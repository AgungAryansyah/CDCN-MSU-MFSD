import itertools
import json
import statistics
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from cdcn_msu_mfsd.config import Config, config_hash, load_config, save_resolved_config
from cdcn_msu_mfsd.data.dataset import MSUMFSDDataset
from cdcn_msu_mfsd.data.preprocessing import load_frame_records
from cdcn_msu_mfsd.data.supervision import (
    BinaryTarget,
    PseudoDepthTarget,
    validate_pseudo_depth_targets,
)
from cdcn_msu_mfsd.evaluation import (
    VideoPrediction,
    evaluate_checkpoint,
    metric_report,
    select_threshold,
)
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.tracking import create_tracker, evaluation_metrics
from cdcn_msu_mfsd.training import (
    Trainer,
    output_directory,
    provenance,
    run_training,
    save_checkpoint,
    seed_everything,
    select_device,
)


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: Path
    name: str
    fixed: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, list[Any]]


def load_matrix(path: Path) -> MatrixConfig:
    raw = yaml.safe_load(path.read_text())
    matrix = MatrixConfig.model_validate(raw)
    return matrix.model_copy(update={"base": (path.parent / matrix.base).resolve()})


def _override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value)}"


def expand_matrix(path: Path, extra_overrides: list[str] | None = None) -> list[Config]:
    matrix = load_matrix(path)
    keys = sorted(matrix.axes)
    configurations = []
    seen = set()
    for values in itertools.product(*(matrix.axes[key] for key in keys)):
        config_overrides = [_override(key, value) for key, value in matrix.fixed.items()]
        config_overrides.extend(
            _override(key, value) for key, value in zip(keys, values, strict=True)
        )
        config_overrides.append(_override("experiment.name", matrix.name))
        config_overrides.extend(extra_overrides or [])
        config = load_config(matrix.base, config_overrides)
        digest = config_hash(config)
        if digest not in seen:
            configurations.append(config)
            seen.add(digest)
    return configurations


def run_matrix(
    path: Path, dry_run: bool = False, overrides: list[str] | None = None
) -> list[dict[str, Any]]:
    results = []
    for config in expand_matrix(path, overrides):
        output_dir = output_directory(config)
        record: dict[str, Any] = {
            "hash": config_hash(config),
            "output_dir": str(output_dir),
            "model": config.model.name,
            "supervision": config.supervision.kind,
            "fold": config.experiment.fold,
            "seed": config.experiment.seed,
        }
        if dry_run:
            record["status"] = "planned"
        elif (
            (output_dir / "summary.json").exists()
            and (output_dir / "best.pt").exists()
            and (output_dir / "validation_videos.jsonl").exists()
        ):
            record["status"] = "skipped_complete"
        elif (output_dir / "summary.json").exists() and (output_dir / "best.pt").exists():
            evaluate_checkpoint(config, output_dir / "best.pt", "validation", output_dir)
            record["status"] = "evaluated_complete_run"
        else:
            resume = output_dir / "last.pt"
            result = run_training(config, resume if resume.exists() else None)
            record.update(
                {
                    "status": "completed",
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                }
            )
        results.append(record)
    return results


def aggregate_oof(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    predictions = []
    seen = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line:
                continue
            item = VideoPrediction(**json.loads(line))
            if item.video_id in seen:
                raise ValueError(f"Duplicate out-of-fold prediction for {item.video_id}")
            predictions.append(item)
            seen.add(item.video_id)
    if len(predictions) != 120:
        raise ValueError(f"Expected 120 out-of-fold videos, received {len(predictions)}")
    labels = [item.label for item in predictions]
    if labels.count(1) != 30 or labels.count(0) != 90:
        raise ValueError("Out-of-fold predictions do not match the official training class counts")
    threshold = select_threshold(predictions)
    report = metric_report(predictions, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oof_videos.jsonl").write_text(
        "\n".join(json.dumps(item.__dict__, sort_keys=True) for item in predictions) + "\n"
    )
    (output_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold}, indent=2) + "\n"
    )
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def median_best_epoch(summary_paths: list[Path]) -> int:
    if len(summary_paths) != 5:
        raise ValueError("Exactly five cross-validation summaries are required")
    epochs = [int(json.loads(path.read_text())["best_epoch"]) + 1 for path in summary_paths]
    return max(1, round(statistics.median(epochs)))


def run_refit(config: Config, epochs: int, threshold: float | None = None) -> Path:
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"epochs": epochs})}
    )
    if config.experiment.phase != "refit":
        raise ValueError("Refit requires experiment.phase=refit")
    seed_everything(config.experiment.seed, config.runtime.deterministic)
    device = select_device(config.runtime.device)
    records = load_frame_records(config.paths.cache_root / "frames.jsonl")
    depth_validation = (
        validate_pseudo_depth_targets(records, config.paths.depth_root)
        if config.supervision.kind == "pseudo_depth"
        else None
    )
    subjects = {record.subject for record in records if record.protocol == "train"}
    target_provider = (
        BinaryTarget()
        if config.supervision.kind == "binary"
        else PseudoDepthTarget(config.paths.depth_root)
    )
    dataset = MSUMFSDDataset(
        records,
        config.paths.cache_root,
        target_provider,
        subjects,
        True,
        config.data.frames_per_video,
        config.experiment.seed,
        config.data.crop_scale,
        (config.data.crop_scale_min, config.data.crop_scale_max),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.workers > 0,
    )
    output_dir = (
        config.paths.output_root
        / config.experiment.name
        / f"{config.model.name}-{config.supervision.kind}"
        / "refit"
        / f"seed-{config.experiment.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "config.yaml")
    run_provenance = provenance(config, depth_validation)
    tracker = create_tracker(config, output_dir, run_provenance)
    trainer = Trainer(
        config,
        build_model(config.model.name, config.model.theta, config.model.width_multiplier),
        loader,
        loader,
        tracker,
        device,
    )
    global_step = 0
    try:
        for epoch in range(epochs):
            metrics, global_step = trainer.train_epoch(epoch, global_step)
            tracker.log(
                {"epoch": epoch, **{f"refit/{key}": value for key, value in metrics.items()}},
                global_step,
            )
            trainer.scheduler.step()
            save_checkpoint(
                output_dir / "final.pt",
                epoch,
                epoch,
                metrics["total"],
                global_step,
                0,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                config,
            )
        summary: dict[str, int | float] = {"refit_epochs": epochs}
        if threshold is not None:
            report = evaluate_checkpoint(
                config,
                output_dir / "final.pt",
                "test",
                output_dir,
                threshold,
            )
            test_metrics = evaluation_metrics("test", report)
            tracker.log(test_metrics)
            summary.update(test_metrics)
        tracker.summary(summary)
    finally:
        tracker.finish()
    return output_dir / "final.pt"


def run_refit_test(config: Config, epochs: int, threshold: float) -> dict[str, Any]:
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"epochs": epochs})}
    )
    checkpoint = run_refit(config, epochs, threshold)
    return cast(dict[str, Any], json.loads((checkpoint.parent / "test_metrics.json").read_text()))
