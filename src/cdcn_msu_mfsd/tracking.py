import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import wandb
from dotenv import load_dotenv

from cdcn_msu_mfsd.config import Config, resolved_config


class Tracker(Protocol):
    def log(self, metrics: Mapping[str, int | float], step: int | None = None) -> None: ...

    def summary(self, metrics: Mapping[str, int | float | str]) -> None: ...

    def finish(self) -> None: ...


class MetricsTracker:
    def __init__(self, local_path: Path, run: Any | None = None) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path = local_path
        self.run = run

    def log(self, metrics: Mapping[str, int | float], step: int | None = None) -> None:
        record: dict[str, int | float] = dict(metrics)
        if step is not None:
            record["step"] = step
        with self.local_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.run is not None:
            try:
                self.run.log(dict(metrics), step=step)
            except Exception as error:
                warnings.warn(
                    f"W&B logging failed; continuing with local metrics: {error}", stacklevel=2
                )

    def summary(self, metrics: Mapping[str, int | float | str]) -> None:
        path = self.local_path.with_name("summary.json")
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update(metrics)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        if self.run is not None:
            try:
                self.run.summary.update(dict(metrics))
            except Exception as error:
                warnings.warn(f"W&B summary failed; local summary is intact: {error}", stacklevel=2)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                warnings.warn(f"W&B finish failed: {error}", stacklevel=2)


def evaluation_metrics(split: str, report: Mapping[str, Any]) -> dict[str, int | float]:
    false_positive = int(report["false_accepts"])
    false_negative = int(report["false_rejects"])
    genuine = int(report["genuine"])
    attacks = int(report["attacks"])
    return {
        f"{split}/acer": float(report["acer"]),
        f"{split}/apcer": float(report["apcer"]),
        f"{split}/bpcer": float(report["bpcer"]),
        f"{split}/eer": float(report["eer"]),
        f"{split}/roc_auc": float(report["auc"]),
        f"{split}/f1_attack": float(report["f1"]),
        f"{split}/accuracy": float(report["accuracy"]),
        f"{split}/threshold": float(report["threshold"]),
        f"{split}/confusion/true_positive": genuine - false_negative,
        f"{split}/confusion/true_negative": attacks - false_positive,
        f"{split}/confusion/false_positive": false_positive,
        f"{split}/confusion/false_negative": false_negative,
        f"{split}/count/genuine": genuine,
        f"{split}/count/attack": attacks,
    }


def create_tracker(
    config: Config, output_dir: Path, provenance: Mapping[str, Any]
) -> MetricsTracker:
    load_dotenv()
    local_path = output_dir / "metrics.jsonl"
    mode = os.getenv("WANDB_MODE", "offline")
    enabled = config.tracking.enabled and mode != "disabled"
    if not enabled:
        return MetricsTracker(local_path)
    if mode not in {"online", "offline"}:
        raise ValueError("WANDB_MODE must be online, offline, or disabled")
    wandb_mode = cast(Literal["online", "offline"], mode)
    project = os.getenv("WANDB_PROJECT")
    if not project:
        raise ValueError("WANDB_PROJECT must be configured in .env")
    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY must be configured in .env for online tracking")
    run_config = resolved_config(config)
    run_config["provenance"] = dict(provenance)
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY") or None,
        project=project,
        dir=output_dir,
        name=f"{config.experiment.name}-f{config.experiment.fold}-s{config.experiment.seed}",
        group=config.experiment.name,
        job_type=config.experiment.phase,
        tags=[config.model.name, config.supervision.kind],
        config=run_config,
        mode=wandb_mode,
        save_code=False,
        settings=wandb.Settings(
            allow_offline_artifacts=False,
            disable_code=True,
            disable_git=True,
            x_save_requirements=False,
            ignore_globs=("*.pt", "*.pth", "*.ckpt", "*.jpg", "*.png", "*.npy"),
        ),
    )
    return MetricsTracker(local_path, run)
