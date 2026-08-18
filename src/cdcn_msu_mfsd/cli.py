import json
import shutil
from pathlib import Path
from typing import Annotated

import torch
import typer

from cdcn_msu_mfsd import __version__
from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.data.manifest import (
    build_manifest,
    load_manifest,
    manifest_summary,
    save_manifest,
)
from cdcn_msu_mfsd.data.preprocessing import (
    load_frame_records,
    prepare_frames,
)
from cdcn_msu_mfsd.data.protocol import make_folds, save_folds
from cdcn_msu_mfsd.data.supervision import export_prnet_inputs, import_prnet_outputs
from cdcn_msu_mfsd.evaluation import evaluate_checkpoint, load_threshold
from cdcn_msu_mfsd.experiments import (
    aggregate_oof,
    expand_matrix,
    median_best_epoch,
    run_matrix,
    run_refit_test,
)
from cdcn_msu_mfsd.training import output_directory, run_training

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
Overrides = Annotated[list[str] | None, typer.Option("--set", help="Override a YAML key")]
OptionalPath = Annotated[Path | None, typer.Option()]


def _echo(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def doctor(require_cuda: Annotated[bool, typer.Option()] = False) -> None:
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vram_gib": properties.total_memory / (1024**3),
            }
        )
    _echo(report)
    if require_cuda and not cuda_available:
        raise typer.Exit(1)


@app.command("inspect-data")
def inspect_data(
    dataset_root: Path = Path("data/raw/scene01"),
    train_subjects: Path = Path("data/raw/train_sub_list.txt"),
    test_subjects: Path = Path("data/raw/test_sub_list.txt"),
    probe: Annotated[bool, typer.Option(help="Read optional video metadata with ffprobe")] = False,
) -> None:
    records = build_manifest(dataset_root, train_subjects, test_subjects, probe)
    _echo(manifest_summary(records))


@app.command()
def prepare(
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    probe: Annotated[bool, typer.Option(help="Read optional video metadata with ffprobe")] = False,
) -> None:
    config = load_config(config_path, override)
    records = build_manifest(
        config.paths.dataset_root,
        config.paths.train_subjects,
        config.paths.test_subjects,
        probe=probe,
    )
    manifest_hash = save_manifest(records, config.paths.cache_root / "manifest.jsonl")
    folds = make_folds(records)
    folds_hash = save_folds(folds, config.paths.cache_root / "folds.json")
    frame_records, frames_hash = prepare_frames(
        records,
        config.paths.dataset_root,
        config.paths.cache_root,
        config.data.candidates_per_video,
        limit,
    )
    _echo(
        {
            **manifest_summary(records),
            "prepared_frames": len(frame_records),
            "manifest_sha256": manifest_hash,
            "folds_sha256": folds_hash,
            "frames_sha256": frames_hash,
        }
    )


@app.command("export-prnet")
def export_prnet(
    export_root: Path,
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
) -> None:
    config = load_config(config_path, override)
    records = load_frame_records(config.paths.cache_root / "frames.jsonl")
    contract = export_prnet_inputs(records, config.paths.cache_root, export_root)
    _echo({"exported": len(contract), "contract": export_root / "contract.jsonl"})


@app.command("import-prnet")
def import_prnet(
    contract: Path,
    output_root: Path,
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
) -> None:
    config = load_config(config_path, override)
    imported = import_prnet_outputs(contract, output_root, config.paths.depth_root)
    _echo({"imported": imported, "depth_root": config.paths.depth_root})


@app.command()
def train(
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
    resume: Path | None = None,
) -> None:
    result = run_training(load_config(config_path, override), resume)
    _echo(result.__dict__)


@app.command()
def evaluate(
    checkpoint: Path,
    split: Annotated[str, typer.Option()] = "validation",
    threshold_path: OptionalPath = None,
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
) -> None:
    config = load_config(config_path, override)
    threshold = load_threshold(threshold_path) if threshold_path is not None else None
    report = evaluate_checkpoint(
        config,
        checkpoint,
        split,
        output_directory(config),
        threshold,
    )
    _echo(report)


@app.command("run-matrix")
def run_matrix_command(matrix_path: Path, dry_run: bool = False) -> None:
    _echo(run_matrix(matrix_path, dry_run))


@app.command("show-matrix")
def show_matrix(matrix_path: Path) -> None:
    configurations = expand_matrix(matrix_path)
    _echo(
        [
            {
                "model": config.model.name,
                "theta": config.model.theta,
                "supervision": config.supervision.kind,
                "fold": config.experiment.fold,
                "seed": config.experiment.seed,
            }
            for config in configurations
        ]
    )


@app.command("aggregate-oof")
def aggregate_oof_command(predictions: list[Path], output_dir: Path) -> None:
    _echo(aggregate_oof(predictions, output_dir))


@app.command("refit-test")
def refit_test(
    threshold_path: Path,
    epochs: Annotated[int | None, typer.Option(min=1)] = None,
    summaries: Annotated[list[Path] | None, typer.Option("--summary")] = None,
    config_path: Path = Path("configs/base.yaml"),
    override: Overrides = None,
) -> None:
    selected_epochs = epochs
    if selected_epochs is None:
        selected_epochs = median_best_epoch(summaries or [])
    overrides = [*(override or []), "experiment.phase=refit"]
    config = load_config(config_path, overrides)
    report = run_refit_test(config, selected_epochs, load_threshold(threshold_path))
    _echo(report)


@app.command("manifest-summary")
def manifest_summary_command(path: Path = Path("cache/manifest.jsonl")) -> None:
    _echo(manifest_summary(load_manifest(path)))


if __name__ == "__main__":
    app()
