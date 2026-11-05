import json
from typing import Any

import torch

from cdcn_msu_mfsd.config import checkpoint_matches_config, load_config
from cdcn_msu_mfsd.pipelines.common import Campaign, write_json
from cdcn_msu_mfsd.tracking import TrackingContext
from cdcn_msu_mfsd.training import output_directory, run_training


def run(campaign: Campaign) -> dict[str, Any]:
    config = load_config(
        campaign.settings.base_config,
        [
            "model.name=cdcn",
            "model.width_multiplier=0.25",
            "model.theta=0.7",
            "supervision.kind=pseudo_depth",
            f"experiment.name={campaign.settings.name}-wandb-smoke",
            "experiment.phase=smoke",
            "experiment.fold=0",
            "experiment.seed=2026",
            "training.epochs=1",
            "training.batch_size=2",
            "training.gradient_accumulation=1",
            "training.patience=1",
            "data.frames_per_video=1",
            "data.workers=0",
            "runtime.device=cuda",
            "runtime.amp=false",
            "tracking.log_every_steps=1",
        ],
    )
    output_dir = output_directory(config)
    final_paths = (
        output_dir / "summary.json",
        output_dir / "best.pt",
        output_dir / "validation_videos.jsonl",
    )
    run_metadata_path = output_dir / "wandb_run.json"
    if all(path.is_file() for path in final_paths) and run_metadata_path.is_file():
        metadata = json.loads(run_metadata_path.read_text())
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        if metadata.get("status") == "finished" and checkpoint_matches_config(
            checkpoint.get("config_hash"), config
        ):
            result = {"status": "reused_complete", "output_dir": str(output_dir)}
            write_json(campaign.root / "wandb_smoke.json", result)
            return result
    resume = output_dir / "last.pt"
    context = TrackingContext(
        name=f"smoke-{campaign.settings.name}-cdcn-a0.25-t0.70-s2026",
        group=campaign.settings.name,
        job_type="smoke",
        tags=("development-smoke", "pseudo_depth", "cdcn-a0.25-t0.70"),
        config={
            "campaign": campaign.settings.name,
            "purpose": "verify online W&B logging before full experiment runs",
            "epochs": 1,
            "frames_per_video": 1,
        },
        required_online=True,
        custom_step_metric=True,
    )
    training_result = run_training(config, resume if resume.is_file() else None, context)
    metadata = json.loads(run_metadata_path.read_text())
    if metadata.get("status") != "finished":
        raise RuntimeError("W&B smoke fit did not finish an online run")
    report = {
        "status": "completed",
        "output_dir": str(output_dir),
        "best_epoch": training_result.best_epoch,
        "epochs_completed": training_result.epochs_completed,
        "wandb_run_id": metadata["run_id"],
        "wandb_url": metadata.get("url"),
    }
    write_json(campaign.root / "wandb_smoke.json", report)
    return report
