import json
import time
from pathlib import Path
from typing import Any

import torch

from cdcn_msu_mfsd.config import checkpoint_matches_config
from cdcn_msu_mfsd.evaluation import evaluate_checkpoint
from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    Condition,
    Progress,
    conditions_for_theta,
    conditions_for_widths,
    write_json,
)
from cdcn_msu_mfsd.tracking import TrackingContext
from cdcn_msu_mfsd.training import run_training


def _complete(config: Any, output_dir: Path, tracking_required: bool) -> bool:
    required = (output_dir / "summary.json", output_dir / "best.pt")
    if not all(path.is_file() for path in required):
        return False
    wandb_path = output_dir / "wandb_run.json"
    if tracking_required and not wandb_path.is_file():
        return False
    if wandb_path.is_file() and json.loads(wandb_path.read_text()).get("status") != "finished":
        return False
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    if not checkpoint_matches_config(checkpoint.get("config_hash"), config):
        raise ValueError(f"Saved checkpoint does not match current config: {output_dir}")
    if not (output_dir / "validation_videos.jsonl").is_file():
        evaluate_checkpoint(config, output_dir / "best.pt", "validation", output_dir)
    return True


def _run_jobs(
    campaign: Campaign,
    stage: str,
    jobs: list[tuple[Condition, int, int]],
) -> dict[str, object]:
    progress = Progress(campaign, stage, len(jobs))
    results = []
    for index, (condition, fold, seed) in enumerate(jobs, start=1):
        config = campaign.config(condition, fold, seed)
        output_dir = campaign.cv_directory(condition, fold, seed)
        started = time.monotonic()
        if _complete(config, output_dir, tracking_required=stage == "theta-cv"):
            status = "reused_complete"
            summary = json.loads((output_dir / "summary.json").read_text())
        else:
            resume = output_dir / "last.pt"
            context = TrackingContext(
                name=f"{stage}-{condition.key}-f{fold}-s{seed}",
                group=f"{campaign.settings.name}-{stage}",
                job_type="fit",
                tags=(stage, condition.key, f"fold-{fold}", f"seed-{seed}"),
                config={
                    "campaign": campaign.settings.name,
                    "stage": stage,
                    "condition": condition.key,
                    "fold": fold,
                    "seed": seed,
                    "resumed_from_checkpoint": resume.is_file(),
                },
                required_online=True,
                custom_step_metric=True,
            )
            result = run_training(config, resume if resume.is_file() else None, context)
            summary = {
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "epochs_completed": result.epochs_completed,
            }
            status = "completed"
        result_record = {
            "condition": condition.key,
            "fold": fold,
            "seed": seed,
            "status": status,
            "output_dir": str(output_dir),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            **summary,
        }
        results.append(result_record)
        write_json(
            campaign.root / "jobs" / stage / f"{condition.key}-f{fold}-s{seed}.json", result_record
        )
        progress.update(
            index, f"{condition.key}, fold {fold}, seed {seed}", result_record["elapsed_seconds"]
        )
    report = {"stage": stage, "jobs": len(results), "results": results}
    write_json(campaign.root / f"{stage}_runs.json", report)
    return report


def run_width(campaign: Campaign) -> dict[str, object]:
    jobs = [
        (condition, fold, seed)
        for condition in conditions_for_widths(campaign)
        for seed in campaign.settings.seeds
        for fold in campaign.settings.folds
    ]
    return _run_jobs(campaign, "width-cv", jobs)


def run_theta(campaign: Campaign) -> dict[str, object]:
    jobs = [
        (condition, fold, seed)
        for condition in conditions_for_theta(campaign)
        if condition.theta != 0.7
        for seed in campaign.settings.seeds
        for fold in campaign.settings.folds
    ]
    if len(jobs) != 60:
        raise ValueError(f"Expected 60 new theta runs, found {len(jobs)}")
    return _run_jobs(campaign, "theta-cv", jobs)
