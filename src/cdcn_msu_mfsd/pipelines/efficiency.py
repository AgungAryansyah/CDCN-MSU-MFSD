import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cdcn_msu_mfsd.efficiency import benchmark_checkpoint
from cdcn_msu_mfsd.pipelines.common import Campaign, Condition, Progress, sha256_file, write_json


def _checkpoint(campaign: Campaign, condition: Condition) -> Path:
    path = campaign.cv_directory(condition, 0, 2026) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing representative fold-0 checkpoint: {path}")
    return path


def run_scope(campaign: Campaign, scope: str) -> dict[str, Any]:
    if scope == "width":
        conditions = [Condition(width, 0.7) for width in campaign.settings.widths]
    elif scope == "theta":
        conditions = [
            Condition(campaign.settings.primary_width, theta)
            for theta in campaign.settings.theta_values
            if theta != 0.7
        ]
    else:
        raise ValueError("Efficiency scope must be width or theta")
    jobs = [(condition, device) for condition in conditions for device in ("cpu", "cuda")]
    progress = Progress(campaign, f"{scope}-efficiency", len(jobs))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 6))
    records = []
    try:
        for index, (condition, device) in enumerate(jobs, start=1):
            checkpoint = _checkpoint(campaign, condition)
            output_path = campaign.root / "efficiency" / condition.key / f"{device}.json"
            checkpoint_digest = sha256_file(checkpoint)
            if output_path.is_file():
                existing = __import__("json").loads(output_path.read_text())
                if existing.get("checkpoint_sha256") == checkpoint_digest:
                    records.append(existing)
                    progress.update(index, f"reused {condition.key} on {device}")
                    continue
            started = time.monotonic()
            report = benchmark_checkpoint(
                campaign.config(condition),
                checkpoint,
                device,
                warmup_runs=20,
                timed_runs=100,
                input_shape=(1, 3, 256, 256),
            )
            record = asdict(report)
            record.update(
                {
                    "condition": condition.key,
                    "checkpoint_sha256": checkpoint_digest,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
            )
            write_json(output_path, record)
            records.append(record)
            progress.update(index, f"{condition.key} on {device}", record["elapsed_seconds"])
    finally:
        torch.set_num_threads(previous_threads)
    result = {"scope": scope, "records": records}
    write_json(campaign.root / f"{scope}_efficiency.json", result)
    return result
