#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from cdcn_msu_mfsd.config import (
    Config,
    checkpoint_matches_config,
    config_hash,
    save_resolved_config,
)
from cdcn_msu_mfsd.data.supervision import PseudoDepthValidation
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    Condition,
    Progress,
    campaign_lock,
    input_fingerprint,
    read_json,
    sha256_file,
    update_campaign_state,
    write_json,
)
from cdcn_msu_mfsd.pipelines.cross_validation import _complete
from cdcn_msu_mfsd.pipelines.refit import _refit_entry
from cdcn_msu_mfsd.tracking import TrackingContext, create_tracker
from cdcn_msu_mfsd.training import (
    Trainer,
    make_refit_loader,
    provenance,
    run_training,
    seed_everything,
    select_device,
)


def _job_key(job: dict[str, Any], stage: str) -> str:
    if stage == "theta":
        return f"{job['condition']}-f{job['fold']}-s{job['seed']}"
    return f"{job['condition']}-s{job['seed']}"


def _run_theta_job(campaign: Campaign, job: dict[str, Any]) -> dict[str, Any]:
    condition = Condition(float(job["width"]), float(job["theta"]))
    fold = int(job["fold"])
    seed = int(job["seed"])
    config = campaign.config(condition, fold, seed)
    output_dir = campaign.cv_directory(condition, fold, seed)
    started = time.monotonic()
    if _complete(config, output_dir, tracking_required=True):
        status = "reused_complete"
        summary = read_json(output_dir / "summary.json")
    else:
        resume = output_dir / "last.pt"
        context = TrackingContext(
            name=f"theta-cv-{condition.key}-f{fold}-s{seed}",
            group=f"{campaign.settings.name}-theta-cv",
            job_type="fit",
            tags=("theta-cv", condition.key, f"fold-{fold}", f"seed-{seed}"),
            config={
                "campaign": campaign.settings.name,
                "stage": "theta-cv",
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
    record = {
        "condition": condition.key,
        "fold": fold,
        "seed": seed,
        "status": status,
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        **summary,
    }
    write_json(
        campaign.root / "jobs" / "theta-cv" / f"{condition.key}-f{fold}-s{seed}.json",
        record,
    )
    return record


def _refit_complete(
    campaign: Campaign, entry: dict[str, Any], plan_hash: str
) -> dict[str, Any] | None:
    condition_key = str(entry["condition"])
    seed = int(entry["seed"])
    condition = next(item for item in campaign.conditions if item.key == condition_key)
    output_dir = campaign.refit_directory(condition, seed)
    complete_path = output_dir / "refit_complete.json"
    final_path = output_dir / "final.pt"
    if not (complete_path.is_file() and final_path.is_file()):
        return None
    complete = read_json(complete_path)
    wandb_path = output_dir / "wandb_run.json"
    if (
        complete.get("frozen_plan_sha256") == plan_hash
        and complete.get("config_hash") == entry["refit_config_hash"]
        and complete.get("checkpoint_sha256") == sha256_file(final_path)
        and wandb_path.is_file()
        and read_json(wandb_path).get("status") == "finished"
    ):
        return {**complete, "status": "reused_complete"}
    raise ValueError(f"Completed refit artifacts do not match the frozen plan: {output_dir}")


def _run_refit_job(
    campaign: Campaign, entry: dict[str, Any], plan_hash: str, depth_data: dict[str, Any]
) -> dict[str, Any]:
    config = Config.model_validate(entry["refit_config"])
    if config_hash(config) != entry["refit_config_hash"]:
        raise ValueError(
            f"Frozen config hash mismatch for {entry['condition']} seed {entry['seed']}"
        )
    condition_key = str(entry["condition"])
    seed = int(entry["seed"])
    condition = next(item for item in campaign.conditions if item.key == condition_key)
    output_dir = campaign.refit_directory(condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = _refit_complete(campaign, entry, plan_hash)
    if previous is not None:
        return previous

    started = time.monotonic()
    save_resolved_config(config, output_dir / "config.yaml")
    seed_everything(config.experiment.seed, config.runtime.deterministic)
    device = select_device(config.runtime.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    loader = make_refit_loader(config, device)
    depth_validation = PseudoDepthValidation(
        int(depth_data["pseudo_depth_targets"]), str(depth_data["pseudo_depth_sha256"])
    )
    run_provenance = provenance(config, depth_validation)
    run_provenance["frozen_plan_sha256"] = plan_hash
    run_provenance["oof_threshold_sha256"] = str(entry["threshold_sha256"])
    context = TrackingContext(
        name=f"refit-{condition_key}-s{seed}",
        group=campaign.settings.name,
        job_type="refit",
        tags=("final-fit", condition_key, f"seed-{seed}"),
        config={
            "campaign": campaign.settings.name,
            "condition": condition_key,
            "seed": seed,
            "epochs": config.training.epochs,
            "threshold": entry["threshold"],
            "frozen_plan_sha256": plan_hash,
        },
        required_online=True,
        custom_step_metric=True,
    )
    tracker = create_tracker(config, output_dir, run_provenance, context)
    try:
        trainer = Trainer(
            config,
            build_model(config.model.name, config.model.theta, config.model.width_multiplier),
            loader,
            loader,
            tracker,
            device,
        )
        last_path = output_dir / "last.pt"
        trainer.fit_fixed_epochs(output_dir, last_path if last_path.is_file() else None)
        final_path = output_dir / "final.pt"
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        if not checkpoint_matches_config(checkpoint.get("config_hash"), config):
            raise ValueError("Refit checkpoint config hash does not match the frozen config")
        complete = {
            "condition": condition_key,
            "seed": seed,
            "status": "completed",
            "refit_epochs": config.training.epochs,
            "threshold": float(entry["threshold"]),
            "frozen_plan_sha256": plan_hash,
            "config_hash": entry["refit_config_hash"],
            "checkpoint": str(final_path),
            "checkpoint_sha256": sha256_file(final_path),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        tracker.summary(
            {
                "refit/epochs": config.training.epochs,
                "refit/checkpoint_bytes": final_path.stat().st_size,
                "refit/frozen_threshold": float(entry["threshold"]),
            }
        )
    finally:
        tracker.finish()
    write_json(output_dir / "refit_complete.json", complete)
    write_json(output_dir / "refit_result.json", complete)
    return complete


def _worker(
    campaign_path: str,
    stage: str,
    gpu_index: int,
    tasks: Any,
    results: Any,
    plan_hash: str | None,
    depth_data: dict[str, Any] | None,
) -> None:
    worker_id = f"gpu-{gpu_index}"
    try:
        torch.cuda.set_device(gpu_index)
        campaign = Campaign(Path(campaign_path))
        while True:
            job = tasks.get()
            if job is None:
                break
            key = _job_key(job, stage)
            results.put({"kind": "started", "worker": worker_id, "key": key})
            try:
                if stage == "theta":
                    record = _run_theta_job(campaign, job)
                else:
                    if plan_hash is None or depth_data is None:
                        raise RuntimeError("Refit worker did not receive frozen plan data")
                    record = _run_refit_job(campaign, job["entry"], plan_hash, depth_data)
                results.put({"kind": "done", "worker": worker_id, "key": key, "record": record})
            except BaseException:
                results.put(
                    {
                        "kind": "error",
                        "worker": worker_id,
                        "key": key,
                        "error": traceback.format_exc(),
                    }
                )
    except BaseException:
        results.put({"kind": "worker_error", "worker": worker_id, "error": traceback.format_exc()})
    finally:
        results.put({"kind": "worker_done", "worker": worker_id})


def _theta_jobs(campaign: Campaign) -> list[dict[str, Any]]:
    jobs = [
        {
            "condition": condition.key,
            "width": condition.width,
            "theta": condition.theta,
            "fold": fold,
            "seed": seed,
        }
        for condition in sorted(campaign.conditions, key=lambda item: (item.width, item.theta))
        if condition.width == campaign.settings.primary_width and condition.theta != 0.7
        for seed in campaign.settings.seeds
        for fold in campaign.settings.folds
    ]
    if len(jobs) != 60:
        raise ValueError(f"Expected 60 theta cross-validation fits, found {len(jobs)}")
    return jobs


def _history(campaign: Campaign, stage: str) -> list[float]:
    folder = campaign.root / "jobs" / ("theta-cv" if stage == "theta" else "refit")
    durations = []
    for path in folder.glob("*.json"):
        record = read_json(path)
        if record.get("status") == "completed" and record.get("elapsed_seconds", 0) > 0:
            durations.append(float(record["elapsed_seconds"]))
    return durations


def _execute(campaign: Campaign, stage: str, gpu_indices: list[int]) -> None:
    component = "theta" if stage == "theta" else "refit"
    progress_stage = "theta-cv" if stage == "theta" else "refit"
    state = read_json(campaign.state_path) if campaign.state_path.exists() else {}
    preflight = state.get("stages", {}).get("preflight", {})
    if preflight.get("status") != "completed":
        raise RuntimeError("Run preflight before parallel fits")
    if preflight.get("input_fingerprint") != input_fingerprint(campaign):
        raise RuntimeError("Campaign inputs or source code changed after preflight")

    update_campaign_state(campaign, component, "running", {"gpu_indices": gpu_indices})
    plan_hash: str | None = None
    depth_data: dict[str, Any] | None = None
    if stage == "theta":
        jobs = _theta_jobs(campaign)
    else:
        entries = _refit_entry(campaign)
        plan_hash = sha256_file(campaign.plan_path)
        depth_data = read_json(campaign.root / "preflight.json")
        jobs = [
            {
                "condition": str(entry["condition"]),
                "seed": int(entry["seed"]),
                "entry": entry,
            }
            for entry in entries
        ]

    progress = Progress(campaign, progress_stage, len(jobs))
    progress.durations.extend(duration / len(gpu_indices) for duration in _history(campaign, stage))
    ordered_records: dict[str, dict[str, Any]] = {}
    pending = []
    for job in jobs:
        key = _job_key(job, stage)
        if stage == "theta":
            condition = Condition(float(job["width"]), float(job["theta"]))
            config = campaign.config(condition, int(job["fold"]), int(job["seed"]))
            output_dir = campaign.cv_directory(condition, int(job["fold"]), int(job["seed"]))
            complete = _complete(config, output_dir, tracking_required=True)
            if complete:
                existing_path = campaign.root / "jobs" / "theta-cv" / f"{key}.json"
                record = (
                    read_json(existing_path)
                    if existing_path.is_file()
                    else {
                        "condition": condition.key,
                        "fold": int(job["fold"]),
                        "seed": int(job["seed"]),
                        "status": "reused_complete",
                        "output_dir": str(output_dir),
                        **read_json(output_dir / "summary.json"),
                    }
                )
                ordered_records[key] = record
            else:
                pending.append(job)
        else:
            entry = job["entry"]
            complete = _refit_complete(campaign, entry, str(plan_hash))
            if complete is not None:
                ordered_records[key] = complete
            else:
                pending.append(job)

    completed = len(ordered_records)
    progress.update(completed, f"resuming; {completed}/{len(jobs)} already complete")
    print(
        f"[{progress_stage}] launching {len(gpu_indices)} fit workers on GPUs "
        f"{', '.join(map(str, gpu_indices))}; {len(pending)} fits remain",
        flush=True,
    )
    if pending:
        context = mp.get_context("spawn")
        task_queues = [context.Queue() for _ in gpu_indices]
        result_queue = context.Queue()
        for index, job in enumerate(pending):
            task_queues[index % len(gpu_indices)].put(job)
        for task_queue in task_queues:
            task_queue.put(None)
        processes = [
            context.Process(
                target=_worker,
                args=(
                    str(campaign.path),
                    stage,
                    gpu_index,
                    task_queue,
                    result_queue,
                    plan_hash,
                    depth_data,
                ),
                name=f"fit-gpu-{gpu_index}",
            )
            for gpu_index, task_queue in zip(gpu_indices, task_queues, strict=True)
        ]
        active: dict[str, str] = {}
        errors: list[str] = []
        worker_done = 0
        for process in processes:
            process.start()
        try:
            while worker_done < len(processes):
                try:
                    message = result_queue.get(timeout=60)
                except queue.Empty:
                    running = "; ".join(f"{worker}: {job}" for worker, job in active.items())
                    progress.update(completed, running or "workers starting")
                    dead = [process for process in processes if not process.is_alive()]
                    if dead and worker_done < len(processes):
                        raise RuntimeError(
                            "Fit worker exited unexpectedly: "
                            + ", ".join(f"{item.name}={item.exitcode}" for item in dead)
                        ) from None
                    continue
                kind = message["kind"]
                worker = str(message["worker"])
                if kind == "started":
                    active[worker] = str(message["key"])
                    running = "; ".join(f"{name}: {job}" for name, job in active.items())
                    progress.update(completed, running)
                elif kind == "done":
                    record = message["record"]
                    key = str(message["key"])
                    ordered_records[key] = record
                    active.pop(worker, None)
                    completed += 1
                    elapsed = float(record.get("elapsed_seconds", 0))
                    current = "; ".join(f"{name}: {job}" for name, job in active.items())
                    progress.update(
                        completed,
                        current or f"completed {key}",
                        elapsed / len(gpu_indices),
                    )
                elif kind == "error":
                    active.pop(worker, None)
                    detail = f"{message['key']} on {worker}:\n{message['error']}"
                    errors.append(detail)
                    print(detail, file=sys.stderr, flush=True)
                    running = "; ".join(f"{name}: {job}" for name, job in active.items())
                    progress.update(completed, running or f"failed {message['key']}")
                elif kind == "worker_error":
                    errors.append(str(message["error"]))
                    print(message["error"], file=sys.stderr, flush=True)
                elif kind == "worker_done":
                    worker_done += 1
            for process in processes:
                process.join()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            for task_queue in task_queues:
                task_queue.close()
            result_queue.close()

        if errors:
            report = {
                "stage": progress_stage,
                "jobs": len(ordered_records),
                "results": list(ordered_records.values()),
            }
            write_json(campaign.root / f"{progress_stage}_runs.json", report)
            update_campaign_state(campaign, component, "failed", {"errors": errors})
            raise RuntimeError(f"{len(errors)} parallel fit worker error(s)")

    report = {
        "stage": progress_stage,
        "jobs": len(jobs),
        "results": [ordered_records[_job_key(job, stage)] for job in jobs],
    }
    if stage == "theta":
        write_json(campaign.root / "theta-cv_runs.json", report)
    else:
        write_json(
            campaign.root / "refit_results.json",
            {"jobs": len(jobs), "results": report["results"]},
        )
    progress.update(len(jobs), "all fits complete")
    update_campaign_state(campaign, component, "completed", {"jobs": len(jobs)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("theta", "refit"), required=True)
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/pipelines/thesis_completion.yaml")
    )
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpu_indices = [int(item) for item in args.gpus.split(",") if item]
    if len(gpu_indices) != 2 or len(set(gpu_indices)) != 2:
        parser.error("--gpus must name two distinct GPU indices, such as 0,1")
    if not torch.cuda.is_available() or max(gpu_indices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPUs {gpu_indices}, but only {torch.cuda.device_count()} are visible"
        )
    campaign = Campaign(args.campaign)
    with campaign_lock(campaign):
        try:
            _execute(campaign, args.stage, gpu_indices)
        except BaseException as error:
            state = read_json(campaign.state_path) if campaign.state_path.exists() else {}
            current = state.get("stages", {}).get("theta" if args.stage == "theta" else "refit", {})
            if current.get("status") == "running":
                update_campaign_state(
                    campaign,
                    "theta" if args.stage == "theta" else "refit",
                    "failed",
                    {"error": str(error)},
                )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
