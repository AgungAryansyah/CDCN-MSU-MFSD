import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from cdcn_msu_mfsd.config import Config, checkpoint_matches_config
from cdcn_msu_mfsd.evaluation import (
    FramePrediction,
    _evaluation_loader,
    aggregate_video_predictions,
    metric_report,
)
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.pipelines.common import Campaign, Progress, read_json, sha256_file, write_json
from cdcn_msu_mfsd.tracking import TrackingContext, create_tracker, evaluation_metrics
from cdcn_msu_mfsd.training import select_device


def _frame_predictions(path: Path) -> dict[tuple[str, int], FramePrediction]:
    results: dict[tuple[str, int], FramePrediction] = {}
    if not path.is_file():
        return results
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
        path.write_bytes(b"".join(lines))
    for line in lines:
        if not line:
            continue
        item = FramePrediction(**json.loads(line))
        key = (item.video_id, item.frame_index)
        if key in results and results[key] != item:
            raise ValueError(f"Conflicting cached prediction for {key}")
        results[key] = item
    return results


@torch.inference_mode()
def _predict_resumable(
    config: Config,
    checkpoint_path: Any,
    output_dir: Any,
    threshold: float,
    campaign: Campaign,
) -> dict[str, Any]:
    device = select_device(config.runtime.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint_matches_config(checkpoint.get("config_hash"), config):
        raise ValueError("Refit checkpoint config hash does not match the test config")
    model = build_model(config.model.name, config.model.theta, config.model.width_multiplier).to(
        device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = _evaluation_loader(config, "test", device)
    frame_path = output_dir / "test_frames.jsonl"
    state_path = output_dir / "test_prediction_state.json"
    cache_state = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_hash": str(checkpoint["config_hash"]),
        "threshold": threshold,
        "frozen_plan_sha256": sha256_file(campaign.plan_path),
    }
    if state_path.is_file():
        if read_json(state_path) != cache_state:
            raise ValueError("Cached test predictions use a different checkpoint or threshold")
    else:
        write_json(state_path, cache_state)
    cached = _frame_predictions(frame_path)
    total = len(cast(Any, loader.dataset))
    if len(cached) > total:
        raise ValueError("Cached test predictions exceed the test frame count")
    progress = Progress(
        campaign, f"test-{config.model.width_multiplier:.2f}-{config.model.theta:.2f}", total
    )
    progress.update(len(cached), f"resuming with {len(cached)} cached test frames")
    for batch_count, batch in enumerate(loader, start=1):
        images = cast(Tensor, batch["image"])
        missing = [
            index
            for index, (video_id, frame_index) in enumerate(
                zip(batch["video_id"], batch["frame_index"].tolist(), strict=True)
            )
            if (video_id, int(frame_index)) not in cached
        ]
        if missing:
            selected = images[missing].to(device, non_blocking=True)
            outputs = cast(Tensor, model(selected))
            scores = outputs.mean(dim=(1, 2)).detach().cpu().tolist()
            with frame_path.open("a") as stream:
                for index, score in zip(missing, scores, strict=True):
                    item = FramePrediction(
                        video_id=batch["video_id"][index],
                        frame_index=int(batch["frame_index"][index]),
                        label=int(batch["label"][index]),
                        score=float(score),
                        attack_type=str(batch["attack_type"][index]),
                        device=str(batch["device"][index]),
                    )
                    key = (item.video_id, item.frame_index)
                    cached[key] = item
                    stream.write(json.dumps(item.__dict__, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        progress.update(len(cached), f"test batches processed {batch_count}")
    if len(cached) != total:
        raise ValueError(f"Expected {total} test frame predictions, found {len(cached)}")
    frames = list(cached.values())
    videos = aggregate_video_predictions(frames)
    labels = [video.label for video in videos]
    if len(videos) != campaign.settings.test_videos:
        raise ValueError(f"Expected 160 test videos, found {len(videos)}")
    if labels.count(1) != campaign.settings.test_genuine:
        raise ValueError("Official test genuine-video count does not match the protocol")
    if labels.count(0) != campaign.settings.test_attacks:
        raise ValueError("Official test attack-video count does not match the protocol")
    report = metric_report(videos, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_videos.jsonl").write_text(
        "\n".join(json.dumps(item.__dict__, sort_keys=True) for item in videos) + "\n"
    )
    write_json(output_dir / "test_metrics.json", report)
    return report


def run(campaign: Campaign) -> dict[str, Any]:
    if (
        not campaign.plan_digest_path.is_file()
        or campaign.plan_digest_path.read_text().strip() != sha256_file(campaign.plan_path)
    ):
        raise ValueError("Frozen plan digest does not match the plan contents")
    plan = read_json(campaign.plan_path)
    entries = plan["conditions"]
    progress = Progress(campaign, "test", len(entries))
    plan_hash = sha256_file(campaign.plan_path)
    results = []
    for index, entry in enumerate(entries, start=1):
        condition_key = str(entry["condition"])
        seed = int(entry["seed"])
        condition = next(item for item in campaign.conditions if item.key == condition_key)
        output_dir = campaign.test_directory(condition, seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        completion_path = output_dir / "test_complete.json"
        if completion_path.is_file():
            completion = read_json(completion_path)
            wandb_path = output_dir / "wandb" / "wandb_run.json"
            if completion.get("frozen_plan_sha256") != plan_hash:
                raise ValueError("Completed test report uses a different frozen plan")
            if (
                (output_dir / "test_metrics.json").is_file()
                and wandb_path.is_file()
                and read_json(wandb_path).get("status") == "finished"
            ):
                results.append({**completion, "status": "reused_complete"})
                progress.update(index, f"reused {condition_key}, seed {seed}")
                continue
            raise ValueError(f"Test completion record has no metrics file: {output_dir}")

        checkpoint_path = campaign.refit_directory(condition, seed) / "final.pt"
        refit_complete = read_json(checkpoint_path.parent / "refit_complete.json")
        if refit_complete.get("frozen_plan_sha256") != plan_hash:
            raise ValueError(
                f"Refit checkpoint is not linked to the frozen plan: {checkpoint_path}"
            )
        config = Config.model_validate(entry["refit_config"])
        report = _predict_resumable(
            config,
            checkpoint_path,
            output_dir,
            float(entry["threshold"]),
            campaign,
        )
        context = TrackingContext(
            name=f"test-{condition_key}-s{seed}",
            group=campaign.settings.name,
            job_type="test",
            tags=("official-test", condition_key, f"seed-{seed}"),
            config={
                "campaign": campaign.settings.name,
                "condition": condition_key,
                "seed": seed,
                "threshold": entry["threshold"],
                "f1_positive_class": "attack",
                "confusion_matrix_positive_class": "genuine",
                "frozen_plan_sha256": plan_hash,
            },
            required_online=True,
        )
        tracker = create_tracker(
            config,
            output_dir / "wandb",
            {
                "frozen_plan_sha256": plan_hash,
                "refit_checkpoint_sha256": refit_complete["checkpoint_sha256"],
                "test_predictions_sha256": sha256_file(output_dir / "test_videos.jsonl"),
            },
            context,
        )
        try:
            metrics = evaluation_metrics("test", report)
            metrics.update(
                {
                    "test/false_accepts": int(report["false_accepts"]),
                    "test/false_rejects": int(report["false_rejects"]),
                    "test/f1_positive_class_attack": float(report["f1"]),
                    "test/count/videos": int(report["genuine"]) + int(report["attacks"]),
                }
            )
            tracker.log(metrics)
            tracker.summary({key: value for key, value in metrics.items()})
        finally:
            tracker.finish()
        complete = {
            "condition": condition_key,
            "seed": seed,
            "status": "completed",
            "threshold": float(entry["threshold"]),
            "frozen_plan_sha256": plan_hash,
            "refit_checkpoint_sha256": refit_complete["checkpoint_sha256"],
            "test_predictions_sha256": sha256_file(output_dir / "test_videos.jsonl"),
            "metrics": report,
        }
        write_json(completion_path, complete)
        results.append(complete)
        progress.update(index, f"{condition_key}, seed {seed}")
    result = {"jobs": len(results), "results": results}
    write_json(campaign.root / "test_results.json", result)
    return result
