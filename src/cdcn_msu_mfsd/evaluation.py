import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from cdcn_msu_mfsd.config import Config, config_hash
from cdcn_msu_mfsd.data.dataset import MSUMFSDDataset, Sample
from cdcn_msu_mfsd.data.preprocessing import load_frame_records
from cdcn_msu_mfsd.data.protocol import load_folds
from cdcn_msu_mfsd.data.supervision import BinaryTarget
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.training import select_device


@dataclass(frozen=True)
class FramePrediction:
    video_id: str
    frame_index: int
    label: int
    score: float
    attack_type: str
    device: str


@dataclass(frozen=True)
class VideoPrediction:
    video_id: str
    label: int
    score: float
    attack_type: str
    device: str


def aggregate_video_predictions(frames: list[FramePrediction]) -> list[VideoPrediction]:
    grouped: dict[str, list[FramePrediction]] = defaultdict(list)
    for frame in frames:
        grouped[frame.video_id].append(frame)
    videos = []
    for video_id, group in sorted(grouped.items()):
        labels = {frame.label for frame in group}
        attack_types = {frame.attack_type for frame in group}
        devices = {frame.device for frame in group}
        if len(labels) != 1 or len(attack_types) != 1 or len(devices) != 1:
            raise ValueError(f"Inconsistent metadata for video {video_id}")
        videos.append(
            VideoPrediction(
                video_id,
                labels.pop(),
                sum(frame.score for frame in group) / len(group),
                attack_types.pop(),
                devices.pop(),
            )
        )
    return videos


def _rates(predictions: list[VideoPrediction], threshold: float) -> dict[str, float | int]:
    genuine = [item for item in predictions if item.label == 1]
    attacks = [item for item in predictions if item.label == 0]
    if not genuine or not attacks:
        raise ValueError("PAD metrics require both genuine and attack videos")
    false_accepts = sum(item.score >= threshold for item in attacks)
    false_rejects = sum(item.score < threshold for item in genuine)
    apcer = false_accepts / len(attacks)
    bpcer = false_rejects / len(genuine)
    correct = len(predictions) - false_accepts - false_rejects
    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "accuracy": correct / len(predictions),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "genuine": len(genuine),
        "attacks": len(attacks),
    }


def auc(predictions: list[VideoPrediction]) -> float:
    genuine = [item.score for item in predictions if item.label == 1]
    attacks = [item.score for item in predictions if item.label == 0]
    if not genuine or not attacks:
        raise ValueError("AUC requires both classes")
    favorable = sum(
        float(genuine_score > attack_score) + 0.5 * float(genuine_score == attack_score)
        for genuine_score in genuine
        for attack_score in attacks
    )
    return favorable / (len(genuine) * len(attacks))


def candidate_thresholds(predictions: list[VideoPrediction]) -> list[float]:
    scores = sorted({item.score for item in predictions})
    if not scores:
        raise ValueError("Cannot select a threshold without predictions")
    epsilon = max(1e-8, (scores[-1] - scores[0]) * 1e-8)
    middle = [(left + right) / 2 for left, right in zip(scores, scores[1:], strict=False)]
    return [scores[0] - epsilon, *middle, scores[-1] + epsilon]


def select_threshold(predictions: list[VideoPrediction]) -> float:
    return min(
        candidate_thresholds(predictions),
        key=lambda threshold: (
            float(_rates(predictions, threshold)["acer"]),
            -float(_rates(predictions, threshold)["accuracy"]),
            threshold,
        ),
    )


def eer(predictions: list[VideoPrediction]) -> float:
    pairs = [
        (
            abs(float(metrics["apcer"]) - float(metrics["bpcer"])),
            (float(metrics["apcer"]) + float(metrics["bpcer"])) / 2,
        )
        for threshold in candidate_thresholds(predictions)
        for metrics in [_rates(predictions, threshold)]
    ]
    return min(pairs)[1]


def metric_report(predictions: list[VideoPrediction], threshold: float) -> dict[str, Any]:
    report: dict[str, Any] = dict(_rates(predictions, threshold))
    report.update({"threshold": threshold, "auc": auc(predictions), "eer": eer(predictions)})
    attacks = sorted({item.attack_type for item in predictions if item.label == 0})
    report["by_attack_type"] = {
        attack: _rates(
            [item for item in predictions if item.label == 1 or item.attack_type == attack],
            threshold,
        )
        for attack in attacks
    }
    devices = sorted({item.device for item in predictions})
    report["by_device"] = {
        device: _rates([item for item in predictions if item.device == device], threshold)
        for device in devices
    }
    return report


def _evaluation_loader(config: Config, split: str, device: torch.device) -> DataLoader[Sample]:
    records = load_frame_records(config.paths.cache_root / "frames.jsonl")
    if split == "validation":
        fold = load_folds(config.paths.cache_root / "folds.json")[config.experiment.fold]
        subjects = set(fold.validation_subjects)
    elif split == "test":
        subjects = {record.subject for record in records if record.protocol == "test"}
    else:
        raise ValueError("split must be validation or test")
    dataset = MSUMFSDDataset(
        records,
        config.paths.cache_root,
        BinaryTarget(),
        subjects,
        False,
        config.data.frames_per_video,
        config.experiment.seed,
        config.data.crop_scale,
    )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.workers > 0,
    )


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader[Sample], device: torch.device
) -> list[FramePrediction]:
    model.eval()
    results = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        maps = cast(Tensor, model(images))
        scores = maps.mean(dim=(1, 2)).detach().cpu().tolist()
        labels = batch["label"].tolist()
        frame_indices = batch["frame_index"].tolist()
        for index, score in enumerate(scores):
            results.append(
                FramePrediction(
                    video_id=batch["video_id"][index],
                    frame_index=int(frame_indices[index]),
                    label=int(labels[index]),
                    score=float(score),
                    attack_type=batch["attack_type"][index],
                    device=batch["device"][index],
                )
            )
    return results


def evaluate_checkpoint(
    config: Config,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    if split == "test" and threshold is None:
        raise ValueError("Official test evaluation requires a frozen validation threshold")
    if split == "test" and config.experiment.phase not in {"refit", "test"}:
        raise ValueError("Official test evaluation requires a refit checkpoint configuration")
    device = select_device(config.runtime.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_hash") != config_hash(config):
        raise ValueError("Checkpoint configuration does not match evaluation configuration")
    model = build_model(config.model.name, config.model.theta).to(device)
    model.load_state_dict(checkpoint["model"])
    frames = predict(model, _evaluation_loader(config, split, device), device)
    videos = aggregate_video_predictions(frames)
    selected_threshold = select_threshold(videos) if threshold is None else threshold
    report = metric_report(videos, selected_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}_frames.jsonl").write_text(
        "\n".join(json.dumps(asdict(item), sort_keys=True) for item in frames) + "\n"
    )
    (output_dir / f"{split}_videos.jsonl").write_text(
        "\n".join(json.dumps(asdict(item), sort_keys=True) for item in videos) + "\n"
    )
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if split == "validation":
        (output_dir / "threshold.json").write_text(
            json.dumps({"threshold": selected_threshold}, indent=2) + "\n"
        )
    return report


def load_threshold(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Frozen threshold file does not exist: {path}")
    return float(json.loads(path.read_text())["threshold"])
