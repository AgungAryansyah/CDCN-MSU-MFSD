import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cdcn_msu_mfsd import evaluation, training
from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.data.preprocessing import FrameRecord
from cdcn_msu_mfsd.data.protocol import Fold
from cdcn_msu_mfsd.data.supervision import PseudoDepthValidation
from cdcn_msu_mfsd.tracking import MetricsTracker
from cdcn_msu_mfsd.training import Trainer, provenance, run_training, split_metadata


class TinyDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": torch.full((3, 16, 16), index / 2),
            "target": torch.full((32, 32), index / 2),
            "label": torch.tensor(index),
            "video_id": f"video-{index}",
            "frame_index": index,
            "attack_type": "genuine",
            "device": "android",
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.interpolate(self.conv(inputs), size=(32, 32)).squeeze(1)


def test_cpu_trainer_writes_resumable_checkpoints(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/experiments/smoke.yaml"),
        ["runtime.device=cpu", "runtime.amp=false", "tracking.log_every_steps=1"],
    )
    loader = DataLoader(TinyDataset(), batch_size=2)
    tracker = MetricsTracker(tmp_path / "metrics.jsonl")
    trainer = Trainer(config, TinyModel(), loader, loader, tracker, torch.device("cpu"))
    result = trainer.fit(tmp_path)
    assert result.epochs_completed == 1
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()

    resumed = Trainer(config, TinyModel(), loader, loader, tracker, torch.device("cpu"))
    resumed_result = resumed.fit(tmp_path, tmp_path / "last.pt")
    assert resumed_result.epochs_completed == 1
    assert resumed_result.best_epoch == result.best_epoch
    assert resumed_result.best_validation_loss == result.best_validation_loss


def test_provenance_includes_pseudo_depth_metadata(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    depth_root = tmp_path / "depth"
    depth_root.mkdir()
    (depth_root / "provenance.json").write_text(
        json.dumps(
            {
                "contract_sha256": "contract",
                "generation_sha256": "generation",
                "checkpoint_sha256": "checkpoint",
            }
        )
    )
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"depth_root": depth_root})}
    )

    metadata = provenance(config, PseudoDepthValidation(10, "targets"))

    assert metadata["pseudo_depth_targets"] == "10"
    assert metadata["pseudo_depth_sha256"] == "targets"
    assert metadata["pseudo_depth_checkpoint_sha256"] == "checkpoint"


def test_split_metadata_records_subjects_and_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    records = [
        FrameRecord("train-video", 0, "frame.jpg", "001", 1, None, "android", "train", 0, 0, 1, 1),
        FrameRecord(
            "validation-video",
            0,
            "frame.jpg",
            "002",
            0,
            "printed_photo",
            "laptop",
            "train",
            0,
            0,
            1,
            1,
        ),
        FrameRecord(
            "test-video", 0, "frame.jpg", "003", 0, "ipad_video", "android", "test", 0, 0, 1, 1
        ),
    ]
    monkeypatch.setattr(
        training,
        "load_folds",
        lambda _: [Fold(0, ("001",), ("002",))],
    )

    split = split_metadata(config, records)

    assert split["train_subjects"] == ["001"]
    assert split["validation_subjects"] == ["002"]
    assert split["test_subjects"] == ["003"]
    assert split["validation_videos"] == ["validation-video"]


def test_training_logs_best_checkpoint_pad_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(
        Path("configs/experiments/smoke.yaml"),
        ["runtime.device=cpu", "runtime.amp=false", "tracking.log_every_steps=1"],
    )
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"output_root": tmp_path / "outputs"})}
    )
    loader = DataLoader(TinyDataset(), batch_size=2)
    monkeypatch.setattr(training, "make_dataloaders", lambda *_: (loader, loader))
    monkeypatch.setattr(training, "load_frame_records", lambda _: [])
    monkeypatch.setattr(
        training,
        "split_metadata",
        lambda *_: {
            "train_subjects": ["001"],
            "validation_subjects": ["002"],
            "test_subjects": ["003"],
            "train_videos": ["train-video"],
            "validation_videos": ["validation-video"],
            "test_videos": ["test-video"],
        },
    )
    monkeypatch.setattr(training, "build_model", lambda *_: TinyModel())
    monkeypatch.setattr(
        training,
        "create_tracker",
        lambda _config, output_dir, _provenance: MetricsTracker(output_dir / "metrics.jsonl"),
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_checkpoint",
        lambda *_: {
            "acer": 0.15,
            "apcer": 0.1,
            "bpcer": 0.2,
            "eer": 0.12,
            "auc": 0.93,
            "accuracy": 0.85,
            "threshold": 0.47,
            "false_accepts": 2,
            "false_rejects": 1,
            "genuine": 5,
            "attacks": 20,
        },
    )

    result = run_training(config)
    summary = json.loads((result.output_dir / "summary.json").read_text())

    assert summary["validation/acer"] == 0.15
    assert summary["validation/roc_auc"] == 0.93
    assert summary["validation/confusion/false_positive"] == 2
    assert json.loads((result.output_dir / "split.json").read_text())["test_subjects"] == ["003"]
