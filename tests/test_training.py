from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.tracking import MetricsTracker
from cdcn_msu_mfsd.training import Trainer


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
