from pathlib import Path

import pytest

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.tracking import create_tracker


def test_disabled_tracking_writes_local_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    tracker = create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {})
    tracker.log({"loss": 1.0}, step=2)
    tracker.summary({"best_epoch": 2})
    assert '"loss": 1.0' in (tmp_path / "metrics.jsonl").read_text()
    assert '"best_epoch": 2' in (tmp_path / "summary.json").read_text()


def test_online_tracking_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {})
