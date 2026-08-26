from pathlib import Path

import pytest

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.tracking import create_tracker, evaluation_metrics


def test_disabled_tracking_writes_local_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    tracker = create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {})
    tracker.log({"loss": 1.0}, step=2)
    tracker.summary({"best_epoch": 2})
    tracker.summary({"validation/acer": 0.1})
    assert '"loss": 1.0' in (tmp_path / "metrics.jsonl").read_text()
    summary = (tmp_path / "summary.json").read_text()
    assert '"best_epoch": 2' in summary
    assert '"validation/acer": 0.1' in summary


def test_online_tracking_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {})


def test_evaluation_metrics_include_pad_rates_and_confusion() -> None:
    metrics = evaluation_metrics(
        "validation",
        {
            "acer": 0.15,
            "apcer": 0.1,
            "bpcer": 0.2,
            "eer": 0.12,
            "auc": 0.93,
            "f1": 0.8,
            "accuracy": 0.85,
            "threshold": 0.47,
            "false_accepts": 2,
            "false_rejects": 1,
            "genuine": 5,
            "attacks": 20,
        },
    )

    assert metrics["validation/roc_auc"] == 0.93
    assert metrics["validation/f1_attack"] == 0.8
    assert metrics["validation/confusion/true_positive"] == 4
    assert metrics["validation/confusion/true_negative"] == 18
    assert metrics["validation/confusion/false_positive"] == 2
    assert metrics["validation/confusion/false_negative"] == 1
