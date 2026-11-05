import json
from pathlib import Path
from typing import Any

import pytest

from cdcn_msu_mfsd import tracking
from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.tracking import (
    MetricsTracker,
    TrackingContext,
    create_tracker,
    evaluation_metrics,
)


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


def test_pipeline_wandb_run_resumes_from_saved_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    class FakeRun:
        id = "run12345"
        name = "fit-condition"
        url = "https://wandb.test/run12345"

        def __init__(self) -> None:
            self.logs = []
            self.summary = self

        def log(self, metrics: dict[str, int | float], **kwargs: Any) -> None:
            self.logs.append((metrics, kwargs))

        def update(self, metrics: dict[str, int | float | str]) -> None:
            self.logs.append((metrics, {}))

        def define_metric(self, *args: Any, **kwargs: Any) -> None:
            return None

        def finish(self) -> None:
            return None

    fake_run = FakeRun()

    def fake_init(**kwargs: Any) -> FakeRun:
        calls.append(kwargs)
        fake_run.id = str(kwargs["id"])
        return fake_run

    monkeypatch.setattr(tracking.wandb, "init", fake_init)
    context = TrackingContext(
        name="fit-condition",
        group="campaign",
        job_type="fit",
        required_online=True,
        custom_step_metric=True,
    )

    first = create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {}, context)
    first.log({"train/loss": 0.5}, step=12)
    first.finish()
    second = create_tracker(load_config(Path("configs/base.yaml")), tmp_path, {}, context)
    second.finish()

    assert calls[0]["id"] == calls[1]["id"]
    assert calls[0]["resume"] == "never"
    assert calls[1]["resume"] == "must"
    assert "step" not in fake_run.logs[0][1]
    assert fake_run.logs[0][0]["pipeline/global_step"] == 12
    metadata = json.loads((tmp_path / "wandb_run.json").read_text())
    assert metadata["status"] == "finished"


def test_required_wandb_failure_keeps_local_metric_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracking.time, "sleep", lambda _: None)

    class BrokenRun:
        def log(self, *_: Any, **__: Any) -> None:
            raise ConnectionError("offline")

    tracker = MetricsTracker(tmp_path / "metrics.jsonl", BrokenRun(), required_online=True)

    with pytest.raises(RuntimeError, match="after three attempts"):
        tracker.log({"train/loss": 0.5}, step=1)

    assert '"train/loss": 0.5' in (tmp_path / "metrics.jsonl").read_text()
