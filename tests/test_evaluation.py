from pathlib import Path

import pytest

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.evaluation import (
    FramePrediction,
    VideoPrediction,
    aggregate_video_predictions,
    auc,
    evaluate_checkpoint,
    metric_report,
    select_threshold,
)


def _videos() -> list[VideoPrediction]:
    return [
        VideoPrediction("real-1", 1, 0.9, "genuine", "android"),
        VideoPrediction("real-2", 1, 0.8, "genuine", "laptop"),
        VideoPrediction("attack-1", 0, 0.1, "printed_photo", "android"),
        VideoPrediction("attack-2", 0, 0.2, "printed_photo", "laptop"),
    ]


def test_video_aggregation_averages_frame_scores() -> None:
    frames = [
        FramePrediction("video", 0, 1, 0.4, "genuine", "android"),
        FramePrediction("video", 1, 1, 0.8, "genuine", "android"),
    ]
    videos = aggregate_video_predictions(frames)
    assert len(videos) == 1
    assert videos[0].score == pytest.approx(0.6)


def test_perfect_predictions_have_zero_acer() -> None:
    videos = _videos()
    threshold = select_threshold(videos)
    report = metric_report(videos, threshold)
    assert 0.2 < threshold < 0.8
    assert report["acer"] == 0.0
    assert report["eer"] == 0.0
    assert auc(videos) == 1.0


def test_official_test_requires_frozen_threshold(tmp_path: Path) -> None:
    config = load_config(Path("configs/base.yaml"))
    with pytest.raises(ValueError, match="frozen validation threshold"):
        evaluate_checkpoint(config, tmp_path / "missing.pt", "test", tmp_path)


def test_official_test_requires_refit_configuration(tmp_path: Path) -> None:
    config = load_config(Path("configs/base.yaml"))
    with pytest.raises(ValueError, match="refit checkpoint"):
        evaluate_checkpoint(config, tmp_path / "missing.pt", "test", tmp_path, threshold=0.5)
