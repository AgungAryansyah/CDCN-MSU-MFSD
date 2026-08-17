from pathlib import Path

import pytest

from cdcn_msu_mfsd.data.manifest import build_manifest, manifest_summary, parse_video_name
from cdcn_msu_mfsd.data.protocol import make_folds


def test_parse_genuine_name() -> None:
    assert parse_video_name("real_client001_android_SD_scene01") == ("001", 1, None, "android")


def test_parse_attack_name() -> None:
    assert parse_video_name("attack_client001_laptop_SD_ipad_video_scene01") == (
        "001",
        0,
        "ipad_video",
        "laptop",
    )


@pytest.mark.dataset
def test_local_dataset_manifest() -> None:
    root = Path("../data/raw/scene01")
    if not root.exists():
        pytest.skip("MSU-MFSD is not available")
    records = build_manifest(
        root,
        Path("../data/raw/train_sub_list.txt"),
        Path("../data/raw/test_sub_list.txt"),
        probe=False,
    )
    summary = manifest_summary(records)
    assert summary["videos"] == 280
    assert summary["subjects"] == 35
    assert summary["genuine"] == 70
    assert summary["attacks"] == 210
    assert summary["attack_types"] == {
        "ipad_video": 70,
        "iphone_video": 70,
        "printed_photo": 70,
    }
    folds = make_folds(records)
    assert len(folds) == 5
    assert all(len(fold.validation_subjects) == 3 for fold in folds)

