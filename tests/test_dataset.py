from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cdcn_msu_mfsd.data.dataset import MSUMFSDDataset
from cdcn_msu_mfsd.data.preprocessing import FrameRecord
from cdcn_msu_mfsd.data.supervision import (
    BinaryTarget,
    PseudoDepthTarget,
    import_prnet_outputs,
    validate_pseudo_depth_targets,
)


def _record(label: int = 1) -> FrameRecord:
    return FrameRecord(
        video_id=(
            "real_client001_android_SD_scene01" if label else "attack_client001_android_SD_scene01"
        ),
        frame_index=0,
        frame_path="frames/video/000000.jpg",
        subject="001",
        label=label,
        attack_type=None if label else "printed_photo",
        device="android",
        protocol="train",
        left=20,
        top=20,
        right=80,
        bottom=80,
    )


def test_binary_target_matches_label() -> None:
    assert BinaryTarget()(_record(1)).mean().item() == 1.0
    assert BinaryTarget()(_record(0)).mean().item() == 0.0


def test_dataset_returns_normalized_sample(tmp_path: Path) -> None:
    frame_path = tmp_path / "frames/video/000000.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(frame_path)
    dataset = MSUMFSDDataset(
        [_record()], tmp_path, BinaryTarget(), {"001"}, False, frames_per_video=1
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["target"].shape == (32, 32)
    assert sample["image"].abs().max().item() < 0.01


def test_imported_prnet_target_is_normalized(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.save(output_root / "real_client001_android_SD_scene01__000000.npy", source)
    contract = tmp_path / "contract.jsonl"
    contract.write_text(
        '{"frame_index": 0, "input_path": "input.png", '
        '"output_stem": "real_client001_android_SD_scene01__000000", '
        '"video_id": "real_client001_android_SD_scene01"}\n'
    )
    depth_root = tmp_path / "depth"
    assert import_prnet_outputs(contract, output_root, depth_root) == 1
    target = PseudoDepthTarget(depth_root)(_record())
    assert target.shape == (32, 32)
    assert 0.0 <= target.min().item() <= target.max().item() <= 1.0


def test_import_validates_all_sources_before_writing(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    np.save(output_root / "video__000000.npy", np.ones((32, 32), dtype=np.float32))
    contract = tmp_path / "contract.jsonl"
    contract.write_text(
        '{"frame_index": 0, "input_path": "first.png", '
        '"output_stem": "video__000000", "video_id": "video"}\n'
        '{"frame_index": 1, "input_path": "second.png", '
        '"output_stem": "video__000001", "video_id": "video"}\n'
    )
    depth_root = tmp_path / "depth"

    with pytest.raises(FileNotFoundError, match="video__000001"):
        import_prnet_outputs(contract, output_root, depth_root)

    assert not depth_root.exists()


def test_pseudo_depth_preflight_hashes_genuine_targets(tmp_path: Path) -> None:
    record = _record()
    destination = tmp_path / record.video_id / "000000.npy"
    destination.parent.mkdir(parents=True)
    np.save(destination, np.ones((32, 32), dtype=np.float32))

    validation = validate_pseudo_depth_targets([record, _record(0)], tmp_path)

    assert validation.targets == 1
    assert len(validation.sha256) == 64

    np.save(destination, np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="1 invalid"):
        validate_pseudo_depth_targets([replace(record, frame_index=0)], tmp_path)
