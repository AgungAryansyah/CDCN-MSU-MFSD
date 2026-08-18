from pathlib import Path

import pytest
from PIL import Image

from cdcn_msu_mfsd.data.manifest import VideoRecord
from cdcn_msu_mfsd.data.preprocessing import (
    FaceAnnotation,
    FrameDecodeError,
    crop_box,
    decode_frame,
    load_annotations,
    prepare_frames,
    select_uniform,
)


def test_uniform_selection_includes_endpoints() -> None:
    assert select_uniform(list(range(10)), 3) == [0, 4, 9]


def test_uniform_selection_can_return_one_item() -> None:
    assert select_uniform(list(range(10)), 1) == [5]


def test_crop_box_clamps_to_image() -> None:
    annotation = FaceAnnotation(0, 5, 10, 105, 130, 30.0, 40.0, 80.0, 40.0)
    left, top, right, bottom = crop_box(annotation, 720, 480, 1.5)
    assert 0 <= left < right <= 720
    assert 0 <= top < bottom <= 480


def test_decode_frame_reports_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cdcn_msu_mfsd.data.preprocessing.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg is required for frame extraction"):
        decode_frame(Path("video.mp4"), 0)


def test_prepare_frames_replaces_trailing_undecodable_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "raw"
    annotation_path = Path("attack/video.face")
    source = dataset_root / annotation_path
    source.parent.mkdir(parents=True)
    source.write_text("\n".join(f"{index}, 1, 1, 7, 7, 2, 3, 6, 3" for index in range(5)) + "\n")
    record = VideoRecord(
        video_id="video",
        video_path="attack/video.mp4",
        annotation_path=str(annotation_path),
        subject="001",
        label=0,
        attack_type="printed_photo",
        device="android",
        protocol="train",
        width=0,
        height=0,
        frame_count=0,
        rotation=0,
    )
    decoded_indices = []

    def fake_decode_frame(video_path: Path, frame_index: int) -> Image.Image:
        decoded_indices.append(frame_index)
        if frame_index == 4:
            raise FrameDecodeError("missing trailing frame")
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(
        "cdcn_msu_mfsd.data.preprocessing.decode_frame",
        fake_decode_frame,
    )
    frame_records, _ = prepare_frames(
        [record],
        dataset_root,
        tmp_path / "cache",
        candidates_per_video=3,
    )

    assert decoded_indices == [0, 2, 4, 3]
    assert [frame.frame_index for frame in frame_records] == [0, 2, 3]


@pytest.mark.dataset
def test_rotated_video_decodes_in_display_orientation() -> None:
    path = Path("data/raw/scene01/real/real_client003_android_SD_scene01.mp4")
    if not path.exists():
        pytest.skip("MSU-MFSD is not available")
    frame = decode_frame(path, 0)
    annotation = load_annotations(path.with_suffix(".face"))[0]
    face = frame.crop(crop_box(annotation, frame.width, frame.height, 1.0))
    assert frame.size == (720, 480)
    assert face.width > 0 and face.height > 0
