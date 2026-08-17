from pathlib import Path

import pytest

from cdcn_msu_mfsd.data.preprocessing import (
    FaceAnnotation,
    crop_box,
    decode_frame,
    load_annotations,
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


@pytest.mark.dataset
def test_rotated_video_decodes_in_display_orientation() -> None:
    path = Path("../data/raw/scene01/real/real_client003_android_SD_scene01.mp4")
    if not path.exists():
        pytest.skip("MSU-MFSD is not available")
    frame = decode_frame(path, 0)
    annotation = load_annotations(path.with_suffix(".face"))[0]
    face = frame.crop(crop_box(annotation, frame.width, frame.height, 1.0))
    assert frame.size == (720, 480)
    assert face.width > 0 and face.height > 0
