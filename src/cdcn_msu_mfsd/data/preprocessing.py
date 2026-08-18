import hashlib
import io
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from cdcn_msu_mfsd.data.manifest import VideoRecord


@dataclass(frozen=True)
class FaceAnnotation:
    frame_index: int
    left: int
    top: int
    right: int
    bottom: int
    left_eye_x: float
    left_eye_y: float
    right_eye_x: float
    right_eye_y: float


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_index: int
    frame_path: str
    subject: str
    label: int
    attack_type: str | None
    device: str
    protocol: str
    left: int
    top: int
    right: int
    bottom: int


def load_annotations(path: Path) -> list[FaceAnnotation]:
    annotations = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 9:
            raise ValueError(f"{path}:{line_number} expected 9 values, received {len(values)}")
        annotations.append(
            FaceAnnotation(
                frame_index=int(values[0]),
                left=int(values[1]),
                top=int(values[2]),
                right=int(values[3]),
                bottom=int(values[4]),
                left_eye_x=float(values[5]),
                left_eye_y=float(values[6]),
                right_eye_x=float(values[7]),
                right_eye_y=float(values[8]),
            )
        )
    if not annotations:
        raise ValueError(f"No face annotations found in {path}")
    return annotations


def select_uniform[T](items: list[T], count: int) -> list[T]:
    if count <= 0:
        raise ValueError("Frame count must be positive")
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def crop_box(
    annotation: FaceAnnotation | FrameRecord,
    image_width: int,
    image_height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    if scale <= 0:
        raise ValueError("Crop scale must be positive")
    width = annotation.right - annotation.left
    height = annotation.bottom - annotation.top
    side = max(width, height) * scale
    center_x = (annotation.left + annotation.right) / 2
    center_y = (annotation.top + annotation.bottom) / 2
    left = max(0, round(center_x - side / 2))
    top = max(0, round(center_y - side / 2))
    right = min(image_width, round(center_x + side / 2))
    bottom = min(image_height, round(center_y + side / 2))
    if right <= left or bottom <= top:
        raise ValueError("Face crop is empty after clamping")
    return left, top, right, bottom


def decode_frame(video_path: Path, frame_index: int) -> Image.Image:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for frame extraction. "
            "Install the FFmpeg tools before running prepare."
        )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    if not result.stdout:
        raise ValueError(f"FFmpeg did not decode frame {frame_index} from {video_path}")
    with Image.open(io.BytesIO(result.stdout)) as image:
        return image.convert("RGB")


def prepare_frames(
    records: list[VideoRecord],
    dataset_root: Path,
    cache_root: Path,
    candidates_per_video: int = 32,
    limit: int | None = None,
) -> tuple[list[FrameRecord], str]:
    frame_records: list[FrameRecord] = []
    selected_records = records[:limit] if limit is not None else records
    for video in selected_records:
        annotations = load_annotations(dataset_root / video.annotation_path)
        for annotation in select_uniform(annotations, candidates_per_video):
            relative_path = Path("frames") / video.video_id / f"{annotation.frame_index:06d}.jpg"
            destination = cache_root / relative_path
            if not destination.exists():
                frame = decode_frame(dataset_root / video.video_path, annotation.frame_index)
                crop_box(annotation, frame.width, frame.height, 1.0)
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.save(destination, quality=95)
            frame_records.append(
                FrameRecord(
                    video_id=video.video_id,
                    frame_index=annotation.frame_index,
                    frame_path=str(relative_path),
                    subject=video.subject,
                    label=video.label,
                    attack_type=video.attack_type,
                    device=video.device,
                    protocol=video.protocol,
                    left=annotation.left,
                    top=annotation.top,
                    right=annotation.right,
                    bottom=annotation.bottom,
                )
            )
    manifest_path = cache_root / "frames.jsonl"
    lines = [json.dumps(asdict(record), sort_keys=True) for record in frame_records]
    payload = "\n".join(lines) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(payload)
    return frame_records, hashlib.sha256(payload.encode()).hexdigest()


def load_frame_records(path: Path) -> list[FrameRecord]:
    return [FrameRecord(**json.loads(line)) for line in path.read_text().splitlines() if line]
