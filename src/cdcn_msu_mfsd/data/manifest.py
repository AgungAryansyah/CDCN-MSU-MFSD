import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_VIDEO_PATTERN = re.compile(
    r"^(?P<label>real|attack)_client(?P<subject>\d{3})_"
    r"(?P<device>android|laptop)_SD"
    r"(?:_(?P<attack>printed_photo|iphone_video|ipad_video))?_scene01$"
)


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    video_path: str
    annotation_path: str
    subject: str
    label: int
    attack_type: str | None
    device: str
    protocol: str
    width: int
    height: int
    frame_count: int
    rotation: int


def _subjects(path: Path) -> set[str]:
    return {f"{int(line.strip()):03d}" for line in path.read_text().splitlines() if line.strip()}


def parse_video_name(stem: str) -> tuple[str, int, str | None, str]:
    match = _VIDEO_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognized MSU-MFSD filename: {stem}")
    label = int(match.group("label") == "real")
    attack = match.group("attack")
    if label == 0 and attack is None:
        raise ValueError(f"Attack type is missing from {stem}")
    return match.group("subject"), label, attack, match.group("device")


def probe_video(path: Path) -> tuple[int, int, int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    side_data = stream.get("side_data_list", [])
    rotation = next((int(item["rotation"]) for item in side_data if "rotation" in item), 0)
    return int(stream["width"]), int(stream["height"]), int(stream["nb_frames"]), rotation


def build_manifest(
    dataset_root: Path,
    train_subjects_path: Path,
    test_subjects_path: Path,
    probe: bool = True,
) -> list[VideoRecord]:
    train_subjects = _subjects(train_subjects_path)
    test_subjects = _subjects(test_subjects_path)
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(f"Official protocols overlap: {sorted(overlap)}")
    records: list[VideoRecord] = []
    for annotation_path in sorted(dataset_root.glob("*/*.face")):
        stem = annotation_path.stem
        subject, label, attack_type, device = parse_video_name(stem)
        candidates = [annotation_path.with_suffix(".mp4"), annotation_path.with_suffix(".mov")]
        video_path = next((path for path in candidates if path.exists()), None)
        if video_path is None:
            raise FileNotFoundError(f"Missing video for {annotation_path}")
        if subject in train_subjects:
            protocol = "train"
        elif subject in test_subjects:
            protocol = "test"
        else:
            raise ValueError(f"Subject {subject} is absent from the official protocol lists")
        width, height, frame_count, rotation = probe_video(video_path) if probe else (0, 0, 0, 0)
        records.append(
            VideoRecord(
                video_id=stem,
                video_path=str(video_path.relative_to(dataset_root)),
                annotation_path=str(annotation_path.relative_to(dataset_root)),
                subject=subject,
                label=label,
                attack_type=attack_type,
                device=device,
                protocol=protocol,
                width=width,
                height=height,
                frame_count=frame_count,
                rotation=rotation,
            )
        )
    validate_manifest(records)
    return records


def validate_manifest(records: Iterable[VideoRecord]) -> None:
    materialized = list(records)
    ids = [record.video_id for record in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate video IDs")
    for record in materialized:
        if record.label == 1 and record.attack_type is not None:
            raise ValueError(f"Genuine video has an attack type: {record.video_id}")
        if record.label == 0 and record.attack_type is None:
            raise ValueError(f"Attack video has no attack type: {record.video_id}")


def save_manifest(records: Iterable[VideoRecord], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), sort_keys=True) for record in records]
    payload = "\n".join(lines) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_manifest(path: Path) -> list[VideoRecord]:
    records = [VideoRecord(**json.loads(line)) for line in path.read_text().splitlines() if line]
    validate_manifest(records)
    return records


def manifest_summary(records: Iterable[VideoRecord]) -> dict[str, Any]:
    materialized = list(records)
    return {
        "videos": len(materialized),
        "subjects": len({record.subject for record in materialized}),
        "genuine": sum(record.label for record in materialized),
        "attacks": sum(1 - record.label for record in materialized),
        "protocols": dict(Counter(record.protocol for record in materialized)),
        "attack_types": dict(
            Counter(record.attack_type for record in materialized if not record.label)
        ),
        "devices": dict(Counter(record.device for record in materialized)),
        "rotations": dict(Counter(str(record.rotation) for record in materialized)),
    }
