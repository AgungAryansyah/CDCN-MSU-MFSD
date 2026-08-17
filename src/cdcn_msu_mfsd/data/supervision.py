import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from cdcn_msu_mfsd.data.preprocessing import FrameRecord, crop_box


class BinaryTarget:
    def __call__(self, record: FrameRecord) -> Tensor:
        return torch.full((32, 32), float(record.label), dtype=torch.float32)


class PseudoDepthTarget:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self, record: FrameRecord) -> Tensor:
        if record.label == 0:
            return torch.zeros((32, 32), dtype=torch.float32)
        path = self.root / record.video_id / f"{record.frame_index:06d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing pseudo-depth target: {path}")
        target = torch.from_numpy(np.load(path).astype(np.float32))
        if target.shape != (32, 32):
            raise ValueError(
                f"Expected a 32x32 pseudo-depth map at {path}, received {target.shape}"
            )
        return target


@dataclass(frozen=True)
class PRNetRecord:
    video_id: str
    frame_index: int
    input_path: str
    output_stem: str


def export_prnet_inputs(
    records: list[FrameRecord],
    cache_root: Path,
    export_root: Path,
    crop_scale: float = 1.3,
) -> list[PRNetRecord]:
    contract = []
    for record in records:
        if record.label == 0:
            continue
        output_stem = f"{record.video_id}__{record.frame_index:06d}"
        relative_path = Path("inputs") / f"{output_stem}.png"
        destination = export_root / relative_path
        if not destination.exists():
            with Image.open(cache_root / record.frame_path) as source:
                image = source.convert("RGB")
            face = image.crop(crop_box(record, image.width, image.height, crop_scale))
            destination.parent.mkdir(parents=True, exist_ok=True)
            face.resize((256, 256)).save(destination)
        contract.append(
            PRNetRecord(
                video_id=record.video_id,
                frame_index=record.frame_index,
                input_path=str(relative_path),
                output_stem=output_stem,
            )
        )
    payload = "\n".join(json.dumps(asdict(item), sort_keys=True) for item in contract) + "\n"
    export_root.mkdir(parents=True, exist_ok=True)
    (export_root / "contract.jsonl").write_text(payload)
    return contract


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("F"))
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D depth map at {path}, received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Pseudo-depth map contains non-finite values: {path}")
    array = np.clip(array, 0, None)
    maximum = float(array.max())
    if maximum > 0:
        array /= maximum
    depth_image = Image.fromarray(array, mode="F").resize((32, 32), Image.Resampling.BILINEAR)
    return np.asarray(depth_image, dtype=np.float32)


def import_prnet_outputs(contract_path: Path, output_root: Path, depth_root: Path) -> int:
    imported = 0
    for line in contract_path.read_text().splitlines():
        if not line:
            continue
        record = PRNetRecord(**json.loads(line))
        candidates = [
            output_root / f"{record.output_stem}{suffix}"
            for suffix in (".npy", ".png", ".jpg")
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Missing PRNet output for {record.output_stem}")
        destination = depth_root / record.video_id / f"{record.frame_index:06d}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, _load_depth(source))
        imported += 1
    return imported
