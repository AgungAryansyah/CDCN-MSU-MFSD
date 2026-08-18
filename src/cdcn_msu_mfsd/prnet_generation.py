import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cdcn_msu_mfsd.config import PRNetConfig
from cdcn_msu_mfsd.data.supervision import PRNetRecord
from cdcn_msu_mfsd.prnet_artifacts import artifact_paths, prnet_artifact_status
from cdcn_msu_mfsd.prnet_model import (
    load_prnet_checkpoint,
    normalize_prnet_inputs,
    resolve_prnet_device,
)


class PRNetInput(TypedDict):
    record_index: int
    image: Tensor


class PRNetInputDataset(Dataset[PRNetInput]):
    def __init__(self, records: list[PRNetRecord], contract_root: Path) -> None:
        self.records = records
        self.contract_root = contract_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PRNetInput:
        path = self.contract_root / self.records[index].input_path
        with Image.open(path) as source:
            image = source.convert("RGB").resize((256, 256))
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return {
            "record_index": index,
            "image": torch.from_numpy(array).permute(2, 0, 1),
        }


@dataclass(frozen=True)
class PRNetGenerationReport:
    total: int
    generated: int
    skipped: int
    output_root: str
    device: str
    contract_sha256: str
    checkpoint_sha256: str
    elapsed_seconds: float


def load_prnet_contract(path: Path) -> list[PRNetRecord]:
    records = [
        PRNetRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"PRNet contract is empty: {path}")
    return records


def rasterize_prnet_depth(position_maps: Tensor, face_indices: Tensor) -> Tensor:
    if position_maps.ndim != 4 or position_maps.shape[1] != 3:
        raise ValueError(f"Expected PRNet position maps shaped Bx3xHxW, got {position_maps.shape}")
    points = position_maps.flatten(2).index_select(2, face_indices) * 255.0
    x = (points[:, 0] * (31.0 / 255.0)).round().clamp(0, 31).long()
    y = (points[:, 1] * (31.0 / 255.0)).round().clamp(0, 31).long()
    z = points[:, 2].clamp_min(0)
    flat_depth = torch.zeros(
        (position_maps.shape[0], 32 * 32),
        dtype=position_maps.dtype,
        device=position_maps.device,
    )
    flat_depth.scatter_reduce_(1, y * 32 + x, z, reduce="amax", include_self=True)
    depth = flat_depth.reshape(-1, 32, 32)
    maximum = depth.amax(dim=(1, 2), keepdim=True)
    if torch.any(maximum <= 0):
        raise ValueError("PRNet produced an empty pseudo-depth map")
    return depth / maximum


def _valid_depth(path: Path) -> bool:
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return bool(
        depth.shape == (32, 32)
        and np.isfinite(depth).all()
        and float(depth.min()) >= 0.0
        and float(depth.max()) <= 1.0
        and float(depth.max()) > 0.0
    )


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        np.save(file, array)
    os.replace(temporary, path)


def _atomic_save_report(path: Path, report: PRNetGenerationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(payload)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def generate_prnet_outputs(
    contract_path: Path,
    config: PRNetConfig,
    limit: int | None = None,
    model: nn.Module | None = None,
) -> PRNetGenerationReport:
    started = time.monotonic()
    artifacts = artifact_paths(config)
    status = prnet_artifact_status(config)
    missing = [name for name, item in status.items() if not item["ready"]]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"PRNet artifacts are not ready ({names}); run fetch-prnet first")

    all_records = load_prnet_contract(contract_path)
    records = all_records[:limit] if limit is not None else all_records
    pending = [
        record
        for record in records
        if not _valid_depth(config.output_root / f"{record.output_stem}.npy")
    ]
    device = resolve_prnet_device(config.device)
    inference_model = model or load_prnet_checkpoint(artifacts["checkpoint"], device)
    inference_model.eval().to(device)
    face_indices = (
        torch.from_numpy(np.loadtxt(artifacts["face_indices"], dtype=np.int64)).flatten().to(device)
    )
    if (
        face_indices.numel() == 0
        or int(face_indices.min()) < 0
        or int(face_indices.max()) >= 256**2
    ):
        raise ValueError(f"Invalid PRNet face indices: {artifacts['face_indices']}")

    dataset = PRNetInputDataset(pending, contract_path.parent)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    generated = 0
    with torch.inference_mode():
        for batch in loader:
            indices = cast(Tensor, batch["record_index"])
            images = normalize_prnet_inputs(cast(Tensor, batch["image"]).to(device))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                positions = inference_model(images)
            depths = rasterize_prnet_depth(positions.float(), face_indices).cpu().numpy()
            for record_index, depth in zip(indices.tolist(), depths, strict=True):
                destination = config.output_root / f"{pending[record_index].output_stem}.npy"
                _atomic_save_array(destination, depth.astype(np.float32))
                generated += 1

    report = PRNetGenerationReport(
        total=len(records),
        generated=generated,
        skipped=len(records) - generated,
        output_root=str(config.output_root),
        device=str(device),
        contract_sha256=_sha256(contract_path),
        checkpoint_sha256=cast(str, status["checkpoint"]["sha256"]),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    _atomic_save_report(config.output_root / "generation.json", report)
    return report
