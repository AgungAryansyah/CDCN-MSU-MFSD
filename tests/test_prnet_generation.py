import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from cdcn_msu_mfsd.config import PRNetConfig
from cdcn_msu_mfsd.prnet_generation import (
    generate_prnet_outputs,
    load_prnet_face_indices,
    rasterize_prnet_depth,
)


class FakePRNet(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        size = inputs.shape[-1]
        coordinates = torch.linspace(0, 1, size, device=inputs.device)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        position = torch.stack((x, y, torch.ones_like(x)))
        return position.expand(inputs.shape[0], -1, -1, -1)


def test_rasterizes_position_map_to_normalized_depth() -> None:
    position = FakePRNet()(torch.zeros(1, 3, 256, 256))
    face_indices = torch.arange(256**2)

    depth = rasterize_prnet_depth(position, face_indices)

    assert depth.shape == (1, 32, 32)
    assert torch.isfinite(depth).all()
    assert depth.min() >= 0
    assert depth.max() == pytest.approx(1.0)


def test_loads_scientific_notation_face_indices(tmp_path: Path) -> None:
    path = tmp_path / "face_ind.txt"
    path.write_text("1.285000000000000000e+03\n6.553500000000000000e+04\n")

    indices = load_prnet_face_indices(path, torch.device("cpu"))

    assert indices.dtype == torch.int64
    assert indices.tolist() == [1285, 65535]


def test_rejects_fractional_face_indices(tmp_path: Path) -> None:
    path = tmp_path / "face_ind.txt"
    path.write_text("1.5\n")

    with pytest.raises(ValueError, match="finite integers"):
        load_prnet_face_indices(path, torch.device("cpu"))


def test_generation_resumes_and_replaces_invalid_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = tmp_path / "export"
    input_path = export_root / "inputs" / "sample.png"
    input_path.parent.mkdir(parents=True)
    Image.new("RGB", (256, 256), color=(127, 127, 127)).save(input_path)
    contract_path = export_root / "contract.jsonl"
    contract_path.write_text(
        json.dumps(
            {
                "video_id": "real_client001_android_SD_scene01",
                "frame_index": 0,
                "input_path": "inputs/sample.png",
                "output_stem": "real_client001_android_SD_scene01__000000",
            }
        )
        + "\n"
    )
    model_root = tmp_path / "model"
    model_root.mkdir()
    checkpoint = model_root / "latest.pth"
    checkpoint.write_bytes(b"checkpoint")
    np.savetxt(model_root / "face_ind.txt", np.arange(256**2), fmt="%d")
    config = PRNetConfig(
        model_root=model_root,
        export_root=export_root,
        output_root=tmp_path / "outputs",
        batch_size=1,
        workers=0,
        device="cpu",
    )
    monkeypatch.setattr(
        "cdcn_msu_mfsd.prnet_generation.prnet_artifact_status",
        lambda _: {
            "checkpoint": {"ready": True, "sha256": "checkpoint-sha256"},
            "face_indices": {"ready": True, "sha256": "indices-sha256"},
        },
    )

    first = generate_prnet_outputs(contract_path, config, model=FakePRNet())
    second = generate_prnet_outputs(contract_path, config, model=FakePRNet())
    output = config.output_root / "real_client001_android_SD_scene01__000000.npy"
    output.write_bytes(b"invalid")
    third = generate_prnet_outputs(contract_path, config, model=FakePRNet())

    assert (first.generated, first.skipped) == (1, 0)
    assert (second.generated, second.skipped) == (0, 1)
    assert (third.generated, third.skipped) == (1, 0)
    assert np.load(output).shape == (32, 32)
    report = json.loads((config.output_root / "generation.json").read_text())
    assert report["checkpoint_sha256"] == "checkpoint-sha256"
