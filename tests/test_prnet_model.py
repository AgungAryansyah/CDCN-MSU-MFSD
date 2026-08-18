from pathlib import Path

import pytest
import torch

from cdcn_msu_mfsd.prnet_model import (
    PRNet,
    load_prnet_checkpoint,
    normalize_prnet_inputs,
    resolve_prnet_device,
)


def test_prnet_preserves_spatial_shape_and_bounds() -> None:
    model = PRNet(size=1).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 32))
    assert output.shape == (1, 3, 32, 32)
    assert 0.0 <= output.min() <= output.max() <= 1.0


def test_prnet_input_normalization_uses_checkpoint_bgr_order() -> None:
    image = torch.tensor([[[[1.0]], [[0.5]], [[0.0]]]])
    normalized = normalize_prnet_inputs(image)
    assert normalized[0, 0, 0, 0] == pytest.approx(-0.485 / 0.229)
    assert normalized[0, 2, 0, 0] == pytest.approx((1.0 - 0.406) / 0.225)


def test_loads_checkpoint_with_original_module_names(tmp_path: Path) -> None:
    model = PRNet(size=1)
    state = {}
    for key, value in model.state_dict().items():
        original_key = key.replace(".shortcut.", ".shortcut_conv.").replace(
            ".normalizer.", ".normalizer_fn."
        )
        state[original_key] = value
    checkpoint = tmp_path / "latest.pth"
    torch.save({"prnet": state}, checkpoint)

    loaded = load_prnet_checkpoint(checkpoint, torch.device("cpu"), size=1)

    assert not loaded.training
    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_cuda_request_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_prnet_device("cuda")
