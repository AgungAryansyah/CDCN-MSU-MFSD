import pytest
import torch
from torch import nn

from cdcn_msu_mfsd.models import CDCN, CDCNPP, CentralDifferenceConv2d, build_model


def test_theta_zero_matches_regular_convolution() -> None:
    layer = CentralDifferenceConv2d(3, 4, theta=0.0)
    inputs = torch.randn(2, 3, 16, 16)
    assert torch.equal(layer(inputs), layer.conv(inputs))


@pytest.mark.parametrize("model_type", [CDCN, CDCNPP])
def test_model_output_shape_and_gradient(model_type: type[nn.Module]) -> None:
    model = model_type()
    inputs = torch.randn(1, 3, 256, 256, requires_grad=True)
    outputs = model(inputs)
    assert outputs.shape == (1, 32, 32)
    outputs.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    ("model_type", "parameter_count"),
    [(CDCN, 2_245_016), (CDCNPP, 2_257_352)],
)
def test_model_parameter_count(model_type: type[nn.Module], parameter_count: int) -> None:
    assert sum(parameter.numel() for parameter in model_type().parameters()) == parameter_count


def test_model_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown")
