import pytest
import torch

from cdcn_msu_mfsd.losses import DepthLoss, contrast_depth


def test_contrast_depth_shape() -> None:
    outputs = contrast_depth(torch.zeros(2, 32, 32))
    assert outputs.shape == (2, 8, 30, 30)


def test_depth_loss_is_zero_for_equal_maps() -> None:
    maps = torch.rand(2, 32, 32)
    components = DepthLoss().components(maps, maps)
    assert components["total"].item() == pytest.approx(0.0)


def test_contrast_depth_validates_shape() -> None:
    with pytest.raises(ValueError, match="BxHxW"):
        contrast_depth(torch.zeros(32, 32))
