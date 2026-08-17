import torch
from torch import Tensor, nn
from torch.nn import functional as F

_NEIGHBOR_KERNELS = torch.tensor(
    [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
    ],
    dtype=torch.float32,
).unsqueeze(1)


def contrast_depth(inputs: Tensor) -> Tensor:
    if inputs.ndim != 3:
        raise ValueError(f"Expected BxHxW depth maps, received {tuple(inputs.shape)}")
    expanded = inputs.unsqueeze(1).expand(-1, 8, -1, -1)
    kernels = _NEIGHBOR_KERNELS.to(device=inputs.device, dtype=inputs.dtype)
    return F.conv2d(expanded, kernels, groups=8)


class DepthLoss(nn.Module):
    def __init__(self, absolute_weight: float = 1.0, contrastive_weight: float = 1.0) -> None:
        super().__init__()
        self.absolute_weight = absolute_weight
        self.contrastive_weight = contrastive_weight

    def components(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        absolute = F.mse_loss(predictions, targets)
        contrastive = F.mse_loss(contrast_depth(predictions), contrast_depth(targets))
        total = self.absolute_weight * absolute + self.contrastive_weight * contrastive
        return {"total": total, "absolute": absolute, "contrastive": contrastive}

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.components(predictions, targets)["total"]
