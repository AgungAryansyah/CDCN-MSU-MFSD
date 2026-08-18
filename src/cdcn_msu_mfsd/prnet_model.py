from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn


class PRNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.normalizer = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        project_shortcut = self.stride != 1 or inputs.shape[-1] != self.out_channels
        shortcut = self.shortcut(inputs) if project_shortcut else inputs
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return cast(Tensor, self.activation(self.normalizer(outputs + shortcut)))


class PRNet(nn.Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.block0 = nn.Conv2d(3, size, kernel_size=3, padding=1, bias=False)
        self.block1 = PRNetResidualBlock(size, size * 2, stride=2)
        self.block2 = PRNetResidualBlock(size * 2, size * 2)
        self.block3 = PRNetResidualBlock(size * 2, size * 4, stride=2)
        self.block4 = PRNetResidualBlock(size * 4, size * 4)
        self.block5 = PRNetResidualBlock(size * 4, size * 8, stride=2)
        self.block6 = PRNetResidualBlock(size * 8, size * 8)
        self.block7 = PRNetResidualBlock(size * 8, size * 16, stride=2)
        self.block8 = PRNetResidualBlock(size * 16, size * 16)
        self.block9 = PRNetResidualBlock(size * 16, size * 32, stride=2)
        self.block10 = PRNetResidualBlock(size * 32, size * 32)

        decoder_channels = [
            (size * 32, size * 32, 3, 1),
            (size * 32, size * 16, 4, 2),
            (size * 16, size * 16, 3, 1),
            (size * 16, size * 16, 3, 1),
            (size * 16, size * 8, 4, 2),
            (size * 8, size * 8, 3, 1),
            (size * 8, size * 8, 3, 1),
            (size * 8, size * 4, 4, 2),
            (size * 4, size * 4, 3, 1),
            (size * 4, size * 4, 3, 1),
            (size * 4, size * 2, 4, 2),
            (size * 2, size * 2, 3, 1),
            (size * 2, size, 4, 2),
            (size, size, 3, 1),
            (size, 3, 3, 1),
            (3, 3, 3, 1),
            (3, 3, 3, 1),
        ]
        for index, (in_channels, out_channels, kernel_size, stride) in enumerate(decoder_channels):
            padding = 1
            layer = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            setattr(self, f"upsample{index}", layer)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for index in range(11):
            outputs = cast(nn.Module, getattr(self, f"block{index}"))(outputs)
        for index in range(17):
            outputs = cast(nn.Module, getattr(self, f"upsample{index}"))(outputs)
        return outputs.sigmoid()


def _rename_checkpoint_key(key: str) -> str:
    replacements = {
        ".shortcut_conv.": ".shortcut.",
        ".normalizer_fn.": ".normalizer.",
    }
    for source, destination in replacements.items():
        key = key.replace(source, destination)
    return key


def load_prnet_checkpoint(path: Path, device: torch.device, size: int = 16) -> PRNet:
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    raw_state = payload.get("prnet")
    if not isinstance(raw_state, dict):
        raise ValueError(f"PRNet checkpoint does not contain a prnet state dict: {path}")
    state = {_rename_checkpoint_key(str(key)): value for key, value in raw_state.items()}
    model = PRNet(size=size)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def normalize_prnet_inputs(images: Tensor) -> Tensor:
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images[:, [2, 1, 0]] - mean) / std


def resolve_prnet_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PRNet CUDA inference was requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)
