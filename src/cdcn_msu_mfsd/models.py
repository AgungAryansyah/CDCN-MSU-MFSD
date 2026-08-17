from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CentralDifferenceConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def forward(self, inputs: Tensor) -> Tensor:
        normal = cast(Tensor, self.conv(inputs))
        if abs(self.theta) < 1e-8:
            return normal
        difference_kernel = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        difference = F.conv2d(
            inputs,
            difference_kernel,
            bias=self.conv.bias,
            stride=self.conv.stride,
            groups=self.conv.groups,
        )
        return normal - self.theta * difference


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        attention = cast(Tensor, self.conv(torch.cat((average, maximum), dim=1)))
        return torch.sigmoid(attention)


ConvFactory = Callable[..., nn.Module]


def _layer(
    conv: ConvFactory,
    in_channels: int,
    out_channels: int,
    theta: float,
) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CDCN(nn.Module):
    def __init__(self, theta: float = 0.7, conv: ConvFactory = CentralDifferenceConv2d) -> None:
        super().__init__()
        self.stem = _layer(conv, 3, 64, theta)
        self.block1 = nn.Sequential(
            _layer(conv, 64, 128, theta),
            _layer(conv, 128, 196, theta),
            _layer(conv, 196, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            _layer(conv, 128, 128, theta),
            _layer(conv, 128, 196, theta),
            _layer(conv, 196, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(
            _layer(conv, 128, 128, theta),
            _layer(conv, 128, 196, theta),
            _layer(conv, 196, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.head = nn.Sequential(
            _layer(conv, 384, 128, theta),
            _layer(conv, 128, 64, theta),
            conv(64, 1, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        first = self.block1(self.stem(inputs))
        second = self.block2(first)
        third = self.block3(second)
        features = torch.cat(tuple(self._resize(block) for block in (first, second, third)), dim=1)
        return cast(Tensor, self.head(features)).squeeze(1)

    @staticmethod
    def _resize(inputs: Tensor) -> Tensor:
        return F.interpolate(inputs, size=(32, 32), mode="bilinear", align_corners=False)


class CDCNPP(nn.Module):
    def __init__(self, theta: float = 0.7, conv: ConvFactory = CentralDifferenceConv2d) -> None:
        super().__init__()
        self.stem = _layer(conv, 3, 64, theta)
        self.block1 = nn.Sequential(
            _layer(conv, 64, 128, theta),
            _layer(conv, 128, 204, theta),
            _layer(conv, 204, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            _layer(conv, 128, 153, theta),
            _layer(conv, 153, 128, theta),
            _layer(conv, 128, 179, theta),
            _layer(conv, 179, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(
            _layer(conv, 128, 128, theta),
            _layer(conv, 128, 153, theta),
            _layer(conv, 153, 128, theta),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.attention = nn.ModuleList(
            (SpatialAttention(7), SpatialAttention(5), SpatialAttention(3))
        )
        self.head = nn.Sequential(
            _layer(conv, 384, 128, theta),
            conv(128, 1, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        first = self.block1(self.stem(inputs))
        second = self.block2(first)
        third = self.block3(second)
        blocks = (first, second, third)
        attended = (
            block * attention(block)
            for block, attention in zip(blocks, self.attention, strict=True)
        )
        features = torch.cat(tuple(self._resize(block) for block in attended), dim=1)
        return cast(Tensor, self.head(features)).squeeze(1)

    @staticmethod
    def _resize(inputs: Tensor) -> Tensor:
        return F.interpolate(inputs, size=(32, 32), mode="bilinear", align_corners=False)


MODELS: dict[str, type[nn.Module]] = {"cdcn": CDCN, "cdcnpp": CDCNPP}


def build_model(name: str, theta: float = 0.7) -> nn.Module:
    try:
        model_type = MODELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {name!r}; choose one of: {choices}") from error
    return model_type(theta=theta)
