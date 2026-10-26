"""Globecom state CNN topology with the local scheme's input channels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GlobecomSeparableConv(nn.Module):
    """3×3 depthwise then 1×1 pointwise, with Keras' output bias."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class StateInit(nn.Module):
    """8 → 64 → 64 → 64 separable CNN, shared across layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = GlobecomSeparableConv(8, 64)
        self.norm1 = ChannelLayerNorm(64)
        self.conv2 = GlobecomSeparableConv(64, 64)
        self.norm2 = ChannelLayerNorm(64)
        self.conv3 = GlobecomSeparableConv(64, 64)
        self.norm3 = ChannelLayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return self.norm3(self.conv3(x))


class StateUpdate(nn.Module):
    """82 → 64 → 64 → 64 separable CNN with a state shortcut."""

    def __init__(self):
        super().__init__()
        self.conv1 = GlobecomSeparableConv(82, 64)
        self.norm1 = ChannelLayerNorm(64)
        self.conv2 = GlobecomSeparableConv(64, 64)
        self.norm2 = ChannelLayerNorm(64)
        self.conv3 = GlobecomSeparableConv(64, 64)
        self.norm3 = ChannelLayerNorm(64)

    def forward(
        self, state: torch.Tensor, message: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        z = torch.cat((state, message, position), dim=1)
        z = F.relu(self.norm1(self.conv1(z)))
        z = F.relu(self.norm2(self.conv2(z)))
        return self.norm3(state + self.conv3(z))
