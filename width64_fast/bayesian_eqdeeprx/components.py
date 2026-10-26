"""Local copies of the EqDeepRx denoiser and pointwise demapper.

The detector is deliberately replaced by the bayesian_eqdeeprx CNN.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import PointwiseResidualBlock, SubsampledResidualBlock, TimeMixer


def _resolve_amp_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError("amp dtype must be bfloat16 or float32")



class DenoiseNN(nn.Module):
    def __init__(
        self,
        *,
        widths: Tuple[int, ...] = (64, 64, 64, 2),
        subsamples: Tuple[int, ...] = (1, 4, 2, 1),
        time_mixer_channels: int = 2,
        residual_projection_bias: bool = False,
        amp_dtype: str | torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.amp_dtype = _resolve_amp_dtype(amp_dtype)
        if len(widths) != len(subsamples):
            raise ValueError("widths and subsamples must have equal length")
        blocks = []
        in_channels = 2
        for width, downsample in zip(widths, subsamples):
            blocks.append(
                SubsampledResidualBlock(
                    in_channels,
                    width,
                    downsample=downsample,
                    frequency_only=True,
                    projection_bias=residual_projection_bias,
                )
            )
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.mixers = nn.ModuleList(
            TimeMixer(mix_channels=min(time_mixer_channels, width)) for width in widths
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex channel estimates are represented by real/imaginary pairs.
        # Use CUDA bfloat16 for the convolutions (float32 exponent range), but
        # return float32 before constructing the complex channel estimate.
        denoise_amp = x.device.type == "cuda"
        with torch.autocast(
            device_type=x.device.type,
            dtype=self.amp_dtype,
            enabled=denoise_amp and self.amp_dtype != torch.float32,
        ):
            out = x.float()
            for block, mixer in zip(self.blocks, self.mixers):
                out = mixer(block(out))
        return out.float()



class DemapperNN(nn.Module):
    def __init__(self, *, in_channels: int = 64, widths: Tuple[int, ...] = (32, 32, 32, 8)):
        super().__init__()
        blocks = []
        current = in_channels
        for width in widths:
            blocks.append(PointwiseResidualBlock(current, width))
            current = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Demapper activations can exceed float16 range at high-SNR samples.
        # Keep this learned boundary in float32 while retaining AMP elsewhere.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = x.float()
            for block in self.blocks:
                out = block(out)
        return out.float()

