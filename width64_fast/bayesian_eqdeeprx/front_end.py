"""The local pilot front end for bayesian_eqdeeprx."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import DenoiseNN
from .config import BayesianEqDeepRxConfig
from .receiver import (
    estimate_incm,
    estimate_raw_channel,
    interpolate_channel,
    lmmse_equalize,
    rzf_equalize,
)


@dataclass(frozen=True)
class FrontEndOutput:
    features: torch.Tensor
    channel: torch.Tensor
    covariance: torch.Tensor
    lmmse: torch.Tensor
    rzf: torch.Tensor


class BayesianEqDeepRxFrontEnd(nn.Module):
    """Run the paper's LS → denoiser → interpolation → INCM → EQ path."""

    def __init__(self, config: BayesianEqDeepRxConfig | None = None):
        super().__init__()
        self.config = config or BayesianEqDeepRxConfig()
        model = self.config.model
        self.denoise = DenoiseNN(
            widths=model.denoise_widths,
            subsamples=model.denoise_subsamples,
            time_mixer_channels=model.time_mixer_channels,
            residual_projection_bias=model.residual_projection_bias,
            amp_dtype=self.config.training.amp_dtype,
        )

    def forward(
        self, received: torch.Tensor, pilot_symbols: torch.Tensor, pilot_mask: torch.Tensor
    ) -> FrontEndOutput:
        raw = estimate_raw_channel(received, pilot_symbols, pilot_mask)
        denoised = self._denoise_channel(raw, pilot_mask)
        channel = interpolate_channel(denoised, pilot_mask)
        bandwidth = self.config.receiver.incm_coherence_bandwidth
        covariance = estimate_incm(
            received, channel, pilot_symbols, pilot_mask,
            coherence_bandwidth=bandwidth,
        )
        lmmse = lmmse_equalize(received, channel, covariance, coherence_bandwidth=bandwidth)
        rzf = rzf_equalize(received, channel, alpha=self.config.receiver.rzf_alpha)
        n, nk, f, s = lmmse.shape
        coordinates = self._coordinates(n, f, s, received.device, received.real.dtype)
        features = torch.cat(
            (
                lmmse.real.unsqueeze(2), lmmse.imag.unsqueeze(2),
                rzf.real.unsqueeze(2), rzf.imag.unsqueeze(2),
                coordinates.unsqueeze(1).expand(-1, nk, -1, -1, -1),
            ), dim=2,
        )
        # The local reference equalizer stores bands last. The final scheme's
        # public contract stores them immediately after the batch dimension.
        return FrontEndOutput(features, channel, covariance.permute(0, 3, 1, 2).contiguous(), lmmse, rzf)

    def _denoise_channel(self, raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        n, nr, nt, f, s = raw.shape
        mask = pilot_mask > 0

        frequency_locations = torch.nonzero(mask.any(dim=-1), as_tuple=False)
        time_locations = torch.nonzero(mask.any(dim=-2), as_tuple=False)
        pair_count = n * nt

        if (
            frequency_locations.shape[0] == 0
            or frequency_locations.shape[0] % pair_count
            or time_locations.shape[0] % pair_count
        ):
            raise ValueError("each batch/layer pair must have a uniform pilot grid")
        n_pilot_frequencies = frequency_locations.shape[0] // pair_count
        n_pilot_symbols = time_locations.shape[0] // pair_count

        frequencies = frequency_locations[:, 2].reshape(n, nt, n_pilot_frequencies)
        symbols = time_locations[:, 2].reshape(n, nt, n_pilot_symbols)

        by_pair = raw.permute(0, 2, 1, 3, 4)
        frequency_indices = frequencies[:, :, None, :, None].expand(
            n, nt, nr, n_pilot_frequencies, s
        )
        compact = torch.gather(by_pair, 3, frequency_indices)
        time_indices = symbols[:, :, None, None, :].expand(
            n, nt, nr, n_pilot_frequencies, n_pilot_symbols
        )
        compact = torch.gather(compact, 4, time_indices)
        denoiser_input = torch.stack((compact.real, compact.imag), dim=3).reshape(
            n * nt * nr, 2, n_pilot_frequencies, n_pilot_symbols
        )
        denoised = self.denoise(denoiser_input).reshape(
            n, nt, nr, 2, n_pilot_frequencies, n_pilot_symbols
        )
        estimate = torch.complex(denoised[:, :, :, 0], denoised[:, :, :, 1])
        flat_indices = (
            frequencies[..., :, None] * s + symbols[..., None, :]
        ).reshape(n, nt, 1, -1).expand(n, nt, nr, -1)
        full = raw.new_zeros(n, nt, nr, f * s)
        full.scatter_(3, flat_indices, estimate.reshape(n, nt, nr, -1))
        return full.reshape(n, nt, nr, f, s).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _coordinates(batch: int, f: int, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freq = torch.linspace(-1.0, 1.0, f, device=device, dtype=dtype).view(1, 1, f, 1).expand(batch, 1, f, s)
        time = torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype).view(1, 1, 1, s).expand(batch, 1, f, s)
        return torch.cat((freq, time), dim=1)


def pilot_distance_map(pilot_mask: torch.Tensor) -> torch.Tensor:
    """Normalized [time distance, frequency distance] for each layer."""

    if pilot_mask.ndim != 4:
        raise ValueError("pilot mask must be [N,K,F,S]")
    n, nk, f, s = pilot_mask.shape
    frequency = torch.arange(f, device=pilot_mask.device)
    time = torch.arange(s, device=pilot_mask.device)
    slots = []
    for batch in range(n):
        layers = []
        for layer in range(nk):
            occupied = pilot_mask[batch, layer] > 0
            time_positions = torch.nonzero(occupied.any(dim=0), as_tuple=False).flatten()
            frequency_positions = torch.nonzero(occupied.any(dim=1), as_tuple=False).flatten()
            if time_positions.numel() == 0 or frequency_positions.numel() == 0:
                raise ValueError("each layer needs pilot locations for positional encoding")
            time_distance = (time[:, None] - time_positions).abs().min(dim=1).values
            frequency_distance = (frequency[:, None] - frequency_positions).abs().min(dim=1).values
            distance = torch.stack(
                (time_distance[None].expand(f, s), frequency_distance[:, None].expand(f, s)),
                dim=0,
            ).to(torch.float32)
            mean = distance.mean(dim=(1, 2), keepdim=True)
            std = distance.std(dim=(1, 2), unbiased=False, keepdim=True)
            distance = torch.where(std > 0, (distance - mean) / std.clamp_min(1e-12), 0)
            layers.append(distance)
        slots.append(torch.stack(layers))
    return torch.stack(slots)
