"""Synchronous four-round bayesian_eqdeeprx receiver."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .components import DemapperNN
from .config import BayesianEqDeepRxConfig

from .belief import project_re_grid
from .front_end import BayesianEqDeepRxFrontEnd, pilot_distance_map
from .networks import StateInit, StateUpdate


class BayesianEqDeepRx(nn.Module):
    """64-QAM receiver with four outer CNN/projection interactions.

    A single ``StateInit`` and ``DemapperNN`` are shared by layers. Four
    ``StateUpdate`` modules have independent parameters and are each shared
    by every layer. The probability inference sees all active layers at once.
    """

    def __init__(
        self, config: BayesianEqDeepRxConfig | None = None, *, message_chunk_size: int = 256
    ):
        super().__init__()
        self.config = config or BayesianEqDeepRxConfig()
        if self.config.modulation != "64QAM":
            raise ValueError("this receiver is fixed to 64QAM")
        if message_chunk_size < 1:
            raise ValueError("message_chunk_size must be positive")
        self.message_chunk_size = message_chunk_size
        self.front_end = BayesianEqDeepRxFrontEnd(self.config)
        # Native PyTorch initialization, as in the EqDeepRx reference training.
        # The final bit/symbol objective trains the denoiser through both
        # equalizers and the four channel-dependent probability projections.
        self.state_init = StateInit()
        self.updates = nn.ModuleList(StateUpdate() for _ in range(4))
        self.demapper = DemapperNN(in_channels=64, widths=(32, 32, 32, 8))
        initial_sigma = 2 / math.sqrt(42.0)
        initial_unconstrained = math.log(math.expm1(initial_sigma - 1e-4))
        self.log_sigma = nn.Parameter(torch.full((4,), initial_unconstrained))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + 1e-4

    def forward(
        self,
        received: torch.Tensor,
        pilot_symbols: torch.Tensor,
        pilot_mask: torch.Tensor,
        data_mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        front = self.front_end(received, pilot_symbols, pilot_mask)
        n, nk, _, f, s = front.features.shape
        if data_mask.shape == (n, 1, f, s):
            data_mask = data_mask.expand(n, nk, f, s)
        if data_mask.shape != (n, nk, f, s):
            raise ValueError("data mask must be [N,K,F,S] or [N,1,F,S]")
        position = pilot_distance_map(pilot_mask)
        flattened_position = position.reshape(n * nk, 2, f, s)
        initial = torch.cat((front.features, position), dim=2)
        state = self.state_init(initial.reshape(n * nk, 8, f, s)).reshape(n, nk, 64, f, s)
        message = state.new_full((n, nk, 16, f, s), 1.0 / 8.0)
        soft_symbols = []
        readouts = []
        bandwidth = self.config.receiver.incm_coherence_bandwidth

        for round_index, update in enumerate(self.updates):
            next_state = update(
                state.reshape(n * nk, 64, f, s),
                message.reshape(n * nk, 16, f, s),
                flattened_position,
            ).reshape(n, nk, 64, f, s)
            soft = torch.complex(next_state[:, :, 0], next_state[:, :, 1])
            if return_aux:
                soft_symbols.append(soft)
            # Both directed factor updates finish inside project_re_grid.
            # Its single returned projection is written back synchronously.
            projection, message = project_re_grid(
                received, front.channel, front.covariance, soft,
                data_mask, pilot_symbols, self.sigma[round_index],
                bandwidth=bandwidth, chunk_size=self.message_chunk_size,
            )
            projected_channels = torch.stack((projection.real, projection.imag), dim=2)
            state = torch.cat((projected_channels, next_state[:, :, 2:]), dim=2)
            if return_aux or round_index == 3:
                llrs = self.demapper(state.reshape(n * nk, 64, f, s))[:, :6]
                readouts.append(
                    llrs.reshape(n, nk, 6, f, s) * data_mask[:, :, None]
                )

        final = readouts[-1]
        if not return_aux:
            return final
        return final, {
            "readouts": readouts,
            "soft_symbols": soft_symbols,
            "messages": message,
            "state": state,
            "front_end": front,
            "position": position,
        }
