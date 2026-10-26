"""EqDeepRx-style bit/symbol objective with the final scheme's data mask."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def receiver_loss_components(
    readouts: list[torch.Tensor],
    soft_symbols: list[torch.Tensor],
    target_bits: torch.Tensor,
    target_symbols: torch.Tensor,
    data_mask: torch.Tensor,
    snr_linear: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Compute mean_n log2(1+SNR_n) × [mean_t CE_n^t + 1e-5 SE_n].

    Readouts are LLRs positive for bit 0; BCE therefore consumes their
    negation as bit-1 logits. Each symbol term is a *sum* over data REs.
    """

    if len(readouts) != 4 or len(soft_symbols) != 4:
        raise ValueError("exactly four readouts and four pre-projection symbols are required")
    n, nk, bits, f, s = readouts[0].shape
    if bits != 6 or target_bits.shape != (n, nk, 6, f, s):
        raise ValueError("bit tensors must be [N,K,6,F,S]")
    if target_symbols.shape != (n, nk, f, s):
        raise ValueError("target symbols must be [N,K,F,S]")
    if data_mask.shape == (n, 1, f, s):
        data_mask = data_mask.expand(n, nk, f, s)
    if data_mask.shape != (n, nk, f, s):
        raise ValueError("data mask must be [N,K,F,S] or [N,1,F,S]")
    mask = data_mask.to(readouts[0].dtype)
    count = mask.sum(dim=(1, 2, 3))
    if bool((count == 0).any()):
        raise ValueError("each training slot needs at least one data symbol")
    ce = readouts[0].new_zeros(n)
    se = readouts[0].new_zeros(n)
    round_ce = []
    for llr, soft in zip(readouts, soft_symbols):
        if llr.shape != (n, nk, 6, f, s) or soft.shape != (n, nk, f, s):
            raise ValueError("round output shape mismatch")
        bit_error = F.binary_cross_entropy_with_logits(
            -llr, target_bits.to(llr.dtype), reduction="none"
        )
        current_ce = (bit_error * mask[:, :, None]).sum(dim=(1, 2, 3, 4)) / (6 * count)
        ce = ce + current_ce
        round_ce.append(current_ce)
        symbol_error = (soft - target_symbols).abs().square()
        se = se + (symbol_error * mask).sum(dim=(1, 2, 3))
    snr = torch.as_tensor(snr_linear, dtype=ce.dtype, device=ce.device).flatten()
    if snr.numel() == 1:
        snr = snr.expand(n)
    if snr.numel() != n or bool((snr < 0).any()):
        raise ValueError("linear SNR must be nonnegative and scalar or per slot")
    weight = torch.log2(1 + snr)
    return {
        "loss": (weight * (ce / 4 + 1e-5 * se)).mean(),
        "bit_loss": (weight * ce / 4).mean(),
        "symbol_loss": (weight * 1e-5 * se).mean(),
        "round_bit_losses": (torch.stack(round_ce) * weight[None]).mean(dim=1),
    }


def receiver_loss(
    readouts: list[torch.Tensor],
    soft_symbols: list[torch.Tensor],
    target_bits: torch.Tensor,
    target_symbols: torch.Tensor,
    data_mask: torch.Tensor,
    snr_linear: torch.Tensor | float,
) -> torch.Tensor:
    """Keep the original four-readout CE and pre-projection symbol objective."""
    return receiver_loss_components(readouts, soft_symbols, target_bits,
                                    target_symbols, data_mask, snr_linear)["loss"]
