"""Two directed factor-message updates and one 64-QAM soft projection.

The observation model is the unquantized, real-valued Gaussian adaptation of
TWC (17)--(26). The second outgoing message is a *normalized*, directed
distribution, as explicitly defined by the local final scheme.
"""

from __future__ import annotations

import math

import torch


def pam_levels(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor((-7, -5, -3, -1, 1, 3, 5, 7)) / math.sqrt(42.0)


def project_active_re(
    received: torch.Tensor,
    channel: torch.Tensor,
    variance: torch.Tensor,
    soft_symbols: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project active layers at M REs.

    Shapes: complex ``received[M,R]``, ``channel[M,R,K]``,
    real complex-disturbance ``variance[M,R]``, complex ``soft[M,K]``.
    Returns complex ``[M,K]`` and beliefs ``[M,2K,8]``. The real and
    imaginary symbol variables occupy the first and second K positions.
    """

    if received.ndim != 2 or channel.shape[:2] != received.shape:
        raise ValueError("received/channel must be [M,R] and [M,R,K]")
    m, nr = received.shape
    nk = channel.shape[-1]
    if nk < 1 or soft_symbols.shape != (m, nk) or variance.shape != (m, nr):
        raise ValueError("soft symbols or disturbance variance has wrong shape")
    if bool((variance <= 0).any()) or not bool(torch.isfinite(variance).all()):
        raise ValueError("the per-antenna complex disturbance variance must be positive")

    # [Re(y), Im(y)] = [[Re(H), -Im(H)], [Im(H), Re(H)]]
    y = torch.cat((received.real, received.imag), dim=-1).float()
    g = torch.cat(
        (
            torch.cat((channel.real, -channel.imag), dim=-1),
            torch.cat((channel.imag, channel.real), dim=-1),
        ),
        dim=1,
    ).float()
    x = torch.cat((soft_symbols.real, soft_symbols.imag), dim=-1).float()
    noise = torch.cat((variance, variance), dim=-1).float() / 2.0
    levels = pam_levels(x)
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    if bool((sigma <= 0).any()):
        raise ValueError("sigma must be positive")
    prior_logits = -(x[..., None] - levels).square() / (2.0 * sigma.square())
    prior = prior_logits.softmax(dim=-1)
    # p is indexed [RE, observation j, symbol component i, level a].
    outgoing = prior[:, None].expand(-1, 2 * nr, -1, -1)
    g2 = g.square()

    for update in range(2):
        mean = (outgoing * levels).sum(dim=-1)
        symbol_variance = ((outgoing * levels.square()).sum(dim=-1) - mean.square()).clamp_min(0.0)
        mui = (g * mean).sum(dim=-1, keepdim=True) - g * mean
        mui_variance = (g2 * symbol_variance).sum(dim=-1, keepdim=True) - g2 * symbol_variance
        observation_variance = torch.maximum(noise[..., None] + mui_variance, noise[..., None])
        residual = y[:, :, None, None] - g[..., None] * levels - mui[..., None]
        squared = residual.square()
        delta = -(squared - squared[..., :1]) / (2.0 * observation_variance[..., None])
        aggregated = delta.sum(dim=1)
        if update == 0:
            # Exclude the target observation; no extra initial prior is
            # multiplied here, matching the locally specified adaptation.
            outgoing = (aggregated[:, None] - delta).softmax(dim=-1)

    belief = aggregated.softmax(dim=-1)
    projection = (belief * levels).sum(dim=-1)
    projected_complex = torch.complex(projection[:, :nk], projection[:, nk:])
    return projected_complex, belief


def project_re_grid(
    received: torch.Tensor,
    channel: torch.Tensor,
    covariance: torch.Tensor,
    soft_symbols: torch.Tensor,
    data_mask: torch.Tensor,
    pilot_symbols: torch.Tensor,
    sigma: torch.Tensor,
    *,
    bandwidth: int = 24,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project jointly per RE, subtracting only known nondetected pilots.

    Inputs: Y[N,R,F,S], H[N,R,K,F,S], R[N,B,R,R],
    soft/pilots/mask[N,K,F,S]. Nondata state is left to the caller; its
    returned projection equals the input and its 16 belief values are uniform.
    """

    n, nr, f, s = received.shape
    nk = channel.shape[2]
    if bandwidth < 1 or chunk_size < 1:
        raise ValueError("bandwidth and chunk_size must be positive")
    if channel.shape != (n, nr, nk, f, s) or soft_symbols.shape != (n, nk, f, s):
        raise ValueError("channel or soft-symbol shape does not match received")
    if pilot_symbols.shape != soft_symbols.shape or data_mask.shape != soft_symbols.shape:
        raise ValueError("pilot symbols and data mask must have one entry per layer and RE")
    if covariance.shape != (n, math.ceil(f / bandwidth), nr, nr):
        raise ValueError("covariance must contain one full RX matrix per frequency band")
    if bool(((data_mask > 0) & (pilot_symbols.abs() > 0)).any()):
        raise ValueError("a data layer cannot simultaneously transmit a known pilot")

    y_flat = received.permute(0, 2, 3, 1).reshape(-1, nr)
    h_flat = channel.permute(0, 3, 4, 1, 2).reshape(-1, nr, nk)
    soft_flat = soft_symbols.permute(0, 2, 3, 1).reshape(-1, nk)
    pilot_flat = pilot_symbols.permute(0, 2, 3, 1).reshape(-1, nk)
    active = (data_mask > 0).permute(0, 2, 3, 1).reshape(-1, nk)
    diag = covariance.diagonal(dim1=2, dim2=3).real  # [N,B,R]
    band = torch.arange(f, device=received.device) // bandwidth
    disturbance = diag[:, band, :][:, :, None, :].expand(n, f, s, nr).reshape(-1, nr)
    projected = soft_flat.clone()
    beliefs = soft_flat.real.new_full((n * f * s, nk, 2, 8), 1.0 / 8.0)
    powers = 2 ** torch.arange(nk, device=received.device)
    patterns = (active.long() * powers).sum(dim=-1)

    for key in range(1, 2**nk):
        positions = torch.nonzero(patterns == key, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        layers = [layer for layer in range(nk) if key & (1 << layer)]
        for start in range(0, positions.numel(), chunk_size):
            where = positions[start : start + chunk_size]
            h = h_flat[where]
            # The reference resource grid has zeros on unused layers. Unknown
            # data are never included in this cancellation term.
            known = torch.where(active[where], 0, pilot_flat[where])
            residual = y_flat[where] - torch.einsum("mrk,mk->mr", h, known)
            result, q = project_active_re(
                residual, h[:, :, layers], disturbance[where],
                soft_flat[where][:, layers], sigma,
            )
            projected[where[:, None], layers] = result
            beliefs[where[:, None], layers] = torch.stack(
                (q[:, : len(layers)], q[:, len(layers) :]), dim=2
            )

    projected = projected.reshape(n, f, s, nk).permute(0, 3, 1, 2).contiguous()
    beliefs = beliefs.reshape(n, f, s, nk, 2, 8).permute(0, 3, 4, 5, 1, 2)
    return projected, beliefs.reshape(n, nk, 16, f, s).contiguous()
