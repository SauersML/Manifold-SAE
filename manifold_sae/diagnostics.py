"""Diagnostics on ManifoldSAEOutput — pure functions, no side effects."""

from __future__ import annotations

import torch

from .sae import ManifoldSAEOutput


def position_variance(output: ManifoldSAEOutput) -> torch.Tensor:
    """Per-feature variance of positions across the batch. Low values flag collapse."""
    return output.positions.var(dim=0, unbiased=False)


def dead_feature_mask(output: ManifoldSAEOutput, amp_threshold: float = 1e-3) -> torch.Tensor:
    """Boolean mask, True where a feature's mean amplitude across the batch is below threshold."""
    return output.amplitudes.mean(dim=0) < amp_threshold


def position_amplitude_grad_ratio(
    loss: torch.Tensor,
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """|dL/dp| / |dL/da| per feature; signals whether the encoder is using positions."""
    grads = torch.autograd.grad(
        loss, [positions, amplitudes], retain_graph=True, allow_unused=True
    )
    g_p, g_a = grads
    g_p = torch.zeros_like(positions) if g_p is None else g_p
    g_a = torch.zeros_like(amplitudes) if g_a is None else g_a
    pos_mag = g_p.abs().mean(dim=0)
    amp_mag = g_a.abs().mean(dim=0)
    ratio = pos_mag / (amp_mag + eps)
    return {
        "position_grad_mag": pos_mag,
        "amplitude_grad_mag": amp_mag,
        "ratio": ratio,
    }
