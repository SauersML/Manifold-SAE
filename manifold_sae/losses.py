"""Loss components for Manifold-SAE training."""

from __future__ import annotations

import torch

from .sae import ManifoldSAEConfig, ManifoldSAEOutput


def _position_coverage_loss(positions: torch.Tensor, mask: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """For each feature, encourage its FIRING positions to spread uniformly
    over [0, 1]. Without this, the encoder can fire feature k only for
    tokens whose position falls in a narrow range — yielding partial-arc
    recovery even when W_k is correct.

    Soft-binned histogram of positions, weighted by mask, normalized per
    feature. Return mean KL(p || uniform) over features.
    """
    B, F = positions.shape
    centers = torch.linspace(0.0, 1.0, n_bins, device=positions.device, dtype=positions.dtype)
    width = 1.0 / max(n_bins - 1, 1)
    # (B, F, n_bins) soft-membership weights
    diff = positions.unsqueeze(-1) - centers.view(1, 1, -1)
    bin_weights = torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)
    # Weight by firing mask
    mw = mask.unsqueeze(-1) * bin_weights  # (B, F, n_bins)
    # Sum over batch, normalize per feature
    p = mw.sum(dim=0)  # (F, n_bins)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # KL from uniform (= -entropy + log n_bins)
    uniform = 1.0 / n_bins
    kl = (p * (torch.log(p.clamp(min=1e-12)) - torch.log(torch.tensor(uniform, device=p.device, dtype=p.dtype)))).sum(dim=-1)
    return kl.mean()


def total_loss(
    output: ManifoldSAEOutput,
    target: torch.Tensor,
    config: ManifoldSAEConfig,
) -> dict[str, torch.Tensor]:
    mse = torch.mean((output.reconstruction - target) ** 2)
    sparsity = output.mask_soft.mean()
    coverage = _position_coverage_loss(output.positions, output.mask_soft)
    total = (
        mse
        + config.sparsity_weight * sparsity
        + config.cumulant_weight * output.cumulant_loss
        + config.ortho_weight * output.ortho_loss
        + 1e-2 * coverage
        + 1e-1 * output.monotonicity_loss
    )
    return {
        "mse": mse,
        "sparsity": sparsity,
        "cumulant": output.cumulant_loss,
        "ortho": output.ortho_loss,
        "coverage": coverage,
        "monotonicity": output.monotonicity_loss,
        "total": total,
    }
