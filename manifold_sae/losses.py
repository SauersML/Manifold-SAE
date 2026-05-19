"""Loss components for Manifold-SAE training."""

from __future__ import annotations

import torch

from .sae import ManifoldSAEConfig, ManifoldSAEOutput


def _position_spread_entropy(positions: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    # Soft-binned histogram of mean position per feature across the batch.
    # Returns negative entropy: minimizing this maximizes spread.
    B, F = positions.shape
    centers = torch.linspace(0.0, 1.0, n_bins, device=positions.device, dtype=positions.dtype)
    width = 1.0 / max(n_bins - 1, 1)
    # (B, F, n_bins)
    diff = positions.unsqueeze(-1) - centers.view(1, 1, -1)
    weights = torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    # Mean across batch -> (F, n_bins) distribution per feature.
    p = weights.mean(dim=0)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum(dim=-1).mean()
    return -entropy


def total_loss(
    output: ManifoldSAEOutput,
    target: torch.Tensor,
    config: ManifoldSAEConfig,
) -> dict[str, torch.Tensor]:
    mse = torch.mean((output.reconstruction - target) ** 2)
    sparsity = output.amplitudes.abs().mean()
    reml = -output.reml_score
    spread = _position_spread_entropy(output.positions)

    total = (
        mse
        + config.sparsity_weight * sparsity
        + config.reml_weight * reml
        + config.position_spread_weight * spread
    )
    return {
        "mse": mse,
        "sparsity": sparsity,
        "reml": reml,
        "position_spread": spread,
        "total": total,
    }
