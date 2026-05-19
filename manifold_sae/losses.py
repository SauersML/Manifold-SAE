"""Loss components for Manifold-SAE training."""

from __future__ import annotations

import torch

from .sae import ManifoldSAEConfig, ManifoldSAEOutput


def total_loss(
    output: ManifoldSAEOutput,
    target: torch.Tensor,
    config: ManifoldSAEConfig,
) -> dict[str, torch.Tensor]:
    mse = torch.mean((output.reconstruction - target) ** 2)
    # Sparsity on PRE-binarization sigmoid probabilities. Gives gradient to
    # all features (not just TopK-selected ones), so non-firing features
    # still get pushed toward zero — helps the encoder converge on which
    # features should fire.
    sparsity = output.mask_soft.mean()
    total = (
        mse
        + config.sparsity_weight * sparsity
        + config.cumulant_weight * output.cumulant_loss
        + config.ortho_weight * output.ortho_loss
    )
    return {
        "mse": mse,
        "sparsity": sparsity,
        "cumulant": output.cumulant_loss,
        "ortho": output.ortho_loss,
        "total": total,
    }
