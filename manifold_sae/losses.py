"""Loss components for Manifold-SAE training.

Under Path B (gamfit-managed B and λ, REML each batch), the smoothness
selection is internal to gamfit. The training loss is:

    MSE(recon, x)                — full SAE reconstruction in ambient
  + sparsity_weight · |amps|     — standard SAE sparsity
  + ortho_weight · ortho_loss    — identification: W_k ≠ W_j across features
  − reml_weight · reml_score     — REML log-likelihood (smoothness via gamfit)
  + light coverage prior         — identification: positions span [0, 1]
  + light monotonicity prior     — parameterization tiebreaker

What's NOT here anymore:
  - quadratic smoothness penalty (lived inside REML now)
  - curve_norm gauge penalty   (REML's likelihood pins the amplitude scale
                                through the per-feature ``by`` weighting)
  - cumulant identifiability    (REML + sparsity is enough on the toy)
"""

from __future__ import annotations

import torch

from .sae import ManifoldSAEConfig, ManifoldSAEOutput


def _position_coverage_loss(positions: torch.Tensor, mask: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Identification prior: firing positions should spread over [0, 1].

    KL from uniform on soft-binned firing-position histogram.
    """
    B, F = positions.shape
    centers = torch.linspace(0.0, 1.0, n_bins, device=positions.device, dtype=positions.dtype)
    width = 1.0 / max(n_bins - 1, 1)
    diff = positions.unsqueeze(-1) - centers.view(1, 1, -1)
    bin_weights = torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)
    mw = mask.unsqueeze(-1) * bin_weights
    p = mw.sum(dim=0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
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
    # NOTE: gamfit already runs REML *internally* to select λ_k per feature
    # given current positions, y_proj, and amplitudes. The fit's B_k and
    # λ_k are then used to build the reconstruction. So smoothness selection
    # happens inside gamfit; the MSE on the SAE reconstruction drives the
    # encoder + W via backprop through the gamfit autograd path.
    #
    # We do NOT add `-reml_score` to the loss as an explicit term: that
    # term has a degenerate maximum at "all amplitudes zero" (gamfit's
    # reml_score for zero-weight features is unbounded above because there's
    # no residual to fit and no penalty paid), which collapses the model.
    # Keep reml_score in the output struct for diagnostics only.
    total = (
        mse
        + config.sparsity_weight * sparsity
        + config.ortho_weight * output.ortho_loss
        + 1e-2 * coverage
        + 1e-2 * output.monotonicity_loss
    )
    return {
        "mse": mse,
        "sparsity": sparsity,
        "ortho": output.ortho_loss,
        "reml": output.reml_score.mean(),
        "coverage": coverage,
        "monotonicity": output.monotonicity_loss,
        "total": total,
    }
