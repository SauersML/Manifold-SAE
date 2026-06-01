"""Loss components for Manifold-SAE training (gamfit-native schema).

The training objective under the gamfit-native :class:`gamfit.torch.ManifoldSAE`
is reconstruction plus the module's own Rust-backed regularizers:

    MSE(x_hat, x)                          — SAE reconstruction in ambient R^D
  + mean(z)                                — activation sparsity surrogate
  + sae.decoder_ortho_penalty()            — per-atom decoder block orthogonality
  + sae.decoder_monotonicity_penalty()     — per-atom curve monotonicity prior
  + light coverage prior                   — firing positions span the manifold

What changed in the cutover:
  - There is no explicit ``-reml_score`` loss term. REML now drives the
    closed-form ``sae.fit()`` solve, not a backprop loss (the old term had a
    degenerate maximum at all-zero amplitudes).
  - Smoothness λ selection is internal to gamfit (``sae.fit`` / ``lambdas``).
  - ``ortho_loss`` / ``monotonicity_loss`` are no longer fields of the output;
    they are *methods on the module* and are queried here when the module is
    passed in.

Signature note: ``total_loss(output, target, sae)`` REQUIRES the SAE *module*
as its third argument so it can call ``decoder_ortho_penalty()`` /
``decoder_monotonicity_penalty()``. Passing anything that is not a
:class:`ManifoldSAE` raises ``TypeError`` — there is no degraded
config-only path.
"""

from __future__ import annotations

import torch

from .sae import ManifoldSAE, ManifoldSAEOutput


def _position_coverage_loss(
    positions: torch.Tensor, mask: torch.Tensor, n_bins: int = 10
) -> torch.Tensor:
    """Identification prior: firing positions should spread over the manifold.

    ``positions`` is ``(N, F, d)`` under the new schema; we use the angular /
    first intrinsic coordinate. ``mask`` is ``(N, F)``. KL from uniform on a
    soft-binned firing-position histogram, averaged over atoms.
    """
    if positions.dim() == 3:
        coord = positions[..., 0]
    else:
        coord = positions
    lo = coord.detach().min()
    hi = coord.detach().max()
    span = (hi - lo).clamp(min=1e-6)
    coord = (coord - lo) / span
    centers = torch.linspace(0.0, 1.0, n_bins, device=coord.device, dtype=coord.dtype)
    width = 1.0 / max(n_bins - 1, 1)
    diff = coord.unsqueeze(-1) - centers.view(1, 1, -1)
    bin_weights = torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)
    mw = mask.unsqueeze(-1) * bin_weights
    p = mw.sum(dim=0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    uniform = 1.0 / n_bins
    kl = (
        p
        * (
            torch.log(p.clamp(min=1e-12))
            - torch.log(torch.tensor(uniform, device=p.device, dtype=p.dtype))
        )
    ).sum(dim=-1)
    return kl.mean()


def total_loss(
    output: ManifoldSAEOutput,
    target: torch.Tensor,
    sae: ManifoldSAE,
    *,
    sparsity_weight: float = 1e-3,
    coverage_weight: float = 1e-2,
    ortho_weight: float = 1e-2,
    monotonicity_weight: float = 1e-2,
) -> dict[str, torch.Tensor]:
    """Reconstruction + Rust-backed regularizers on the gamfit-native output.

    ``sae`` MUST be the :class:`ManifoldSAE` module so the decoder penalties
    can be evaluated off it. Passing anything else raises ``TypeError``.
    """
    if not isinstance(sae, ManifoldSAE):
        raise TypeError(
            "total_loss requires the ManifoldSAE module as its third argument "
            f"(to read decoder penalties), got {type(sae).__name__}"
        )
    mse = torch.mean((output.x_hat - target) ** 2)
    # Activation-magnitude sparsity surrogate (z = assignments * amplitudes).
    sparsity = output.z.abs().mean()
    coverage = _position_coverage_loss(output.positions, output.amplitudes)

    ortho = sae.decoder_ortho_penalty()
    monotonicity = sae.decoder_monotonicity_penalty()

    total = (
        mse
        + sparsity_weight * sparsity
        + ortho_weight * ortho
        + coverage_weight * coverage
        + monotonicity_weight * monotonicity
    )
    return {
        "mse": mse,
        "sparsity": sparsity,
        "ortho": ortho,
        "reml": output.reml_score.reshape(-1).mean(),
        "coverage": coverage,
        "monotonicity": monotonicity,
        "total": total,
    }
