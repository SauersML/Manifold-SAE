"""SAE diagnostics — thin gamfit-0.1.241 layer.

Trust / atom-typing diagnostics are no longer hand-rolled here: they are owned
by gamfit. For a **facade** fit (``gamfit.sae_manifold_fit`` /
:class:`gamfit.ManifoldSAE`) call :func:`trust_diagnostics` /
:func:`atom_trust_scores` / :func:`atom_trust` / :func:`atom_diagnostics` /
:func:`summary`, all of which delegate straight into gamfit's Rust-validated
trust block.

GAP (kept deliberately): the closed-form trust block only exists on the
*facade* result. The rest of this package trains the **torch** module
(:class:`gamfit.torch.ManifoldSAE`, output :class:`ManifoldSAEOutput`), for
which gamfit exposes no per-step trust surface. The three
``ManifoldSAEOutput`` probes below (position variance, dead-feature mask,
position/amplitude grad ratio) are the torch-training analogs and stay because
gamfit provides no drop-in replacement for that output type.
"""

from __future__ import annotations

from typing import Any

import torch

from gamfit import atom_trust_scores, sae_trust_diagnostics

from .sae import ManifoldSAEOutput

__all__ = [
    # gamfit trust surface (facade fits)
    "sae_trust_diagnostics",
    "trust_diagnostics",
    "atom_trust_scores",
    "atom_trust",
    "atom_diagnostics",
    "summary",
    # torch-training probes (ManifoldSAEOutput)
    "position_variance",
    "dead_feature_mask",
    "position_amplitude_grad_ratio",
]


# --------------------------------------------------------------------------- #
# gamfit-native trust diagnostics (facade `gamfit.ManifoldSAE` fits)
# --------------------------------------------------------------------------- #
def trust_diagnostics(fit: Any) -> dict[str, Any]:
    """Validated ``{atom_trust, atoms}`` trust block for a fitted facade SAE.

    Delegates to :func:`gamfit.sae_trust_diagnostics`, feeding it the fit's own
    ``diagnostics`` block. Accepts either a fitted :class:`gamfit.ManifoldSAE`
    or a raw payload/serialized-dict already carrying a ``diagnostics`` key.
    """
    if hasattr(fit, "diagnostics") and not isinstance(fit, dict):
        payload = {"diagnostics": fit.diagnostics}
    else:
        payload = fit  # type: ignore[assignment]
    return sae_trust_diagnostics(payload)


def atom_trust(fit: Any, atom: int) -> float:
    """Per-atom trust score, straight from gamfit's fitted SAE."""
    return fit.atom_trust(atom)


def atom_diagnostics(fit: Any, atom: int) -> dict[str, Any]:
    """Full per-atom diagnostic record (coverage, sigma_min_tangent, ...)."""
    return fit.atom_diagnostics(atom)


def summary(fit: Any) -> dict[str, Any]:
    """gamfit's own fit summary (trust histogram, typed/untyped counts, ...)."""
    return fit.summary()


# --------------------------------------------------------------------------- #
# torch-training probes on ManifoldSAEOutput (no gamfit equivalent — see GAP)
# --------------------------------------------------------------------------- #
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
    # positions are (N, F, d) for intrinsic_rank d > 1, so pos_mag is (F, d);
    # collapse the trailing intrinsic dim to a per-feature scalar so it
    # broadcasts against the (F,) amplitude magnitudes.
    if pos_mag.dim() > amp_mag.dim():
        pos_mag = pos_mag.flatten(amp_mag.dim()).mean(dim=amp_mag.dim())
    ratio = pos_mag / (amp_mag + eps)
    return {
        "position_grad_mag": pos_mag,
        "amplitude_grad_mag": amp_mag,
        "ratio": ratio,
    }
