"""Thin seam over gamfit_glue.manifold_fit; swap here to replace the inner solve."""

from __future__ import annotations

import torch

from .gamfit_glue import BasisSpec, manifold_fit


def decode(
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    basis_spec: BasisSpec,
) -> dict:
    """Call manifold_fit and return its dict of reconstruction, reml_score, lambdas, edf."""
    return manifold_fit(positions, amplitudes, targets, basis_spec)


@torch.no_grad()
def extract_feature_curves(
    sae,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Probe each feature's learned curve on a position grid.

    The decoder coefficients are solved fresh per-batch by gamfit (no learned
    decoder weights), so "feature k's curve" only exists relative to a batch
    context. We:

    1. Run the encoder on the real activation batch to get the actual
       (positions, amplitudes) the model uses at inference.
    2. Run the gamfit inner solve once on that batch; it returns per-feature
       coefficients ``B_k`` of shape ``(K, D)``.
    3. For each feature ``k``, evaluate the Duchon basis at ``t_grid`` to get
       a ``(T, K)`` design matrix, then ``curve_k = phi(t_grid) @ B_k``.

    This is the *trained* coefficient evaluated at user-chosen positions —
    not a re-solve that biases each feature with the full target signal.

    Parameters
    ----------
    sae:
        Trained :class:`~manifold_sae.sae.ManifoldSAE` instance.
    activations:
        ``(N, D)`` real activations. Used as the batch context for the inner
        solve that yields the trained coefficients.
    t_grid:
        ``(T,)`` positions to evaluate each feature's curve at.

    Returns
    -------
    ``(F, T, D)`` tensor; ``curves[k, i]`` is feature k's value at
    ``t_grid[i]`` in R^D.
    """
    import gamfit
    import numpy as np

    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid = t_grid.to(device=device, dtype=activations.dtype)

    sae.eval()
    out = sae(activations)
    coefficients = out.coefficients  # (F, K, D)

    K = sae.basis_spec.n_basis
    centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
    t_grid_np = t_grid.detach().to(device="cpu", dtype=torch.float64).numpy()
    phi = gamfit.duchon_basis_1d(t_grid_np, centers, m=2, periodic=False)  # (T, K)
    phi_t = torch.as_tensor(np.asarray(phi), dtype=activations.dtype, device=device)

    # The trained coefficients absorb an inverse-`by` factor: at training time
    # the reconstruction is `by[i,k] * phi(t[i,k]) @ B_k`, so B_k is sized to
    # the typical amplitude scale. Multiply by the mean of amplitudes for
    # firing tokens (amplitude > small threshold) per feature so the probed
    # curve is reported at the scale a typical-firing token would see — this
    # matches the ground-truth convention `amp * curve_gt(t)`.
    amp = out.amplitudes  # (N, F)
    firing = amp > 1e-3
    denom = firing.to(amp.dtype).sum(dim=0).clamp(min=1.0)
    mean_amp = (amp * firing.to(amp.dtype)).sum(dim=0) / denom  # (F,)
    coef_scaled = coefficients.to(activations.dtype) * mean_amp.view(-1, 1, 1)

    # curves[k] = phi @ (mean_amp[k] * coefficients[k]) -> einsum over K axis.
    curves = torch.einsum("tk,fkd->ftd", phi_t, coef_scaled)
    return curves
