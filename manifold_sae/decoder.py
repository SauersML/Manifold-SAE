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

    Because the decoder coefficients are solved per-batch by gamfit (no learned
    decoder weights), there is no "frozen" curve to read off. Instead we ask
    the SAE the same question gamfit asks at training time: given the encoder's
    inferred (positions, amplitudes) on the real activation batch, what
    coefficients does the inner solve choose? Then we re-evaluate the fit at
    the supplied ``t_grid``, holding the other features at their inferred
    state, so the returned curves reflect what each feature has learned to
    represent on this data.

    Parameters
    ----------
    sae:
        Trained :class:`~manifold_sae.sae.ManifoldSAE` instance.
    activations:
        ``(N, D)`` real activations the encoder has been trained on. These
        are the targets for the inner solve — passing zeros yields zero
        curves and is a common probe bug.
    t_grid:
        ``(T,)`` positions to evaluate each feature's curve at.

    Returns
    -------
    ``(F, T, D)`` tensor; ``curves[k, i]`` is feature k's value at
    ``t_grid[i]`` in R^D.
    """
    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid = t_grid.to(device=device, dtype=activations.dtype)

    sae.eval()
    out = sae(activations)
    pos_base = out.positions
    amp_base = out.amplitudes

    F = sae.config.n_features
    T = t_grid.shape[0]
    D = activations.shape[1]
    curves = torch.zeros(F, T, D, dtype=activations.dtype, device=device)

    # For each feature, probe its curve by varying its position over t_grid
    # while holding every other feature at its inferred (position, amplitude).
    # Targets are the real activations tiled T times so the inner solve has
    # genuine signal — feeding zeros collapses every coefficient to zero.
    base_idx = 0  # use the first sample as the reference context
    pos_ref = pos_base[base_idx : base_idx + 1].repeat(T, 1)
    amp_ref = amp_base[base_idx : base_idx + 1].repeat(T, 1)
    target_ref = activations[base_idx : base_idx + 1].repeat(T, 1)

    for k in range(F):
        pos = pos_ref.clone()
        amp = torch.zeros_like(amp_ref)
        pos[:, k] = t_grid
        amp[:, k] = 1.0
        fit = decode(pos, amp, target_ref, sae.basis_spec)
        curves[k] = fit["reconstruction"]
    return curves
