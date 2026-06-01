"""Manifold-SAE — thin gamfit-native wrapper over ``gamfit.torch.ManifoldSAE``.

FULL BREAKING CUTOVER (gamfit 0.1.123+)
=======================================
This module no longer contains a hand-rolled SAE. It re-exports the gamfit
primitive :class:`gamfit.torch.ManifoldSAE` (a trainable ``nn.Module``) and its
config/output dataclasses verbatim, and adds only Manifold-SAE-side glue:

* a module-level :func:`extract_feature_curves` helper that adapts the
  primitive's :meth:`ManifoldSAE.extract_feature_curves` (grid -> per-atom
  ambient curves) to the ``(sae, activations, t_grid) -> (F, T, D)`` signature
  the synthetic-recovery experiment expects, and
* :func:`lift_atom_curve`, a decoder-side probe that gives the per-atom ambient
  curve ``g_k(t)`` used by the interpretability tools (atom_traversal,
  feature_dashboard) in place of the old ``W_k`` (``directions``) lift.

SEMANTIC CUTOVER NOTES
----------------------
* Output schema is the new gamfit bundle
  ``{z, x_hat, positions, amplitudes, curves, gate, assignments, reml_score,
  lambdas}``. The old ``{reconstruction, mask_soft, coefficients, lam, fitted,
  directions, ortho_loss, monotonicity_loss}`` schema is GONE. Use ``x_hat``
  for reconstruction, ``amplitudes``/``z`` for activation, and call
  ``sae.decoder_ortho_penalty()`` / ``sae.decoder_monotonicity_penalty()`` /
  ``sae.regularization(logits)`` for the regularizers (they are no longer
  fields of the output).
* ``positions`` is ``(N, F, d)`` (per-atom manifold coordinate), NOT ``(N, F)``.
  The intrinsic manifold dimension ``d`` is ``cfg.intrinsic_rank``.
* There is no ``directions`` / ``W_k`` parameter. An atom's ambient behaviour is
  fully described by its on-manifold curve ``curves @ decoder_blocks[k]`` —
  i.e. ``decoder_blocks`` already lives in ambient ``R^D``. Use
  :func:`lift_atom_curve` for the per-atom curve.
* Training uses backprop on encoder/anchor/log_lambda plus the closed-form REML
  solve via :meth:`ManifoldSAE.fit`; deploy via :meth:`ManifoldSAE.lock_snapshot`.
  The old per-batch ``gamfit.torch.fit`` + ``update_snapshot`` /
  ``inference_mode`` dance is gone.

Config field renames (old -> new) for all Manifold-SAE callers:
    n_features        -> n_atoms
    n_basis           -> n_basis_per_atom
    top_k             -> sparsity.target_k   (SparsityConfig)
    sparsity_weight   -> (drop; sparsity is structural via SparsityConfig)
    ortho_weight      -> decoder.ortho_weight (DecoderConfig)
    reml_weight       -> (drop; REML drives the closed-form solve, not a loss term)
    encoder_type      -> encoder_hidden (0 == linear encoder)
    continuous_amp    -> (drop; amplitudes are always continuous softplus)
    periodic          -> atom_manifold="circle"
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from gamfit.torch import (
    DecoderConfig,
    ManifoldSAE,
    ManifoldSAEConfig,
    ManifoldSAEOutput,
    RemlConfig,
    SparsityConfig,
)

__all__ = [
    "DecoderConfig",
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "RemlConfig",
    "SparsityConfig",
    "load_sae",
    "extract_feature_curves",
    "lift_atom_curve",
]


def load_sae(
    path: str | Path,
    input_dim: int,
    device: torch.device,
    dtype: Any = torch.float32,
) -> ManifoldSAE:
    """Load a gamfit-native ManifoldSAE checkpoint.

    Expects a new-API checkpoint dict carrying a stored
    :class:`ManifoldSAEConfig` under ``"config"`` and the module weights under
    ``"state_dict"``::

        {"config": ManifoldSAEConfig(...), "state_dict": sae.state_dict(),
         "locked": bool}

    The config is reconstructed directly — no legacy ``sig``-key translation.
    A checkpoint that lacks ``"config"`` raises a clear error rather than
    guessing the architecture.

    ``input_dim`` is validated against the stored config's ``input_dim``.
    ``dtype`` overrides the stored config dtype (defaults to ``float32``, the
    LM-inference regime). Use ``float64`` if you intend to call ``sae.fit``.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(
            f"{path!r} is not a new-API ManifoldSAE checkpoint: expected a dict "
            "with a stored ManifoldSAEConfig under 'config'. Legacy 'sig'-keyed "
            "checkpoints are no longer supported."
        )
    cfg = ckpt["config"]
    if not isinstance(cfg, ManifoldSAEConfig):
        raise TypeError(
            f"checkpoint 'config' must be a ManifoldSAEConfig, got "
            f"{type(cfg).__name__}"
        )
    if int(cfg.input_dim) != int(input_dim):
        raise ValueError(
            f"input_dim mismatch: checkpoint config has input_dim={cfg.input_dim}, "
            f"caller requested {input_dim}"
        )
    if dtype is not None and cfg.dtype != dtype:
        cfg = dataclasses.replace(cfg, dtype=dtype)
    if "state_dict" not in ckpt:
        raise ValueError(
            f"{path!r} is missing module weights under 'state_dict'"
        )
    sae = ManifoldSAE(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    if bool(ckpt.get("locked", False)) and not sae.is_locked:
        sae.lock_snapshot()
    return sae


@torch.no_grad()
def extract_feature_curves(
    sae: ManifoldSAE,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Per-atom learned curves sampled on ``t_grid`` in ambient ``R^D``.

    Adapts the primitive's :meth:`ManifoldSAE.extract_feature_curves`
    (which returns a ``{atom -> (grid, D)}`` dict over its own internal grid)
    to the ``(F, T, D)`` tensor the synthetic-recovery experiment consumes.

    ``activations`` seeds the closed-form REML solve when the SAE has not yet
    been locked (so the decoder blocks reflect the planted manifold rather than
    random init). ``t_grid`` only sets the grid resolution ``T``; the primitive
    samples its own manifold-appropriate grid (e.g. ``[0, 1]`` for circle), so
    the exact sample locations are owned by gamfit.
    """
    device = next(sae.parameters()).device
    if not bool(getattr(sae, "is_locked", False)):
        # Drive the closed-form solve so decoder_blocks reflect the data, then
        # freeze for deterministic curve extraction.
        sae.fit(activations.to(device=device, dtype=sae.cfg.dtype))
        sae.lock_snapshot()
    grid_size = int(t_grid.shape[0])
    curve_dict = sae.extract_feature_curves(grid_size=grid_size)
    F = int(sae.cfg.n_atoms)
    # Stack atom -> (T, D) into (F, T, D); each value already lives in ambient R^D.
    ordered = [curve_dict[i] for i in range(F)]
    curves = torch.stack(ordered, dim=0)
    return curves.to(dtype=activations.dtype)


@torch.no_grad()
def lift_atom_curve(
    sae: ManifoldSAE,
    atom_k: int,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Ambient curve ``g_k(t)`` for a single atom, shape ``(len(t_grid), D)``.

    Replaces the old ``(phi @ B_k) @ W_k^T`` lift. The new decoder block already
    lives in ambient ``R^D``, so the curve is ``phi(t) @ decoder_blocks[k]``;
    we obtain it by slicing the primitive's per-atom grid curve. ``t_grid`` only
    sets the grid resolution; the primitive owns the manifold sample locations.
    """
    grid_size = int(t_grid.shape[0])
    curve_dict = sae.extract_feature_curves(grid_size=grid_size)
    return curve_dict[int(atom_k)].to(dtype=t_grid.dtype)
