"""Steering primitives for ManifoldSAE, LinearSAE, and the diff-of-means baseline.

Three steering operators, all returning a tensor of the same shape as the input
activations:

- ``steer_manifold``: shift the intrinsic position of a chosen ManifoldSAE feature.
  Caller declares ``cyclic=True`` to wrap modulo 1 (days, hours); otherwise clamps to
  [0, 1]. This is the primitive the project hypothesizes wins on curved geometry: the
  activation moves *along* the learned manifold rather than teleporting in ambient space.

- ``steer_linear``: add a scalar (or per-example) amplitude bump to a chosen LinearSAE
  feature and decode. This is the LRH-style steering operator that the literature shows
  produces brittle behaviour for curved concepts.

- ``steer_baseline_diff_means``: the AxBench / Panickssery-style "difference of means"
  contrast vector applied directly in activation space. No SAE involved. AxBench shows
  this beats linear SAEs on most behavioural steering tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .sae import ManifoldSAE

DeltaLike = float | torch.Tensor


def _broadcast_delta(delta: DeltaLike, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(delta, torch.Tensor):
        t = delta.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.expand(batch)
        if t.ndim == 1 and t.shape[0] == batch:
            return t
        raise ValueError(f"delta tensor shape {tuple(t.shape)} incompatible with batch {batch}")
    return torch.full((batch,), float(delta), device=device, dtype=dtype)


def _shift_position(
    positions: torch.Tensor,
    feature_idx: int,
    delta: torch.Tensor,
    periodic: bool,
) -> torch.Tensor:
    """Return positions with feature ``feature_idx`` moved by ``delta`` along [0, 1]."""
    new_positions = positions.clone()
    moved = new_positions[:, feature_idx] + delta
    if periodic:
        moved = moved - torch.floor(moved)  # wrap to [0, 1)
    else:
        moved = moved.clamp(0.0, 1.0)
    new_positions[:, feature_idx] = moved
    return new_positions


def steer_manifold(
    sae: ManifoldSAE,
    x: torch.Tensor,
    feature_idx: int,
    delta_position: DeltaLike,
    *,
    cyclic: bool = False,
) -> torch.Tensor:
    """Move feature ``feature_idx``'s intrinsic position by ``delta_position`` and decode.

    Other features' inferred (position, amplitude) pairs stay fixed; only
    feature_idx's manifold contribution is re-evaluated at the new coordinate.

    ``cyclic`` is a caller-level declaration about the *concept* being steered
    (days-of-week wraps, magnitude doesn't). When True, positions wrap modulo 1;
    when False, positions clamp to [0, 1]. This is independent of the SAE's
    basis (v1 fits everything as open 1D Duchon).
    """
    sae.eval()
    with torch.no_grad():
        out = sae(x)
        positions = out.positions
        amplitudes = out.amplitudes

    if feature_idx < 0 or feature_idx >= positions.shape[1]:
        raise IndexError(f"feature_idx {feature_idx} out of range for n_features={positions.shape[1]}")

    delta = _broadcast_delta(delta_position, batch=x.shape[0], device=x.device, dtype=positions.dtype)
    new_positions = _shift_position(positions, feature_idx, delta, periodic=cyclic)

    # Re-decode with the shifted positions. We use ``x`` as the target because
    # manifold_fit needs a target to fit smoother coefficients; the steered output is
    # the resulting reconstruction at the new positions.
    from .decoder import decode

    with torch.no_grad():
        fit = decode(new_positions, amplitudes, x, sae.basis_spec)
    return fit["reconstruction"]


def steer_linear(
    sae,
    x: torch.Tensor,
    feature_idx: int,
    delta_amplitude: DeltaLike,
) -> torch.Tensor:
    """Encode x, add ``delta_amplitude`` to feature_idx's amplitude, decode.

    This is the LRH steering operator: a translation in dictionary-coefficient space.
    On non-linear / curved concepts this is the operation that produces the
    "teleportation" failure mode the paper is targeting.
    """
    sae.eval()
    with torch.no_grad():
        amps = sae.encode(x)
        if feature_idx < 0 or feature_idx >= amps.shape[1]:
            raise IndexError(f"feature_idx {feature_idx} out of range for n_features={amps.shape[1]}")
        delta = _broadcast_delta(delta_amplitude, batch=x.shape[0], device=x.device, dtype=amps.dtype)
        new_amps = amps.clone()
        new_amps[:, feature_idx] = new_amps[:, feature_idx] + delta
        recon = sae.decode(new_amps)
    return recon


def steer_baseline_diff_means(
    activations: torch.Tensor,
    labels_source: Sequence,
    labels_target: Sequence,
    x: torch.Tensor,
    alpha: float,
    all_labels: Sequence | None = None,
) -> torch.Tensor:
    """Diff-of-means steering: x + alpha * (mean_target - mean_source).

    ``activations`` is the harvested dataset (N, D); ``all_labels`` aligns 1:1 with its
    rows. ``labels_source`` / ``labels_target`` are the label values that define the two
    contrast groups. If ``all_labels`` is None we expect ``labels_source`` and
    ``labels_target`` to already be parallel index arrays into ``activations``.
    """
    if all_labels is None:
        # Treat the label arguments as direct index lists.
        src_idx = torch.as_tensor(list(labels_source), dtype=torch.long, device=activations.device)
        tgt_idx = torch.as_tensor(list(labels_target), dtype=torch.long, device=activations.device)
        src = activations.index_select(0, src_idx)
        tgt = activations.index_select(0, tgt_idx)
    else:
        labels_list = list(all_labels)
        src_set = set(labels_source)
        tgt_set = set(labels_target)
        src_mask = torch.tensor([lab in src_set for lab in labels_list], dtype=torch.bool, device=activations.device)
        tgt_mask = torch.tensor([lab in tgt_set for lab in labels_list], dtype=torch.bool, device=activations.device)
        if not src_mask.any():
            raise ValueError(f"no rows in activations match labels_source={labels_source}")
        if not tgt_mask.any():
            raise ValueError(f"no rows in activations match labels_target={labels_target}")
        src = activations[src_mask]
        tgt = activations[tgt_mask]

    direction = tgt.mean(dim=0) - src.mean(dim=0)
    return x + alpha * direction.to(x.device, x.dtype)


__all__ = [
    "steer_manifold",
    "steer_linear",
    "steer_baseline_diff_means",
]
