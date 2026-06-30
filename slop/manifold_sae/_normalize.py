"""Activation normalization utilities — the proper way.

The original pipeline used `(X - X.mean(0)) / X.std()` where `X.std()`
returns a single scalar over all elements. On LM residuals this is
catastrophic: the residual stream has one direction (the "norm direction")
carrying ~99% of total variance, so a scalar std preserves rank-1 structure.
Diagnostic at Qwen-1.5B L4/L8/L12/L18 showed 99% variance in 1 PC under
this normalization vs. 925+ PCs under per-dimension normalization.

This module exposes `normalize_activations` and `inverse_normalize` that
do per-dimension std normalization (standard SAE practice). The
`NormalizationStats` dataclass keeps the per-dim mean + std so we can
invert when needed (e.g. for downstream LM patching).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalizationStats:
    """Per-feature mean and std for invertible normalization."""

    mean: torch.Tensor   # shape (D,)
    std: torch.Tensor    # shape (D,), clamped to ≥ eps


def fit_normalize(X: torch.Tensor, eps: float = 1e-6) -> NormalizationStats:
    """Compute per-dimension mean + std stats from `X` of shape `(N, D)`.

    `std` is clamped to `eps` to avoid divide-by-zero in flat dimensions
    (rare but possible at boundary layers).
    """
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=eps)
    return NormalizationStats(mean=mean, std=std)


def normalize_activations(
    X: torch.Tensor,
    stats: NormalizationStats | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, NormalizationStats]:
    """Normalize `X` per dimension: ``(X - mean) / std``.

    Centers by per-feature mean and scales by per-feature std. This
    PRESERVES rank — each dimension contributes proportionally — unlike
    the scalar-std normalization we previously used.

    If `stats` is given, applies the same normalization (use this when
    normalizing eval/holdout data with stats fit on train). Otherwise
    fits stats from `X` and returns them so they can be re-used.

    Returns: `(X_normalized, stats)`. Stats live on `X`'s device/dtype.
    """
    if stats is None:
        stats = fit_normalize(X, eps=eps)
    X_norm = (X - stats.mean.to(X)) / stats.std.to(X)
    return X_norm, stats


def inverse_normalize(
    X_norm: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """Map normalized activations back to the original activation space."""
    return X_norm * stats.std.to(X_norm) + stats.mean.to(X_norm)
