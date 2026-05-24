"""Autointerp `score` stage — simulation accuracy + bootstrap CIs.

Anthropic-style protocol:
  For each atom h, build features F(h) from its structured hypothesis (HSV
  ranges, name regex match, template-id membership). Fit a per-atom
  regression on the val set predicting actual atom activation from F(h),
  and report R². Hypothesis R² ≈ "if the hypothesis is right, how well does
  it predict held-out firings?"

  Aggregate across atoms (median + 95% bootstrap CI) per model. The model
  whose atoms are most simulation-predictable IS the more interpretable
  one, modulo capacity-equalized R² on reconstruction (reported in the
  comparison table for context).
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .explain import AtomHypothesis


def simulation_features(
    h: AtomHypothesis,
    color_hsv: np.ndarray,
    color_names: list[str],
    row_color: np.ndarray,
    row_template: np.ndarray,
) -> np.ndarray:
    """Build (N, 5) feature matrix from hypothesis h evaluated on val rows.

    Features:
      0: in_hue_range  (circular)
      1: in_sat_range
      2: in_val_range
      3: name_regex_match
      4: template_in_pattern
    """
    n = len(row_color)
    hsv = color_hsv[row_color]
    feats = np.zeros((n, 5), dtype=np.float32)

    h_lo, h_hi = h.hue_range
    h_vals = hsv[:, 0]
    if h_hi >= h_lo:
        feats[:, 0] = ((h_vals >= h_lo) & (h_vals <= h_hi)).astype(np.float32)
    else:
        # wrap
        feats[:, 0] = ((h_vals >= h_lo) | (h_vals <= h_hi)).astype(np.float32)

    s_lo, s_hi = h.saturation_range
    feats[:, 1] = ((hsv[:, 1] >= s_lo) & (hsv[:, 1] <= s_hi)).astype(np.float32)
    v_lo, v_hi = h.lightness_range
    feats[:, 2] = ((hsv[:, 2] >= v_lo) & (hsv[:, 2] <= v_hi)).astype(np.float32)

    if h.name_pattern_regex:
        try:
            patt = re.compile(h.name_pattern_regex, re.IGNORECASE)
            name_hits = np.array([1.0 if patt.search(color_names[ci]) else 0.0 for ci in row_color], dtype=np.float32)
        except re.error:
            name_hits = np.zeros(n, dtype=np.float32)
    else:
        name_hits = np.zeros(n, dtype=np.float32)
    feats[:, 3] = name_hits

    if h.template_pattern:
        tset = set(h.template_pattern)
        feats[:, 4] = np.array([1.0 if int(t) in tset else 0.0 for t in row_template], dtype=np.float32)

    return feats


def score_hypothesis(
    h: AtomHypothesis,
    acts_val: np.ndarray,
    color_hsv: np.ndarray,
    color_names: list[str],
    row_color: np.ndarray,
    row_template: np.ndarray,
) -> dict:
    """Fit linear regression: hypothesis features → atom activation.

    Returns dict with r2, n_active, mean_act.
    """
    y = acts_val[:, h.atom_id].astype(np.float64)
    if y.var() < 1e-12 or h.n_active == 0:
        return {"r2": 0.0, "n_active": int((y > 0).sum()), "mean_act": float(y.mean())}

    F = simulation_features(h, color_hsv, color_names, row_color, row_template).astype(np.float64)
    # Add intercept
    X = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
    # least squares
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"r2": 0.0, "n_active": int((y > 0).sum()), "mean_act": float(y.mean())}
    y_hat = X @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "r2": float(np.clip(r2, -1.0, 1.0)),
        "n_active": int((y > 0).sum()),
        "mean_act": float(y.mean()),
    }


def aggregate_model_scores(
    per_atom_scores: list[dict],
    min_active: int = 5,
) -> dict:
    """Aggregate per-atom hypothesis R²s into model-level summary.

    Filter to atoms with at least min_active firings; otherwise R² is noise.
    """
    qualifying = [s for s in per_atom_scores if s["n_active"] >= min_active]
    if not qualifying:
        return {
            "n_atoms_evaluated": 0,
            "median_r2": 0.0,
            "mean_r2": 0.0,
            "frac_above_0.5": 0.0,
        }
    r2s = np.array([s["r2"] for s in qualifying], dtype=np.float64)
    return {
        "n_atoms_evaluated": len(qualifying),
        "median_r2": float(np.median(r2s)),
        "mean_r2": float(r2s.mean()),
        "frac_above_0.5": float((r2s >= 0.5).mean()),
        "r2_array": r2s.tolist(),
    }


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    statistic: str = "mean",
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(v)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = v[idx]
        boots[i] = sample.mean() if statistic == "mean" else np.median(sample)
    point = float(v.mean()) if statistic == "mean" else float(np.median(v))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi
