"""Validated diagnostics for gauge-fitted concept manifolds."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .gauge import GaugeFit, fit_gauge


def per_anchor_curvature(
    gauge: GaugeFit,
    activations: NDArray[np.floating],
    anchor_rows: Mapping[str, Sequence[int]],
    *,
    k: int = 8,
) -> dict[str, float]:
    """Estimate per-anchor curvature from local ambient residual variance.

    Implements the ``auto_exp_52`` diagnostic: high scores indicate that a
    single flat anchor offset is locally under-specified.
    """
    x = np.asarray(activations, dtype=np.float64)
    z = gauge.transform(x)
    out: dict[str, float] = {}
    for name, rows_seq in anchor_rows.items():
        rows = np.asarray(rows_seq, dtype=np.int64)
        center = z[rows].mean(axis=0)
        nearest = np.argsort(((z - center) ** 2).sum(axis=1))[: max(k, gauge.d + 1)]
        local_z = z[nearest] - z[nearest].mean(axis=0)
        local_x = x[nearest] - x[nearest].mean(axis=0)
        latent_var = float((local_z**2).sum())
        ambient_var = float((local_x**2).sum()) + 1e-12
        out[name] = float(np.clip(1.0 - latent_var / ambient_var, 0.0, 1.0))
    return out


def null_topology_control(
    gauge: GaugeFit,
    activations: NDArray[np.floating],
    labels: Mapping[str, Sequence[float] | NDArray[np.floating]],
    *,
    n_perm: int = 100,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Permutation null for topology and target-predictiveness claims.

    This is the ``auto_exp_42`` control. Each target is shuffled, refit with
    the same PCA budget, and compared to observed in-sample R-squared.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for target in gauge.targets:
        if target not in labels:
            continue
        observed = max((v for k, v in gauge.r2.items() if k == target or k.startswith(f"{target}_")), default=0.0)
        null = np.empty(n_perm, dtype=np.float64)
        label_arr = np.asarray(labels[target], dtype=np.float64)
        for i in range(n_perm):
            shuffled = label_arr.copy()
            rng.shuffle(shuffled, axis=0)
            refit = fit_gauge(activations, {target: shuffled}, targets=[target], d=gauge.d, k=gauge.pca_basis.shape[1])
            null[i] = max(refit.r2.values())
        out[target] = {
            "r2": float(observed),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p": float((null >= observed).mean()),
        }
    return out


def variance_vs_concept_locality(
    gauge: GaugeFit,
    activations: NDArray[np.floating],
    anchor_rows: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, float]]:
    """Measure variance-locality tradeoffs for anchor steering.

    Implements the ``auto_85`` check: reliable steering tends to have high
    anchor-offset variance inside the fitted subspace and compact anchor
    neighborhoods.
    """
    x = np.asarray(activations, dtype=np.float64)
    z = gauge.transform(x)
    out: dict[str, dict[str, float]] = {}
    for name, rows_seq in anchor_rows.items():
        rows = np.asarray(rows_seq, dtype=np.int64)
        offset = x[rows].mean(axis=0) - gauge.mu
        proj = gauge.axes.T @ offset
        frac = float((proj**2).sum() / max(float((offset**2).sum()), 1e-12))
        radius = float(np.linalg.norm(z[rows] - z[rows].mean(axis=0), axis=1).mean())
        out[name] = {"subspace_fraction": frac, "latent_radius": radius}
    return out


def validated_diagnostics(
    gauge: GaugeFit,
    activations: NDArray[np.floating],
    labels: Mapping[str, Sequence[float] | NDArray[np.floating]],
    anchor_rows: Mapping[str, Sequence[int]],
    *,
    n_perm: int = 100,
) -> dict[str, object]:
    """Run all validated diagnostics and return JSON-serializable records."""
    return {
        "per_anchor_curvature_auto_exp_52": per_anchor_curvature(gauge, activations, anchor_rows),
        "null_topology_control_auto_exp_42": null_topology_control(
            gauge, activations, labels, n_perm=n_perm
        ),
        "variance_vs_concept_locality_auto_85": variance_vs_concept_locality(
            gauge, activations, anchor_rows
        ),
    }


def plot_diagnostics(report: Mapping[str, object], path: str | Path) -> Path:
    """Render a compact diagnostics figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise NotImplementedError("plot_diagnostics requires matplotlib") from exc
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    curv = report.get("per_anchor_curvature_auto_exp_52", {})
    if isinstance(curv, Mapping) and curv:
        axes[0].bar(list(curv), list(curv.values()))
        axes[0].set_title("curvature")
        axes[0].tick_params(axis="x", rotation=45)
    null = report.get("null_topology_control_auto_exp_42", {})
    if isinstance(null, Mapping) and null:
        names = list(null)
        vals = [float(null[n]["r2"]) for n in names]  # type: ignore[index]
        axes[1].bar(names, vals)
        axes[1].set_title("target R2")
        axes[1].tick_params(axis="x", rotation=45)
    loc = report.get("variance_vs_concept_locality_auto_85", {})
    if isinstance(loc, Mapping) and loc:
        xs = [float(loc[n]["subspace_fraction"]) for n in loc]  # type: ignore[index]
        ys = [float(loc[n]["latent_radius"]) for n in loc]  # type: ignore[index]
        axes[2].scatter(xs, ys)
        axes[2].set_title("locality")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=140)
    return out
