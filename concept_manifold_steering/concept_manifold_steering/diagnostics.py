"""Pre-flight diagnostics for a fitted :class:`GaugeFix`.

Three checks, each backed by a cogito experiment that flagged a real
failure mode:

* :func:`per_anchor_curvature`  -- auto_exp_52
* :func:`null_topology_control` -- auto_exp_42
* :func:`variance_vs_locality`  -- auto_85

These return plain dicts / arrays; :func:`plot_diagnostics` will render
them with matplotlib if installed.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np

from .gauge import GaugeFix


# ---------------------------------------------------------------------------
# 1. Per-anchor curvature ranking  (auto_exp_52)
# ---------------------------------------------------------------------------

def per_anchor_curvature(
    gauge: GaugeFix,
    X: np.ndarray,
    anchor_rows: Mapping[str, Sequence[int]],
    *,
    k: int = 8,
) -> dict[str, float]:
    """Empirical curvature of the gauge-fixed chart around each anchor.

    For each concept, we collect the k nearest training rows in the
    gauge-fixed latent space, fit a local affine plane, and report the
    fraction of residual variance that lives OUTSIDE that plane.  High
    values indicate the local manifold is not flat and steering with a
    single offset vector will under- or over-shoot (auto_exp_52).

    Returns a dict ``{concept_name: curvature_score in [0, 1]}``.
    """
    Z = gauge.transform(X)            # (N, d)
    d = Z.shape[1]
    out: dict[str, float] = {}
    for cname, rows in anchor_rows.items():
        rows = np.asarray(list(rows), dtype=int)
        center = Z[rows].mean(0)
        # k-NN in latent space (over the full harvest)
        d2 = np.sum((Z - center) ** 2, axis=1)
        order = np.argsort(d2)[: max(k, d + 1)]
        local = Z[order] - center
        # Total local variance
        v_tot = float((local ** 2).sum())
        # Variance captured by best d-flat is just the trace (already d-dim)
        # so we measure how much the *raw* activations bulge off this flat:
        Xc = X[order] - X[order].mean(0)
        v_ambient = float((Xc ** 2).sum())
        # Curvature ~ ambient_var - latent_var (normalised by ambient)
        out[cname] = float(max(0.0, 1.0 - v_tot / max(v_ambient, 1e-12)))
    return out


# ---------------------------------------------------------------------------
# 2. Null control for topology claims (auto_exp_42)
# ---------------------------------------------------------------------------

def null_topology_control(
    gauge: GaugeFix,
    X: np.ndarray,
    labels: Mapping[str, np.ndarray],
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Permutation null for each target's gauge-fixed R^2.

    Repeatedly shuffle each label, re-fit the OLS step (in PC space,
    using ``gauge.pca_basis_``), and compare to the observed R^2.
    Returns ``{target: {"r2": float, "null_mean": float,
    "null_p": float}}``.  ``null_p`` near 1.0 means your "manifold" is
    just noise.
    """
    if gauge.pca_basis_ is None or gauge.mu_ is None:
        raise RuntimeError("gauge not fitted")
    rng = np.random.default_rng(seed)
    Xc = X - gauge.mu_
    Z = Xc @ gauge.pca_basis_                # (N, K)
    ZtZ_inv = np.linalg.pinv(Z.T @ Z + 1e-4 * np.eye(Z.shape[1]))
    out: dict[str, dict[str, float]] = {}
    obs_r2 = gauge.r2()
    for name in obs_r2:
        # Reconstruct the target column (only numeric supported in null)
        key = name.split("=")[0]
        if key not in labels:
            continue
        col = np.asarray(labels[key])
        if col.dtype.kind in ("U", "S", "O"):
            # Use dummy column for the level matching the name suffix
            if "=" in name:
                lvl = name.split("=", 1)[1]
                y = (col == lvl).astype(np.float32)
            else:
                continue
        else:
            y = col.astype(np.float32)
        y = (y - y.mean()) / (y.std() + 1e-8)

        nulls = np.empty(n_perm, dtype=np.float32)
        for i in range(n_perm):
            yp = rng.permutation(y)
            w = ZtZ_inv @ (Z.T @ yp)
            yhat = Z @ w
            ss_res = float(((yp - yhat) ** 2).sum())
            ss_tot = float(((yp - yp.mean()) ** 2).sum()) + 1e-12
            nulls[i] = 1.0 - ss_res / ss_tot
        p = float((nulls >= obs_r2[name]).mean())
        out[name] = {
            "r2": float(obs_r2[name]),
            "null_mean": float(nulls.mean()),
            "null_std": float(nulls.std()),
            "null_p": p,
        }
        if p > 0.05:
            warnings.warn(
                f"target {name!r}: observed R^2={obs_r2[name]:.3f} is "
                f"not significant vs permutation null (p={p:.3f}); "
                f"per auto_exp_42 your steering signal is likely spurious.",
                stacklevel=2,
            )
    return out


# ---------------------------------------------------------------------------
# 3. Variance vs concept locality (auto_85)
# ---------------------------------------------------------------------------

def variance_vs_locality(
    gauge: GaugeFix,
    X: np.ndarray,
    anchor_rows: Mapping[str, Sequence[int]],
) -> dict[str, tuple[float, float]]:
    """For each concept, report (in-subspace variance fraction,
    within-anchor latent radius).  Concepts that have low subspace
    variance OR large within-anchor radius destabilise steering."""
    Z = gauge.transform(X)
    out: dict[str, tuple[float, float]] = {}
    for cname, rows in anchor_rows.items():
        rows = np.asarray(list(rows), dtype=int)
        v = X[rows].mean(0) - gauge.mu_
        proj = gauge.axes_.T @ v
        frac = float((proj ** 2).sum() / max((v ** 2).sum(), 1e-12))
        rad = float(np.linalg.norm(Z[rows] - Z[rows].mean(0), axis=1).mean())
        out[cname] = (frac, rad)
    return out


# ---------------------------------------------------------------------------
# Aggregate plot
# ---------------------------------------------------------------------------

def plot_diagnostics(
    gauge: GaugeFix,
    X: np.ndarray,
    labels: Mapping[str, np.ndarray],
    anchor_rows: Mapping[str, Sequence[int]],
    out_path: str | None = None,
):
    """Render a 1x3 diagnostic panel.  Requires the 'plot' extra."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_diagnostics requires matplotlib; "
                          "pip install concept_manifold_steering[plot]") from e

    curv = per_anchor_curvature(gauge, X, anchor_rows)
    null = null_topology_control(gauge, X, labels)
    locality = variance_vs_locality(gauge, X, anchor_rows)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # 1. curvature bar
    names, vals = zip(*sorted(curv.items(), key=lambda kv: -kv[1]))
    axs[0].barh(names, vals)
    axs[0].set_title("per-anchor curvature\n(auto_exp_52)")
    axs[0].set_xlabel("residual fraction off d-flat")
    # 2. null R^2
    if null:
        nm, rec = zip(*null.items())
        r2s = [r["r2"] for r in rec]
        ps  = [r["null_p"] for r in rec]
        axs[1].bar(nm, r2s, color=["C2" if p < 0.05 else "C3" for p in ps])
        axs[1].set_title("target R^2 vs null\n(auto_exp_42)")
        axs[1].set_xticklabels(nm, rotation=45, ha="right")
    # 3. locality scatter
    if locality:
        nm, pts = zip(*locality.items())
        fracs = [p[0] for p in pts]; rads = [p[1] for p in pts]
        axs[2].scatter(fracs, rads)
        for n, x, y in zip(nm, fracs, rads):
            axs[2].annotate(n, (x, y), fontsize=7)
        axs[2].set_xlabel("in-subspace variance fraction")
        axs[2].set_ylabel("within-anchor latent radius")
        axs[2].set_title("variance vs concept locality\n(auto_85)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig
