"""auto_exp_15: persistent homology of cogito L40 per-color centroids.

GOAL
----
Does cogito's color manifold contain a robust H¹ generator (a topological
circle), discovered from activations alone (not prescribed by RGB
metadata)?  Prior evidence (auto_54, auto_61, auto_66) found
hue-aligned tangents and near-linear hue-rotation group action, but
those analyses used RGB labels. This experiment asks the unsupervised
topological question.

METHOD
------
1. Build per-color centroids via the canonical pipeline:
   - mmap'd X_L40.npy harvest
   - mean over TOP_TEMPLATES=[8,13,16,17,18,5]
   - drop bad color names via color_filter_list.filter_colors
   - project to top-64 PCs via _pca_basis.load_pc_basis(K=64) + project()
2. Pairwise Euclidean distance matrix.
3. ripser.ripser(D, maxdim=1, distance_matrix=True).
4. Report top-3 H¹ persistences vs median noise H¹ persistence.
5. Persistence ratio: top H¹ / (max finite H₀ persistence).
6. Sweep K_PC in {3, 8, 16, 32, 64}.

NO Gaussian RBF, NO Duchon length_scale, NO B-splines.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_exp_15.png"
OUT_JSON = OUT_DIR / "auto_exp_15.json"

K_TOP = 64
K_SWEEP = [3, 8, 16, 32, 64]


def _ensure_ripser():
    try:
        import ripser  # noqa: F401
        return "ripser"
    except ImportError:
        pass
    print("[install] ripser not found; running `uv pip install ripser`", flush=True)
    rc = subprocess.call(["uv", "pip", "install", "ripser"])
    if rc == 0:
        try:
            import ripser  # noqa: F401
            return "ripser"
        except ImportError:
            pass
    print("[install] ripser install failed; trying gudhi", flush=True)
    try:
        import gudhi  # noqa: F401
        return "gudhi"
    except ImportError:
        pass
    rc = subprocess.call(["uv", "pip", "install", "gudhi"])
    if rc == 0:
        try:
            import gudhi  # noqa: F401
            return "gudhi"
        except ImportError:
            pass
    raise RuntimeError("Neither ripser nor gudhi available; bailing.")


def _compute_ph(D: np.ndarray, maxdim: int, backend: str):
    """Return (h0, h1) lists of (birth, death) pairs."""
    if backend == "ripser":
        from ripser import ripser
        out = ripser(D, maxdim=maxdim, distance_matrix=True)
        dgms = out["dgms"]
        return dgms[0], dgms[1]
    elif backend == "gudhi":
        import gudhi
        rc = gudhi.RipsComplex(distance_matrix=D.tolist(),
                               max_edge_length=float(D.max()))
        st = rc.create_simplex_tree(max_dimension=maxdim + 1)
        st.compute_persistence()
        h0 = np.array([(b, d) for dim, (b, d) in st.persistence_intervals_in_dimension(0)
                       .reshape(-1, 2)]) if False else np.array(st.persistence_intervals_in_dimension(0))
        h1 = np.array(st.persistence_intervals_in_dimension(1))
        if h0.size == 0:
            h0 = np.zeros((0, 2))
        if h1.size == 0:
            h1 = np.zeros((0, 2))
        return h0, h1
    else:
        raise ValueError(backend)


def _top_h1(h1: np.ndarray, k: int = 3) -> list[tuple[float, float, float]]:
    """Return top-k (birth, death, persistence) sorted by persistence desc."""
    if h1.size == 0:
        return []
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return []
    pers = finite[:, 1] - finite[:, 0]
    order = np.argsort(pers)[::-1]
    out = []
    for i in order[:k]:
        out.append((float(finite[i, 0]), float(finite[i, 1]), float(pers[i])))
    return out


def _max_finite_h0_pers(h0: np.ndarray) -> float:
    if h0.size == 0:
        return 0.0
    finite = h0[np.isfinite(h0[:, 1])]
    if finite.size == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())


def _betti1_curve(h1: np.ndarray, n_pts: int = 400):
    if h1.size == 0:
        return np.linspace(0, 1, n_pts), np.zeros(n_pts)
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return np.linspace(0, 1, n_pts), np.zeros(n_pts)
    s_max = float(finite[:, 1].max())
    scales = np.linspace(0.0, s_max * 1.02, n_pts)
    counts = np.zeros(n_pts, dtype=np.int64)
    for s_idx, s in enumerate(scales):
        counts[s_idx] = int(((finite[:, 0] <= s) & (finite[:, 1] > s)).sum())
    return scales, counts


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    backend = _ensure_ripser()
    print(f"[backend] using {backend}", flush=True)

    # ---- Load harvest mmap'd, build TOP_TEMPLATES centroids ----
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    n_colors_raw = n_total // N_TEMPLATES
    print(f"[load] X={X.shape} -> n_colors_raw={n_colors_raw}", flush=True)

    centroids = np.zeros((n_colors_raw, H), dtype=np.float64)
    for ci in range(n_colors_raw):
        base = ci * N_TEMPLATES
        rows = [base + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(axis=0)
    print(f"[cent] centroids={centroids.shape} (avg over {len(TOP_TEMPLATES)} templates)",
          flush=True)

    # ---- Filter color names ----
    colors_all = load_xkcd_colors()[:n_colors_raw]
    kept, kept_idx = filter_colors(colors_all)
    kept_idx = np.array(kept_idx, dtype=np.int64)
    centroids = centroids[kept_idx]
    n_colors = centroids.shape[0]
    print(f"[filt] kept {n_colors} colors after filter_colors", flush=True)

    # ---- Project to top-64 PCs via canonical basis ----
    basis64 = load_pc_basis(K=K_TOP)
    Z64 = project(centroids, basis64)               # (n_colors, 64)
    print(f"[pca ] Z64={Z64.shape}, EVR_sum={basis64['evr'][:K_TOP].sum():.3f}",
          flush=True)

    # ---- PH on Z64 ----
    D = np.linalg.norm(Z64[:, None, :] - Z64[None, :, :], axis=-1)
    print(f"[dist] D={D.shape}, max={D.max():.3f}, median={np.median(D):.3f}",
          flush=True)

    print(f"[ph  ] computing maxdim=1 on K={K_TOP} ...", flush=True)
    h0, h1 = _compute_ph(D, maxdim=1, backend=backend)
    n_h1 = int(h1.shape[0]) if h1.size else 0
    print(f"[ph  ] H0={h0.shape[0]} bars, H1={n_h1} bars", flush=True)

    top3 = _top_h1(h1, k=3)
    max_h0 = _max_finite_h0_pers(h0)
    # median noise persistence: all H1 bars (each represents a generator)
    if n_h1 > 0:
        all_pers = h1[:, 1] - h1[:, 0]
        all_pers = all_pers[np.isfinite(all_pers)]
        med_noise = float(np.median(all_pers))
        mean_noise = float(np.mean(all_pers))
    else:
        med_noise = 0.0
        mean_noise = 0.0
    top1_pers = top3[0][2] if top3 else 0.0
    pers_ratio_top1_over_h0 = top1_pers / max(max_h0, 1e-12)
    pers_ratio_top1_over_noise = top1_pers / max(med_noise, 1e-12)

    print(f"[ph  ] top-3 H1 persistences: "
          + ", ".join(f"{p:.3f}" for _, _, p in top3), flush=True)
    print(f"[ph  ] median noise H1 pers = {med_noise:.4f}", flush=True)
    print(f"[ph  ] top1/max_H0 ratio = {pers_ratio_top1_over_h0:.3f}", flush=True)
    print(f"[ph  ] top1/median_noise ratio = {pers_ratio_top1_over_noise:.2f}",
          flush=True)

    # ---- PC sweep ----
    sweep = []
    for K in K_SWEEP:
        Z = Z64[:, :K]
        DK = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        print(f"[swp ] K={K} ...", flush=True)
        _h0K, h1K = _compute_ph(DK, maxdim=1, backend=backend)
        topK = _top_h1(h1K, k=3)
        sweep.append({
            "K": K,
            "n_h1": int(h1K.shape[0]) if h1K.size else 0,
            "top3": topK,
            "top1_persistence": topK[0][2] if topK else 0.0,
            "median_h1_pers": float(np.median(h1K[:, 1] - h1K[:, 0]))
                              if h1K.size else 0.0,
        })
        print(f"[swp ] K={K}: top1={sweep[-1]['top1_persistence']:.3f}, "
              f"n_H1={sweep[-1]['n_h1']}", flush=True)

    # ---- Betti curve from K=64 result ----
    scales, betti1 = _betti1_curve(h1, n_pts=400)

    # ---- Verdict ----
    # Robust circle if: top1 >> median noise (ratio > ~3) and not vanishing
    # in low-K projection.
    robust_circle = (pers_ratio_top1_over_noise > 3.0
                     and top1_pers > 0
                     and sweep[0]["top1_persistence"] > 0)
    survives_low_pc = (sweep[0]["top1_persistence"] > 0
                       and (sweep[0]["top1_persistence"]
                            / max(top1_pers, 1e-12)) > 0.5)
    verdict = (
        "ROBUST CIRCLE PRESENT" if robust_circle else
        "MARGINAL / NO ROBUST CIRCLE"
    )

    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors_raw": int(n_colors_raw),
            "n_colors_after_filter": int(n_colors),
            "top_templates": list(TOP_TEMPLATES),
            "K_top": int(K_TOP),
            "K_sweep": list(K_SWEEP),
            "backend": backend,
        },
        "K64": {
            "n_h0_bars": int(h0.shape[0]) if h0.size else 0,
            "n_h1_bars": int(n_h1),
            "top3_h1": [{"birth": b, "death": d, "persistence": p}
                        for (b, d, p) in top3],
            "max_finite_h0_persistence": max_h0,
            "median_noise_h1_persistence": med_noise,
            "mean_noise_h1_persistence": mean_noise,
            "ratio_top1_over_max_h0": pers_ratio_top1_over_h0,
            "ratio_top1_over_median_noise": pers_ratio_top1_over_noise,
        },
        "pc_sweep": sweep,
        "verdict": {
            "robust_h1_generator": bool(robust_circle),
            "loop_survives_low_pc_K3": bool(survives_low_pc),
            "summary": verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) Persistence diagram on Z64.
    ax = axes[0]
    if h0.size:
        h0f = h0[np.isfinite(h0[:, 1])]
        if h0f.size:
            ax.scatter(h0f[:, 0], h0f[:, 1], s=14, c="steelblue",
                       alpha=0.5, label=f"H0 ({h0f.shape[0]})", edgecolor="none")
    if h1.size:
        h1f = h1[np.isfinite(h1[:, 1])]
        if h1f.size:
            ax.scatter(h1f[:, 0], h1f[:, 1], s=18, c="firebrick",
                       alpha=0.45, label=f"H1 ({h1f.shape[0]})",
                       edgecolor="none")
    # Highlight top-3 H1
    for rank, (b, d, p) in enumerate(top3):
        ax.scatter([b], [d], s=160, facecolor="none",
                   edgecolor="black", lw=1.6, zorder=10)
        ax.annotate(f"#{rank+1} p={p:.2f}", (b, d),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    lim_lo = 0.0
    if h1.size and np.isfinite(h1[:, 1]).any():
        lim_hi = max(float(h1[np.isfinite(h1[:, 1])][:, 1].max()),
                     float(h0[np.isfinite(h0[:, 1])][:, 1].max())
                     if (h0.size and np.isfinite(h0[:, 1]).any()) else 0.0)
    else:
        lim_hi = float(D.max())
    lim_hi *= 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.6)
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(f"Persistence diagram on Z(K={K_TOP})\n"
                 f"top-1 H1 / median-noise = {pers_ratio_top1_over_noise:.1f}×")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)

    # (b) Top-1 H1 persistence vs K_PC.
    ax = axes[1]
    Ks = [s["K"] for s in sweep]
    top1s = [s["top1_persistence"] for s in sweep]
    meds = [s["median_h1_pers"] for s in sweep]
    x = np.arange(len(Ks))
    ax.bar(x - 0.18, top1s, width=0.36, color="firebrick",
           edgecolor="k", lw=0.4, label="top-1 H1 persistence")
    ax.bar(x + 0.18, meds, width=0.36, color="gray",
           edgecolor="k", lw=0.4, label="median H1 persistence (noise)")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K_PC (number of principal components)")
    ax.set_ylabel("persistence (death - birth)")
    ax.set_title("Top-1 H1 persistence vs PC count")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    # (c) Betti-1 curve (K=64).
    ax = axes[2]
    ax.plot(scales, betti1, color="firebrick", lw=1.4)
    ax.fill_between(scales, 0, betti1, color="firebrick", alpha=0.18)
    if top3:
        for rank, (b, d, p) in enumerate(top3):
            ax.axvspan(b, d, color="black", alpha=0.04 + 0.04 * (2 - rank))
    ax.set_xlabel("scale (Euclidean distance)")
    ax.set_ylabel("number of alive H1 generators")
    ax.set_title(f"Betti-1 curve (K={K_TOP})  max β₁ = {int(betti1.max())}")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"auto_exp_15: persistent homology of cogito L40 color centroids  "
        f"({n_colors} colors, top-{K_TOP} PCs)\n"
        f"VERDICT: {verdict}   "
        f"(top-1 H1 pers={top1_pers:.3f}, median noise={med_noise:.3f}, "
        f"ratio={pers_ratio_top1_over_noise:.1f}×, top1/max_H0={pers_ratio_top1_over_h0:.2f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
