"""auto_19: eigenvalue spectrum + knee marker (idea q).

Question
--------
What does the *full* singular-value spectrum of the per-color centroid
matrix look like, and where is the elbow? Prior analyses report only
top-64 EVR (sum ≈ 0.82). We compute the full spectrum (min(n_colors,
D) eigenvalues), normalize, and locate the knee three ways:
  * Kneedle (max-distance to chord on log-EVR)
  * Participation ratio (effective rank = (Σλ)² / Σλ²)
  * 90% / 95% / 99% cumulative-variance thresholds

Plot: 2-panel
  (a) log-scale EVR per index with knee/PR/threshold markers
  (b) cumulative EVR with the same thresholds shaded

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_19.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_19.json"
OUT_PNG = OUT_DIR / "auto_19.png"
N_TEMPLATES = 28


def kneedle(y: np.ndarray) -> int:
    """Index of max perpendicular distance from log-curve to chord."""
    n = len(y)
    x = np.arange(n, dtype=np.float64)
    ly = np.log10(y.clip(min=1e-12))
    # Normalize both
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (ly - ly.min()) / (ly.max() - ly.min())
    # Chord from (0,1) to (1,0) — distance = (xn + yn - 1)/sqrt(2) for descending normalized
    # Use generic distance to line through endpoints
    p0 = np.array([xn[0], yn[0]])
    p1 = np.array([xn[-1], yn[-1]])
    v = p1 - p0
    vn = v / np.linalg.norm(v)
    pts = np.stack([xn, yn], axis=1) - p0
    proj = pts @ vn
    perp = pts - proj[:, None] * vn[None, :]
    d = np.linalg.norm(perp, axis=1)
    return int(np.argmax(d))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    print(f"[load] X={X.shape} n_colors={n_colors}", flush=True)

    # Per-color centroids
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)

    # Standardize per-feature (match prior auto_18 convention) then center
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma
    Cc = Cn - Cn.mean(0, keepdims=True)

    print(f"[svd] centroid matrix {Cc.shape} ...", flush=True)
    # economy SVD: returns min(n_colors, D) singular values
    s = np.linalg.svd(Cc, compute_uv=False)
    print(f"[svd] {len(s)} singular values, top={s[0]:.3f} tail={s[-1]:.3e}", flush=True)

    var = s ** 2
    evr = var / var.sum()
    cum = np.cumsum(evr)

    # Knee on log EVR — restrict to non-degenerate range (drop the
    # near-zero numerical tail where centroid matrix loses rank, else
    # log distance is dominated by 1e-13 outliers)
    eff = int(np.sum(evr > 1e-6))
    knee = kneedle(evr[:eff])

    # Participation ratio
    pr = float(var.sum() ** 2 / (var ** 2).sum())

    thresh = {}
    for t in (0.50, 0.80, 0.90, 0.95, 0.99):
        idx = int(np.searchsorted(cum, t) + 1)
        thresh[f"k_{int(t*100)}"] = idx

    out = {
        "n_components": int(len(s)),
        "n_effective_components": eff,
        "centroid_matrix_shape": list(Cc.shape),
        "knee_index": int(knee),
        "knee_evr": float(evr[knee]),
        "knee_cum": float(cum[knee]),
        "participation_ratio": pr,
        "cum_var_thresholds": thresh,
        "evr_top20": [float(v) for v in evr[:20]],
        "evr_full": [float(v) for v in evr],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[knee] idx={knee} EVR={evr[knee]:.4f} cum={cum[knee]:.3f}", flush=True)
    print(f"[PR ] effective rank = {pr:.2f}", flush=True)
    print(f"[cum] {thresh}", flush=True)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(1, len(evr) + 1)

    ax = axes[0]
    ax.semilogy(idx, evr, "-", color="#1f77b4", lw=1.2, label="EVR")
    ax.axvline(knee + 1, color="red", ls="--", lw=1.3,
               label=f"knee k={knee+1} (cum={cum[knee]:.2f})")
    ax.axvline(pr, color="orange", ls=":", lw=1.5,
               label=f"participation ratio = {pr:.1f}")
    ax.axvline(thresh["k_90"], color="green", ls="-.", lw=1, alpha=0.7,
               label=f"90% var @ k={thresh['k_90']}")
    ax.set_xlabel("component index")
    ax.set_ylabel("EVR (log)")
    ax.set_title(f"Centroid eigenvalue spectrum (n={n_colors}, D={D})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(idx, cum, "-", color="#1f77b4", lw=1.5)
    for t, color in [(0.50, "#aaaaaa"), (0.80, "#888888"),
                     (0.90, "green"), (0.95, "purple"), (0.99, "black")]:
        k = thresh[f"k_{int(t*100)}"]
        ax.axhline(t, color=color, ls=":", lw=0.8, alpha=0.6)
        ax.axvline(k, color=color, ls=":", lw=0.8, alpha=0.6)
        ax.plot([k], [t], "o", color=color, ms=5,
                label=f"{int(t*100)}% @ k={k}")
    ax.axvline(knee + 1, color="red", ls="--", lw=1.2, alpha=0.8, label=f"knee k={knee+1}")
    ax.set_xlabel("component index")
    ax.set_ylabel("cumulative EVR")
    ax.set_title("Cumulative variance + thresholds")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    fig.suptitle("auto_19 — Centroid eigenvalue spectrum + knee (idea q)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
