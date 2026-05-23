"""auto_exp_07: local intrinsic dimensionality of T_3d via local-PCA.

auto_exp_06 asked "what global d explains held-out variance best?" and
found a single elbow.  But the color manifold may have *varying* local
dimensionality — a near-1d hue circle in one region, a 2d L*-C* sheet
elsewhere, plus dark/grey collapse points.  This experiment estimates
local intrinsic dim around each centroid by local-PCA on its k-NN in
T_3d (the fitted U_3d latent), reports the per-color participation-ratio
and 90%-variance threshold dim, and maps the result back to hue.

Method (cheap, no server, no harvest):
  1. Load cached residuals + reproduce 954x64 centroid PCA target.
  2. Fit one U_3d (full data, no CV) using cmg.fit_unsupervised_manifold.
  3. For each centroid i, take its k-NN in T (k in {12, 24}) and run
     PCA on the neighborhood in the original 64-PC target space.  Two
     intrinsic-dim estimators per neighborhood:
       (a) participation ratio  PR = (sum eig)^2 / sum(eig^2)
       (b) d90 = #PCs needed to reach 90% variance
  4. Aggregate: histogram of local dims, hue-vs-localdim scatter
     (uses XKCD rgb file already shipped in experiments/).
  5. Save JSON summary + 2x2 PNG.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_07_local_dim.json"
OUT_PNG = OUT_DIR / "auto_exp_07_local_dim.png"

N_TEMPLATES = 28
N_PCS = 64
D_LATENT = 3
K_NEIGHBORS = [12, 24]
VAR_THRESH = 0.90


def _load_xkcd_hues(n_colors_expected: int) -> np.ndarray | None:
    """Best-effort: parse xkcd_colors.txt -> hue in [0,1).  Returns NaN array
    if file is missing/mismatched (analysis just skips hue scatter then)."""
    if not XKCD.exists():
        return None
    hexes: list[str] = []
    for line in XKCD.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # tolerate "name\t#rrggbb" or just "#rrggbb"
        parts = line.split()
        hx = next((p for p in parts if p.startswith("#") and len(p) == 7), None)
        if hx is not None:
            hexes.append(hx)
    if len(hexes) != n_colors_expected:
        print(f"[hue] xkcd parse got {len(hexes)} != {n_colors_expected}; "
              "skipping hue map", flush=True)
        return None
    import colorsys
    hues = np.zeros(n_colors_expected, dtype=np.float64)
    for i, hx in enumerate(hexes):
        r, g, b = (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))
        h, _, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        hues[i] = h
    return hues


def _local_pca_dims(Z: np.ndarray, T: np.ndarray, k: int) -> dict:
    """For each row i: find k nearest neighbors in T, run PCA on the
    corresponding rows of Z (mean-centered), return PR and d90."""
    n = T.shape[0]
    # pairwise dists in latent T (small: 954x954 x 3 floats, ~22MB peak fine)
    diff = T[:, None, :] - T[None, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]

    pr = np.zeros(n)
    d90 = np.zeros(n, dtype=np.int32)
    eig_all = []
    for i in range(n):
        nbrs = Z[nn_idx[i]]
        nbrs = nbrs - nbrs.mean(0, keepdims=True)
        # SVD on (k, 64); sing values^2 ~ eigenvalues of cov
        s = np.linalg.svd(nbrs, compute_uv=False)
        eig = s ** 2
        eig = eig[eig > 1e-12]
        total = eig.sum()
        if total <= 0:
            pr[i] = np.nan
            d90[i] = 0
            continue
        pr[i] = float((total ** 2) / (eig ** 2).sum())
        cum = np.cumsum(eig) / total
        d90[i] = int(np.searchsorted(cum, VAR_THRESH) + 1)
        eig_all.append(eig[:8] / total)  # top-8 ev ratios for diagnostics
    return {
        "k": k,
        "pr": pr,
        "d90": d90,
        "pr_mean": float(np.nanmean(pr)),
        "pr_median": float(np.nanmedian(pr)),
        "d90_mean": float(d90.mean()),
        "d90_median": float(np.median(d90)),
        "eig_top8_mean": np.mean(np.stack(eig_all, 0), axis=0).tolist()
            if eig_all else None,
        "nn_idx": nn_idx,  # not serialized
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # ---- 954x64 PCA target (matches auto_exp_04/05/06) ----
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z = centroids_n @ V_topK.T          # (954, 64)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # ---- Fit U_3d on full data (no CV, single fit) ----
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=1,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))
    print("[fit] U_3d on all 954 centroids ...", flush=True)
    t0 = time.time()
    fit = cmg.fit_unsupervised_manifold(Z, d=D_LATENT, cfg=cfg,
                                         n_iters=15, verbose=True)
    T = np.asarray(fit["T"])             # (954, 3)
    print(f"[fit] T={T.shape}  centers={fit['centers'].shape}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---- Local-PCA per centroid ----
    results = {}
    for k in K_NEIGHBORS:
        print(f"[localPCA] k={k} ...", flush=True)
        r = _local_pca_dims(Z, T, k)
        print(f"  k={k}  PR median={r['pr_median']:.2f}  "
              f"d90 median={r['d90_median']:.1f}", flush=True)
        results[k] = r

    # ---- Hue map (optional) ----
    hues = _load_xkcd_hues(n_colors)

    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors": int(n_colors), "n_templates": N_TEMPLATES,
            "n_pcs": N_PCS, "d_latent": D_LATENT,
            "k_neighbors": K_NEIGHBORS, "var_thresh": VAR_THRESH,
        },
        "evr_topK_sum": float(evr.sum()),
        "results": {
            str(k): {
                "pr_mean": results[k]["pr_mean"],
                "pr_median": results[k]["pr_median"],
                "d90_mean": results[k]["d90_mean"],
                "d90_median": results[k]["d90_median"],
                "eig_top8_mean": results[k]["eig_top8_mean"],
                "pr_per_color": results[k]["pr"].tolist(),
                "d90_per_color": results[k]["d90"].tolist(),
            } for k in K_NEIGHBORS
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # ---- Plot 2x2 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) PR histograms
    ax = axes[0, 0]
    for k in K_NEIGHBORS:
        pr = results[k]["pr"]
        ax.hist(pr[np.isfinite(pr)], bins=40, alpha=0.55,
                label=f"k={k}  med={np.nanmedian(pr):.2f}")
    ax.set_xlabel("participation ratio (local intrinsic dim)")
    ax.set_ylabel("# colors")
    ax.set_title("Local PR histogram across 954 centroids")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    # (0,1) d90 histograms
    ax = axes[0, 1]
    max_d = max(results[k]["d90"].max() for k in K_NEIGHBORS)
    bins = np.arange(1, max_d + 2) - 0.5
    for k in K_NEIGHBORS:
        ax.hist(results[k]["d90"], bins=bins, alpha=0.55,
                label=f"k={k}  med={results[k]['d90_median']:.0f}")
    ax.set_xlabel(f"d90  (#PCs to reach {VAR_THRESH*100:.0f}% local var)")
    ax.set_ylabel("# colors")
    ax.set_title("Local d90 histogram")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    # (1,0) Hue vs PR (or scatter PR-vs-d90 if no hues)
    ax = axes[1, 0]
    k_show = K_NEIGHBORS[-1]
    pr = results[k_show]["pr"]
    d90 = results[k_show]["d90"]
    if hues is not None:
        ax.scatter(hues, pr, c=hues, cmap="hsv", s=10, alpha=0.7)
        ax.set_xlabel("hue (xkcd RGB -> HSV)")
        ax.set_ylabel(f"local PR (k={k_show})")
        ax.set_title("Local intrinsic dim vs hue")
    else:
        ax.scatter(d90, pr, s=10, alpha=0.5)
        ax.set_xlabel(f"d90 (k={k_show})")
        ax.set_ylabel(f"PR (k={k_show})")
        ax.set_title("PR vs d90  (hue map unavailable)")
    ax.grid(linestyle=":", alpha=0.4)

    # (1,1) Mean top-8 eigenvalue spectrum (k=24)
    ax = axes[1, 1]
    for k in K_NEIGHBORS:
        eg = results[k]["eig_top8_mean"]
        if eg is None: continue
        ax.plot(np.arange(1, len(eg) + 1), eg, marker="o",
                label=f"k={k}")
    ax.axhline(0.1, color="grey", linestyle=":", lw=0.8)
    ax.set_xlabel("local PC index")
    ax.set_ylabel("mean variance ratio")
    ax.set_title("Mean local eigenvalue spectrum  (top-8)")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle("auto_exp_07 — local intrinsic dim on U_3d (cogito L40)",
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
