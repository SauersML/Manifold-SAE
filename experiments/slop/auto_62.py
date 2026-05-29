"""auto_62: Per-color cluster compactness in T-space (idea lllllll).

For each xkcd color, its 28 templates form a small point cloud in the
top-K PC ("T") space. We ask: which colors live in *tight* clusters
(strong color identity, template-invariant) and which are *smeared*
(template variation dominates)?

Three complementary tightness scalars per color:
  (1) Convex-hull volume of the 28 points after PCA-to-3D within that
      color's local frame (color-local 3D convex hull, scipy.QHull).
  (2) Mean pairwise distance in full K-dim T-space.
  (3) Mean 5-NN distance (k=5, excluding self) in full T-space.

Then:
  - Histogram each metric.
  - Scatter (1) vs (2) and (2) vs (3) -- consistency check.
  - Show the 12 tightest and 12 loosest colors as swatches with their
    metrics; the RGB swatches come from the run's color_axes arrays.
  - Color-of-each-marker = the color's actual RGB.

Pure numpy + scipy.ConvexHull + matplotlib. NO Gaussian RBF, no Duchon
length_scale, no fancy regressors. Uses PCA basis from results.json
(Vt_topK, mu, sigma), same basis as published R^2.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_62.{json,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull, QhullError

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG = RUN_DIR / "auto_62.png"
OUT_JSON = RUN_DIR / "auto_62.json"


def local_pca_hull_volume(P: np.ndarray) -> float:
    """Convex hull volume of P (n_pts x K) after PCA-to-3D in its local frame.

    Returns NaN if QHull cannot build a full-dim hull.
    """
    n = P.shape[0]
    if n < 4:
        return float("nan")
    Pc = P - P.mean(0, keepdims=True)
    # SVD: cheaper than full eig since n=28, K up to 64
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    k = min(3, S.size)
    Y = (U[:, :k] * S[:k])  # (n, k)
    if k < 3:
        return float("nan")
    try:
        hull = ConvexHull(Y)
        return float(hull.volume)
    except QhullError:
        return float("nan")


def mean_pairwise(P: np.ndarray) -> float:
    n = P.shape[0]
    sq = (P * P).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * (P @ P.T)
    d2 = np.maximum(d2, 0)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(d2[iu]).mean())


def mean_knn(P: np.ndarray, k: int = 5) -> float:
    n = P.shape[0]
    sq = (P * P).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * (P @ P.T)
    d2 = np.maximum(d2, 0)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    nn = np.take_along_axis(d2, idx, axis=1)
    return float(np.sqrt(nn).mean())


def swatch_strip(ax, rgbs, labels, vals, title):
    ax.set_xlim(0, len(rgbs))
    ax.set_ylim(0, 1)
    for i, (rgb, lab, v) in enumerate(zip(rgbs, labels, vals)):
        ax.add_patch(Rectangle((i, 0.3), 1, 0.7, color=tuple(rgb)))
        ax.text(i + 0.5, 0.22, lab[:14], ha="center", va="top",
                fontsize=6.5, rotation=40)
        ax.text(i + 0.5, 0.08, f"{v:.2g}", ha="center", va="top",
                fontsize=6.5, color="black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {RESULTS}")
    res = json.loads(RESULTS.read_text())
    pl = res["per_layer"]["L40"]
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float32)
    mu = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)
    K, D = Vt.shape

    n_t = len(res["templates"])
    R = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    G = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    B = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.stack([R, G, B], axis=1)
    n_c = rgb.shape[0]
    N = n_c * n_t
    print(f"[layout] n_colors={n_c} n_templates={n_t} K={K} D={D}")

    # Try to recover color names if present anywhere; otherwise fall back to idx
    names = None
    for key in ("color_names", "colors", "names"):
        if key in res and len(res[key]) == n_c:
            names = list(res[key])
            break
    if names is None:
        try:
            xtxt = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt").read_text().splitlines()
            # file is often "<name>\t#hex"; extract first column
            cand = []
            for ln in xtxt:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                cand.append(ln.split("\t")[0].strip())
            if len(cand) >= n_c:
                names = cand[:n_c]
        except Exception:
            pass
    if names is None:
        names = [f"c{i:04d}" for i in range(n_c)]

    # Project X to Z = top-K PCs
    print(f"[load] X {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, (X.shape, N)
    Z = np.zeros((N, K), dtype=np.float32)
    chunk = 4096
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Xc = (Xc - mu) / sigma
        Z[i:j] = Xc @ Vt.T
    Z = Z.reshape(n_c, n_t, K)

    # Global per-PC std for normalization (use centered-by-color points)
    Zc = Z - Z.mean(1, keepdims=True)
    global_scale = float(np.linalg.norm(Zc.reshape(-1, K).std(0)))
    print(f"[scale] global ||sigma_Z_centered|| = {global_scale:.3f}")

    print("[compute] per-color metrics ...")
    hull_v  = np.full(n_c, np.nan)
    pair_d  = np.zeros(n_c)
    knn5_d  = np.zeros(n_c)
    for ci in range(n_c):
        P = Z[ci].astype(np.float64)
        hull_v[ci] = local_pca_hull_volume(P)
        pair_d[ci] = mean_pairwise(P)
        knn5_d[ci] = mean_knn(P, k=5)

    # Normalize by global scale where appropriate (distances)
    pair_n = pair_d / max(global_scale, 1e-9)
    knn_n  = knn5_d / max(global_scale, 1e-9)
    # cubic root of hull volume to get a length unit, then normalize
    hull_len = np.where(np.isfinite(hull_v), np.cbrt(np.maximum(hull_v, 0)), np.nan)
    hull_n   = hull_len / max(global_scale, 1e-9)

    # Rank by mean-pairwise (most stable scalar)
    order = np.argsort(pair_n)
    tight_idx = order[:12]
    loose_idx = order[-12:][::-1]

    summary = {
        "config": {
            "n_colors": int(n_c), "n_templates": int(n_t), "K": int(K),
            "knn_k": 5, "hull_dim": 3,
        },
        "global_centered_scale": global_scale,
        "median_pair_dist_norm": float(np.median(pair_n)),
        "median_knn5_norm": float(np.median(knn_n)),
        "median_hull_len_norm": float(np.nanmedian(hull_n)),
        "frac_hull_degenerate": float(np.isnan(hull_v).mean()),
        "spearman_pair_vs_knn5": float(
            np.corrcoef(np.argsort(np.argsort(pair_n)),
                        np.argsort(np.argsort(knn_n)))[0, 1]
        ),
        "spearman_pair_vs_hull": float(
            np.corrcoef(
                np.argsort(np.argsort(pair_n[np.isfinite(hull_n)])),
                np.argsort(np.argsort(hull_n[np.isfinite(hull_n)])),
            )[0, 1]
        ),
        "tightest_12": [
            {"idx": int(i), "name": names[i], "rgb": rgb[i].tolist(),
             "pair_norm": float(pair_n[i]), "knn5_norm": float(knn_n[i]),
             "hull_len_norm": float(hull_n[i])}
            for i in tight_idx
        ],
        "loosest_12": [
            {"idx": int(i), "name": names[i], "rgb": rgb[i].tolist(),
             "pair_norm": float(pair_n[i]), "knn5_norm": float(knn_n[i]),
             "hull_len_norm": float(hull_n[i])}
            for i in loose_idx
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {OUT_JSON}")
    print("median pair_n:", summary["median_pair_dist_norm"],
          "median knn5_n:", summary["median_knn5_norm"],
          "median hull_len_n:", summary["median_hull_len_norm"])
    print("rank corr pair-vs-knn:", summary["spearman_pair_vs_knn5"],
          "pair-vs-hull:", summary["spearman_pair_vs_hull"])

    # --- Plot ---
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95],
                          hspace=0.55, wspace=0.30)

    # (a) hist mean pairwise
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(pair_n, bins=40, color="#4060c0", edgecolor="black", lw=0.3)
    ax.axvline(np.median(pair_n), ls="--", color="black", lw=1,
               label=f"median={np.median(pair_n):.2f}")
    ax.set_xlabel(r"mean pairwise dist  /  $\|\sigma_Z^{centered}\|$")
    ax.set_ylabel("# colors")
    ax.set_title("per-color mean pairwise spread")
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.4)

    # (b) hist 5-NN
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(knn_n, bins=40, color="#c06040", edgecolor="black", lw=0.3)
    ax.axvline(np.median(knn_n), ls="--", color="black", lw=1,
               label=f"median={np.median(knn_n):.2f}")
    ax.set_xlabel(r"mean 5-NN dist  /  $\|\sigma_Z^{centered}\|$")
    ax.set_ylabel("# colors")
    ax.set_title("per-color local (5-NN) tightness")
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.4)

    # (c) hist hull length (cube-rooted volume)
    ax = fig.add_subplot(gs[0, 2])
    finite = np.isfinite(hull_n)
    ax.hist(hull_n[finite], bins=40, color="#60a060", edgecolor="black", lw=0.3)
    ax.axvline(np.nanmedian(hull_n), ls="--", color="black", lw=1,
               label=f"median={np.nanmedian(hull_n):.2f}")
    ax.set_xlabel(r"local-3D hull $V^{1/3}$  /  $\|\sigma_Z^{centered}\|$")
    ax.set_ylabel("# colors")
    ax.set_title(f"local-PCA(3D) convex-hull length\n(deg={int((~finite).sum())} of {n_c})")
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.4)

    # (d) scatter pair vs knn, colored by actual color RGB
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(pair_n, knn_n, c=np.clip(rgb, 0, 1), s=14, alpha=0.85,
               edgecolors="black", linewidths=0.2)
    rho1 = summary["spearman_pair_vs_knn5"]
    ax.set_xlabel("mean-pairwise (norm)"); ax.set_ylabel("mean 5-NN (norm)")
    ax.set_title(f"pair vs 5-NN  (rank corr = {rho1:.3f})")
    ax.grid(ls=":", alpha=0.4)

    # (e) scatter pair vs hull
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(pair_n[finite], hull_n[finite], c=np.clip(rgb[finite], 0, 1),
               s=14, alpha=0.85, edgecolors="black", linewidths=0.2)
    rho2 = summary["spearman_pair_vs_hull"]
    ax.set_xlabel("mean-pairwise (norm)")
    ax.set_ylabel(r"hull $V^{1/3}$ (norm)")
    ax.set_title(f"pair vs hull-len  (rank corr = {rho2:.3f})")
    ax.grid(ls=":", alpha=0.4)

    # (f) HSV.S vs pair_n  (does low-saturation -> looser cluster?)
    import colorsys
    hsv = np.stack([np.array(colorsys.rgb_to_hsv(*np.clip(c, 0, 1))) for c in rgb])
    ax = fig.add_subplot(gs[1, 2])
    sc = ax.scatter(hsv[:, 1], pair_n, c=hsv[:, 2], cmap="cividis",
                    s=12, alpha=0.85)
    ax.set_xlabel("HSV saturation"); ax.set_ylabel("mean-pairwise (norm)")
    ax.set_title("template spread vs HSV saturation")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85); cbar.set_label("HSV value")
    ax.grid(ls=":", alpha=0.4)

    # (g) tightest 12 swatches
    ax = fig.add_subplot(gs[2, :])
    # split row in half: tight on left, loose on right
    ax.axis("off")
    # use two inner axes
    bb = ax.get_position()
    w = bb.width / 2 - 0.01
    ax_t = fig.add_axes([bb.x0, bb.y0, w, bb.height])
    ax_l = fig.add_axes([bb.x0 + w + 0.02, bb.y0, w, bb.height])
    swatch_strip(
        ax_t,
        [np.clip(rgb[i], 0, 1) for i in tight_idx],
        [names[i] for i in tight_idx],
        [pair_n[i] for i in tight_idx],
        "TIGHTEST 12 (most template-invariant)  -- value = pair-dist / sigma",
    )
    swatch_strip(
        ax_l,
        [np.clip(rgb[i], 0, 1) for i in loose_idx],
        [names[i] for i in loose_idx],
        [pair_n[i] for i in loose_idx],
        "LOOSEST 12 (most template-smeared)     -- value = pair-dist / sigma",
    )

    fig.suptitle(
        "auto_62  -- per-color k-NN / convex-hull / pairwise compactness in T-space\n"
        f"K={K} PCs, n_templates={n_t}, n_colors={n_c}; rank corr pair-vs-knn={rho1:.2f}, pair-vs-hull={rho2:.2f}",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[done] -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
