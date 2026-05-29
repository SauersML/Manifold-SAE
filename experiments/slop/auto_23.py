"""auto_23: per-color k-NN hue purity (idea nn).

For each xkcd color, find its k nearest neighbors in the cogito L40
per-color centroid PCA space (the same standardize→center→Vt pipeline
the gam fit uses). Ask: are those neighbors hue-coherent? Compare
against RGB-space k-NN (an upper bound on what a literal color encoder
could achieve) and against a random-permutation baseline.

Hue purity for a color c with neighbors N_k(c) is defined as the mean
HSV-hue circular cosine similarity between c and each neighbor (range
[-1, +1], +1 = identical hue). Achromatic colors (saturation < 0.1) are
masked out because hue is undefined.

Outputs
-------
- Top panel: histogram of per-color hue purity for residual-PCA kNN
  vs RGB kNN vs shuffled baseline.
- Middle panel: hue purity vs HSV saturation (binned).
- Bottom panel: swatches of the 6 best and 6 worst colors (by
  residual-kNN hue purity, among chromatic colors), each shown with
  its 5 nearest residual-space neighbors.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_23.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path("/Users/user/Manifold-SAE")
RESULTS = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
HARVEST = ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"
XKCD = ROOT / "experiments/xkcd_colors.txt"
OUT_PNG = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_23.png"
OUT_JSON = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_23.json"

N_COLORS = 949
N_TEMPLATES = 28
K = 10  # neighbours per color


def load_xkcd_rgb(n: int) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        h = parts[1].lstrip("#")
        names.append(parts[0])
        rgb.append((int(h[0:2], 16) / 255., int(h[2:4], 16) / 255., int(h[4:6], 16) / 255.))
        if len(names) >= n:
            break
    return names, np.array(rgb, dtype=np.float64)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    # squared euclidean
    g = X @ X.T
    sq = np.diag(g)
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, kth=k, axis=1)[:, :k]


def hue_circular_sim(h_self: float, h_neighbors: np.ndarray) -> np.ndarray:
    # hue in [0, 1) interpreted as angle. cos(2π·Δh) ∈ [-1, 1].
    return np.cos(2.0 * np.pi * (h_neighbors - h_self))


def per_color_purity(neighbors: np.ndarray, hsv: np.ndarray, chromatic_mask: np.ndarray) -> np.ndarray:
    purity = np.full(neighbors.shape[0], np.nan)
    for i in range(neighbors.shape[0]):
        if not chromatic_mask[i]:
            continue
        nb = neighbors[i]
        nb = nb[chromatic_mask[nb]]
        if nb.size == 0:
            continue
        sims = hue_circular_sim(hsv[i, 0], hsv[nb, 0])
        purity[i] = float(np.mean(sims))
    return purity


def main() -> None:
    d = json.loads(RESULTS.read_text())
    pl = d["per_layer"]["L40"]
    mu = np.array(pl["mu"], dtype=np.float32)
    sigma = np.array(pl["sigma"], dtype=np.float32)
    Vt = np.array(pl["Vt_topK"], dtype=np.float32)
    Kpc, D = Vt.shape
    print(f"[load] D={D} K={Kpc}")

    X = np.load(HARVEST, mmap_mode="r")
    assert X.shape[0] == N_COLORS * N_TEMPLATES

    centroids = np.zeros((N_COLORS, D), dtype=np.float32)
    for ci in range(N_COLORS):
        s = ci * N_TEMPLATES
        centroids[ci] = X[s : s + N_TEMPLATES].mean(0)

    Xn = (centroids - mu) / sigma
    Xn_c = Xn - Xn.mean(0, keepdims=True)
    Z = Xn_c @ Vt.T  # (n_c, K)

    names, rgb = load_xkcd_rgb(N_COLORS)
    hsv = rgb_to_hsv(rgb)
    chromatic = hsv[:, 1] >= 0.1  # saturation ≥ 0.1
    print(f"[chromatic] {int(chromatic.sum())} / {N_COLORS}")

    nn_resid = knn_indices(Z.astype(np.float64), K)
    nn_rgb = knn_indices(rgb, K)

    purity_resid = per_color_purity(nn_resid, hsv, chromatic)
    purity_rgb = per_color_purity(nn_rgb, hsv, chromatic)

    # shuffled baseline: random K neighbors (no self)
    rng = np.random.default_rng(0)
    nn_shuf = np.zeros((N_COLORS, K), dtype=np.int64)
    for i in range(N_COLORS):
        choices = rng.choice(N_COLORS - 1, size=K, replace=False)
        choices = np.where(choices >= i, choices + 1, choices)
        nn_shuf[i] = choices
    purity_shuf = per_color_purity(nn_shuf, hsv, chromatic)

    p_resid = purity_resid[~np.isnan(purity_resid)]
    p_rgb = purity_rgb[~np.isnan(purity_rgb)]
    p_shuf = purity_shuf[~np.isnan(purity_shuf)]

    summary = {
        "k": K,
        "n_chromatic": int(chromatic.sum()),
        "purity_resid_mean": float(p_resid.mean()),
        "purity_resid_median": float(np.median(p_resid)),
        "purity_rgb_mean": float(p_rgb.mean()),
        "purity_rgb_median": float(np.median(p_rgb)),
        "purity_shuf_mean": float(p_shuf.mean()),
        "purity_shuf_median": float(np.median(p_shuf)),
        "frac_resid_above_0p5": float((p_resid > 0.5).mean()),
        "frac_rgb_above_0p5": float((p_rgb > 0.5).mean()),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---------------- plot ----------------
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    bins = np.linspace(-1, 1, 41)
    ax0.hist(p_resid, bins=bins, alpha=0.55, color="#1f77b4",
             label=f"residual-PCA kNN  (mean={p_resid.mean():+.3f})")
    ax0.hist(p_rgb, bins=bins, alpha=0.55, color="#2ca02c",
             label=f"RGB kNN  (mean={p_rgb.mean():+.3f})")
    ax0.hist(p_shuf, bins=bins, alpha=0.45, color="#888888",
             label=f"shuffled  (mean={p_shuf.mean():+.3f})")
    ax0.axvline(0, color="k", lw=0.5)
    ax0.set_xlabel("hue circular cosine purity of k=10 neighbours")
    ax0.set_ylabel("# colors")
    ax0.set_title(f"per-color k-NN hue purity, k={K}, chromatic only "
                  f"(n={int(chromatic.sum())} of {N_COLORS})")
    ax0.legend(loc="upper left")
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(gs[1])
    sat_vals = hsv[chromatic, 1]
    sat_bins = np.linspace(0.1, 1.0, 10)
    bcent = 0.5 * (sat_bins[:-1] + sat_bins[1:])
    def bin_mean(p):
        out, ns = [], []
        pc = p[chromatic]
        for lo, hi in zip(sat_bins[:-1], sat_bins[1:]):
            m = (sat_vals >= lo) & (sat_vals < hi)
            if m.sum() == 0:
                out.append(np.nan); ns.append(0); continue
            out.append(np.nanmean(pc[m])); ns.append(int(m.sum()))
        return np.array(out), ns
    mr, nr = bin_mean(purity_resid)
    mg, _ = bin_mean(purity_rgb)
    ms, _ = bin_mean(purity_shuf)
    ax1.plot(bcent, mr, "o-", color="#1f77b4", label="residual-PCA kNN")
    ax1.plot(bcent, mg, "s-", color="#2ca02c", label="RGB kNN")
    ax1.plot(bcent, ms, "x--", color="#888888", label="shuffled")
    for x, y, n in zip(bcent, mr, nr):
        ax1.annotate(f"n={n}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color="#1f77b4")
    ax1.set_xlabel("HSV saturation (binned)")
    ax1.set_ylabel("mean hue purity")
    ax1.set_title("hue purity vs saturation — does the manifold help more on vivid colors?")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)

    # swatch panel
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("residual-kNN exemplars — top-6 best (left) and bottom-6 worst (right) chromatic colors")
    chrom_idx = np.where(chromatic)[0]
    order = chrom_idx[np.argsort(-purity_resid[chrom_idx])]
    best = order[:6]
    worst = order[-6:][::-1]
    rows = list(best) + list(worst)
    n_rows = len(rows)
    cols = K + 1
    ax2.set_xlim(0, cols)
    ax2.set_ylim(0, n_rows)
    ax2.invert_yaxis()
    for ri, ci in enumerate(rows):
        # anchor swatch
        ax2.add_patch(plt.Rectangle((0, ri), 1, 1, color=tuple(rgb[ci]),
                                    ec="k", lw=0.8))
        label = f"{names[ci][:18]}\nρ={purity_resid[ci]:+.2f}"
        ax2.text(0.5, ri + 0.5, label, ha="center", va="center",
                 fontsize=7,
                 color="white" if (0.299 * rgb[ci, 0] + 0.587 * rgb[ci, 1] + 0.114 * rgb[ci, 2]) < 0.5 else "black")
        for j, nb in enumerate(nn_resid[ci]):
            ax2.add_patch(plt.Rectangle((j + 1, ri), 1, 1, color=tuple(rgb[nb]),
                                        ec="0.5", lw=0.4))
        # separator between best (0..5) and worst (6..11)
        if ri == 5:
            ax2.axhline(ri + 1, color="k", lw=1.2)
    ax2.text(0, -0.15, "anchor", fontsize=8)
    ax2.text(1, -0.15, "← 10 residual-PCA nearest →", fontsize=8)

    fig.suptitle("auto_23 — per-color k-NN hue purity (idea nn)  "
                 "cogito L40, 949 xkcd colors", fontsize=13)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "summary": summary,
        "top10_best_chromatic": [
            {"name": names[i], "purity_resid": float(purity_resid[i]),
             "purity_rgb": float(purity_rgb[i])}
            for i in chrom_idx[np.argsort(-purity_resid[chrom_idx])[:10]]
        ],
        "top10_worst_chromatic": [
            {"name": names[i], "purity_resid": float(purity_resid[i]),
             "purity_rgb": float(purity_rgb[i])}
            for i in chrom_idx[np.argsort(purity_resid[chrom_idx])[:10]]
        ],
    }, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
