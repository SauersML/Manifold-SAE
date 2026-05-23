"""auto_21: cosine-similarity matrix of canonical color-name centroids (idea zz).

Question
--------
Pick a small set of canonical anchor color names (rainbow + neutrals).
Compute each one's per-color centroid in cogito L40 residual space
(mean across 28 templates), then compute the pairwise cosine-similarity
matrix. Is the angular pattern interpretable -- i.e., does cosine
similarity track perceptual closeness on the hue wheel, and do neutrals
(black/white/grey) form a distinct off-diagonal block?

Diagnostics
-----------
1. cosine-sim matrix (ordered around the hue wheel, neutrals last)
2. for chromatic anchors only: spearman(cosine-sim, -hue circular distance)
   to test whether closer hues -> higher cosine sim.
3. mean within-neutral cosine vs mean neutral<->chromatic cosine.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_21.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.stats import spearmanr

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_21.json"
OUT_PNG = OUT_DIR / "auto_21.png"
N_TEMPLATES = 28

# Hue-wheel order, then neutrals at the end.
CHROMATIC = [
    "red", "orange", "yellow", "lime", "green",
    "teal", "cyan", "turquoise", "blue", "navy",
    "purple", "magenta", "pink", "maroon", "brown", "olive",
]
NEUTRALS = ["black", "grey", "white"]
ANCHORS = CHROMATIC + NEUTRALS


def load_xkcd_rgb(n: int) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[0]
        hex_code = parts[1].lstrip("#")
        r = int(hex_code[0:2], 16) / 255.0
        g = int(hex_code[2:4], 16) / 255.0
        b = int(hex_code[4:6], 16) / 255.0
        names.append(name)
        rgb.append((r, g, b))
        if len(names) >= n:
            break
    return names, np.array(rgb, dtype=np.float64)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); df = mx - mn
    h = np.zeros_like(mx)
    safe = df > 1e-12
    rmax = safe & (mx == r); gmax = safe & (mx == g); bmax = safe & (mx == b)
    h[rmax] = ((g[rmax] - b[rmax]) / df[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / df[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / df[bmax]) + 4
    h = (h / 6.0) % 1.0
    s = np.where(mx > 0, df / np.where(mx > 0, mx, 1), 0.0)
    return np.stack([h, s, mx], axis=-1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float32)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    names, rgb = load_xkcd_rgb(n_colors)
    name_to_idx = {n: i for i, n in enumerate(names)}
    for a in ANCHORS:
        if a not in name_to_idx:
            raise SystemExit(f"missing anchor {a}")
    idx = np.array([name_to_idx[a] for a in ANCHORS], dtype=int)

    # Per-color centroids (mean across templates)
    centroids = X.reshape(n_colors, N_TEMPLATES, D).mean(axis=1).astype(np.float64)
    # De-mean over all colors so cosine sim measures direction in residual space.
    mu = centroids.mean(0, keepdims=True)
    C = centroids - mu
    A = C[idx]  # (k, D)
    norms = np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    An = A / norms
    S = An @ An.T  # cosine similarity, k x k

    # Anchor hue (for chromatic ones)
    anchor_rgb = rgb[idx]
    anchor_hsv = rgb_to_hsv(anchor_rgb)
    anchor_hue = anchor_hsv[:, 0]
    anchor_sat = anchor_hsv[:, 1]

    n_chrom = len(CHROMATIC)
    # Hue circular distance matrix for chromatic anchors
    hr = 2 * np.pi * anchor_hue[:n_chrom]
    HD = np.abs(((hr[:, None] - hr[None, :]) + np.pi) % (2 * np.pi) - np.pi)
    Schrom = S[:n_chrom, :n_chrom]
    iu = np.triu_indices(n_chrom, k=1)
    rho_hue_cos, p_hue_cos = spearmanr(HD[iu], Schrom[iu])
    print(f"[chrom] spearman(hue-dist, cos-sim) = {rho_hue_cos:+.3f}  (p={p_hue_cos:.2e})", flush=True)

    # Neutrals block analysis
    n_neut = len(NEUTRALS)
    Sneut = S[n_chrom:, n_chrom:]
    iu_n = np.triu_indices(n_neut, k=1)
    within_neut = float(Sneut[iu_n].mean()) if iu_n[0].size else float("nan")
    Scross = S[:n_chrom, n_chrom:]
    cross_mean = float(Scross.mean())
    within_chrom = float(Schrom[iu].mean())
    print(f"[blocks] within_chrom={within_chrom:+.3f}  within_neut={within_neut:+.3f}  cross={cross_mean:+.3f}", flush=True)

    # Average cosine to nearest non-self chromatic anchor: does it match hue-adjacent?
    # For each chromatic anchor, find the argmax of S over the other chromatic anchors;
    # compare to true hue-nearest neighbor.
    matches = 0
    for i in range(n_chrom):
        s_i = Schrom[i].copy(); s_i[i] = -np.inf
        nn_cos = int(np.argmax(s_i))
        d_i = HD[i].copy(); d_i[i] = np.inf
        nn_hue = int(np.argmin(d_i))
        matches += int(nn_cos == nn_hue)
    nn_agreement = matches / n_chrom
    print(f"[nn] cosine-NN == hue-NN agreement: {matches}/{n_chrom} = {nn_agreement:.2f}", flush=True)

    out = {
        "n_colors": int(n_colors),
        "n_templates": int(N_TEMPLATES),
        "anchors": ANCHORS,
        "chromatic": CHROMATIC,
        "neutrals": NEUTRALS,
        "cosine_sim": S.tolist(),
        "anchor_hue": anchor_hue.tolist(),
        "anchor_sat": anchor_sat.tolist(),
        "spearman_hue_dist_vs_cos_sim_chrom": float(rho_hue_cos),
        "spearman_p": float(p_hue_cos),
        "within_chrom_mean_cos": within_chrom,
        "within_neut_mean_cos": within_neut,
        "chrom_neut_cross_mean_cos": cross_mean,
        "cosineNN_eq_hueNN_agreement": nn_agreement,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- plot ----
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.35)

    # (a) cosine-sim matrix
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.max(np.abs(S - np.eye(len(ANCHORS)))))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ANCHORS))); ax.set_yticks(range(len(ANCHORS)))
    ax.set_xticklabels(ANCHORS, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(ANCHORS, fontsize=8)
    # block divider between chromatic and neutrals
    ax.axhline(n_chrom - 0.5, color="k", lw=0.8)
    ax.axvline(n_chrom - 0.5, color="k", lw=0.8)
    # color-swatch ticks
    for i, a in enumerate(ANCHORS):
        ax.add_patch(plt.Rectangle((-1.6, i - 0.4), 0.8, 0.8,
                                   transform=ax.transData, clip_on=False,
                                   facecolor=anchor_rgb[i], edgecolor="k", lw=0.4))
        ax.add_patch(plt.Rectangle((i - 0.4, -1.6), 0.8, 0.8,
                                   transform=ax.transData, clip_on=False,
                                   facecolor=anchor_rgb[i], edgecolor="k", lw=0.4))
    ax.set_title("cosine sim — residual-space color centroids")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    # (b) chromatic-only: cos-sim vs hue circular distance
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(HD[iu], Schrom[iu], s=22, c="black", alpha=0.7)
    ax.set_xlabel("hue circular distance (rad)")
    ax.set_ylabel("cosine similarity (residual)")
    ax.set_title(f"chromatic anchors: cos vs hue-dist\nspearman={rho_hue_cos:+.2f}  (p={p_hue_cos:.1e})")
    ax.grid(True, alpha=0.25)
    # least-squares trend line
    x = HD[iu]; y = Schrom[iu]
    if x.std() > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b * xs + a, "r-", lw=1.2, alpha=0.7, label=f"slope={b:+.2f}")
        ax.legend(loc="upper right", fontsize=8)

    # (c) block-mean bar chart + NN agreement
    ax = fig.add_subplot(gs[0, 2])
    bars = ["within\nchromatic", "within\nneutrals", "chromatic\n<-> neutral"]
    vals = [within_chrom, within_neut, cross_mean]
    cols = ["#3b6", "#888", "#c83"]
    ax.bar(bars, vals, color=cols)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"block-mean cosine\nhue-NN agreement = {matches}/{n_chrom}")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"auto_21 — cosine-sim matrix of {len(ANCHORS)} canonical anchor colors "
        f"(cogito L40 centroids, de-meaned over {n_colors} colors) — idea zz",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
