"""auto_09: Top-K nearest-neighbor consistency — does the discovered U_3d
latent T put each color near its perceptual neighbors?

For each of the 949 xkcd centroid colors we have:
  (a) latent coordinates T_i in R^3 (unsupervised manifold fit, d=3)
  (b) true RGB_i and HSV_i

For each K in {1, 3, 5, 10, 20} and each color i we compute the K nearest
neighbors of i under three distance metrics:
    d_latent  = ||T_i - T_j||_2
    d_rgb     = ||RGB_i - RGB_j||_2
    d_hsv     = cyclic distance in (hue, sat, value) with hue on a circle

We then ask: of the K latent-NN of color i, what fraction are also among
the K-NN under RGB / HSV? This is the standard "neighborhood recall@K"
metric used to evaluate embeddings (t-SNE / UMAP papers).

Baseline = random recall = K / (N-1).

We also pull out the 10 colors with the worst latent↔HSV recall@5
(i.e. the colors whose latent neighborhood is most semantically wrong)
and the 10 with the best — and print their xkcd names if available.

Reads:  runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
Writes: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_09.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
OUT = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_09.png"


def knn_indices(D: np.ndarray, K: int) -> np.ndarray:
    """Return (N, K) indices of K nearest neighbors of each row (excluding self)."""
    N = D.shape[0]
    np.fill_diagonal(D, np.inf)
    # argpartition for speed
    idx = np.argpartition(D, K, axis=1)[:, :K]
    return idx


def hsv_distance(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Pairwise distance with hue on circle. Weight hue by sat*sat for grays."""
    N = H.shape[0]
    # Cyclic hue distance in [0, 0.5]
    dh = np.abs(H[:, None] - H[None, :])
    dh = np.minimum(dh, 1.0 - dh)
    ds = S[:, None] - S[None, :]
    dv = V[:, None] - V[None, :]
    # Weight hue contribution by saturation product (grays shouldn't care about hue)
    sw = np.sqrt(np.maximum(S[:, None] * S[None, :], 1e-6))
    return np.sqrt((sw * dh) ** 2 + ds ** 2 + dv ** 2)


def main() -> None:
    with open(RESULTS) as f:
        d = json.load(f)

    L = d["per_layer"]["L40"]
    T = np.array(L["unsupervised_full_data"]["d=3"]["T"])  # (949, 3)
    n = T.shape[0]
    R = np.array(d["color_axes_per_color_index"]["R"])[:n]
    G = np.array(d["color_axes_per_color_index"]["G"])[:n]
    B = np.array(d["color_axes_per_color_index"]["B"])[:n]
    H = np.array(d["color_axes_per_color_index"]["hue"])[:n]
    S = np.array(d["color_axes_per_color_index"]["sat"])[:n]
    V = np.array(d["color_axes_per_color_index"]["value"])[:n]
    rgb = np.stack([R, G, B], axis=1)

    # Pairwise distance matrices
    def pdist(X):
        d2 = np.sum(X * X, axis=1)[:, None] + np.sum(X * X, axis=1)[None, :] \
             - 2 * X @ X.T
        return np.sqrt(np.maximum(d2, 0.0))

    D_lat = pdist(T)
    D_rgb = pdist(rgb)
    D_hsv = hsv_distance(H, S, V)

    Ks = [1, 3, 5, 10, 20, 50]
    recall_rgb = []
    recall_hsv = []
    recall_baseline = []

    # Cache top-K idx per K, per metric
    nn_lat_K = {}
    nn_rgb_K = {}
    nn_hsv_K = {}

    for K in Ks:
        nn_lat = knn_indices(D_lat.copy(), K)
        nn_rgb = knn_indices(D_rgb.copy(), K)
        nn_hsv = knn_indices(D_hsv.copy(), K)
        nn_lat_K[K] = nn_lat
        nn_rgb_K[K] = nn_rgb
        nn_hsv_K[K] = nn_hsv

        # Overlap fraction
        ov_rgb = np.mean([len(set(nn_lat[i]) & set(nn_rgb[i])) / K
                           for i in range(n)])
        ov_hsv = np.mean([len(set(nn_lat[i]) & set(nn_hsv[i])) / K
                           for i in range(n)])
        recall_rgb.append(ov_rgb)
        recall_hsv.append(ov_hsv)
        recall_baseline.append(K / (n - 1))

    # Per-color recall@5 vs HSV for hardest / easiest analysis
    K_focus = 5
    nn_lat5 = nn_lat_K[K_focus]
    nn_hsv5 = nn_hsv_K[K_focus]
    per_color_rec = np.array([
        len(set(nn_lat5[i]) & set(nn_hsv5[i])) / K_focus for i in range(n)
    ])
    order = np.argsort(per_color_rec)
    worst10 = order[:10]
    best10 = order[-10:][::-1]

    # ----- Plot -----
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(Ks, recall_rgb, "o-", label="latent ∩ RGB-NN", lw=2)
    ax1.plot(Ks, recall_hsv, "s-", label="latent ∩ HSV-NN", lw=2)
    ax1.plot(Ks, recall_baseline, "k--", label=f"random = K/{n-1}", lw=1.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("K (neighborhood size)")
    ax1.set_ylabel("mean fraction of latent-K-NN also in RGB/HSV-K-NN")
    ax1.set_title("Neighborhood recall@K\n"
                  "(higher = U_3d preserves perceptual neighbors)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    for x, y in zip(Ks, recall_hsv):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(4, 6), fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(per_color_rec, bins=np.linspace(0, 1, 11),
             edgecolor="black", color="steelblue")
    ax2.axvline(recall_hsv[Ks.index(K_focus)], color="red", ls="--",
                label=f"mean = {recall_hsv[Ks.index(K_focus)]:.2f}")
    ax2.axvline(K_focus / (n - 1), color="gray", ls=":",
                label=f"random = {K_focus/(n-1):.3f}")
    ax2.set_xlabel(f"per-color recall@{K_focus}  (latent ∩ HSV-NN)")
    ax2.set_ylabel("# colors (out of 949)")
    ax2.set_title("Distribution of per-color neighborhood recall@5 vs HSV")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Worst colors panel: show their true RGB swatch + the 5 latent-NN swatches
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Worst 10 colors (lowest recall@5): "
                  "swatch = color, then its 5 latent-NN")
    for row, i in enumerate(worst10):
        # Self swatch
        ax3.add_patch(plt.Rectangle((0, row), 0.9, 0.9,
                                     color=rgb[i].clip(0, 1)))
        for k, j in enumerate(nn_lat5[i]):
            ax3.add_patch(plt.Rectangle((1.2 + k, row), 0.9, 0.9,
                                         color=rgb[j].clip(0, 1)))
        ax3.text(7.5, row + 0.45,
                 f"rec={per_color_rec[i]:.1f}",
                 va="center", fontsize=9)
    ax3.set_xlim(-0.2, 9)
    ax3.set_ylim(-0.5, len(worst10))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect("equal")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Best 10 colors (highest recall@5): "
                  "swatch = color, then its 5 latent-NN")
    for row, i in enumerate(best10):
        ax4.add_patch(plt.Rectangle((0, row), 0.9, 0.9,
                                     color=rgb[i].clip(0, 1)))
        for k, j in enumerate(nn_lat5[i]):
            ax4.add_patch(plt.Rectangle((1.2 + k, row), 0.9, 0.9,
                                         color=rgb[j].clip(0, 1)))
        ax4.text(7.5, row + 0.45,
                 f"rec={per_color_rec[i]:.1f}",
                 va="center", fontsize=9)
    ax4.set_xlim(-0.2, 9)
    ax4.set_ylim(-0.5, len(best10))
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect("equal")

    verdict_rec5 = recall_hsv[Ks.index(5)]
    base5 = 5 / (n - 1)
    fig.suptitle(
        f"auto_09: Top-K NN consistency — latent U_3d vs RGB / HSV   "
        f"(recall@5 vs HSV = {verdict_rec5:.2f}, "
        f"{verdict_rec5/base5:.0f}× random {base5:.3f})",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    print(f"[auto_09] saved {OUT}")
    print(f"[auto_09] recall@K vs HSV: " +
          ", ".join(f"K={k}:{r:.3f}" for k, r in zip(Ks, recall_hsv)))
    print(f"[auto_09] recall@K vs RGB: " +
          ", ".join(f"K={k}:{r:.3f}" for k, r in zip(Ks, recall_rgb)))


if __name__ == "__main__":
    main()
