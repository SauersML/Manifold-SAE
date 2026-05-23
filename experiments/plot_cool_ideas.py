"""Three quick analyses on existing harvest data:

  A) Per-PC interpretation
     For PCs 1..6 of the per-color centroids, show the 8 colors that
     score highest and lowest on that PC. Reveals what each PC encodes
     (lightness? warm/cool? chroma? something semantic?).

  B) Per-prompt UMAP
     Embed all 9212 individual prompts (not per-color means) into UMAP-2D.
     Color each dot by its underlying color. If templates dominate, dots
     scatter randomly; if color identity dominates, all 28 prompts of
     each color cluster together. Tests "color invariance across
     templates" at a glance.

  C) Pairwise-distance correlation matrix
     A 5×5 heatmap of Pearson r between cogito's pairwise distance
     matrix and RGB / Lab / Oklab / HSV / Lch pairwise distance matrices.
     Says which input geometry cogito's residual geometry is closest
     to globally.
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))


N_T = 28


def load_xkcd_colors():
    from plot_color_geometry import load_xkcd_colors as _f
    return _f()


def load_harvest(p: Path) -> np.ndarray:
    from plot_color_geometry import load_harvest as _f
    return _f(p)


def umap_to_d(X, d, seed=0):
    from plot_color_geometry import umap_to_d as _f
    return _f(X, d, seed)


def rgb_to_lab_local(rgb01: np.ndarray) -> np.ndarray:
    from color_manifold_gam import rgb_to_lab
    return rgb_to_lab(rgb01)


def rgb_to_oklab_local(rgb01: np.ndarray) -> np.ndarray:
    from color_manifold_gam import rgb_to_oklab
    return rgb_to_oklab(rgb01)


def rgb_to_lch_local(rgb01: np.ndarray) -> np.ndarray:
    from color_manifold_gam import rgb_to_lch
    return rgb_to_lch(rgb01)


# ---------------------------------------------------------------------------
# A) Per-PC interpretation: top/bottom 8 colors per PC
# ---------------------------------------------------------------------------


def render_per_pc_panel(centroids: np.ndarray, names: list[str],
                          rgb_per_color: np.ndarray, out_path: Path,
                          n_pcs: int = 6, n_top: int = 8):
    # PCA on per-color centroids
    Xc = centroids - centroids.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T[:, :n_pcs]                                 # (n_colors, n_pcs)

    fig, axes = plt.subplots(n_pcs, 1, figsize=(14, n_pcs * 1.6))
    for k in range(n_pcs):
        ax = axes[k]
        scores = Z[:, k]
        order = np.argsort(scores)
        lo_idx = order[:n_top]
        hi_idx = order[-n_top:][::-1]
        labels = ["highest"] + [names[i] for i in hi_idx] + [
            "...", "lowest"
        ] + [names[i] for i in lo_idx]
        rgbs = [(0, 0, 0)] + [tuple(rgb_per_color[i]) for i in hi_idx] + [
            (1, 1, 1), (0, 0, 0),
        ] + [tuple(rgb_per_color[i]) for i in lo_idx]

        for xi, (lbl, rgb) in enumerate(zip(labels, rgbs)):
            if lbl in ("highest", "lowest"):
                ax.text(xi + 0.5, 0.5, lbl, ha="center", va="center",
                        fontsize=9, fontweight="bold")
            elif lbl == "...":
                ax.text(xi + 0.5, 0.5, "…", ha="center", va="center", fontsize=14)
            else:
                ax.add_patch(Rectangle((xi, 0), 1, 1, facecolor=rgb, edgecolor="black",
                                        linewidth=0.4))
                # Choose white/black text based on perceptual brightness
                lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(xi + 0.5, 0.5, lbl, ha="center", va="center",
                        fontsize=8, color=txt_color, rotation=0)
        ax.set_xlim(0, len(labels))
        ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PC {k+1}   ·   variance explained: "
                     f"{(Vt[k] ** 2).sum() / (centroids.var(0).sum()) * 100:.1f}%",
                     fontsize=10)

    plt.suptitle(
        "Per-PC interpretation — what each principal component of cogito's per-color centroid space encodes",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[per_pc] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# B) Per-prompt UMAP — 9212 dots, color by RGB
# ---------------------------------------------------------------------------


def render_per_prompt_umap(X_full: np.ndarray, n_colors: int,
                            rgb_per_color: np.ndarray, names: list[str],
                            out_path: Path):
    # Per-prompt: each of 9212 dots. Color each by its underlying color RGB.
    n_rows = n_colors * N_T
    X_full = X_full[: n_rows]
    rgb_per_prompt = np.repeat(rgb_per_color, N_T, axis=0)        # (n_rows, 3)

    # Standardize per dim (same as centroid pipeline)
    mu = X_full.mean(0, keepdims=True)
    sigma = X_full.std(0, keepdims=True).clip(min=1e-6)
    Xn = (X_full - mu) / sigma
    Z = umap_to_d(Xn, 2, seed=0)

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.scatter(Z[:, 0], Z[:, 1], c=rgb_per_prompt, s=14,
                edgecolors="none", alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f8f8f8")
    ax.set_title(
        f"Per-PROMPT UMAP-2D — {n_rows} prompts ({n_colors} colors × {N_T} templates), "
        f"each dot colored by its color\n"
        f"If color identity dominates, all 28 prompts of each color cluster together. "
        f"If templates dominate, dots scatter randomly.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[per_prompt] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# C) Pairwise-distance correlation matrix
# ---------------------------------------------------------------------------


def pairwise_dist(X):
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(sq, 0.0))


def upper_tri(M):
    return M[np.triu_indices_from(M, k=1)]


def render_distance_corr(centroids: np.ndarray, rgb_per_color: np.ndarray,
                          out_path: Path):
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_per_color])
    spaces = {
        "cogito": centroids,
        "RGB":    rgb_per_color,
        "Lab":    rgb_to_lab_local(rgb_per_color),
        "Oklab":  rgb_to_oklab_local(rgb_per_color),
        "Lch":    rgb_to_lch_local(rgb_per_color),
        "HSV-cyclic": np.stack([
            np.cos(2*np.pi*hsv[:, 0]), np.sin(2*np.pi*hsv[:, 0]),
            hsv[:, 1], hsv[:, 2],
        ], axis=1),
        "luminance": (0.299 * rgb_per_color[:, 0] + 0.587 * rgb_per_color[:, 1]
                       + 0.114 * rgb_per_color[:, 2])[:, None],
    }
    dist_vecs = {label: upper_tri(pairwise_dist(X)) for label, X in spaces.items()}
    labels = list(spaces.keys())
    n = len(labels)
    M = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            va, vb = dist_vecs[a], dist_vecs[b]
            va = va - va.mean(); vb = vb - vb.mean()
            denom = np.sqrt((va * va).sum() * (vb * vb).sum()) + 1e-12
            M[i, j] = float((va * vb).sum() / denom)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r of upper-triangle pairwise distances")
    ax.set_title(
        "Pairwise color-distance correlation across spaces\n"
        "How similar are 'A and B are close colors' across these definitions of close?",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[dist_corr] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    snapshot = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.snapshot.npz",
    ))
    out_dir = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = load_harvest(snapshot)
    n_colors = X_full.shape[0] // N_T
    X_full = X_full[: n_colors * N_T]
    colors = load_xkcd_colors()[:n_colors]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    names = [c[0] for c in colors]
    centroids = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X_full[ci * N_T:(ci + 1) * N_T].mean(0)
    # Per-dim standardize for PCA
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Xn = (centroids - mu) / sigma

    render_per_pc_panel(Xn, names, rgb, out_dir / "per_pc_interpretation.png")
    render_distance_corr(centroids, rgb, out_dir / "distance_correlation.png")
    render_per_prompt_umap(X_full, n_colors, rgb, names,
                              out_dir / "per_prompt_umap.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
