"""Quantitative analyses of cogito's color geometry vs RGB / HSV references.

Loads the same harvest as plot_color_geometry.py, builds per-color centroids,
projects each space (cogito residual, RGB cube, HSV-periodic) to 2D via UMAP,
and runs four quantitative comparisons + one diagnostic visualization:

1. PROCRUSTES R²
   Find the optimal rigid alignment (rotation+scale+reflection) between
   cogito's UMAP-2D and each reference's UMAP-2D. The remaining residual
   variance tells us how much of cogito's 2D geometry is NOT explained by
   the reference shape. Reported as 1 - (||T_cog - T_ref_aligned||² / ||T_cog||²).

2. k-NN JACCARD
   For each color find its top-k nearest neighbors in cogito-residual space
   and in each reference space (separately). The Jaccard overlap |A ∩ B| /
   |A ∪ B|, averaged over colors, says "which input geometry does cogito's
   local structure agree with most?". Reported for k ∈ {5, 10, 20}.

3. MANTEL DISTANCE-CORRELATION
   Pearson r between the upper-triangle of cogito's pairwise-distance matrix
   and each reference's. Global metric — does cogito preserve gross
   pairwise distance structure?

4. HUE-ANGLE RECOVERY
   In each space's UMAP-2D, compute the angle of each point around the
   centroid. Correlate with the true hue (Spearman + circular). If cogito
   has learned a hue wheel, this should be high (>0.7); if scattered,
   low (<0.3).

5. COLOR-BY-AXIS PANEL  (saved as PNG)
   Cogito's UMAP-2D re-colored by hue, saturation, value, R, G, B,
   luminance. Reveals which subset of color axes each region of cogito's
   manifold is encoding.
"""

from __future__ import annotations

import colorsys
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N_TEMPLATES = 28
XKCD_URL = "https://xkcd.com/color/rgb.txt"


def load_xkcd_colors():
    cache = Path(__file__).parent / "xkcd_colors.txt"
    if cache.exists():
        text = cache.read_text()
    else:
        with urllib.request.urlopen(XKCD_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        try:
            cache.write_text(text)
        except OSError:
            pass
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("License") or s.startswith("Copyright"):
            continue
        m = re.match(r"^(.+?)\s+#?([0-9a-fA-F]{6})$", s)
        if not m:
            continue
        name, hex_ = m.group(1).strip(), m.group(2)
        out.append((name, int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)))
    return out


def load_harvest(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.asarray(np.load(path, allow_pickle=False)["X"])
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(path.suffix)


def per_dim_standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True).clip(min=1e-6)


def umap_to_d(X: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    import umap
    n = X.shape[0]
    return umap.UMAP(
        n_components=d, n_neighbors=min(15, max(2, n // 3)),
        min_dist=0.1, random_state=seed,
    ).fit_transform(X)


# -------------- Analysis 1: Procrustes ---------------------------------------


def procrustes_r2(A: np.ndarray, B: np.ndarray) -> float:
    """Find the best rotation/scale/reflection mapping A → B, return the
    fraction of variance in B explained after alignment.

    Reports 1 - ||B - T(A)||² / ||B - mean(B)||² where T is the optimal
    rigid+scale alignment. Same shape semantics as orthogonal Procrustes.
    """
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    M = B0.T @ A0
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    s = np.trace(B0.T @ A0 @ R.T) / max(np.sum(A0 ** 2), 1e-12)
    pred = s * (A0 @ R.T)
    ss_res = np.sum((B0 - pred) ** 2)
    ss_tot = np.sum(B0 ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


# -------------- Analysis 2: k-NN Jaccard -------------------------------------


def knn_jaccard(X1: np.ndarray, X2: np.ndarray, k: int) -> float:
    """Mean Jaccard overlap of top-k nearest neighbors in two spaces."""
    n = X1.shape[0]
    def knn(X):
        # Use Euclidean distance; exclude self.
        sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq, np.inf)
        return np.argsort(sq, axis=1)[:, :k]
    knn1, knn2 = knn(X1), knn(X2)
    out = []
    for i in range(n):
        a, b = set(knn1[i]), set(knn2[i])
        u = a | b
        out.append(len(a & b) / max(len(u), 1))
    return float(np.mean(out))


# -------------- Analysis 3: Mantel-style distance correlation ----------------


def pairwise_dist(X: np.ndarray) -> np.ndarray:
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(sq, 0.0))


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


def mantel_pearson(D1: np.ndarray, D2: np.ndarray) -> float:
    v1, v2 = upper_tri(D1), upper_tri(D2)
    v1 = v1 - v1.mean(); v2 = v2 - v2.mean()
    return float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


# -------------- Analysis 4: Hue-angle recovery -------------------------------


def hue_angle_recovery(Z2: np.ndarray, hues: np.ndarray) -> dict:
    """Compute angle of each point around the centroid in a 2D embedding;
    correlate with true hue. Returns Spearman + circular correlation.
    """
    Zc = Z2 - Z2.mean(0, keepdims=True)
    angles = np.arctan2(Zc[:, 1], Zc[:, 0])               # (-π, π]
    angles_unit = (angles + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)   # [0, 1)
    rho_lin = float(spearman(angles_unit, hues))
    # Circular Pearson: r = sum(sin(2π(h - h̄)) * sin(2π(θ - θ̄))) ...
    h_rad = 2 * np.pi * hues
    a_rad = 2 * np.pi * angles_unit
    def circ_centered(x):
        return np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    h_bar = circ_centered(h_rad)
    a_bar = circ_centered(a_rad)
    num = np.sum(np.sin(h_rad - h_bar) * np.sin(a_rad - a_bar))
    den = np.sqrt(np.sum(np.sin(h_rad - h_bar) ** 2) *
                  np.sum(np.sin(a_rad - a_bar) ** 2))
    rho_circ = float(num / max(den, 1e-12))
    return {"spearman_angle_vs_hue": rho_lin,
            "circular_correlation_angle_vs_hue": rho_circ}


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


# -------------- Analysis 5: color-by-axis panel ------------------------------


def color_by_axis_panel(Z2: np.ndarray, rgb_per_color: np.ndarray,
                        out_path: Path):
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_per_color])
    luminance = 0.299 * rgb_per_color[:, 0] + 0.587 * rgb_per_color[:, 1] + 0.114 * rgb_per_color[:, 2]
    axes_data = {
        "TRUE RGB color": None,           # special: use rgb_per_color directly
        "hue":  hsv[:, 0],
        "saturation": hsv[:, 1],
        "value (lightness)": hsv[:, 2],
        "R": rgb_per_color[:, 0],
        "G": rgb_per_color[:, 1],
        "B": rgb_per_color[:, 2],
        "luminance": luminance,
    }
    n = len(axes_data)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axs = axs.flatten()
    for ax, (label, vals) in zip(axs, axes_data.items()):
        if vals is None:
            ax.scatter(Z2[:, 0], Z2[:, 1], c=rgb_per_color, s=70,
                       edgecolors="black", linewidth=0.3)
            ax.set_title(label, fontsize=11)
        else:
            cmap = "hsv" if label == "hue" else "viridis"
            sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=vals, s=70, cmap=cmap,
                            edgecolors="black", linewidth=0.3)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f5f5f5")
    for ax in axs[n:]:
        ax.set_visible(False)
    plt.suptitle("cogito L40 UMAP-2D, re-colored by each color axis",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------- main ---------------------------------------------------------


def main() -> int:
    cache_path = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.snapshot.npz",
    ))
    out_dir = Path(os.environ.get(
        "OUTPUT_DIR",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40",
    ))

    X_full = load_harvest(cache_path)
    n_colors = X_full.shape[0] // N_TEMPLATES
    X_full = X_full[: n_colors * N_TEMPLATES]
    colors = load_xkcd_colors()[: n_colors]
    rgb_per_color = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    hsv_per_color = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_per_color])
    print(f"[load] {cache_path.name}  n_colors={n_colors}", flush=True)

    # Per-color centroids.
    centroids = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X_full[ci * N_TEMPLATES:(ci + 1) * N_TEMPLATES].mean(0)
    X_cog = per_dim_standardize(centroids)
    X_rgb = rgb_per_color - rgb_per_color.mean(0, keepdims=True)
    X_hsv_periodic = np.stack([
        np.cos(2 * np.pi * hsv_per_color[:, 0]),
        np.sin(2 * np.pi * hsv_per_color[:, 0]),
        hsv_per_color[:, 1], hsv_per_color[:, 2],
    ], axis=1) - 0  # already centered enough; not subtracting for clarity
    X_hsv_periodic -= X_hsv_periodic.mean(0, keepdims=True)

    # UMAP-2D of each.
    print("[umap] cogito / rgb / hsv-periodic ...", flush=True)
    Z_cog = umap_to_d(X_cog, 2, seed=0)
    Z_rgb = umap_to_d(X_rgb, 2, seed=0)
    Z_hsv = umap_to_d(X_hsv_periodic, 2, seed=0)

    results = {"n_colors": n_colors, "source": str(cache_path)}

    # ---- Analysis 1: Procrustes ----
    print("\n=== Procrustes R² (cogito UMAP-2D explained by reference UMAP-2D) ===")
    procrustes = {
        "RGB":  procrustes_r2(Z_rgb, Z_cog),
        "HSV":  procrustes_r2(Z_hsv, Z_cog),
    }
    for k, v in procrustes.items():
        print(f"  cogito vs {k:3}: R² = {v:+.3f}")
    results["procrustes_R2"] = procrustes

    # ---- Analysis 2: k-NN Jaccard (in ORIGINAL high-D residual + 4D HSV,
    # not in the 2D projection — that's the right comparison) ----
    print("\n=== k-NN Jaccard (cogito-residual vs RGB / HSV) ===")
    knn = {}
    for k in (5, 10, 20):
        knn[f"k={k}_RGB"] = knn_jaccard(centroids, rgb_per_color, k)
        knn[f"k={k}_HSV"] = knn_jaccard(centroids, X_hsv_periodic, k)
    print(f"  {'k':>4}  {'Jaccard vs RGB':>16}  {'Jaccard vs HSV':>16}")
    for k in (5, 10, 20):
        print(f"  {k:>4}  {knn[f'k={k}_RGB']:>16.3f}  {knn[f'k={k}_HSV']:>16.3f}")
    results["knn_jaccard"] = knn

    # ---- Analysis 3: Mantel ----
    print("\n=== Mantel pairwise-distance correlation ===")
    D_cog = pairwise_dist(centroids)
    D_rgb = pairwise_dist(rgb_per_color)
    D_hsv = pairwise_dist(X_hsv_periodic)
    mantel = {
        "cogito_vs_RGB": mantel_pearson(D_cog, D_rgb),
        "cogito_vs_HSV": mantel_pearson(D_cog, D_hsv),
    }
    for k, v in mantel.items():
        print(f"  {k:18}: r = {v:+.3f}")
    results["mantel_r"] = mantel

    # ---- Analysis 4: Hue-angle recovery ----
    print("\n=== Hue-angle recovery (angle around centroid in UMAP-2D) ===")
    hue_results = {
        "cogito_umap2d": hue_angle_recovery(Z_cog, hsv_per_color[:, 0]),
        "rgb_umap2d":    hue_angle_recovery(Z_rgb, hsv_per_color[:, 0]),
        "hsv_umap2d":    hue_angle_recovery(Z_hsv, hsv_per_color[:, 0]),
    }
    print(f"  {'embedding':18}  {'Spearman ρ':>11}  {'circ. ρ':>9}")
    for k, v in hue_results.items():
        print(f"  {k:18}  {v['spearman_angle_vs_hue']:+11.3f}  "
              f"{v['circular_correlation_angle_vs_hue']:+9.3f}")
    results["hue_angle"] = hue_results

    # ---- Analysis 5: color-by-axis panel ----
    panel_path = out_dir / "color_by_axis_panel.png"
    color_by_axis_panel(Z_cog, rgb_per_color, panel_path)
    print(f"\n[panel] {panel_path}")
    results["panel"] = str(panel_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analyses.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] {out_dir / 'analyses.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
