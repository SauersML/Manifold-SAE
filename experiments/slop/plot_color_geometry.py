"""Visualize cogito's per-color residual centroids in 2D + 3D, side-by-side
with pure-RGB and pure-HSV reference geometries.

Reads a (N_prompts, D) residual harvest produced by color_geometry.py,
collapses to one (D,) centroid per color via mean over its templates,
and projects to 2D / 3D via PCA and UMAP. Each projection is rendered for
THREE sources so you can eyeball which one cogito looks most like:

  * cogito centroids — what cogito has learned at layer 40
  * RGB cube reference — colors plotted purely by (R, G, B)
  * HSV-periodic reference — colors plotted by (cos 2πh, sin 2πh, s, v)

Outputs:
  color_centroids_2d.png   — 3×2 panel grid (rows: cogito / RGB / HSV;
                              columns: PCA-2D / UMAP-2D)
  color_centroids_3d.png   — same 3×2 grid in 3D
  color_centroids_3d.html  — interactive plotly 3D (cogito only, rotate
                              with your mouse in the browser)
"""

from __future__ import annotations

import colorsys
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection


N_TEMPLATES = 28
XKCD_URL = "https://xkcd.com/color/rgb.txt"


def load_xkcd_colors() -> list[tuple[str, int, int, int]]:
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


def load_harvest(cache_path: Path) -> np.ndarray:
    if cache_path.suffix == ".npz":
        d = np.load(cache_path, allow_pickle=False)
        return np.asarray(d["X"])
    if cache_path.suffix == ".npy":
        return np.load(cache_path)
    raise ValueError(f"need .npy or .npz; got {cache_path.suffix}")


def pca_to_d(X: np.ndarray, d: int) -> np.ndarray:
    """First d PCs of X (N, D). Returns (N, d). Center but don't scale.

    Delegates to ``_pca_basis.top_pcs`` (sklearn.decomposition.PCA) so all
    PCA in this repo uses one code path.
    """
    from _pca_basis import top_pcs
    return top_pcs(X, d=d, standardize=False)


def umap_to_d(X: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    """UMAP to d dimensions. Falls back to PCA if umap-learn isn't installed."""
    try:
        import umap
    except ImportError:
        print(f"[umap_{d}d] umap-learn not installed; falling back to PCA",
              file=sys.stderr)
        return pca_to_d(X, d)
    n = X.shape[0]
    reducer = umap.UMAP(
        n_components=d, n_neighbors=min(15, max(2, n // 3)),
        min_dist=0.1, random_state=seed, metric="euclidean",
    )
    return reducer.fit_transform(X)


def pick_label_indices(rgb_per_color: np.ndarray, n_labels: int = 12) -> list[int]:
    """Pick high-chroma representatives near canonical hues so labels are
    interpretable on the scatter."""
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_per_color])
    targets = np.linspace(0, 1, n_labels, endpoint=False)
    out = []
    for th in targets:
        chroma = hsv[:, 1] * hsv[:, 2]
        d = np.minimum(np.abs(hsv[:, 0] - th), 1 - np.abs(hsv[:, 0] - th))
        score = d - 0.5 * chroma
        out.append(int(np.argmin(score)))
    return out


def draw_2d(ax, Z, rgb, names, label_idx, title):
    ax.scatter(Z[:, 0], Z[:, 1], c=rgb, s=120, edgecolors="black", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    for i in label_idx:
        ax.annotate(names[i], (Z[i, 0], Z[i, 1]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")


def draw_3d(ax, Z, rgb, names, label_idx, title):
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=rgb, s=70,
               edgecolors="black", linewidth=0.3, depthshade=False)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for i in label_idx:
        ax.text(Z[i, 0], Z[i, 1], Z[i, 2], names[i], fontsize=6, alpha=0.85)


def per_dim_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True).clip(min=1e-6)
    return (X - mu) / sigma


def write_interactive_3d(sources: dict[str, np.ndarray],
                          rgb_per_color: np.ndarray, names: list[str],
                          out_path: Path):
    """One HTML with N subplots, each rotateable independently."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[3d_html] plotly not installed; skipping interactive HTML",
              file=sys.stderr)
        return
    hex_colors = [
        "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in rgb_per_color
    ]
    titles = list(sources.keys())
    fig = make_subplots(
        rows=1, cols=len(sources),
        specs=[[{"type": "scatter3d"}] * len(sources)],
        subplot_titles=titles,
    )
    for col, (label, Z3) in enumerate(sources.items(), start=1):
        fig.add_trace(
            go.Scatter3d(
                x=Z3[:, 0], y=Z3[:, 1], z=Z3[:, 2],
                mode="markers",
                marker=dict(size=4, color=hex_colors,
                            line=dict(color="black", width=0.3)),
                text=names, hoverinfo="text",
                name=label, showlegend=False,
            ),
            row=1, col=col,
        )
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=40, b=0),
        title="cogito L40 — interactive 3D projections "
              "(drag to rotate, scroll to zoom)",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[3d_html] {out_path}", flush=True)


def main() -> int:
    cache_path = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.snapshot.npz",
    ))
    out_dir = Path(os.environ.get(
        "OUTPUT_DIR",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40",
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {cache_path}", flush=True)
    X_full = load_harvest(cache_path)
    n_colors = X_full.shape[0] // N_TEMPLATES
    print(f"[shape] X={X_full.shape}  n_colors={n_colors}", flush=True)

    X_full = X_full[: n_colors * N_TEMPLATES]
    colors = load_xkcd_colors()[: n_colors]
    rgb_per_color = np.array([(r, g, b) for _, r, g, b in colors],
                              dtype=np.float64) / 255.0
    names = [c[0] for c in colors]

    # Cogito centroids — mean over 28 templates per color.
    centroids = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X_full[ci * N_TEMPLATES:(ci + 1) * N_TEMPLATES].mean(axis=0)

    # Three input spaces to compare. Cogito is high-D (~7168) → per-dim
    # standardized; RGB and HSV-periodic are low-D and used as-is.
    X_cog = per_dim_standardize(centroids)
    X_rgb = rgb_per_color - rgb_per_color.mean(axis=0, keepdims=True)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_per_color])
    X_hsv = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]),
        np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1], hsv[:, 2],
    ], axis=1)
    X_hsv = X_hsv - X_hsv.mean(axis=0, keepdims=True)

    sources = {"cogito L40": X_cog, "RGB cube": X_rgb, "HSV-periodic": X_hsv}

    # Cache projections
    print("[project] computing PCA/UMAP for each source ...", flush=True)
    projections = {}
    for label, X in sources.items():
        projections[label] = {
            "pca_2d": pca_to_d(X, 2),
            "umap_2d": umap_to_d(X, 2),
            "pca_3d": pca_to_d(X, 3),
            "umap_3d": umap_to_d(X, 3),
        }
        print(f"  {label}: done", flush=True)

    label_idx = pick_label_indices(rgb_per_color, n_labels=12)

    # === 2D PNG: 3 rows × 2 cols (PCA / UMAP for each of 3 sources) ===
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    for row, (label, projs) in enumerate(projections.items()):
        draw_2d(axes[row, 0], projs["pca_2d"],  rgb_per_color, names, label_idx,
                f"{label} — PCA-2D")
        draw_2d(axes[row, 1], projs["umap_2d"], rgb_per_color, names, label_idx,
                f"{label} — UMAP-2D")
    plt.suptitle(
        f"each point = one xkcd color   ·   cogito centroid = mean over 28 templates\n"
        f"n_colors = {n_colors}   ·   source: {cache_path.name}",
        fontsize=11, y=1.003,
    )
    plt.tight_layout()
    png2d = out_dir / "color_centroids_2d.png"
    plt.savefig(png2d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2d] {png2d}", flush=True)

    # === 3D PNG: 3 rows × 2 cols ===
    fig = plt.figure(figsize=(15, 18))
    for row, (label, projs) in enumerate(projections.items()):
        ax = fig.add_subplot(3, 2, 2 * row + 1, projection="3d")
        draw_3d(ax, projs["pca_3d"],  rgb_per_color, names, label_idx,
                f"{label} — PCA-3D")
        ax = fig.add_subplot(3, 2, 2 * row + 2, projection="3d")
        draw_3d(ax, projs["umap_3d"], rgb_per_color, names, label_idx,
                f"{label} — UMAP-3D")
    plt.suptitle(
        f"3D projections   ·   n_colors = {n_colors}   ·   source: {cache_path.name}",
        fontsize=11, y=1.001,
    )
    plt.tight_layout()
    png3d = out_dir / "color_centroids_3d.png"
    plt.savefig(png3d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3d] {png3d}", flush=True)

    # === Interactive 3D HTML — one subplot per source × method ===
    html_sources = {
        "cogito PCA-3D":  projections["cogito L40"]["pca_3d"],
        "cogito UMAP-3D": projections["cogito L40"]["umap_3d"],
        "RGB cube PCA-3D": projections["RGB cube"]["pca_3d"],
    }
    html_path = out_dir / "color_centroids_3d.html"
    write_interactive_3d(html_sources, rgb_per_color, names, html_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
