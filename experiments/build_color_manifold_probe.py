"""Build a MANIFOLD-FLAVORED color probe (not PCA) for cogito-probed.

User wants manifold directions, not PCA components. Manifold directions
here = (1) directions TOWARD specific xkcd named colors (the "raw" axes
of cogito's color manifold), and (2) hand-crafted contrast axes (warmness
etc.) defined by extremes WITHIN the manifold. NO PCA components.

Output: /Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/color_manifold_probes_layer40.npz

The cogito-probed server's reload_probes_if_changed() (file watcher)
auto-loads this on a 5s interval. File name must match
``*_probes_layer*.npz``.
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors, load_harvest
from color_filter_list import filter_colors


N_T = 28
LAYER = 40
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]   # color-focused templates


def main() -> int:
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X_full = load_harvest(cache)
    n_colors = X_full.shape[0] // N_T
    X_full = X_full[: n_colors * N_T]

    # Per-color centroids — top-6 color-focused templates only
    centroids_all = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        base = ci * N_T
        rows = [base + ti for ti in TOP_TEMPLATES]
        centroids_all[ci] = X_full[rows].mean(axis=0)

    colors_all = load_xkcd_colors()[:n_colors]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids_all[kept_idx]
    names = [c[0] for c in kept]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    mean_centroid = centroids.mean(0)

    print(f"[probe] {n_colors} raw → {len(kept)} after filter")

    # --- (1) Direction TOWARD each canonical color (manifold-native)
    canonical = [
        ("red", "red"), ("orange", "orange"), ("yellow", "yellow"),
        ("green", "green"), ("blue", "blue"), ("purple", "purple"),
        ("pink", "pink"), ("brown", "brown"),
        ("black", "black"), ("white", "white"), ("grey", "grey"),
        ("cyan", "cyan"), ("magenta", "magenta"),
    ]
    name_to_idx = {n: i for i, n in enumerate(names)}
    canonical_dirs = []
    canonical_labels = []
    for label, key in canonical:
        if key not in name_to_idx:
            print(f"  ! {key} not in filtered colors, skipping")
            continue
        i = name_to_idx[key]
        v = (centroids[i] - mean_centroid).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-9)
        canonical_dirs.append(v)
        canonical_labels.append(label)

    # --- (2) Hand-crafted axes from extremes (manifold-defined contrasts)
    def axis_direction(score: np.ndarray, name: str) -> tuple[np.ndarray, str]:
        order = np.argsort(score)
        n_extreme = 12
        low = centroids[order[:n_extreme]].mean(0)
        high = centroids[order[-n_extreme:]].mean(0)
        v = (high - low).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-9)
        return v, name

    axis_specs = [
        (rgb[:, 0] - rgb[:, 1] - rgb[:, 2], "redness"),
        (rgb[:, 1] - rgb[:, 0] - rgb[:, 2], "greenness"),
        (rgb[:, 2] - rgb[:, 0] - rgb[:, 1], "blueness"),
        (hsv[:, 2], "lightness"),
        (hsv[:, 1], "saturation"),
        (hsv[:, 1] * hsv[:, 2], "chroma"),
        (rgb[:, 0] - rgb[:, 2], "warmness"),
    ]
    axis_dirs = []
    axis_labels = []
    for score, name in axis_specs:
        v, lbl = axis_direction(score, name)
        axis_dirs.append(v); axis_labels.append(lbl)

    # --- (3) Hue-wheel directions: 12 hue spokes
    hue_dirs = []
    hue_labels = []
    for i, target_h in enumerate(np.linspace(0, 1, 12, endpoint=False)):
        d_hue = np.minimum(np.abs(hsv[:, 0] - target_h),
                            1 - np.abs(hsv[:, 0] - target_h))
        # Weight by chroma to bias toward saturated representatives
        weight = (hsv[:, 1] * hsv[:, 2]) * np.exp(-(d_hue / 0.05) ** 2)
        if weight.sum() < 1e-6:
            continue
        target = (weight[:, None] * centroids).sum(0) / weight.sum()
        v = (target - mean_centroid).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-9)
        hue_label = f"hue_{int(target_h*360):03d}"
        hue_dirs.append(v); hue_labels.append(hue_label)

    all_dirs = np.stack(canonical_dirs + axis_dirs + hue_dirs).astype(np.float32)
    all_labels = canonical_labels + axis_labels + hue_labels
    assert all_dirs.shape[0] == len(all_labels)
    assert all_dirs.shape[1] == 7168

    out_path = cache.parent / "color_manifold_probes_layer40.npz"
    desc = (
        f"Color manifold probes: {len(canonical_labels)} canonical-color directions "
        f"+ {len(axis_labels)} hand-crafted contrast axes "
        f"+ {len(hue_labels)} hue-wheel spokes. Built from {len(kept)} xkcd "
        f"centroids at layer 40 using the 6 color-focused templates "
        f"(top per-template alignment R²). NO PCA."
    )
    np.savez(out_path,
              directions=all_dirs,
              labels=np.array(all_labels),
              description=desc)
    print(f"\n[saved] {out_path}")
    print(f"  total directions: {len(all_labels)}")
    print(f"  labels: {all_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
