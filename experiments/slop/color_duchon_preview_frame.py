"""Render one high-resolution still of an actual gamfit Duchon color fit.

This is a direct still render from checkpoint data, not a frame extracted from a
movie. It uses the same core colors and gamfit periodic Duchon fit as
``color_duchon_loop_training_video.py``.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_duchon_loop_training_video import (
    fit_hue_periodic_loop,
    label_from_order,
    load_color_means,
    sort_key,
    stage_from_order,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="/tmp/colall/*.npz")
    parser.add_argument("--checkpoint", default="last", choices=["first", "middle", "last"])
    parser.add_argument("--out", default="results/color_duchon_training_real/preview_final_rl31.png")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob), key=sort_key)
    if not files:
        raise SystemExit(f"no files matched {args.input_glob}")
    idx = {"first": 0, "middle": len(files) // 2, "last": len(files) - 1}[args.checkpoint]
    path = files[idx]

    vectors, _names, rgbs = load_color_means(path)
    centered = vectors - vectors.mean(axis=0)
    display_basis = np.linalg.svd(centered, full_matrices=False)[2][:3]
    points = centered @ display_basis.T
    curve, score, edf, _lam = fit_hue_periodic_loop(points, rgbs)

    all_points = np.concatenate([points, curve], axis=0)
    center = all_points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_points - center, axis=1)) * 1.22)
    order = sort_key(path)
    label = label_from_order(order)
    stage = stage_from_order(order)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 9.2), dpi=args.dpi)
    fig.patch.set_facecolor("#080a0f")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#080a0f")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("color PC1", color="#aeb7c8", labelpad=14)
    ax.set_ylabel("color PC2", color="#aeb7c8", labelpad=14)
    ax.set_zlabel("color PC3", color="#aeb7c8", labelpad=14)
    ax.tick_params(colors="#556176", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.05, 0.06, 0.10, 0.26))
        axis.pane.set_edgecolor((0.62, 0.70, 0.85, 0.26))
    ax.grid(True, color="#1f2937")
    ax.view_init(elev=22, azim=42)

    colors = [(r / 255.0, g / 255.0, b / 255.0, 1.0) for r, g, b in rgbs]
    for size, alpha in [(240, 0.08), (110, 0.26), (48, 1.0)]:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=[(*c[:3], alpha) for c in colors], depthshade=False)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=(1, 1, 1, 0.13), lw=14)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=(1, 1, 1, 0.97), lw=4.2)

    fig.text(0.055, 0.94, "Gamfit Duchon color loop", color="#f8fafc", fontsize=30, weight="semibold")
    fig.text(
        0.055,
        0.902,
        f"{label} · {stage} · actual checkpoint data · closed hue-periodic Duchon decoder · R² {score:.2f} · EDF {edf:.1f}",
        color="#aeb7c8",
        fontsize=13,
    )
    fig.text(0.055, 0.873, Path(path).name, color="#64748b", fontsize=10)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
