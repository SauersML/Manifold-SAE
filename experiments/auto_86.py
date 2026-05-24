"""auto_86: Direct PC2 x PC4 hue-ring scatter of 949 xkcd-color centroids.

Motivation: auto_82 (J-S circ-corr ~ -0.72 for PC2+PC4), auto_85 (per-PC
variance vs hue-circularity), and auto_exp_47/49 all *inferred* a hue ring
sitting in the (PC2, PC4) plane of standardized L40 centroids, but no
prior auto_* actually rendered that 2D scatter colored by true xkcd-RGB.

This script does exactly that and adds two companion panels so the ring
is visually unambiguous:

  Left:   PC2 vs PC4 scatter, points colored by xkcd-RGB (the hue ring).
  Middle: Same points colored by HSV hue (continuous viridis-like wheel),
          to make ring-ordering obvious independent of saturation/value.
  Right:  Polar plot of angle(PC2, PC4) on the unit circle vs true hue
          (rad), showing 1:1 alignment (with the empirical sign flip).

The headline number printed: Jammalamadaka-Sarma circular correlation
between angle(PC2, PC4) and 2*pi*hue.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_86.png"

K_PCS = 16
PC_I, PC_J = 2, 4   # auto_82's "PC2+PC4" — i=2, j=4 (3rd and 5th singular dirs)


def circ_corr_js(a, b):
    a_bar = np.angle(np.mean(np.exp(1j * a)))
    b_bar = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - a_bar) * np.sin(b - b_bar))
    den = np.sqrt(np.sum(np.sin(a - a_bar) ** 2)
                  * np.sum(np.sin(b - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def main():
    t0 = time.time()
    print("[auto_86] PC2 x PC4 hue-ring direct scatter")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    hue = hsv[:, 0]
    hue_rad = 2 * np.pi * hue
    rgb_clip = np.clip(rgb, 0, 1)

    # PCA on centered standardized centroids (matches auto_82 convention)
    Tc = T0 - T0.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    print(f"[pca] top singular values: {S[:6].round(3)}")
    plane = Tc @ Vt[[PC_I, PC_J]].T
    th = np.arctan2(plane[:, 1], plane[:, 0])
    cc = circ_corr_js(th, hue_rad)
    print(f"[circ-corr] angle(PC{PC_I}, PC{PC_J}) vs hue: {cc:+.3f}")

    # Standardize axes for plotting (zero-mean, unit-std per axis)
    p = plane.copy()
    p_n = (p - p.mean(0)) / p.std(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    # Panel 1: scatter colored by true xkcd-RGB
    ax = axes[0]
    ax.scatter(p_n[:, 0], p_n[:, 1], c=rgb_clip, s=18,
               edgecolors="black", linewidths=0.15)
    ax.set_xlabel(f"PC{PC_I} (standardized)")
    ax.set_ylabel(f"PC{PC_J} (standardized)")
    ax.set_title(f"Centroids in PC{PC_I} x PC{PC_J} plane\n"
                 f"colored by xkcd-RGB  (N={n})")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)

    # Panel 2: same scatter colored continuously by hue (hsv colormap)
    ax = axes[1]
    sc = ax.scatter(p_n[:, 0], p_n[:, 1], c=hue, cmap="hsv",
                    s=18, edgecolors="black", linewidths=0.15,
                    vmin=0, vmax=1)
    ax.set_xlabel(f"PC{PC_I} (standardized)")
    ax.set_ylabel(f"PC{PC_J} (standardized)")
    ax.set_title(f"Same scatter, colored by HSV hue\n"
                 f"J-S circ-corr(angle, hue) = {cc:+.3f}")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="true HSV hue")

    # Panel 3: polar 1:1 — recovered PC-plane angle vs true hue angle
    ax = axes[2]
    # Flip sign of recovered angle if circ-corr is negative, so points hug y=x
    sign = 1.0 if cc >= 0 else -1.0
    th_show = (sign * th) % (2 * np.pi)
    ax.scatter(hue_rad, th_show, c=rgb_clip, s=14,
               edgecolors="black", linewidths=0.15)
    diag = np.linspace(0, 2 * np.pi, 200)
    ax.plot(diag, diag, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("true hue angle (rad)")
    ax.set_ylabel(f"angle(PC{PC_I}, PC{PC_J})  [sign-flipped]")
    ax.set_title("Hue-ring alignment: recovered vs true angle")
    ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")

    fig.suptitle(
        f"auto_86 — Hue ring in (PC{PC_I}, PC{PC_J}) "
        f"of L40 standardized centroids   |   "
        f"|circ-corr| = {abs(cc):.3f}", fontsize=12)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
