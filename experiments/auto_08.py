"""auto_08: Is the U_3d latent manifold actually CIRCULAR in hue?

Follow-up to auto_07, which found t3 ↔ hue Spearman ρ=+0.41 (the strongest
single latent↔axis alignment in U_3d). A Spearman rank correlation is
*linear* in rank, but hue is a *cyclic* variable on [0, 1). If the manifold
really encodes hue, we'd expect points to wrap around a closed loop, not
to sit on a monotonic line.

This script tests the circularity hypothesis directly:

  1. Take the 949-point U_3d latent T.
  2. Find the 2D plane that best explains hue variation
     (linear regress hue's sin/cos onto T → projection axes).
  3. Project T onto that plane → (x, y).
  4. Convert to polar (r, θ_latent).
  5. Compute circular correlation between θ_latent and true hue*2π.
  6. Plot:
       - latent points on the 2D plane, colored by true RGB
       - same, colored by pure-hue rainbow
       - θ_latent vs hue scatter (should be a diagonal if circular)
       - histogram of residual angle (hue*2π − θ_latent) mod 2π,
         peaked if alignment is good

Reads:  runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
Writes: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_08.png
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
OUT = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_08.png"


def circ_corr(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Jammalamadaka & Sarma circular correlation coefficient (radians)."""
    a_bar = np.arctan2(np.sin(alpha).mean(), np.cos(alpha).mean())
    b_bar = np.arctan2(np.sin(beta).mean(),  np.cos(beta).mean())
    num = np.sum(np.sin(alpha - a_bar) * np.sin(beta - b_bar))
    den = np.sqrt(np.sum(np.sin(alpha - a_bar) ** 2) *
                  np.sum(np.sin(beta - b_bar) ** 2))
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    with open(RESULTS) as f:
        d = json.load(f)
    L = d["per_layer"]["L40"]
    T = np.array(L["unsupervised_full_data"]["d=3"]["T"])  # (949, 3)
    n = T.shape[0]
    R = np.array(d["color_axes_per_color_index"]["R"])[:n]
    G = np.array(d["color_axes_per_color_index"]["G"])[:n]
    B = np.array(d["color_axes_per_color_index"]["B"])[:n]
    hue = np.array(d["color_axes_per_color_index"]["hue"])[:n]
    sat = np.array(d["color_axes_per_color_index"]["sat"])[:n]
    val = np.array(d["color_axes_per_color_index"]["value"])[:n]

    rgb = np.stack([R, G, B], axis=1).clip(0, 1)
    rgb_hue = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])

    # ---- Find plane in latent space that best aligns with sin/cos(2π·hue)
    Tc = T - T.mean(axis=0, keepdims=True)
    theta_true = 2 * np.pi * hue           # (n,)
    Y = np.stack([np.cos(theta_true), np.sin(theta_true)], axis=1)  # (n,2)
    # Least squares: Tc @ W = Y  →  W = (TcᵀTc)⁻¹ Tcᵀ Y, then plane = column-
    # space of Tc @ W (≈ the 2 latent directions most predictive of sin/cos).
    W, *_ = np.linalg.lstsq(Tc, Y, rcond=None)           # (3,2)
    # Orthonormalize columns of W in latent space:
    Q, _ = np.linalg.qr(W)
    XY = Tc @ Q                                          # (n,2) projection
    x, y = XY[:, 0], XY[:, 1]

    # Polar coords of projection
    r_latent = np.hypot(x, y)
    theta_latent = np.arctan2(y, x)                       # (-π, π]
    theta_latent_unit = (theta_latent + 2 * np.pi) % (2 * np.pi)  # [0,2π)

    # Circular correlation true hue ↔ latent angle
    rho_circ = circ_corr(theta_true, theta_latent_unit)
    # Best global rotation of latent angle to match hue (1-D search):
    grid = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    losses = np.array([
        np.mean(1 - np.cos(theta_true - (theta_latent_unit + s)))
        for s in grid
    ])
    shift = grid[int(np.argmin(losses))]
    theta_aligned = (theta_latent_unit + shift) % (2 * np.pi)
    # Also try sign-flip (handedness)
    losses_flip = np.array([
        np.mean(1 - np.cos(theta_true - ((-theta_latent_unit + s) % (2*np.pi))))
        for s in grid
    ])
    if losses_flip.min() < losses.min():
        shift = grid[int(np.argmin(losses_flip))]
        theta_aligned = ((-theta_latent_unit) + shift) % (2 * np.pi)
        handed = "flipped"
    else:
        handed = "same"
    residual = np.arctan2(np.sin(theta_true - theta_aligned),
                          np.cos(theta_true - theta_aligned))
    mean_abs_resid_deg = np.degrees(np.mean(np.abs(residual)))

    # ---- Plot
    fig = plt.figure(figsize=(15, 11))

    # (1) latent 2D projection, true RGB
    a1 = fig.add_subplot(2, 3, 1)
    a1.scatter(x, y, c=rgb, s=18, alpha=0.9, edgecolors="none")
    a1.set_aspect("equal")
    a1.axhline(0, color="0.7", lw=0.5); a1.axvline(0, color="0.7", lw=0.5)
    a1.set_title("Hue-aligned latent plane\n(true xkcd RGB)")
    a1.set_xlabel("plane axis 1"); a1.set_ylabel("plane axis 2")

    # (2) latent 2D projection, pure hue rainbow
    a2 = fig.add_subplot(2, 3, 2)
    a2.scatter(x, y, c=rgb_hue, s=18, alpha=0.9, edgecolors="none")
    a2.set_aspect("equal")
    a2.axhline(0, color="0.7", lw=0.5); a2.axvline(0, color="0.7", lw=0.5)
    a2.set_title("Same plane, colored by HUE only\n"
                 "(pure rainbow → wraparound?)")
    a2.set_xlabel("plane axis 1"); a2.set_ylabel("plane axis 2")

    # (3) polar plot (θ_latent, r_latent), colored by hue
    a3 = fig.add_subplot(2, 3, 3, projection="polar")
    a3.scatter(theta_latent_unit, r_latent, c=rgb_hue, s=14, alpha=0.9,
               edgecolors="none")
    a3.set_title("Polar (θ_latent, r_latent), hue colors", pad=15)

    # (4) θ_latent (aligned) vs hue scatter: should be y=x diagonal if hue
    a4 = fig.add_subplot(2, 3, 4)
    a4.scatter(hue, theta_aligned / (2 * np.pi), c=rgb_hue, s=14, alpha=0.9,
               edgecolors="none")
    a4.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    a4.set_xlabel("true hue"); a4.set_ylabel("aligned θ_latent / 2π")
    a4.set_title(f"Hue vs aligned latent angle\n"
                 f"circ_corr ρ={rho_circ:+.3f}  ({handed} handedness)")
    a4.set_aspect("equal"); a4.set_xlim(0, 1); a4.set_ylim(0, 1)

    # (5) residual histogram in degrees
    a5 = fig.add_subplot(2, 3, 5)
    a5.hist(np.degrees(residual), bins=60, color="steelblue",
            edgecolor="white")
    a5.axvline(0, color="k", lw=0.8)
    a5.set_xlabel("angular residual (deg)")
    a5.set_ylabel("count")
    a5.set_title(f"Residual hue−θ_latent\nmean |resid| = {mean_abs_resid_deg:.1f}°"
                 f"  (chance = 90°)")

    # (6) summary panel
    a6 = fig.add_subplot(2, 3, 6); a6.axis("off")
    # Look at r vs saturation: chromatic colors should sit at larger r if the
    # manifold is actually conic.
    rho_r_sat = np.corrcoef(r_latent, sat)[0, 1]
    rho_r_val = np.corrcoef(r_latent, val)[0, 1]
    rho_r_lum = np.corrcoef(
        r_latent,
        0.299 * R + 0.587 * G + 0.114 * B
    )[0, 1]
    lines = [
        "auto_08 — Is the U_3d latent really circular in hue?",
        "",
        f"n = {n} xkcd colors, layer 40 (Cogito-27B)",
        "",
        "Picked the 2D latent plane that best predicts",
        "  (cos 2π·hue, sin 2π·hue) via least-squares on T_3d.",
        "Projected, then converted to polar.",
        "",
        f"Circular correlation  ρ_circ(hue, θ_latent) = {rho_circ:+.3f}",
        f"Best 1-D rotation (handedness={handed}):",
        f"  mean |angular residual| = {mean_abs_resid_deg:.1f}°  "
        f"(chance≈90°, perfect=0°)",
        "",
        "Radial structure (does r ≈ saturation/lightness?):",
        f"  corr(r_latent, sat)       = {rho_r_sat:+.3f}",
        f"  corr(r_latent, value)     = {rho_r_val:+.3f}",
        f"  corr(r_latent, luminance) = {rho_r_lum:+.3f}",
        "",
        "Interpretation:",
        ("  Manifold IS closed-loop in hue" if rho_circ > 0.5
         else "  Manifold is only partially circular in hue"),
        ("  r encodes saturation (cone-like)" if rho_r_sat > 0.3
         else "  r is NOT cleanly saturation"),
    ]
    a6.text(0.0, 1.0, "\n".join(lines), family="monospace",
            fontsize=9.5, va="top")

    fig.suptitle("auto_08: polar projection of U_3d latent — is hue a circle?",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    print(f"[auto_08] saved {OUT}")
    print(f"[auto_08] circ_corr={rho_circ:+.3f}  mean|resid|={mean_abs_resid_deg:.1f}°  "
          f"r/sat={rho_r_sat:+.3f}  r/val={rho_r_val:+.3f}")


if __name__ == "__main__":
    main()
