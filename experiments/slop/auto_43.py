"""auto_43 [idea nnnnn]: hue-color ring synthesis.

Take the best *supervised* RGB→PC spec (L_joint_rgb, a 3D Duchon spline
on the RGB cube, R²~0.234) refit on the full filtered centroid set,
then synthesize predictions for 360 evenly-spaced hues at S=V=1 (the
pure-saturation chromatic ring of HSV).

For each predicted centroid we plot:
  (a) the trajectory through the top-3 PC space, coloured by hue
      (so the curve is literally the colour wheel rendered in PC₁₂₃);
  (b) PC1, PC2, PC3 vs hue angle (1D unrolling — should be roughly
      sinusoidal if the model has learned a circular hue code);
  (c) the same ring traced on top of the actual centroid scatter in
      PC1/PC2 (do the predicted "pure" hues sit near a 1-parameter
      curve through observed colour data?);
  (d) distance from each predicted ring point to its nearest actual
      centroid, plotted vs hue — peaks tell you which hues the model
      *invents* (no real swatch nearby) vs. which are well-supported.

This complements auto_42's structural alignment story: that showed how
much variance is RGB-reachable; this shows what trajectory the model
*actually traces* when you sweep one parameter (hue) through it.

Constraints respected: PCA + Duchon (no length_scale) + k-NN distance.
No Gaussian RBF.
"""
from __future__ import annotations

import colorsys
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402
from color_manifold_gam import (  # noqa: E402
    duchon_basis_radial,
    lattice_centers,
    reml_fit,
)

N_T = 28
RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_43.png"
RESULTS = RUN / "results.json"
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
N_HUE = 360


def main():
    with open(RESULTS) as f:
        res = json.load(f)
    L = res["per_layer"]["L40"]
    Vt = np.asarray(L["Vt_topK"], dtype=np.float64)
    mu = np.asarray(L["mu"], dtype=np.float64)
    sigma = np.asarray(L["sigma"], dtype=np.float64)
    r2_macro = L["specs"]["L_joint_rgb"]["r2_macro_mean"]
    K, D = Vt.shape

    X_full = load_harvest(HARVEST)
    n_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_raw * N_T]
    centroids_all = X_full.reshape(n_raw, N_T, -1).mean(1)
    colors_all = load_xkcd_colors()[:n_raw]
    _, kept = filter_colors(colors_all)
    centroids = centroids_all[kept]
    colors = [colors_all[i] for i in kept]
    n_c = len(colors)
    assert centroids.shape[1] == D

    Xn = (centroids - mu[None, :]) / np.maximum(sigma[None, :], 1e-8)
    Z = Xn @ Vt.T                                    # (n_c, K) PC scores
    rgb = np.array([(r, g, b) for _, r, g, b in colors],
                   dtype=np.float64) / 255.0
    print(f"[auto_43] n_c={n_c}  K={K}  D={D}", flush=True)

    # --- refit L_joint_rgb on full data ---
    centers_rgb = lattice_centers(5)
    Phi_tr, P = duchon_basis_radial(rgb, centers_rgb)
    print(f"[auto_43] Phi {Phi_tr.shape}  P {P.shape}  centers {centers_rgb.shape}",
          flush=True)
    B, lam = reml_fit(Phi_tr, Z, P, init_log_lambda=0.0)
    print(f"[auto_43] reml log-λ = {np.log(lam):+.3f}", flush=True)
    Z_train_pred = Phi_tr @ B
    train_r2 = 1.0 - ((Z - Z_train_pred) ** 2).sum() / (Z ** 2).sum()
    print(f"[auto_43] full-data training R² (macro) = {train_r2:+.4f}  "
          f"(held-out from results.json = {r2_macro:+.4f})", flush=True)

    # --- synthesize 360-hue ring at S=V=1 (pure spectrum) ---
    hue = np.linspace(0.0, 1.0, N_HUE, endpoint=False)
    rgb_ring = np.stack([hsv_to_rgb(np.stack([h * np.ones(1),
                                              np.ones(1), np.ones(1)], 1))[0]
                          for h in hue])
    Phi_ring, _ = duchon_basis_radial(rgb_ring, centers_rgb)
    Z_ring = Phi_ring @ B                            # (N_HUE, K)
    print(f"[auto_43] ring score range PC1=[{Z_ring[:,0].min():+.2f},"
          f"{Z_ring[:,0].max():+.2f}]  PC2=[{Z_ring[:,1].min():+.2f},"
          f"{Z_ring[:,1].max():+.2f}]  PC3=[{Z_ring[:,2].min():+.2f},"
          f"{Z_ring[:,2].max():+.2f}]", flush=True)

    # --- nearest-actual-centroid distance per hue (in PC space, top-8) ---
    K_DIST = 8
    diff = Z_ring[:, None, :K_DIST] - Z[None, :, :K_DIST]
    d2 = (diff ** 2).sum(-1)
    nn_dist = np.sqrt(d2.min(1))
    nn_idx = d2.argmin(1)

    # Hue swatch colours for plotting (use the synthesised pure-spectrum RGB)
    ring_rgb_plot = rgb_ring.copy()

    # --- plot ---
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # (a) 3D trajectory in PC₁₂₃, coloured by hue
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    # actual centroid cloud, lightly
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],
               c=rgb, s=8, alpha=0.18, edgecolor="none", depthshade=False)
    # ring
    ax.plot(Z_ring[:, 0], Z_ring[:, 1], Z_ring[:, 2],
            color="k", lw=0.5, alpha=0.5)
    ax.scatter(Z_ring[:, 0], Z_ring[:, 1], Z_ring[:, 2],
               c=ring_rgb_plot, s=22, edgecolor="black", lw=0.25,
               depthshade=False)
    # close the loop
    ax.plot([Z_ring[-1, 0], Z_ring[0, 0]],
            [Z_ring[-1, 1], Z_ring[0, 1]],
            [Z_ring[-1, 2], Z_ring[0, 2]], color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("(a) predicted ring through PC₁₂₃\n"
                 "360 pure-saturation hues (S=V=1) via L_joint_rgb",
                 fontsize=10)

    # (b) PC1, PC2, PC3 vs hue angle
    ax = fig.add_subplot(gs[0, 1])
    hue_deg = hue * 360.0
    for j, lbl, col in [(0, "PC1", "#1f77b4"),
                         (1, "PC2", "#d62728"),
                         (2, "PC3", "#2ca02c")]:
        ax.plot(hue_deg, Z_ring[:, j], color=col, lw=1.6, label=lbl)
    # colour-strip along the bottom showing hue
    ymin, ymax = ax.get_ylim()
    strip_h = (ymax - ymin) * 0.04
    for k in range(N_HUE):
        ax.axvspan(hue_deg[k] - 0.5, hue_deg[k] + 0.5,
                    ymin=0, ymax=strip_h / (ymax - ymin),
                    color=ring_rgb_plot[k], alpha=1.0, lw=0)
    ax.set_xlabel("hue (deg)")
    ax.set_ylabel("predicted PC score")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_title("(b) per-PC traces vs hue angle\n"
                 "should look ~sinusoidal if hue is encoded ~circularly",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (c) PC1 vs PC2 — ring drawn on real centroid cloud
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(Z[:, 0], Z[:, 1], c=rgb, s=18, alpha=0.55,
               edgecolor="black", lw=0.15)
    ax.plot(np.r_[Z_ring[:, 0], Z_ring[:1, 0]],
            np.r_[Z_ring[:, 1], Z_ring[:1, 1]],
            color="k", lw=0.8, alpha=0.7)
    ax.scatter(Z_ring[:, 0], Z_ring[:, 1],
               c=ring_rgb_plot, s=42, edgecolor="black", lw=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(c) predicted hue ring (large dots)\n"
                 "overlaid on real xkcd centroids (small)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    # (d) distance to nearest actual centroid, vs hue
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(hue_deg, nn_dist, color="gray", lw=1.0)
    ax.scatter(hue_deg, nn_dist, c=ring_rgb_plot, s=22,
               edgecolor="black", lw=0.25)
    # mark globally-supported vs invented hues
    med = float(np.median(nn_dist))
    ax.axhline(med, color="black", ls="--", lw=0.6,
                label=f"median = {med:.2f}")
    worst = np.argmax(nn_dist)
    best = np.argmin(nn_dist)
    ax.annotate(f"worst-supported\nhue {hue_deg[worst]:.0f}°\n"
                 f"NN: {colors[nn_idx[worst]][0]}",
                 (hue_deg[worst], nn_dist[worst]),
                 fontsize=8, xytext=(8, -2), textcoords="offset points",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    ax.annotate(f"best\n{hue_deg[best]:.0f}° → {colors[nn_idx[best]][0]}",
                 (hue_deg[best], nn_dist[best]),
                 fontsize=8, xytext=(8, 10), textcoords="offset points",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xlabel("hue (deg)")
    ax.set_ylabel(f"distance to nearest centroid in PC₁..₈")
    ax.set_title("(d) hue support: distance to nearest real xkcd centroid\n"
                 "peaks = hues the model 'invents' with no nearby swatch",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"auto_43 · hue-ring synthesis through L_joint_rgb · "
        f"cogito L40 · n_c={n_c} · K={K} PCs · "
        f"held-out R²={r2_macro:+.3f}  full-fit R²={train_r2:+.3f} "
        f"· {N_HUE} pure-spectrum hues  [idea nnnnn]",
        fontsize=12, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
