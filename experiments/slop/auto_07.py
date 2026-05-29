"""auto_07: Visualize U_3d discovered latent T colored by actual RGB.

Question: do the 3 unsupervised latent axes (t1, t2, t3) discovered by the
nonlinear manifold fit align visually/semantically with R, G, B (or with
hue/value/sat)? We plot the 949 xkcd centroids in the (t1,t2,t3) latent
space, coloring each point by its true xkcd RGB. We also show three side
panels — one per latent axis pair — and one panel where points are colored
by hue only, to confirm the visual story matches the Spearman numbers
(latent_2 ↔ hue, ρ=0.41 being strongest).

Reads:  runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
Writes: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_07.png
"""
from __future__ import annotations

import json
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
OUT = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_07.png"


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
    val = np.array(d["color_axes_per_color_index"]["value"])[:n]
    sat = np.array(d["color_axes_per_color_index"]["sat"])[:n]

    rgb = np.stack([R, G, B], axis=1).clip(0, 1)
    # Pure-hue rgb (saturate hue, full value) for "rainbow" coloring panel
    rgb_hue = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])

    rhos = L["unsupervised_full_data"]["d=3"]["axis_to_latent_spearman"]
    rho_str = {
        ax: f"R={rhos[ax]['per_latent_rho'][0]:+.2f} "
            f"G={rhos[ax]['per_latent_rho'][1]:+.2f} "  # actually labels are per LATENT here
            f"B={rhos[ax]['per_latent_rho'][2]:+.2f}"
        for ax in ("R", "G", "B", "hue", "value", "sat", "luminance")
    }

    fig = plt.figure(figsize=(17, 11))

    # --- Big 3D scatter colored by true RGB
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.scatter(T[:, 0], T[:, 1], T[:, 2], c=rgb, s=14, alpha=0.9,
                 edgecolors="none")
    ax3d.set_xlabel("t1"); ax3d.set_ylabel("t2"); ax3d.set_zlabel("t3")
    ax3d.set_title("Discovered U_3d latent T, colored by true xkcd RGB\n"
                   "(949 colors, layer 40)")

    # --- 3D scatter colored by pure hue (rainbow)
    ax3h = fig.add_subplot(2, 3, 4, projection="3d")
    ax3h.scatter(T[:, 0], T[:, 1], T[:, 2], c=rgb_hue, s=14, alpha=0.9,
                 edgecolors="none")
    ax3h.set_xlabel("t1"); ax3h.set_ylabel("t2"); ax3h.set_zlabel("t3")
    ax3h.set_title("Same T, colored by HUE only (sat=val=1)\n"
                   "hue↔t2 ρ=+0.41 is strongest single latent↔axis")

    # --- 2D pair plots: each colored by true RGB
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(pairs):
        a = fig.add_subplot(2, 3, 2 + k if k < 2 else 6)
        a.scatter(T[:, i], T[:, j], c=rgb, s=18, alpha=0.9, edgecolors="none")
        a.set_xlabel(f"t{i+1}"); a.set_ylabel(f"t{j+1}")
        a.set_title(f"t{i+1} vs t{j+1}  (true RGB)")
        a.grid(alpha=0.25)

    # --- Spearman summary panel (replaces position 5)
    ax_text = fig.add_subplot(2, 3, 5)
    ax_text.axis("off")
    lines = ["Spearman ρ( latent_k , known_axis )", ""]
    lines.append(f"{'axis':<10}{'t1':>8}{'t2':>8}{'t3':>8}")
    for ax_name in ("R", "G", "B", "hue", "sat", "value", "luminance"):
        rr = rhos[ax_name]["per_latent_rho"]
        lines.append(f"{ax_name:<10}{rr[0]:+8.2f}{rr[1]:+8.2f}{rr[2]:+8.2f}")
    lines += ["", "Best per latent:"]
    for entry in L["unsupervised_full_data"]["d=3"]["best_axis_per_latent"]:
        lines.append(
            f"  t{entry['latent_idx']+1} → {entry['best_axis']:<5}"
            f" (ρ={entry['rho']:+.2f})"
        )
    lines += ["",
              "Verdict: latents are NOT a clean R/G/B rotation;",
              "the dominant alignment is t2 ↔ hue (cyclic).",
              "RGB cloud shows mixing — not three orthogonal RGB axes."]
    ax_text.text(0.0, 1.0, "\n".join(lines), family="monospace",
                 fontsize=10, va="top")

    fig.suptitle("auto_07: Does the unsupervised 3D manifold latent T align "
                 "with R, G, B?", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    print(f"[auto_07] saved {OUT}")


if __name__ == "__main__":
    main()
