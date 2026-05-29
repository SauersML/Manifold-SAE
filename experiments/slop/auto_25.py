"""auto_25 (p): parallel coordinates of T_3d for canonical anchor colors.

Take the d=3 unsupervised latents T (949 colors x 3 latents). Identify a
small set of canonical anchor colors (the 11 basic color terms + a few
more) by nearest-neighbour in RGB to a hard-coded anchor palette, then
draw a parallel-coordinates plot of those anchors over the 3 latents.

Annotate each latent axis with its best-correlated supervised axis
(spearman rho) so the reader can interpret what each latent "means".
Also show the swatches of the chosen anchors as a side column.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_25.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_25.png"
OUT_JSON = RUN / "auto_25.json"

# Canonical anchor palette: (label, target_rgb in [0,1]).
ANCHORS = [
    ("red",     (1.00, 0.00, 0.00)),
    ("orange",  (1.00, 0.55, 0.00)),
    ("yellow",  (1.00, 1.00, 0.00)),
    ("green",   (0.00, 0.60, 0.00)),
    ("cyan",    (0.00, 1.00, 1.00)),
    ("blue",    (0.00, 0.00, 1.00)),
    ("purple",  (0.55, 0.00, 0.70)),
    ("magenta", (1.00, 0.00, 1.00)),
    ("pink",    (1.00, 0.65, 0.80)),
    ("brown",   (0.55, 0.30, 0.10)),
    ("black",   (0.00, 0.00, 0.00)),
    ("white",   (1.00, 1.00, 1.00)),
    ("grey",    (0.50, 0.50, 0.50)),
]


def main() -> None:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    U3 = L["unsupervised_full_data"]["d=3"]
    T = np.array(U3["T"])                       # (N, 3)
    N = T.shape[0]

    ca = d["color_axes_per_color_index"]
    R = np.array(ca["R"])
    G = np.array(ca["G"])
    B = np.array(ca["B"])
    rgb = np.stack([R, G, B], axis=1)           # (N, 3) in [0,1]

    # Nearest-neighbour in RGB to each anchor target.
    chosen_idx = []
    chosen_meta = []
    used = set()
    for name, tgt in ANCHORS:
        t = np.array(tgt)
        d2 = np.sum((rgb - t) ** 2, axis=1)
        order = np.argsort(d2)
        # avoid duplicates
        for i in order:
            if int(i) not in used:
                used.add(int(i))
                chosen_idx.append(int(i))
                chosen_meta.append({
                    "label": name,
                    "color_index": int(i),
                    "actual_rgb": rgb[i].tolist(),
                    "target_rgb": list(tgt),
                    "rgb_dist": float(np.sqrt(d2[i])),
                    "T_3d": T[i].tolist(),
                })
                break

    chosen_idx = np.array(chosen_idx)
    T_anchor = T[chosen_idx]                    # (K, 3)
    rgb_anchor = rgb[chosen_idx]                # (K, 3)
    labels = [m["label"] for m in chosen_meta]
    K = len(chosen_idx)

    # Z-score each latent (across all 949 colors) for visual fairness.
    Tz_full = (T - T.mean(axis=0, keepdims=True)) / (T.std(axis=0, keepdims=True) + 1e-9)
    Tz_anchor = Tz_full[chosen_idx]

    # Best-correlated supervised axis per latent (for axis annotation).
    bal = U3["best_axis_per_latent"]
    latent_labels = [
        f"latent {b['latent_idx']}\nbest: {b['best_axis']} (ρ={b['rho']:+.2f})"
        for b in bal
    ]

    # ----- Figure -----
    fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_sw = fig.add_subplot(gs[0, 1])

    xs = np.array([0, 1, 2])

    # Background: faint grey traces for all 949 colors.
    for i in range(N):
        ax.plot(xs, Tz_full[i], color=(0.75, 0.75, 0.75, 0.05), lw=0.5, zorder=1)

    # Anchor traces, coloured by their actual RGB.
    for k in range(K):
        c = tuple(np.clip(rgb_anchor[k], 0.02, 0.98))
        # outline (dark) for contrast on light traces
        ax.plot(xs, Tz_anchor[k], color="black", lw=3.2, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, Tz_anchor[k], color=c, lw=2.0,
                solid_capstyle="round", zorder=3,
                label=labels[k])
        # label at the right
        ax.annotate(labels[k], xy=(xs[-1], Tz_anchor[k, -1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, va="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=c, ec="black", lw=0.5, alpha=0.85))

    ax.set_xticks(xs)
    ax.set_xticklabels(latent_labels, fontsize=9)
    ax.set_ylabel("latent value (z-scored across all 949 colors)")
    ax.set_title(
        "auto_25 (p): parallel coordinates of T_3d for canonical anchor colors\n"
        "(grey = all 949 colors; coloured lines = nearest match to canonical anchor)",
        fontsize=11,
    )
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.1, xs[-1] + 0.55)

    # ----- Swatch column -----
    ax_sw.set_xlim(0, 1)
    ax_sw.set_ylim(0, K)
    ax_sw.invert_yaxis()
    ax_sw.axis("off")
    ax_sw.set_title("anchor swatches\n(actual nearest match)", fontsize=9)
    for k in range(K):
        c = tuple(np.clip(rgb_anchor[k], 0.0, 1.0))
        ax_sw.add_patch(Rectangle((0.02, k + 0.10), 0.30, 0.80,
                                  facecolor=c, edgecolor="black", lw=0.6))
        ax_sw.text(0.36, k + 0.50,
                   f"{labels[k]}\nrgb=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})",
                   va="center", ha="left", fontsize=7.5)

    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "anchors": chosen_meta,
        "latent_best_axis": bal,
        "T_mean": T.mean(axis=0).tolist(),
        "T_std": T.std(axis=0).tolist(),
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
