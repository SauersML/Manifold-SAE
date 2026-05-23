"""auto_06: cumulative variance-weighted signal-recovery curves.

auto_03 plots per-PC R^2 as a heatmap, but it does not tell us how much
of the *total* centroid signal each spec recovers, because high-index
PCs carry far less variance than low-index ones. Here we weight each
PC's R^2 by its explained-variance ratio and accumulate:

    signal_recovered(k) = sum_{i<=k} EVR[i] * max(0, r2_per_pc[i])

The curve answers: "of the total variance in the centroid space, what
fraction does this spec actually predict, using only the first k PCs?"

The cumulative-EVR curve (sum_{i<=k} EVR[i]) is the perfect-prediction
ceiling. The vertical gap between a spec's curve and the ceiling at
k=64 is the variance the spec leaves on the table.

We pick a representative subset of the zoo (linear baselines in each
colorspace, top non-linear / Duchon variants, and the const-mean
floor) so the plot stays legible.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_06.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json")
OUT = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_06.png")


# Curated spec subset: linear baselines per colorspace, top joints, top
# polys, top Duchon variants, kNN, const floor. Keeps the plot legible.
HIGHLIGHT_SPECS = [
    ("L_const_mean",                 "const mean (floor)",       "#888888", "--"),
    ("L_lin_luminance",              "lin luminance (1d)",        "#bbbbbb", "-"),
    ("L_lin_rgb",                    "lin RGB",                   "#9ec7e8", "-"),
    ("L_lin_hsv",                    "lin HSV",                   "#7baed1", "-"),
    ("L_lin_lab",                    "lin Lab",                   "#356d96", "-"),
    ("L_lin_oklab",                  "lin Oklab",                 "#1d3f5c", "-"),
    ("L_poly_hsv",                   "poly2 HSV",                 "#f4c26b", "-"),
    ("L_poly4_hsv",                  "poly4 HSV",                 "#d98c2b", "-"),
    ("L_joint_lab",                  "joint Lab",                 "#7ec07e", "-"),
    ("L_joint_oklab",                "joint Oklab",               "#3d8c3d", "-"),
    ("L_tensor_bspline_rgb",         "tensor B-spline RGB",       "#c45a93", "-"),
    ("L_kernel_rbf_rgb",             "kernel RBF RGB",            "#7b3fa0", "-"),
    ("N_knn_lab_k10",                "kNN Lab k=10",              "#b25c5c", "-"),
    ("U_pca3_duchon_joint",          "U PCA-3 Duchon joint",      "#1f9d8f", "-"),
    ("U_pca8_duchon_add1d",          "U PCA-8 Duchon add",        "#0d5f55", "-"),
    ("U_pca_64d",                    "U PCA-64 (linear ceiling)", "#000000", ":"),
]


def main() -> int:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    specs = L["specs"]
    evr = np.array(L["explained_variance_ratio_topK"], dtype=float)
    K = len(evr)
    cum_evr = np.cumsum(evr)
    total_evr = cum_evr[-1]

    available = [(sid, lbl, col, ls) for (sid, lbl, col, ls) in HIGHLIGHT_SPECS
                 if sid in specs and "r2_per_pc_mean" in specs[sid]]
    missing = [s for (s, *_rest) in HIGHLIGHT_SPECS if s not in specs]
    if missing:
        print(f"[warn] missing specs skipped: {missing}", flush=True)

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [3.2, 1.0]},
    )

    # Ceiling: cumulative EVR (perfect prediction)
    xs = np.arange(1, K + 1)
    ax.plot(xs, cum_evr * 100, color="black", lw=1.8, ls="-",
            label=f"ceiling: cumulative EVR  (top-{K} = {total_evr*100:.1f}%)")
    ax.fill_between(xs, cum_evr * 100, color="black", alpha=0.04)

    final_recovery = []
    for sid, lbl, col, ls in available:
        rpc = np.array(specs[sid]["r2_per_pc_mean"], dtype=float)
        # Variance-weighted recovery, clipped at 0 (negative R^2 means
        # the spec is worse than predicting the mean: it adds no signal).
        contrib = evr * np.clip(rpc, 0.0, None)
        cum = np.cumsum(contrib)
        ax.plot(xs, cum * 100, color=col, lw=1.6, ls=ls, label=lbl)
        final_recovery.append((sid, lbl, col, float(cum[-1])))

    ax.set_xlabel("number of leading PCs included  (k)", fontsize=11)
    ax.set_ylabel("cumulative recovered centroid variance  (% of total)", fontsize=11)
    ax.set_title(
        "Cumulative variance-weighted signal recovery  ·  cogito L40\n"
        "sum_{i<=k} EVR[i] * max(0, r2_per_pc[i])  —  how much real centroid signal each spec predicts",
        fontsize=12, loc="left",
    )
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_xlim(1, K)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2, framealpha=0.92)

    # Right: final recovery bar (k=K), as fraction of ceiling
    final_recovery.sort(key=lambda x: x[3], reverse=True)
    labels = [r[1] for r in final_recovery]
    fracs_of_ceiling = [r[3] / total_evr for r in final_recovery]
    cols = [r[2] for r in final_recovery]
    y = np.arange(len(final_recovery))
    ax_bar.barh(y, fracs_of_ceiling, color=cols, edgecolor="black", linewidth=0.4)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=8.5)
    ax_bar.set_xlabel("recovered / ceiling  (k=64)", fontsize=10)
    ax_bar.set_xlim(0, max(fracs_of_ceiling) * 1.15)
    ax_bar.axvline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.4)
    for yi, fv in zip(y, fracs_of_ceiling):
        ax_bar.text(fv + 0.005, yi, f"{fv*100:.1f}%", va="center", fontsize=7.5)
    ax_bar.set_title("final recovery (k=64)", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT}", flush=True)

    # Console summary
    print("\n[summary] cumulative recovered variance @ k=64  (% of ceiling)")
    for sid, lbl, _col, recov in final_recovery:
        print(f"  {recov/total_evr*100:6.2f}%   {sid:32s}  {lbl}")
    # Where does each spec hit half of its final value?
    print("\n[summary] PCs needed to reach 50% / 90% of each spec's k=64 recovery")
    for sid, lbl, _col, ls in available:
        rpc = np.array(specs[sid]["r2_per_pc_mean"], dtype=float)
        cum = np.cumsum(evr * np.clip(rpc, 0.0, None))
        if cum[-1] <= 0:
            print(f"  {sid:32s}  (no positive recovery)")
            continue
        k50 = int(np.searchsorted(cum, 0.5 * cum[-1]) + 1)
        k90 = int(np.searchsorted(cum, 0.9 * cum[-1]) + 1)
        print(f"  {sid:32s}  k50={k50:>3d}  k90={k90:>3d}  final={cum[-1]*100:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
