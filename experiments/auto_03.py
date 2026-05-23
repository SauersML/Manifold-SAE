"""auto_03: per-PC R² heatmap across the supervised GAM zoo.

The macro R² reported in results.json collapses 64 PCs into a single
number, hiding *which* PCs each parameterization captures. Here we
break that out:

  * X = the 64 PCA components (ordered by explained variance, high → low)
  * Y = each supervised spec L_* in the zoo
  * cell = held-out r2_per_pc_mean for that (spec, PC) pair
  * adjacent panels: explained-variance bar per PC, and bar of macro R²
    per spec (for cross-checking)

What this reveals:
  - Are low-variance PCs (PC>20) actually carrying color information
    that the supervised fits can recover, or is color confined to the
    top few PCs?
  - Which parameterizations recover deep (high-index) PCs the linear
    baselines miss (e.g. tensor B-spline, Duchon joints)?
  - The diagonal-like falloff per spec tells us the "depth" each
    color basis reaches before noise dominates.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_03.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


RESULTS = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json")
OUT = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_03.png")


def main() -> int:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    specs = L["specs"]
    evr = np.array(L["explained_variance_ratio_topK"], dtype=float)
    K = len(evr)

    # Restrict to the supervised L_* zoo (linear / polynomial / kernel /
    # joint color-coordinate fits). Skip specs missing per-PC traces.
    spec_names = [
        s for s in specs.keys()
        if s.startswith("L_") and "r2_per_pc_mean" in specs[s]
    ]
    macro = np.array([specs[s]["r2_macro_mean"] for s in spec_names])
    per_pc = np.array([specs[s]["r2_per_pc_mean"] for s in spec_names])  # (S, K)

    # Sort specs by macro R² descending so the strongest models sit on top.
    order = np.argsort(-macro)
    spec_names = [spec_names[i] for i in order]
    macro = macro[order]
    per_pc = per_pc[order]

    S = len(spec_names)

    fig = plt.figure(figsize=(17, max(6, 0.42 * S + 3)))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[5.5, 1.0],
        height_ratios=[0.7, 5.5],
        hspace=0.06, wspace=0.04,
    )
    ax_evr = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_evr)
    ax_macro = fig.add_subplot(gs[1, 1], sharey=ax_hm)
    ax_topright = fig.add_subplot(gs[0, 1])
    ax_topright.axis("off")

    # ---- Top: explained variance per PC ----
    ax_evr.bar(np.arange(K), evr * 100, color="#777", width=0.85)
    ax_evr.set_ylabel("EVR  (%)", fontsize=9)
    ax_evr.set_yscale("log")
    ax_evr.tick_params(axis="x", labelbottom=False)
    ax_evr.set_xlim(-0.5, K - 0.5)
    ax_evr.grid(axis="y", linestyle=":", alpha=0.4)
    ax_evr.set_title(
        "Per-PC R² across the supervised GAM zoo  ·  cogito L40  ·  64 PCs of the 7168-d centroid space",
        fontsize=12, loc="left",
    )

    # ---- Heatmap ----
    # Diverging colormap pinned at 0 so negative R² (worse than mean) shows red.
    vmax = max(0.9, float(np.nanmax(per_pc)))
    vmin = min(-0.1, float(np.nanmin(per_pc)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax_hm.imshow(
        per_pc, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest",
    )
    ax_hm.set_yticks(np.arange(S))
    ax_hm.set_yticklabels(spec_names, fontsize=9)
    ax_hm.set_xticks(np.arange(0, K, 4))
    ax_hm.set_xticklabels([str(i) for i in range(0, K, 4)], fontsize=8)
    ax_hm.set_xlabel("PCA component index  (ordered by explained variance)", fontsize=10)
    cbar = fig.colorbar(
        im, ax=ax_hm, fraction=0.018, pad=0.01, orientation="vertical",
    )
    cbar.set_label("held-out R²  per PC", fontsize=9)

    # ---- Right: macro R² bars (cross-check) ----
    ax_macro.barh(np.arange(S), macro, color="#356d96", edgecolor="black", linewidth=0.4)
    ax_macro.set_xlabel("macro R²", fontsize=9)
    ax_macro.tick_params(axis="y", labelleft=False)
    ax_macro.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_macro.grid(axis="x", linestyle=":", alpha=0.4)
    ax_macro.set_xlim(min(0.0, macro.min() - 0.02), max(macro) * 1.05)
    for i, v in enumerate(macro):
        ax_macro.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5)

    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT}", flush=True)

    # Brief quantitative summary
    best_spec = spec_names[0]
    # PCs where the best spec achieves r2>0.5
    deep = int(np.sum(per_pc[0] > 0.5))
    deepest = int(np.where(per_pc[0] > 0.2)[0].max()) if (per_pc[0] > 0.2).any() else -1
    print(
        f"[summary] best spec = {best_spec}  macro={macro[0]:.3f}  "
        f"#PCs with r2>0.5: {deep}  deepest PC with r2>0.2: {deepest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
