"""auto_26 (mmm): per-PC residual variance heatmap (spec x PC).

For each supervised spec, we have per-PC R^2 (mean over folds) for the first
64 PCs. Combined with each PC's explained-variance ratio, the residual
*share of total variance* contributed by PC k under spec s is

    resid_share[s, k] = (1 - max(0, r2[s, k])) * evr[k]

This shows, for each spec, where in PC-space the unexplained variance lives
(low PCs = head, high PCs = tail). Plot two panels:

  (left)  heatmap of resid_share over spec x PC (log color scale),
          rows sorted by macro R^2 descending.
  (right) bar chart of EVR (top row) and per-spec total residual share
          (sum across PCs) on the same y-axis as the heatmap rows.

Also overlay, per spec, the per-PC R^2 as a line on a twin panel below the
heatmap so you can directly see where each spec wins/loses.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_26.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_26.png"
OUT_JSON = RUN / "auto_26.json"


def main() -> None:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    evr = np.array(L["explained_variance_ratio_topK"])  # (K,)
    K = evr.shape[0]

    specs = L["specs"]
    names = [n for n, v in specs.items() if "r2_per_pc_mean" in v]
    r2_pc = np.array([specs[n]["r2_per_pc_mean"] for n in names])  # (S, K)
    r2_macro = np.array([specs[n]["r2_macro_mean"] for n in names])
    r2_macro_std = np.array(
        [specs[n].get("r2_macro_std", 0.0) for n in names]
    )

    # Clip negative R^2 to 0 for residual interpretation (a model worse than
    # mean leaves >100% residual on that PC; cap it so the heatmap stays
    # comparable).
    r2_clip = np.clip(r2_pc, 0.0, 1.0)
    resid_share = (1.0 - r2_clip) * evr[None, :]   # (S, K)
    total_resid = resid_share.sum(axis=1)           # (S,)

    # Sort specs by macro R^2 descending (best at top).
    order = np.argsort(-r2_macro)
    names_o = [names[i] for i in order]
    r2_macro_o = r2_macro[order]
    r2_macro_std_o = r2_macro_std[order]
    r2_pc_o = r2_pc[order]
    resid_o = resid_share[order]
    total_resid_o = total_resid[order]
    S = len(names_o)

    # ----- Figure -----
    fig = plt.figure(figsize=(15.5, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[6.0, 0.9, 0.25],
        height_ratios=[1.0, 2.6],
    )

    # Top: EVR per PC
    ax_evr = fig.add_subplot(gs[0, 0])
    ax_evr.bar(np.arange(K), evr, color="0.4", width=0.9)
    ax_evr.set_yscale("log")
    ax_evr.set_xlim(-0.5, K - 0.5)
    ax_evr.set_ylabel("EVR (log)")
    ax_evr.set_title(
        "auto_26 (mmm): residual variance share per spec x PC at L40\n"
        "resid_share[s,k] = (1 - clip(R^2[s,k],0,1)) * EVR[k]",
        fontsize=11,
    )
    ax_evr.set_xticks([])
    ax_evr.grid(True, axis="y", alpha=0.3, which="both")

    # Main heatmap
    ax = fig.add_subplot(gs[1, 0], sharex=ax_evr)
    vmin = max(1e-6, np.min(resid_o[resid_o > 0]) * 0.5)
    vmax = resid_o.max()
    im = ax.imshow(
        resid_o,
        aspect="auto",
        cmap="magma_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        extent=[-0.5, K - 0.5, S - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(S))
    ax.set_yticklabels(
        [f"{n}  (R2={r:.3f}+/-{s:.3f})"
         for n, r, s in zip(names_o, r2_macro_o, r2_macro_std_o)],
        fontsize=8.5,
    )
    ax.set_xlabel("PC index (0 = top eigenvector)")
    ax.set_xticks(np.arange(0, K, 4))

    # Side bar: total residual share per spec
    ax_b = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_b.barh(np.arange(S), total_resid_o, color="0.35", height=0.8)
    ax_b.invert_yaxis()
    ax_b.set_yticks([])
    ax_b.set_xlabel("sum resid_share\n(top-64 PCs)")
    ax_b.grid(True, axis="x", alpha=0.3)
    # annotate
    for i, v in enumerate(total_resid_o):
        ax_b.text(v, i, f"  {v:.3f}", va="center", fontsize=7.5)

    # Colorbar
    cax = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=cax, label="residual share (log)")

    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "spec_names_sorted": names_o,
        "r2_macro_sorted": r2_macro_o.tolist(),
        "total_resid_share_sorted": total_resid_o.tolist(),
        "evr_topK": evr.tolist(),
        "argmax_resid_pc_per_spec": {
            n: int(np.argmax(resid_o[i])) for i, n in enumerate(names_o)
        },
        "notes": (
            "resid_share weights (1-R^2_pc) by that PC's EVR, so the heatmap "
            "row sums to ~total unexplained fraction of the top-64 PC variance."
        ),
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
