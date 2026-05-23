"""
auto_56 - (llllll) per-PC explained-variance-ratio vs per-PC prediction R^2.

Question
--------
For the layer-40 residual stream of cogito, we have a top-64 PCA fit
(explained_variance_ratio per PC) and, for many specs (linear/poly/
joint over RGB, HSV, Lab, Oklab, LCh, plus k-NN baselines), a per-PC
R^2 from color-only predictors.

Is variance-explained correlated with predictability? In other words,
do high-EVR PCs carry color, or do they carry mostly template/style
information that's invisible to color-only predictors? Conversely, do
small-EVR PCs encode crisp, low-variance color directions (which
would mean color is *not* a top-variance axis here)?

Method (allow-listed primitives: PCA + linear/ridge from results)
-----------------------------------------------------------------
1. Load EVR_topK (64,) and r2_per_pc_mean for every spec.
2. For each PC k, compute:
     - EVR_k
     - best_R2_k = max over specs of r2_per_pc_mean[k]
     - mean_R2_k over a representative subset of specs
3. Plot:
   (a) Bar chart: EVR per PC overlaid with best-spec R^2 per PC.
   (b) Scatter: log10(EVR) vs best-spec R^2 + Spearman/Pearson corr.
   (c) Heatmap: PC index x spec, color = R^2 (top 32 PCs, all specs).
   (d) Cumulative "color-explained variance": sum_k EVR_k * R2_k vs k,
       which tells us how much of total residual-stream variance is
       linearly/parametrically explained by color across the top PCs.

Outputs
-------
PNG : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_56.png
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_56.json

Constraints satisfied: no Gaussian RBF used; no Duchon length_scale
set (Duchon not used here); uses only PCA + ridge results already in
results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
OUT_PNG = RUN_DIR / "auto_56.png"
OUT_JSON = RUN_DIR / "auto_56.json"


def main():
    d = json.loads(RESULTS.read_text())
    pl = d["per_layer"]["L40"]
    evr = np.asarray(pl["explained_variance_ratio_topK"], dtype=float)
    K = evr.shape[0]

    specs = pl["specs"]
    # Collect per-PC R^2 for *color-predictor* specs. Skip the
    # unsupervised reconstruction baselines (U_*) which trivially get
    # R^2 ~ 1 because they reconstruct the PCA target from PCA itself.
    spec_names, R2 = [], []
    for name, info in specs.items():
        if not isinstance(info, dict):
            continue
        if "r2_per_pc_mean" not in info:
            continue
        if name.startswith("U_"):
            continue  # unsupervised reconstructors, not color predictors
        r = np.asarray(info["r2_per_pc_mean"], dtype=float)
        if r.shape[0] != K:
            continue
        spec_names.append(name)
        R2.append(r)
    R2 = np.stack(R2, axis=0)  # (S, K)
    print(f"Loaded {len(spec_names)} specs, K={K} PCs")

    best_R2 = R2.max(axis=0)  # (K,)
    best_spec_idx = R2.argmax(axis=0)
    mean_R2 = R2.mean(axis=0)
    # A pinned reference: the best single overall spec (max macro R^2).
    macro = np.array(
        [specs[s]["r2_macro_mean"] for s in spec_names], dtype=float
    )
    top_spec_i = int(np.argmax(macro))
    top_spec_name = spec_names[top_spec_i]
    print(f"Top spec by macro R^2: {top_spec_name} = {macro[top_spec_i]:.3f}")

    # Correlations between EVR and predictability.
    # Use only PCs with EVR > 0 (always true here, but be safe).
    valid = evr > 0
    log_evr = np.log10(evr[valid])

    sp_rho_best, sp_p_best = spearmanr(evr[valid], best_R2[valid])
    pe_r_best, pe_p_best = pearsonr(log_evr, best_R2[valid])
    sp_rho_mean, _ = spearmanr(evr[valid], mean_R2[valid])

    # Cumulative color-explained variance per PC (clip negatives to 0).
    R2_pos = np.clip(best_R2, 0.0, None)
    cum_color_var = np.cumsum(evr * R2_pos)
    cum_total_var = np.cumsum(evr)

    # ---------------- plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) bars EVR + line best R^2
    ax = axes[0, 0]
    ks = np.arange(K)
    ax2 = ax.twinx()
    ax.bar(ks, evr, color="steelblue", alpha=0.75, label="EVR")
    ax2.plot(ks, best_R2, "o-", color="crimson",
             ms=3.5, lw=1.0, label="best-spec $R^2$")
    ax2.plot(ks, R2[top_spec_i], "s-", color="darkorange",
             ms=2.5, lw=0.7, alpha=0.85,
             label=f"{top_spec_name} $R^2$")
    ax2.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Explained variance ratio", color="steelblue")
    ax2.set_ylabel("per-PC $R^2$ (color predictors)", color="crimson")
    ax.set_title("(a) per-PC EVR (bars) vs per-PC color $R^2$ (lines)")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)

    # (b) scatter log10(EVR) vs best R^2
    ax = axes[0, 1]
    ax.scatter(log_evr, best_R2[valid], c=ks[valid], cmap="viridis",
               s=40, edgecolor="k", lw=0.4)
    # Annotate a few extreme points
    order = np.argsort(-best_R2[valid])
    for j in order[:5]:
        k = ks[valid][j]
        ax.annotate(f"PC{k}", (log_evr[j], best_R2[valid][j]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(r"$\log_{10}$ EVR")
    ax.set_ylabel("best-spec per-PC $R^2$")
    ax.set_title(
        f"(b) Spearman EVR-vs-best$R^2$ = {sp_rho_best:+.3f} "
        f"(p={sp_p_best:.2g})\n"
        f"Pearson logEVR-vs-best$R^2$ = {pe_r_best:+.3f} "
        f"(p={pe_p_best:.2g})\n"
        f"Spearman EVR-vs-mean$R^2$ = {sp_rho_mean:+.3f}"
    )
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("PC index")

    # (c) heatmap of R^2 across specs x PCs (top 32 PCs)
    ax = axes[1, 0]
    # Sort specs by macro R^2 descending so eye finds patterns.
    s_order = np.argsort(-macro)
    R2_sorted = R2[s_order][:, :32]
    names_sorted = [spec_names[i] for i in s_order]
    im = ax.imshow(R2_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-0.3, vmax=0.7, interpolation="nearest")
    ax.set_xlabel("PC index (0..31)")
    ax.set_yticks(np.arange(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=5)
    ax.set_title("(c) per-PC $R^2$ heatmap (specs sorted by macro $R^2$)")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # (d) cumulative variance: total vs color-explained
    ax = axes[1, 1]
    ax.plot(ks, cum_total_var, "-", color="steelblue",
            lw=1.6, label="cumulative EVR (total)")
    ax.plot(ks, cum_color_var, "-", color="crimson",
            lw=1.6, label=r"cumulative EVR $\times$ best $R^2$ (color)")
    ax.fill_between(ks, 0, cum_color_var, color="crimson", alpha=0.15)
    ax.fill_between(ks, cum_color_var, cum_total_var,
                    color="steelblue", alpha=0.12)
    ax.set_xlabel("PC index (cumulative through k)")
    ax.set_ylabel("cumulative variance fraction")
    frac = cum_color_var[-1] / cum_total_var[-1]
    ax.set_title(
        f"(d) color explains {cum_color_var[-1]:.3f} / "
        f"{cum_total_var[-1]:.3f} of top-{K} variance ({frac:.1%})"
    )
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "auto_56 (llllll): per-PC EVR vs per-PC color-prediction $R^2$ "
        "(cogito L40, top-64 PCA)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    # ---- json summary ----
    summary = {
        "n_specs": len(spec_names),
        "K": int(K),
        "spec_names": spec_names,
        "evr": evr.tolist(),
        "best_R2_per_pc": best_R2.tolist(),
        "best_spec_idx_per_pc": best_spec_idx.tolist(),
        "mean_R2_per_pc": mean_R2.tolist(),
        "top_macro_spec": top_spec_name,
        "top_macro_value": float(macro[top_spec_i]),
        "spearman_evr_vs_bestR2": float(sp_rho_best),
        "spearman_evr_vs_bestR2_p": float(sp_p_best),
        "pearson_logEVR_vs_bestR2": float(pe_r_best),
        "pearson_logEVR_vs_bestR2_p": float(pe_p_best),
        "spearman_evr_vs_meanR2": float(sp_rho_mean),
        "cum_color_variance_topK": float(cum_color_var[-1]),
        "cum_total_variance_topK": float(cum_total_var[-1]),
        "color_fraction_of_topK": float(frac),
        "top5_PCs_by_bestR2": [
            {"pc": int(k),
             "evr": float(evr[k]),
             "best_R2": float(best_R2[k]),
             "best_spec": spec_names[int(best_spec_idx[k])]}
            for k in np.argsort(-best_R2)[:5]
        ],
        "top5_PCs_by_evr_with_R2": [
            {"pc": int(k),
             "evr": float(evr[k]),
             "best_R2": float(best_R2[k]),
             "best_spec": spec_names[int(best_spec_idx[k])]}
            for k in np.argsort(-evr)[:5]
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_JSON}")
    print(f"Spearman EVR vs best R^2: {sp_rho_best:+.3f} (p={sp_p_best:.2g})")
    print(f"Color fraction of top-{K} variance: {frac:.1%}")


if __name__ == "__main__":
    main()
