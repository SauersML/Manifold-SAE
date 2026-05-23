"""
auto_46: Variance partition (ANOVA) of per-prompt PCA features.

Idea (yyyyy): For each top-K principal component of the per-prompt residuals
X (N=26572 = 949 colors × 28 templates), decompose the variance into:

    SS_total = SS_color  +  SS_template  +  SS_residual

where SS_color and SS_template are between-group sums of squares of the
marginal means (one-observation-per-cell two-way layout), and SS_residual
captures the color×template interaction plus noise (they are confounded
in a one-cell-per-(c,t) design).

Plots:
  (1) Stacked bar of variance shares for the top-32 PCs.
  (2) Aggregate pie of pooled variance shares across the top-K PCs.
  (3) Per-PC color-fraction vs template-fraction scatter (top-K PCs).
  (4) Cumulative "color-explained" variance vs "template-explained" vs
      residual along the PCA spectrum (top-K PCs, weighted by EVR).

No Gaussian RBF, no length_scale on Duchon; only PCA + arithmetic (mean,
group-means). Pure numpy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG = RUN_DIR / "auto_46.png"


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    with RESULTS.open() as f:
        res = json.load(f)

    pl = res["per_layer"]["L40"]
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float32)        # (K, D)
    mu = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)        # (1, D) per-color centroid mean
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)  # (1, D)
    evr = np.asarray(pl["explained_variance_ratio_topK"], dtype=np.float64)  # (K,)
    K, D = Vt.shape
    print(f"[pca] K={K} D={D}", flush=True)

    n_t = len(res["templates"])
    n_c = len(res["color_axes_per_color_index"]["R"])
    print(f"[layout] n_colors={n_c} n_templates={n_t} -> N={n_c*n_t}", flush=True)

    X = np.load(X_PATH, mmap_mode="r")  # (Nfull, D) float32
    N = n_c * n_t
    assert X.shape[0] >= N, (X.shape, N)
    assert X.shape[1] == D, (X.shape, D)

    # Project to top-K PCs using the same normalization as the GAM pipeline.
    # We'll stream in chunks of ~2k rows to avoid materializing (26572,7168) at once.
    chunk = 2048
    Z = np.zeros((N, K), dtype=np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Xc = (Xc - mu) / sigma
        Z[i:j] = Xc @ Vt.T
    print(f"[project] Z {Z.shape}", flush=True)

    # Row r corresponds to color = r // n_t, template = r % n_t  (per harvest
    # loop in color_manifold_gam.py: outer color, inner template).
    color_idx = np.repeat(np.arange(n_c), n_t)
    templ_idx = np.tile(np.arange(n_t), n_c)
    assert color_idx.shape[0] == N

    # ANOVA decomposition per PC
    # grand mean (over all N rows)
    gm = Z.mean(axis=0, keepdims=True)                                # (1, K)
    Z0 = Z - gm

    # color marginal means: (n_c, K)
    col_means = np.zeros((n_c, K), dtype=np.float64)
    np.add.at(col_means, color_idx, Z0)
    col_means /= n_t                                                  # balanced design

    # template marginal means: (n_t, K)
    tmpl_means = np.zeros((n_t, K), dtype=np.float64)
    np.add.at(tmpl_means, templ_idx, Z0)
    tmpl_means /= n_c

    SS_total = (Z0.astype(np.float64) ** 2).sum(axis=0)               # (K,)
    SS_color = n_t * (col_means ** 2).sum(axis=0)                     # (K,)
    SS_templ = n_c * (tmpl_means ** 2).sum(axis=0)                    # (K,)
    SS_resid = SS_total - SS_color - SS_templ                         # interaction + noise
    SS_resid = np.maximum(SS_resid, 0.0)

    frac_color = SS_color / np.maximum(SS_total, 1e-12)
    frac_templ = SS_templ / np.maximum(SS_total, 1e-12)
    frac_resid = SS_resid / np.maximum(SS_total, 1e-12)

    # Sanity report on the first 8 PCs
    for k in range(min(8, K)):
        print(f"[PC{k+1:02d}] evr={evr[k]:.4f}  "
              f"color={frac_color[k]:.3f}  templ={frac_templ[k]:.3f}  "
              f"resid={frac_resid[k]:.3f}", flush=True)

    # Pooled (EVR-weighted) variance shares across all K PCs
    w = evr / evr.sum()
    pooled = np.array([
        (w * frac_color).sum(),
        (w * frac_templ).sum(),
        (w * frac_resid).sum(),
    ])
    print(f"[pooled-evr-weighted] color={pooled[0]:.3f}  "
          f"templ={pooled[1]:.3f}  resid={pooled[2]:.3f}", flush=True)

    # ---------------- Plot ----------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # (1) stacked bar of top-32 PCs
    ax1 = fig.add_subplot(gs[0, 0])
    K_show = min(32, K)
    idx = np.arange(K_show)
    c, t, r = frac_color[:K_show], frac_templ[:K_show], frac_resid[:K_show]
    ax1.bar(idx, c, color="#1f77b4", label="color")
    ax1.bar(idx, t, bottom=c, color="#ff7f0e", label="template")
    ax1.bar(idx, r, bottom=c + t, color="#999999", label="residual (interaction+noise)")
    ax1.set_xlabel("PC index (1..32)")
    ax1.set_ylabel("variance share")
    ax1.set_title("Per-PC ANOVA variance partition (top 32 PCs)")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", fontsize=8)

    # (2) pooled pie
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(
        pooled,
        labels=[f"color\n{pooled[0]*100:.1f}%",
                f"template\n{pooled[1]*100:.1f}%",
                f"residual\n{pooled[2]*100:.1f}%"],
        colors=["#1f77b4", "#ff7f0e", "#999999"],
        startangle=90,
        wedgeprops=dict(linewidth=1, edgecolor="white"),
    )
    ax2.set_title(f"Pooled variance shares (EVR-weighted, K={K} PCs)")

    # (3) scatter: color-frac vs templ-frac, size ~ EVR
    ax3 = fig.add_subplot(gs[1, 0])
    sizes = 30 + 4000 * (evr / evr.max())
    sc = ax3.scatter(frac_color, frac_templ, s=sizes,
                     c=np.arange(K), cmap="viridis", alpha=0.75,
                     edgecolor="black", linewidth=0.4)
    # annotate first 6
    for k in range(min(6, K)):
        ax3.annotate(f"PC{k+1}", (frac_color[k], frac_templ[k]),
                     fontsize=8, ha="left", va="bottom")
    lim = max(frac_color.max(), frac_templ.max()) * 1.05
    ax3.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax3.set_xlabel("color variance fraction")
    ax3.set_ylabel("template variance fraction")
    ax3.set_title("Per-PC color- vs template- variance (marker size ∝ EVR)")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)
    cb = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cb.set_label("PC index")

    # (4) cumulative variance attributed to each source along the spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    cum_c = np.cumsum(evr * frac_color)
    cum_t = np.cumsum(evr * frac_templ)
    cum_r = np.cumsum(evr * frac_resid)
    xs = np.arange(1, K + 1)
    ax4.plot(xs, cum_c, color="#1f77b4", label="color")
    ax4.plot(xs, cum_t, color="#ff7f0e", label="template")
    ax4.plot(xs, cum_r, color="#999999", label="residual")
    ax4.plot(xs, np.cumsum(evr), color="black", linestyle="--",
             linewidth=0.8, label="total EVR")
    ax4.set_xlabel("# top PCs included")
    ax4.set_ylabel("cumulative variance share (of full activation var)")
    ax4.set_title("Cumulative ANOVA variance along the PCA spectrum")
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "auto_46 — Variance partition of L40 PCA features: "
        f"color vs template vs interaction  (N={N} = {n_c}c × {n_t}t, K={K})",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)

    # tiny json summary
    summary = {
        "pooled_evr_weighted": {
            "color": float(pooled[0]),
            "template": float(pooled[1]),
            "residual": float(pooled[2]),
        },
        "per_pc_frac_color_top16": [float(x) for x in frac_color[:16]],
        "per_pc_frac_template_top16": [float(x) for x in frac_templ[:16]],
        "per_pc_frac_residual_top16": [float(x) for x in frac_resid[:16]],
        "evr_top16": [float(x) for x in evr[:16]],
        "K": int(K), "n_colors": int(n_c), "n_templates": int(n_t),
    }
    (RUN_DIR / "auto_46.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {RUN_DIR/'auto_46.json'}", flush=True)


if __name__ == "__main__":
    main()
