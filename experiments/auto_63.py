"""auto_63: Color-PC vs template-PC EVR decomposition (idea kkkkkkk).

For each of the K=64 top PCs of the L40 representation, we ask:
  - how much of its variance comes from *between-color* shifts
    (the colour identity signal), vs
  - how much from *within-color* template variation
    (the template/wording noise).

Decomposition (per PC k):
    var_total(k)   = mean_{c,t} [ (Z[c,t,k] - mean_{c,t} Z)^2 ]
    var_between(k) = mean_c     [ (mean_t Z[c,t,k] - mean_{c,t} Z)^2 ]
    var_within(k)  = mean_{c,t} [ (Z[c,t,k] - mean_t Z[c,t,k])^2 ]
    var_total       = var_between + var_within   (exact, balanced design)

This gives an explained-variance-ratio (EVR) split per PC.  We rank
PCs by total variance (the published EVR), and overlay between/within
shares.  Then we also build *color-only* and *template-only* sub-PCAs
on Z (each ran on the (n_c, K) color-mean matrix and on the residual
(N, K) within-color matrix respectively) and plot their cumulative
EVR vs the joint PCA --- this tells us the intrinsic rank of "what
the model knows about color name" vs "what changes with wording".

Pure numpy + matplotlib.  Uses PCA basis from results.json (Vt_topK,
mu, sigma).  No Gaussian RBF, no Duchon, no kernel tricks.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_63.{json,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG  = RUN_DIR / "auto_63.png"
OUT_JSON = RUN_DIR / "auto_63.json"


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {RESULTS}")
    res = json.loads(RESULTS.read_text())
    pl = res["per_layer"]["L40"]
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float32)
    mu = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)
    evr_pub = np.asarray(pl["explained_variance_ratio_topK"], dtype=np.float64)
    K, D = Vt.shape

    n_t = len(res["templates"])
    R = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    n_c = R.shape[0]
    N = n_c * n_t
    print(f"[layout] n_colors={n_c} n_templates={n_t} K={K} D={D}")

    # Project X -> Z (N, K) using the published PCA basis.
    print(f"[load] X {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, (X.shape, N)
    Z = np.zeros((N, K), dtype=np.float32)
    chunk = 4096
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Xc = (Xc - mu) / sigma
        Z[i:j] = Xc @ Vt.T
    Z = Z.reshape(n_c, n_t, K).astype(np.float64)
    print(f"[project] Z {Z.shape}")

    # Per-PC variance decomposition (balanced design):
    grand = Z.mean(axis=(0, 1), keepdims=True)        # (1,1,K)
    color_mean = Z.mean(axis=1, keepdims=True)         # (n_c,1,K)
    var_total   = ((Z - grand) ** 2).mean(axis=(0, 1))             # (K,)
    var_between = ((color_mean - grand) ** 2).mean(axis=(0, 1))    # (K,)
    var_within  = ((Z - color_mean) ** 2).mean(axis=(0, 1))        # (K,)
    # numerical check
    drift = float(np.max(np.abs(var_total - (var_between + var_within))))
    print(f"[check] max |total - (between+within)| = {drift:.2e}")

    share_between = var_between / np.maximum(var_total, 1e-12)
    share_within  = 1.0 - share_between

    # Sub-PCAs on the (n_c, K) between matrix and the (N, K) within matrix.
    B = (color_mean[:, 0, :] - grand[0, 0, :])                     # (n_c, K)
    W = (Z - color_mean).reshape(-1, K)                            # (N, K)
    # eigenvalues of cov: just SVD
    s_b = np.linalg.svd(B, compute_uv=False)
    s_w = np.linalg.svd(W, compute_uv=False)
    # joint full-K PCA on Z (already PCs, so eigvals are var per axis):
    Zc_full = (Z - grand).reshape(-1, K)
    s_j = np.linalg.svd(Zc_full, compute_uv=False)

    def cum_evr(s):
        v = s ** 2
        return np.cumsum(v) / v.sum()

    cum_b = cum_evr(s_b)
    cum_w = cum_evr(s_w)
    cum_j = cum_evr(s_j)

    # How many subspace dims to reach 90% / 99% var?
    def first_at(c, thr):
        idx = np.searchsorted(c, thr) + 1
        return int(min(idx, c.size))

    summary = {
        "n_colors": int(n_c),
        "n_templates": int(n_t),
        "K": int(K),
        "total_variance_sum": float(var_total.sum()),
        "between_variance_sum": float(var_between.sum()),
        "within_variance_sum": float(var_within.sum()),
        "overall_between_share": float(var_between.sum() / var_total.sum()),
        "rank90_between": first_at(cum_b, 0.90),
        "rank99_between": first_at(cum_b, 0.99),
        "rank90_within":  first_at(cum_w, 0.90),
        "rank99_within":  first_at(cum_w, 0.99),
        "rank90_joint":   first_at(cum_j, 0.90),
        "rank99_joint":   first_at(cum_j, 0.99),
        "per_pc": {
            "var_total":   var_total.tolist(),
            "var_between": var_between.tolist(),
            "var_within":  var_within.tolist(),
            "share_between": share_between.tolist(),
        },
        "cum_evr_between_subpca": cum_b.tolist(),
        "cum_evr_within_subpca":  cum_w.tolist(),
        "cum_evr_joint_full":     cum_j.tolist(),
    }

    print(f"[stat] overall between-color share = "
          f"{summary['overall_between_share']*100:.1f}%")
    print(f"[stat] rank90 between/within/joint = "
          f"{summary['rank90_between']}/{summary['rank90_within']}/"
          f"{summary['rank90_joint']}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26)

    # (a) stacked bar per PC: between vs within share, sorted by total var
    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(-var_total)
    xs = np.arange(K)
    ax.bar(xs, share_between[order], color="#d95f02",
           label="between-color")
    ax.bar(xs, share_within[order], bottom=share_between[order],
           color="#7570b3", label="within-color (template)")
    ax.set_xlabel("PC rank (by total variance)")
    ax.set_ylabel("share of PC variance")
    ax.set_title("(a) Per-PC variance decomposition")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # (b) total variance per PC (published EVR) overlaid with between/within absolute
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(xs + 1, var_total[order], "k-", lw=1.5, label="total var")
    ax.semilogy(xs + 1, var_between[order], color="#d95f02",
                lw=1.2, label="between-color var")
    ax.semilogy(xs + 1, var_within[order], color="#7570b3",
                lw=1.2, label="within-color var")
    ax.set_xlabel("PC rank (1=largest)")
    ax.set_ylabel("variance (log)")
    ax.set_title("(b) Absolute variance per PC")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # (c) cumulative EVR for sub-PCAs
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.arange(1, cum_j.size + 1), cum_j, "k-", lw=1.6,
            label=f"joint (full Z)  90%@{summary['rank90_joint']}")
    ax.plot(np.arange(1, cum_b.size + 1), cum_b, color="#d95f02",
            lw=1.6, label=f"between-color sub-PCA  90%@{summary['rank90_between']}")
    ax.plot(np.arange(1, cum_w.size + 1), cum_w, color="#7570b3",
            lw=1.6, label=f"within-color sub-PCA  90%@{summary['rank90_within']}")
    ax.axhline(0.90, color="grey", ls=":", lw=0.8)
    ax.axhline(0.99, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative EVR")
    ax.set_title("(c) Cumulative EVR of color-only vs template-only sub-PCAs")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (d) ratio: between/within per PC (sorted by total var), and a
    #     reference line at the overall ratio.
    ax = fig.add_subplot(gs[1, 1])
    ratio = var_between[order] / np.maximum(var_within[order], 1e-12)
    overall_ratio = summary["between_variance_sum"] / max(
        summary["within_variance_sum"], 1e-12)
    ax.semilogy(xs + 1, ratio, "o-", color="#1b9e77",
                ms=3.5, lw=1, label="PC between/within ratio")
    ax.axhline(overall_ratio, color="k", ls="--", lw=1,
               label=f"overall = {overall_ratio:.2f}")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("PC rank (by total variance)")
    ax.set_ylabel("between / within (log)")
    ax.set_title("(d) Color-signal vs template-noise per PC")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"auto_63 - color-PC vs template-PC EVR | L40 | n_c={n_c} n_t={n_t} "
        f"K={K} | between share = "
        f"{summary['overall_between_share']*100:.1f}%",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
