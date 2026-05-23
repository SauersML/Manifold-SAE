"""auto_exp_14: per-template color signal-to-noise (option rr).

Motivation
----------
auto_exp_08 (template-OOD CV) gave a *fold-aggregate* view of how well a
held-out template's color signal lives on the manifold learned from the
other 27 templates. But it doesn't tell us *which* templates are the
clean color carriers vs the noisy ones — and that per-template ranking
is what we want if we ever drop templates (or weight them) for
production harvesting at L40.

This experiment computes a per-template ANOVA-style F-statistic on the
PC-64 representation of the (949 colors × 28 templates) panel:

  For template t and PC index k, view the 949 within-template values
  Z_t[c, k] as one realization per color, and the 27 sibling templates'
  values as the "noise" cloud around each color's centroid. Then:

    signal_t[k] = (1/(C-1)) Σ_c (Z_t[c, k] - Z_t[·, k])^2
                = between-color variance of template t at PC k
    noise_t[k]  = (1/(C·26)) Σ_c Σ_{s≠t} (Z_s[c, k] - μ_{¬t}[c, k])^2
                = pooled within-color residual variance after removing
                  the *other-templates* centroid μ_{¬t}[c, k]
    F_t[k]      = signal_t[k] / noise_t[k]

Aggregating across PCs:
  - F_trace_t = (Σ_k signal_t[k]) / (Σ_k noise_t[k])    "trace-F"
  - F_geo_t   = exp( mean_k log F_t[k] )                "geo-mean-F"

The trace-F is the most natural color-SNR per template. Templates with
F >> 1 carry color identity above their cross-template noise; F ~ 1
means template t lives inside the inter-template scatter.

We also report:
  - Pearson correlation between per-template F_trace and the per-template
    R^2 you'd get by predicting that template's per-color centroid from
    the OTHER 27 templates' centroid via a single linear scalar
    (well-defined since both are 949-vectors per PC, and is the
    simplest "is template t close to consensus?" probe).
  - The per-template residual norm from auto_exp_13's residual matrix is
    NOT recomputed (different question); we stay in raw PC-64 here.

Cheap: zero server traffic, pure numpy, ~seconds on existing X_L40.npy.

NO Gaussian RBF. NO length_scale on Duchon. (No Duchon fit at all here —
this experiment characterizes the *data* prior to manifold fitting.)

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_14_template_snr.{json,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_14_template_snr.json"
OUT_PNG = OUT_DIR / "auto_exp_14_template_snr.png"

N_TEMPLATES = 28
N_PCS = 64


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, H = X.shape
    assert N % N_TEMPLATES == 0, f"N={N} not divisible by {N_TEMPLATES}"
    n_colors = N // N_TEMPLATES
    # Row layout per run.log + auto_exp_13: color-major
    #   c_idx = repeat(arange(n_colors), N_TEMPLATES)
    #   t_idx = tile  (arange(N_TEMPLATES), n_colors)
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}  n_templates={N_TEMPLATES}",
          flush=True)

    # ---- centroids (per color), normalize, fixed top-PCA basis (match exp_13) ----
    centroids = np.zeros((n_colors, H), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma
    Cn = Cn - Cn.mean(0, keepdims=True)
    _, s_full, Vt_full = np.linalg.svd(Cn, full_matrices=False)
    V_topK = Vt_full[:N_PCS]
    evr_topK = (s_full ** 2 / (s_full ** 2).sum())[:N_PCS]
    print(f"[pca] fixed top-{N_PCS} EVR sum = {evr_topK.sum():.3f}", flush=True)

    # Project every sample through the same normalization + PCA basis.
    Xn = (X - mu) / sigma
    Xn = Xn - Cn.mean(0, keepdims=True)
    Z_all = Xn @ V_topK.T                       # (N, N_PCS)

    # Reshape into a (C, T, K) panel.
    K = N_PCS
    panel = np.zeros((n_colors, N_TEMPLATES, K), dtype=np.float64)
    for c in range(n_colors):
        rows = (c_idx == c)
        Zc = Z_all[rows]                        # (T, K) — already in t order
        # Confirm order: t_idx[rows] should equal arange(N_TEMPLATES).
        assert np.array_equal(t_idx[rows], np.arange(N_TEMPLATES))
        panel[c] = Zc

    # ---- core decomposition ----
    # For each template t:
    #   signal_t[k] = Var_c panel[c, t, k]   (between-color spread inside t)
    #   noise_t[k]  = mean_c Var_{s != t} panel[c, s, k]
    #                = mean_c [ (1/26) Σ_{s≠t} (panel[c,s,k] - μ_{¬t}[c,k])^2 ]
    # F_t[k] = signal_t[k] / noise_t[k]
    signal = panel.var(axis=0, ddof=1)          # (T, K)
    # Per-color mean across all 28 templates and sum:
    sum_all = panel.sum(axis=1)                 # (C, K)
    sumsq_all = (panel ** 2).sum(axis=1)        # (C, K)
    noise = np.zeros((N_TEMPLATES, K), dtype=np.float64)
    for t in range(N_TEMPLATES):
        sum_not = sum_all - panel[:, t, :]      # (C, K)  sum over 27 templates
        mean_not = sum_not / (N_TEMPLATES - 1)
        sumsq_not = sumsq_all - panel[:, t, :] ** 2
        # Σ (x - m)^2 = Σ x^2 - n m^2  with n = 27
        ss_not = sumsq_not - (N_TEMPLATES - 1) * mean_not ** 2
        var_not_c = ss_not / (N_TEMPLATES - 2)   # unbiased per-color noise var
        noise[t] = var_not_c.mean(axis=0)        # average over colors

    F = signal / np.maximum(noise, 1e-30)        # (T, K)
    f_trace = signal.sum(axis=1) / np.maximum(noise.sum(axis=1), 1e-30)  # (T,)
    f_geo = np.exp(np.log(np.maximum(F, 1e-30)).mean(axis=1))            # (T,)
    f_median_pc = np.median(F, axis=1)                                   # (T,)

    print(f"[snr ] F_trace  min={f_trace.min():.2f}  "
          f"median={float(np.median(f_trace)):.2f}  max={f_trace.max():.2f}",
          flush=True)
    print(f"[snr ] F_geo    min={f_geo.min():.2f}  "
          f"median={float(np.median(f_geo)):.2f}  max={f_geo.max():.2f}",
          flush=True)

    # Consensus alignment: per template, predict its per-color value at each PC
    # from the OTHER 27 templates' per-color centroid via a single scalar slope
    # (centered). R^2_t[k] = corr(panel[:,t,k], μ_{¬t}[·,k])^2.
    r2_consensus = np.zeros((N_TEMPLATES, K), dtype=np.float64)
    for t in range(N_TEMPLATES):
        mean_not = (sum_all - panel[:, t, :]) / (N_TEMPLATES - 1)  # (C, K)
        a = panel[:, t, :] - panel[:, t, :].mean(0, keepdims=True)
        b = mean_not - mean_not.mean(0, keepdims=True)
        num = (a * b).sum(0)
        den = np.sqrt((a * a).sum(0) * (b * b).sum(0)).clip(min=1e-30)
        r = num / den
        r2_consensus[t] = r * r
    # Variance-weighted aggregate R^2 per template (weighted by signal[t,k]).
    w = signal / signal.sum(axis=1, keepdims=True).clip(min=1e-30)
    r2_weighted = (r2_consensus * w).sum(axis=1)                          # (T,)

    # Correlation between F_trace and weighted consensus R^2 across templates.
    a = f_trace - f_trace.mean()
    b = r2_weighted - r2_weighted.mean()
    corr_FT_R2 = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()).clip(min=1e-30))
    print(f"[snr ] corr(F_trace, R^2_consensus_weighted) = {corr_FT_R2:+.3f}",
          flush=True)

    # Rankings (descending = better templates first).
    order_F = np.argsort(f_trace)[::-1]
    order_R = np.argsort(r2_weighted)[::-1]
    print("[rank] top-5 by F_trace  : "
          + " ".join(f"t{int(t)}({f_trace[t]:.1f})" for t in order_F[:5]),
          flush=True)
    print("[rank] bot-5 by F_trace  : "
          + " ".join(f"t{int(t)}({f_trace[t]:.1f})" for t in order_F[-5:]),
          flush=True)
    print("[rank] top-5 by R^2_cons : "
          + " ".join(f"t{int(t)}({r2_weighted[t]:.3f})" for t in order_R[:5]),
          flush=True)

    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors": int(n_colors), "n_templates": int(N_TEMPLATES),
            "n_pcs": int(N_PCS),
        },
        "f_trace": f_trace.tolist(),
        "f_geo_mean_over_pcs": f_geo.tolist(),
        "f_median_over_pcs": f_median_pc.tolist(),
        "r2_consensus_weighted": r2_weighted.tolist(),
        "rank_by_f_trace_desc": [int(x) for x in order_F.tolist()],
        "rank_by_r2_consensus_desc": [int(x) for x in order_R.tolist()],
        "stats": {
            "f_trace_min": float(f_trace.min()),
            "f_trace_median": float(np.median(f_trace)),
            "f_trace_max": float(f_trace.max()),
            "corr_F_trace_vs_r2_consensus_weighted": corr_FT_R2,
        },
        "interpretation": (
            "Per-template color SNR. F_trace_t > 1 means template t's "
            "between-color variance (signal) exceeds its leave-one-out "
            "cross-template noise. High F_trace + high R^2_consensus = a "
            "clean color carrier that agrees with the other templates; "
            "low F_trace = template is dominated by template-specific noise."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) per-template F_trace (sorted)
    ax = axes[0, 0]
    o = order_F
    ax.bar(np.arange(N_TEMPLATES), f_trace[o],
           color="steelblue", edgecolor="k", lw=0.4)
    ax.axhline(1.0, color="firebrick", ls="--", lw=0.8, label="F=1 (signal=noise)")
    ax.set_xticks(np.arange(N_TEMPLATES))
    ax.set_xticklabels([str(int(t)) for t in o], fontsize=7, rotation=90)
    ax.set_xlabel("template id (sorted by F_trace)")
    ax.set_ylabel("F_trace = Σ_k signal_k / Σ_k noise_k")
    ax.set_title(f"Per-template color SNR  (median={float(np.median(f_trace)):.1f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (2) per-template variance-weighted consensus R^2
    ax = axes[0, 1]
    ax.bar(np.arange(N_TEMPLATES), r2_weighted[o],
           color="darkorange", edgecolor="k", lw=0.4)
    ax.set_xticks(np.arange(N_TEMPLATES))
    ax.set_xticklabels([str(int(t)) for t in o], fontsize=7, rotation=90)
    ax.set_xlabel("template id (sorted by F_trace)")
    ax.set_ylabel("variance-weighted R^2 vs leave-one-out consensus")
    ax.set_title("Per-template agreement with other 27 templates")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")

    # (3) scatter F_trace vs R^2_consensus
    ax = axes[1, 0]
    ax.scatter(f_trace, r2_weighted, s=40, c="seagreen", edgecolor="k", lw=0.5)
    for t in range(N_TEMPLATES):
        ax.annotate(str(t), (f_trace[t], r2_weighted[t]),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("F_trace (color SNR)")
    ax.set_ylabel("R^2 vs consensus (weighted)")
    ax.set_title(f"corr = {corr_FT_R2:+.3f}\n"
                 "upper-right = clean color carriers")
    ax.grid(alpha=0.3)

    # (4) F per PC heatmap (templates sorted by F_trace, log scale)
    ax = axes[1, 1]
    im = ax.imshow(np.log10(np.maximum(F[o], 1e-3)),
                   aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("PC index"); ax.set_ylabel("template (sorted by F_trace, desc)")
    ax.set_yticks(np.arange(N_TEMPLATES))
    ax.set_yticklabels([str(int(t)) for t in o], fontsize=6)
    ax.set_title("log10(F) per (template, PC)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(F)")

    fig.suptitle(
        f"auto_exp_14: per-template color SNR  (PC-{N_PCS}, "
        f"{n_colors} colors × {N_TEMPLATES} templates)\n"
        f"F_trace range [{f_trace.min():.1f}, {f_trace.max():.1f}]  "
        f"median={float(np.median(f_trace)):.1f}  "
        f"corr(F_trace, R^2_cons)={corr_FT_R2:+.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
