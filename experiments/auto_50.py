"""
auto_50 — cumulative R² adding PCs one at a time (idea hhhhhh).

For RGB(+hue/sat/val + per-template one-hot) -> Z_top64 (the L40 PCA codes from
the GAM run), we ask: as we walk k = 1..K from PC1 to PC64, how much of the
*total* held-out residual sum-of-squares does our Ridge regression recover when
we are only allowed to predict the first k PCs (zero-imputing the rest)?

Compares three curves on the same axis:
  (1) cumulative PCA variance of Z itself (the upper bound: if our regression
      were perfect on the first k PCs, this is the R² we'd get),
  (2) cumulative held-out R² of Ridge fitted to all K PCs but only the first
      k predictions are used (rest zero),
  (3) ratio (2)/(1) = "fraction of available variance the regressor recovered".

A flat (2) curve past k* means later PCs are unpredictable from RGB+template,
even though they carry variance.  A curve (3) that *rises* with k means later
PCs are individually easier (per unit variance) to predict than early ones.

5-fold KFold over colors so a held-out color's 28 template rows are never seen
in training.  No Gaussian RBF.  No length_scale on Duchon.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
X_PATH   = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG  = RUN_DIR / "auto_50.png"
OUT_JSON = RUN_DIR / "auto_50.json"


def hsv_from_rgb(R, G, B):
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    v = mx
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0.0)
    h = np.zeros_like(R)
    safe = d > 1e-12
    rmax = (mx == R) & safe
    gmax = (mx == G) & safe & ~rmax
    bmax = (mx == B) & safe & ~rmax & ~gmax
    h_r = ((G - B) / np.maximum(d, 1e-12)) % 6.0
    h_g = ((B - R) / np.maximum(d, 1e-12)) + 2.0
    h_b = ((R - G) / np.maximum(d, 1e-12)) + 4.0
    h = np.where(rmax, h_r, h)
    h = np.where(gmax, h_g, h)
    h = np.where(bmax, h_b, h)
    return (h / 6.0) % 1.0, s, v


def main():
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    templates = res["templates"]
    pl = res["per_layer"]["L40"]
    Vt    = np.asarray(pl["Vt_topK"], dtype=np.float32)
    mu    = np.asarray(pl["mu"],     dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(pl["sigma"],  dtype=np.float32).reshape(1, -1)
    K = Vt.shape[0]
    R = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    G = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    B = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    n_c = R.size
    n_t = len(templates)
    N = n_c * n_t
    print(f"[layout] n_c={n_c} n_t={n_t} N={N} K={K}", flush=True)

    # ---- Stream X -> Z = standardised PCA codes ----
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, X.shape
    Z = np.zeros((N, K), dtype=np.float32)
    chunk = 2048
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Z[i:j] = ((np.asarray(X[i:j], dtype=np.float32) - mu) / sigma) @ Vt.T
    Z = Z.astype(np.float64)
    print(f"[project] Z {Z.shape}  Z.var_total={float((Z.var(axis=0)).sum()):.4f}", flush=True)

    color_idx = np.repeat(np.arange(n_c), n_t)
    templ_idx = np.tile  (np.arange(n_t), n_c)
    Rr = R[color_idx]; Gr = G[color_idx]; Bb = B[color_idx]
    h, s, v = hsv_from_rgb(Rr, Gr, Bb)
    feat_color = np.column_stack([
        Rr, Gr, Bb,
        np.sin(2*np.pi*h), np.cos(2*np.pi*h),
        s, v,
    ])
    T_onehot = np.eye(n_t, dtype=np.float64)[templ_idx]
    Phi = np.concatenate([feat_color, T_onehot], axis=1)
    print(f"[feat] Phi {Phi.shape}", flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(kf.split(np.arange(n_c)))

    alpha = 1.0   # tuned to be near-best in auto_49
    Zhat = np.zeros_like(Z)
    for fi, (tr_c, te_c) in enumerate(folds):
        tr_mask = np.isin(color_idx, tr_c)
        te_mask = ~tr_mask
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(Phi[tr_mask], Z[tr_mask])
        Zhat[te_mask] = mdl.predict(Phi[te_mask])
        ss_res_fold = float(((Z[te_mask] - Zhat[te_mask]) ** 2).sum())
        ss_tot_fold = float(((Z[te_mask] - Z[tr_mask].mean(axis=0)) ** 2).sum())
        print(f"[fold {fi}] full-K R²={1 - ss_res_fold/ss_tot_fold:.4f}", flush=True)

    # ---- Cumulative curves ----
    pc_mean = Z.mean(axis=0, keepdims=True)
    Zc = Z - pc_mean
    Zhc = Zhat - pc_mean

    # variance per PC and per-PC SS_res from the held-out predictions
    var_pc    = (Zc  ** 2).sum(axis=0)            # ss_tot per PC (vs global mean)
    ssres_pc  = ((Zc - Zhc) ** 2).sum(axis=0)     # ss_res per PC
    r2_pc     = 1.0 - ssres_pc / np.maximum(var_pc, 1e-12)

    var_total = float(var_pc.sum())
    cum_var_share   = np.cumsum(var_pc) / var_total      # upper bound on cum R² if perfect on first k
    # Cumulative regression R²: treat predictions for PCs > k as the global mean
    # (=> zero residual contribution from those PCs in the *numerator*, but they
    # still contribute their full variance to the denominator).
    cum_ssres = np.cumsum(ssres_pc) + (var_total - np.cumsum(var_pc))   # res from first-k + worst-case (=mean) for rest
    cum_r2_reg = 1.0 - cum_ssres / var_total
    # Per-PC efficiency: regression R² on PC k itself
    # Cumulative "efficiency" = variance-weighted average of per-PC R² over first k
    cum_var_explained_by_reg = np.cumsum(var_pc * r2_pc)
    cum_eff = cum_var_explained_by_reg / np.cumsum(var_pc)   # this is exactly cum_r2_reg / cum_var_share

    # Reference: cumulative variance share if we kept only first k PCs (ceiling)
    # plotted alongside cum_r2_reg.

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ks = np.arange(1, K + 1)
    ax = axes[0]
    ax.plot(ks, cum_var_share,  marker="o", ms=3, color="black",
            label="cum PCA var share (ceiling)")
    ax.plot(ks, cum_r2_reg,     marker="o", ms=3, color="#1f77b4",
            label=f"cum held-out R² of Ridge (α={alpha})")
    ax.fill_between(ks, cum_r2_reg, cum_var_share, color="#1f77b4", alpha=0.10)
    ax.set_xlabel("k (number of leading PCs predicted)")
    ax.set_ylabel("cumulative R² of Z (held-out, total-variance basis)")
    ax.set_title("Cumulative R² as we add PCs one at a time")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.bar(ks, r2_pc, color=np.where(r2_pc > 0, "#1f77b4", "#d62728"),
           edgecolor="black", linewidth=0.2)
    ax.axhline(0, color="black", linewidth=0.6)
    mean_r2 = float((var_pc * r2_pc).sum() / var_total)
    ax.set_xlabel("PC index k")
    ax.set_ylabel("per-PC held-out R²")
    ax.set_title(f"Per-PC R² (variance-weighted mean = {mean_r2:.3f})")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot(ks, cum_eff, marker="o", ms=3, color="#2ca02c",
            label="cum efficiency = cum_R²_reg / cum_var_share")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6,
               label="perfect-on-first-k upper bound")
    ax.set_xlabel("k")
    ax.set_ylabel("fraction of available variance recovered")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Efficiency: how well RGB+template explains the first k PCs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"auto_50 — cumulative R² adding PCs one at a time  "
        f"(L40, K={K}, n_colors={n_c}, n_templates={n_t}, 5-fold over colors)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "alpha": alpha,
        "K": int(K), "n_colors": int(n_c), "n_templates": int(n_t),
        "var_total": var_total,
        "var_pc":        var_pc.tolist(),
        "r2_pc":         r2_pc.tolist(),
        "cum_var_share": cum_var_share.tolist(),
        "cum_r2_reg":    cum_r2_reg.tolist(),
        "cum_eff":       cum_eff.tolist(),
        "variance_weighted_mean_per_pc_r2": mean_r2,
        "k_first_negative_r2": int(np.argmax(r2_pc < 0)) if np.any(r2_pc < 0) else -1,
        "k_for_50pct_cum_r2": int(np.searchsorted(cum_r2_reg, 0.5) + 1) if cum_r2_reg.max() >= 0.5 else -1,
        "k_for_90pct_of_ceiling": int(np.searchsorted(cum_eff, 0.9 * cum_eff.max()) + 1),
    }, indent=2))
    print(f"[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
