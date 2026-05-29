"""
auto_49: RGB -> Z_top64 via PLS regression vs Ridge (idea gggggg).

Does Partial Least Squares find a lower-dim latent regression that beats Ridge
when mapping RGB (+ poly-2 cross terms + cyclic hue + per-template one-hots)
into the L40 top-64 PCA codes?  Both models are evaluated in identical 5-fold
KFold-over-colors so a held-out color's 28 template rows are never seen during
fit.  We sweep PLS n_components in {2,4,8,16,32,48,64} and compare to Ridge at
alpha in {0.1,1,10,100}.

No Gaussian RBF.  No length_scale on Duchon.  Only sklearn (Ridge, PLSRegression)
+ KFold over colors + numpy.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG = RUN_DIR / "auto_49.png"
OUT_JSON = RUN_DIR / "auto_49.json"


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


def macro_r2(Z_true, Z_pred, train_mean):
    ss_res = float(((Z_true - Z_pred) ** 2).sum())
    ss_tot = float(((Z_true - train_mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot


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

    # ----- Stream-project X -> Z -----
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, X.shape
    chunk = 2048
    Z = np.zeros((N, K), dtype=np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Z[i:j] = ((np.asarray(X[i:j], dtype=np.float32) - mu) / sigma) @ Vt.T
    Z = Z.astype(np.float64)
    print(f"[project] Z {Z.shape}", flush=True)

    color_idx = np.repeat(np.arange(n_c), n_t)
    templ_idx = np.tile  (np.arange(n_t), n_c)
    Rr = R[color_idx]; Gr = G[color_idx]; Bb = B[color_idx]
    h, s, v = hsv_from_rgb(Rr, Gr, Bb)
    feat_color = np.column_stack([
        Rr, Gr, Bb,
        Rr*Gr, Rr*Bb, Gr*Bb,
        Rr*Rr, Gr*Gr, Bb*Bb,
        np.sin(2*np.pi*h), np.cos(2*np.pi*h),
        s, v,
    ])
    T_onehot = np.eye(n_t, dtype=np.float64)[templ_idx]
    Phi = np.concatenate([feat_color, T_onehot], axis=1)
    print(f"[feat] Phi {Phi.shape}", flush=True)

    pls_ncomp = [2, 4, 8, 16, 24, 32, 40]
    ridge_alpha = [0.1, 1.0, 10.0, 100.0]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(kf.split(np.arange(n_c)))

    # ----- Ridge sweep -----
    ridge_results = {a: {"fold_r2": [], "Zhat": np.zeros_like(Z)} for a in ridge_alpha}
    pls_results   = {k: {"fold_r2": [], "Zhat": np.zeros_like(Z)} for k in pls_ncomp}

    for fi, (tr_c, te_c) in enumerate(folds):
        tr_mask = np.isin(color_idx, tr_c)
        te_mask = ~tr_mask
        Phi_tr, Phi_te = Phi[tr_mask], Phi[te_mask]
        Z_tr,   Z_te   = Z[tr_mask],   Z[te_mask]
        train_mean = Z_tr.mean(axis=0)

        for a in ridge_alpha:
            mdl = Ridge(alpha=a, fit_intercept=True)
            mdl.fit(Phi_tr, Z_tr)
            yh = mdl.predict(Phi_te)
            ridge_results[a]["Zhat"][te_mask] = yh
            ridge_results[a]["fold_r2"].append(macro_r2(Z_te, yh, train_mean))

        for k in pls_ncomp:
            mdl = PLSRegression(n_components=k, scale=False, max_iter=500, tol=1e-6)
            mdl.fit(Phi_tr, Z_tr)
            yh = mdl.predict(Phi_te)
            pls_results[k]["Zhat"][te_mask] = yh
            pls_results[k]["fold_r2"].append(macro_r2(Z_te, yh, train_mean))

        rstr = " ".join(f"R(a={a})={ridge_results[a]['fold_r2'][-1]:.3f}" for a in ridge_alpha)
        pstr = " ".join(f"P(k={k})={pls_results[k]['fold_r2'][-1]:.3f}"   for k in pls_ncomp)
        print(f"[fold {fi}] {rstr}  {pstr}", flush=True)

    pc_mean = Z.mean(axis=0, keepdims=True)
    den_total = float(((Z - pc_mean) ** 2).sum())

    def overall_r2(Zhat):
        return 1.0 - float(((Z - Zhat) ** 2).sum()) / den_total

    ridge_overall = {a: overall_r2(ridge_results[a]["Zhat"]) for a in ridge_alpha}
    pls_overall   = {k: overall_r2(pls_results  [k]["Zhat"]) for k in pls_ncomp}

    # Per-PC R² for best models (vs Z baseline = global mean)
    def per_pc_r2(Zhat):
        ss_res = ((Z - Zhat) ** 2).sum(axis=0)
        ss_tot = ((Z - pc_mean) ** 2).sum(axis=0)
        return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    best_a = max(ridge_alpha, key=lambda a: ridge_overall[a])
    best_k = max(pls_ncomp,   key=lambda k: pls_overall  [k])
    print(f"[best] ridge alpha={best_a} R2={ridge_overall[best_a]:.4f}  pls k={best_k} R2={pls_overall[best_k]:.4f}", flush=True)
    r2pc_ridge = per_pc_r2(ridge_results[best_a]["Zhat"])
    r2pc_pls   = per_pc_r2(pls_results  [best_k]["Zhat"])

    # Per-color R² for best of each
    sq_res_r = (Z - ridge_results[best_a]["Zhat"]) ** 2
    sq_res_p = (Z - pls_results  [best_k]["Zhat"]) ** 2
    sq_tot   = (Z - pc_mean) ** 2
    r2c_r = np.zeros(n_c); r2c_p = np.zeros(n_c)
    for c in range(n_c):
        m = color_idx == c
        d = max(sq_tot[m].sum(), 1e-12)
        r2c_r[c] = 1.0 - sq_res_r[m].sum() / d
        r2c_p[c] = 1.0 - sq_res_p[m].sum() / d

    # ----- Plot -----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.27)

    ax = fig.add_subplot(gs[0, 0])
    ks = np.array(pls_ncomp, dtype=float)
    pls_mean = np.array([np.mean(pls_results[k]["fold_r2"]) for k in pls_ncomp])
    pls_std  = np.array([np.std (pls_results[k]["fold_r2"]) for k in pls_ncomp])
    ax.errorbar(ks, pls_mean, yerr=pls_std, marker="o", color="#1f77b4",
                linewidth=1.6, capsize=3, label="PLS (mean ± std over 5 folds)")
    for a in ridge_alpha:
        rm = np.mean(ridge_results[a]["fold_r2"])
        rs = np.std (ridge_results[a]["fold_r2"])
        ax.axhline(rm, linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Ridge α={a}: {rm:.3f}±{rs:.3f}")
    ax.set_xscale("log"); ax.set_xticks(pls_ncomp); ax.set_xticklabels(pls_ncomp)
    ax.set_xlabel("PLS n_components")
    ax.set_ylabel("macro R² (5-fold over colors)")
    ax.set_title("PLS vs Ridge: cross-validated R²")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    idx_pc = np.arange(K)
    ax.bar(idx_pc - 0.2, r2pc_ridge, width=0.4, color="#d62728", label=f"Ridge α={best_a}")
    ax.bar(idx_pc + 0.2, r2pc_pls  , width=0.4, color="#1f77b4", label=f"PLS k={best_k}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("PCA component index")
    ax.set_ylabel("per-PC held-out R²")
    ax.set_title(f"per-PC R² (best Ridge vs best PLS; mean Ridge={r2pc_ridge.mean():.3f}, PLS={r2pc_pls.mean():.3f})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    rgb_each = np.stack([R, G, B], axis=1).clip(0, 1)
    lo = float(min(r2c_r.min(), r2c_p.min())); hi = float(max(r2c_r.max(), r2c_p.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.scatter(r2c_r, r2c_p, c=rgb_each, s=12, edgecolor="black", linewidth=0.15, alpha=0.85)
    n_better = int((r2c_p > r2c_r).sum())
    ax.set_xlabel(f"per-color R² (Ridge α={best_a})")
    ax.set_ylabel(f"per-color R² (PLS k={best_k})")
    ax.set_title(f"per-color R²: PLS vs Ridge  ({n_better}/{n_c} colors PLS-better; "
                 f"mean Δ={float((r2c_p - r2c_r).mean()):+.4f})")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    diff = r2c_p - r2c_r
    ax.hist(diff, bins=50, color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(float(diff.mean()), color="red", linewidth=1.0,
               label=f"mean Δ={float(diff.mean()):+.4f}")
    ax.set_xlabel("ΔR² = R²_PLS − R²_Ridge (per color)")
    ax.set_ylabel("count")
    ax.set_title("distribution of per-color R² improvement")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"auto_49 — PLS vs Ridge for RGB(+poly2+hue+template)→Z_top{K}  "
        f"(5-fold over colors; best Ridge α={best_a}: R²={ridge_overall[best_a]:.4f}, "
        f"best PLS k={best_k}: R²={pls_overall[best_k]:.4f})",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "ridge_alpha":  ridge_alpha,
        "pls_ncomp":    pls_ncomp,
        "ridge_fold_r2": {str(a): ridge_results[a]["fold_r2"] for a in ridge_alpha},
        "pls_fold_r2":   {str(k): pls_results  [k]["fold_r2"] for k in pls_ncomp},
        "ridge_overall_r2": {str(a): ridge_overall[a] for a in ridge_alpha},
        "pls_overall_r2":   {str(k): pls_overall  [k] for k in pls_ncomp},
        "best_ridge_alpha": best_a,
        "best_pls_k":       best_k,
        "per_color_r2_mean_ridge": float(r2c_r.mean()),
        "per_color_r2_mean_pls":   float(r2c_p.mean()),
        "n_colors_pls_better":     int(n_better),
        "mean_delta_r2_per_color": float(diff.mean()),
        "K": int(K), "n_colors": int(n_c), "n_templates": int(n_t),
    }, indent=2))
    print(f"[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
