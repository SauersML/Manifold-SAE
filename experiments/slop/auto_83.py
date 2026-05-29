"""
auto_83 — Per-template (28) breakdown of held-out R^2 for predicting the 64
cogito-L40 PCs from color features.

Question this answers (not covered by auto_77..auto_82):
  Which prompt template gives the cleanest, most colour-explainable cogito
  activation? Templates that name an object whose colour competes with {x}
  (e.g. "{x} milk", "{x} stallion") should bleed; abstract templates ("a {x}
  pool of paint") should be cleaner.

Method:
  - Load harvest X (26572, 7168) = 949 colors x 28 templates (color-major).
  - Use cached mu / sigma / Vt_topK (K=64) from results.json to project each
    row to a 64-D PC code Z_full of shape (26572, 64).
  - For each template t in 0..27:
      take the 949 rows belonging to that template,
      build feature matrix F = [R G B, hue (cos/sin), sat, value, lum,
                                 RGB^2 pairwise, R*G, G*B, R*B]  (12-D),
      fit 5-fold ridge regression F -> Z[:, :64] (closed-form per-fold),
      hold-out predict, compute per-PC R^2, then macro-R^2 (mean across PCs).
  - Also compute, per-template, the "bulk R^2" using just the top-8 PCs
    (the meaningfully-explainable block per auto_77/81), to separate
    head-vs-tail signal-to-noise.

Plot: horizontal bar chart of 28 templates sorted by macro-R^2 (top-8 block),
overlaid with full-64 macro-R^2 as scatter. Template text on the y-axis
abbreviated.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_83.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_83.png"

N_FOLDS = 5
RIDGE_LAM = 1e-2
TOP_PCS = 8
RNG = np.random.default_rng(0)


def build_features(R, G, B, H, S, V, L):
    # 12-D feature block: linear + cyclic hue + low-order interactions
    Hc = np.cos(2 * np.pi * H)
    Hs = np.sin(2 * np.pi * H)
    feats = np.stack([
        np.ones_like(R), R, G, B, Hc, Hs, S, V, L,
        R * G, G * B, R * B,
    ], axis=1)
    return feats


def kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def ridge_cv_r2(F, Z, n_folds=5, lam=1e-2, rng=None):
    """Return per-PC R^2 (length P) averaged across folds via held-out preds."""
    n, P = Z.shape
    folds = kfold_indices(n, n_folds, rng)
    preds = np.zeros_like(Z)
    for fi in range(n_folds):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        Ft, Zt = F[tr], Z[tr]
        A = Ft.T @ Ft + lam * np.eye(Ft.shape[1])
        W = np.linalg.solve(A, Ft.T @ Zt)
        preds[te] = F[te] @ W
    # per-PC R^2 vs constant-mean baseline (mean fit per fold's train would
    # be overkill; use overall mean baseline which matches macro-R^2 conv.)
    ss_res = ((Z - preds) ** 2).sum(axis=0)
    ss_tot = ((Z - Z.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def abbrev(t, n=44):
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def main():
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    L = d["per_layer"]["L40"]
    Vt = np.asarray(L["Vt_topK"], dtype=np.float64)       # (K, D)
    mu = np.asarray(L["mu"], dtype=np.float64)            # (D,)
    sigma = np.asarray(L["sigma"], dtype=np.float64)      # (D,)
    K = Vt.shape[0]

    ax = d["color_axes_per_color_index"]
    R = np.asarray(ax["R"], dtype=np.float64)
    G = np.asarray(ax["G"], dtype=np.float64)
    B = np.asarray(ax["B"], dtype=np.float64)
    H = np.asarray(ax["hue"], dtype=np.float64)
    S = np.asarray(ax["sat"], dtype=np.float64)
    V = np.asarray(ax["value"], dtype=np.float64)
    Lm = np.asarray(ax["luminance"], dtype=np.float64)
    n_c = len(R)
    print(f"[meta] n_c={n_c}, n_t={n_t}, K={K}")

    X = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X.shape
    assert n_rows == n_c * n_t, (n_rows, n_c, n_t)

    F = build_features(R, G, B, H, S, V, Lm)  # (n_c, P_feat)

    macro_top = np.zeros(n_t)
    macro_full = np.zeros(n_t)
    per_pc_top = np.zeros((n_t, TOP_PCS))

    # Project per template (chunked to keep memory low).
    for t in range(n_t):
        # rows for color c at template t: row = c * n_t + t (color-major)
        rows = np.arange(n_c) * n_t + t
        Xt = np.asarray(X[rows], dtype=np.float64)  # (949, 7168)
        Zt = ((Xt - mu) / np.maximum(sigma, 1e-8)) @ Vt.T  # (949, 64)
        r2 = ridge_cv_r2(F, Zt, n_folds=N_FOLDS, lam=RIDGE_LAM,
                         rng=np.random.default_rng(0))
        macro_full[t] = r2.mean()
        macro_top[t] = r2[:TOP_PCS].mean()
        per_pc_top[t] = r2[:TOP_PCS]
        print(f"[t={t:02d}] R2_top{TOP_PCS}={macro_top[t]:+.3f}  "
              f"R2_full64={macro_full[t]:+.3f}  | {abbrev(templates[t], 60)}")

    order = np.argsort(macro_top)  # ascending
    labels = [abbrev(templates[t]) for t in order]
    y = np.arange(n_t)

    fig, ax = plt.subplots(1, 2, figsize=(15, 9),
                           gridspec_kw={"width_ratios": [3.2, 2.2]})

    a = ax[0]
    a.barh(y, macro_top[order], color="#4477aa", alpha=0.85,
           label=f"macro R$^2$ over top-{TOP_PCS} PCs")
    a.scatter(macro_full[order], y, color="#cc3311", s=24, zorder=5,
              label="macro R$^2$ over all 64 PCs")
    a.set_yticks(y)
    a.set_yticklabels(labels, fontsize=8)
    a.set_xlabel("held-out R$^2$ (5-fold ridge, features = RGB + cyclic hue + SV + lum + 3 interactions)")
    a.set_title("auto_83  Per-template colour-explainability of cogito-L40 PCs"
                f"\n(949 xkcd colours per template, K=64, ridge $\\lambda$={RIDGE_LAM})")
    a.axvline(0, color="k", lw=0.6)
    a.legend(loc="lower right", fontsize=9)
    a.grid(axis="x", alpha=0.3)

    # right: per-PC heatmap (sorted same order)
    b = ax[1]
    im = b.imshow(per_pc_top[order], aspect="auto", cmap="RdBu_r",
                  vmin=-0.5, vmax=0.5, interpolation="nearest")
    b.set_yticks(y)
    b.set_yticklabels([""] * n_t)
    b.set_xticks(np.arange(TOP_PCS))
    b.set_xticklabels([f"PC{i+1}" for i in range(TOP_PCS)], fontsize=8)
    b.set_title(f"per-PC R$^2$ (top-{TOP_PCS})")
    plt.colorbar(im, ax=b, fraction=0.04, pad=0.02, label="R$^2$")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[done] wrote {OUT}")

    # quick textual ranking
    print("\n[ranking by R2_top]")
    for rk, t in enumerate(order[::-1]):
        print(f"  #{rk+1:02d}  R2_top={macro_top[t]:+.3f}  R2_full={macro_full[t]:+.3f}  "
              f"{abbrev(templates[t], 70)}")


if __name__ == "__main__":
    main()
