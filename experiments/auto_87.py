"""auto_87: Per-color name-token-count vs prediction error.

Fresh angle — NOT covered by auto_77..86. Asks: does the GAM-class fit
predict short-named xkcd colors ("lemon") better than long-named ones
("pale dusty olive green")? If yes, the model is leaning on
name-template surface morphology more than on intrinsic chroma; if no,
the manifold genuinely encodes the perceptual concept regardless of
how flowery the prompt string is.

Method (using only artifacts already on disk — no new harvest):
  1. Load 949 standardized PC-16 centroids T0 via per_color_stats_mmap.
  2. Fit a 5-fold-CV ridge of HSV (with circular hue encoding sin/cos)
     onto each of the top-16 PCs. This is the cheapest fair stand-in
     for "best supervised spec" (auto_77 showed HSV-bases dominate the
     low-K leaderboard).
  3. Per-color residual norm e_i = ||T0_i - T0_hat_i|| (test-fold
     predictions only).
  4. Tokenize each xkcd color name on whitespace; count tokens (1..5).
  5. Plot: (a) per-token-count violin of e_i, (b) scatter of e_i vs
     n_tokens colored by xkcd-RGB, (c) top-/bottom-10 hardest/easiest
     named colors. Print Spearman r(n_tokens, e_i).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_87.png"

K_PCS = 16
N_FOLDS = 5
ALPHA = 1.0


def encode_hsv(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, 0]
    return np.column_stack([
        np.sin(2 * np.pi * h),
        np.cos(2 * np.pi * h),
        hsv[:, 1],
        hsv[:, 2],
        hsv[:, 1] * hsv[:, 2],
    ])


def main():
    t0 = time.time()
    print("[auto_87] per-color name-token-count vs prediction error")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    feats = encode_hsv(hsv)

    # 5-fold CV ridge per PC; collect out-of-fold predictions.
    T0_hat = np.zeros_like(T0)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(kf.split(T0)):
        m = Ridge(alpha=ALPHA, fit_intercept=True)
        m.fit(feats[tr], T0[tr])
        T0_hat[te] = m.predict(feats[te])
    resid = T0 - T0_hat
    e = np.linalg.norm(resid, axis=1)
    ss_res = (resid ** 2).sum()
    ss_tot = ((T0 - T0.mean(0)) ** 2).sum()
    r2_macro = 1.0 - ss_res / ss_tot
    print(f"[ridge] CV R²_macro={r2_macro:+.4f}  (HSV->PC16, K={K_PCS})")

    n_tokens = np.array([len(nm.split()) for nm in names], dtype=int)
    print(f"[tokens] counts: " +
          ", ".join(f"{c}:{int((n_tokens==c).sum())}" for c in sorted(set(n_tokens))))

    rho, p = spearmanr(n_tokens, e)
    print(f"[spearman] n_tokens vs residual_norm: rho={rho:+.4f} p={p:.2e}")

    # ---- plotting ----
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.1])

    # (a) violin per token-count
    ax = fig.add_subplot(gs[0, 0])
    groups = sorted(set(n_tokens.tolist()))
    data = [e[n_tokens == g] for g in groups]
    parts = ax.violinplot(data, positions=groups, showmedians=True,
                          widths=0.75)
    for pc_ in parts["bodies"]:
        pc_.set_alpha(0.55)
    for g, d in zip(groups, data):
        ax.scatter(np.full_like(d, g, dtype=float) +
                   np.random.uniform(-0.12, 0.12, size=d.shape),
                   d, s=4, alpha=0.35, color="black")
    ax.set_xlabel("name token count")
    ax.set_ylabel("test-fold residual norm  ||T_i - T̂_i||")
    ax.set_title(f"Residual by name length\n"
                 f"Spearman ρ = {rho:+.3f}  (p={p:.1e})")
    ax.set_xticks(groups)

    # (b) scatter colored by xkcd-RGB with jitter
    ax = fig.add_subplot(gs[0, 1])
    jit = np.random.uniform(-0.18, 0.18, size=n)
    ax.scatter(n_tokens + jit, e, c=np.clip(rgb, 0, 1), s=14,
               edgecolors="black", linewidths=0.15)
    # Theil-Sen-ish linear overlay using means per group
    gx = np.array(groups, dtype=float)
    gy = np.array([d.mean() for d in data])
    ax.plot(gx, gy, color="black", lw=1.5, marker="o",
            ms=6, label="group mean")
    ax.set_xlabel("name token count")
    ax.set_ylabel("residual norm")
    ax.set_title("Per-color residual vs n_tokens\n"
                 f"(CV R²_macro of HSV→PC{K_PCS} = {r2_macro:+.3f})")
    ax.legend(loc="upper left", fontsize=8)

    # (c) top-10 hardest + 10 easiest named colors as colored swatches
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(e)
    easy = order[:10]
    hard = order[-10:][::-1]
    rows = list(easy) + list(hard)
    for row_idx, i in enumerate(rows):
        y = -row_idx
        ax.add_patch(plt.Rectangle((0, y - 0.4), 0.8, 0.8,
                                    facecolor=np.clip(rgb[i], 0, 1),
                                    edgecolor="black", lw=0.4))
        ax.text(0.95, y, f"{names[i]}", va="center", fontsize=8)
        ax.text(5.6, y, f"e={e[i]:.2f}  nt={n_tokens[i]}",
                va="center", fontsize=8, family="monospace")
    ax.axhline(-9.5, color="red", lw=1.0, ls="--", alpha=0.7)
    ax.text(3.4, -9.5, " easiest ↑   hardest ↓ ",
            color="red", ha="center", va="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.85))
    ax.set_xlim(-0.2, 7.5)
    ax.set_ylim(-len(rows) - 0.5, 0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Top-10 easiest / hardest to predict")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"auto_87 — name-token-count vs L40 prediction error  "
        f"(N={n} xkcd colors, K_PCs={K_PCS}, HSV ridge, 5-fold CV)",
        fontsize=12)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
