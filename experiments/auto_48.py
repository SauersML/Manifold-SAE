"""
auto_48: Color-name length vs per-color R² (idea jj).

We fit a ridge-regression mapping  φ(R,G,B,hue_cyclic) -> Z_top64 (PCA codes
of the L40 residual stream) on the full N = 949 colors × 28 templates =
26 572 rows, using 5-fold KFold over colors (so a held-out color's 28 rows
are never seen during fit).  This is essentially the `L_joint_rgb_with_hue`
spec from the GAM grid (best supervised spec in results.json, R² ≈ 0.239).

For every held-out color c we compute its per-color R²:

  num_c = Σ_{t,k}        (Z_{c,t,k} - Ẑ_{c,t,k})²
  den_c = Σ_{t,k}        (Z_{c,t,k} - Z̄_{·,·,k}  )²
  R²_c  = 1 - num_c / den_c

then plot R²_c vs `len(color_name)` (in characters and in tokens/words),
overlay a linear (ridge) trend and report Pearson + Spearman correlations.

No Gaussian RBF.  No length_scale on Duchon.  Tools: numpy + scikit-learn
Ridge + KFold; xkcd color CSV for names.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD    = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG = RUN_DIR / "auto_48.png"
OUT_JSON= RUN_DIR / "auto_48.json"


def load_xkcd_names() -> list[str]:
    names: list[str] = []
    with XKCD.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("\t", 1)[0].strip()
            if name:
                names.append(name)
    return names


def hsv_from_rgb(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    h = (h / 6.0) % 1.0
    return h, s, v


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    templates: list[str] = res["templates"]
    pl = res["per_layer"]["L40"]
    Vt    = np.asarray(pl["Vt_topK"], dtype=np.float32)             # (K, D)
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

    # ----- Build per-row features.  Row r -> color c = r // n_t, template t = r % n_t. -----
    color_idx = np.repeat(np.arange(n_c), n_t)
    templ_idx = np.tile  (np.arange(n_t), n_c)
    Rr = R[color_idx]; Gr = G[color_idx]; Bb = B[color_idx]
    h, s, v = hsv_from_rgb(Rr, Gr, Bb)
    # Joint RGB feature block (centered+poly-2), plus cyclic hue, plus per-template one-hots
    # so that template-level mean offsets do not leak into the per-color residual.
    feat_color = np.column_stack([
        Rr, Gr, Bb,
        Rr*Gr, Rr*Bb, Gr*Bb,
        Rr*Rr, Gr*Gr, Bb*Bb,
        np.sin(2*np.pi*h), np.cos(2*np.pi*h),
        s, v,
    ])                                                                # (N, 13)
    T_onehot = np.eye(n_t, dtype=np.float64)[templ_idx]               # (N, n_t)
    Phi = np.concatenate([feat_color, T_onehot], axis=1)              # (N, 13+n_t)
    print(f"[feat] Phi {Phi.shape}", flush=True)

    # ----- 5-fold KFold over *colors* -----
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    Zhat = np.zeros_like(Z)
    fold_r2 = []
    for fi, (tr_c, te_c) in enumerate(kf.split(np.arange(n_c))):
        tr_mask = np.isin(color_idx, tr_c)
        te_mask = ~tr_mask
        mdl = Ridge(alpha=1.0, fit_intercept=True)
        mdl.fit(Phi[tr_mask], Z[tr_mask])
        Zhat[te_mask] = mdl.predict(Phi[te_mask])
        ss_res = float(((Z[te_mask] - Zhat[te_mask]) ** 2).sum())
        ss_tot = float(((Z[te_mask] - Z[tr_mask].mean(axis=0)) ** 2).sum())
        fold_r2.append(1.0 - ss_res / ss_tot)
        print(f"[fold {fi}] r2_global={fold_r2[-1]:.4f} (test colors={te_c.size})", flush=True)

    # ----- Per-color R² (held-out predictions only) -----
    # baseline: per-PC global mean over the entire dataset (matches macro convention well enough)
    pc_mean = Z.mean(axis=0, keepdims=True)
    sq_res = (Z - Zhat) ** 2
    sq_tot = (Z - pc_mean) ** 2
    num_per_color = np.zeros(n_c); den_per_color = np.zeros(n_c)
    for c in range(n_c):
        m = color_idx == c
        num_per_color[c] = sq_res[m].sum()
        den_per_color[c] = sq_tot[m].sum()
    r2_per_color = 1.0 - num_per_color / np.maximum(den_per_color, 1e-12)
    print(f"[r2_per_color] mean={r2_per_color.mean():.4f} median={np.median(r2_per_color):.4f}", flush=True)

    # ----- Name lengths -----
    names = load_xkcd_names()
    if len(names) > n_c:
        names = names[:n_c]
    elif len(names) < n_c:
        names = names + [f"color_{i}" for i in range(len(names), n_c)]
    char_len = np.array([len(n) for n in names], dtype=np.float64)
    word_len = np.array([len(re.findall(r"\S+", n)) for n in names], dtype=np.float64)

    # ----- Correlations -----
    r_pear_chr, p_pear_chr = pearsonr (char_len, r2_per_color)
    r_spr_chr , p_spr_chr  = spearmanr(char_len, r2_per_color)
    r_pear_wrd, p_pear_wrd = pearsonr (word_len, r2_per_color)
    r_spr_wrd , p_spr_wrd  = spearmanr(word_len, r2_per_color)
    print(f"[corr char_len] pearson={r_pear_chr:+.3f} (p={p_pear_chr:.2e})  spearman={r_spr_chr:+.3f}", flush=True)
    print(f"[corr word_len] pearson={r_pear_wrd:+.3f} (p={p_pear_wrd:.2e})  spearman={r_spr_wrd:+.3f}", flush=True)

    # ----- Plot -----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.2, 1.0],
                          hspace=0.32, wspace=0.25)

    # (top-left) scatter char_len vs R², colored by RGB
    ax = fig.add_subplot(gs[0, 0])
    rgb_each = np.stack([R, G, B], axis=1).clip(0, 1)
    ax.scatter(char_len + np.random.RandomState(0).uniform(-0.25, 0.25, size=n_c),
               r2_per_color, c=rgb_each, s=18, edgecolor="black", linewidth=0.15, alpha=0.9)
    # Linear trend
    coef = np.polyfit(char_len, r2_per_color, 1)
    xs = np.linspace(char_len.min(), char_len.max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="black", linewidth=1.2, linestyle="--",
            label=f"linear fit: slope={coef[0]:+.4f}/char")
    ax.axhline(r2_per_color.mean(), color="grey", linewidth=0.6, alpha=0.7,
               label=f"mean R²={r2_per_color.mean():.3f}")
    ax.set_xlabel("color-name length (characters)")
    ax.set_ylabel("per-color held-out R²")
    ax.set_title(f"name length vs R² (Pearson r={r_pear_chr:+.3f}, Spearman ρ={r_spr_chr:+.3f}; N={n_c})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    # (top-right) word count strip / box
    ax = fig.add_subplot(gs[0, 1])
    word_vals = sorted(np.unique(word_len).astype(int))
    box_data = [r2_per_color[word_len == w] for w in word_vals]
    bp = ax.boxplot(box_data, positions=word_vals, widths=0.55, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#cbd5e1", edgecolor="black", linewidth=0.6)
    rng = np.random.RandomState(1)
    for w in word_vals:
        ys = r2_per_color[word_len == w]
        xs = w + rng.uniform(-0.18, 0.18, size=ys.size)
        rgbs = rgb_each[word_len == w]
        ax.scatter(xs, ys, c=rgbs, s=10, edgecolor="black", linewidth=0.15, alpha=0.85)
    ax.set_xlabel("color-name length (words)")
    ax.set_ylabel("per-color held-out R²")
    ax.set_title(f"R² by word count (Pearson r={r_pear_wrd:+.3f}, Spearman ρ={r_spr_wrd:+.3f})")
    ax.grid(True, alpha=0.25, axis="y")

    # (bottom-left) top-10 best / worst predicted names as swatches
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(r2_per_color)
    worst = order[:10]
    best  = order[-10:][::-1]
    rows = list(best) + [None] + list(worst)
    ax.set_xlim(0, 6); ax.set_ylim(-1, len(rows))
    ax.invert_yaxis()
    for i, idx in enumerate(rows):
        if idx is None:
            ax.text(0.05, i, "— — — — — — — — — — — — — — — — — —", fontsize=8, va="center")
            continue
        ax.add_patch(plt.Rectangle((0.1, i - 0.4), 0.8, 0.8,
                                   facecolor=rgb_each[idx], edgecolor="black", linewidth=0.4))
        ax.text(1.05, i, f"{names[idx]}", fontsize=8, va="center")
        ax.text(3.8, i, f"R²={r2_per_color[idx]:+.3f}  ({int(char_len[idx])} char, {int(word_len[idx])} wrd)",
                fontsize=8, va="center", family="monospace")
    ax.axis("off")
    ax.set_title("top-10 best (above) / worst (below) predicted xkcd colors")

    # (bottom-right) running R² vs char_len (sorted)
    ax = fig.add_subplot(gs[1, 1])
    ord_chr = np.argsort(char_len)
    win = 40
    smooth = np.convolve(r2_per_color[ord_chr], np.ones(win)/win, mode="valid")
    chr_smooth = np.convolve(char_len     [ord_chr], np.ones(win)/win, mode="valid")
    ax.plot(chr_smooth, smooth, color="black", linewidth=1.4, label=f"rolling mean R² (w={win})")
    ax.scatter(char_len, r2_per_color, c=rgb_each, s=6, alpha=0.45, edgecolor="none")
    ax.axhline(r2_per_color.mean(), color="grey", linewidth=0.6, alpha=0.7,
               label=f"mean R²={r2_per_color.mean():.3f}")
    ax.set_xlabel("color-name length (chars, sorted)")
    ax.set_ylabel("R²")
    ax.set_title("rolling-mean R² as name length grows")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    fold_str = ", ".join(f"{r:.3f}" for r in fold_r2)
    fig.suptitle(
        f"auto_48 — color-name length vs per-color R²  "
        f"(L40, K={K}, ridge on RGB+poly2+cyclic-hue+template-onehots, "
        f"5-fold over colors; per-fold R² macro = [{fold_str}])",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "fold_r2_macro": fold_r2,
        "r2_per_color_mean": float(r2_per_color.mean()),
        "r2_per_color_median": float(np.median(r2_per_color)),
        "pearson_char_len_r2": [float(r_pear_chr), float(p_pear_chr)],
        "spearman_char_len_r2": [float(r_spr_chr ), float(p_spr_chr )],
        "pearson_word_len_r2": [float(r_pear_wrd), float(p_pear_wrd)],
        "spearman_word_len_r2": [float(r_spr_wrd ), float(p_spr_wrd )],
        "top10_best":  [{"name": names[int(i)], "r2": float(r2_per_color[int(i)])} for i in best],
        "top10_worst": [{"name": names[int(i)], "r2": float(r2_per_color[int(i)])} for i in worst],
        "n_colors": int(n_c), "n_templates": int(n_t), "K": int(K),
    }, indent=2))
    print(f"[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
