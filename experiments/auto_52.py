"""
auto_52 — Prediction quality: common-named ("red") vs rare-named ("celadon") colors.

Idea (oooooo): Does the cogito residual-stream embedding encode colors that
have a common English name *better* than colors with only a niche/compound
name? If yes, the model's color geometry is shaped by linguistic frequency
(more text -> sharper, more accurate per-color representation). If no,
the geometry is name-agnostic and driven by the underlying perceptual axes.

Setup
-----
- Read the d=3 unsupervised GAM manifold T (949 colors x 3) from results.json.
  T is the cogito-derived low-D coordinate of each xkcd color (no Gaussian RBF,
  no Duchon length_scale set: we just consume the existing fit). T was learned
  unsupervised from cogito hidden states, so it carries no RGB information
  by construction.
- For each color compute leave-one-out (LOO) prediction of its (R,G,B)
  from T using k-NN (k=10, distance-weighted) — k-NN is on the allowed list
  and gives a per-color residual that doesn't smear globally like ridge would.
  This produces a per-color RGB error in [0, sqrt(3)].
- Bucket colors:
    common = single-token name that is one of the canonical English color
             words people learn as children (the 11 "Berlin & Kay basic
             color terms" plus a handful of universally-known extras).
    rare   = everything else (compound names "navy blue", obscure names
             "celadon", "puce", "vermilion", body-fluid words, etc.).
- Compare: mean per-color L2 error, with bootstrap 95% CIs; plus a swatch
  panel showing the 12 best-predicted and 12 worst-predicted colors so the
  reader can eyeball the qualitative split. Also a swarm scatter of error
  vs bucket with the common colors labelled.

Outputs
-------
PNG: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_52.png  (3 panels)
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_52.json (numeric summary)

Constraints respected: no Gaussian RBF used; no Duchon length_scale set
(no Duchon fit here at all); PCA optional (none needed in 3-D); k-NN allowed.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
XKCD     = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG  = RUN_DIR / "auto_52.png"
OUT_JSON = RUN_DIR / "auto_52.json"

# Berlin & Kay's 11 basic color terms + a few obvious universally-known
# English colors. Kept single-token and lowercase to match xkcd entries.
COMMON_BASIC = {
    "black", "white", "red", "green", "yellow", "blue",
    "brown", "purple", "pink", "orange", "grey", "gray",
    # extras virtually every English speaker knows:
    "tan", "beige", "cream", "navy", "magenta", "cyan",
    "violet", "indigo", "gold", "silver", "lime", "maroon",
    "olive", "teal", "turquoise",
}


def load_xkcd() -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # "name\t#rrggbb"
        name, hex_ = [p.strip() for p in s.split("\t")[:2]]
        hex_ = hex_.lstrip("#")
        if len(hex_) != 6:
            continue
        r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
        names.append(name)
        rgb.append([r / 255.0, g / 255.0, b / 255.0])
    return names, np.asarray(rgb, dtype=np.float64)


def loo_knn_predict(T: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    """Leave-one-out k-NN regression on T -> Y. Distance-weighted.

    Returns predictions of shape Y.shape. We query k+1 neighbors and drop
    the self-match (distance 0) to implement LOO without refitting.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(T)
    dist, idx = nn.kneighbors(T, return_distance=True)
    # drop self (first column)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    # inverse-distance weights with tiny floor to avoid div-by-zero
    w = 1.0 / np.maximum(dist, 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    Y_neigh = Y[idx]                          # (n, k, dy)
    pred = (w[:, :, None] * Y_neigh).sum(axis=1)
    return pred


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def is_common(name: str, basic: set[str]) -> bool:
    """True iff name is exactly one common English single-token color word."""
    tokens = name.lower().split()
    return len(tokens) == 1 and tokens[0] in basic


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    T = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb_used = np.column_stack([Rax, Gax, Bax])
    n_c = rgb_used.shape[0]
    assert T.shape == (n_c, 3)

    names, _xkcd_rgb_full = load_xkcd()
    # The GAM run uses the first n_c xkcd colors in file order (see
    # color_manifold_gam.py: colors = colors[:n_full_colors]).
    assert len(names) >= n_c
    names = names[:n_c]
    print(f"[align] using first {n_c} xkcd names; "
          f"max |rgb_axes - parsed_xkcd_rgb| = "
          f"{np.abs(rgb_used - _xkcd_rgb_full[:n_c]).max():.4f}")

    # Predict RGB from cogito 3-D manifold via LOO k-NN.
    K = 10
    pred = loo_knn_predict(T, rgb_used, k=K)
    err = np.linalg.norm(pred - rgb_used, axis=1)           # (n_c,)
    print(f"[knn] k={K}  mean LOO err={err.mean():.4f}  median={np.median(err):.4f}")

    # Bucket colors.
    common_mask = np.array([is_common(nm, COMMON_BASIC) for nm in names])
    n_common = int(common_mask.sum())
    n_rare = n_c - n_common
    print(f"[bucket] common={n_common}  rare={n_rare}")
    common_names = [names[i] for i in np.where(common_mask)[0]]
    print(f"[bucket] common names: {common_names}")

    err_common = err[common_mask]
    err_rare   = err[~common_mask]

    mc, lo_c, hi_c = bootstrap_ci(err_common)
    mr, lo_r, hi_r = bootstrap_ci(err_rare)
    diff_mean = mr - mc
    # bootstrap CI on the difference (rare - common)
    rng = np.random.default_rng(1)
    n_boot = 5000
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s_c = err_common[rng.integers(0, n_common, n_common)].mean()
        s_r = err_rare[rng.integers(0, n_rare, n_rare)].mean()
        diffs[b] = s_r - s_c
    dlo = float(np.quantile(diffs, 0.025))
    dhi = float(np.quantile(diffs, 0.975))
    print(f"[stats] mean err  common={mc:.4f} [{lo_c:.4f},{hi_c:.4f}]  "
          f"rare={mr:.4f} [{lo_r:.4f},{hi_r:.4f}]")
    print(f"[stats] diff (rare-common)={diff_mean:.4f}  95% CI [{dlo:.4f},{dhi:.4f}]")

    # Also: total R^2 in each bucket (treating RGB as 3-D target).
    def r2_block(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean(0)) ** 2))
        return 1.0 - ss_res / ss_tot
    r2_common = r2_block(rgb_used[common_mask], pred[common_mask])
    r2_rare   = r2_block(rgb_used[~common_mask], pred[~common_mask])
    r2_all    = r2_block(rgb_used, pred)
    print(f"[r2] common={r2_common:.3f}  rare={r2_rare:.3f}  all={r2_all:.3f}")

    # ---- Plot: 3 panels ---------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85],
                          width_ratios=[1.0, 1.0, 1.0],
                          hspace=0.35, wspace=0.25,
                          left=0.06, right=0.98, top=0.92, bottom=0.07)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_hist  = fig.add_subplot(gs[0, 1])
    ax_swatch_best = fig.add_subplot(gs[0, 2])
    ax_swatch_worst_common = fig.add_subplot(gs[1, :])

    # Panel A: per-color error strip, common labelled
    rng2 = np.random.default_rng(0)
    x_common = 0.0 + rng2.uniform(-0.15, 0.15, n_common)
    x_rare   = 1.0 + rng2.uniform(-0.15, 0.15, n_rare)
    bg_rgb_c = rgb_used[common_mask]
    bg_rgb_r = rgb_used[~common_mask]
    ax_strip.scatter(x_rare,   err_rare,   s=12, c=bg_rgb_r, alpha=0.55,
                     edgecolors="none", label=f"rare/compound (n={n_rare})")
    ax_strip.scatter(x_common, err_common, s=70, c=bg_rgb_c, alpha=0.95,
                     edgecolors="black", linewidths=0.6,
                     label=f"common basic (n={n_common})")
    # mean lines with CI
    ax_strip.errorbar([0.0], [mc], yerr=[[mc - lo_c], [hi_c - mc]],
                      fmt="_", color="black", capsize=6, lw=2,
                      markersize=24)
    ax_strip.errorbar([1.0], [mr], yerr=[[mr - lo_r], [hi_r - mr]],
                      fmt="_", color="black", capsize=6, lw=2,
                      markersize=24)
    ax_strip.set_xticks([0.0, 1.0])
    ax_strip.set_xticklabels(["common", "rare"])
    ax_strip.set_ylabel(f"LOO k-NN ({K}) RGB error (L2)")
    ax_strip.set_title(f"Per-color cogito->RGB prediction error\n"
                       f"diff (rare-common)={diff_mean:+.4f}  "
                       f"95% CI [{dlo:+.4f},{dhi:+.4f}]")
    # Label a handful of common-color dots with their names.
    common_idx = np.where(common_mask)[0]
    for xi, ci in zip(x_common, common_idx):
        ax_strip.annotate(names[ci], (xi, err[ci]),
                          fontsize=7, alpha=0.85,
                          xytext=(4, 2), textcoords="offset points")
    ax_strip.set_xlim(-0.5, 1.5)
    ax_strip.legend(loc="upper right", fontsize=8)

    # Panel B: overlaid histograms
    bins = np.linspace(0, max(err.max(), 0.5), 36)
    ax_hist.hist(err_rare,   bins=bins, alpha=0.55, color="#777777",
                 density=True, label=f"rare (n={n_rare})")
    ax_hist.hist(err_common, bins=bins, alpha=0.75, color="#d62728",
                 density=True, label=f"common (n={n_common})")
    ax_hist.axvline(mr, color="#444444", lw=1.5, ls="--",
                    label=f"rare mean={mr:.3f}")
    ax_hist.axvline(mc, color="#d62728", lw=1.5, ls="--",
                    label=f"common mean={mc:.3f}")
    ax_hist.set_xlabel("LOO RGB error (L2)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"Error distributions\nR^2: common={r2_common:.3f}  rare={r2_rare:.3f}")
    ax_hist.legend(fontsize=8)

    # Helper to render a row/grid of swatches.
    def draw_swatches(ax, indices: Iterable[int], title: str, ncol: int = 6):
        idxs = list(indices)
        nrow = (len(idxs) + ncol - 1) // ncol
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for k, ci in enumerate(idxs):
            r, c = divmod(k, ncol)
            ax.add_patch(Rectangle((c + 0.05, r + 0.05), 0.9, 0.6,
                                   facecolor=tuple(rgb_used[ci]),
                                   edgecolor="black", lw=0.4))
            # prediction swatch directly below
            ax.add_patch(Rectangle((c + 0.05, r + 0.65), 0.9, 0.2,
                                   facecolor=tuple(np.clip(pred[ci], 0, 1)),
                                   edgecolor="black", lw=0.4))
            ax.text(c + 0.5, r + 0.95,
                    f"{names[ci]}\nerr={err[ci]:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10)

    # Panel C: 12 best-predicted overall (any bucket)
    best_idx = np.argsort(err)[:12]
    draw_swatches(ax_swatch_best, best_idx,
                  "12 best-predicted colors (top swatch=actual, thin strip=pred)",
                  ncol=6)

    # Panel D: worst-predicted *common* colors + worst-predicted rare colors
    # side-by-side strip (1 row of 6 + 1 row of 6).
    worst_common = common_idx[np.argsort(-err[common_mask])][:6]
    worst_rare_pool = np.where(~common_mask)[0]
    worst_rare = worst_rare_pool[np.argsort(-err[~common_mask])][:6]
    # Combine into one panel with two labelled rows.
    combo = list(worst_common) + list(worst_rare)
    draw_swatches(ax_swatch_worst_common, combo,
                  "Worst-predicted: top row = common basics (cogito gets these wrong) | "
                  "bottom row = rare names",
                  ncol=6)

    fig.suptitle("auto_52 (oooooo): does cogito predict 'red' better than 'celadon'?",
                 fontsize=13)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    # ---- JSON summary -----------------------------------------------------
    summary = {
        "idea": "oooooo",
        "n_colors": int(n_c),
        "n_common": n_common,
        "n_rare": n_rare,
        "common_names_found": common_names,
        "knn_k": K,
        "mean_err_common": mc,
        "mean_err_rare": mr,
        "ci95_common": [lo_c, hi_c],
        "ci95_rare": [lo_r, hi_r],
        "diff_rare_minus_common": diff_mean,
        "diff_ci95": [dlo, dhi],
        "r2_common": r2_common,
        "r2_rare": r2_rare,
        "r2_all": r2_all,
        "best_predicted_top12": [{"name": names[i], "err": float(err[i])}
                                 for i in best_idx],
        "worst_common_top6":    [{"name": names[i], "err": float(err[i])}
                                 for i in worst_common],
        "worst_rare_top6":      [{"name": names[i], "err": float(err[i])}
                                 for i in worst_rare],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
