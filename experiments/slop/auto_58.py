"""
auto_58 — (ddddddd) Concentration of variance: how many colors contribute
50% / 80% of the total cogito prediction error?

Question
--------
The cogito L40 residual stream "knows" colors with varying fidelity. If we
take the d=3 unsupervised manifold T (949 x 3) and predict each color's
RGB from its cogito neighbourhood (LOO k-NN, k=10, distance-weighted —
all allow-listed primitives, no Gaussian RBF, no Duchon length_scale set),
we get a per-color squared-error e_i = ||pred_i - rgb_i||^2.

How peaked is the distribution of e_i? Concretely:
  - sort colors by e_i descending,
  - compute the cumulative share of total error contributed by the
    top-k worst colors,
  - report k50, k80, k95 (how many colors needed to explain 50/80/95%
    of total error), and the Gini coefficient of the error distribution.

Compare three predictors:
  1. LOO k-NN in cogito-T (k=10, distance-weighted)   "cogito"
  2. LOO k-NN in RGB        (k=10, distance-weighted) "RGB-knn"   (control)
  3. Global mean of RGB                               "mean"      (null)
This isolates *which* errors are an artefact of the LOO estimator vs
genuinely cogito-specific.

Outputs
-------
PNG : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_58.png   (4 panels)
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_58.json

Constraints respected: PCA / linear-ridge / Duchon (unused) / k-NN /
t-SNE / UMAP only. No Gaussian RBF. No Duchon length_scale.
"""
from __future__ import annotations

import json
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
XKCD     = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG  = RUN_DIR / "auto_58.png"
OUT_JSON = RUN_DIR / "auto_58.json"

K_NN = 10


def load_names(n: int) -> list[str]:
    if not XKCD.exists():
        return [f"color_{i}" for i in range(n)]
    names = []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t") if p.strip()]
        if len(parts) >= 2:
            names.append(parts[0])
    while len(names) < n:
        names.append(f"color_{len(names)}")
    return names[:n]


def loo_knn_predict(X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    dist, idx = dist[:, 1:], idx[:, 1:]
    w = 1.0 / np.maximum(dist, 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * Y[idx]).sum(axis=1)


def concentration_stats(err: np.ndarray) -> dict:
    """Return Lorenz / concentration stats for a non-negative vector."""
    n = err.size
    total = float(err.sum())
    order_desc = np.argsort(-err)
    cum = np.cumsum(err[order_desc]) / total
    k50 = int(np.searchsorted(cum, 0.50) + 1)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    order_asc = np.argsort(err)
    cum_asc = np.cumsum(err[order_asc]) / total
    # Gini via the trapezoidal formula on the Lorenz curve.
    pct_pop = np.arange(1, n + 1) / n
    lorenz = np.concatenate([[0.0], cum_asc])
    pct_pop_aug = np.concatenate([[0.0], pct_pop])
    auc_lorenz = float(np.trapezoid(lorenz, pct_pop_aug))
    gini = float(1.0 - 2.0 * auc_lorenz)
    return {
        "k50": k50, "k80": k80, "k95": k95,
        "frac50": k50 / n, "frac80": k80 / n, "frac95": k95 / n,
        "gini": gini,
        "order_desc": order_desc,
        "lorenz_pop": pct_pop_aug,
        "lorenz_share": lorenz,
    }


def rgb_to_hsv_arr(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for i, (r, g, b) in enumerate(rgb):
        out[i] = colorsys.rgb_to_hsv(r, g, b)
    return out


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    T = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.column_stack([Rax, Gax, Bax])
    n_c = rgb.shape[0]
    assert T.shape == (n_c, 3)
    names = load_names(n_c)
    print(f"[load] n_colors={n_c}  d=3 T shape={T.shape}")

    # ---- per-color squared errors for three predictors --------------------
    pred_cogito = loo_knn_predict(T,  rgb, k=K_NN)
    pred_rgbnn  = loo_knn_predict(rgb, rgb, k=K_NN)  # LOO k-NN in RGB itself
    pred_mean   = np.broadcast_to(rgb.mean(axis=0, keepdims=True), rgb.shape)

    err_cogito = np.sum((pred_cogito - rgb) ** 2, axis=1)
    err_rgbnn  = np.sum((pred_rgbnn  - rgb) ** 2, axis=1)
    err_mean   = np.sum((pred_mean   - rgb) ** 2, axis=1)

    # global R^2 sanity (vs mean predictor)
    ss_tot = float(np.sum((rgb - rgb.mean(axis=0)) ** 2))
    r2_cogito = 1.0 - float(err_cogito.sum()) / ss_tot
    r2_rgbnn  = 1.0 - float(err_rgbnn.sum())  / ss_tot
    print(f"[r2] cogito={r2_cogito:.4f}  rgb-knn={r2_rgbnn:.4f}  "
          f"(both LOO k={K_NN})")

    stats_cog  = concentration_stats(err_cogito)
    stats_rgb  = concentration_stats(err_rgbnn)
    stats_mean = concentration_stats(err_mean)

    for nm, st in [("cogito", stats_cog),
                   ("rgb-knn", stats_rgb),
                   ("mean",    stats_mean)]:
        print(f"[concentration {nm:>7s}]  "
              f"k50={st['k50']:>3d} ({100*st['frac50']:.1f}%)  "
              f"k80={st['k80']:>3d} ({100*st['frac80']:.1f}%)  "
              f"k95={st['k95']:>3d} ({100*st['frac95']:.1f}%)  "
              f"gini={st['gini']:.3f}")

    # ratio: how much more concentrated than mean?
    excess_50 = stats_mean['k50'] / max(1, stats_cog['k50'])
    print(f"[fold-excess] cogito is {excess_50:.2f}x more concentrated "
          f"than the mean predictor at the 50% mark "
          f"(k50: cogito={stats_cog['k50']} vs mean={stats_mean['k50']})")

    # ---- plot --------------------------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                           hspace=0.36, wspace=0.26,
                           left=0.06, right=0.98, top=0.91, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # --- (A) Lorenz curve (cumulative-share view, descending) -------------
    # x = fraction of worst colors, y = cumulative share of total error
    def cum_desc(err):
        s = np.sort(err)[::-1]
        n = s.size
        return np.arange(1, n + 1) / n, np.cumsum(s) / s.sum()

    for nm, err, col in [
        ("cogito (d=3 kNN-T)", err_cogito, "#d62728"),
        ("RGB-kNN (control)",  err_rgbnn,  "#1f77b4"),
        ("mean predictor",     err_mean,   "#7f7f7f"),
    ]:
        x, y = cum_desc(err)
        axA.plot(x, y, lw=2.0, color=col, label=nm)
    for thr, col in [(0.50, "#222222"), (0.80, "#666666"), (0.95, "#aaaaaa")]:
        axA.axhline(thr, color=col, lw=0.6, ls="--")
        axA.text(0.99, thr + 0.005, f"{int(100*thr)}%", ha="right",
                 fontsize=8, color=col)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.005)
    axA.set_xlabel("fraction of colors (sorted worst → best)")
    axA.set_ylabel("cumulative share of total squared error")
    axA.set_title("Concentration of error: cumulative share vs population fraction")
    axA.legend(loc="lower right", fontsize=9)
    axA.grid(alpha=0.3)

    # --- (B) Standard Lorenz (ascending), with Gini area shaded ---------
    for nm, st, col in [
        ("cogito", stats_cog, "#d62728"),
        ("RGB-kNN", stats_rgb, "#1f77b4"),
        ("mean",    stats_mean, "#7f7f7f"),
    ]:
        axB.plot(st['lorenz_pop'], st['lorenz_share'], lw=2.0,
                 color=col, label=f"{nm}  (Gini={st['gini']:.3f})")
    axB.plot([0, 1], [0, 1], "k:", lw=1, label="equality line")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_xlabel("fraction of colors (sorted best → worst)")
    axB.set_ylabel("cumulative share of total squared error")
    axB.set_title("Lorenz curve of per-color error (lower = more peaked)")
    axB.legend(loc="upper left", fontsize=9)
    axB.grid(alpha=0.3)

    # --- (C) bar of k50/k80/k95 across predictors -----------------------
    labels = ["k50", "k80", "k95"]
    preds  = [("cogito", stats_cog, "#d62728"),
              ("RGB-kNN", stats_rgb, "#1f77b4"),
              ("mean",    stats_mean, "#7f7f7f")]
    xpos = np.arange(3)
    w = 0.27
    for i, (nm, st, col) in enumerate(preds):
        vals = [st['k50'], st['k80'], st['k95']]
        bars = axC.bar(xpos + (i - 1) * w, vals, width=w,
                       color=col, label=nm, edgecolor="black", lw=0.5)
        for bar, v in zip(bars, vals):
            axC.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 8, f"{v}",
                     ha="center", fontsize=8)
    axC.set_xticks(xpos); axC.set_xticklabels(labels)
    axC.set_ylabel(f"# colors (of {n_c}) to reach threshold")
    axC.set_title("How many colors account for 50/80/95% of total error?")
    axC.legend(fontsize=9)
    axC.grid(axis="y", alpha=0.3)
    axC.axhline(n_c / 2, color="black", lw=0.4, ls=":")
    axC.text(2.45, n_c / 2 + 8, f"n/2 = {n_c // 2}",
             ha="right", fontsize=8, color="black")

    # --- (D) top-30 worst colors under cogito (with RGB swatches) -------
    TOP = 30
    worst = stats_cog['order_desc'][:TOP]
    ncols = 6
    nrows = TOP // ncols
    axD.set_xlim(0, ncols)
    axD.set_ylim(0, nrows)
    axD.invert_yaxis()
    axD.set_xticks([]); axD.set_yticks([])
    for spine in axD.spines.values():
        spine.set_visible(False)
    for rank, ci in enumerate(worst):
        r = rank // ncols
        c = rank %  ncols
        axD.add_patch(Rectangle((c + 0.05, r + 0.15), 0.55, 0.7,
                                facecolor=tuple(rgb[ci]),
                                edgecolor="black", lw=0.5))
        axD.text(c + 0.65, r + 0.42,
                 f"{names[ci][:18]}",
                 fontsize=8, va="center", ha="left")
        axD.text(c + 0.65, r + 0.70,
                 f"e²={err_cogito[ci]:.3f}",
                 fontsize=7, va="center", ha="left", color="#444444")
    pct_top30 = 100.0 * float(err_cogito[worst].sum()) / float(err_cogito.sum())
    axD.set_title(
        f"Top {TOP} worst-predicted colors under cogito "
        f"(account for {pct_top30:.1f}% of total error)")

    fig.suptitle(
        "auto_58 (ddddddd): Concentration of variance — how many colors "
        f"contribute 50/80/95% of cogito's prediction error? "
        f"  [L40, d=3 manifold, LOO k-NN k={K_NN};   "
        f"k50(cogito)={stats_cog['k50']}/{n_c} "
        f"({100*stats_cog['frac50']:.1f}%),  Gini={stats_cog['gini']:.3f}]",
        fontsize=12)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    # ---- relationship between error and HSV (quick correlate) ----------
    hsv = rgb_to_hsv_arr(rgb)
    pearson_sat = float(np.corrcoef(hsv[:, 1], err_cogito)[0, 1])
    pearson_val = float(np.corrcoef(hsv[:, 2], err_cogito)[0, 1])
    print(f"[correlate] corr(error_cogito, sat)={pearson_sat:+.3f}  "
          f"corr(error_cogito, val)={pearson_val:+.3f}")

    summary = {
        "idea": "ddddddd",
        "n_colors": int(n_c),
        "knn_k": int(K_NN),
        "r2_cogito_loo_knn": r2_cogito,
        "r2_rgbnn_loo_knn":  r2_rgbnn,
        "concentration": {
            "cogito":  {k: stats_cog[k]  for k in
                        ("k50","k80","k95","frac50","frac80","frac95","gini")},
            "rgb_knn": {k: stats_rgb[k]  for k in
                        ("k50","k80","k95","frac50","frac80","frac95","gini")},
            "mean":    {k: stats_mean[k] for k in
                        ("k50","k80","k95","frac50","frac80","frac95","gini")},
        },
        "top30_worst_cogito_indices": [int(i) for i in stats_cog['order_desc'][:30]],
        "top30_worst_cogito_names":   [names[int(i)] for i in stats_cog['order_desc'][:30]],
        "top30_share_of_error":       pct_top30 / 100.0,
        "corr_err_cogito_saturation": pearson_sat,
        "corr_err_cogito_value":      pearson_val,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
