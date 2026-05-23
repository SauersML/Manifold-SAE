"""
auto_53 — (rrrrrr) Regress per-color goodness-of-fit on (name_length,
saturation, lightness): partial-coefficient decomposition.

Question
--------
Holding other factors constant, what *independently* makes a color easier
for cogito (residual-stream L40) to encode well?  Is it:
  - having a short, frequent name  (name_length, in tokens)
  - being vivid / saturated        (HSV saturation)
  - being mid-lightness vs extreme (HSV value, plus |value - 0.5|)

Method
------
- Read the d=3 unsupervised GAM manifold T (949 x 3) from results.json.
  This is the cogito-derived coordinate of each xkcd color (no Gaussian
  RBF, no Duchon length_scale used).
- Per-color prediction of (R,G,B) from T via LOO k-NN (k=10, distance
  weighted) — k-NN is on the allow-list. This produces a per-color
  residual sum-of-squares ss_res_i = ||pred_i - rgb_i||^2.
- Per-color "goodness-of-fit" defined as the deviance-share form,
      gof_i = 1 - ss_res_i / (mean ss_tot_i over all colors)
  i.e. how much better than the mean baseline this individual color is
  predicted. This is a well-defined per-color analogue of R^2 because
  the denominator is a single global scalar; summing across colors
  recovers the standard cross-validated R^2.
- Standardise features (z-score) and run OLS with statsmodels-free pure
  numpy: solve via normal equations + pseudo-inverse, recover
  standardised partial coefficients beta_k (one per feature), and
  decompose R^2 of the *gof regression* into per-feature share via the
  "Pratt index"  pratt_k = beta_k * r_k  where r_k = corr(x_k, y).
  Pratt shares sum to R^2 and are signed — they are the cleanest
  variance decomposition for non-orthogonal predictors.
- Features: [name_length (tokens), name_length (chars), HSV saturation,
  HSV value, |HSV value - 0.5| (mid-grey bias), HSV value squared].
  We also report bivariate Pearson r for each feature so the reader can
  see the unconditional vs partial effect.

Outputs
-------
PNG:  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_53.png  (4 panels)
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_53.json

Constraints: no Gaussian RBF; no Duchon length_scale set; only PCA /
ridge / k-NN / Duchon (not used here) allowed. Per the task: pick ONE
fresh idea.
"""
from __future__ import annotations

import json
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
XKCD     = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG  = RUN_DIR / "auto_53.png"
OUT_JSON = RUN_DIR / "auto_53.json"


def load_xkcd() -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        name, hex_ = parts[0], parts[1].lstrip("#")
        if len(hex_) != 6:
            continue
        r = int(hex_[0:2], 16) / 255.0
        g = int(hex_[2:4], 16) / 255.0
        b = int(hex_[4:6], 16) / 255.0
        names.append(name)
        rgb.append([r, g, b])
    return names, np.asarray(rgb, dtype=np.float64)


def loo_knn_predict(T: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(T)
    dist, idx = nn.kneighbors(T, return_distance=True)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    w = 1.0 / np.maximum(dist, 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * Y[idx]).sum(axis=1)


def rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for i, (r, g, b) in enumerate(rgb):
        out[i] = colorsys.rgb_to_hsv(r, g, b)
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return x - m
    return (x - m) / s


def ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (beta, R^2, y_hat). X assumed already to include any intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, yhat


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

    names, _ = load_xkcd()
    names = names[:n_c]

    # --- per-color goodness-of-fit (deviance-share form) ------------------
    K = 10
    pred = loo_knn_predict(T, rgb, k=K)
    ss_res = np.sum((pred - rgb) ** 2, axis=1)             # (n_c,)
    rgb_centered = rgb - rgb.mean(axis=0, keepdims=True)
    ss_tot_per_color = np.sum(rgb_centered ** 2, axis=1)   # (n_c,)
    ss_tot_mean = float(ss_tot_per_color.mean())
    gof = 1.0 - ss_res / ss_tot_mean
    r2_global = 1.0 - ss_res.sum() / ss_tot_per_color.sum()
    print(f"[knn k={K}] global R^2 = {r2_global:.4f}  "
          f"per-color gof  mean={gof.mean():.4f} std={gof.std():.4f}  "
          f"min={gof.min():.3f} max={gof.max():.3f}")

    # --- features ---------------------------------------------------------
    hsv = rgb_to_hsv_array(rgb)
    saturation = hsv[:, 1]
    value      = hsv[:, 2]
    mid_bias   = np.abs(value - 0.5)
    name_tokens = np.array([len(nm.split()) for nm in names], dtype=np.float64)
    name_chars  = np.array([len(nm) for nm in names], dtype=np.float64)

    feat_names = ["name_tokens", "name_chars", "saturation",
                  "value", "|value-0.5|"]
    F_raw = np.column_stack([name_tokens, name_chars, saturation,
                             value, mid_bias])

    # standardise both X and y for clean partial-coefficient interpretation
    Fz = np.column_stack([zscore(F_raw[:, k]) for k in range(F_raw.shape[1])])
    yz = zscore(gof)

    # bivariate Pearson r per feature
    bivar_r = np.array([np.corrcoef(F_raw[:, k], gof)[0, 1]
                        for k in range(F_raw.shape[1])])
    print("[bivariate] feature : Pearson r with gof")
    for nm, r in zip(feat_names, bivar_r):
        print(f"   {nm:>14s}  r = {r:+.4f}")

    # multivariate OLS (no intercept needed since both sides z-scored)
    beta, r2_full, yhat = ols_fit(Fz, yz)
    print(f"[ols]  R^2(model) = {r2_full:.4f}")
    print("[ols]  standardised partial coefficient per feature")
    for nm, b in zip(feat_names, beta):
        print(f"   {nm:>14s}  beta = {b:+.4f}")

    # Pratt variance decomposition: pratt_k = beta_k * r_k, sums to R^2.
    pratt = beta * bivar_r
    pratt_share = pratt / pratt.sum() if pratt.sum() != 0 else pratt
    print(f"[pratt] sum = {pratt.sum():.4f}  (should == R^2 = {r2_full:.4f})")

    # bootstrap CIs on standardised partial coefficients
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot_beta = np.empty((n_boot, Fz.shape[1]))
    for b in range(n_boot):
        sel = rng.integers(0, n_c, n_c)
        bb, _, _ = ols_fit(Fz[sel], yz[sel])
        boot_beta[b] = bb
    beta_lo = np.quantile(boot_beta, 0.025, axis=0)
    beta_hi = np.quantile(boot_beta, 0.975, axis=0)

    # bootstrap CIs on pratt shares
    boot_pratt = np.empty((n_boot, Fz.shape[1]))
    for b in range(n_boot):
        sel = rng.integers(0, n_c, n_c)
        bb, _, _ = ols_fit(Fz[sel], yz[sel])
        rr = np.array([np.corrcoef(F_raw[sel, k], gof[sel])[0, 1]
                       for k in range(F_raw.shape[1])])
        boot_pratt[b] = bb * rr
    pratt_lo = np.quantile(boot_pratt, 0.025, axis=0)
    pratt_hi = np.quantile(boot_pratt, 0.975, axis=0)

    # --- plot -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                          hspace=0.42, wspace=0.32,
                          left=0.07, right=0.98, top=0.92, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # A: bar of bivariate r vs standardised partial beta (side-by-side)
    xpos = np.arange(len(feat_names))
    w = 0.38
    axA.bar(xpos - w/2, bivar_r, width=w, color="#888888",
            label="bivariate r")
    axA.bar(xpos + w/2, beta,    width=w, color="#d62728",
            label="partial beta (std.)")
    yerr_lo = beta - beta_lo
    yerr_hi = beta_hi - beta
    axA.errorbar(xpos + w/2, beta, yerr=[yerr_lo, yerr_hi],
                 fmt="none", color="black", capsize=3, lw=1)
    axA.axhline(0, color="black", lw=0.6)
    axA.set_xticks(xpos)
    axA.set_xticklabels(feat_names, rotation=25, ha="right", fontsize=9)
    axA.set_ylabel("effect on per-color gof (std. units)")
    axA.set_title(f"Bivariate vs partial effects  "
                  f"(model R^2 = {r2_full:.3f})")
    axA.legend(fontsize=8, loc="best")
    axA.grid(axis="y", alpha=0.3)

    # B: Pratt variance shares
    colors_bar = ["#888888" if p >= 0 else "#cc4444" for p in pratt]
    axB.bar(xpos, pratt, color=colors_bar, edgecolor="black", lw=0.6)
    perr_lo = pratt - pratt_lo
    perr_hi = pratt_hi - pratt
    axB.errorbar(xpos, pratt, yerr=[perr_lo, perr_hi],
                 fmt="none", color="black", capsize=3, lw=1)
    axB.axhline(0, color="black", lw=0.6)
    axB.set_xticks(xpos)
    axB.set_xticklabels(feat_names, rotation=25, ha="right", fontsize=9)
    axB.set_ylabel("Pratt share of R^2  (beta * r)")
    axB.set_title(f"Pratt decomposition  (sum = R^2 = {pratt.sum():.3f})")
    axB.grid(axis="y", alpha=0.3)

    # C: gof vs saturation, colored by RGB
    axC.scatter(saturation, gof, s=14, c=rgb, alpha=0.85,
                edgecolors="none")
    # ridge-style smoother: bin and plot mean
    bins = np.linspace(0, 1, 11)
    bcent, bmean = [], []
    for i in range(len(bins) - 1):
        m = (saturation >= bins[i]) & (saturation < bins[i+1])
        if m.sum() >= 5:
            bcent.append(0.5 * (bins[i] + bins[i+1]))
            bmean.append(gof[m].mean())
    axC.plot(bcent, bmean, "k-", lw=2, label="bin mean")
    axC.set_xlabel("HSV saturation")
    axC.set_ylabel("per-color gof  (1 - ss_res / mean ss_tot)")
    axC.set_title(f"gof vs saturation  (r={bivar_r[2]:+.3f})")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)

    # D: gof vs HSV value
    axD.scatter(value, gof, s=14, c=rgb, alpha=0.85, edgecolors="none")
    bcent, bmean = [], []
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        m = (value >= bins[i]) & (value < bins[i+1])
        if m.sum() >= 5:
            bcent.append(0.5 * (bins[i] + bins[i+1]))
            bmean.append(gof[m].mean())
    axD.plot(bcent, bmean, "k-", lw=2, label="bin mean")
    axD.set_xlabel("HSV value (lightness)")
    axD.set_ylabel("per-color gof")
    axD.set_title(f"gof vs value  (r={bivar_r[3]:+.3f})")
    axD.legend(fontsize=8)
    axD.grid(alpha=0.3)

    # E: gof vs name token count (boxplot per token count)
    unique_tok = sorted(set(int(t) for t in name_tokens))
    box_data = [gof[name_tokens == t] for t in unique_tok]
    bp = axE.boxplot(box_data, positions=unique_tok, widths=0.6,
                     patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#ddddee")
    axE.set_xticks(unique_tok)
    axE.set_xticklabels([f"{t}\n(n={(name_tokens==t).sum():.0f})"
                         for t in unique_tok], fontsize=8)
    axE.set_xlabel("name length (tokens)")
    axE.set_ylabel("per-color gof")
    axE.set_title(f"gof vs name_tokens  (r={bivar_r[0]:+.3f})")
    axE.grid(axis="y", alpha=0.3)

    # F: gof predicted vs observed (sanity)
    axF.scatter(yhat, yz, s=12, c=rgb, alpha=0.7, edgecolors="none")
    lo = float(min(yhat.min(), yz.min()))
    hi = float(max(yhat.max(), yz.max()))
    axF.plot([lo, hi], [lo, hi], "k--", lw=1)
    axF.set_xlabel("OLS prediction of gof (z)")
    axF.set_ylabel("observed gof (z)")
    axF.set_title(f"OLS fit  R^2 = {r2_full:.3f}")
    axF.grid(alpha=0.3)

    fig.suptitle(
        "auto_53 (rrrrrr): partial-coefficient decomposition of per-color "
        f"gof on (name_length, saturation, lightness)  "
        f"[cogito L40, d=3 manifold, LOO k-NN k={K}]",
        fontsize=12,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    # --- JSON summary -----------------------------------------------------
    summary = {
        "idea": "rrrrrr",
        "n_colors": int(n_c),
        "knn_k": K,
        "r2_global_knn": float(r2_global),
        "gof_mean": float(gof.mean()),
        "gof_std": float(gof.std()),
        "gof_min": float(gof.min()),
        "gof_max": float(gof.max()),
        "features": feat_names,
        "bivariate_pearson_r": [float(x) for x in bivar_r],
        "ols_standardised_beta": [float(x) for x in beta],
        "ols_beta_ci95": [[float(lo), float(hi)]
                          for lo, hi in zip(beta_lo, beta_hi)],
        "ols_R2": float(r2_full),
        "pratt_share": [float(x) for x in pratt],
        "pratt_share_ci95": [[float(lo), float(hi)]
                             for lo, hi in zip(pratt_lo, pratt_hi)],
        "pratt_sum_equals_R2": float(pratt.sum()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
