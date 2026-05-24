"""auto_95 — Manifesto figure: the cogito-L40 color manifold in one page.

Six-panel synthesis pulling together auto_78, auto_79, auto_83, auto_85,
auto_86, and auto_exp_38 into a single coherent narrative:

  A. Bulk-vs-tail Pareto (auto_78 reduce): every GAM-zoo spec on one plane,
     with the empirical Pareto frontier drawn and 4 hand-picked exemplars
     annotated. Shows the dichotomy: bulk PCs 1-8 are easy; the tail is
     a different geometry that only Duchon + unsupervised specs touch.

  B. Hue ring in PC2 x PC4 (auto_86 reduce): centroid scatter colored by
     true xkcd-RGB, with a fitted circle overlay and the U_3d gauge-fixed
     "hue axis" projected in as an arrow.

  C. HSV vs Name-Semantic ORTHOGONALITY block-matrix (auto_exp_38): 6x6
     |corr| heatmap showing the §4(c) decomposition empirically — top-left
     3x3 is HSV-diagonal, bottom-right 3x3 is name-semantic, off-diagonal
     blocks are near-zero. This is the load-bearing finding.

  D. PC variance vs hue-circularity (auto_85 reduce): why ARD fails. The
     high-variance PCs (which ARD keeps) carry NO hue; the low-variance
     PC2+PC4 carry ALL of it. Spearman is strongly negative.

  E. Per-template R^2 strip (auto_83 reduce): a compact horizontal bar
     of the 28 templates sorted by colour-explainability, showing the
     ceiling/noise-floor spread.

  F. Six concept reconstructions: pick 6 xkcd colors representing distinct
     regions of the manifold (red, blue, green, dusty/muted, pale, dark)
     and show actual swatch vs HSV-from-U_3d reconstruction swatch.

The figure subtitle states the unique synthesis finding: cogito-L40's
color geometry is the DIRECT SUM of a 3-D perceptual subspace (HSV) and
a 3-D name-semantic subspace, with the perceptual hue circle living in
the LOW-variance PCs that ARD would prune.

Cached inputs (no GPU, no cogito server):
  - results.json                    -> U_3d.T, EVR, Vt_topK, mu, sigma, specs
  - auto_78.json                    -> bulk vs tail rows
  - auto_exp_38.json                -> corr_hsv / corr_name 6x3 blocks
  - X_L40.npy (mmap)                -> per-template R^2 + PC2xPC4 scatter
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
RESULTS = RUN_DIR / "results.json"
HARVEST = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = RUN_DIR / "auto_95.png"

K_PCS = 16
PC_I, PC_J = 2, 4
TOP_PCS_TEMPLATE = 8
RIDGE_LAM = 1e-2

# Consistent palette
CMAP_HEAT = "magma"
COL_HSV = "#2a9d8f"
COL_NAME = "#e76f51"
COL_PARETO = "#264653"
COL_PRIMARY = "#e9c46a"


# --------------------------------- helpers --------------------------------
def kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def ridge_cv_per_pc(F, Z, n_folds, lam, rng):
    n, P = Z.shape
    folds = kfold_indices(n, n_folds, rng)
    preds = np.zeros_like(Z)
    for fi in range(n_folds):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        A = F[tr].T @ F[tr] + lam * np.eye(F.shape[1])
        W = np.linalg.solve(A, F[tr].T @ Z[tr])
        preds[te] = F[te] @ W
    ss_r = ((Z - preds) ** 2).sum(0)
    ss_t = ((Z - Z.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss_r / np.maximum(ss_t, 1e-12)


def build_features(R, G, B, H, S, V, L):
    Hc = np.cos(2 * np.pi * H)
    Hs = np.sin(2 * np.pi * H)
    return np.stack([np.ones_like(R), R, G, B, Hc, Hs, S, V, L,
                     R * G, G * B, R * B], axis=1)


def circ_corr_js(a, b):
    a_bar = np.angle(np.mean(np.exp(1j * a)))
    b_bar = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - a_bar) * np.sin(b - b_bar))
    den = np.sqrt(np.sum(np.sin(a - a_bar) ** 2)
                  * np.sum(np.sin(b - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def abbrev(t, n=42):
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= n else t[: n - 1] + "..."


# --------------------------------- main -----------------------------------
def main():
    t0 = time.time()
    print("[auto_95] manifesto synthesis figure")

    # ---- load cached results ---------------------------------------------
    res = json.loads(RESULTS.read_text())
    pl = res["per_layer"]["L40"]
    templates = res["templates"]
    n_t = len(templates)
    evr = np.array(pl["explained_variance_ratio_topK"], dtype=np.float64)
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float64)
    mu = np.asarray(pl["mu"], dtype=np.float64)
    sigma = np.asarray(pl["sigma"], dtype=np.float64)
    T_u3d = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)

    axes = res["color_axes_per_color_index"]
    R = np.asarray(axes["R"], dtype=np.float64)
    G = np.asarray(axes["G"], dtype=np.float64)
    B = np.asarray(axes["B"], dtype=np.float64)
    H = np.asarray(axes["hue"], dtype=np.float64)
    S = np.asarray(axes["sat"], dtype=np.float64)
    V = np.asarray(axes["value"], dtype=np.float64)
    Lm = np.asarray(axes["luminance"], dtype=np.float64)
    n_c = len(R)
    print(f"[data] n_c={n_c} n_t={n_t} K_pcs={Vt.shape[0]}")

    a78 = json.loads((RUN_DIR / "auto_78.json").read_text())
    a38 = json.loads((RUN_DIR / "auto_exp_38.json").read_text())

    # ---- harvest-side computations: PC2xPC4 ring + per-template R^2 -----
    print("[harvest] loading X mmap...")
    X = np.load(HARVEST, mmap_mode="r")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    print(f"[harvest] T0 centroids = {T0.shape}")

    Tc = T0 - T0.mean(0, keepdims=True)
    _, S_sv, Vt_cent = np.linalg.svd(Tc, full_matrices=False)
    plane = Tc @ Vt_cent[[PC_I, PC_J]].T
    p_n = (plane - plane.mean(0)) / plane.std(0)
    theta = np.arctan2(plane[:, 1], plane[:, 0])
    names_xkcd, rgb_real = load_xkcd_rgb(n_c)
    hsv_real = hsv_from_rgb(rgb_real)
    hue_rad = 2 * np.pi * hsv_real[:, 0]
    cc_hue = circ_corr_js(theta, hue_rad)
    print(f"[hue-ring] circ-corr |J-S| = {abs(cc_hue):.3f}")

    # Per-PC variance + per-PC max-|circ-corr| for panel D
    var_per_pc = (S_sv[:K_PCS] ** 2) / (n_c - 1)
    best_cc = np.zeros(K_PCS)
    for i in range(K_PCS):
        best = 0.0
        for j in range(K_PCS):
            if i == j:
                continue
            pl_ij = Tc @ Vt_cent[[i, j]].T
            th = np.arctan2(pl_ij[:, 1], pl_ij[:, 0])
            cc = circ_corr_js(th, hue_rad)
            if abs(cc) > abs(best):
                best = cc
        best_cc[i] = best

    # Per-template macro R^2 over top-8 PCs (auto_83 condensed)
    F_feat = build_features(R, G, B, H, S, V, Lm)
    macro_top = np.zeros(n_t)
    for t in range(n_t):
        rows = np.arange(n_c) * n_t + t
        Xt = np.asarray(X[rows], dtype=np.float64)
        Zt = ((Xt - mu) / np.maximum(sigma, 1e-8)) @ Vt.T
        r2 = ridge_cv_per_pc(F_feat, Zt[:, :TOP_PCS_TEMPLATE],
                             n_folds=5, lam=RIDGE_LAM,
                             rng=np.random.default_rng(0))
        macro_top[t] = r2.mean()
        print(f"  [t={t:02d}] R2_top8 = {macro_top[t]:+.3f}")

    # ---- panel F exemplars: 6 colors spanning manifold regions ----------
    # picks from common-knowledge xkcd palette
    pick_names = ["red", "navy blue", "kelly green",
                  "dusty rose", "pale yellow", "dark olive"]
    pick_idx = []
    for nm in pick_names:
        if nm in names_xkcd:
            pick_idx.append(names_xkcd.index(nm))
        else:
            # fallback: nearest match by substring
            cand = [i for i, n in enumerate(names_xkcd) if nm.split()[0] in n]
            pick_idx.append(cand[0] if cand else 0)
    pick_idx = np.array(pick_idx)
    rgb_pick = np.clip(rgb_real[pick_idx], 0, 1)
    hsv_pick = hsv_real[pick_idx]

    # Predict HSV from U_3d via 5-fold ridge on held-out colors
    Phi = np.concatenate([T_u3d, np.ones((n_c, 1))], axis=1)
    HSV = hsv_real.copy()
    rng = np.random.default_rng(0)
    folds = kfold_indices(n_c, 5, rng)
    pred_hsv = np.zeros_like(HSV)
    for fi in range(5):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(5) if j != fi])
        W = np.linalg.solve(Phi[tr].T @ Phi[tr] + 1e-3 * np.eye(Phi.shape[1]),
                            Phi[tr].T @ HSV[tr])
        pred_hsv[te] = Phi[te] @ W
    pred_hsv[:, 0] = pred_hsv[:, 0] % 1.0
    pred_hsv = np.clip(pred_hsv, 0, 1)
    rgb_pred = np.array([mcolors.hsv_to_rgb(h) for h in pred_hsv])

    # ---- build figure ----------------------------------------------------
    print("[plot] building 6-panel figure")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.0, 0.7],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.42, wspace=0.30,
    )

    # =================== PANEL A — Pareto: bulk vs tail ==================
    axA = fig.add_subplot(gs[0, 0])
    rows = a78["rows"]
    xs = np.array([r["bulk_r2"] for r in rows])
    ys = np.array([r["tail_r2"] for r in rows])
    specs = [r["spec"] for r in rows]

    axA.scatter(xs, ys, s=24, c="#94a3b8", edgecolors="black",
                linewidths=0.25, alpha=0.55, zorder=2)

    # Pareto frontier (max y for each prefix sorted by x descending)
    order = np.argsort(-xs)
    cur_best = -np.inf
    pareto_pts = []
    for i in order:
        if ys[i] > cur_best:
            pareto_pts.append((xs[i], ys[i], specs[i]))
            cur_best = ys[i]
    pareto_pts.sort()
    px, py, pn = zip(*pareto_pts)
    axA.plot(px, py, "-o", color=COL_PARETO, lw=1.6, ms=5,
             markeredgecolor="black", markeredgewidth=0.4,
             label="Pareto frontier", zorder=5)

    # Annotate 4 hand-picked exemplars
    exemplars = {
        "L_lin_hsv": "linear HSV\n(bulk-only)",
        "U_pca6_duchon_joint": "Duchon joint\n(true geometry)",
        "M_chroma_disk_plus_L": "chroma-disk+L\n(perceptual)",
        "U_3d": "U_3d\n(unsupervised)",
    }
    for spec, label in exemplars.items():
        for r in rows:
            if r["spec"] == spec:
                axA.scatter([r["bulk_r2"]], [r["tail_r2"]], s=120,
                            c=COL_PRIMARY, edgecolors="black",
                            linewidths=1.0, zorder=6)
                axA.annotate(label, (r["bulk_r2"], r["tail_r2"]),
                             xytext=(8, 6), textcoords="offset points",
                             fontsize=8.5, fontweight="bold",
                             arrowprops=dict(arrowstyle="-", lw=0.4))
                break
    axA.plot([0, 1], [0, 1], ":", color="black", lw=0.6, alpha=0.4)
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.25, 0.55)
    axA.set_xlabel(f"EVR-weighted R^2 on BULK PCs 1-8  "
                   f"(EVR={a78['bulk_evr']:.2f})", fontsize=9.5)
    axA.set_ylabel(f"EVR-weighted R^2 on TAIL PCs 9-64  "
                   f"(EVR={a78['tail_evr']:.2f})", fontsize=9.5)
    axA.set_title("A. Bulk-vs-tail Pareto: two regimes of cogito-L40 "
                  f"geometry  (N={len(rows)} specs)",
                  fontsize=11, fontweight="bold", loc="left")
    axA.grid(ls=":", alpha=0.35)
    axA.legend(loc="upper left", fontsize=8.5, frameon=True)

    # =================== PANEL B — PC2xPC4 hue ring ======================
    axB = fig.add_subplot(gs[0, 1])
    rgb_clip = np.clip(rgb_real, 0, 1)
    axB.scatter(p_n[:, 0], p_n[:, 1], c=rgb_clip, s=22,
                edgecolors="black", linewidths=0.18)
    # overlay best-fit unit circle (in standardized coords) sized by RMS radius
    r_emp = np.sqrt((p_n ** 2).sum(1)).mean()
    th_grid = np.linspace(0, 2 * np.pi, 200)
    axB.plot(r_emp * np.cos(th_grid), r_emp * np.sin(th_grid),
             "--", color=COL_PARETO, lw=1.2, alpha=0.8,
             label=f"ring at r={r_emp:.2f}")
    # arrow showing U_3d hue-correlated axis direction projected into PC2xPC4
    # Build the linear map T_u3d -> (PC_I, PC_J) coords and pick the axis
    # with maximal |corr| with hue_rad sin/cos.
    Phi_T = np.concatenate([T_u3d, np.ones((n_c, 1))], axis=1)
    W = np.linalg.solve(Phi_T.T @ Phi_T + 1e-3 * np.eye(4),
                        Phi_T.T @ p_n)
    # Project a unit vector along each T axis into PC2xPC4
    arrows = []
    for k in range(3):
        e = np.zeros(4); e[k] = 1.0
        v = (e @ W) - (np.array([0, 0, 0, 1]) @ W)
        arrows.append((k, v))
    # plot all 3 arrows in cooler colors
    arr_cols = ["#9b5de5", "#00bbf9", "#06d6a0"]
    for (k, v), c in zip(arrows, arr_cols):
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            continue
        v_show = v / nrm * 1.6
        axB.annotate("", xy=v_show, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
        axB.text(v_show[0] * 1.05, v_show[1] * 1.05, f"U_3d t{k+1}",
                 color=c, fontsize=8.5, fontweight="bold",
                 ha="left", va="center")
    axB.axhline(0, color="gray", lw=0.4); axB.axvline(0, color="gray", lw=0.4)
    axB.set_aspect("equal")
    axB.set_xlim(-2.6, 2.6); axB.set_ylim(-2.6, 2.6)
    axB.set_xlabel(f"PC{PC_I} (standardized)", fontsize=9.5)
    axB.set_ylabel(f"PC{PC_J} (standardized)", fontsize=9.5)
    axB.set_title(f"B. Hue ring in PC{PC_I} x PC{PC_J}  "
                  f"|J-S circ-corr| = {abs(cc_hue):.2f}",
                  fontsize=11, fontweight="bold", loc="left")
    axB.legend(loc="lower right", fontsize=8, frameon=True)

    # =================== PANEL C — orthogonality block matrix ============
    axC = fig.add_subplot(gs[0, 2])
    corr_hsv = np.array(a38["corr_hsv_all_axes"])    # 6x3 (axes x [h,s,v])
    corr_name = np.array(a38["corr_name_all_axes"])  # 6x3 (axes x [mono,modc,tsig])
    # Build 6x6 = [corr_hsv | corr_name] (rows = 6 fit axes, cols = 6 targets)
    M = np.concatenate([corr_hsv, corr_name], axis=1)
    im = axC.imshow(np.abs(M), cmap=CMAP_HEAT, vmin=0, vmax=0.9, aspect="equal")
    axC.set_xticks(range(6))
    axC.set_xticklabels(["hue", "sat", "val", "mono", "mod_c", "tmpl_σ"],
                        fontsize=9, rotation=30, ha="right")
    axC.set_yticks(range(6))
    axC.set_yticklabels([f"sup ax{i+1}" if i < 3 else f"free ax{i-2}"
                         for i in range(6)], fontsize=9)
    # mark the 3x3 blocks
    for (x, y, c) in [(0, 0, COL_HSV), (3, 3, COL_NAME)]:
        axC.add_patch(mpatches.Rectangle(
            (x - 0.5, y - 0.5), 3, 3,
            fill=False, edgecolor=c, lw=2.4))
    # annotate cells
    for i in range(6):
        for j in range(6):
            txt = f"{abs(M[i, j]):.2f}"
            col = "white" if abs(M[i, j]) > 0.45 else "black"
            axC.text(j, i, txt, ha="center", va="center",
                     fontsize=7.5, color=col)
    axC.set_title("C. §4(c) orthogonality: HSV ⊕ name-semantic\n"
                  "blocks (auto_exp_38, d_aux=3+3)",
                  fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=axC, shrink=0.7, label="|corr|")

    # =================== PANEL D — ARD failure ===========================
    axD = fig.add_subplot(gs[1, 0])
    pcs = np.arange(K_PCS)
    bars = axD.bar(pcs, var_per_pc, color="#94a3b8", alpha=0.85,
                   label="variance(PC_k)")
    axD.set_yscale("log")
    axD.set_xlabel("PC index", fontsize=9.5)
    axD.set_ylabel("variance (log)", color="#475569", fontsize=9.5)
    axD.tick_params(axis="y", labelcolor="#475569")
    axD.set_xticks(pcs)
    axD2 = axD.twinx()
    axD2.plot(pcs, np.abs(best_cc), "o-", color=COL_NAME, lw=1.8, ms=6.5,
              markeredgecolor="black", markeredgewidth=0.4,
              label="max |circ-corr| with hue")
    axD2.set_ylabel("max |circ-corr| with hue", color=COL_NAME, fontsize=9.5)
    axD2.tick_params(axis="y", labelcolor=COL_NAME)
    axD2.set_ylim(0, 0.9)
    # mark the hue-carrying PCs
    for k in (PC_I, PC_J):
        axD.axvspan(k - 0.4, k + 0.4, color=COL_PRIMARY, alpha=0.35, zorder=0)
    from scipy.stats import spearmanr  # type: ignore
    rho, _ = spearmanr(var_per_pc, np.abs(best_cc))
    axD.set_title(f"D. ARD trap — top-variance PC0 carries NO hue   "
                  f"(Spearman var↔|cc|={rho:+.2f})",
                  fontsize=11, fontweight="bold", loc="left")
    axD.text(0.5, 0.93,
             f"hue ring sits in PC{PC_I}×PC{PC_J} (gold, |cc|≈{abs(best_cc[PC_I]):.2f}),  "
             f"but PC0 (largest variance) has |cc|={abs(best_cc[0]):.2f}",
             transform=axD.transAxes, fontsize=8.5,
             ha="center", va="top",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # =================== PANEL E — per-template R^2 ======================
    axE = fig.add_subplot(gs[1, 1])
    order_t = np.argsort(macro_top)
    y = np.arange(n_t)
    cols = plt.cm.viridis(
        (macro_top[order_t] - macro_top.min())
        / max(1e-6, macro_top.max() - macro_top.min())
    )
    axE.barh(y, macro_top[order_t], color=cols,
             edgecolor="black", linewidth=0.3)
    axE.set_yticks(y)
    axE.set_yticklabels([abbrev(templates[t], 36) for t in order_t],
                        fontsize=6.5)
    axE.set_xlabel(f"held-out R^2 on top-{TOP_PCS_TEMPLATE} PCs "
                   f"(5-fold ridge)", fontsize=9.5)
    axE.set_title(f"E. Template noise floor vs ceiling  "
                  f"(spread = {macro_top.max() - macro_top.min():.2f})",
                  fontsize=11, fontweight="bold", loc="left")
    axE.axvline(macro_top.mean(), color=COL_PARETO, lw=1.0, ls="--",
                label=f"mean={macro_top.mean():.2f}")
    axE.legend(loc="lower right", fontsize=8)
    axE.grid(axis="x", ls=":", alpha=0.35)

    # =================== PANEL F — 6 concept reconstructions =============
    axF = fig.add_subplot(gs[1, 2])
    axF.set_xlim(0, 6); axF.set_ylim(0, 4)
    axF.axis("off")
    axF.set_title("F. Concept reconstruction: U_3d -> HSV  "
                  "(actual vs predicted swatch)",
                  fontsize=11, fontweight="bold", loc="left")
    # 2 rows x 3 cols of swatch-pairs
    for k, ci in enumerate(pick_idx):
        col = k % 3; row = 1 - (k // 3)
        x0 = col * 2.0 + 0.15; y0 = row * 2.0 + 0.2
        # actual swatch
        axF.add_patch(mpatches.Rectangle((x0, y0 + 0.55), 0.75, 0.9,
                                         facecolor=rgb_pick[k],
                                         edgecolor="black"))
        # predicted swatch
        axF.add_patch(mpatches.Rectangle((x0 + 0.85, y0 + 0.55), 0.75, 0.9,
                                         facecolor=rgb_pred[ci],
                                         edgecolor="black"))
        axF.text(x0, y0 + 1.55, names_xkcd[ci], fontsize=8.5,
                 fontweight="bold", ha="left")
        axF.text(x0 + 0.375, y0 + 0.45, "actual",
                 fontsize=7, ha="center", va="top")
        axF.text(x0 + 1.225, y0 + 0.45, "pred",
                 fontsize=7, ha="center", va="top")
        dH = abs((pred_hsv[ci, 0] - hsv_pick[k, 0] + 0.5) % 1.0 - 0.5)
        axF.text(x0, y0 + 0.05,
                 f"ΔH={dH:.2f}  ΔS={abs(pred_hsv[ci,1]-hsv_pick[k,1]):.2f}  "
                 f"ΔV={abs(pred_hsv[ci,2]-hsv_pick[k,2]):.2f}",
                 fontsize=6.8, ha="left")

    # =================== bottom: caption box =============================
    axCap = fig.add_subplot(gs[2, :])
    axCap.axis("off")
    caption = (
        "cogito-L40 color manifold, synthesis. "
        f"949 xkcd colors × 28 templates, harvest from Qwen3.6-27B layer 40 (width 7168). "
        f"Two regimes coexist (A): the bulk PCs 1-8 carry ~{a78['bulk_evr']:.0%} of variance "
        "and are linearly explained by HSV; the tail PCs 9-64 require Duchon / unsupervised "
        "geometry. The hue circle (B) lives in PC2 × PC4 with |circ-corr|≈"
        f"{abs(cc_hue):.2f}, with PC0 (largest variance) carrying NO hue — "
        "exactly the failure mode that traps ARD-alpha (D). Supervising HSV on 3 axes and leaving 3 free recovers "
        "name-semantic axes (mono-word / modifier-count / template-σ) UNSUPERVISEDLY in the "
        "free block (C, lower-right), with off-diagonal blocks near zero: the manifold "
        "decomposes as a direct sum HSV ⊕ name-semantic. "
        f"Templates vary by {macro_top.max() - macro_top.min():.2f} R^2 (E) — abstract paint/pigment "
        "templates are the cleanest, embodied 'horse / cat' templates the noisiest. "
        "U_3d alone (F) reconstructs hue/value well but loses some saturation. "
        "  >>> SYNTHESIS FINDING: the §4(c) decomposition (HSV ⊕ name-semantic) is geometrically "
        "REAL — but it is invisible to ARD because the perceptual subspace is variance-poor; "
        "you must gauge-fix HSV to make the name-semantic axes emerge."
    )
    axCap.text(0.01, 0.95, caption, fontsize=10, va="top", ha="left",
               wrap=True,
               bbox=dict(facecolor="#f8f9fa", edgecolor="#adb5bd",
                         boxstyle="round,pad=0.6"))

    fig.suptitle(
        "Cogito-L40 color manifold: a direct sum of a perceptual ring and "
        "a name-semantic 3-frame  —  six-panel synthesis (auto_95)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
