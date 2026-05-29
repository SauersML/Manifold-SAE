"""auto_67.py — Gauge-free supervised-ceiling decomposition.

Tests whether the kNN-Lab ceiling on Z_top64 (R^2 ≈ 0.246), which is
already matched by a 1D periodic Duchon on hue alone (auto_66: R^2 ≈
0.251 at n_centers=40), gains any incremental capacity from a smooth
(sat, val) term g(sat,val).

Model:   Z ≈ γ(hue) + g(sat, val)
  γ : 1D periodic Duchon, m=2, n_centers=40
      gamfit.duchon_basis(hue, centers, m=2, periodic_per_axis=[True])
  g : 2D non-periodic Duchon on [0,1]^2 with a 6×6 uniform grid (36
      centers). gamfit's pure-Duchon m=2 requires 2*(p+s) > dim+2, so
      in dim=2 we lift the nullspace order to "degree2" (the natural
      thin-plate analog) and keep m=2 — still pure Duchon, still
      length_scale=None.

Fit:  single gaussian_reml_fit on Phi = [Phi_hue | Phi_sv] with
      block-diagonal penalty P = blockdiag(P_hue, P_sv). One REML λ
      governs both blocks (so both terms see the same shrinkage trade).

Report:
  * 5-fold macro CV R^2 for: hue-only, sv-only, joint hue+sv
  * incremental R^2 = joint - hue-only
  * per-PC CV R^2 (heatmap)
  * γ(hue) curve for the top-5 most-color-predictable PCs (joint fit)

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
N_FOLDS = 5
K_PC = 64

HUE_CENTERS = 40
SV_GRID = 6  # 6x6 = 36 centers on [0,1]^2


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def r2_per_col(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    out = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    return out


def hue_basis(hue01, n_centers=HUE_CENTERS):
    """1D periodic Duchon m=2 on the hue circle."""
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    )
    return Phi, P, centers


def sv_basis(sat, val, grid=SV_GRID):
    """2D Duchon m=2 on [0,1]^2 (non-periodic) with degree-2 nullspace.

    Pure Duchon (length_scale=None). Centers are a uniform grid x grid mesh.
    """
    import gamfit
    g = np.linspace(0.0, 1.0, grid)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, nullspace_order="degree2")
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, nullspace_order="degree2")
    )
    return Phi, P, centers


def reml_fit(Phi, Y, P):
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return np.asarray(out["coefficients"]), float(out["lambda"])


def main():
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[load] X mmap shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = len(kept)
    print(f"[load] N={N} filtered colors")

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    print(f"[load] Z shape={Z.shape}  EVR_top{K_PC}={float(basis['evr'].sum()):.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # ---- Build full bases on all data (per-fold subselect rows) ----
    Phi_h_full, P_h, _ = hue_basis(hue)
    Phi_sv_full, P_sv, sv_centers = sv_basis(sat, val)
    Kh = Phi_h_full.shape[1]
    Ksv = Phi_sv_full.shape[1]
    print(f"[basis] Phi_hue cols={Kh}  Phi_sv cols={Ksv}")

    # Block-diag penalty for joint fit (single shared λ via gaussian_reml_fit)
    P_joint = sla.block_diag(P_h, P_sv)
    Phi_joint_full = np.concatenate([Phi_h_full, Phi_sv_full], axis=1)

    preds = {"hue": np.zeros_like(Z), "sv": np.zeros_like(Z), "joint": np.zeros_like(Z)}
    lambdas = {"hue": [], "sv": [], "joint": []}

    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        # hue-only
        Bh, lh = reml_fit(Phi_h_full[tr], Z[tr], P_h)
        preds["hue"][te] = Phi_h_full[te] @ Bh
        lambdas["hue"].append(lh)
        # sv-only
        Bs, ls = reml_fit(Phi_sv_full[tr], Z[tr], P_sv)
        preds["sv"][te] = Phi_sv_full[te] @ Bs
        lambdas["sv"].append(ls)
        # joint
        Bj, lj = reml_fit(Phi_joint_full[tr], Z[tr], P_joint)
        preds["joint"][te] = Phi_joint_full[te] @ Bj
        lambdas["joint"].append(lj)
        print(f"[fold {f}] λ hue={lh:.3g}  sv={ls:.3g}  joint={lj:.3g}")

    macro = {k: float(r2_macro(Z, v)) for k, v in preds.items()}
    per_pc = {k: r2_per_col(Z, v) for k, v in preds.items()}
    incr_sv_over_hue = macro["joint"] - macro["hue"]

    print("\n[CV macro R^2]")
    for k in ("hue", "sv", "joint"):
        print(f"   {k:6s}  R^2 = {macro[k]:+.4f}   λ̄ = {np.mean(lambdas[k]):.3g}")
    print(f"   incremental (joint − hue) = {incr_sv_over_hue:+.4f}")

    # Per-PC differential (which PCs benefit most from sat/val on top of hue)
    delta_pc = per_pc["joint"] - per_pc["hue"]
    top_color_pcs = np.argsort(-per_pc["joint"])[:5]
    top_delta_pcs = np.argsort(-delta_pc)[:5]
    print(f"[top-5 joint R^2 PCs] {top_color_pcs.tolist()}  "
          f"R^2={per_pc['joint'][top_color_pcs].round(3).tolist()}")
    print(f"[top-5 Δ(joint-hue) PCs] {top_delta_pcs.tolist()}  "
          f"Δ={delta_pc[top_delta_pcs].round(3).tolist()}")

    # In-sample joint fit for γ(hue) curves (gauge-free decomposition)
    B_joint_full, _ = reml_fit(Phi_joint_full, Z, P_joint)
    Bh_part = B_joint_full[:Kh]
    Bsv_part = B_joint_full[Kh:]
    hue_dense = np.linspace(0.0, 1.0, 360, endpoint=False)
    Phi_h_dense, _, _ = hue_basis(hue_dense)
    gamma_dense = Phi_h_dense @ Bh_part  # (360, K_PC)

    # ---- Plot ----
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.2])

    # (1) bar chart of macro R^2
    ax = fig.add_subplot(gs[0, 0])
    names = ["hue γ", "sat,val g", "joint γ+g"]
    vals = [macro["hue"], macro["sv"], macro["joint"]]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    ax.bar(names, vals, color=colors, edgecolor="black")
    ax.axhline(0.246, color="green", ls="--", lw=1, label="kNN-Lab ceiling 0.246")
    ax.axhline(0.237, color="purple", ls=":", lw=1, label="L_joint_lab 0.237")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    ax.set_ylabel("CV macro R²  (target Z_top64)")
    ax.set_title(f"(a) Gauge-free supervised ceiling decomposition\n"
                 f"Δ_sv = {incr_sv_over_hue:+.4f}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")

    # (2) per-PC R^2 heatmap
    ax = fig.add_subplot(gs[0, 1:])
    M = np.stack([per_pc["hue"], per_pc["sv"], per_pc["joint"], delta_pc], axis=0)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   vmin=-float(np.abs(M).max()), vmax=float(np.abs(M).max()))
    ax.set_yticks(range(4))
    ax.set_yticklabels(["hue γ", "sat,val g", "joint γ+g", "Δ (joint−hue)"])
    ax.set_xlabel(f"PC index (0 .. {K_PC-1})")
    ax.set_title("Per-PC CV R²")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # (3) γ(hue) for top-5 joint-R^2 PCs
    ax = fig.add_subplot(gs[1, :])
    theta = hue_dense * 2 * np.pi
    cmap_hsv = plt.get_cmap("hsv")
    for rank, pc in enumerate(top_color_pcs):
        ax.plot(hue_dense, gamma_dense[:, pc], lw=2,
                label=f"PC{pc}  (joint R²={per_pc['joint'][pc]:+.2f}, "
                      f"hue R²={per_pc['hue'][pc]:+.2f})")
    # color hue axis with HSV strip at bottom
    yl = ax.get_ylim()
    for h in np.linspace(0, 1, 60, endpoint=False):
        c = colorsys.hsv_to_rgb(h, 1, 1)
        ax.axvspan(h, h + 1/60, ymin=0.0, ymax=0.03, color=c, alpha=1.0)
    ax.set_ylim(yl)
    ax.set_xlim(0, 1)
    ax.set_xlabel("hue ∈ [0,1)")
    ax.set_ylabel("γ(hue) component (Z-units)")
    ax.set_title(f"(c) γ(hue) curves for top-5 joint-R² PCs "
                 f"(from joint REML fit; sat/val partialled out)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # (4) Δ-per-PC bar (top 20)
    ax = fig.add_subplot(gs[2, 0])
    order = np.argsort(-delta_pc)[:20]
    ax.bar(range(20), delta_pc[order],
           color=["#1f77b4" if d > 0 else "#d62728" for d in delta_pc[order]])
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(p) for p in order], rotation=60, fontsize=7)
    ax.set_xlabel("PC")
    ax.set_ylabel("Δ R²  (joint − hue)")
    ax.set_title("(d) Top-20 PCs by sat/val incremental R²")
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)

    # (5) scatter joint vs hue per-PC
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(per_pc["hue"], per_pc["joint"], s=24,
               c=delta_pc, cmap="RdBu_r",
               vmin=-float(np.abs(delta_pc).max()), vmax=float(np.abs(delta_pc).max()),
               edgecolor="black", linewidth=0.3)
    lo = min(per_pc["hue"].min(), per_pc["joint"].min())
    hi = max(per_pc["hue"].max(), per_pc["joint"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("hue-only R²")
    ax.set_ylabel("joint R²")
    ax.set_title("(e) Per-PC: joint vs hue-only")
    ax.grid(alpha=0.3)

    # (6) summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    txt = (
        f"N = {N} filtered xkcd colors\n"
        f"Z = top-{K_PC} sklearn-PCA of cogito L40\n"
        f"  EVR cum = {float(basis['evr'].sum()):.3f}\n\n"
        f"γ : 1D periodic Duchon m=2, n_centers={HUE_CENTERS}\n"
        f"g : 2D Duchon m=2 (degree2 nullspace),\n"
        f"    {SV_GRID}×{SV_GRID}={SV_GRID*SV_GRID} centers on [0,1]²\n"
        f"single REML, block-diag penalty\n\n"
        f"5-fold macro CV R²:\n"
        f"   hue γ      : {macro['hue']:+.4f}\n"
        f"   sat,val g  : {macro['sv']:+.4f}\n"
        f"   joint γ+g  : {macro['joint']:+.4f}\n"
        f"   Δ (joint−hue) = {incr_sv_over_hue:+.4f}\n\n"
        f"top-5 joint R² PCs:\n"
        f"   {top_color_pcs.tolist()}\n"
        f"top-5 Δ PCs:\n"
        f"   {top_delta_pcs.tolist()}\n"
        f"   Δ vals = {[round(float(delta_pc[i]),3) for i in top_delta_pcs]}\n"
    )
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9,
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(
        f"auto_67 · ceiling decomposition  Z ≈ γ(hue) + g(sat,val) · cogito L40",
        fontsize=13,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "auto_67.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    payload = {
        "n_colors": int(N),
        "K_PC": K_PC,
        "evr_top_K_PC": float(basis["evr"].sum()),
        "hue_centers": HUE_CENTERS,
        "sv_grid": SV_GRID,
        "sv_centers_total": int(SV_GRID * SV_GRID),
        "hue_basis_cols": int(Kh),
        "sv_basis_cols": int(Ksv),
        "cv_macro_r2": {k: float(v) for k, v in macro.items()},
        "incremental_r2_satval_over_hue": float(incr_sv_over_hue),
        "mean_lambda": {k: float(np.mean(v)) for k, v in lambdas.items()},
        "fold_lambdas": {k: [float(x) for x in v] for k, v in lambdas.items()},
        "per_pc_r2_hue": per_pc["hue"].tolist(),
        "per_pc_r2_sv": per_pc["sv"].tolist(),
        "per_pc_r2_joint": per_pc["joint"].tolist(),
        "per_pc_delta_joint_minus_hue": delta_pc.tolist(),
        "top5_joint_r2_pcs": [int(i) for i in top_color_pcs],
        "top5_joint_r2_values": [float(per_pc["joint"][i]) for i in top_color_pcs],
        "top5_delta_pcs": [int(i) for i in top_delta_pcs],
        "top5_delta_values": [float(delta_pc[i]) for i in top_delta_pcs],
        "notes": (
            "2D Duchon m=2 requires nullspace_order='degree2' in dim=2 "
            "(pure Duchon, no length_scale). 1D hue is periodic Duchon m=2. "
            "Joint fit: single gaussian_reml_fit with block-diagonal penalty "
            "(one shared λ). No B-splines, no Gaussian RBF."
        ),
    }
    (OUT_DIR / "auto_67.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_67.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
