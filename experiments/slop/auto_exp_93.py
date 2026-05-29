"""auto_exp_93: visualize the name-semantic latent subspace from auto_exp_54.

auto_exp_54 found that the HSV gauge-fix recipe (RRR on cogito-L40 centroids,
K=16 PCs, ridge λ=1.0) ALSO works on non-perceptual targets:
    modifier_count R²_CV=0.763  monoword R²_CV=0.733  template_σ R²_CV=0.620
all exceeding HSV hue's R²=0.700.

This script plots the recovered 949 × 3 latent T_namesem and quantifies its
relation to {hue, sat, val, R, G, B, mod_count, monoword, template_σ}.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (
    X_PATH, N_TEMPLATES, K_PCS,
    per_color_stats_mmap, load_xkcd_rgb, hsv_from_rgb,
)
from auto_exp_53 import fit_rrr

ROOT = Path("/Users/user/Manifold-SAE")
NPZ_54 = ROOT / "runs" / "auto_exp_54_nonhsv_gauge.npz"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_93.png"
RIDGE_LAM = 1.0


def main():
    print("[auto_exp_93] loading T_namesem from auto_exp_54 npz")
    d54 = np.load(NPZ_54, allow_pickle=True)
    T_ns = d54["T_joint"]            # (949, 3)
    targets54 = list(d54["targets"]) # ['modifier_count','monoword','template_sigma']
    print(f"  T_namesem shape={T_ns.shape}  targets={targets54}")

    print("[data] loading cogito-L40 centroids + xkcd labels")
    X = np.load(X_PATH, mmap_mode="r")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    name_token_count = np.array([len(n.split()) for n in names], dtype=np.float64)
    print(f"  T0={T0.shape} rgb={rgb.shape}  n_c={n_c}")

    # ---- Recompute HSV gauge-fit (auto_exp_38 recipe) to get U_hsv (949, 3) ----
    Tc = T0 - T0.mean(0, keepdims=True)
    Y_hsv = hsv.copy()
    y_mu = Y_hsv.mean(0, keepdims=True)
    y_sd = Y_hsv.std(0, keepdims=True).clip(min=1e-8)
    Y_std = (Y_hsv - y_mu) / y_sd
    Yc = Y_std - Y_std.mean(0, keepdims=True)
    fit_hsv = fit_rrr(Tc, Yc, 3, lam=RIDGE_LAM)
    T_hsv = fit_hsv["T"]              # (949, 3) latent in HSV-supervised space
    print(f"  T_hsv (auto_exp_38 recipe) shape={T_hsv.shape}")

    # ----------- Cross-correlation matrix -----------
    feats = {
        "hue": hsv[:, 0], "sat": hsv[:, 1], "val": hsv[:, 2],
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "mod_count": modc, "monoword": mono, "template_sigma": tsig,
        "name_token_count": name_token_count,
    }
    feat_names = list(feats.keys())
    n_ax = T_ns.shape[1]
    corr = np.zeros((n_ax, len(feat_names)))
    pval = np.zeros_like(corr)
    for i in range(n_ax):
        for j, fn in enumerate(feat_names):
            r, p = spearmanr(T_ns[:, i], feats[fn])
            corr[i, j] = r
            pval[i, j] = p
    # Also: mod_count axis (T_ns[:,0]) vs HSV-gauge hue axis (T_hsv[:,0])
    r_mc_hue, _ = spearmanr(T_ns[:, 0], T_hsv[:, 0])
    r_mc_hue_raw, _ = spearmanr(T_ns[:, 0], hsv[:, 0])

    print("\n=== Spearman corr: T_namesem.axes vs features ===")
    hdr = f"{'axis':>20} | " + " ".join(f"{fn:>10}" for fn in feat_names)
    print(hdr)
    for i in range(n_ax):
        label = f"axis{i} ({targets54[i]})"
        row = f"{label:>20} | " + " ".join(f"{corr[i,j]:>+10.3f}" for j in range(len(feat_names)))
        print(row)
    print(f"\nT_namesem axis0 (mod_count) vs T_hsv axis0 (hue-gauge): r={r_mc_hue:+.3f}")
    print(f"T_namesem axis0 (mod_count) vs raw hue:                  r={r_mc_hue_raw:+.3f}")

    # ----------- Plot -----------
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1.0],
                          hspace=0.42, wspace=0.32)

    # Row 1: 3 pairwise panels of T_ns coloured by xkcd RGB
    rgb_clip = np.clip(rgb, 0, 1)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(T_ns[:, a], T_ns[:, b], c=rgb_clip, s=14,
                   edgecolor="black", linewidth=0.15)
        ax.set_xlabel(f"axis{a} ({targets54[a]})")
        ax.set_ylabel(f"axis{b} ({targets54[b]})")
        ax.set_title(f"T_namesem axes {a}-{b}  |  xkcd RGB")
        ax.grid(alpha=0.3)

    # Row 1 col 3: 3D scatter coloured by xkcd RGB
    ax3d = fig.add_subplot(gs[0, 3], projection="3d")
    ax3d.scatter(T_ns[:, 0], T_ns[:, 1], T_ns[:, 2],
                 c=rgb_clip, s=12, depthshade=False,
                 edgecolor="black", linewidth=0.1)
    ax3d.set_xlabel(f"a0 {targets54[0]}")
    ax3d.set_ylabel(f"a1 {targets54[1]}")
    ax3d.set_zlabel(f"a2 {targets54[2]}")
    ax3d.set_title("T_namesem 3D (xkcd RGB)")

    # Row 2: same panels coloured by modifier_count
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=max(3, modc.max()))
    for k, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, k])
        sc = ax.scatter(T_ns[:, a], T_ns[:, b], c=modc, cmap=cmap, norm=norm,
                        s=14, edgecolor="black", linewidth=0.15)
        ax.set_xlabel(f"axis{a} ({targets54[a]})")
        ax.set_ylabel(f"axis{b} ({targets54[b]})")
        ax.set_title(f"T_namesem axes {a}-{b}  |  modifier_count")
        ax.grid(alpha=0.3)
    cax = fig.add_subplot(gs[1, 3])
    cax.set_axis_off()
    # Side-by-side: mod_count axis vs HSV-gauge hue axis
    inset = fig.add_axes([0.78, 0.40, 0.18, 0.20])
    inset.scatter(T_ns[:, 0], T_hsv[:, 0], c=rgb_clip, s=14,
                  edgecolor="black", linewidth=0.15)
    inset.set_xlabel(f"T_namesem axis0\n({targets54[0]})")
    inset.set_ylabel("T_hsv axis0\n(hue-gauge)")
    inset.set_title(f"mod_count axis vs hue axis\nSpearman r={r_mc_hue:+.3f}")
    inset.grid(alpha=0.3)
    fig.colorbar(sc, ax=cax, fraction=0.6, label="modifier_count")

    # Row 3: correlation heatmap
    ax_h = fig.add_subplot(gs[2, :3])
    im = ax_h.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_h.set_xticks(range(len(feat_names)))
    ax_h.set_xticklabels(feat_names, rotation=35, ha="right")
    ax_h.set_yticks(range(n_ax))
    ax_h.set_yticklabels([f"axis{i}\n({targets54[i]})" for i in range(n_ax)])
    for i in range(n_ax):
        for j in range(len(feat_names)):
            star = "*" if pval[i, j] < 0.001 else ""
            ax_h.text(j, i, f"{corr[i,j]:+.2f}{star}",
                      ha="center", va="center",
                      color="white" if abs(corr[i, j]) > 0.5 else "black",
                      fontsize=8)
    fig.colorbar(im, ax=ax_h, label="Spearman ρ", shrink=0.85)
    ax_h.set_title("Spearman correlation: T_namesem axes vs perceptual + name features"
                   "  (* = p<0.001)")

    # bottom-right summary panel
    ax_s = fig.add_subplot(gs[2, 3])
    ax_s.axis("off")
    lines = [
        "auto_exp_93 — name-semantic latent",
        "",
        f"n_colors = {n_c}, axes = {n_ax}",
        "Recipe: RRR on cogito-L40 (K=16 PCs),",
        "supervised by [mod_count, monoword,",
        " template_sigma]  (auto_exp_54).",
        "",
        f"axis0 vs hue (raw):    r={r_mc_hue_raw:+.3f}",
        f"axis0 vs T_hsv hue ax: r={r_mc_hue:+.3f}",
        "",
        "Top |ρ| per axis:",
    ]
    for i in range(n_ax):
        order = np.argsort(-np.abs(corr[i]))
        top = ", ".join(f"{feat_names[j]}={corr[i,j]:+.2f}" for j in order[:3])
        lines.append(f" axis{i}: {top}")
    ax_s.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.suptitle("Name-semantic latent subspace from cogito-L40 (auto_exp_54 → auto_exp_93)",
                 fontsize=14, y=0.995)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[png] saved {OUT_PNG}")

    # Persist a small npz of the cross-correlations
    out_npz = ROOT / "runs" / "auto_exp_93_namesem_corr.npz"
    np.savez(out_npz, corr=corr, pval=pval,
             feat_names=np.array(feat_names),
             axis_targets=np.array(targets54),
             T_namesem=T_ns, T_hsv=T_hsv,
             r_axis0_vs_T_hsv_hue=r_mc_hue,
             r_axis0_vs_raw_hue=r_mc_hue_raw)
    print(f"[npz] saved {out_npz}")


if __name__ == "__main__":
    main()
