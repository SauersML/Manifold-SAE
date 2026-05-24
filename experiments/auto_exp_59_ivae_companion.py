"""auto_exp_59: iVAE-companion gauge-fix on cogito-L40.

Replaces the auto_exp_38 "fit_aux_supervised_hsv + PCA residual" hack with a
principled iVAE-based identifiability primitive:

  - Supervised block (3 axes): iVAE conditional Gaussian log-prior with
    mean μ_i(HSV) and scale σ_i(HSV) from piecewise-linear smooths
    (Khemakhem 2107.10098).
  - Free block (3 axes): standard N(0, 1) prior.
  - Decoder W: mechanism-sparsity Jacobian column-2-norm penalty
    (Lachapelle 2401.04890).

If the 2026 unified identifiability theorem (2512.05534) is right, the iVAE
auxiliary-conditional prior alone is enough to gauge-fix the supervised
block, and the mechanism-sparsity-induced column structure on the free
block lets it freely concentrate on whatever residual structure
maximises sparsity — empirically the name-semantic axes (modifier count,
monoword, template σ).

Falsifiable comparison vs auto_exp_38:
  - (a) supervised-block R²(hue, sat, val) — must match or beat 0.65 / 0.55 / 0.55
  - (b) free-block max corr with each of {monoword, mod_count, template_σ}
        must be ≥ what auto_exp_38 found (the hack hit ~0.67 on mod_count).

Outputs:
  - runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_59.png
  - runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_59.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/user/Manifold-SAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _pca_basis import load_pc_basis  # type: ignore  # noqa: E402
from manifold_sae.identifiable import (  # noqa: E402
    abs_corr,
    identifiable_manifold_sae,
)

RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
RUN_DIR.mkdir(parents=True, exist_ok=True)
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_PNG = RUN_DIR / "auto_exp_59.png"
OUT_JSON = RUN_DIR / "auto_exp_59.json"

N_TEMPLATES = 28
K_PCS = 16
N_SUP = 3
N_FREE = 3

AUX_LABELS_HSV = ["hue", "sat", "val"]
AUX_LABELS_NAME = ["monoword", "mod_count", "template_sigma"]


def load_xkcd_rgb(n_colors: int):
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append(
                (
                    int(hexs[0:2], 16) / 255.0,
                    int(hexs[2:4], 16) / 255.0,
                    int(hexs[4:6], 16) / 255.0,
                )
            )
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def per_color_stats(X, n_t, basis, k_pcs):
    n_rows, _ = X.shape
    n_c = n_rows // n_t
    mu = basis["mu"]; sigma = basis["sigma"]; Vt = basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t; e = ce * n_t
        chunk = np.asarray(X[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T
        Z = Z[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names, tsig):
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


def main():
    t0 = time.time()
    print("[auto_exp_59] iVAE+MechSparsity identifiable manifold SAE on cogito-L40")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    print("[pca] basis loaded K=64")

    T0, tsig = per_color_stats(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)
    print(f"[aux] hsv={hsv.shape} (supervised) namef={namef.shape} (held-out)")

    # ---- One-shot identifiable fit
    # Standardise aux for stable smooth fitting
    aux_mu = hsv.mean(0, keepdims=True)
    aux_sd = hsv.std(0, keepdims=True).clip(min=1e-8)
    aux_hsv_std = (hsv - aux_mu) / aux_sd

    fit = identifiable_manifold_sae(
        X=T0,
        aux_hsv=aux_hsv_std,
        n_supervised=N_SUP,
        n_free=N_FREE,
        weight_recon=1.0,
        weight_ivae=60.0,
        weight_free_prior=1.0e-3,
        weight_mech=5.0e-3,
        epsilon_mech=1.0e-6,
        n_centres=8,
        n_iter=250,
        smooth_refit_every=5,
        sigma_floor=0.15,
        seed=59,
    )
    print(f"[fit] used_rust={fit.used_rust}  final loss={fit.losses[-1]['total']:.3f}")

    # ---- Eval: per-axis correlations and supervised R²
    T_all = fit.T
    corr_hsv = abs_corr(T_all, hsv)   # (6, 3)
    corr_name = abs_corr(T_all, namef)  # (6, 3)
    print("[corr] |corr(latent, HSV)| =")
    print(np.round(corr_hsv, 2))
    print("[corr] |corr(latent, name-features)| =")
    print(np.round(corr_name, 2))

    # Best-axis R² wrt each HSV component (uses per-axis OLS)
    def per_target_r2(T, y):
        Tc = T - T.mean(0, keepdims=True); yc = y - y.mean()
        beta, *_ = np.linalg.lstsq(Tc, yc, rcond=None)
        pred = Tc @ beta
        return float(1.0 - np.sum((yc - pred) ** 2) / max(np.sum(yc ** 2), 1e-12))

    r2_hsv_sup = np.array(
        [per_target_r2(T_all[:, :N_SUP], hsv[:, i]) for i in range(3)]
    )
    r2_name_free = np.array(
        [per_target_r2(T_all[:, N_SUP:], namef[:, i]) for i in range(3)]
    )
    print(f"[R²] HSV from supervised block: {r2_hsv_sup}")
    print(f"[R²] name-features from FREE block (held-out): {r2_name_free}")

    free_axes = list(range(N_SUP, N_SUP + N_FREE))
    free_axis_max_corr_name = [float(corr_name[j].max()) for j in free_axes]
    print(f"[free axes] max name-corr per free axis: {free_axis_max_corr_name}")

    # ---- Hypotheses vs auto_exp_38 baseline
    baseline_free_max = 0.67  # auto_exp_38 mod_count alignment on axis 4
    hyps = {
        "r2_hue_ge_0.65": bool(r2_hsv_sup[0] >= 0.65),
        "r2_sat_ge_0.55": bool(r2_hsv_sup[1] >= 0.55),
        "r2_val_ge_0.55": bool(r2_hsv_sup[2] >= 0.55),
        "any_free_axis_aligns_name": bool(max(free_axis_max_corr_name) >= 0.50),
        "matches_or_beats_auto_38_baseline": bool(
            max(free_axis_max_corr_name) >= baseline_free_max - 0.05
        ),
    }
    print(f"[hyps] {hyps}")

    # ---- Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    ax = axs[0, 0]
    im = ax.imshow(corr_hsv, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_HSV)
    ax.set_yticks(range(N_SUP + N_FREE))
    ax.set_yticklabels(
        [f"axis {j}{' [SUP]' if j < N_SUP else ' [FREE]'}" for j in range(N_SUP + N_FREE)]
    )
    ax.set_title("|corr(latent, HSV)|")
    for j in range(N_SUP + N_FREE):
        for k in range(3):
            ax.text(k, j, f"{corr_hsv[j, k]:.2f}", ha="center", va="center",
                    color="white" if corr_hsv[j, k] < 0.6 else "black", fontsize=9)
    ax.axhline(N_SUP - 0.5, color="red", lw=1.5, ls="--")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axs[0, 1]
    free_name_block = corr_name[free_axes]
    im2 = ax.imshow(free_name_block, vmin=0, vmax=1.0, cmap="magma", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_NAME, rotation=20, ha="right")
    ax.set_yticks(range(N_FREE)); ax.set_yticklabels([f"free axis {j}" for j in free_axes])
    ax.set_title("|corr(FREE axis, name-feature)| — held out")
    for j in range(N_FREE):
        for k in range(3):
            ax.text(k, j, f"{free_name_block[j, k]:.2f}", ha="center", va="center",
                    color="white" if free_name_block[j, k] < 0.6 else "black", fontsize=10)
    fig.colorbar(im2, ax=ax, shrink=0.85)

    ax = axs[1, 0]
    losses = fit.losses
    iters = [r["iter"] for r in losses]
    ax.plot(iters, [r["total"] for r in losses], label="total")
    ax.plot(iters, [r["recon"] for r in losses], label="recon")
    ax.plot(iters, [r["ivae"] for r in losses], label="ivae")
    ax.plot(iters, [r["mech"] for r in losses], label="mech")
    ax.set_xlabel("iter"); ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("training losses")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    labels = AUX_LABELS_HSV + AUX_LABELS_NAME
    vals = list(r2_hsv_sup) + list(r2_name_free)
    bcolors = ["#d62728"] * 3 + ["#9467bd"] * 3
    ax.bar(range(len(vals)), vals, color=bcolors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² (block→target)")
    ax.set_title("Supervised R² (HSV from sup block) | Held-out R² (name from free block)")
    ax.axhline(0.65, ls=":", color="k", lw=0.8, label="hue target 0.65")
    ax.set_ylim(min(0, min(vals) - 0.05), 1.0)
    ax.legend(fontsize=8)

    fig.suptitle("auto_exp_59 — iVAE + Mechanism-Sparsity identifiable manifold SAE", fontsize=13)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] wrote {OUT_PNG}")

    out = {
        "experiment": "auto_exp_59_ivae_companion",
        "used_rust_ffi": bool(fit.used_rust),
        "n_supervised": N_SUP,
        "n_free": N_FREE,
        "n_iter": len(fit.losses),
        "final_loss": fit.losses[-1],
        "r2_hsv_from_sup": r2_hsv_sup.tolist(),
        "r2_name_from_free": r2_name_free.tolist(),
        "corr_hsv": corr_hsv.tolist(),
        "corr_name": corr_name.tolist(),
        "free_axis_max_name_corr": free_axis_max_corr_name,
        "hypotheses": hyps,
        "elapsed_s": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] wrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
