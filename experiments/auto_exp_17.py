"""auto_exp_17 — Unified composition-engine validation on cogito L40 colors.

GOAL
----
Test the claim from project_gamfit_composition_engine.md that a single call
to gamfit's new composition-engine primitives (`LatentCoord`, joint REML,
topology selection) reproduces the hand-rolled stack we built across
auto_67 + auto_74 (HSV-supervised hue+sv + ICA-on-residual, CV R² 0.61
ceiling, U_3d alternating fit R² 0.608).

DESIGN
------
The new API (`LatentCoord`, `select_topology`, `sae_manifold_fit`) lives in
the gamfit source tree and is shipping in v0.1.114 (currently 0.1.112 is on
PyPI). So this script must:

  - graceful-degrade detection via try/except ImportError
  - if HAS_NEW_API: run gamfit.fit(..., latents=LatentCoord(d=3, manifold='circle'))
    joint REML on the 886 centroids, score CV R² in PC space, compare
  - if NOT HAS_NEW_API: run the auto_74 pipeline as the "predictive
    baseline" (the value the new pipeline would produce if it worked as
    claimed) and stash the gamfit-fit slot as null in the JSON.
    When v0.1.114 lands, re-run; the gamfit block is filled in automatically.

This is the LLM-applicability gate: if the composition engine can't match
the hand-rolled R² on this dataset, it's not a credible replacement for
the auto_67 + auto_74 stack.

RAM
---
- mmap=r on X_L40.npy (760 MB)
- cached PC basis via _pca_basis.load_pc_basis(K=64); we use top-16 PCs
- never load the harvest fully into memory
"""

from __future__ import annotations

import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from sklearn.decomposition import FastICA

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_17.png"
OUT_JSON = OUT_DIR / "auto_exp_17.json"

K_PC          = 16
HUE_CENTERS   = 40
SV_GRID       = 6
N_FOLDS       = 5
N_ICA         = 6
RGB_BINS      = 2          # 2^3 = 8 octants
LATENT_D      = 3          # request 3 manifold dims (matches U_3d ceiling)
SEED          = 0

# Cached known-baselines (gauge-free supervised CV R² from prior auto_*):
REF_HUE_SV_R2_AUTO_67 = 0.32
REF_U3D_R2_AUTO_EXP_06 = 0.608
REF_HSV_ICA_R2_AUTO_74 = 0.61   # ~ U_3d ceiling


# ---------------------------------------------------------------------------
# New composition-engine API import (v0.1.114+ only)
# ---------------------------------------------------------------------------
HAS_NEW_API = False
NEW_API_REASON = ""
try:
    from gamfit import LatentCoord, select_topology, sae_manifold_fit  # noqa: F401
    HAS_NEW_API = True
    NEW_API_REASON = "v0.1.114 primitives present"
except ImportError as e:
    NEW_API_REASON = f"ImportError: {e}"

import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")


# ---------------------------------------------------------------------------
# Centroid builder (mmap)
# ---------------------------------------------------------------------------
def build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    n_raw = n_total // N_TEMPLATES
    centroids = np.zeros((n_raw, H), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_TEMPLATES + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(axis=0)
    del X  # release mmap
    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    names = [n for n, *_ in kept]
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, names, rgb, hsv


# ---------------------------------------------------------------------------
# Bases (auto_67 / auto_74 spec)
# ---------------------------------------------------------------------------
def hue_basis(hue01):
    centers = np.linspace(0.0, 1.0, HUE_CENTERS, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(gamfit.duchon_basis(
        pts, centers, m=2, periodic_per_axis=[True]))
    P = np.asarray(gamfit.duchon_function_norm_penalty(
        centers, m=2, periodic_per_axis=[True]))
    return Phi, P


def sv_basis(sat, val):
    g = np.linspace(0.0, 1.0, SV_GRID)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi = np.asarray(gamfit.duchon_basis(
        pts, centers, m=2, nullspace_order="degree2"))
    P = np.asarray(gamfit.duchon_function_norm_penalty(
        centers, m=2, nullspace_order="degree2"))
    return Phi, P


def reml_fit(Phi, Y, P):
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return np.asarray(out["coefficients"]), float(out["lambda"])


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Hand-rolled stack: hue + sv + ICA-on-residual, CV (mirrors auto_74)
# ---------------------------------------------------------------------------
def run_handrolled_stack(Z, hue, sat, val, rgb, fold):
    Phi_h,  P_h  = hue_basis(hue)
    Phi_sv, P_sv = sv_basis(sat, val)
    Phi_joint = np.concatenate([Phi_h, Phi_sv], axis=1)
    P_joint = sla.block_diag(P_h, P_sv)
    N = Z.shape[0]

    pred_hue = np.zeros_like(Z)
    pred_joint = np.zeros_like(Z)
    pred_stack = np.zeros_like(Z)

    bin_id = np.minimum((rgb * RGB_BINS).astype(int), RGB_BINS - 1)
    bin_id = bin_id[:, 0] * RGB_BINS * RGB_BINS + bin_id[:, 1] * RGB_BINS + bin_id[:, 2]

    K_use_ic = min(3, N_ICA)
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        Bh, _ = reml_fit(Phi_h[tr], Z[tr], P_h)
        pred_hue[te] = Phi_h[te] @ Bh
        Bj, _ = reml_fit(Phi_joint[tr], Z[tr], P_joint)
        pred_joint[te] = Phi_joint[te] @ Bj

        # ICA on training residual
        Z_res_tr = Z[tr] - Phi_joint[tr] @ Bj
        Z_res_te = Z[te] - Phi_joint[te] @ Bj
        ica = FastICA(n_components=N_ICA, whiten="unit-variance",
                      random_state=0, max_iter=2000, tol=1e-5)
        S_tr = ica.fit_transform(Z_res_tr)
        # MI on training to pick top-K_use_ic
        bid = bin_id[tr]
        uniq = np.unique(bid)
        S_std = (S_tr - S_tr.mean(0)) / S_tr.std(0).clip(min=1e-12)
        MI = np.zeros(N_ICA)
        for k in range(N_ICA):
            s = S_std[:, k]
            means = np.array([s[bid == u].mean() for u in uniq])
            wvar = np.array([s[bid == u].var() for u in uniq])
            wts = np.array([(bid == u).sum() for u in uniq], float)
            wts /= wts.sum()
            Es_var = float((wvar * wts).sum())
            Var_Es = float(((means - (means * wts).sum()) ** 2 * wts).sum())
            MI[k] = Var_Es / max(Es_var, 1e-12)
        sel = np.argsort(-MI)[:K_use_ic]
        # Project test residuals via unmixing
        S_te = (Z_res_te - ica.mean_) @ ica.components_.T
        sm, ss = S_tr.mean(0), S_tr.std(0).clip(min=1e-12)
        S_tr_top = ((S_tr - sm) / ss)[:, sel]
        S_te_top = ((S_te - sm) / ss)[:, sel]
        Phi_stack_tr = np.concatenate([Phi_joint[tr], S_tr_top], axis=1)
        Phi_stack_te = np.concatenate([Phi_joint[te], S_te_top], axis=1)
        P_stack = sla.block_diag(P_joint, np.zeros((K_use_ic, K_use_ic)))
        Bs, _ = reml_fit(Phi_stack_tr, Z[tr], P_stack)
        pred_stack[te] = Phi_stack_te @ Bs

    return {
        "r2_hue_cv": r2_macro(Z, pred_hue),
        "r2_hue_sv_cv": r2_macro(Z, pred_joint),
        "r2_stack_cv": r2_macro(Z, pred_stack),
    }


# ---------------------------------------------------------------------------
# Unified composition-engine fit (only runs if HAS_NEW_API)
# ---------------------------------------------------------------------------
def run_unified_gamfit(Z, fold):
    """Joint REML on per-row latent t_i ∈ S¹ × R² (circle + 2 euclid)."""
    # We attempt the most likely v0.1.114 surface; if any signature mismatch
    # we catch and return a structured failure so the JSON is honest.
    N = Z.shape[0]
    pred = np.zeros_like(Z)
    err = None
    try:
        for f in range(N_FOLDS):
            tr = fold != f
            te = ~tr
            latents = LatentCoord(d=LATENT_D, manifold="circle")  # noqa: F821
            model = gamfit.fit(
                Y=Z[tr],
                latents=latents,
                method="reml",
                seed=SEED,
            )
            # Project / predict on held-out rows. The unified API exposes
            # both a posterior latent inference and a `predict` step.
            pred[te] = np.asarray(model.predict(Z[te]))
        return {"r2_unified_cv": r2_macro(Z, pred), "error": None}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return {"r2_unified_cv": None, "error": err}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    print(f"[api ] HAS_NEW_API = {HAS_NEW_API}  ({NEW_API_REASON})", flush=True)

    centroids, names, rgb, hsv = build_centroids()
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC]
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = "
          f"{float(basis['evr'][:K_PC].sum()):.3f}", flush=True)

    # CV fold assignment
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # --- Hand-rolled baseline (always runs) ---
    print("[hand] running auto_67 + auto_74 stack ...", flush=True)
    hr = run_handrolled_stack(Z, hue, sat, val, rgb, fold)
    print(f"[hand] CV R² : hue={hr['r2_hue_cv']:+.3f}  "
          f"hue+sv={hr['r2_hue_sv_cv']:+.3f}  "
          f"hue+sv+ICA={hr['r2_stack_cv']:+.3f}", flush=True)

    # --- Unified composition-engine fit (if v0.1.114) ---
    if HAS_NEW_API:
        print("[gam ] running unified LatentCoord REML ...", flush=True)
        gam = run_unified_gamfit(Z, fold)
        if gam["error"]:
            print(f"[gam ] FAILED: {gam['error']}", flush=True)
        else:
            print(f"[gam ] CV R² = {gam['r2_unified_cv']:+.3f}", flush=True)
    else:
        gam = {"r2_unified_cv": None,
               "error": (f"new API not in gamfit=={GAMFIT_VERSION}; "
                         "re-run when v0.1.114 wheels are on PyPI")}
        print("[gam ] SKIPPED (no LatentCoord); "
              "JSON slot is null for fill-in later", flush=True)

    # --- Honest comparison ---
    # The hand-rolled CV stack IS the predictive baseline for what the
    # unified engine should match. If HAS_NEW_API is True, the delta is
    # the test; if False, the hand-rolled value is the *prediction* for
    # the future re-run, and we record it as such.
    expected_unified = hr["r2_stack_cv"]
    matched = None
    delta = None
    if gam["r2_unified_cv"] is not None:
        delta = gam["r2_unified_cv"] - expected_unified
        # Within 0.03 absolute R² counts as matching the ceiling.
        matched = bool(abs(delta) <= 0.03)

    summary = {
        "gamfit_version": GAMFIT_VERSION,
        "has_new_api": HAS_NEW_API,
        "new_api_reason": NEW_API_REASON,
        "config": {
            "K_PC": K_PC,
            "hue_centers": HUE_CENTERS,
            "sv_grid": SV_GRID,
            "n_folds": N_FOLDS,
            "n_ica": N_ICA,
            "rgb_bins": RGB_BINS,
            "latent_d_request": LATENT_D,
            "n_colors": int(N),
        },
        "handrolled_cv": hr,
        "unified_cv": gam,
        "references": {
            "auto_67_hue_sv_ceiling": REF_HUE_SV_R2_AUTO_67,
            "auto_74_HSV_plus_ICA_ceiling": REF_HSV_ICA_R2_AUTO_74,
            "auto_exp_06_U3d_alternating": REF_U3D_R2_AUTO_EXP_06,
        },
        "verdict": {
            "expected_unified_r2": expected_unified,
            "actual_unified_r2": gam["r2_unified_cv"],
            "delta": delta,
            "matched_handrolled_ceiling_within_0.03": matched,
            "comment": (
                "Hand-rolled stack value is the *prediction* for the "
                "unified-engine re-run when v0.1.114 lands."
                if not HAS_NEW_API else
                ("Unified engine matches hand-rolled ceiling."
                 if matched else
                 "Unified engine fell short of hand-rolled ceiling.")
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bar_names = [
        "hue\nauto_67",
        "hue+sv\nauto_67",
        "hue+sv+ICA\nauto_74 (CV)",
        "U_3d alt\nauto_exp_06",
        "UNIFIED gamfit\nLatentCoord d=3" + ("" if HAS_NEW_API else "\n(PENDING v0.1.114)"),
    ]
    bar_vals = [
        hr["r2_hue_cv"],
        hr["r2_hue_sv_cv"],
        hr["r2_stack_cv"],
        REF_U3D_R2_AUTO_EXP_06,
        gam["r2_unified_cv"] if gam["r2_unified_cv"] is not None else 0.0,
    ]
    bar_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    bars = ax.bar(bar_names, bar_vals, color=bar_colors,
                  edgecolor="black", lw=0.8)
    # Hatch the pending bar
    if not HAS_NEW_API:
        bars[-1].set_hatch("//")
        bars[-1].set_alpha(0.45)
    for i, v in enumerate(bar_vals):
        label = f"{v:+.3f}" if (HAS_NEW_API or i != len(bar_vals) - 1) else "n/a"
        ax.text(i, max(v, 0.01) + 0.012, label, ha="center", fontsize=9)
    ax.axhline(REF_HSV_ICA_R2_AUTO_74, color="green", ls="--", lw=1, alpha=0.6,
               label=f"hand-rolled ceiling (R²={REF_HSV_ICA_R2_AUTO_74})")
    ax.axhline(REF_HUE_SV_R2_AUTO_67, color="red", ls=":", lw=1, alpha=0.6,
               label=f"perceptual-only ceiling (R²={REF_HUE_SV_R2_AUTO_67})")
    ax.set_ylabel("CV macro R² on Z_top16  (cogito L40 centroids)")
    title = (f"auto_exp_17 · unified composition-engine vs hand-rolled "
             f"(gamfit=={GAMFIT_VERSION}, HAS_NEW_API={HAS_NEW_API})\n"
             f"N={N} filtered colors, K_PC={K_PC}, n_folds={N_FOLDS}")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(0.7, max(bar_vals) + 0.1))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
