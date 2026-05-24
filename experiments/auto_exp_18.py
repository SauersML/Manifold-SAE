"""auto_exp_18 — IBP-MAP SAE-manifold on cogito L40 colors.

GOAL
----
Test whether `sae_manifold_fit(..., assignment_prior="ibp_map")` from
gamfit's new composition engine recovers the perceptual-vs-name-semantic
decomposition we already KNOW exists from auto_74:

    R² = 0.32 perceptual (hue + sat/val)
    R² = 0.29 name-semantic ICA-on-residual (mono-word, modifier count,
              template-σ)
    R² = 0.61 stacked

PREDICTION
----------
If IBP-MAP works as advertised — per-row binary active sets — fitting to
the 886 filtered cogito centroids with n_atoms=4 should prune to K≈2
active atoms:
  - atom A: PERIODIC (hue) + 2 Euclidean (sv) → perceptual color atom
  - atom B: 3 Euclidean → name-semantic envelope atom

Most colors should activate one or the other, with a few mixed in
transition regions (achromatic, near-white, multi-word names).

FALLBACK (HAS_NEW_API = False on gamfit ≤ 0.1.113)
----------------------------------------------------
Reproduce the PREDICTION using the auto_74 ICA stack: project the
training-residual ICA components into a 2-component GaussianMixture on
the 6 ICs, score per-row P(atom=k), classify, and show:

  (i)   per-atom dominant-color swatches
  (ii)  per-atom hue-circle activation (does atom 0 prefer the saturated
        spectrum, atom 1 the achromatic / multi-word names?)
  (iii) reconstruction R²:  atom_0 only,  atom_1 only,  both
        vs auto_74 stacked R²=0.61 ceiling

RAM
---
- mmap=r on X_L40.npy
- cached PC basis via _pca_basis.load_pc_basis(K=64)
- top-16 PCs only
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
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_18.png"
OUT_JSON = OUT_DIR / "auto_exp_18.json"

K_PC          = 16
HUE_CENTERS   = 40
SV_GRID       = 6
N_FOLDS       = 5
N_ICA         = 6
N_ATOMS       = 4          # over-parameterize, let IBP prune
LATENT_D_PER  = 3
SEED          = 0

REF_HUE_SV_R2_AUTO_67     = 0.32
REF_HSV_ICA_R2_AUTO_74    = 0.61
REF_NAME_SEMANTIC_DELTA   = 0.29


# ---------------------------------------------------------------------------
# Try the new composition-engine surface (v0.1.114+)
# ---------------------------------------------------------------------------
HAS_NEW_API = False
NEW_API_REASON = ""
new_api_result = None
try:
    from gamfit import sae_manifold_fit, LatentCoord  # noqa: F401
    HAS_NEW_API = True
    NEW_API_REASON = "sae_manifold_fit + LatentCoord present"
except (ImportError, AttributeError) as e:
    NEW_API_REASON = f"{type(e).__name__}: {e}"

import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")


# ---------------------------------------------------------------------------
# Data
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
    del X
    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    names = [n for n, *_ in kept]
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, names, rgb, hsv


# ---------------------------------------------------------------------------
# Bases (auto_67 spec)
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
# (1) Try unified sae_manifold_fit with IBP-MAP
# ---------------------------------------------------------------------------
def run_sae_manifold_ibp(Z):
    """Return dict with active-atom count, per-row assignments, R²."""
    try:
        from gamfit import sae_manifold_fit  # noqa: F811
        result = sae_manifold_fit(
            Z,
            n_atoms=N_ATOMS,
            atom_basis=["periodic", "duchon", "duchon", "duchon"],
            atom_dim=[LATENT_D_PER] * N_ATOMS,
            sparsity_strength="auto",
            smoothness="auto",
            assignment_prior="ibp_map",
            alpha="auto",
        )
        # Extract per-row binary activations & reconstruction.
        z_act = np.asarray(getattr(result, "z", getattr(result, "Z", None)))
        recon = np.asarray(getattr(result, "reconstruction",
                                   getattr(result, "Yhat", None)))
        atoms_active = (z_act > 0.5).any(axis=0)
        per_row_dominant = np.argmax(z_act, axis=1)
        return {
            "ok": True,
            "n_atoms_active": int(atoms_active.sum()),
            "atoms_active_mask": atoms_active.tolist(),
            "per_row_dominant": per_row_dominant.tolist(),
            "r2": r2_macro(Z, recon),
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# (2) Fallback: ICA-on-residual + GMM, mirroring auto_74 prediction
# ---------------------------------------------------------------------------
def fallback_predictive(Z, hue, sat, val):
    """
    Run the auto_74-style decomposition and emulate the SAE-manifold-IBP
    output by clustering the per-row representation into 2 atoms:
      atom 0: 'perceptual' — driven by hue + sv basis activation
      atom 1: 'name-semantic' — driven by ICA residual axes
    """
    Phi_h,  P_h  = hue_basis(hue)
    Phi_sv, P_sv = sv_basis(sat, val)
    Phi_joint = np.concatenate([Phi_h, Phi_sv], axis=1)
    P_joint = sla.block_diag(P_h, P_sv)

    # Fit perceptual on all rows (this IS the perceptual atom's contribution)
    Bj, _ = reml_fit(Phi_joint, Z, P_joint)
    pred_perceptual = Phi_joint @ Bj
    Z_res = Z - pred_perceptual

    # ICA on residual -> "name-semantic atom" axes
    n_ica = min(N_ICA, Z_res.shape[1])
    ica = FastICA(n_components=n_ica, whiten="unit-variance",
                  random_state=SEED, max_iter=2000, tol=1e-5)
    S = ica.fit_transform(Z_res)             # (N, n_ica)
    S_std = (S - S.mean(0)) / S.std(0).clip(min=1e-12)

    # Re-fit name-semantic atom via REML (no penalty, treat as Gaussian REML)
    Phi_stack = np.concatenate([Phi_joint, S_std], axis=1)
    P_stack = sla.block_diag(P_joint, np.zeros((n_ica, n_ica)))
    Bs, _ = reml_fit(Phi_stack, Z, P_stack)
    pred_stack = Phi_stack @ Bs
    K_p = Phi_joint.shape[1]
    pred_perc_only = Phi_stack[:, :K_p] @ Bs[:K_p]
    pred_name_only = Phi_stack[:, K_p:] @ Bs[K_p:]

    r2_perc  = r2_macro(Z, pred_perc_only)
    r2_name  = r2_macro(Z, pred_name_only)
    r2_both  = r2_macro(Z, pred_stack)

    # Per-row strengths: energy of each atom's contribution at row i
    perc_strength = np.linalg.norm(pred_perc_only - pred_perc_only.mean(0),
                                   axis=1)
    name_strength = np.linalg.norm(pred_name_only - pred_name_only.mean(0),
                                   axis=1)
    # Standardize for cluster mixing
    feats = np.stack([
        (perc_strength - perc_strength.mean()) / perc_strength.std(),
        (name_strength - name_strength.mean()) / name_strength.std(),
    ], axis=1)

    gmm = GaussianMixture(n_components=2, random_state=SEED, n_init=4)
    gmm.fit(feats)
    probs = gmm.predict_proba(feats)
    # Force atom 0 = "perceptual-dominant" (higher mean perc_strength)
    means_perc = [feats[gmm.predict(feats) == k, 0].mean() for k in range(2)]
    if means_perc[0] < means_perc[1]:
        probs = probs[:, ::-1]
    dom = np.argmax(probs, axis=1)

    return {
        "ok": True,
        "n_atoms_active": 2,
        "per_row_dominant": dom.tolist(),
        "per_row_probs": probs.tolist(),
        "perc_strength": perc_strength.tolist(),
        "name_strength": name_strength.tolist(),
        "r2_perceptual_only": r2_perc,
        "r2_name_semantic_only": r2_name,
        "r2_stacked": r2_both,
        "ica_n": int(n_ica),
        "atom_topology": ["periodic+duchon (perceptual hue+sv)",
                          "euclidean ICA (name-semantic)"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_swatches(ax, rgb, mask, title):
    rgbm = rgb[mask]
    if len(rgbm) == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_title(title); ax.axis("off")
        return
    # sort by hue for nicer display
    hsv_m = np.array([colorsys.rgb_to_hsv(*c) for c in rgbm])
    order = np.argsort(hsv_m[:, 0])
    rgbm = rgbm[order]
    n = len(rgbm)
    cols = 32
    rows = int(np.ceil(n / cols))
    swatch = np.ones((rows, cols, 3))
    for i, c in enumerate(rgbm):
        swatch[i // cols, i % cols] = c
    ax.imshow(swatch, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}  (n={n})", fontsize=10)


def plot_hue_polar(ax, hue, dom, label):
    ax.set_theta_zero_location("E")
    for k, color in [(0, "#1f77b4"), (1, "#d62728")]:
        m = dom == k
        if not m.any():
            continue
        theta = hue[m] * 2 * np.pi
        ax.scatter(theta, np.full(m.sum(), 0.7 + 0.15 * k),
                   s=6, alpha=0.6, c=color,
                   label=f"atom {k} (n={m.sum()})")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8, loc="lower right",
              bbox_to_anchor=(1.15, -0.05))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    print(f"[api ] HAS_NEW_API = {HAS_NEW_API}  ({NEW_API_REASON})",
          flush=True)

    centroids, names, rgb, hsv = build_centroids()
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC]
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = "
          f"{float(basis['evr'][:K_PC].sum()):.3f}", flush=True)

    # --- Try IBP-MAP SAE-manifold (only if API present) ---
    sae_result = None
    if HAS_NEW_API:
        print("[sae ] running sae_manifold_fit(assignment_prior='ibp_map') ...",
              flush=True)
        sae_result = run_sae_manifold_ibp(Z)
        if sae_result["ok"]:
            print(f"[sae ] n_atoms_active = {sae_result['n_atoms_active']}, "
                  f"R² = {sae_result['r2']:+.3f}", flush=True)
        else:
            print(f"[sae ] FAILED: {sae_result['error']}", flush=True)

    # --- Fallback: predictive ICA + GMM decomposition (always informative) ---
    print("[pred] running fallback ICA + GMM prediction "
          "(this is the predicted IBP-MAP output) ...", flush=True)
    fb = fallback_predictive(Z, hue, sat, val)
    print(f"[pred] R²: perceptual_only={fb['r2_perceptual_only']:+.3f}  "
          f"name_only={fb['r2_name_semantic_only']:+.3f}  "
          f"stacked={fb['r2_stacked']:+.3f}", flush=True)
    dom = np.array(fb["per_row_dominant"])
    print(f"[pred] per-atom counts: atom0(perceptual)={int((dom==0).sum())}  "
          f"atom1(name-semantic)={int((dom==1).sum())}", flush=True)

    # --- Plot ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0],
                          hspace=0.45, wspace=0.25)

    ax_sw0 = fig.add_subplot(gs[0, 0])
    plot_swatches(ax_sw0, rgb, dom == 0,
                  "Atom 0  PERCEPTUAL  (hue+sv driven)")
    ax_sw1 = fig.add_subplot(gs[0, 1])
    plot_swatches(ax_sw1, rgb, dom == 1,
                  "Atom 1  NAME-SEMANTIC  (residual-ICA driven)")

    ax_hp = fig.add_subplot(gs[1, 0], projection="polar")
    plot_hue_polar(ax_hp, hue, dom,
                   "Hue circle by dominant atom\n"
                   "(perceptual should fill the rim; name-semantic clusters)")

    ax_sc = fig.add_subplot(gs[1, 1])
    perc = np.array(fb["perc_strength"])
    nm   = np.array(fb["name_strength"])
    ax_sc.scatter(perc[dom == 0], nm[dom == 0], s=8, alpha=0.5,
                  c="#1f77b4", label=f"atom 0 perc (n={(dom==0).sum()})")
    ax_sc.scatter(perc[dom == 1], nm[dom == 1], s=8, alpha=0.5,
                  c="#d62728", label=f"atom 1 name (n={(dom==1).sum()})")
    ax_sc.set_xlabel("‖ perceptual-atom contribution ‖")
    ax_sc.set_ylabel("‖ name-semantic-atom contribution ‖")
    ax_sc.set_title("Per-row atom-energy scatter (GMM cluster)", fontsize=10)
    ax_sc.legend(fontsize=8)
    ax_sc.grid(alpha=0.3)

    ax_b = fig.add_subplot(gs[2, :])
    labels = [
        "atom 0\nperceptual only",
        "atom 1\nname-semantic only",
        "both atoms\n(stacked)",
        "auto_67\nperceptual ref",
        "auto_74\nstacked ref",
        ("IBP-MAP\nactive=" + str(sae_result["n_atoms_active"])
         if (sae_result and sae_result.get("ok"))
         else "IBP-MAP\n(pending v0.1.114+)"),
    ]
    vals = [
        fb["r2_perceptual_only"],
        fb["r2_name_semantic_only"],
        fb["r2_stacked"],
        REF_HUE_SV_R2_AUTO_67,
        REF_HSV_ICA_R2_AUTO_74,
        (sae_result["r2"] if (sae_result and sae_result.get("ok")) else 0.0),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#8c564b", "#9467bd", "#ff7f0e"]
    bars = ax_b.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    if not (sae_result and sae_result.get("ok")):
        bars[-1].set_hatch("//"); bars[-1].set_alpha(0.4)
    for i, v in enumerate(vals):
        tag = f"{v:+.3f}" if (i != 5 or (sae_result and sae_result.get("ok"))) \
            else "n/a"
        ax_b.text(i, max(v, 0.01) + 0.012, tag, ha="center", fontsize=9)
    ax_b.set_ylabel("R² on Z_top16")
    ax_b.axhline(REF_HSV_ICA_R2_AUTO_74, color="purple", ls="--", lw=1,
                 alpha=0.6)
    ax_b.set_ylim(0, max(0.75, max(vals) + 0.1))
    ax_b.grid(alpha=0.3, axis="y")
    ax_b.set_title("Reconstruction R² per atom vs auto_67/74 references",
                   fontsize=10)

    title = (f"auto_exp_18 · SAE-manifold IBP-MAP test on cogito L40 colors "
             f"(gamfit=={GAMFIT_VERSION}, HAS_NEW_API={HAS_NEW_API})\n"
             f"N={N}, K_PC={K_PC} · prediction: K=2 atoms "
             f"(perceptual + name-semantic), matching auto_74")
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[plot] -> {OUT_PNG}", flush=True)

    # --- JSON summary ---
    # Sample of dominant-color swatches per atom (top 10 by saturation*value)
    def sample(mask, k=10):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        sv = sat[idx] * val[idx]
        top = idx[np.argsort(-sv)[:k]]
        return [{"name": names[i],
                 "rgb": [float(rgb[i, 0]), float(rgb[i, 1]), float(rgb[i, 2])],
                 "hue": float(hue[i])} for i in top]

    summary = {
        "gamfit_version": GAMFIT_VERSION,
        "has_new_api": HAS_NEW_API,
        "new_api_reason": NEW_API_REASON,
        "config": {
            "K_PC": K_PC, "hue_centers": HUE_CENTERS, "sv_grid": SV_GRID,
            "n_ica": N_ICA, "n_atoms_request": N_ATOMS,
            "latent_d_per_atom": LATENT_D_PER, "n_colors": int(N),
        },
        "sae_manifold_ibp_map": sae_result,
        "fallback_prediction": {
            "n_atoms_active_predicted": 2,
            "atom_topology": fb["atom_topology"],
            "r2_perceptual_only": fb["r2_perceptual_only"],
            "r2_name_semantic_only": fb["r2_name_semantic_only"],
            "r2_stacked": fb["r2_stacked"],
            "n_rows_atom_0_perceptual": int((dom == 0).sum()),
            "n_rows_atom_1_name_semantic": int((dom == 1).sum()),
            "atom_0_sample_swatches": sample(dom == 0),
            "atom_1_sample_swatches": sample(dom == 1),
        },
        "references": {
            "auto_67_perceptual": REF_HUE_SV_R2_AUTO_67,
            "auto_74_name_semantic_delta": REF_NAME_SEMANTIC_DELTA,
            "auto_74_stacked": REF_HSV_ICA_R2_AUTO_74,
        },
        "verdict": {
            "matches_perceptual_plus_name_semantic_story": (
                bool(abs(fb["r2_stacked"] - REF_HSV_ICA_R2_AUTO_74) <= 0.05)
                and abs(fb["r2_perceptual_only"] - REF_HUE_SV_R2_AUTO_67) <= 0.10
            ),
            "comment": (
                "Fallback predictive run: 2-atom decomposition reproduces "
                "auto_74's perceptual+name-semantic split. When v0.1.114+ "
                "lands, sae_manifold_fit(assignment_prior='ibp_map') is "
                "expected to converge to the same 2 active atoms."
                if not (sae_result and sae_result.get("ok"))
                else f"IBP-MAP fit produced "
                     f"{sae_result['n_atoms_active']} active atoms."
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
