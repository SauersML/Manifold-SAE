"""auto_76.py — Bayesian topology selection over manifold shapes for
cogito's color centroids via gamfit's REML evidence.

The TDA approach in auto_70/auto_72 was ambiguous (no dominant H¹ class,
persistence ratio ~ 1). The right resolution is: don't eyeball
persistence diagrams — fit the model under each candidate topology and
let the marginal likelihood select. The log|H| - log|S|_+ Occam factors
in gamfit's gaussian_reml_fit are built precisely to compare models of
different complexity.

Candidate topologies (each fit as supervised Z ≈ Φ(coord) @ B):
 1. Euclidean 1D (hue, non-periodic)         — Duchon m=2
 2. Circle S¹ (hue, periodic)                — Duchon m=2 periodic[True]
 3. Sphere S² (lat=val, lon=hue)             — gamfit.sphere_basis
 4. Euclidean 2D (sat, val)                  — Duchon m=2 degree2 nullspace
 5. Cylinder S¹ × R² (hue periodic ⊗ s,v)    — Kronecker (auto_69 pattern)
 6. Torus T²  (hue × sat both periodic)      — Kronecker of two 1D periodic
 7. Torus T³  (h × s × v all periodic)       — Kronecker of three 1D periodic

For each: gaussian_reml_fit -> reml_score (LAML/REML log marginal),
plus 5-fold color-grouped CV macro R² for comparison.

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import colorsys
import json
import sys
import traceback
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
K_PC = 16


# ----------------------------- helpers -----------------------------
def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def r2_per_col(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


# ----------------------------- bases ------------------------------
def basis_hue_line(hue, n_centers=20, train_idx=None):
    """(1) Euclidean 1D non-periodic Duchon m=2 on hue."""
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers).reshape(-1, 1)
    pts = np.asarray(hue, float).reshape(-1, 1)
    Phi = np.asarray(gamfit.duchon_basis(pts, centers, m=2))
    P = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=2))
    return Phi, P


def basis_hue_circle(hue, n_centers=20):
    """(2) Circle S¹: 1D periodic Duchon m=2 on hue."""
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue, float).reshape(-1, 1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    )
    return Phi, P


def basis_sphere(hue, sat, val, n_centers=64):
    """(3) Sphere S²: parameterize hue as longitude, val as latitude.
    sat is dropped (this is the 'no sat direction' interpretation in spec).
    lon = hue * 360 deg in [0, 360),  lat = (val - 0.5) * 180 deg in [-90, 90].
    """
    import gamfit
    lon = (np.asarray(hue, float) % 1.0) * 360.0
    lat = (np.asarray(val, float) - 0.5) * 180.0
    pts = np.stack([lat, lon], axis=1)  # (N, 2) lat, lon, in degrees
    Phi, P = gamfit.sphere_basis(pts, n_centers, kernel="sobolev")
    return np.asarray(Phi), np.asarray(P)


def basis_sv_plane(sat, val, grid=6):
    """(4) Euclidean 2D plane (sat, val): Duchon m=2 with degree-2 nullspace."""
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
    return Phi, P


def _kron_rowwise(A, B):
    """Row-wise Kronecker (Khatri-Rao on rows): out[i, a*Kb + b] = A[i,a]*B[i,b]."""
    Na, Ka = A.shape
    Nb, Kb = B.shape
    assert Na == Nb
    return (A[:, :, None] * B[:, None, :]).reshape(Na, Ka * Kb)


def basis_cylinder(hue, sat, val, n_h=8, n_grid=4):
    """(5) Cylinder S¹ × R²: tensor product of (hue periodic Duchon 1D, m=2)
    ⊗ (sat,val Duchon 2D, m=3 thin-plate). Mirrors auto_69's Kronecker
    fallback for mixed-periodicity.
    """
    import gamfit
    h_ctr = np.linspace(0.0, 1.0, n_h, endpoint=False).reshape(-1, 1)
    g = np.linspace(0.0, 1.0, n_grid)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    sv_ctr = np.stack([cx.ravel(), cy.ravel()], axis=1)

    Phi_h = np.asarray(
        gamfit.duchon_basis(np.asarray(hue, float).reshape(-1, 1), h_ctr,
                            m=2, periodic_per_axis=[True])
    )
    P_h = np.asarray(
        gamfit.duchon_function_norm_penalty(h_ctr, m=2, periodic_per_axis=[True])
    )
    sv_pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi_sv = np.asarray(gamfit.duchon_basis(sv_pts, sv_ctr, m=3))
    P_sv = np.asarray(gamfit.duchon_function_norm_penalty(sv_ctr, m=3))

    Phi = _kron_rowwise(Phi_h, Phi_sv)
    Kh, Ksv = Phi_h.shape[1], Phi_sv.shape[1]
    I_h, I_sv = np.eye(Kh), np.eye(Ksv)
    P = np.kron(P_h, I_sv) + np.kron(I_h, P_sv)
    P = 0.5 * (P + P.T)
    return Phi, P


def basis_torus2(hue, sat, n_h=8, n_s=8):
    """(6) Torus T² (h × s both periodic): Kronecker of two 1D periodic
    Duchon factors. (We avoid B-splines, so PeriodicSpline is not used.)
    """
    import gamfit
    h_ctr = np.linspace(0.0, 1.0, n_h, endpoint=False).reshape(-1, 1)
    s_ctr = np.linspace(0.0, 1.0, n_s, endpoint=False).reshape(-1, 1)
    Phi_h = np.asarray(
        gamfit.duchon_basis(np.asarray(hue, float).reshape(-1, 1), h_ctr,
                            m=2, periodic_per_axis=[True])
    )
    P_h = np.asarray(
        gamfit.duchon_function_norm_penalty(h_ctr, m=2, periodic_per_axis=[True])
    )
    Phi_s = np.asarray(
        gamfit.duchon_basis(np.asarray(sat, float).reshape(-1, 1), s_ctr,
                            m=2, periodic_per_axis=[True])
    )
    P_s = np.asarray(
        gamfit.duchon_function_norm_penalty(s_ctr, m=2, periodic_per_axis=[True])
    )
    Phi = _kron_rowwise(Phi_h, Phi_s)
    Kh, Ks = Phi_h.shape[1], Phi_s.shape[1]
    P = np.kron(P_h, np.eye(Ks)) + np.kron(np.eye(Kh), P_s)
    P = 0.5 * (P + P.T)
    return Phi, P


def basis_torus3(hue, sat, val, n_h=6, n_s=6, n_v=6):
    """(7) Torus T³ (all three periodic): triple Kronecker of 1D periodic
    Duchon factors.
    """
    import gamfit
    h_ctr = np.linspace(0.0, 1.0, n_h, endpoint=False).reshape(-1, 1)
    s_ctr = np.linspace(0.0, 1.0, n_s, endpoint=False).reshape(-1, 1)
    v_ctr = np.linspace(0.0, 1.0, n_v, endpoint=False).reshape(-1, 1)

    def _periodic_1d(x, ctr):
        Phi = np.asarray(
            gamfit.duchon_basis(np.asarray(x, float).reshape(-1, 1), ctr,
                                m=2, periodic_per_axis=[True])
        )
        P = np.asarray(
            gamfit.duchon_function_norm_penalty(ctr, m=2, periodic_per_axis=[True])
        )
        return Phi, P

    Phi_h, P_h = _periodic_1d(hue, h_ctr)
    Phi_s, P_s = _periodic_1d(sat, s_ctr)
    Phi_v, P_v = _periodic_1d(val, v_ctr)
    Kh, Ks, Kv = Phi_h.shape[1], Phi_s.shape[1], Phi_v.shape[1]

    # Row-wise triple Kron: ((Phi_h ⊠ Phi_s) ⊠ Phi_v)
    Phi_hs = _kron_rowwise(Phi_h, Phi_s)
    Phi = _kron_rowwise(Phi_hs, Phi_v)

    I_h, I_s, I_v = np.eye(Kh), np.eye(Ks), np.eye(Kv)
    P = (np.kron(np.kron(P_h, I_s), I_v)
         + np.kron(np.kron(I_h, P_s), I_v)
         + np.kron(np.kron(I_h, I_s), P_v))
    P = 0.5 * (P + P.T)
    return Phi, P


# ------------------------- fit / evaluate -------------------------
def reml_fit(Phi, Y, P):
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return (
        np.asarray(out["coefficients"]),
        float(out["lambda"]),
        float(out["reml_score"]),
    )


def evaluate_topology(name, build_fn, build_args, Z, fold, notes=""):
    """Build the basis on all rows, then do REML on full data (for evidence)
    and 5-fold CV for R²."""
    try:
        Phi_full, P_full = build_fn(**build_args)
        K = Phi_full.shape[1]
        # Full-data REML score (evidence)
        B_full, lam_full, reml_full = reml_fit(Phi_full, Z, P_full)

        # CV macro R²
        preds = np.zeros_like(Z)
        ll_folds = []
        for f in range(N_FOLDS):
            tr = fold != f
            te = ~tr
            B, lam, _ = reml_fit(Phi_full[tr], Z[tr], P_full)
            preds[te] = Phi_full[te] @ B
            ll_folds.append(np.log(lam) if lam > 0 else float("-inf"))
        cv_r2 = float(r2_macro(Z, preds))
        per_pc = r2_per_col(Z, preds)
        return {
            "name": name,
            "K_basis": int(K),
            "reml_score": float(reml_full),
            "cv_r2": cv_r2,
            "log_lambda": float(np.log(lam_full)) if lam_full > 0 else float("-inf"),
            "log_lambda_cv_mean": float(np.mean(ll_folds)),
            "per_pc_cv_r2": [float(x) for x in per_pc],
            "status": "ok",
            "notes": notes,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FAIL] {name}: {e}")
        print(tb)
        return {
            "name": name,
            "K_basis": -1,
            "reml_score": float("nan"),
            "cv_r2": float("nan"),
            "log_lambda": float("nan"),
            "log_lambda_cv_mean": float("nan"),
            "per_pc_cv_r2": [],
            "status": f"failed: {e}",
            "notes": notes,
        }


# ----------------------------- main -------------------------------
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
    evr = float(basis["evr"].sum())
    print(f"[load] Z shape={Z.shape}  EVR_top{K_PC}={evr:.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # ----- Run all 7 candidate topologies -----
    candidates = [
        ("1. Euclidean R¹ (hue)",
         basis_hue_line, dict(hue=hue, n_centers=20),
         "Duchon m=2 non-periodic, 20 centers on hue"),
        ("2. Circle S¹ (hue periodic)",
         basis_hue_circle, dict(hue=hue, n_centers=20),
         "Duchon m=2 periodic[True], 20 centers (auto_66 spec)"),
        ("3. Sphere S² (val=lat, hue=lon)",
         basis_sphere, dict(hue=hue, sat=sat, val=val, n_centers=64),
         "gamfit.sphere_basis(kernel='sobolev'), 64 Wahba centers; sat dropped"),
        ("4. Euclidean R² (sat, val)",
         basis_sv_plane, dict(sat=sat, val=val, grid=6),
         "Duchon m=2 degree2 nullspace, 6×6=36 centers (auto_67 spec)"),
        ("5. Cylinder S¹×R² (h, s, v)",
         basis_cylinder, dict(hue=hue, sat=sat, val=val, n_h=8, n_grid=4),
         "Kronecker fallback (auto_69): Φ_h(periodic m=2) ⊗ Φ_sv(m=3); "
         "mixed-periodic Duchon hits admissibility p=1,s=0 constraint"),
        ("6. Torus T² (h, s periodic)",
         basis_torus2, dict(hue=hue, sat=sat, n_h=8, n_s=8),
         "Kronecker of two 1D periodic Duchon m=2; "
         "2D periodic Duchon hits admissibility constraint"),
        ("7. Torus T³ (h, s, v periodic)",
         basis_torus3, dict(hue=hue, sat=sat, val=val, n_h=6, n_s=6, n_v=6),
         "Triple Kronecker of 1D periodic Duchon m=2; "
         "3D periodic Duchon hits admissibility constraint"),
    ]

    results = []
    for name, fn, args, notes in candidates:
        print(f"\n[topology] {name}")
        r = evaluate_topology(name, fn, args, Z, fold, notes=notes)
        print(f"   K_basis={r['K_basis']}  reml_score={r['reml_score']:+.2f}  "
              f"cv_r2={r['cv_r2']:+.4f}  log_lambda={r['log_lambda']:+.2f}")
        results.append(r)

    # ----- Ranking -----
    valid = [r for r in results if r["status"] == "ok"]
    by_reml = sorted(valid, key=lambda r: -r["reml_score"])
    by_cv = sorted(valid, key=lambda r: -r["cv_r2"])
    winner_reml = by_reml[0]["name"] if by_reml else None
    winner_cv = by_cv[0]["name"] if by_cv else None

    # Bayes factor of winner over runner-up (in log-nats)
    if len(by_reml) >= 2:
        margin_nats = by_reml[0]["reml_score"] - by_reml[1]["reml_score"]
    else:
        margin_nats = float("nan")

    print("\n========== RANKING by REML evidence (higher = better) ==========")
    for r in by_reml:
        print(f"  {r['reml_score']:+10.2f}   {r['name']:40s}  K={r['K_basis']:5d}  cv_r2={r['cv_r2']:+.3f}")
    print("\n========== RANKING by CV R² ==========")
    for r in by_cv:
        print(f"  {r['cv_r2']:+8.4f}   {r['name']:40s}  reml={r['reml_score']:+.2f}")

    # ----- Plot -----
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    # (1) Bar: reml_score
    ax = fig.add_subplot(gs[0, 0])
    names = [r["name"].split(".")[1].strip().split(" (")[0]
             if "." in r["name"] else r["name"] for r in by_reml]
    scores = [r["reml_score"] for r in by_reml]
    bcols = plt.cm.viridis(np.linspace(0.15, 0.85, len(by_reml)))
    bars = ax.barh(range(len(by_reml)), scores, color=bcols, edgecolor="black")
    ax.set_yticks(range(len(by_reml)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("REML score (log marginal likelihood, nats)")
    ax.set_title(f"(a) Bayesian evidence per topology\n"
                 f"(higher = better; includes Occam factor)")
    ax.axvline(by_reml[0]["reml_score"], color="red", ls="--", lw=0.8, alpha=0.5,
               label=f"winner")
    for i, s in enumerate(scores):
        ax.text(s, i, f"  {s:+.1f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)

    # (2) Bar: CV R²
    ax = fig.add_subplot(gs[0, 1])
    names_cv = [r["name"].split(".")[1].strip().split(" (")[0]
                if "." in r["name"] else r["name"] for r in by_cv]
    cvs = [r["cv_r2"] for r in by_cv]
    bcols = plt.cm.plasma(np.linspace(0.15, 0.85, len(by_cv)))
    ax.barh(range(len(by_cv)), cvs, color=bcols, edgecolor="black")
    ax.set_yticks(range(len(by_cv)))
    ax.set_yticklabels(names_cv, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV macro R²")
    ax.set_title("(b) CV R² per topology\n(favors complexity)")
    ax.axvline(0, color="black", lw=0.5)
    for i, s in enumerate(cvs):
        ax.text(s, i, f"  {s:+.3f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # (3) Per-PC R² heatmap for top-3 evidence winners
    ax = fig.add_subplot(gs[0, 2])
    top3 = by_reml[:3]
    M = np.array([r["per_pc_cv_r2"] for r in top3 if r["per_pc_cv_r2"]])
    if M.size > 0:
        n_show = min(K_PC, M.shape[1])
        im = ax.imshow(M[:, :n_show], aspect="auto", cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5, interpolation="nearest")
        ax.set_yticks(range(len(top3)))
        ax.set_yticklabels([r["name"][:24] for r in top3], fontsize=8)
        ax.set_xlabel(f"PC index (0..{n_show-1})")
        ax.set_title("(c) Per-PC CV R² for top-3 evidence winners")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # (4) Text panel: ranking + verdict
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")

    def _fmt_row(r):
        return (f"  {r['name']:40s}  K={r['K_basis']:5d}  "
                f"reml={r['reml_score']:+9.2f}  cv_r2={r['cv_r2']:+.4f}  "
                f"log_λ={r['log_lambda']:+6.2f}  status={r['status']}")

    txt_lines = [
        f"N = {N} filtered xkcd colors  ·  Z = top-{K_PC} cogito-L40 PCs (EVR cum = {evr:.3f})",
        "",
        "Candidates (in input order):",
    ]
    for r in results:
        txt_lines.append(_fmt_row(r))
    txt_lines += [
        "",
        f"Ranking by REML evidence (highest = best supported topology):",
    ]
    for i, r in enumerate(by_reml):
        txt_lines.append(f"   #{i+1}  reml={r['reml_score']:+9.2f}   {r['name']}")
    txt_lines += [
        "",
        f"Ranking by 5-fold CV R²:",
    ]
    for i, r in enumerate(by_cv):
        txt_lines.append(f"   #{i+1}  cv_r2={r['cv_r2']:+.4f}   {r['name']}")
    agree = (winner_reml == winner_cv)
    if margin_nats == margin_nats:  # not NaN
        bf_str = f"Bayes factor ≈ exp({margin_nats:.2f}) = {np.exp(min(margin_nats, 700)):.2e}"
    else:
        bf_str = "Bayes factor: n/a"
    txt_lines += [
        "",
        "============================== ENGINE'S VERDICT ==============================",
        f"  Winner by REML evidence : {winner_reml}",
        f"  Winner by CV R²         : {winner_cv}",
        f"  Agreement               : {'YES' if agree else 'NO — REML and CV disagree'}",
        f"  Evidence margin over runner-up : Δ reml_score = {margin_nats:+.2f} nats   ({bf_str})",
    ]
    ax.text(0.0, 1.0, "\n".join(txt_lines), family="monospace", fontsize=8,
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(
        f"auto_76 · Bayesian topology selection (REML evidence) for cogito L40 color manifold\n"
        f"7 candidate topologies fit via gamfit.gaussian_reml_fit  ·  K_PC={K_PC}",
        fontsize=12,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "auto_76.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    # ----- Verdict text -----
    if agree:
        verdict = (
            f"REML evidence and CV R² agree: the best-supported topology for cogito's "
            f"color manifold (top-{K_PC} PCs of L40 centroids) is '{winner_reml}'. "
            f"The evidence margin over the runner-up is {margin_nats:+.2f} nats."
        )
    else:
        verdict = (
            f"REML evidence picks '{winner_reml}' as the best-supported topology, "
            f"while CV R² favours '{winner_cv}'. The disagreement is the Occam "
            f"factor at work: REML penalises model complexity (via log|H|-log|S|_+) "
            f"that CV does not see directly. Δ_reml over runner-up = {margin_nats:+.2f} nats."
        )

    payload = {
        "n_colors": int(N),
        "K_PC": K_PC,
        "evr_top_K_PC": evr,
        "n_folds": N_FOLDS,
        "candidates": results,
        "ranking_by_reml": [r["name"] for r in by_reml],
        "ranking_by_cv": [r["name"] for r in by_cv],
        "winner_by_evidence": winner_reml,
        "winner_by_cv": winner_cv,
        "evidence_margin_nats": float(margin_nats),
        "verdict": verdict,
        "hard_rules": (
            "Pure Duchon (length_scale=None), gamfit.sphere_basis for S², "
            "no Gaussian RBF, no B-splines. Mixed-periodic Duchon in d≥2 is "
            "pinned to p=1,s=0 (admissibility 2(p+s)>d fails) so cylinder/T²/T³ "
            "use Kronecker tensor product of 1D/2D Duchon factors (auto_69 pattern). "
            "2D Duchon m=2 uses nullspace_order='degree2' (auto_67 pattern). "
            "X_L40.npy loaded via mmap. PC basis via _pca_basis.load_pc_basis+project."
        ),
    }
    (OUT_DIR / "auto_76.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_76.json'}")
    print(f"\n[verdict] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
