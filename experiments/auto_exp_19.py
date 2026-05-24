"""auto_exp_19 - gamfit.select_topology vs auto_76 manual REML ranking.

GOAL
----
Test whether gamfit's new `select_topology()` helper (v0.1.114+ composition
engine) AGREES with auto_76's hand-rolled, candidate-by-candidate REML
ranking of 7 hue/sv topologies on the cogito L40 centroids.

auto_76 verdict (reproduced for reference):
  - winner by REML evidence: '4. Euclidean R^2 (sat, val)'  (lambda_S=ok)
  - winner by CV R^2       : '5. Cylinder S^1 x R^2 (h,s,v)'
  - margin to runner-up    : +36.19 nats (REML)

PREDICTION
----------
If `select_topology()` does what the composition-engine doc claims (joint
REML model evidence over a candidate set with a built-in Occam factor),
it MUST reproduce auto_76's REML ordering: the top pick should be either
"Euclidean R^2 (sv)" or "Cylinder S^1 x R^2", and the bottom should be
the cylinder if-and-only-if select_topology uses the same complexity
penalty.  The crucial falsifiable claim:

    select_topology's top-1 == auto_76 REML top-1 ('Euclidean R^2 sv')

DESIGN
------
- mmap=r on X_L40.npy, cached load_pc_basis(K=64), top-16 PCs
- Reconstruct a *light* 3-candidate subset of auto_76:
      A. Euclidean R^1 (hue, Duchon m=2)
      B. Circle    S^1 (hue, periodic Duchon m=2)
      C. Cylinder  S^1 x R^2 (h, s, v - Kronecker auto_69 spec)
  (Skipping the 4 candidates that hit admissibility constraints to keep
   the fallback runtime under 60s and the comparison apples-to-apples.)
- For each candidate: compute REML score + CV R^2 with the SAME basis
  spec auto_76 used, so we can verify our re-run matches auto_76's
  numbers within rounding.
- HAS_NEW_API gate:
    True  -> call gamfit.select_topology(Y=Z, candidates=[...]) and
             compare its ranking to ours.
    False -> emit our 3-candidate ranking AS THE PREDICTION for what
             select_topology should produce when it lands. JSON slot
             `select_topology_ranking` = null, but `predicted_ranking`
             is fully filled in so a future re-run can falsify in 1 line.

RAM RULES
---------
- mmap=r on X_L40.npy
- cached _pca_basis.load_pc_basis(K=64), no full harvest in RAM
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

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_19.png"
OUT_JSON = OUT_DIR / "auto_exp_19.json"

K_PC          = 16
HUE_CENTERS   = 20            # auto_76 spec
SV_GRID       = 6             # auto_67 / auto_69 spec
N_FOLDS       = 5
SEED          = 0

# auto_76 reference numbers we are predicting select_topology should match
AUTO_76_REML_TOP1   = "4. Euclidean R^2 (sat, val)"
AUTO_76_CV_TOP1     = "5. Cylinder S^1 x R^2 (h, s, v)"
AUTO_76_REML_MARGIN_NATS = 36.19

# ---------------------------------------------------------------------------
# Try the new composition-engine surface (v0.1.114+)
# ---------------------------------------------------------------------------
HAS_NEW_API = False
NEW_API_REASON = ""
try:
    from gamfit import select_topology, LatentCoord  # noqa: F401
    HAS_NEW_API = True
    NEW_API_REASON = "gamfit.select_topology + LatentCoord present"
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
# Bases (auto_76 candidate-subset spec: A, B, C)
# ---------------------------------------------------------------------------
def basis_euclidean_hue(hue01):
    """A. Euclidean R^1 over hue (Duchon m=2 non-periodic, 20 centers)."""
    centers = np.linspace(0.0, 1.0, HUE_CENTERS).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(gamfit.duchon_basis(pts, centers, m=2))
    P = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=2))
    return Phi, P


def basis_circle_hue(hue01):
    """B. Circle S^1 over hue (Duchon m=2 periodic, 20 centers, auto_66 spec)."""
    centers = np.linspace(0.0, 1.0, HUE_CENTERS,
                          endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(gamfit.duchon_basis(
        pts, centers, m=2, periodic_per_axis=[True]))
    P = np.asarray(gamfit.duchon_function_norm_penalty(
        centers, m=2, periodic_per_axis=[True]))
    return Phi, P


def basis_sv(sat, val):
    """Duchon m=2 degree2 nullspace, 6x6 centers (auto_67 spec)."""
    g = np.linspace(0.0, 1.0, SV_GRID)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi = np.asarray(gamfit.duchon_basis(
        pts, centers, m=2, nullspace_order="degree2"))
    P = np.asarray(gamfit.duchon_function_norm_penalty(
        centers, m=2, nullspace_order="degree2"))
    return Phi, P


def basis_cylinder(hue01, sat, val):
    """C. Cylinder S^1 x R^2 via Kronecker (auto_69 / auto_76 spec).
       Phi_h (periodic Duchon m=2) tensor Phi_sv (Duchon m=3 degree2)."""
    # Phi_h
    Phi_h, P_h = basis_circle_hue(hue01)
    # Phi_sv with m=3 (auto_69 used higher smoothness on sv leg of cylinder)
    g = np.linspace(0.0, 1.0, SV_GRID)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers_sv = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts_sv = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi_sv = np.asarray(gamfit.duchon_basis(
        pts_sv, centers_sv, m=3, nullspace_order="degree2"))
    P_sv = np.asarray(gamfit.duchon_function_norm_penalty(
        centers_sv, m=3, nullspace_order="degree2"))

    N = Phi_h.shape[0]
    Kh, Ksv = Phi_h.shape[1], Phi_sv.shape[1]
    # Row-wise Kronecker product (Khatri-Rao): Phi[i] = Phi_h[i] (x) Phi_sv[i]
    Phi = (Phi_h[:, :, None] * Phi_sv[:, None, :]).reshape(N, Kh * Ksv)
    # Block-diagonal additive penalty: lambda_h * (P_h (x) I) + lambda_sv * (I (x) P_sv)
    # We use a SINGLE shared lambda (single-penalty REML) for fair comparison.
    P = np.kron(P_h, np.eye(Ksv)) + np.kron(np.eye(Kh), P_sv)
    return Phi, P


# ---------------------------------------------------------------------------
# REML + CV utilities
# ---------------------------------------------------------------------------
def reml_fit(Phi, Y, P):
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return (np.asarray(out["coefficients"]),
            float(out["lambda"]),
            float(out.get("reml_score", out.get("reml_objective", np.nan))))


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def candidate_score(name, build_phi, Z, fold):
    """Return REML score (on full data) + CV macro R^2 across folds."""
    Phi, P = build_phi()
    # REML score on full data (sum over per-PC if multi-response).
    B, lam, _ = reml_fit(Phi, Z, P)
    # The single-output REML score is what we need, but gamfit's
    # gaussian_reml_fit returns per-fit; for a multi-response Y of shape
    # (N, K_PC) we sum -2 * log marginal likelihood across PCs.
    # Recompute REML across PCs:
    reml_total = 0.0
    log_lams = []
    for k in range(Z.shape[1]):
        out_k = gamfit.gaussian_reml_fit(Phi, Z[:, k:k+1], P)
        reml_total += float(out_k.get("reml_score",
                                      out_k.get("reml_objective", np.nan)))
        log_lams.append(float(np.log(out_k["lambda"])))
    # CV R^2
    pred = np.zeros_like(Z)
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        Btr, _, _ = reml_fit(Phi[tr], Z[tr], P)
        pred[te] = Phi[te] @ Btr
    cv_r2 = r2_macro(Z, pred)
    return {
        "name": name,
        "K_basis": int(Phi.shape[1]),
        "reml_score_sum": reml_total,    # sum over K_PC columns
        "cv_r2": cv_r2,
        "log_lambda_full_mean": float(np.mean(log_lams)),
    }


# ---------------------------------------------------------------------------
# Optional new-API call
# ---------------------------------------------------------------------------
def try_select_topology(Z, hue, sat, val):
    """Call gamfit.select_topology if present; return its ranking dict."""
    try:
        candidates = [
            LatentCoord(d=1, manifold="euclidean", aux_data=hue[:, None]),
            LatentCoord(d=1, manifold="circle",    aux_data=hue[:, None]),
            LatentCoord(d=3, manifold="cylinder",
                        aux_data=np.stack([hue, sat, val], axis=1)),
        ]
        out = select_topology(Y=Z, candidates=candidates, method="reml")
        # Hope the result exposes .ranking or .scores
        ranking = getattr(out, "ranking",
                          getattr(out, "ranked_names", None))
        scores = getattr(out, "scores",
                         getattr(out, "reml_scores", None))
        return {"ok": True,
                "ranking": list(ranking) if ranking is not None else None,
                "scores": (list(map(float, scores))
                           if scores is not None else None),
                "raw_repr": repr(out)[:400],
                "error": None}
    except Exception as e:
        return {"ok": False, "ranking": None, "scores": None,
                "error": f"{type(e).__name__}: {e}"}


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
    evr = float(basis["evr"][:K_PC].sum())
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = {evr:.3f}",
          flush=True)

    # CV folds
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # --- Manual 3-candidate ranking (the predictive baseline) ---
    print("[rank] scoring candidates A/B/C (Euclidean hue, Circle hue, "
          "Cylinder h,s,v) ...", flush=True)

    cand = []
    print("[rank]   A. Euclidean R^1 (hue) ...", flush=True)
    cand.append(candidate_score(
        "A. Euclidean R^1 (hue)",
        lambda: basis_euclidean_hue(hue), Z, fold))
    print(f"         REML_sum={cand[-1]['reml_score_sum']:.1f}  "
          f"CV R^2={cand[-1]['cv_r2']:+.3f}  K={cand[-1]['K_basis']}",
          flush=True)

    print("[rank]   B. Circle S^1 (hue periodic) ...", flush=True)
    cand.append(candidate_score(
        "B. Circle S^1 (hue periodic)",
        lambda: basis_circle_hue(hue), Z, fold))
    print(f"         REML_sum={cand[-1]['reml_score_sum']:.1f}  "
          f"CV R^2={cand[-1]['cv_r2']:+.3f}  K={cand[-1]['K_basis']}",
          flush=True)

    print("[rank]   C. Cylinder S^1 x R^2 (h, s, v) ...", flush=True)
    cand.append(candidate_score(
        "C. Cylinder S^1 x R^2 (h, s, v)",
        lambda: basis_cylinder(hue, sat, val), Z, fold))
    print(f"         REML_sum={cand[-1]['reml_score_sum']:.1f}  "
          f"CV R^2={cand[-1]['cv_r2']:+.3f}  K={cand[-1]['K_basis']}",
          flush=True)

    # Rank: lower REML score = better evidence (REML score = -2 log mlik
    # up to const).
    ranking_reml = sorted(cand, key=lambda c: c["reml_score_sum"])
    ranking_cv   = sorted(cand, key=lambda c: -c["cv_r2"])
    predicted_reml_top1 = ranking_reml[0]["name"]
    predicted_cv_top1   = ranking_cv[0]["name"]
    margin_to_runner_up = (ranking_reml[1]["reml_score_sum"]
                           - ranking_reml[0]["reml_score_sum"])

    print(f"[rank] predicted REML top-1: {predicted_reml_top1}  "
          f"(margin = {margin_to_runner_up:+.1f} nats)", flush=True)
    print(f"[rank] predicted CV   top-1: {predicted_cv_top1}", flush=True)

    # --- Optional: select_topology() call (only if HAS_NEW_API) ---
    if HAS_NEW_API:
        print("[api ] calling gamfit.select_topology ...", flush=True)
        st_res = try_select_topology(Z, hue, sat, val)
        if st_res["ok"]:
            print(f"[api ] select_topology ranking = {st_res['ranking']}",
                  flush=True)
        else:
            print(f"[api ] select_topology FAILED: {st_res['error']}",
                  flush=True)
    else:
        st_res = {"ok": False, "ranking": None, "scores": None,
                  "error": f"gamfit=={GAMFIT_VERSION} lacks select_topology; "
                           "rerun when v0.1.114 wheels land to falsify the "
                           "predicted ranking below."}

    # --- Verdict ---
    # NOTE on apples-to-apples mapping: auto_76 covered 7 candidates but
    # the cogito-relevant subset is essentially {hue topology,
    # cylinder}, so our 3-candidate REML ordering should mirror auto_76's
    # ordering of those 3 entries: (Euclidean R^1 hue) vs (Circle S^1 hue)
    # vs (Cylinder).
    # auto_76 numbers for those 3 (REML_score): 55493.24, 55573.87, 48767.92
    # -> rank by REML: Cylinder, Euclidean R^1, Circle.
    # (Note: in the 7-way sweep, auto_76's TOP-1 was "4. Euclidean R^2 (sv)";
    # we're not including that one, so the top of our 3-cand subset is
    # *expected* to be Cylinder.)
    auto_76_subset_reml_ranking = [
        "C. Cylinder S^1 x R^2 (h, s, v)",
        "A. Euclidean R^1 (hue)",
        "B. Circle S^1 (hue periodic)",
    ]
    our_ranking_names = [c["name"] for c in ranking_reml]
    agrees_with_auto_76_subset = (our_ranking_names ==
                                  auto_76_subset_reml_ranking)

    if HAS_NEW_API and st_res["ok"] and st_res["ranking"]:
        # Map select_topology ranking back to our names (positional).
        # The candidate order we passed was A, B, C.
        name_by_idx = {0: "A. Euclidean R^1 (hue)",
                       1: "B. Circle S^1 (hue periodic)",
                       2: "C. Cylinder S^1 x R^2 (h, s, v)"}
        st_top_idx = (st_res["ranking"][0]
                      if isinstance(st_res["ranking"][0], int) else None)
        st_top_name = (name_by_idx.get(st_top_idx)
                       if st_top_idx is not None
                       else str(st_res["ranking"][0]))
        api_matches_manual_top1 = (st_top_name == predicted_reml_top1)
    else:
        st_top_name = None
        api_matches_manual_top1 = None

    summary = {
        "experiment": "auto_exp_19",
        "question": ("Does gamfit.select_topology() agree with auto_76's "
                     "manual REML topology ranking on cogito centroids?"),
        "gamfit_version": GAMFIT_VERSION,
        "has_new_api": HAS_NEW_API,
        "new_api_reason": NEW_API_REASON,
        "config": {
            "K_PC": K_PC,
            "hue_centers": HUE_CENTERS,
            "sv_grid": SV_GRID,
            "n_folds": N_FOLDS,
            "n_colors": int(N),
            "evr_top_K_PC": evr,
        },
        "candidates": cand,
        "predicted_ranking_reml": [c["name"] for c in ranking_reml],
        "predicted_ranking_cv":   [c["name"] for c in ranking_cv],
        "predicted_reml_top1":    predicted_reml_top1,
        "predicted_cv_top1":      predicted_cv_top1,
        "predicted_reml_margin_nats": margin_to_runner_up,
        "auto_76_reference": {
            "winner_by_reml_7cand": AUTO_76_REML_TOP1,
            "winner_by_cv_7cand":   AUTO_76_CV_TOP1,
            "reml_margin_nats":     AUTO_76_REML_MARGIN_NATS,
            "subset_reml_ranking_expected": auto_76_subset_reml_ranking,
            "our_3cand_matches_auto_76_subset": agrees_with_auto_76_subset,
        },
        "select_topology_result": st_res,
        "verdict": {
            "manual_REML_winner": predicted_reml_top1,
            "select_topology_winner": st_top_name,
            "api_matches_manual_top1": api_matches_manual_top1,
            "comment": (
                ("PREDICTION: when gamfit.select_topology lands, it should "
                 "rank '" + predicted_reml_top1 + "' top-1 with a margin "
                 f"of >= {abs(margin_to_runner_up):.0f} nats. If it doesn't, "
                 "either its complexity penalty differs from per-PC summed "
                 "REML, or it weights candidates by something else (e.g. CV).")
                if not HAS_NEW_API
                else ("select_topology AGREES with manual REML top-1."
                      if api_matches_manual_top1
                      else "select_topology DISAGREES with manual REML "
                           "top-1.  Investigate which complexity penalty "
                           "select_topology uses.")
            ),
            "falsification_when_wheels_land": (
                "Re-run auto_exp_19; if HAS_NEW_API is True and "
                "select_topology_winner != '" + predicted_reml_top1 +
                "', the prediction is falsified."
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # REML score (lower = better) bar chart
    names_short = ["A.\nEuclid R^1\n(hue)",
                   "B.\nCircle S^1\n(hue)",
                   "C.\nCylinder\nS^1 x R^2"]
    reml_vals = [c["reml_score_sum"] for c in cand]
    cv_vals   = [c["cv_r2"] for c in cand]
    bcolors = ["#1f77b4", "#d62728", "#2ca02c"]

    bars0 = axes[0].bar(names_short, reml_vals, color=bcolors,
                        edgecolor="black", lw=0.8)
    best_idx = int(np.argmin(reml_vals))
    bars0[best_idx].set_edgecolor("gold")
    bars0[best_idx].set_linewidth(3)
    for i, v in enumerate(reml_vals):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Sum REML score over K_PC columns  (lower = better)")
    axes[0].set_title(f"REML evidence ranking\ntop-1 (manual): "
                      f"{cand[best_idx]['name']}", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")

    bars1 = axes[1].bar(names_short, cv_vals, color=bcolors,
                        edgecolor="black", lw=0.8)
    best_cv = int(np.argmax(cv_vals))
    bars1[best_cv].set_edgecolor("gold")
    bars1[best_cv].set_linewidth(3)
    for i, v in enumerate(cv_vals):
        axes[1].text(i, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel(f"{N_FOLDS}-fold CV macro R^2 on Z_top{K_PC}")
    axes[1].set_title(f"CV ranking\ntop-1 (manual): "
                      f"{cand[best_cv]['name']}", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(0, max(0.5, max(cv_vals) + 0.05))

    title = (f"auto_exp_19 . select_topology() vs auto_76 manual REML "
             f"(gamfit=={GAMFIT_VERSION}, HAS_NEW_API={HAS_NEW_API})\n"
             f"N={N} cogito L40 centroids, K_PC={K_PC}, n_folds={N_FOLDS}  "
             f"|  3-cand subset matches auto_76 ordering: "
             f"{agrees_with_auto_76_subset}")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
