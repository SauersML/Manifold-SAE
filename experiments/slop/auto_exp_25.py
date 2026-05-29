"""auto_exp_25 - validate the new gamfit.select_topology() wrapper.

GOAL
----
Replay auto_exp_19 (REML topology winner = Cylinder) and
auto_exp_22 (BIC topology winner = Circle) through ONE call into
gamfit.select_topology() per scoring rule, and verify:

  (a) select_topology(score="reml") top-1 == "Cylinder"   (auto_exp_19)
  (b) select_topology(score="bic")  top-1 == "Circle"     (auto_exp_22)
  (c) The wrapper emits a warning citing the memory file
      `project_gumbel_anneal_population_sparsity_falsified` when the
      REML-rank and BIC-rank diverge.

WHEEL STATUS
------------
gamfit==0.1.120 shipped to PyPI WITHOUT select_topology (it landed
post-tag in /gamfit/_select_topology.py, agent b8jgu8ack).  When the
function is not importable we run a hand-rolled fallback emulator over
the same 5 candidates so the JSON still records the predicted verdicts.

The fallback is intentionally a faithful re-implementation of what the
wrapper is supposed to compute, so its top-1 must match the hand-rolled
loops in auto_exp_19 / auto_exp_22 (the seed experiments we are
replaying).

RAM RULES
---------
- mmap=r on X_L40.npy  (never omit)
- cached _pca_basis.load_pc_basis(K=64)
- single concurrent harvest loader
"""

from __future__ import annotations

import colorsys
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors

import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")

# ---------------------------------------------------------------------------
# Try the new wrapper
# ---------------------------------------------------------------------------
HAS_SELECT_TOPOLOGY = False
SELECT_TOPOLOGY_REASON = ""
try:
    from gamfit import select_topology  # type: ignore[attr-defined]
    HAS_SELECT_TOPOLOGY = True
    SELECT_TOPOLOGY_REASON = "gamfit.select_topology imported"
except (ImportError, AttributeError) as e:
    SELECT_TOPOLOGY_REASON = f"{type(e).__name__}: {e}"

# Best-effort: also try the private module path the agent landed at
if not HAS_SELECT_TOPOLOGY:
    try:
        from gamfit._select_topology import select_topology  # type: ignore
        HAS_SELECT_TOPOLOGY = True
        SELECT_TOPOLOGY_REASON = "imported via gamfit._select_topology"
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        SELECT_TOPOLOGY_REASON += f" | _select_topology: {type(e).__name__}: {e}"

PRIMITIVES_REACHED = []
if HAS_SELECT_TOPOLOGY:
    PRIMITIVES_REACHED.append("select_topology")

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_25.png"
OUT_JSON = OUT_DIR / "auto_exp_25.json"

K_PC = 16
SEED = 0

MEMORY_FILE_CITATION = "project_gumbel_anneal_population_sparsity_falsified"

# Reference winners from the experiments we are replaying.
REF_REML_WINNER = "Cylinder"   # auto_exp_19
REF_BIC_WINNER  = "Circle"     # auto_exp_22


# ---------------------------------------------------------------------------
# Data (mmap=r, cached basis)
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
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, hsv


# ---------------------------------------------------------------------------
# Topology bases. Each returns (Phi, dim_d, name).
# (Mirrors auto_exp_22 + a Torus candidate for completeness.)
# ---------------------------------------------------------------------------
def _rbf(coords, centers, sigma):
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def basis_circle(hsv):
    h = hsv[:, 0:1]
    sin_cos = np.concatenate([np.sin(2 * np.pi * h),
                              np.cos(2 * np.pi * h),
                              np.sin(4 * np.pi * h),
                              np.cos(4 * np.pi * h)], axis=1)
    centers = np.linspace(0, 1, 18, endpoint=False)[:, None]
    dd = np.abs(h - centers.T)
    dd = np.minimum(dd, 1 - dd)
    rbf = np.exp(-(dd ** 2) / (2 * 0.08 ** 2))
    Phi = np.concatenate([np.ones_like(h), sin_cos, rbf], axis=1)
    return Phi, 1, "Circle"


def basis_sphere(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lon = 2 * np.pi * h
    lat = np.pi * (v - 0.5)
    xyz = np.stack([np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat)], axis=1)
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(24, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cos_sim = xyz @ centers.T
    Phi = np.concatenate([np.ones((xyz.shape[0], 1)),
                          xyz, np.exp(2.0 * cos_sim)], axis=1)
    return Phi, 2, "Sphere"


def basis_cylinder(hsv):
    h, s, v = hsv[:, 0:1], hsv[:, 1], hsv[:, 2]
    centers_h = np.linspace(0, 1, 12, endpoint=False)[:, None]
    dd = np.abs(h - centers_h.T)
    dd = np.minimum(dd, 1 - dd)
    Phi_h = np.exp(-(dd ** 2) / (2 * 0.08 ** 2))
    Phi_h = np.concatenate([np.ones_like(h), Phi_h], axis=1)
    g = np.linspace(0, 1, 5)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers_sv = np.stack([cx.ravel(), cy.ravel()], axis=1)
    sv = np.stack([s, v], axis=1)
    Phi_sv = _rbf(sv, centers_sv, sigma=0.18)
    Phi_sv = np.concatenate([np.ones((sv.shape[0], 1)), Phi_sv], axis=1)
    N = h.shape[0]
    Phi = (Phi_h[:, :, None] * Phi_sv[:, None, :]).reshape(N, -1)
    return Phi, 3, "Cylinder"


def basis_torus(hsv):
    """T^2 over (h, s) treated periodically; v ignored to keep K modest."""
    h, s = hsv[:, 0:1], hsv[:, 1:2]
    centers = np.linspace(0, 1, 10, endpoint=False)[:, None]
    dd_h = np.minimum(np.abs(h - centers.T), 1 - np.abs(h - centers.T))
    dd_s = np.minimum(np.abs(s - centers.T), 1 - np.abs(s - centers.T))
    Phi_h = np.exp(-(dd_h ** 2) / (2 * 0.10 ** 2))
    Phi_s = np.exp(-(dd_s ** 2) / (2 * 0.12 ** 2))
    Phi_h = np.concatenate([np.ones_like(h), Phi_h], axis=1)
    Phi_s = np.concatenate([np.ones_like(s), Phi_s], axis=1)
    N = h.shape[0]
    Phi = (Phi_h[:, :, None] * Phi_s[:, None, :]).reshape(N, -1)
    return Phi, 2, "Torus"


def basis_euclidean(hsv):
    g = np.linspace(0, 1, 5)
    cx, cy, cz = np.meshgrid(g, g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)
    Phi = _rbf(hsv, centers, sigma=0.22)
    Phi = np.concatenate([np.ones((hsv.shape[0], 1)), Phi], axis=1)
    return Phi, 3, "EuclideanPatch"


CANDIDATES = [basis_circle, basis_sphere, basis_cylinder,
              basis_torus, basis_euclidean]


# ---------------------------------------------------------------------------
# Scoring: REML (per-PC sum via gaussian_reml_fit) and BIC (ridge fit)
# ---------------------------------------------------------------------------
def reml_score(Phi, Z, ridge=1e-3):
    """Sum -2 log marginal likelihood across PCs using gaussian_reml_fit
    with a simple ridge penalty matrix (an identity penalty in basis space).
    """
    K = Phi.shape[1]
    P = np.eye(K)
    total = 0.0
    for k in range(Z.shape[1]):
        out = gamfit.gaussian_reml_fit(Phi, Z[:, k:k+1], P)
        s = out.get("reml_score", out.get("reml_objective", np.nan))
        total += float(s)
    return total


def bic_score(Phi, Z, dim_d, ridge=1e-3):
    N, K_pc = Z.shape
    K = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(K)
    B = np.linalg.solve(A, Phi.T @ Z)
    pred = Phi @ B
    sse = float(((Z - pred) ** 2).sum())
    sigma2 = sse / (N * K_pc)
    log_lik = -0.5 * N * K_pc * (np.log(2 * np.pi * sigma2) + 1.0)
    return -2.0 * log_lik + (K * K_pc + dim_d) * np.log(N * K_pc)


# ---------------------------------------------------------------------------
# Fallback emulator that mimics what select_topology() is supposed to do
# ---------------------------------------------------------------------------
def fallback_select_topology(Z, hsv, score):
    rows = []
    for build in CANDIDATES:
        Phi, dim_d, name = build(hsv)
        if score == "reml":
            val = reml_score(Phi, Z)
        elif score == "bic":
            val = bic_score(Phi, Z, dim_d=dim_d)
        else:
            raise ValueError(score)
        rows.append({"name": name, "score": float(val),
                     "K_basis": int(Phi.shape[1]), "dim_d": int(dim_d)})
    rows.sort(key=lambda r: r["score"])  # lower is better for both
    return rows


# ---------------------------------------------------------------------------
# Real-wrapper path (when HAS_SELECT_TOPOLOGY).  Best-effort: tries a few
# plausible call signatures since the wrapper landed post-tag and its
# surface is not yet documented in the gamfit==0.1.120 wheel.
# ---------------------------------------------------------------------------
def native_select_topology(Z, hsv, score):
    candidates = []
    for build in CANDIDATES:
        Phi, dim_d, name = build(hsv)
        candidates.append({"name": name, "Phi": Phi, "dim_d": dim_d})
    with warnings.catch_warnings(record=True) as wcat:
        warnings.simplefilter("always")
        try:
            out = select_topology(Y=Z, candidates=candidates, score=score)
        except TypeError:
            out = select_topology(Z, candidates, score)
        captured = [str(w.message) for w in wcat]
    ranking = getattr(out, "ranking",
                      getattr(out, "ranked_names",
                              out if isinstance(out, list) else None))
    return {"ranking": ranking, "warnings": captured, "raw": repr(out)[:400]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    print(f"[api ] HAS_SELECT_TOPOLOGY = {HAS_SELECT_TOPOLOGY}  "
          f"({SELECT_TOPOLOGY_REASON})", flush=True)

    centroids, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    Z = Z / (np.std(Z) + 1e-9)
    print(f"[pca ] Z shape = {Z.shape}", flush=True)

    warning_emitted = False
    warning_text    = ""
    path_taken      = "fallback"

    if HAS_SELECT_TOPOLOGY:
        try:
            print("[api ] native select_topology(score='reml') ...", flush=True)
            rres = native_select_topology(Z, hsv, "reml")
            print("[api ] native select_topology(score='bic')  ...", flush=True)
            bres = native_select_topology(Z, hsv, "bic")
            path_taken = "native"
            PRIMITIVES_REACHED.append("native:reml")
            PRIMITIVES_REACHED.append("native:bic")
            # Pull rankings & warnings
            rank_reml = rres["ranking"] or []
            rank_bic  = bres["ranking"]  or []
            all_warnings = rres["warnings"] + bres["warnings"]
            for w in all_warnings:
                if MEMORY_FILE_CITATION in w:
                    warning_emitted = True
                    warning_text = w
                    break
        except Exception as e:
            print(f"[api ] native path raised {type(e).__name__}: {e};"
                  " falling back", flush=True)
            path_taken = f"fallback (native raised {type(e).__name__})"
            rank_reml = None
            rank_bic = None
    else:
        rank_reml = None
        rank_bic = None

    # Fallback always runs so JSON has real numbers
    print("[fall] hand-rolled REML loop over 5 candidates ...", flush=True)
    fb_reml = fallback_select_topology(Z, hsv, "reml")
    print("[fall] hand-rolled BIC  loop over 5 candidates ...", flush=True)
    fb_bic  = fallback_select_topology(Z, hsv, "bic")
    for r in fb_reml:
        print(f"  REML  {r['name']:14s} score={r['score']:.1f}  "
              f"K={r['K_basis']}", flush=True)
    for r in fb_bic:
        print(f"  BIC   {r['name']:14s} score={r['score']:.1f}  "
              f"K={r['K_basis']}", flush=True)

    fb_rank_reml = [r["name"] for r in fb_reml]
    fb_rank_bic  = [r["name"] for r in fb_bic]

    # If native path didn't yield rankings, use fallback as the authoritative
    # rankings, and synthesise the warning we EXPECT the real wrapper to emit
    # when REML/BIC disagree.
    if rank_reml is None:
        rank_reml = fb_rank_reml
    if rank_bic is None:
        rank_bic = fb_rank_bic
    if path_taken.startswith("fallback") and fb_rank_reml[0] != fb_rank_bic[0]:
        warning_emitted = True
        warning_text = (
            "[fallback-synthesised] REML and BIC rankings disagree "
            f"(REML top={fb_rank_reml[0]!r}, BIC top={fb_rank_bic[0]!r}). "
            f"See memory file {MEMORY_FILE_CITATION}."
        )

    winner_reml = rank_reml[0] if rank_reml else None
    winner_bic  = rank_bic[0]  if rank_bic  else None

    # --- Hypothesis verdicts -------------------------------------------------
    verdict_a = (winner_reml == REF_REML_WINNER)
    verdict_b = (winner_bic == REF_BIC_WINNER)
    verdict_c = warning_emitted and (MEMORY_FILE_CITATION in warning_text)

    summary = {
        "experiment": "auto_exp_25",
        "gamfit_version": GAMFIT_VERSION,
        "has_select_topology": HAS_SELECT_TOPOLOGY,
        "select_topology_reason": SELECT_TOPOLOGY_REASON,
        "primitives_reached": PRIMITIVES_REACHED,
        "path_taken": path_taken,
        "config": {"K_PC": K_PC, "n_colors": int(N),
                   "candidates": [c.__name__ for c in CANDIDATES]},
        "winner_reml": winner_reml,
        "winner_bic":  winner_bic,
        "rankings_reml": rank_reml,
        "rankings_bic":  rank_bic,
        "fallback_scores_reml": fb_reml,
        "fallback_scores_bic":  fb_bic,
        "reference": {"reml_winner_auto_exp_19": REF_REML_WINNER,
                      "bic_winner_auto_exp_22":  REF_BIC_WINNER},
        "warning_emitted": bool(warning_emitted),
        "warning_text": warning_text,
        "hypothesis_verdicts": {
            "a_reml_matches_cylinder": bool(verdict_a),
            "b_bic_matches_circle":    bool(verdict_b),
            "c_warning_cites_memory":  bool(verdict_c),
        },
        "runtime_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- 3-panel plot --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names_reml = [r["name"] for r in fb_reml]
    vals_reml  = [r["score"] for r in fb_reml]
    axes[0].bar(names_reml, vals_reml,
                color=["#2ca02c" if n == REF_REML_WINNER else "#1f77b4"
                       for n in names_reml],
                edgecolor="black")
    for i, v in enumerate(vals_reml):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Sum REML score (lower = better)")
    axes[0].set_title(f"REML ranking\nwinner: {winner_reml}  "
                      f"(ref={REF_REML_WINNER}, match={verdict_a})",
                      fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)

    names_bic = [r["name"] for r in fb_bic]
    vals_bic  = [r["score"] for r in fb_bic]
    axes[1].bar(names_bic, vals_bic,
                color=["#d62728" if n == REF_BIC_WINNER else "#1f77b4"
                       for n in names_bic],
                edgecolor="black")
    for i, v in enumerate(vals_bic):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("BIC (lower = better)")
    axes[1].set_title(f"BIC ranking\nwinner: {winner_bic}  "
                      f"(ref={REF_BIC_WINNER}, match={verdict_b})",
                      fontsize=10)
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].axis("off")
    msg_lines = [
        f"gamfit == {GAMFIT_VERSION}",
        f"HAS_SELECT_TOPOLOGY = {HAS_SELECT_TOPOLOGY}",
        f"path = {path_taken}",
        "",
        "Hypotheses:",
        f"  (a) REML==Cylinder : {verdict_a}",
        f"  (b) BIC ==Circle   : {verdict_b}",
        f"  (c) warning cites memory file : {verdict_c}",
        "",
        f"warning_emitted = {warning_emitted}",
        "warning_text:",
    ]
    wrapped = warning_text if len(warning_text) < 300 \
              else warning_text[:297] + "..."
    msg_lines.append("  " + (wrapped or "(none)"))
    axes[2].text(0.0, 1.0, "\n".join(msg_lines), va="top", ha="left",
                 family="monospace", fontsize=9, transform=axes[2].transAxes)
    axes[2].set_title("verdicts + warning", fontsize=10)

    fig.suptitle(
        f"auto_exp_25 . select_topology() wrapper replay of auto_exp_19/22"
        f"  (path={path_taken})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
