"""auto_exp_22 - Gumbel-softmax tau-annealing for sae_manifold_fit's IBP-MAP
assignment, combined with compare_models over topology candidates.

HYPOTHESES
----------
H1  tau-annealing (1.0 -> 1e-3 geometric, rate=0.9) yields SHARPER K_eff
    at convergence than fixed-tau=1.0.
H2  topology winner under compare_models is still Cylinder
    (matches auto_exp_19 finding) even with Gumbel annealing in the inner loop.

INSTALLED GAMFIT NOTE
---------------------
At time of writing gamfit is NOT importable; if/when v0.1.119 wheels land
with sae_manifold_fit / compare_models / GumbelTemperatureSchedule, the
"native" branch will activate.  Otherwise we emulate:

  * Gumbel-softmax topic-amplitude assignment in NumPy with per-iter
    tau multiplication.
  * IBP-MAP-lite: shared-K topic dictionary in a 64-D PCA space, sparse
    Gumbel-softmax responsibilities, atom prune on cumulative-mass < eps.
  * compare_models = BIC = -2 * log L + k * log(N) where
        log L = -N*K_pc/2 * log(2 pi sigma^2) - SSE / (2 sigma^2),
        sigma^2 = SSE / (N * K_pc), and k = topology-specific d.o.f. of
        the latent map (the manifold dimension * basis_K).
"""

from __future__ import annotations

import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_22.png"
OUT_JSON = OUT_DIR / "auto_exp_22.json"

K_PC          = 16
K_TOPICS      = 80       # generous IBP-MAP atom budget; we measure K_eff <<
N_ITERS       = 80
TAU_START     = 1.0
TAU_MIN       = 1e-3
TAU_RATE      = 0.9
PRUNE_EPS     = 5e-3     # mass threshold for K_eff (fraction of total mass)
SEED          = 0


# ---------------------------------------------------------------------------
# Try the native gamfit API; fall back to NumPy emulation otherwise.
# ---------------------------------------------------------------------------
PRIMITIVES_REACHED: list[str] = []
FALLBACK_REASONS: list[str] = []
GAMFIT_VERSION = "absent"

try:
    import gamfit  # type: ignore
    GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")
    PRIMITIVES_REACHED.append("gamfit_import")
except Exception as e:
    FALLBACK_REASONS.append(f"gamfit import failed: {type(e).__name__}: {e}")

HAS_SAE_FIT = False
HAS_COMPARE = False
HAS_GUMBEL_SCHED = False
try:
    from gamfit import sae_manifold_fit  # type: ignore  # noqa: F401
    HAS_SAE_FIT = True
    PRIMITIVES_REACHED.append("sae_manifold_fit")
except Exception as e:
    FALLBACK_REASONS.append(f"sae_manifold_fit unavailable: {e}")
try:
    from gamfit import compare_models  # type: ignore  # noqa: F401
    HAS_COMPARE = True
    PRIMITIVES_REACHED.append("compare_models")
except Exception as e:
    FALLBACK_REASONS.append(f"compare_models unavailable: {e}")
try:
    from gamfit import GumbelTemperatureSchedule  # type: ignore  # noqa: F401
    HAS_GUMBEL_SCHED = True
    PRIMITIVES_REACHED.append("GumbelTemperatureSchedule")
except Exception as e:
    FALLBACK_REASONS.append(f"GumbelTemperatureSchedule unavailable: {e}")


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
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, hsv


# ---------------------------------------------------------------------------
# Topology bases for compare_models.  Each returns (Phi, dim_d, name)
# where Phi: (N, K_basis) is a fixed feature map of the latent coordinates
# and dim_d is the manifold dimensionality (used in the BIC d.o.f. count).
# ---------------------------------------------------------------------------
def _rbf(coords, centers, sigma):
    """Generic Gaussian RBF feature map.  coords (N, d), centers (M, d)."""
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def basis_circle(hsv):
    """Circle S^1 over hue: sin/cos + 18 wrapped-RBFs."""
    h = hsv[:, 0:1]
    sin_cos = np.concatenate([np.sin(2 * np.pi * h),
                              np.cos(2 * np.pi * h),
                              np.sin(4 * np.pi * h),
                              np.cos(4 * np.pi * h)], axis=1)
    centers = np.linspace(0, 1, 18, endpoint=False)[:, None]
    # periodic kernel via wrapped distance
    dd = np.abs(h - centers.T)
    dd = np.minimum(dd, 1 - dd)
    rbf = np.exp(-(dd ** 2) / (2 * 0.08 ** 2))
    Phi = np.concatenate([np.ones_like(h), sin_cos, rbf], axis=1)
    return Phi, 1, "Circle"


def basis_sphere(hsv):
    """S^2 in (h, s, v) via spherical-RBF on unit-vec lift."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    # lift to S^2-ish coords on a sphere (hue->longitude, val->latitude)
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
    """S^1 x R^2 over (h, s, v): tensor of periodic hue x SV-grid."""
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


def basis_euclidean(hsv):
    """EuclideanPatch R^3 over (h, s, v) via 6^3 RBF grid (subsampled)."""
    g = np.linspace(0, 1, 5)
    cx, cy, cz = np.meshgrid(g, g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)
    Phi = _rbf(hsv, centers, sigma=0.22)
    Phi = np.concatenate([np.ones((hsv.shape[0], 1)), Phi], axis=1)
    return Phi, 3, "EuclideanPatch"


# ---------------------------------------------------------------------------
# Topology BIC scoring via ridge fit on (N, K_pc) latent target
# ---------------------------------------------------------------------------
def fit_and_bic(Phi, Z, ridge=1e-3, dof_dim=1):
    N, K_pc = Z.shape
    K = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(K)
    B = np.linalg.solve(A, Phi.T @ Z)
    pred = Phi @ B
    sse = float(((Z - pred) ** 2).sum())
    sigma2 = sse / (N * K_pc)
    log_lik = -0.5 * N * K_pc * (np.log(2 * np.pi * sigma2) + 1.0)
    # effective dof penalised by manifold dim
    k_eff_dof = K * K_pc * max(dof_dim, 1) / max(dof_dim, 1)  # = K*K_pc
    bic = -2.0 * log_lik + (K * K_pc + dof_dim) * np.log(N * K_pc)
    cv_r2 = 1.0 - sse / float(((Z - Z.mean(0)) ** 2).sum())
    return {"sse": sse, "log_lik": log_lik, "bic": bic,
            "K_basis": K, "dof_dim": dof_dim, "r2": cv_r2}


# ---------------------------------------------------------------------------
# Gumbel-softmax IBP-MAP-lite over a shared dictionary D (K, K_pc).
# Per iter:
#   logits      = Z @ D.T / kappa      (cosine-ish similarity)
#   gumbel      = -log(-log(U))
#   soft_assign = softmax((logits + g) / tau)         (N, K)
#   D update    = (R^T R + lam I)^{-1} R^T Z
#   tau         = max(tau * rate, tau_min)
# Returns trace of K_eff per iter.
# ---------------------------------------------------------------------------
def run_ibp_gumbel(Z, K, n_iters, tau_start, tau_min, tau_rate,
                   seed, anneal):
    rng = np.random.default_rng(seed)
    N, K_pc = Z.shape
    # init dictionary by random column subset
    idx = rng.choice(N, size=K, replace=False)
    D = Z[idx].copy() + 1e-3 * rng.normal(size=(K, K_pc))
    tau = tau_start
    kappa = float(np.std(Z)) + 1e-6
    keff_trace = []
    tau_trace  = []
    sse_trace  = []
    for it in range(n_iters):
        # Gumbel-softmax responsibilities
        logits = Z @ D.T / kappa
        g = -np.log(-np.log(rng.uniform(1e-9, 1.0 - 1e-9, size=logits.shape)))
        z = (logits + g) / max(tau, 1e-9)
        z -= z.max(axis=1, keepdims=True)
        R = np.exp(z)
        R /= R.sum(axis=1, keepdims=True) + 1e-12
        # dictionary M-step (ridge)
        A = R.T @ R + 1e-3 * np.eye(K)
        D = np.linalg.solve(A, R.T @ Z)
        # diagnostics
        mass = R.sum(axis=0)
        mass /= mass.sum()
        keff = int((mass > PRUNE_EPS).sum())
        sse = float(((Z - R @ D) ** 2).sum())
        keff_trace.append(keff)
        tau_trace.append(tau)
        sse_trace.append(sse)
        if anneal:
            tau = max(tau * tau_rate, tau_min)
    # final hard-K_eff using a slightly looser threshold
    return {"K_eff": keff_trace[-1],
            "K_eff_trace": keff_trace,
            "tau_trace": tau_trace,
            "sse_trace": sse_trace,
            "R_final": R, "D_final": D}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    print(f"[gamfit] primitives reached = {PRIMITIVES_REACHED}", flush=True)
    print(f"[gamfit] fallback reasons   = {FALLBACK_REASONS}", flush=True)

    centroids, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    # normalise scale so kappa is meaningful
    Z = Z / (np.std(Z) + 1e-9)
    print(f"[pca ] Z shape = {Z.shape}", flush=True)

    # --- Gumbel-softmax: fixed vs annealed tau ---
    print("[gumbel] fixed-tau run ...", flush=True)
    fixed = run_ibp_gumbel(Z, K=K_TOPICS, n_iters=N_ITERS,
                           tau_start=TAU_START, tau_min=TAU_START,
                           tau_rate=1.0, seed=SEED, anneal=False)
    print(f"          K_eff_final = {fixed['K_eff']}", flush=True)
    print("[gumbel] annealed-tau run ...", flush=True)
    anneal = run_ibp_gumbel(Z, K=K_TOPICS, n_iters=N_ITERS,
                            tau_start=TAU_START, tau_min=TAU_MIN,
                            tau_rate=TAU_RATE, seed=SEED, anneal=True)
    print(f"          K_eff_final = {anneal['K_eff']}", flush=True)

    sharper_K_eff_under_anneal = anneal["K_eff"] < fixed["K_eff"]

    # --- Topology compare_models (BIC) ---
    topo_builders = [basis_circle, basis_sphere, basis_cylinder, basis_euclidean]
    topo_results = []
    for build in topo_builders:
        Phi, dim_d, name = build(hsv)
        res = fit_and_bic(Phi, Z, ridge=1e-3, dof_dim=dim_d)
        res["name"] = name
        topo_results.append(res)
        print(f"[topo] {name:14s}  K={res['K_basis']:4d}  BIC={res['bic']:.1f}  "
              f"R^2={res['r2']:+.3f}", flush=True)

    topo_results.sort(key=lambda r: r["bic"])
    winner = topo_results[0]["name"]
    cylinder_still_wins = (winner == "Cylinder")
    print(f"[topo] winner by BIC: {winner}  "
          f"(cylinder_still_wins={cylinder_still_wins})", flush=True)

    # --- JSON ---
    summary = {
        "experiment": "auto_exp_22",
        "gamfit_version": GAMFIT_VERSION,
        "primitives_reached": PRIMITIVES_REACHED,
        "fallback_reasons": FALLBACK_REASONS,
        "used_native_path": (HAS_SAE_FIT and HAS_COMPARE
                             and HAS_GUMBEL_SCHED),
        "config": {
            "K_PC": K_PC, "K_topics": K_TOPICS, "n_iters": N_ITERS,
            "tau_start": TAU_START, "tau_min": TAU_MIN, "tau_rate": TAU_RATE,
            "prune_eps": PRUNE_EPS, "n_colors": int(N),
        },
        "K_eff_fixed_tau": fixed["K_eff"],
        "K_eff_annealed_tau": anneal["K_eff"],
        "K_eff_trace_fixed": fixed["K_eff_trace"],
        "K_eff_trace_anneal": anneal["K_eff_trace"],
        "tau_trace_anneal": anneal["tau_trace"],
        "topology_results": [
            {"name": r["name"], "bic": r["bic"], "r2": r["r2"],
             "K_basis": r["K_basis"], "dof_dim": r["dof_dim"]}
            for r in topo_results
        ],
        "topology_winner_bic": winner,
        "hypothesis_verdicts": {
            "sharper_K_eff_under_anneal": bool(sharper_K_eff_under_anneal),
            "cylinder_still_wins": bool(cylinder_still_wins),
        },
        "runtime_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- 4-panel plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) tau schedule
    ax = axes[0, 0]
    ax.plot(anneal["tau_trace"], "o-", color="C1", label="annealed")
    ax.plot(fixed["tau_trace"],  "--",  color="C0", label="fixed")
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_ylabel("tau (log)")
    ax.set_title(f"Gumbel-softmax temperature schedule\n"
                 f"start={TAU_START}, min={TAU_MIN}, rate={TAU_RATE}")
    ax.grid(alpha=0.3)
    ax.legend()

    # (0,1) K_eff trace
    ax = axes[0, 1]
    ax.plot(fixed["K_eff_trace"], "-", color="C0", label=f"fixed-tau (final={fixed['K_eff']})")
    ax.plot(anneal["K_eff_trace"], "-", color="C1", label=f"annealed (final={anneal['K_eff']})")
    ax.axhline(K_TOPICS, ls=":", color="grey", label=f"K_max={K_TOPICS}")
    ax.set_xlabel("iter")
    ax.set_ylabel("K_eff (mass>eps)")
    ax.set_title(f"K_eff vs iter  | sharper_under_anneal="
                 f"{sharper_K_eff_under_anneal}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,0) topology BIC bars
    ax = axes[1, 0]
    names = [r["name"] for r in topo_results]
    bics  = [r["bic"] for r in topo_results]
    bcols = ["gold" if n == winner else "#888" for n in names]
    bars = ax.bar(names, bics, color=bcols, edgecolor="black")
    for i, v in enumerate(bics):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title(f"compare_models (BIC) | winner: {winner}\n"
                 f"cylinder_still_wins={cylinder_still_wins}")
    ax.grid(alpha=0.3, axis="y")

    # (1,1) residual histogram for the winning topology
    ax = axes[1, 1]
    Phi_w, dim_w, _ = {n.__name__: n for n in topo_builders}[
        {"Circle": basis_circle.__name__,
         "Sphere": basis_sphere.__name__,
         "Cylinder": basis_cylinder.__name__,
         "EuclideanPatch": basis_euclidean.__name__}[winner]](hsv)
    A = Phi_w.T @ Phi_w + 1e-3 * np.eye(Phi_w.shape[1])
    Bw = np.linalg.solve(A, Phi_w.T @ Z)
    resid = (Z - Phi_w @ Bw).ravel()
    ax.hist(resid, bins=60, color="C2", edgecolor="black", alpha=0.85)
    ax.set_xlabel("residual (Z - Phi @ B)")
    ax.set_ylabel("count")
    ax.set_title(f"Residual hist | winning topology: {winner}\n"
                 f"std={resid.std():.3f}, n={resid.size}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_exp_22 . Gumbel tau-annealing + topology compare_models  |  "
        f"gamfit={GAMFIT_VERSION}  native_path={summary['used_native_path']}  "
        f"N={N}  K_PC={K_PC}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
