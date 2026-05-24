"""auto_exp_29 - score_scale=per_observation vs legacy vs per_effective_dim vs BIC.

GOAL
----
Test whether the *new* `score_scale=` kwarg on `gamfit.select_topology()`
(Tierney-Kadane normalizer + per-observation/per-effective-dim divisions)
actually changes the topology ranking on cogito L40 centroids vs the
legacy un-normalized REML score, and how those rankings stack against BIC.

HYPOTHESES
----------
(a) Rankings under per_observation != rankings under raw, for at least one
    pair of topologies.
(b) per_effective_dim is MORE CONSERVATIVE than per_observation (favors
    smaller-K basis).
(c) BIC ranking MORE EXTREMELY favors small-K than either per_observation
    or per_effective_dim REML.

PATH
----
gamfit==0.1.112 lacks select_topology and score_scale=, so this script
goes down the FALLBACK path, emulating the Tierney-Kadane normalizer in
Python (see auto_exp_25 for the same fallback shape). The four rankings
are computed by hand and a prediction slot is left for the v0.1.121 re-run.
"""

from __future__ import annotations

import colorsys
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

import gamfit
from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_29.png"
OUT_JSON = OUT_DIR / "auto_exp_29.json"
K_PC     = 16

GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")
HAS_NEW_API = hasattr(gamfit, "select_topology")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape  # H=7168 on cogito L40, NOT 4096
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
# Topology bases (cribbed from auto_exp_25 so this is apples-to-apples)
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
# Scoring primitives
# ---------------------------------------------------------------------------
def reml_raw_score(Phi, Z):
    """Sum raw REML score (-2 log mlik up to const) across PC columns."""
    K = Phi.shape[1]
    P = np.eye(K)
    total = 0.0
    for k in range(Z.shape[1]):
        out = gamfit.gaussian_reml_fit(Phi, Z[:, k:k+1], P)
        s = out.get("reml_score", out.get("reml_objective", np.nan))
        total += float(s)
    return total


def effective_dim(Phi, lam=1.0):
    """tr( Phi (Phi^T Phi + lam I)^-1 Phi^T ) — effective dof of a ridge fit."""
    K = Phi.shape[1]
    A = Phi.T @ Phi
    H = A @ np.linalg.solve(A + lam * np.eye(K), np.eye(K))
    return float(np.trace(H))


def bic_score(Phi, Z, dim_d, ridge=1e-3):
    """BIC = -2 log_lik + k log N (Python, k = K_basis * K_pc + dim_d)."""
    N, K_pc = Z.shape
    K = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(K)
    B = np.linalg.solve(A, Phi.T @ Z)
    pred = Phi @ B
    sse = float(((Z - pred) ** 2).sum())
    sigma2 = sse / (N * K_pc)
    log_lik = -0.5 * N * K_pc * (math.log(2 * math.pi * sigma2) + 1.0)
    return -2.0 * log_lik + (K * K_pc + dim_d) * math.log(N * K_pc)


def tierney_kadane_normalize(raw_score, N, K_basis, rank_S, K_pc):
    """Emulate the Tierney-Kadane Laplace normalizer.

    Per-column TK correction:  - 0.5 * (dim_H - rank_S) * log(2*pi) - 0.5 * log(N)
    where dim_H = K_basis (Hessian dim) and rank_S = rank of penalty
    (here rank_S = K_basis for identity penalty -> correction is just -0.5 log N).

    Multi-response sums over K_pc columns.
    """
    per_col = -0.5 * (K_basis - rank_S) * math.log(2 * math.pi) - 0.5 * math.log(N)
    return raw_score + K_pc * per_col


def kendall_tau(rank_a, rank_b):
    """Kendall's tau-b on two lists of names (same set, order = rank)."""
    names = list(rank_a)
    pos_a = {n: i for i, n in enumerate(rank_a)}
    pos_b = {n: i for i, n in enumerate(rank_b)}
    concord = discord = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = pos_a[names[i]] - pos_a[names[j]]
            b = pos_b[names[i]] - pos_b[names[j]]
            if a * b > 0:
                concord += 1
            elif a * b < 0:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}  has_new_api={HAS_NEW_API}",
          flush=True)

    centroids, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors, H = {centroids.shape[1]}",
          flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    Z = Z / (np.std(Z) + 1e-9)
    print(f"[pca ] Z shape = {Z.shape}", flush=True)

    path_taken = "fallback (Python TK normalizer emulator)"
    if HAS_NEW_API:
        path_taken = "native (gamfit.select_topology with score_scale=)"

    rows = []
    for build in CANDIDATES:
        Phi, dim_d, name = build(hsv)
        K_basis = int(Phi.shape[1])
        rank_S = K_basis  # identity penalty -> full rank
        edim = effective_dim(Phi, lam=1.0)
        raw = reml_raw_score(Phi, Z)
        # multi-response TK correction
        tk_corrected = tierney_kadane_normalize(raw, N, K_basis, rank_S, K_PC)
        per_obs = tk_corrected / N
        per_eff = tk_corrected / max(edim, 1.0)
        bic = bic_score(Phi, Z, dim_d=dim_d)
        rows.append({
            "name": name,
            "dim_d": int(dim_d),
            "K_basis": K_basis,
            "effective_dim": float(edim),
            "raw_reml": float(raw),
            "tk_corrected": float(tk_corrected),
            "per_observation": float(per_obs),
            "per_effective_dim": float(per_eff),
            "bic": float(bic),
        })
        print(f"  {name:15s}  K={K_basis:3d}  edim={edim:6.2f}  "
              f"raw={raw:+11.1f}  per_obs={per_obs:+8.3f}  "
              f"per_eff={per_eff:+8.3f}  bic={bic:+10.1f}", flush=True)

    def ranked(key):
        return [r["name"] for r in sorted(rows, key=lambda r: r[key])]

    rank_raw     = ranked("raw_reml")
    rank_per_obs = ranked("per_observation")
    rank_per_eff = ranked("per_effective_dim")
    rank_bic     = ranked("bic")

    print(f"[rank] raw           : {rank_raw}", flush=True)
    print(f"[rank] per_observation: {rank_per_obs}", flush=True)
    print(f"[rank] per_eff_dim   : {rank_per_eff}", flush=True)
    print(f"[rank] bic           : {rank_bic}", flush=True)

    # Pairwise Kendall tau
    rankings = {
        "raw": rank_raw, "per_observation": rank_per_obs,
        "per_effective_dim": rank_per_eff, "bic": rank_bic,
    }
    tau = {}
    keys = list(rankings.keys())
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            tau[f"{a}__vs__{b}"] = float(kendall_tau(rankings[a], rankings[b]))

    # --- Hypothesis verdicts ---
    # (a) per_observation != raw for at least one pair
    hyp_a = rank_raw != rank_per_obs

    # (b) per_effective_dim is MORE CONSERVATIVE (favors smaller K) than
    #     per_observation: compare top-1 K_basis. Smaller K = more conservative.
    top_obs_K = next(r["K_basis"] for r in rows if r["name"] == rank_per_obs[0])
    top_eff_K = next(r["K_basis"] for r in rows if r["name"] == rank_per_eff[0])
    hyp_b = top_eff_K < top_obs_K

    # (c) BIC top-1 has smaller K than per_observation AND per_effective_dim
    top_bic_K = next(r["K_basis"] for r in rows if r["name"] == rank_bic[0])
    hyp_c = (top_bic_K < top_obs_K) and (top_bic_K <= top_eff_K)

    print(f"[hyp]  (a) per_obs != raw         -> {hyp_a}", flush=True)
    print(f"[hyp]  (b) per_eff more conservative-> {hyp_b}  "
          f"(top_K obs={top_obs_K} eff={top_eff_K})", flush=True)
    print(f"[hyp]  (c) BIC most conservative  -> {hyp_c}  "
          f"(top_K bic={top_bic_K})", flush=True)

    summary = {
        "experiment": "auto_exp_29",
        "question": ("Does score_scale= on gamfit.select_topology change "
                     "topology ranking on cogito L40 vs the legacy un-"
                     "normalized score?"),
        "gamfit_version": GAMFIT_VERSION,
        "has_new_api": HAS_NEW_API,
        "path_taken": path_taken,
        "config": {
            "K_PC": K_PC, "N_colors": int(N),
            "H_residual": int(centroids.shape[1]),
            "candidates": [r["name"] for r in rows],
        },
        "rows": rows,
        "rankings": rankings,
        "kendall_tau_pairwise": tau,
        "kendall_tau_per_obs_vs_bic": tau["per_observation__vs__bic"],
        "kendall_tau_per_obs_vs_per_eff":
            tau["per_observation__vs__per_effective_dim"],
        "hypotheses": {
            "a_per_obs_differs_from_raw": bool(hyp_a),
            "b_per_eff_more_conservative_than_per_obs": bool(hyp_b),
            "c_bic_more_extreme_small_K_than_reml": bool(hyp_c),
        },
        "prediction_slot_for_v0_1_121_rerun": {
            "expected_native_per_observation_top1": rank_per_obs[0],
            "expected_native_per_eff_dim_top1":    rank_per_eff[0],
            "expected_native_raw_top1":            rank_raw[0],
            "note": ("When gamfit>=0.1.121 lands with select_topology + "
                     "score_scale=, re-run this script. If HAS_NEW_API is "
                     "True and the native ranking under each score_scale "
                     "differs from the fallback ranking above, the Python "
                     "TK normalizer emulator is wrong."),
        },
        "runtime_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- Plot: 4 ranking bars + concordance heatmap ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    score_keys = ["raw_reml", "per_observation", "per_effective_dim", "bic"]
    score_titles = ["raw REML (legacy)",
                    "per_observation (TK/N)",
                    "per_effective_dim (TK/edim)",
                    "BIC (Python)"]
    panels = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    for ax, key, title in zip(panels, score_keys, score_titles):
        srt = sorted(rows, key=lambda r: r[key])
        names = [r["name"] for r in srt]
        vals  = [r[key]    for r in srt]
        bars = ax.barh(range(len(names)), vals,
                       color=["gold", "#1f77b4", "#1f77b4",
                              "#1f77b4", "#1f77b4"],
                       edgecolor="black", lw=0.8)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{title}\ntop-1: {names[0]}", fontsize=10)
        ax.grid(alpha=0.3, axis="x")

    # Concordance heatmap
    keys = list(rankings.keys())
    M = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            M[i, j] = kendall_tau(rankings[a], rankings[b])
    ax = axes[1, 1]
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(M[i,j]) > 0.5 else "black")
    ax.set_title("Kendall tau pairwise concordance", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    verdict_text = (
        f"gamfit version: {GAMFIT_VERSION}\n"
        f"path: {path_taken}\n\n"
        f"Hypotheses\n"
        f"  (a) per_obs != raw          : {hyp_a}\n"
        f"  (b) per_eff more conservative: {hyp_b}\n"
        f"      top-K: obs={top_obs_K}  eff={top_eff_K}\n"
        f"  (c) BIC most extreme small-K: {hyp_c}\n"
        f"      top-K: bic={top_bic_K}\n\n"
        f"Top-1 by score:\n"
        f"  raw           : {rank_raw[0]}\n"
        f"  per_obs       : {rank_per_obs[0]}\n"
        f"  per_eff_dim   : {rank_per_eff[0]}\n"
        f"  bic           : {rank_bic[0]}\n\n"
        f"tau(per_obs, bic)      = {tau['per_observation__vs__bic']:+.3f}\n"
        f"tau(per_obs, per_eff)  = "
        f"{tau['per_observation__vs__per_effective_dim']:+.3f}\n"
    )
    ax.text(0.0, 1.0, verdict_text, va="top", ha="left", family="monospace",
            fontsize=9)

    fig.suptitle(f"auto_exp_29  score_scale= ranking comparison "
                 f"(gamfit=={GAMFIT_VERSION}, {path_taken})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
