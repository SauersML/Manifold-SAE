"""auto_71.py - Joint Levenberg-Marquardt with geodesic-acceleration
correction for the unsupervised color-manifold Duchon fit.

Replaces the alternating (T, B) optimization in
`color_manifold_gam.fit_unsupervised_manifold` with a joint LM step on
p = (T, B), plus the Transtrum-Sethna second-order "geodesic"
correction (a directional finite-difference of the residual's second
derivative along the LM step).

We use the 4^3 = 64-center Duchon basis (m=3, periodic=None) which the
spec authorizes as a fallback when wall-time tightens; with 886 colors
and K_basis=64 the joint normal equations have ~6754 params, which is
the largest we can solve densely per LM iteration in a reasonable time
budget. The alternating baseline uses the same 64-center basis for an
apples-to-apples comparison.

Compared to auto_exp_11/12: those documented (a) PCA-init underfit
ceiling ~0.72-0.74 train R^2 and (b) seed-dependence with pairwise
Procrustes disparities 0.85-0.99 across random T inits. We test
whether joint LM + geodesic acceleration breaks both.

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

import gamfit  # noqa: E402
from plot_color_geometry import load_xkcd_colors  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402
from _pca_basis import load_pc_basis, project  # noqa: E402
import color_manifold_gam as cmg  # noqa: E402


OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_71.png"
OUT_JSON = OUT_DIR / "auto_71.json"

N_TEMPLATES = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
K_PC = 64
D_LATENT = 3
CENTERS_PER_AXIS = 4            # 4^3 = 64 centers (fallback per spec hard rule)
DUCHON_M = 3                    # m=3 required for d=3 with periodic=False
N_SEEDS = 3                     # honoring the 10-min wall-time rule
LM_MAX_ITERS = 20
ALT_N_ITERS = 50                # match auto_exp_12's "still drifting" budget
ALPHA_GA = 0.75                 # geodesic-acceleration accept threshold
EPS_GA = 1e-3                   # FD step for the directional 2nd-derivative


# ---------------------------------------------------------------------------
# Duchon basis -- thin wrapper around gamfit (matches color_manifold_gam,
# but pinned to m=3 to avoid the per-call m-search).
# ---------------------------------------------------------------------------
def duchon_basis(T: np.ndarray, centers: np.ndarray) -> np.ndarray:
    per = (False,) * T.shape[1]
    return np.asarray(gamfit.duchon_basis(T, centers, m=DUCHON_M,
                                            periodic_per_axis=per))


def duchon_penalty(centers: np.ndarray) -> np.ndarray:
    per = (False,) * centers.shape[1]
    P = np.asarray(gamfit.duchon_function_norm_penalty(
        centers, m=DUCHON_M, periodic_per_axis=per))
    P = 0.5 * (P + P.T)
    diag_max = float(np.max(np.abs(np.diag(P)))) if P.shape[0] > 0 else 1.0
    return P + 1e-8 * max(diag_max, 1.0) * np.eye(P.shape[0])


def lattice_centers(per_side: int, d: int) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, per_side) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.flatten() for m in mesh], axis=1)


# ---------------------------------------------------------------------------
# Residual + Jacobian
#
# r = vec(Z - Phi(T) @ B)        shape (N*Kpc,)
# p = (T flat, B flat)           shape (N*d + Kb*Kpc,)
# We use the row-major flatten:  r[i*Kpc + j] = Z[i,j] - sum_k Phi[i,k] B[k,j]
#
# T-Jacobian: dr[i,j]/dT[i,s] = -(dPhi[i,:]/dT[i,s]) @ B[:, j]
#   computed via forward-mode finite differences -- ONE basis recompute per
#   spatial axis, because perturbing all rows in axis s by the same epsilon
#   shifts each row's contribution independently (cross-row coupling is
#   exactly zero in Phi, since Phi[i, k] depends only on T[i, :]).
#
# B-Jacobian: dr[i,j]/dB[k,j'] = -Phi[i,k] * delta_{j,j'}
#   block-diagonal across the Kpc output channels.
# ---------------------------------------------------------------------------
def compute_dPhi_dT(T: np.ndarray, centers: np.ndarray, eps: float = 1e-4
                     ) -> np.ndarray:
    """Per-row directional derivative of Phi w.r.t. each T-axis.

    Returns ``dPhi`` with shape (d, N, Kb) such that
    ``dPhi[s, i, k] = d Phi[i, k] / d T[i, s]``.
    """
    d = T.shape[1]
    out = []
    for s in range(d):
        Tp = T.copy(); Tp[:, s] += eps
        Tm = T.copy(); Tm[:, s] -= eps
        out.append((duchon_basis(Tp, centers) - duchon_basis(Tm, centers))
                   / (2.0 * eps))
    return np.stack(out, axis=0)  # (d, N, Kb)


def jt_vec(Phi: np.ndarray, dPhi: np.ndarray, B: np.ndarray,
             v: np.ndarray) -> np.ndarray:
    """Compute J^T v where v is in residual space (shape N*Kpc).
    Reuses the analytic block structure; no J materialization."""
    N, Kb = Phi.shape
    d = dPhi.shape[0]
    Kpc = B.shape[1]
    V = v.reshape(N, Kpc)
    g = np.zeros(N * d + Kb * Kpc, dtype=np.float64)
    g[N * d:] = (-(Phi.T @ V)).reshape(-1)
    VB = V @ B.T  # (N, Kb)
    g_T = np.zeros((N, d), dtype=np.float64)
    for s in range(d):
        g_T[:, s] = -(dPhi[s] * VB).sum(axis=1)
    g[:N * d] = g_T.reshape(-1)
    return g


def build_normal_eq(Phi: np.ndarray, dPhi: np.ndarray, B: np.ndarray,
                     R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the joint normal equations (H = J^T J, g = J^T r) WITHOUT
    materializing J -- J has shape (N*Kpc, N*d + Kb*Kpc) which is multi-GB.

    Layout: parameter vector p = [T flat (rowmajor, (i, s)), B flat (rowmajor, (k, j))].
    R is the residual reshaped to (N, Kpc).

    Returns (H, g) of shape (n_p, n_p) and (n_p,).
    """
    N, Kb = Phi.shape
    d = dPhi.shape[0]
    Kpc = B.shape[1]
    n_T = N * d
    n_B = Kb * Kpc
    n_p = n_T + n_B

    # ----- gradient g = J^T r -----
    g = np.zeros(n_p, dtype=np.float64)
    # B-block:   g_B = -(Phi^T @ R)   shape (Kb, Kpc), flat in (k, j).
    g_B = -(Phi.T @ R)
    g[n_T:] = g_B.reshape(-1)
    # T-block:   g_T[i, s] = -dPhi[s, i, :] @ B @ R[i, :]
    #          = -sum_k dPhi[s, i, k] * (B @ R[i, :])_k
    # Compute (R @ B.T) once, shape (N, Kb).
    RB = R @ B.T  # (N, Kb)
    g_T = np.zeros((N, d), dtype=np.float64)
    for s in range(d):
        g_T[:, s] = -(dPhi[s] * RB).sum(axis=1)
    g[:n_T] = g_T.reshape(-1)

    # ----- Hessian H = J^T J -----
    H = np.zeros((n_p, n_p), dtype=np.float64)

    # H_BB: block-diagonal in j, each block = Phi^T Phi (Kb, Kb).
    PtP = Phi.T @ Phi  # (Kb, Kb)
    for j in range(Kpc):
        idx = n_T + np.arange(Kb) * Kpc + j
        H[np.ix_(idx, idx)] = PtP

    # H_TT: block-diagonal in i, each block (d, d).
    # block i = (dPhi[:, i, :] @ B) @ (dPhi[:, i, :] @ B)^T,  shape (d, d).
    dPhi_T_swap = np.transpose(dPhi, (1, 0, 2))  # (N, d, Kb)
    # M[i] = dPhi_T_swap[i] @ B  -> (N, d, Kpc) via einsum
    M = np.einsum("ndk,kj->ndj", dPhi_T_swap, B)  # (N, d, Kpc)
    # Per-row d×d gram:
    Htt_blocks = np.einsum("ndj,nej->nde", M, M)  # (N, d, d)
    for i in range(N):
        idx = i * d + np.arange(d)
        H[np.ix_(idx, idx)] = Htt_blocks[i]

    # H_TB:  H_TB[(i, s), (k, j)] = Phi[i, k] * (dPhi[s, i, :] @ B[:, j])
    #     = Phi[i, k] * M[i, s, j]
    # Sign: J_T entry is -dPhi @ B, J_B entry is -Phi, product = + Phi * (dPhi @ B).
    # Flat T-index (i, s) -> i*d + s; flat B-index (k, j) -> k*Kpc + j.
    # Build block of shape (N*d, Kb*Kpc):
    # Reshape M -> (N, d, Kpc); for each (i, s, k, j): Phi[i,k] * M[i, s, j].
    # That's an outer product over (k, j) per (i, s).
    # Vectorized: H_TB[i*d+s, k*Kpc+j] = Phi[i,k] * M[i,s,j]
    # Use einsum producing shape (N, d, Kb, Kpc) then reshape.
    # Want H_TB[n, s, k, j] = Phi[n, k] * M[n, s, j]  (positive sign because
    # J_T * J_B = (-dPhi B) * (-Phi) = +Phi * dPhi*B).
    H_TB = np.einsum("nk,nsj->nskj", Phi, M)  # (N, d, Kb, Kpc)
    H_TB_flat = H_TB.reshape(N * d, Kb * Kpc)
    H[:n_T, n_T:] = H_TB_flat
    H[n_T:, :n_T] = H_TB_flat.T

    return H, g


def residual(Z: np.ndarray, Phi: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (Z - Phi @ B).reshape(-1)


def unpack(p: np.ndarray, N: int, d: int, Kb: int, Kpc: int
            ) -> tuple[np.ndarray, np.ndarray]:
    T = p[: N * d].reshape(N, d)
    B = p[N * d :].reshape(Kb, Kpc)
    return T, B


def pack(T: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.concatenate([T.reshape(-1), B.reshape(-1)])


def loss_from_p(p: np.ndarray, Z: np.ndarray, centers: np.ndarray,
                  N: int, d: int, Kb: int, Kpc: int) -> tuple[float, np.ndarray, np.ndarray]:
    T, B = unpack(p, N, d, Kb, Kpc)
    Phi = duchon_basis(T, centers)
    r = residual(Z, Phi, B)
    return 0.5 * float(r @ r), Phi, r


# ---------------------------------------------------------------------------
# Joint LM + geodesic acceleration
# ---------------------------------------------------------------------------
def joint_lm_ga(Z: np.ndarray, centers: np.ndarray, T0: np.ndarray,
                 max_iters: int = LM_MAX_ITERS, verbose: bool = False
                 ) -> dict:
    """Joint LM on p = (T, B) with Transtrum-Sethna 2nd-order correction."""
    N, Kpc = Z.shape
    d = T0.shape[1]
    Kb = duchon_basis(T0[:2], centers).shape[1]

    # Initialize B by ridge on Phi(T0).
    Phi = duchon_basis(T0, centers)
    Ppen = duchon_penalty(centers)
    Aint = Phi.T @ Phi + 1e-3 * Ppen
    B = np.linalg.solve(Aint, Phi.T @ Z)
    p = pack(T0, B)

    lam = 1e-2
    history = []
    accepts = 0
    rejects = 0

    f_curr, Phi, r = loss_from_p(p, Z, centers, N, d, Kb, Kpc)
    history.append({"iter": -1, "loss": f_curr, "lam": lam,
                     "ga_used": False, "step_norm": 0.0})

    for it in range(max_iters):
        Tcur = p[: N * d].reshape(N, d)
        Bcur = p[N * d :].reshape(Kb, Kpc)
        # Per-row directional derivatives of Phi w.r.t. T (axis-wise FD).
        dPhi = compute_dPhi_dT(Tcur, centers, eps=1e-4)
        R = r.reshape(N, Kpc)
        H, g = build_normal_eq(Phi, dPhi, Bcur, R)
        diagH = np.diag(H).copy()
        diagH = np.maximum(diagH, 1e-12)

        # LM solve.  Factor once via Cholesky and reuse for the geodesic
        # correction (same A, different RHS).
        import scipy.linalg as sla
        A = H + lam * np.diag(diagH)
        try:
            cho = sla.cho_factor(A, lower=True)
            delta = sla.cho_solve(cho, -g)
        except Exception:
            lam *= 10.0
            rejects += 1
            history.append({"iter": it, "loss": f_curr, "lam": lam,
                              "ga_used": False, "step_norm": 0.0,
                              "rejected": True, "reason": "lin_solve_fail"})
            if lam > 1e12:
                break
            continue

        # Geodesic-acceleration correction.
        h = EPS_GA
        f_plus, _, r_plus = loss_from_p(p + h * delta, Z, centers, N, d, Kb, Kpc)
        f_minus, _, r_minus = loss_from_p(p - h * delta, Z, centers, N, d, Kb, Kpc)
        # K = d^2 r / dp^2 . delta delta  (FD)
        K = (r_plus - 2.0 * r + r_minus) / (h * h)
        try:
            delta2 = sla.cho_solve(cho, -0.5 * jt_vec(Phi, dPhi, Bcur, K))
        except Exception:
            delta2 = np.zeros_like(delta)

        nrm1 = float(np.linalg.norm(delta))
        nrm2 = float(np.linalg.norm(delta2))
        use_ga = nrm1 > 0 and (nrm2 / max(nrm1, 1e-30)) <= ALPHA_GA
        step = delta + delta2 if use_ga else delta

        # Try the step.
        p_new = p + step
        f_new, Phi_new, r_new = loss_from_p(p_new, Z, centers, N, d, Kb, Kpc)

        if f_new < f_curr:
            # Accept.
            accepts += 1
            p = p_new
            f_curr = f_new
            Phi = Phi_new
            r = r_new
            lam = max(lam / 3.0, 1e-9)
            step_norm = float(np.linalg.norm(step))
            history.append({"iter": it, "loss": f_curr, "lam": lam,
                              "ga_used": use_ga, "step_norm": step_norm,
                              "accepted": True})
            if verbose:
                print(f"    iter {it:2d} ACC loss={f_curr:.4e} lam={lam:.2e}"
                      f" ga={use_ga} |delta|={nrm1:.2e} |d2|={nrm2:.2e}")
            # Convergence: relative gradient norm tiny.
            if step_norm / max(1.0, np.linalg.norm(p)) < 1e-6:
                break
        else:
            # Reject -> bump lam.
            rejects += 1
            lam *= 5.0
            history.append({"iter": it, "loss": f_curr, "lam": lam,
                              "ga_used": use_ga, "step_norm": 0.0,
                              "accepted": False})
            if verbose:
                print(f"    iter {it:2d} REJ loss={f_curr:.4e} lam={lam:.2e}")
            if lam > 1e12:
                break

    T_fin, B_fin = unpack(p, N, d, Kb, Kpc)
    return {"T": T_fin, "B": B_fin, "loss_curve": [h["loss"] for h in history],
            "accepts": accepts, "rejects": rejects, "history": history,
            "n_iters_run": len(history) - 1}


# ---------------------------------------------------------------------------
# Alternating reference fit -- same 64-center basis, PCA init, N_iters=50.
# Re-implement here so we share the basis sizing with the LM run.
# ---------------------------------------------------------------------------
def alternating_fit(Z: np.ndarray, centers: np.ndarray, T_init: np.ndarray,
                     n_iters: int = ALT_N_ITERS) -> dict:
    d = T_init.shape[1]
    grid_per_axis = 9 if d == 4 else 20 if d == 3 else 40
    T = T_init.copy()
    history = []

    # Build a coarse grid for the projection step.
    g = grid_per_axis
    grid = lattice_centers(g, d)
    Pgrid = duchon_basis(grid, centers)

    for it in range(n_iters):
        Phi = duchon_basis(T, centers)
        Pen = duchon_penalty(centers)
        A = Phi.T @ Phi + 1e-3 * Pen
        B = np.linalg.solve(A, Phi.T @ Z)

        # Projection: nearest grid point on the fitted surface.
        Z_grid = Pgrid @ B
        zn = (Z ** 2).sum(1, keepdims=True)
        zg = (Z_grid ** 2).sum(1, keepdims=True).T
        sqd = zn - 2 * Z @ Z_grid.T + zg
        best = np.argmin(sqd, axis=1)
        T_new = grid[best]

        # Per-axis rescale to keep T inside [0,1] (matches color_manifold_gam).
        lo = T_new.min(0, keepdims=True); hi = T_new.max(0, keepdims=True)
        span = hi - lo
        T_new = np.where(span > 1e-8, (T_new - lo) / np.maximum(span, 1e-8),
                          0.5 * np.ones_like(T_new))

        Z_hat = Phi @ B
        loss = 0.5 * float(((Z - Z_hat) ** 2).sum())
        history.append(loss)
        T = T_new

    # Final B given converged T.
    Phi = duchon_basis(T, centers)
    Pen = duchon_penalty(centers)
    A = Phi.T @ Phi + 1e-3 * Pen
    B = np.linalg.solve(A, Phi.T @ Z)
    Z_hat = Phi @ B
    loss = 0.5 * float(((Z - Z_hat) ** 2).sum())
    history.append(loss)
    return {"T": T, "B": B, "loss_curve": history}


# ---------------------------------------------------------------------------
# Procrustes disparity (matches auto_exp_11).
# ---------------------------------------------------------------------------
def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    nA = np.linalg.norm(A0); nB = np.linalg.norm(B0)
    if nA < 1e-12 or nB < 1e-12:
        return float("nan")
    A0n = A0 / nA; B0n = B0 / nB
    U, _, Vt = np.linalg.svd(B0n.T @ A0n, full_matrices=False)
    R = Vt.T @ U.T
    M = A0 @ R
    s = float((B0 * M).sum() / max((M * M).sum(), 1e-12))
    A_aligned = s * M + B.mean(0, keepdims=True)
    return float(((B - A_aligned) ** 2).sum() / max((B0 ** 2).sum(), 1e-12))


def train_r2(Z: np.ndarray, Phi: np.ndarray, B: np.ndarray) -> float:
    Z_hat = Phi @ B
    ss_res = float(((Z - Z_hat) ** 2).sum())
    ss_tot = float(((Z - Z.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] X_L40 mmap", flush=True)
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_TEMPLATES
    print(f"[load] X.shape={X.shape} n_raw={n_raw}", flush=True)

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_TEMPLATES + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    N = len(kept)
    print(f"[load] N={N} filtered colors", flush=True)

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    print(f"[load] Z.shape={Z.shape} EVR={float(basis['evr'].sum()):.3f}",
          flush=True)

    centers = lattice_centers(CENTERS_PER_AXIS, D_LATENT)
    Kb = duchon_basis(centers[:2], centers).shape[1]
    print(f"[setup] centers={centers.shape}  Kb={Kb}  total params="
          f"{N * D_LATENT + Kb * K_PC}", flush=True)

    # Pre-build PCA-init T for alternating baseline.
    Zc = Z - Z.mean(0, keepdims=True)
    Vt = cmg._pca_Vt(Zc)
    T_pca = Zc @ Vt.T[:, :D_LATENT]
    lo, hi = T_pca.min(0, keepdims=True), T_pca.max(0, keepdims=True)
    T_pca_unit = (T_pca - lo) / np.maximum(hi - lo, 1e-8)

    lm_runs = []
    alt_runs = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        T_rand = rng.uniform(0.0, 1.0, size=(N, D_LATENT))

        # Geodesic LM with random init.
        print(f"\n[seed {seed}] joint LM + GA (random init)", flush=True)
        t0 = time.time()
        try:
            lm_res = joint_lm_ga(Z, centers, T_rand, max_iters=LM_MAX_ITERS,
                                   verbose=False)
            lm_wall = time.time() - t0
            Phi_lm = duchon_basis(lm_res["T"], centers)
            r2_lm = train_r2(Z, Phi_lm, lm_res["B"])
            print(f"  R2={r2_lm:+.4f}  iters={lm_res['n_iters_run']}  "
                  f"acc/rej={lm_res['accepts']}/{lm_res['rejects']}  "
                  f"wall={lm_wall:.1f}s", flush=True)
        except Exception as exc:
            print(f"  LM failed: {exc!r}", flush=True)
            raise

        lm_runs.append({"seed": seed, "T": lm_res["T"], "r2": r2_lm,
                          "wall": lm_wall, "loss_curve": lm_res["loss_curve"],
                          "n_iters": lm_res["n_iters_run"],
                          "accepts": lm_res["accepts"],
                          "rejects": lm_res["rejects"]})

        # Alternating with PCA-init for seed 0, random-init thereafter
        # (mirrors auto_exp_11's protocol: 1 PCA + N random).
        T_alt_init = T_pca_unit if seed == 0 else T_rand
        label = "alt(pca-init)" if seed == 0 else f"alt(rand seed={seed})"
        print(f"[seed {seed}] {label}", flush=True)
        t0 = time.time()
        alt_res = alternating_fit(Z, centers, T_alt_init, n_iters=ALT_N_ITERS)
        alt_wall = time.time() - t0
        Phi_alt = duchon_basis(alt_res["T"], centers)
        r2_alt = train_r2(Z, Phi_alt, alt_res["B"])
        print(f"  R2={r2_alt:+.4f}  wall={alt_wall:.1f}s", flush=True)
        alt_runs.append({"seed": seed, "T": alt_res["T"], "r2": r2_alt,
                           "wall": alt_wall, "loss_curve": alt_res["loss_curve"],
                           "init": "pca" if seed == 0 else "rand",
                           "label": label})

    # -------------------- Procrustes among LM seeds --------------------
    n_lm = len(lm_runs)
    D_lm = np.zeros((n_lm, n_lm))
    for i in range(n_lm):
        for j in range(n_lm):
            if i == j:
                continue
            D_lm[i, j] = procrustes_disparity(lm_runs[i]["T"], lm_runs[j]["T"])
    iu = np.triu_indices(n_lm, k=1)
    lm_proc_median = float(np.median(D_lm[iu])) if iu[0].size > 0 else float("nan")

    n_alt = len(alt_runs)
    D_alt = np.zeros((n_alt, n_alt))
    for i in range(n_alt):
        for j in range(n_alt):
            if i == j:
                continue
            D_alt[i, j] = procrustes_disparity(alt_runs[i]["T"], alt_runs[j]["T"])
    iu_a = np.triu_indices(n_alt, k=1)
    alt_proc_median = float(np.median(D_alt[iu_a])) if iu_a[0].size > 0 else float("nan")

    # ---------------------------- Plot ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: train-loss curves
    ax = axes[0, 0]
    for r in lm_runs:
        ax.semilogy(r["loss_curve"], color="#1f77b4", alpha=0.7,
                     label=f"LM seed={r['seed']}" if r["seed"] == 0 else None)
    for r in alt_runs:
        col = "#d62728" if r["init"] == "pca" else "#ff7f0e"
        ax.semilogy(r["loss_curve"], color=col, alpha=0.7,
                     ls="--" if r["init"] == "pca" else ":",
                     label=r["label"] if r["seed"] == 0 or r["seed"] == 1 else None)
    ax.set_xlabel("iter"); ax.set_ylabel("0.5 * ||r||^2 (train)")
    ax.set_title("(1) train-loss curves: joint LM+GA vs alternating")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: final train R2 (box-and-whisker)
    ax = axes[0, 1]
    r2_lm_vals = [r["r2"] for r in lm_runs]
    r2_alt_vals = [r["r2"] for r in alt_runs]
    parts = ax.boxplot([r2_alt_vals, r2_lm_vals],
                          labels=["alternating", "joint LM+GA"],
                          patch_artist=True)
    for patch, color in zip(parts["boxes"], ["#d62728", "#1f77b4"]):
        patch.set_facecolor(color); patch.set_alpha(0.45)
    # overlay individual points
    for i, vals in enumerate([r2_alt_vals, r2_lm_vals]):
        ax.scatter(np.full(len(vals), i + 1) + 0.05 * np.random.randn(len(vals)),
                    vals, color="black", s=22, zorder=5)
    ax.set_ylabel("final train R^2")
    ax.set_title(f"(2) final R^2 across {N_SEEDS} seeds")
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: pairwise Procrustes heatmap for LM seeds
    ax = axes[1, 0]
    im = ax.imshow(D_lm, cmap="viridis", vmin=0, vmax=max(D_lm.max(), 1e-6))
    ax.set_xticks(range(n_lm)); ax.set_xticklabels([f"s{r['seed']}" for r in lm_runs])
    ax.set_yticks(range(n_lm)); ax.set_yticklabels([f"s{r['seed']}" for r in lm_runs])
    for i in range(n_lm):
        for j in range(n_lm):
            ax.text(j, i, f"{D_lm[i, j]:.2f}", ha="center", va="center",
                     color="white" if D_lm[i, j] > D_lm.max() / 2 else "black",
                     fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"(3) pairwise Procrustes among LM seeds\n"
                  f"median={lm_proc_median:.3f}  "
                  f"(alt median={alt_proc_median:.3f}; "
                  f"auto_exp_11 reported 0.85-0.99)")

    # Panel 4: wall time
    ax = axes[1, 1]
    walls_lm = [r["wall"] for r in lm_runs]
    walls_alt = [r["wall"] for r in alt_runs]
    x = np.arange(N_SEEDS)
    w = 0.4
    ax.bar(x - w/2, walls_alt, w, color="#d62728", label="alternating", alpha=0.85)
    ax.bar(x + w/2, walls_lm, w, color="#1f77b4", label="joint LM+GA", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {i}" for i in range(N_SEEDS)])
    ax.set_ylabel("wall time (s)")
    ax.set_title("(4) wall time per seed")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"auto_71 · joint LM + geodesic-acceleration on the unsupervised "
        f"Duchon manifold · cogito L40 · N={N} colors · "
        f"d={D_LATENT} · {CENTERS_PER_AXIS}^3 centers (Kb={Kb})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT_PNG}", flush=True)

    payload = {
        "config": {
            "N": int(N), "K_PC": int(K_PC), "D_latent": int(D_LATENT),
            "centers_per_axis": int(CENTERS_PER_AXIS),
            "K_basis": int(Kb), "duchon_m": int(DUCHON_M),
            "n_seeds": int(N_SEEDS), "lm_max_iters": int(LM_MAX_ITERS),
            "alt_n_iters": int(ALT_N_ITERS),
            "alpha_ga": float(ALPHA_GA), "eps_ga": float(EPS_GA),
        },
        "joint_lm_ga": {
            "r2_per_seed": [float(r["r2"]) for r in lm_runs],
            "r2_mean": float(np.mean([r["r2"] for r in lm_runs])),
            "r2_std": float(np.std([r["r2"] for r in lm_runs])),
            "wall_per_seed": [float(r["wall"]) for r in lm_runs],
            "wall_mean": float(np.mean([r["wall"] for r in lm_runs])),
            "iters_per_seed": [int(r["n_iters"]) for r in lm_runs],
            "accepts_per_seed": [int(r["accepts"]) for r in lm_runs],
            "rejects_per_seed": [int(r["rejects"]) for r in lm_runs],
            "procrustes_median": float(lm_proc_median),
            "procrustes_matrix": D_lm.tolist(),
        },
        "alternating": {
            "labels": [r["label"] for r in alt_runs],
            "r2_per_seed": [float(r["r2"]) for r in alt_runs],
            "r2_mean": float(np.mean([r["r2"] for r in alt_runs])),
            "r2_std": float(np.std([r["r2"] for r in alt_runs])),
            "wall_per_seed": [float(r["wall"]) for r in alt_runs],
            "wall_mean": float(np.mean([r["wall"] for r in alt_runs])),
            "procrustes_median": float(alt_proc_median),
            "procrustes_matrix": D_alt.tolist(),
        },
        "notes": (
            "Joint LM with Transtrum-Sethna geodesic-acceleration "
            "correction. p = (T, B), residual r = vec(Z - Phi(T) B), "
            "Jacobian assembled analytically for B-block and via "
            "axis-wise forward-mode FD for the T-block (Phi[i,k] depends "
            "only on T[i,:], so a single FD recompute per spatial axis "
            "gives the entire T-Jacobian).  Geodesic correction uses a "
            "central-difference 2nd derivative of r along the LM step, "
            "accepted when ||delta2|| <= 0.75 ||delta||.  LM lambda "
            "adapts as /3 on accept, *5 on reject.  Both methods share "
            "the same 4^3=64-center Duchon basis (m=3, non-periodic) "
            "for an apples-to-apples comparison; the spec authorises "
            "this when wall-time tightens.  Alternating baseline uses "
            "PCA init for seed 0 and the same random init as LM for "
            "later seeds (mirroring auto_exp_11)."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
