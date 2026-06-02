"""Alternating manifold recovery — fast, no neural encoder, no slow joint solver.

The Manifold-SAE encoder fails to learn token positions (the gradient-trained
amortized net falls into a linear basin and never wraps a circle), and gam's
joint `sae_manifold_fit` recovers good positions but is impractically slow. This
takes a third path: classic alternating optimization where every step is cheap.

Per atom k, repeat:
  1. residual   r = X - sum_{j!=k} a_j * g_j(t_j)         (backfit de-contamination)
  2. fit curve  g_k via gamfit Gaussian-REML on (t_k, r, by=a_k)   (~20 ms)
  3. REPLACE positions+amps by GLOBAL grid search: evaluate g_k on a dense grid
     and, for each token, snap t_k to the grid point that best explains r, with
     the optimal amplitude in closed form. This is the key move — a global
     argmin per token can place a point anywhere on the curve (incl. the far
     side of a circle), so it escapes the local-minima trap that kills the
     gradient-trained encoder.
  4. soft top-k: keep each token's K_active strongest atoms, zero the rest.

Open atoms use a Duchon spline; closed atoms a periodic cyclic B-spline. The
topology that reconstructs better per atom is kept automatically.
"""
from __future__ import annotations

import numpy as np
import torch
import gamfit

torch.set_default_dtype(torch.float64)


def _fit_curve_open(t, y, by, grid):
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(t), torch.from_numpy(y), basis="duchon", basis_order=2,
        by=torch.from_numpy(by),
    )
    coef = np.asarray(res["coefficients"])
    knots = np.asarray(res["knots_or_centers"]).reshape(-1, 1)
    Bg = np.asarray(gamfit.duchon_basis(torch.from_numpy(grid.reshape(-1, 1)),
                                        torch.from_numpy(knots), m=2))
    return Bg @ coef


def _fit_curve_closed(t, y, by, grid, n_knots=14):
    B, P = gamfit.periodic_spline_curve_basis(torch.from_numpy(t), n_knots=n_knots,
                                              degree=3, penalty_order=2)
    B = np.asarray(B)
    X = by[:, None] * B
    beta, _ = gamfit.gaussian_weighted_ridge(
        torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(np.asarray(P)),
        torch.from_numpy(np.ones(len(t))), ridge_lambda=1e-3)
    beta = np.asarray(beta)
    Bg, _ = gamfit.periodic_spline_curve_basis(torch.from_numpy(grid), n_knots=n_knots,
                                               degree=3, penalty_order=2)
    return np.asarray(Bg) @ beta


def _refit_atom(t, y, by, grid):
    """Fit the atom curve on a dense grid, choosing open vs closed by residual."""
    best, best_resid = None, np.inf
    for fn in (_fit_curve_open, _fit_curve_closed):
        try:
            G = fn(t, y, by, grid)
        except Exception:
            continue
        # reconstruction residual at the training points (nearest-grid eval)
        # cheap proxy: project y onto G's span via the grid assignment below;
        # use the produced curve's own fit quality via a quick grid snap.
        resid = _snap_residual(y, by, G)
        if resid < best_resid:
            best, best_resid = G, resid
    return best


def _snap_residual(y, by, G):
    """Sum of squared residuals after snapping each (y,by) to its best grid point."""
    Gn2 = (G ** 2).sum(1)                       # (T,)
    proj = y @ G.T                              # (N, T) = <y_n, G_g>
    # optimal scalar s per (n,g): s = proj/Gn2; residual = ||y||^2 - proj^2/Gn2
    r2 = (y ** 2).sum(1)[:, None] - proj ** 2 / np.maximum(Gn2, 1e-12)[None, :]
    return float(r2.min(1).sum())


def _snap(y, G, grid):
    """Global per-token grid search: returns (t_new, a_new) placing each token at
    the grid point that best explains y with the closed-form optimal amplitude."""
    Gn2 = np.maximum((G ** 2).sum(1), 1e-12)    # (T,)
    proj = y @ G.T                              # (N, T)
    r2 = (y ** 2).sum(1)[:, None] - proj ** 2 / Gn2[None, :]   # (N, T)
    g = r2.argmin(1)                            # (N,)
    a = proj[np.arange(len(y)), g] / Gn2[g]
    return grid[g], np.clip(a, 0.0, None)


def als_recover(X, K, grid_size=200, n_iter=40, k_active=2, seed=0, verbose=False,
                init="subspace", rank=2):
    N, D = X.shape
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    t = rng.random((N, K))
    a = np.full((N, K), 0.1)
    if init == "subspace":
        # Init each atom as a smooth random curve in its OWN random 2D subspace,
        # so atoms start in distinct directions (breaks the identifiability
        # symmetry that lets ALS converge to mixed/leaky atoms).
        curves = np.zeros((K, grid_size, D))
        for k in range(K):
            Bk, _ = np.linalg.qr(rng.standard_normal((D, 2)))   # (D,2) orthonormal
            ang = 2 * np.pi * grid
            intr = np.stack([np.cos(ang), np.sin(ang)], -1)      # smooth 2D loop
            curves[k] = 0.3 * intr @ Bk.T
    else:
        curves = rng.standard_normal((K, grid_size, D)) * 0.05
    for it in range(n_iter):
        for k in range(K):
            recon_others = np.zeros((N, D))
            for j in range(K):
                if j == k:
                    continue
                idx = np.clip((t[:, j] * grid_size).astype(int), 0, grid_size - 1)
                recon_others += a[:, j, None] * curves[j][idx]
            r = X - recon_others
            G = _refit_atom(t[:, k], r, a[:, k], grid)
            if G is None:
                continue
            if rank is not None:
                # Project the curve onto its own dominant `rank`-D subspace: an
                # atom is a curve in a low-D plane, so energy outside that plane
                # is cross-talk leaked from other manifolds. Stripping it each
                # step keeps atoms in distinct orthogonal subspaces.
                mu = G.mean(0, keepdims=True)
                Gc = G - mu
                U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
                G = mu + (U[:, :rank] * S[:rank]) @ Vt[:rank]
            curves[k] = G
            t[:, k], a[:, k] = _snap(r, G, grid)
        # soft top-k sparsity across atoms per token
        if k_active < K:
            order = np.argsort(-a, axis=1)
            keep = np.zeros_like(a, dtype=bool)
            rows = np.arange(N)[:, None]
            keep[rows, order[:, :k_active]] = True
            a = np.where(keep, a, 0.0)
        if verbose:
            recon = _reconstruct(curves, t, a, grid_size)
            ev = 1 - ((X - recon) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
            print(f"  [als it {it:2d}] EV={ev:.4f}", flush=True)
    return curves, t, a


def _reconstruct(curves, t, a, grid_size):
    N, K = t.shape
    D = curves.shape[2]
    out = np.zeros((N, D))
    for k in range(K):
        idx = np.clip((t[:, k] * grid_size).astype(int), 0, grid_size - 1)
        out += a[:, k, None] * curves[k][idx]
    return out
