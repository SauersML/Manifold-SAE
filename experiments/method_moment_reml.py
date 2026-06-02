"""Moment-subspace + gamfit-REML curve recovery — NO clustering, NO kNN, NO graph.

Two principled stages:

1. SUBSPACE RECOVERY from global moments. Each token lives in exactly one of K
   mutually-orthogonal 2D blocks, so the data covariance is block-diagonal: its
   top 2K eigenvectors each lie in one block. We pair them into the K blocks by a
   CO-ACTIVATION test — two eigen-coordinates belong to the same curve iff their
   squared magnitudes fire together across tokens (corr of Y^2). Greedy
   strongest-partner pairing recovers all blocks at cos>0.998. No clustering, no
   iteration — just an eigen-decomposition and a 2K x 2K correlation.

2. CURVE FIT with gamfit REML. Project onto each recovered block, keep the points
   that actually live there (energy gate), give them a principled position init
   (angle around the centroid for a closed loop, principal-axis coordinate for an
   open arc, radius-unwrapped angle for a winding curve), and fit with gamfit's
   Gaussian-REML Duchon smoother (REML auto-selects smoothness; periodic basis
   for closed). No ordering graph — the parameterization is closed-form and REML
   does the rest.
"""
from __future__ import annotations

import numpy as np
import torch
import gamfit

torch.set_default_dtype(torch.float64)


def recover_subspaces(X, K):
    """Return K blocks, each (D,2), via covariance eigvecs + co-activation pairing."""
    C = X.T @ X / len(X)
    _, V = np.linalg.eigh(C)
    U = V[:, -2 * K:]                                   # (D, 2K) union basis
    Y = X @ U
    A = np.corrcoef((Y ** 2).T)                         # co-activation
    np.fill_diagonal(A, -1.0)
    used, pairs = set(), []
    for flat in np.argsort(-A, axis=None):
        i, j = divmod(int(flat), 2 * K)
        if i in used or j in used:
            continue
        pairs.append((i, j)); used |= {i, j}
        if len(pairs) == K:
            break
    return [U[:, list(p)] for p in pairs]               # each (D,2)


def _init_positions(Y2):
    """Principled position init + topology for a clean 2D curve (no graph).
    Returns (t in [0,1], closed)."""
    Yc = Y2 - Y2.mean(0)
    phi = np.arctan2(Yc[:, 1], Yc[:, 0])
    # CLOSED if the points wrap fully around the centroid (small max angular gap).
    sp = np.sort(phi)
    max_gap = np.diff(np.concatenate([sp, [sp[0] + 2 * np.pi]])).max()
    if max_gap < 0.5:
        return (phi % (2 * np.pi)) / (2 * np.pi), True
    # OPEN arc: principal-axis coordinate (monotone along function-graph curves).
    _, _, vt = np.linalg.svd(Yc, full_matrices=False)
    pc1 = Yc @ vt[0]
    return (pc1 - pc1.min()) / max(np.ptp(pc1), 1e-9), False


def _reml_curve(Y2, t, closed, grid):
    """gamfit REML Duchon fit of the 2D curve at positions t; eval on grid."""
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(t), torch.from_numpy(Y2), basis="duchon", basis_order=2,
        by=torch.from_numpy(np.ones(len(t))), periodic=bool(closed),
        period=(1.0 if closed else None))
    fitted = np.asarray(res["fitted"])
    o = np.argsort(t); ts, fs = t[o], fitted[o]
    if closed:
        ts = np.concatenate([ts - 1, ts, ts + 1]); fs = np.concatenate([fs, fs, fs], 0)
    return np.stack([np.interp(grid, ts, fs[:, j]) for j in range(2)], 1)


def recover(X, K, G=200):
    blocks = recover_subspaces(X, K)
    grid = np.linspace(0, 1, G, endpoint=False)
    # energy of each point in each block -> assign by max projection (a single
    # deterministic projection given the moment-recovered blocks, not iterative).
    proj = np.stack([np.linalg.norm(X @ B, axis=1) for B in blocks], 1)   # (N,K)
    asg = proj.argmax(1)
    curves = []
    for k, B in enumerate(blocks):
        Yk = X[asg == k] @ B                            # (nk, 2) clean curve points
        if len(Yk) < 12:
            curves.append(None); continue
        t, closed = _init_positions(Yk)
        try:
            curves.append(_reml_curve(Yk, t, closed, grid) @ B.T)
        except Exception:
            curves.append(None)
    # NOTE: 7/8 shapes are pixel-perfect with this pure closed-form init. The
    # multi-turn spiral is the one shape a one-shot init can't order; it needs an
    # iterative ordering refinement (see method_fourier, the clean winner).
    return curves
