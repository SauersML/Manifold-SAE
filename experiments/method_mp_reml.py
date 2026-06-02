"""Matching-pursuit encode + gamfit-REML decode — sparse curve-atom coding for
the SUPERPOSED regime.  No clustering, no kNN, no graph.

The joint free-position SAE converges to an entangled ~28% attractor: with free
per-atom positions, reconstruction is *lower* at a smeared configuration than at
the true curves, so gradient descent (any init) slides into the smear. The cure
is HARD assignment in the encode (a token may use an atom at exactly one place)
plus a CONSTRAINED decode (REML smooth curve) — alternated, this is sparse coding
with curve atoms and it has the true curves as its stable point.

  ENCODE (per token, matching pursuit): greedily pick the (atom, position) whose
    point on the current curve best explains the residual, subtract, repeat up to
    `na` times. Hard, sparse, no smear.
  DECODE (per atom, REML): gather the tokens that selected it, backfit-subtract
    the other selected atoms to isolate this atom's contribution, and REML-fit a
    smooth curve through (position, contribution). gamfit picks the smoothness.

Warm-started from the moment/co-activation curves, then alternated to clean up the
overlap that non-orthogonal superposition introduces.
"""
from __future__ import annotations

import numpy as np
import torch
import gamfit

from experiments.method_moment_reml import recover as moment_recover

torch.set_default_dtype(torch.float64)


def _reml_once(pos, tgt, closed):
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(pos), torch.from_numpy(tgt), basis="duchon", basis_order=2,
        by=torch.from_numpy(np.ones(len(pos))), periodic=bool(closed),
        period=(1.0 if closed else None))
    return np.asarray(res["fitted"])


def _reml_fit(pos, tgt, G, trim=0.15):
    """ROBUST REML Duchon fit of D-dim targets at positions pos in [0,1).

    Matching-pursuit assignment and backfit subtraction leave a minority of
    corrupted targets (wrong-atom selections, co-active-atom leakage) which a
    plain least-squares fit threads through, creating phantom interior loops. We
    fit once, drop the `trim` fraction with the largest residual, and refit on the
    clean majority -- a one-shot trimmed estimator that removes the tangles."""
    grid = np.linspace(0, 1, G, endpoint=False)
    U = np.linalg.svd(tgt - tgt.mean(0), full_matrices=False)[2][:2]
    Y2 = (tgt - tgt.mean(0)) @ U.T
    phi = np.arctan2(Y2[:, 1], Y2[:, 0]); sp = np.sort(phi)
    closed = np.diff(np.concatenate([sp, [sp[0] + 2 * np.pi]])).max() < 0.6
    fitted = _reml_once(pos, tgt, closed)
    if trim > 0 and len(pos) > 30:
        resid = np.linalg.norm(tgt - fitted, axis=1)
        keep = resid <= np.quantile(resid, 1 - trim)
        if keep.sum() > 20:
            pos, tgt = pos[keep], tgt[keep]
            fitted = _reml_once(pos, tgt, closed)
    o = np.argsort(pos); ts, fs = pos[o], fitted[o]
    if closed:
        ts = np.concatenate([ts - 1, ts, ts + 1]); fs = np.concatenate([fs, fs, fs], 0)
    return np.stack([np.interp(grid, ts, fs[:, j]) for j in range(tgt.shape[1])], 1)


def _encode(X, C, na):
    """Matching pursuit: for each token return up to na selected (atom, grididx)."""
    N, D = X.shape; K, G, _ = C.shape
    R = X.copy()
    sel_k = np.full((N, na), -1, int); sel_g = np.zeros((N, na), int)
    for step in range(na):
        best_k = np.zeros(N, int); best_g = np.zeros(N, int); best_d = np.full(N, np.inf)
        for k in range(K):
            d2 = ((R[:, None, :] - C[k][None]) ** 2).sum(2)      # (N,G)
            g = d2.argmin(1); dd = d2[np.arange(N), g]
            upd = dd < best_d
            best_d = np.where(upd, dd, best_d)
            best_k = np.where(upd, k, best_k); best_g = np.where(upd, g, best_g)
        sel_k[:, step] = best_k; sel_g[:, step] = best_g
        R = R - C[best_k, best_g]
    return sel_k, sel_g


def recover(X, K, na=3, n_iter=6, G=200):
    X = np.asarray(X, float); N, D = X.shape
    grid = np.linspace(0, 1, G, endpoint=False)
    curves = moment_recover(X, K, G=G)
    curves = [c if c is not None else 1e-3 * np.random.default_rng(k).standard_normal((G, D))
              for k, c in enumerate(curves)]
    for it in range(n_iter):
        C = np.stack(curves, 0)
        sel_k, sel_g = _encode(X, C, na)
        new = []
        for k in range(K):
            rows = np.where((sel_k == k).any(1))[0]
            if len(rows) < 12:
                new.append(curves[k]); continue
            pos, tgt = [], []
            for n in rows:
                slot = np.where(sel_k[n] == k)[0][0]
                others = sum(C[sel_k[n, s], sel_g[n, s]] for s in range(na)
                             if s != slot and sel_k[n, s] >= 0)
                pos.append(grid[sel_g[n, slot]]); tgt.append(X[n] - others)
            try:
                new.append(_reml_fit(np.array(pos), np.array(tgt), G))
            except Exception:
                new.append(curves[k])
        curves = new
    return curves
