"""General curve recovery for points sampled ON a 1D manifold (any shape).

Well-posed model: each token is a point on a curve g(t) (+ small noise), with
features superposed additively across orthogonal subspaces. Projecting a cluster
onto its recovered subspace gives clean curve points (orthogonality cancels the
other features), so per cluster we have points scattered on an unknown 1D curve.

Recovery is shape-AGNOSTIC, via the minimum spanning tree of a tangent-aware
k-NN graph:
  1. ORDER + TOPOLOGY from the MST. The MST of points on a curve is a path along
     it. For a CLOSED loop the MST is the loop minus its single weakest edge, so
     the path's two ends are SPATIALLY ADJACENT (the broken closing edge); for an
     OPEN curve the two ends are the true, far-apart endpoints. So ordering by
     tree distance from one end gives the arc-length order for either topology,
     and "is the broken edge a normal curve step or a long jump?" gives open vs
     closed — a topological signal, not a point-space heuristic.
  2. FIT a smooth spline (open Duchon or closed periodic Duchon, gamfit >=0.1.144)
     along the recovered arc-length parameter.

Tangent-aware edges (only connect points whose local tangents and connecting edge
align) stop the graph short-cutting across nearby-but-different parts of the curve
(adjacent spiral turns, sine peaks, thin-loop sides). No shape assumptions — open
or closed, star or non-star (offset circles, cardioids, ellipses, spirals, ...).
"""
from __future__ import annotations

import numpy as np
import torch
import gamfit
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components, minimum_spanning_tree

torch.set_default_dtype(torch.float64)


def _order_mst(Y, k):
    """Unified ordering + topology via the minimum spanning tree.

    The MST of points on a curve is a path along the curve. For a CLOSED loop the
    MST is the loop with its single weakest edge removed, so the two ends of the
    path are SPATIALLY ADJACENT (the broken closing edge); for an OPEN curve the
    two ends are the true, far-apart endpoints. So: order = MST path (tree
    distance from one end); topology = whether the two ends are spatially close.
    Robust because it uses tree connectivity, not point-space gaps or per-point
    scores, and works for any shape (loops, arcs, spirals, waves)."""
    n = len(Y)
    A = _knn_graph(Y, k, tangent_aware=True)
    nc, _ = connected_components(A, directed=False)
    if nc > 1:                                          # tangent filter split it
        A = _knn_graph(Y, k, tangent_aware=False)
    mst = minimum_spanning_tree(A)
    mst = mst + mst.T
    d0 = shortest_path(mst, indices=0, directed=False)
    d0[~np.isfinite(d0)] = -1
    L1 = int(np.argmax(d0))
    dL1 = shortest_path(mst, indices=L1, directed=False)
    dL1[~np.isfinite(dL1)] = dL1[np.isfinite(dL1)].max()
    L2 = int(np.argmax(dL1))
    order = np.argsort(dL1)
    # Closure: is the broken MST edge (Y[L1]->Y[L2]) a normal curve step (the
    # loop's closing edge) or a long jump (an open curve's far endpoints)?
    # Compare to the typical step along the ordered curve, NOT the diameter, so
    # it is robust to elongated/non-uniform shapes.
    steps = np.linalg.norm(np.diff(Y[order], axis=0), axis=1)
    med_step = np.median(steps) + 1e-12
    closed = np.linalg.norm(Y[L1] - Y[L2]) < 6.0 * med_step
    return order, closed


def _tangents(Y, k):
    """Local unit tangent at each point = 1st PCA direction of its k neighbours."""
    n = len(Y)
    tree = cKDTree(Y)
    _, idx = tree.query(Y, min(k + 1, n))
    Tn = np.zeros_like(Y)
    for i in range(n):
        nb = Y[idx[i]] - Y[idx[i]].mean(0)
        _, _, vt = np.linalg.svd(nb, full_matrices=False)
        Tn[i] = vt[0]
    return Tn


def _knn_graph(Y, k, tangent_aware=True):
    """Symmetric kNN distance graph. With `tangent_aware`, an edge (i, j) is kept
    only when the two local tangents AND the edge direction all align — so points
    that are spatially close but on different parts of the curve (adjacent spiral
    turns, sine peaks, the two sides of a thin loop) are NOT connected, which
    kills the graph shortcuts that otherwise tangle the ordering."""
    n = len(Y)
    tree = cKDTree(Y)
    dist, idx = tree.query(Y, min(k + 1, n))
    Tn = _tangents(Y, k) if tangent_aware else None
    rows, cols, vals = [], [], []
    for i in range(n):
        for jj in range(idx.shape[1]):
            j = idx[i, jj]
            if j == i:
                continue
            if tangent_aware:
                e = Y[j] - Y[i]
                ne = np.linalg.norm(e) + 1e-12
                if abs(Tn[i] @ Tn[j]) < 0.5 or abs((e / ne) @ Tn[i]) < 0.5:
                    continue
            rows.append(i); cols.append(j); vals.append(dist[i, jj])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def _fit_spline(Y_ordered, grid, closed):
    """Fit a smooth curve along arc length of the ordered points."""
    C = Y_ordered
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if closed:
        d = np.r_[d, d[-1] + np.linalg.norm(C[0] - C[-1])]
        C = np.vstack([C, C[:1]])
    t = d / max(d[-1], 1e-12)
    by = np.ones(len(t))
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(t), torch.from_numpy(C), basis="duchon", basis_order=2,
        by=torch.from_numpy(by), periodic=bool(closed),
        period=(1.0 if closed else None))
    fitted = np.asarray(res["fitted"])
    order = np.argsort(t)
    ts, fs = t[order], fitted[order]
    if closed:
        ts = np.concatenate([ts - 1, ts, ts + 1]); fs = np.concatenate([fs, fs, fs], 0)
    return np.stack([np.interp(grid, ts, fs[:, j]) for j in range(C.shape[1])], 1)


def recover_curve(Yk, grid, k=10):
    if len(Yk) < k + 2:
        return None
    order, closed = _order_mst(Yk, k)
    return _fit_spline(Yk[order], grid, closed)


def k_subspaces(X, K, dim=2, n_iter=40, seed=0, restarts=10):
    best, best_err = None, np.inf
    D = X.shape[1]
    for rs in range(restarts):
        rng = np.random.default_rng(seed + rs)
        B = [np.linalg.qr(rng.standard_normal((D, dim)))[0] for _ in range(K)]
        res = None
        for _ in range(n_iter):
            res = np.stack([((X - X @ (b @ b.T)) ** 2).sum(1) for b in B], 1)
            asg = res.argmin(1)
            for k in range(K):
                Xk = X[asg == k]
                if len(Xk) < dim + 1:
                    continue
                _, _, Vt = np.linalg.svd(Xk - Xk.mean(0), full_matrices=False)
                B[k] = Vt[:dim].T
        err = res.min(1).sum()
        if err < best_err:
            best_err, best = err, ([b.copy() for b in B], res.argmin(1))
    return best


def recover_manifolds(X, K, dim=2, grid_size=200, seed=0, k=10):
    B, asg = k_subspaces(X, K, dim=dim, seed=seed)
    grid = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    curves = []
    for j in range(K):
        Yk = X[asg == j] @ B[j]
        c = recover_curve(Yk, grid, k=k)
        curves.append(None if c is None else c @ B[j].T)
    return curves, asg, B
