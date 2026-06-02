"""Density-ridge principal-curve recovery via Subspace-Constrained Mean Shift.

The K curves are the 1D ridges of the data's kernel-density estimate. We build a
single global-bandwidth Gaussian KDE over ALL points (a dense kernel, not a kNN
graph) and move probe points to the ridge by SCMS: at each probe we form the
KDE gradient and Hessian, take the trailing (smallest-curvature) Hessian
eigenvector as the ridge tangent, and project the mean-shift step onto the
orthogonal complement so probes flow up onto the 1D ridge. To split/order the
ridge points without clustering or kNN we exploit that the K curves live in
MUTUALLY ORTHOGONAL 2D subspaces: we recover the K 2D blocks from the data's
covariance eigen-structure, assign each ridge point to the block with the
smallest projection residual, and order each curve as a polyline by its angle
(closed) or projected arc-coordinate (open) within its own 2D plane.

Exposes ``recover(X, K) -> list of K (G, D) arrays``.  No k-means/K-subspaces,
no kNN graph, no trained encoder.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Subspace recovery: find the K orthogonal 2D blocks                          #
# --------------------------------------------------------------------------- #
def recover_subspaces(X, K, seed=0):
    """Recover K mutually-orthogonal 2D subspaces spanning the data by peeling.

    The top-2K covariance eigenvectors span the union U of the K planes. Within
    U every token lies (up to noise) in exactly ONE of the K orthogonal 2D
    coordinate planes, so we recover the planes one at a time: seed a direction,
    grow the 2D plane that best contains the points aligned with it (top-2 PCA of
    those points, alternating until the plane stabilizes), claim its member
    points, and repeat on the remainder. This needs no clustering of the raw
    data — it is pure spectral structure of the orthogonal-block covariance.
    """
    Xc = X - X.mean(0)
    N = len(Xc)
    C = Xc.T @ Xc / N
    w, V = np.linalg.eigh(C)
    U = V[:, ::-1][:, : 2 * K]            # (D, 2K) union basis
    Y = Xc @ U                            # (N, 2K)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    Yn = Y / (nrm + 1e-12)                # unit directions in the union

    remaining = np.ones(N, bool)
    blocks = []
    for _ in range(K):
        idx = np.where(remaining)[0]
        Yr = Y[idx]
        # seed: highest-norm remaining point direction
        seed_vec = Yn[idx][np.argmax(np.linalg.norm(Yr, axis=1))]
        B2 = None
        for it in range(10):
            if B2 is None:
                align = np.abs(Yn[idx] @ seed_vec)
            else:
                align = np.linalg.norm(Yn[idx] @ B2, axis=1)
            thr = 0.85 if B2 is not None else 0.7
            sel = align > thr
            if sel.sum() < 6:
                sel = align >= np.quantile(align, 0.9)
            P = Yr[sel]
            cc = P.T @ P
            _, vv = np.linalg.eigh(cc)
            B2 = vv[:, -2:]               # (2K, 2) plane in union coords
        # claim members (this plane contains them well)
        proj = np.linalg.norm(Yn @ B2, axis=1)
        member = (proj > 0.85) & remaining
        if member.sum() < 3:             # fallback: nearest unclaimed
            member = (proj > 0.6) & remaining
        remaining = remaining & ~member
        blocks.append(U @ B2)            # (D, 2) ambient orthonormal plane
    return blocks, None


# --------------------------------------------------------------------------- #
#  SCMS: move probe points to the 1D density ridge                            #
# --------------------------------------------------------------------------- #
def scms(X, probes, h, n_steps=60, tol=1e-5, batch=256):
    """Subspace-Constrained Mean Shift toward the 1D density ridge.

    Global-bandwidth Gaussian KDE over all of X. At each probe we compute the
    mean-shift vector m and the local Hessian of log-density; the ridge tangent
    is the top Hessian eigenvector (largest eigenvalue, smallest |curvature|),
    and we project the step onto its orthogonal complement.
    """
    X = np.asarray(X)
    P = probes.copy()
    h2 = h * h
    for _ in range(n_steps):
        newP = P.copy()
        for s in range(0, len(P), batch):
            p = P[s:s + batch]                       # (b, D)
            diff = X[None, :, :] - p[:, None, :]     # (b, N, D)
            d2 = np.einsum("bnd,bnd->bn", diff, diff)
            wgt = np.exp(-0.5 * d2 / h2)             # (b, N)
            Wsum = wgt.sum(1) + 1e-300
            # mean shift  m = sum w*(x-p)/sum w
            m = np.einsum("bn,bnd->bd", wgt, diff) / Wsum[:, None]
            # local covariance of (x-p) weighted -> Hessian of density ~ outer
            # H ∝ (1/h2)[ sum w (x-p)(x-p)^T / h2 - I*sum w ] ; we use the
            # weighted covariance about the mean-shift target for the ridge frame.
            mu = m                                    # weighted mean of (x-p)
            cen = diff - mu[:, None, :]               # (b, N, D)
            Cov = np.einsum("bn,bni,bnj->bij", wgt, cen, cen) / Wsum[:, None, None]
            # Hessian of log-density (Gaussian KDE): H = (1/h2)(Cov/h2 - I)
            Dn = X.shape[1]
            Hess = Cov / h2 - np.eye(Dn)[None]        # up to positive scale 1/h2
            # ridge tangent = eigenvector of Hess with the LARGEST eigenvalue
            evals, evecs = np.linalg.eigh(Hess)       # ascending
            tang = evecs[:, :, -1]                     # (b, D) largest eval
            # project mean shift onto complement of tangent
            proj = np.einsum("bd,bd->b", m, tang)
            step = m - proj[:, None] * tang
            newP[s:s + batch] = p + step
        shift = np.linalg.norm(newP - P, axis=1).max()
        P = newP
        if shift < tol:
            break
    return P


# --------------------------------------------------------------------------- #
#  Ordering a 2D plane's ridge points into an open/closed polyline            #
# --------------------------------------------------------------------------- #
def order_curve_2d(pts2d, G=200):
    """Order 2D ridge points into a polyline; detect open vs closed.

    The points lie on one smooth 1D curve in 2D (any shape: open, closed, or
    self-approaching like a spiral). We order them by a greedy nearest-point walk
    (polyline stitching of points already known to be on ONE curve — a local
    reconstruction step, not a manifold-discovery kNN graph) and decide open vs
    closed from whether the two ends are close relative to the polyline length.
    """
    P = _greedy_chain(pts2d)
    # Decide open vs closed BEFORE trimming, from the raw chain geometry.
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total = seglen.sum() + 1e-12
    typical = np.median(seglen) + 1e-12
    end_gap = np.linalg.norm(P[0] - P[-1])
    closed = end_gap < 3.0 * typical and end_gap < 0.25 * total
    if not closed:
        # Trim stranded endpoints: a greedy walk on a sharply-turning or
        # self-approaching open curve can append a few points reached only by an
        # anomalously long final/first jump. Peel such jumps off the ends.
        P = _trim_ends(P, typical)
    return P, closed


def _trim_ends(P, typical, factor=4.0, max_strand=4):
    """Drop a short stranded run at either end of the chain. A greedy walk on a
    sharply-turning/self-approaching open curve sometimes appends a few points
    reachable only via an anomalously long jump. If a long jump sits within
    `max_strand` points of an end, cut the smaller piece off."""
    n = len(P)
    if n < 6:
        return P
    s = np.linalg.norm(np.diff(P, axis=0), axis=1)
    while len(P) >= 6:
        s = np.linalg.norm(np.diff(P, axis=0), axis=1)
        j = int(np.argmax(s))
        if s[j] < factor * typical:
            break
        tail = len(P) - 1 - j          # points after the jump (excl. P[j])
        head = j + 1                   # points up to and incl. P[j]
        if tail <= max_strand and tail < head:
            P = P[: j + 1]
        elif head <= max_strand and head < tail:
            P = P[j + 1:]
        else:
            break
    return P


def _greedy_chain(pts):
    """Greedy nearest-neighbour chaining starting from an extreme endpoint.

    Start at the point farthest from the centroid (a likely endpoint / extreme),
    repeatedly hop to the nearest unvisited point. Then refine by also growing
    from the other side, so a mid-curve start cannot strand half the curve.
    """
    n = len(pts)
    if n <= 2:
        return pts.copy()
    c = pts.mean(0)

    def walk(seed):
        visited = np.zeros(n, bool)
        order = [seed]
        visited[seed] = True
        cur = seed
        prev_dir = None
        for _ in range(n - 1):
            vec = pts - pts[cur]
            d = np.linalg.norm(vec, axis=1)
            d[visited] = np.inf
            if prev_dir is not None:
                # direction-aware cost: prefer continuing roughly straight so the
                # walk does not jump between nearby arms of a spiral. Cost =
                # distance * (1 + turn-penalty for reversing direction).
                with np.errstate(invalid="ignore"):
                    dirs = vec / (d[:, None] + 1e-12)
                cos = dirs @ prev_dir          # +1 straight ahead, -1 backward
                cost = d * (1.0 + 1.5 * (1.0 - cos))
            else:
                cost = d
            cost[visited] = np.inf
            nxt = int(np.argmin(cost))
            step = pts[nxt] - pts[cur]
            prev_dir = step / (np.linalg.norm(step) + 1e-12)
            visited[nxt] = True
            order.append(nxt)
            cur = nxt
        return order

    start = int(np.argmax(np.linalg.norm(pts - c, axis=1)))
    return pts[walk(start)]


# --------------------------------------------------------------------------- #
#  Main entry point                                                           #
# --------------------------------------------------------------------------- #
def recover(X, K, n_probe=900, G=200, seed=0):
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    Xc = X - X.mean(0)

    # 1) recover the K orthogonal 2D subspaces
    blocks, _ = recover_subspaces(Xc, K, seed=seed)

    # 2) global KDE bandwidth (single scalar). Scott-ish but tuned small since
    #    the data is 1D-on-curves; pick from intrinsic scale.
    scale = np.sqrt((Xc ** 2).sum(1).mean())
    h = 0.10 * scale

    # 3) probe points = random subset of data, pushed to the ridge by SCMS
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(N, size=min(n_probe, N), replace=False)
    probes = X[idx].copy()
    ridge = scms(X, probes, h, n_steps=80)
    ridge_c = ridge - X.mean(0)

    # 4) assign each ridge point to the block with smallest projection residual
    curves = []
    energy = np.einsum("nd,nd->n", ridge_c, ridge_c) + 1e-12
    proj_res = np.empty((len(ridge_c), K))
    coords = []
    for k, B in enumerate(blocks):
        co = ridge_c @ B                  # (n, 2)
        rec = co @ B.T
        res = np.einsum("nd,nd->n", ridge_c - rec, ridge_c - rec)
        proj_res[:, k] = res / energy
        coords.append(co)
    assign = proj_res.argmin(1)

    mean = X.mean(0)
    for k, B in enumerate(blocks):
        mask = assign == k
        if mask.sum() < 5:
            curves.append(None)
            continue
        p2 = coords[k][mask]
        P2, closed = order_curve_2d(p2, G=G)
        # final smooth fit: gamfit REML auto-selects the smoothness (periodic
        # Duchon for closed curves), fitting each plane-coordinate vs arc-length.
        poly = _gam_fit_curve(P2, closed, G=G)
        if poly is None:                      # fallback if the GAM solve fails
            poly = _resample(_smooth(P2, closed), G, closed)
        curve = poly @ B.T + mean
        curves.append(curve)
    return curves


def _gam_fit_curve(P, closed, G=200):
    """Smooth the ordered 2D polyline with gamfit's closed-form Gaussian REML.

    Parameterize by normalized arc length t in [0,1); fit each of the two plane
    coordinates as a smooth function of t with a Duchon (thin-plate, m=2) basis,
    letting REML auto-select the smoothing penalty (periodic Duchon for closed
    curves). The evaluation grid is appended as near-zero-weight rows so its
    ``fitted`` values are read straight from the same periodic-aware fit (no need
    to reconstruct the periodic design matrix). Returns the (G, 2) grid curve."""
    import gamfit

    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n < 8:
        return None
    if closed:
        Pc = np.vstack([P, P[:1]])
        seg = np.linalg.norm(np.diff(Pc, axis=0), axis=1)
    else:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = s[-1]
    if L < 1e-9:
        return None
    t = s[:n] / L                            # (n,) in [0,1)
    grid = np.linspace(0.0, 1.0, G, endpoint=closed)
    tt = np.concatenate([t, grid])
    w = np.concatenate([np.ones(n), np.full(G, 1e-8)])
    # Enough basis functions to resolve high-curvature shapes (e.g. a 2-turn
    # spiral); REML still picks the smoothing penalty, so simpler curves stay
    # smooth via a larger lambda rather than overfitting the extra knots.
    nk = int(min(max(n // 4, 12), 25))
    try:
        out = np.empty((G, 2))
        for j in range(2):
            yy = np.concatenate([P[:, j], np.zeros(G)])
            res = gamfit.gaussian_reml_fit_positions(
                tt, yy, basis="duchon", basis_order=2, knots_or_centers=nk,
                periodic=bool(closed), period=1.0 if closed else None, weights=w)
            out[:, j] = np.asarray(res["fitted"]).ravel()[n:]
        return out
    except Exception:
        return None


def _smooth(P, closed, win=5):
    """Moving-average smoothing along the ordered polyline to suppress the
    residual noise of the ridge points (the points are ordered, so this is a 1D
    low-pass along arc-order)."""
    n = len(P)
    if n < win + 2:
        return P
    if closed:
        ext = np.vstack([P[-win:], P, P[:win]])
        k = np.ones(2 * win + 1) / (2 * win + 1)
        sm = np.stack([np.convolve(ext[:, j], k, "same") for j in range(P.shape[1])], 1)
        return sm[win:win + n]
    else:
        k = np.ones(win) / win
        pad = win // 2
        ext = np.vstack([np.repeat(P[:1], pad, 0), P, np.repeat(P[-1:], pad, 0)])
        sm = np.stack([np.convolve(ext[:, j], k, "same") for j in range(P.shape[1])], 1)
        return sm[pad:pad + n]


def _resample(P, G, closed):
    if closed:
        P = np.vstack([P, P[:1]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return np.repeat(P[:1], G, 0)
    u = np.linspace(0, d[-1], G, endpoint=not closed)
    return np.stack([np.interp(u, d, P[:, j]) for j in range(P.shape[1])], 1)
