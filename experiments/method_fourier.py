"""Ordering-free harmonic / Fourier recovery of a union of K 1D curves living in
K mutually-orthogonal 2D subspaces of R^D.

NO point clustering (k-means / K-subspaces / mixture-EM), NO k-NN graphs, NO
trained encoder.  Only global moment / spectral / harmonic statistics are used.

Pipeline
--------
1. SUBSPACE RECOVERY by moments.  The data covariance is block-diagonal in the
   orthogonal 2D blocks, so its top 2K eigenvectors span the union of the K
   planes (one shared isotropic noise floor fixes the count 2K).  Every signal
   eigenvector turns out 100%-pure to a single block, so we only need to *pair*
   them.  We pair via a 4th-order co-activation statistic A[a,b]=E[|y_a||y_b|]
   on the eigenvector coordinates: the two axes of the SAME curve are
   simultaneously active on that curve's points, while a cross-block pair has
   one coordinate ~0 (a point active in block i is pure noise in block j).  A
   greedy max-weight perfect matching on A recovers the K planes -- no point
   clustering.  Each plane is then polished as the top-2 PCA of the points that
   stick out of the central noise blob (those points are pure to one block, so
   their span is exactly that block).

2. WITHIN-PLANE FIT (ordering-free).  Projected onto a plane, this curve's
   points spread out while every OTHER curve collapses to a tight Gaussian noise
   blob at the origin (radius ~ sigma).  A global radius threshold isolates this
   curve's points (a threshold on a global statistic -- not a cluster
   assignment).  We then estimate a curve PARAMETER t per point without ordering
   them: a Fourier curve is fit to the unordered cloud by the principal-curve
   fixed point -- alternately (a) give each point the parameter of its nearest
   point on the current smooth curve (a geometric projection, no neighbour
   graph) and (b) refit the Fourier coefficients by least squares.  Closed,
   open and multiply-winding (spiral) curves are auto-detected from the fit
   residual + coverage geometry, the spiral being initialised by a polar unwrap
   about a moment-estimated centre.

3. FINAL SMOOTH via gamfit REML.  Given the recovered t, each coordinate is
   smoothed by ``gamfit.gaussian_reml_fit_positions`` with a Duchon (m=2) basis,
   periodic for closed curves -- REML auto-selects the smoothness.  A couple of
   project-then-REML rounds polish the parameterisation.  The result is lifted
   back to R^D through the recovered orthonormal plane basis.
"""
from __future__ import annotations

import itertools
import numpy as np

try:
    import gamfit
    _HAVE_GAMFIT = True
except Exception:                       # pragma: no cover
    _HAVE_GAMFIT = False


# --------------------------------------------------------------------------- #
# 1. subspace recovery
# --------------------------------------------------------------------------- #
def _cumulant_matrices(Yw):
    """4th-order cumulant slice matrices Q_kl of whitened data (the JADE set)."""
    N, m = Yw.shape
    cov = (Yw.T @ Yw) / N                       # ~ I after whitening
    mats = []
    for k in range(m):
        for l in range(k, m):
            wkl = Yw[:, k] * Yw[:, l]
            M = (wkl[:, None, None] * Yw[:, :, None] * Yw[:, None, :]).mean(0)
            M -= cov * (k == l)
            M -= np.outer(cov[k], cov[l]) + np.outer(cov[l], cov[k])
            mats.append(M)
    return mats


def _joint_diagonalize(mats, m, sweeps=40, tol=1e-9):
    """Cardoso--Souloumiac Jacobi joint diagonalizer: orthogonal Q making the
    cumulant matrices as diagonal as possible.  Resolves blocks even when the
    covariance is degenerate (curves with near-equal variance)."""
    mats = [M.copy() for M in mats]
    Q = np.eye(m)
    for _ in range(sweeps):
        moved = 0.0
        for p in range(m):
            for q in range(p + 1, m):
                g = np.array([[M[p, p] - M[q, q], M[p, q] + M[q, p]] for M in mats])
                G = g.T @ g
                ev, Vg = np.linalg.eigh(G)
                x, y = Vg[:, -1]
                if x < 0:
                    x, y = -x, -y
                theta = 0.5 * np.arctan2(y, x)
                if abs(theta) < tol:
                    continue
                moved += abs(theta)
                c, s = np.cos(theta), np.sin(theta)
                for M in mats:
                    cp, cq = M[:, p].copy(), M[:, q].copy()
                    M[:, p] = c * cp + s * cq
                    M[:, q] = -s * cp + c * cq
                    rp, rq = M[p, :].copy(), M[q, :].copy()
                    M[p, :] = c * rp + s * rq
                    M[q, :] = -s * rp + c * rq
                Qp, Qq = Q[:, p].copy(), Q[:, q].copy()
                Q[:, p] = c * Qp + s * Qq
                Q[:, q] = -s * Qp + c * Qq
        if moved < tol:
            break
    return Q


def _recover_planes(X, K):
    """Return list of K (D,2) orthonormal plane bases + noise sigma estimate.

    Top-2K eigenvectors of the covariance span the union of the K planes.  We
    whiten that subspace and JADE-rotate it to (near-)independent axes, then pair
    the two axes of each curve by co-activation E[|y_a||y_b|] (high within a
    block, ~0 across blocks because a point is active in exactly one block)."""
    N, D = X.shape
    Xc = X - X.mean(0)
    C = (Xc.T @ Xc) / N
    w, V = np.linalg.eigh(C)
    w = w[::-1]
    V = V[:, ::-1]
    m = 2 * K
    U = V[:, :m]
    sigma = np.sqrt(max(w[m:].mean(), 1e-12)) if m < D else np.sqrt(max(w[-1], 1e-12))

    Y = Xc @ U
    cov = (Y.T @ Y) / N
    ew, eV = np.linalg.eigh(cov)
    Wh = eV @ np.diag(1.0 / np.sqrt(np.maximum(ew, 1e-12))) @ eV.T
    Yw = Y @ Wh                                 # whitened signal coords
    Q = _joint_diagonalize(_cumulant_matrices(Yw), m)
    Yf = Yw @ Q                                 # ~independent source coords
    T = U @ Wh @ Q                              # ambient directions

    Ya = np.abs(Yf)
    A = (Ya.T @ Ya) / N                         # co-activation; high within a block
    np.fill_diagonal(A, 0.0)

    used, pairs = set(), []
    for i, j in sorted(itertools.combinations(range(m), 2), key=lambda p: -A[p]):
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.update((i, j))
        if len(pairs) == K:
            break

    planes = []
    for (i, j) in pairs:
        P, _ = np.linalg.qr(T[:, [i, j]])
        planes.append(P)
    return planes, float(sigma)


def _refine_plane(Xc, P, thr):
    """Re-estimate the plane from the curve's own (pure) points, dropping the
    central noise blob; tightens the plane and rejects leaked points."""
    r = np.linalg.norm(Xc @ P, axis=1)
    keep = r > thr
    if keep.sum() < 10:
        return P
    M = Xc[keep]
    Mc = M - M.mean(0)
    _, _, vt = np.linalg.svd(Mc, full_matrices=False)
    return vt[:2].T


# --------------------------------------------------------------------------- #
# 2. ordering-free parameter recovery (Fourier principal-curve)
# --------------------------------------------------------------------------- #
def _design(t, H, closed):
    t = np.asarray(t)
    if closed:
        cols = [np.ones_like(t)]
        for h in range(1, H + 1):
            cols += [np.cos(2 * np.pi * h * t), np.sin(2 * np.pi * h * t)]
    else:
        cols = [np.ones_like(t), t]
        for h in range(1, H + 1):
            cols += [np.cos(np.pi * h * t), np.sin(np.pi * h * t)]
    return np.stack(cols, 1)


def _fit_W(Z, t, H, closed, ridge=1e-6):
    Phi = _design(t, H, closed)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ Z)


def _project(Z, W, H, closed, G):
    grid = np.linspace(0, 1, G + 1)[:-1] if closed else np.linspace(0, 1, G)
    Pg = _design(grid, H, closed) @ W
    d2 = ((Z[:, None, :] - Pg[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    return grid[idx], d2.min(1)


def _refine(Z, t, H, closed, iters=40, G=600, ridge=1e-6):
    for _ in range(iters):
        W = _fit_W(Z, t, H, closed, ridge)
        t, _ = _project(Z, W, H, closed, G)
    W = _fit_W(Z, t, H, closed, ridge)
    _, dmin = _project(Z, W, H, closed, G)
    return W, t, float(dmin.mean())


def _curve_fourier(W, H, closed, G=200):
    grid = np.linspace(0, 1, G + 1)[:-1] if closed else np.linspace(0, 1, G)
    return _design(grid, H, closed) @ W


def _t_open(Z, H=6):
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    p = Zc @ vt[0]
    t0 = (p - p.min()) / (np.ptp(p) + 1e-12)
    return _refine(Z, t0, H, closed=False)


def _t_closed(Z, H=8):
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    t0 = (np.arctan2(Zc @ vt[1], Zc @ vt[0]) / (2 * np.pi)) % 1.0
    return _refine(Z, t0, H, closed=True)


def _spiral_center_turns(Z):
    """Estimate the polar centre + #turns: the centre at which the unwrapped
    angle (radius-ordered) is most linear is the curve's winding centre, and the
    span of that ramp is 2*pi*turns."""
    mu = Z.mean(0)
    best = None
    for cx in np.linspace(mu[0] - 0.5, mu[0] + 0.5, 13):
        for cy in np.linspace(mu[1] - 0.5, mu[1] + 0.5, 13):
            c = np.array([cx, cy])
            v = Z - c
            rad = np.linalg.norm(v, axis=1)
            ang = np.arctan2(v[:, 1], v[:, 0])
            aw = np.unwrap(ang[np.argsort(rad)])
            xq = np.linspace(0, 1, len(aw))
            A = np.c_[xq, np.ones_like(xq)]
            coef, *_ = np.linalg.lstsq(A, aw, rcond=None)
            resid = np.std(aw - A @ coef)
            if best is None or resid < best[0]:
                best = (resid, c, abs(aw[-1] - aw[0]) / (2 * np.pi))
    return best[1], best[2]


def _t_spiral(Z):
    """Init parameter for a winding (spiral) curve.  Radius is monotone in arc
    length for such curves, so the rank of the radius about the winding centre
    IS the arc-length order -- no angle unwrapping or point ordering needed.
    The REML project-refine step then polishes it."""
    c, _ = _spiral_center_turns(Z)
    rad = np.linalg.norm(Z - c, axis=1)
    order = np.argsort(rad)
    t = np.empty(len(rad))
    t[order] = np.arange(len(rad)) / (len(rad) - 1)
    return t


def _coverage_gap(Z, W, H, G=400):
    grid = np.linspace(0, 1, G + 1)[:-1]
    Pg = _design(grid, H, True) @ W
    d2 = ((Z[:, None, :] - Pg[None, :, :]) ** 2).sum(-1)
    near = np.sqrt(d2.min(0))
    seg = np.linalg.norm(np.diff(Pg, axis=0, append=Pg[:1]), axis=1)
    empty = near > (np.median(near) * 4 + 1e-9)
    return seg[empty].sum() / (seg.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# 3. final smooth through gamfit REML  (+ fallback)
# --------------------------------------------------------------------------- #
def _reml_fitted(t, Z, periodic):
    """Smooth each coordinate y=Z[:,j] over positions t with a Duchon-m2 REML
    GAM.  Returns the fitted (denoised) points (same order as input)."""
    out = []
    for j in range(2):
        res = gamfit.gaussian_reml_fit_positions(
            np.ascontiguousarray(t), np.ascontiguousarray(Z[:, j]),
            basis="duchon", basis_order=2,
            periodic=periodic, period=1.0 if periodic else None,
        )
        out.append(np.asarray(res["fitted"]).ravel())
    return np.stack(out, 1)


def _smooth_reml(Z, t, periodic, rounds=4):
    """Alternate (REML smooth) <-> (project points onto the smooth curve) to
    polish the parameterisation, then return an ordered (n,2) polyline."""
    t = (t - t.min()) / (np.ptp(t) + 1e-12)
    F = _reml_fitted(t, Z, periodic)
    for _ in range(rounds):
        o = np.argsort(t)
        dense, td = F[o], t[o]
        d2 = ((Z[:, None, :] - dense[None, :, :]) ** 2).sum(-1)
        t = td[d2.argmin(1)]
        t = (t - t.min()) / (np.ptp(t) + 1e-12)
        F = _reml_fitted(t, Z, periodic)
    o = np.argsort(t)
    return F[o]


def _smooth_spiral(Z, t, rounds=8, Gd=1200, window=0.08):
    """REML smoother for a winding curve.  Projection is done onto a densely
    resampled curve and constrained to a local arc-length window so points
    cannot jump between adjacent turns of the spiral."""
    t = (t - t.min()) / (np.ptp(t) + 1e-12)
    F = _reml_fitted(t, Z, periodic=False)
    for _ in range(rounds):
        o = np.argsort(t)
        d0 = F[o]
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(d0, axis=0), axis=1))]
        d /= d[-1] + 1e-12
        s = np.linspace(0, 1, Gd)
        dense = np.stack([np.interp(s, d, d0[:, j]) for j in range(2)], 1)
        d2 = ((Z[:, None, :] - dense[None, :, :]) ** 2).sum(-1)
        gi = np.clip((t * (Gd - 1)).astype(int), 0, Gd - 1)
        win = max(int(window * Gd), 1)
        tnew = t.copy()
        for n in range(len(t)):
            a = max(0, gi[n] - win)
            b = min(Gd, gi[n] + win + 1)
            tnew[n] = s[a + d2[n, a:b].argmin()]
        t = (tnew - tnew.min()) / (np.ptp(tnew) + 1e-12)
        F = _reml_fitted(t, Z, periodic=False)
    return F[np.argsort(t)]


def _resample(curve, G, closed):
    """Arc-length resample an ordered polyline to G points."""
    C = np.vstack([curve, curve[:1]]) if closed else curve
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return np.repeat(curve[:1], G, 0)
    s = np.linspace(0, d[-1], G, endpoint=not closed)
    return np.stack([np.interp(s, d, C[:, j]) for j in range(C.shape[1])], 1)


def _cloud_resid(Z, poly, closed):
    dense = _resample(poly, 400, closed)
    return ((Z[:, None, :] - dense[None, :, :]) ** 2).sum(-1).min(1).mean()


def _fit_plane_curve(Z, sigma, G=200):
    """Recover the 2D curve (G,2) from the unordered plane cloud Z.

    Topology (open / closed / spiral) is chosen by comparing the point-to-curve
    residual of an open vs a closed REML fit: a closed fit of an open curve adds
    a spurious 'bridge' (large residual) and vice-versa, while a spiral defeats
    both basic parameterisations and leaves a residual far above the noise floor.
    """
    Wc, tc, rc = _t_closed(Z, H=8)
    Wo, to, ro = _t_open(Z, H=6)
    noise = sigma ** 2

    if _HAVE_GAMFIT:
        polyC = _smooth_reml(Z, tc, periodic=True)
        polyO = _smooth_reml(Z, to, periodic=False)
        resC = _cloud_resid(Z, polyC, True)
        resO = _cloud_resid(Z, polyO, False)

        # spiral: even the better basic fit is far from the data
        if min(resC, resO) > 5.0 * noise:
            ts = _t_spiral(Z)
            poly = _smooth_spiral(Z, ts)
            return _resample(poly, G, closed=False)

        # closed only if it fits clearly better than open (margin guards
        # near-closed open curves like the S-curve / semicircle)
        if resC < 0.4 * resO:
            return _resample(polyC, G, closed=True)
        return _resample(polyO, G, closed=False)

    # ---- fallback (no gamfit): Fourier residuals ----
    if min(rc, ro) > 6.0 * noise:
        Ws, ts2, rs = _refine(Z, _t_spiral(Z), H=12, closed=False, iters=60, G=1500)
        return _resample(_curve_fourier(Ws, 12, False), G, False)
    gap = _coverage_gap(Z, Wc, 8)
    if gap < 0.25 and rc <= ro * 1.5:
        return _resample(_curve_fourier(Wc, 8, True), G, True)
    return _resample(_curve_fourier(Wo, 6, False), G, False)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def recover(X, K):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    planes, sigma = _recover_planes(X, K)
    thr = 6.0 * sigma * np.sqrt(2)          # noise-blob radius cutoff in a plane

    curves = []
    for P in planes:
        P = _refine_plane(Xc, P, thr)        # polish plane from pure points
        Z = Xc @ P
        r = np.linalg.norm(Z, axis=1)
        keep = r > thr
        if keep.sum() < 20:
            keep = r > np.quantile(r, 0.5)
        Zk = Z[keep]
        crv2d = _fit_plane_curve(Zk, sigma)
        curves.append(crv2d @ P.T)
    return curves
