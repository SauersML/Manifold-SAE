"""Algebraic / GPCA recovery of a union of K planar curves in orthogonal 2D blocks.

Pipeline (NO clustering, NO k-NN graph, NO trained encoder):

1.  PCA of the covariance C = X^T X.  Because the K curves live in mutually
    ORTHOGONAL 2D subspaces, every eigenvector of C lies (to noise) inside a
    single 2D block.  The top 2K eigenvectors are 2K block-aligned directions.

2.  Pair the 2K eigen-directions into K blocks with a 4th-moment co-activation
    matrix  M[a,b] = E[(v_a.x)^2 (v_b.x)^2]  (variance-normalised).  Two
    directions in the same block co-activate; cross-block pairs do not.  This is
    a pure algebraic/moment separation (a max-weight perfect matching), not
    clustering of points.

    Plain covariance eigenvectors are block-pure ONLY when the 2K covariance
    eigenvalues are distinct; with near-equal eigenvalues across blocks they
    rotate together and mix.  To be robust we whiten X to its 2K-dim signal
    subspace and JOINTLY DIAGONALISE a family of reweighted covariances
    E[g(y) yy^T] (Cardoso Jacobi).  The block 2-planes are a common invariant
    subspace of every such matrix, so joint diagonalisation recovers a
    block-aligned basis even under 2nd-moment degeneracy.

3.  Each point is assigned to the block capturing >99% of its energy by simple
    projection (deterministic, model-based -- not k-means / soft assignment),
    giving ~N/K noisy 2D points per curve.

4.  Within each 2D block recover the 1D curve x(t), y(t).  A global
    parameterisation (angle for closed, principal-axis projection for open,
    unwrapped-angle for spirals) is refined by projecting points onto the
    current curve, and the coordinates are smoothed with gamfit's closed-form
    Gaussian REML on a (periodic) Duchon basis -- REML auto-selects smoothness,
    so there is no hand-tuned bandwidth.  Re-embed the (G,2) polyline into R^D.
"""
from __future__ import annotations

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
#  block recovery (algebraic)                                                  #
# --------------------------------------------------------------------------- #
def _joint_diag(mats, n_sweeps=40, tol=1e-10):
    """Cardoso-style Jacobi joint diagonalisation of symmetric matrices.

    Returns an orthogonal V approximately diagonalising every matrix in ``mats``
    simultaneously.  The common (block) eigenstructure is recovered even when any
    single matrix has repeated eigenvalues."""
    n = mats[0].shape[0]
    V = np.eye(n)
    A = [m.copy() for m in mats]
    for _ in range(n_sweeps):
        moved = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                g = np.array([[A[k][p, p] - A[k][q, q],
                               A[k][p, q] + A[k][q, p]] for k in range(len(A))])
                G = g.T @ g
                ton = G[0, 0] - G[1, 1]
                toff = G[0, 1] + G[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton ** 2 + toff ** 2))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) > 1e-14:
                    moved += abs(s)
                    for k in range(len(A)):
                        cp, cq = A[k][:, p].copy(), A[k][:, q].copy()
                        A[k][:, p] = c * cp + s * cq
                        A[k][:, q] = -s * cp + c * cq
                        rp, rq = A[k][p, :].copy(), A[k][q, :].copy()
                        A[k][p, :] = c * rp + s * rq
                        A[k][q, :] = -s * rp + c * rq
                    vp, vq = V[:, p].copy(), V[:, q].copy()
                    V[:, p] = c * vp + s * vq
                    V[:, q] = -s * vp + c * vq
        if moved < tol:
            break
    return V


def _recover_blocks(X, K):
    """Return list of K (2, D) orthonormal frames, one per 2D block."""
    d = 2 * K
    C = X.T @ X / len(X)
    w, U = np.linalg.eigh(C)
    order = np.argsort(w)[::-1][:d]
    w, U = w[order], U[:, order]                  # top-d signal subspace
    Wm = U / np.sqrt(np.maximum(w, 1e-12))        # (D, d) whitening map
    Y = X @ Wm                                     # (N, d) whitened coords

    # family of reweighted covariances E[g(y) y y^T]; the K block 2-planes are a
    # common invariant subspace of every one of these (4th-moment structure).
    mats = []
    r2 = (Y ** 2).sum(1)
    for p in (1, 2, 3):
        gw = r2 ** p
        mats.append((Y * gw[:, None]).T @ Y / len(Y))
    for a in range(0, d, 2):                       # per-coordinate slices
        gw = Y[:, a] ** 2
        mats.append((Y * gw[:, None]).T @ Y / len(Y))

    R = _joint_diag(mats)                          # (d, d) orthogonal, block-aligned
    B = Y @ R                                       # (N, d) coords in block basis

    # pair the d axes into K blocks with the 4th-moment co-activation matrix
    B2 = B ** 2
    B2 = B2 / (B2.mean(0) + 1e-12)
    Mco = (B2.T @ B2) / len(B)
    np.fill_diagonal(Mco, -np.inf)

    avail = list(range(d))
    pairs = []
    while avail:
        a = avail[0]
        b = max((j for j in avail if j != a), key=lambda j: Mco[a, j])
        pairs.append((a, b))
        avail.remove(a)
        avail.remove(b)

    # direction in ambient space = whitening map composed with the rotation,
    # then orthonormalise each 2-plane (whitening is not orthogonal).
    Dir = Wm @ R                                    # (D, d) columns = block dirs
    frames = []
    for a, b in pairs:
        Q, _ = np.linalg.qr(Dir[:, [a, b]])
        frames.append(Q[:, :2].T)
    return frames


# --------------------------------------------------------------------------- #
#  within-block 1D curve recovery (principal curve)                            #
#                                                                              #
#  The curve x(t), y(t) is smoothed with gamfit's closed-form Gaussian REML on #
#  a Duchon (thin-plate) basis -- REML auto-selects the smoothing parameter,   #
#  so there is no hand-tuned bandwidth.  For closed curves we use the periodic #
#  Duchon basis.  The model is evaluated on the query grid ``tq`` by appending #
#  the grid as zero-weight rows and reading their fitted values.               #
# --------------------------------------------------------------------------- #
def _reml_curve(t, P, tq, periodic):
    """Fit each coordinate of P as a REML smooth of the parameter t, evaluated
    on the grid tq.  Returns a (len(tq), P.shape[1]) polyline."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    tq = np.asarray(tq, float)
    n, m = len(t), len(tq)
    t_all = np.r_[t, tq]
    w = np.r_[np.ones(n), np.zeros(m)]          # grid rows do not affect the fit
    out = np.empty((m, P.shape[1]))
    for j in range(P.shape[1]):
        y_all = np.r_[P[:, j], np.zeros(m)]
        try:
            res = gamfit.gaussian_reml_fit_positions(
                t_all, y_all, basis="duchon", basis_order=2,
                periodic=periodic, period=1.0 if periodic else None, weights=w)
            out[:, j] = np.asarray(res["fitted"]).ravel()[n:]
        except Exception:
            # rare numerically-degenerate parameterisation: fall back to a
            # weighted local mean so the candidate can still be scored.
            out[:, j] = _local_mean(t, P[:, j], tq, periodic)
    return out


def _local_mean(t, y, tq, periodic, bw=0.05):
    out = np.empty(len(tq))
    for k, t0 in enumerate(tq):
        d = np.abs(t - t0)
        if periodic:
            d = np.minimum(d, 1 - d)
        wk = np.exp(-0.5 * (d / bw) ** 2)
        out[k] = (wk * y).sum() / (wk.sum() + 1e-12)
    return out


def _smooth_periodic(t, P, tq, bw=None):
    return _reml_curve(t, P, tq, periodic=True)


def _smooth_open(t, P, tq, bw=None):
    return _reml_curve(t, P, tq, periodic=False)


def _project_param(P, curve, closed):
    """Assign each point the arc-length parameter of its nearest curve vertex."""
    # nearest vertex (brute force projection onto the 1D model, O(N*G))
    d2 = ((P[:, None, :] - curve[None, :, :]) ** 2).sum(2)
    idx = d2.argmin(1)
    G = len(curve)
    seg = np.vstack([curve, curve[:1]]) if closed else curve
    dl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    s = dl[:G] if not closed else dl[:G]
    t = s[idx] / (dl[-1] + 1e-12)
    return t


def _ang_gap(P):
    """Largest angular gap (radians) seen from the centroid."""
    Q = P - P.mean(0)
    ang = np.sort(np.arctan2(Q[:, 1], Q[:, 0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return gaps.max()


def _candidate_inits(P):
    """Yield (name, closed, t) global parameterisations to try."""
    Q = P - P.mean(0)
    r = np.linalg.norm(Q, axis=1)
    ang = np.arctan2(Q[:, 1], Q[:, 0])
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    full_wrap = _ang_gap(P) < 1.1

    cands = []
    # 1. angle (good for closed star-shaped: ellipse, circle, cardioid)
    cands.append(("angle", True, (ang + np.pi) / (2 * np.pi)))
    # 2. principal-axis projection (good for open single-valued: S, parabola,
    #    semicircle, wave)
    a = Q @ Vt[0]
    cands.append(("axis", False, (a - a.min()) / (np.ptp(a) + 1e-12)))
    # 3. unwrapped-angle / spiral inits.  A spiral has radius monotone in
    #    arc-length, so r linearly predicts the *unwrapped* angle.  We don't
    #    know how many turns, so emit one init per turn-count hypothesis.
    for turns in (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
        phi_guess = (r - r.min()) / (np.ptp(r) + 1e-12) * turns * 2 * np.pi
        n = np.round((phi_guess - ang) / (2 * np.pi))
        phi = ang + 2 * np.pi * n
        cands.append((f"spiral{turns}", False,
                      (phi - phi.min()) / (np.ptp(phi) + 1e-12)))
    # 3b. parameter-free spiral init: order by radius, unwrap the angle along
    #     that order (no turn-count guess needed).
    o = np.argsort(r)
    phi = np.unwrap(ang[o])
    t_uw = np.empty(len(P))
    t_uw[o] = (phi - phi.min()) / (np.ptp(phi) + 1e-12)
    cands.append(("spiralUW", False, t_uw))
    return cands, full_wrap


def _refine(P, t, closed, G, n_iter=3):
    """Alternate (REML smooth of x(t), y(t))  <->  (re-project points onto the
    curve to update t).  REML auto-selects the smoothness, so no bandwidth
    schedule is needed; a few projection sweeps suffice to settle the ordering."""
    tq = np.linspace(0, 1, G, endpoint=not closed)
    smooth = _smooth_periodic if closed else _smooth_open
    curve = smooth(t, P, tq)
    for _ in range(n_iter):
        t = _project_param(P, curve, closed)
        curve = smooth(t, P, tq)
    return curve


def _refine_spiral_idx(P, t, G, n_iter=4):
    """Spiral refinement by index-based (nearest-vertex) ordering: it keeps the
    monotone winding structure, then REML smooths x(t), y(t)."""
    tq = np.linspace(0, 1, G)
    curve = _smooth_open(t, P, tq)
    for _ in range(n_iter):
        idx = ((P[:, None, :] - curve[None, :, :]) ** 2).sum(2).argmin(1)
        t = idx / (G - 1)
        curve = _smooth_open(t, P, tq)
    return curve


def _residual(P, curve, closed):
    d2 = ((P[:, None, :] - curve[None, :, :]) ** 2).sum(2)
    return np.sqrt(d2.min(1).mean())


def _bending(curve, closed, m=60):
    """Total squared turning angle along the polyline (smoothness penalty).

    Measured on an arc-length resampling to ``m`` vertices so that per-vertex
    noise wiggle (from a fine smoother bandwidth) does not inflate the score; the
    quantity we care about is the *global* zig-zag, not local jitter."""
    seg = np.vstack([curve, curve[:1]]) if closed else curve
    dl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    if dl[-1] < 1e-9:
        return 0.0
    tt = np.linspace(0, dl[-1], m)
    rs = np.stack([np.interp(tt, dl, seg[:, j]) for j in range(seg.shape[1])], 1)
    d = np.diff(rs, axis=0)
    tn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    ang = np.arccos(np.clip((tn[:-1] * tn[1:]).sum(1), -1, 1))
    return float((ang ** 2).sum())


def _score(P, curve, closed, lam=0.03):
    """Fit residual + smoothness penalty.  The penalty is what distinguishes a
    correct ordering (smooth, low total curvature) from a wrong one that zig-zags
    across the cloud while still hugging the points.  A *correctly* traced curve
    -- even a spiral -- is locally smooth and so has low total turning; a wrong
    ordering (over-/under-counted turns, angle-ordered open arc) zig-zags and is
    penalised."""
    return _residual(P, curve, closed) + lam * _bending(curve, closed)


def _fit_candidate(P, name, closed, t, G):
    """Return a list of (curve, closed, variant) fits for one parameterisation.
    Spirals get two refinements (index-based and arc-length projection); the
    correct ordering for a given seed may come from either."""
    if name.startswith("spiral"):
        return [(_refine_spiral_idx(P, t, G), False, "idx"),
                (_refine(P, t, False, G), False, "arc")]
    return [(_refine(P, t, closed, G), closed, "")]


def _fit_curve(P, G=200):
    """Principal-curve fit of unordered noisy 2D points -> (G,2) polyline.

    Tries several global parameterisations and keeps the refined principal curve
    (REML-smoothed x(t), y(t)) minimising  (point-to-curve residual) +
    lambda*(total squared curvature).  The curvature term rejects orderings that
    make the curve zig-zag (e.g. angle-ordering an open arc).  Selection runs at
    a coarse grid; the winner is refit at full resolution."""
    P = P - P.mean(0)
    cands, full_wrap = _candidate_inits(P)
    # A near-constant radius (low coeff. of variation) means a near-circular
    # CLOSED curve; an open spiral candidate must not be allowed to hijack it.
    r = np.linalg.norm(P, axis=1)
    near_circular = full_wrap and (r.std() / (r.mean() + 1e-12)) < 0.12
    best_curve, best_closed = None, False
    best_score = np.inf
    # Selection and the returned curve come from the SAME fit at a fixed grid Gs
    # (the index-based spiral refinement is grid-dependent, so we must not refit
    # the winner at a different grid).  REML curves are smooth, so we then simply
    # arc-length-resample the winner to the requested output resolution G.
    Gs = 120
    for name, closed, t in cands:
        if name.startswith("spiral") and near_circular:
            continue                             # no spirals for a near-circle
        if name == "angle" and not full_wrap:   # angular init must fully wrap
            closed = False
        for curve, cl, variant in _fit_candidate(P, name, closed, t, Gs):
            score = _score(P, curve, cl)
            # Tie-break toward a CLOSED model when the cloud fully wraps the
            # centroid: a genuinely closed curve (circle/ellipse) should not be
            # narrowly out-scored by an open spiral candidate that happens to hug
            # it.  A true spiral keeps a comfortable residual margin and still
            # wins despite this small handicap.
            if name.startswith("spiral") and full_wrap:
                score += 0.02
            if score < best_score:
                best_score = score
                best_curve, best_closed = curve, cl
    return _resample(best_curve, G, best_closed)


def _resample(curve, G, closed):
    """Arc-length resample a (Gs,2) polyline to G vertices."""
    seg = np.vstack([curve, curve[:1]]) if closed else curve
    dl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    if dl[-1] < 1e-12:
        return np.repeat(curve[:1], G, 0)
    tt = np.linspace(0, dl[-1], G, endpoint=not closed)
    return np.stack([np.interp(tt, dl, seg[:, j]) for j in range(seg.shape[1])], 1)


# --------------------------------------------------------------------------- #
#  public API                                                                  #
# --------------------------------------------------------------------------- #
def recover(X, K, G=200):
    X = np.asarray(X, dtype=np.float64)
    frames = _recover_blocks(X, K)

    # assign points to blocks by projected energy (deterministic projection)
    en = np.stack([((X @ F.T) ** 2).sum(1) for F in frames], 1)   # (N, K)
    full = (X ** 2).sum(1) + 1e-12
    owner = en.argmax(1)

    curves = []
    for i, F in enumerate(frames):
        mask = owner == i
        if mask.sum() < 10:
            curves.append(None)
            continue
        P2 = X[mask] @ F.T                          # (n_i, 2)
        c2 = _fit_curve(P2, G=G)
        curves.append(c2 @ F)                       # lift to R^D
    return curves
