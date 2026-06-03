"""Independent Subspace Analysis (ISA) recovery of a union of K 1D curves,
each living in its own 2D linear subspace, the K subspaces mutually orthogonal.

Pipeline (NO clustering, NO kNN, NO trained encoder):

1. Center + PCA to the 2K-dim span of the orthogonal blocks; whiten.
2. JADE-style joint diagonalization of 4th-order cumulant matrices to find the
   orthogonal rotation that separates the whitened data into independent
   coordinates. Because only one 2D block is "active" per row (others are ~0 +
   noise), the 4th-order cumulant structure is block-diagonal in the true
   blocks, so the recovered independent directions align with the 2D blocks.
3. Group the 2K recovered directions into K pairs (2D subspaces) using a
   4th-order subspace-dependence affinity (quadratic-correlation of squared
   projections), spectrally clustered into K groups of size 2 via a global
   greedy matching on the affinity -- this is grouping of *axes*, not points.
4. For each 2D subspace: select the rows whose energy concentrates there (a
   soft, per-point gating by relative block energy -- not a partition / not
   k-means), recover a 1D parameterization, and decide closed vs open topology:
      - closed: a complex Fourier series z(t)=sum c_m e^{2 pi i m t} recovers
        the cyclic phase by iterative reparameterization (init = analytic
        angle, then project onto the continuous model);
      - open: a dense heat-kernel Laplacian eigenmap (ALL pairs -- not a k-NN
        graph) gives a fold-robust global ordering, parameterized by arc length.
   Closed/open is decided by a support-gap topology test plus a Fourier-vs-poly
   residual comparison (a spiral tiles the angle circle but the closed model
   cannot capture its winding, so its residual flags it open).
5. Fit the final curve with gamfit's Gaussian-REML Duchon smoother (REML
   auto-selects the smoothing parameter; periodic Duchon for closed curves) as a
   function of normalized cumulative ARC LENGTH, resample to a uniform (G,2)
   grid, and lift back to ambient R^D.
"""
from __future__ import annotations

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
# gamfit REML resampling of a 2D curve given a 1D parameterization
# --------------------------------------------------------------------------- #
def _reml_resample(order, P2, w, G, periodic, period=1.0):
    """Fit x(s), y(s) with gamfit's Gaussian-REML Duchon smoother -- REML
    auto-selects the smoothing parameter (no hand-rolled spline) -- where s is
    the *normalized cumulative arc length* along the supplied 1D ordering, then
    resample on a uniform arc-length grid. Arc-length (vs phase) parameterizing
    keeps samples dense through high-curvature regions (e.g. a cardioid cusp).
    A uniform grid is appended as ~zero-weight rows so the REML ``fitted`` values
    at those rows give the resampled curve.

    ``order``: per-point scalar whose sort gives the curve traversal order.
    """
    order = np.asarray(order, float)
    P2 = np.asarray(P2, float)
    w = np.asarray(w, float)
    N = len(order)
    o = np.argsort(order)
    Po = P2[o]
    if periodic:
        seg = np.linalg.norm(np.diff(np.vstack([Po, Po[:1]]), axis=0), axis=1)
        cal = np.r_[0.0, np.cumsum(seg)]
        total = cal[-1] if cal[-1] > 1e-12 else 1.0
        s = np.empty(N)
        s[o] = cal[:-1] / total
        tg = np.linspace(0.0, 1.0, G, endpoint=False)
        per_kw = dict(periodic=True, period=1.0)
    else:
        seg = np.linalg.norm(np.diff(Po, axis=0), axis=1)
        cal = np.r_[0.0, np.cumsum(seg)]
        total = cal[-1] if cal[-1] > 1e-12 else 1.0
        s = np.empty(N)
        s[o] = cal / total
        tg = np.linspace(0.0, 1.0, G)
        per_kw = dict(periodic=False)

    t_all = np.r_[s, tg]
    wgt = np.r_[w, np.full(G, 1e-8)]
    out = np.empty((G, 2))
    for ax in range(2):
        y_all = np.r_[P2[:, ax], np.zeros(G)]
        r = gamfit.gaussian_reml_fit_positions(
            t_all, y_all, basis="duchon", basis_order=2, weights=wgt, **per_kw,
        )
        out[:, ax] = np.asarray(r["fitted"]).ravel()[N:]
    return out


# --------------------------------------------------------------------------- #
# whitening
# --------------------------------------------------------------------------- #
def _whiten(X, n_comp):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(n_comp, len(S))
    V = Vt[:n_comp].T                      # (D, n_comp) ambient basis of the PCA subspace
    lam = (S[:n_comp] ** 2) / Xc.shape[0]  # eigenvalues (variances)
    # whitened coords  Z = Xc @ V / sqrt(lam)
    Z = (Xc @ V) / np.sqrt(lam)
    # de-whitening (ambient) : x = Z * sqrt(lam) @ V.T + mu
    return Z, V, lam, mu


# --------------------------------------------------------------------------- #
# JADE: joint diagonalization of 4th-order cumulant matrices
# --------------------------------------------------------------------------- #
def _cumulant_matrices(Z):
    """Set of n^2 (n,n) 4th-order cumulant matrices Q_M (JADE)."""
    N, n = Z.shape
    R = np.eye(n)                          # cov of whitened data == I
    mats = []
    for p in range(n):
        for q in range(p, n):
            # E[(zp zq) z z^T]
            w = Z[:, p] * Z[:, q]
            M = (Z * w[:, None]).T @ Z / N
            # subtract Gaussian part: R_pq R + R_p R_q^T + R_q R_p^T
            Epq = R[p, q]
            M = M - Epq * R
            M = M - np.outer(R[:, p], R[:, q]) - np.outer(R[:, q], R[:, p])
            mats.append(M)
    return mats


def _joint_diag(mats, n_iter=200, tol=1e-10):
    """Jacobi joint diagonalization (Cardoso & Souloumiac) of symmetric mats."""
    n = mats[0].shape[0]
    V = np.eye(n)
    A = np.stack(mats).astype(np.float64)   # (m, n, n)
    for _ in range(n_iter):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                h = np.array([A[:, p, p] - A[:, q, q], A[:, p, q] + A[:, q, p]])
                G = h @ h.T
                ton = G[0, 0] - G[1, 1]
                toff = G[0, 1] + G[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) < 1e-15:
                    continue
                off += abs(s)
                # rotate columns p,q of every matrix (both sides)
                cp = A[:, :, p].copy(); cq = A[:, :, q].copy()
                A[:, :, p] = c * cp + s * cq
                A[:, :, q] = -s * cp + c * cq
                rp = A[:, p, :].copy(); rq = A[:, q, :].copy()
                A[:, p, :] = c * rp + s * rq
                A[:, q, :] = -s * rp + c * rq
                vp = V[:, p].copy(); vq = V[:, q].copy()
                V[:, p] = c * vp + s * vq
                V[:, q] = -s * vp + c * vq
        if off < tol:
            break
    return V


# --------------------------------------------------------------------------- #
# group independent axes into K 2D subspaces
# --------------------------------------------------------------------------- #
def _group_axes(S, K):
    """S: (N, 2K) independent-ish coords. Return list of K index-pairs.

    Affinity = correlation of squared coords (4th-order dependence): axes in the
    same 2D curve-block share a point's energy, so |s_i|^2 and |s_j|^2 covary
    strongly *across rows* (both ~0 when another block is active, both nonzero
    together). Pure-noise cross terms vanish. Greedy global matching into pairs.
    """
    P = S ** 2
    P = P - P.mean(0)
    C = P.T @ P
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    A = C / np.outer(d, d)
    n = S.shape[1]
    np.fill_diagonal(A, -np.inf)
    # greedy maximum-weight matching into K pairs
    pairs, used = [], set()
    order = np.argsort(-A, axis=None)
    for idx in order:
        i, j = divmod(idx, n)
        if i in used or j in used or i == j:
            continue
        pairs.append((i, j))
        used.add(i); used.add(j)
        if len(pairs) == K:
            break
    # safety: pair up any leftovers
    rest = [k for k in range(n) if k not in used]
    for a in range(0, len(rest) - 1, 2):
        pairs.append((rest[a], rest[a + 1]))
    return pairs[:K]


# --------------------------------------------------------------------------- #
# 2D parametric curve fits
# --------------------------------------------------------------------------- #
def _basis_closed(t, M):
    ms = np.arange(-M, M + 1)
    return np.exp(2j * np.pi * np.outer(t, ms))


def _eval_closed(c, t, M):
    ms = np.arange(-M, M + 1)
    return np.exp(2j * np.pi * np.outer(t, ms)) @ c


def _closed_param(P2, w, M=10, iters=14):
    """Estimate the closed-curve *phase* t in [0,1) per point via iterative
    Fourier reparameterization (init = analytic angle, then alternate
    weighted-LS refit and projection onto the continuous model). The Fourier
    model is used only to recover the cyclic ordering/phase; the final curve is
    fit by gamfit periodic-Duchon REML. Returns (t, residual)."""
    z = P2[:, 0] + 1j * P2[:, 1]
    t = (np.angle(z) / (2 * np.pi)) % 1.0
    sw = np.sqrt(w)
    grid = np.linspace(0, 1, 512, endpoint=False)
    c = None
    for _ in range(iters):
        Phi = _basis_closed(t, M)
        c, *_ = np.linalg.lstsq(Phi * sw[:, None], z * sw, rcond=None)
        zg = _eval_closed(c, grid, M)
        d = np.abs(z[:, None] - zg[None, :])
        t = grid[np.argmin(d, axis=1)]
    res = np.sqrt(np.average(np.abs(_eval_closed(c, t, M) - z) ** 2, weights=w))
    return t, res


def _spectral_orders(Q, wq):
    """Candidate global 1D orderings from the leading nontrivial eigenvectors of
    a *dense* heat-kernel Laplacian (ALL pairs -- NOT a k-NN graph). For a smooth
    1D curve such a Fiedler coordinate is monotone in arc length, so it orders
    even folded curves (spiral/semicircle/wave) that no linear projection can.
    Sweep the bandwidth; yield candidates ranked by ordering smoothness."""
    D2 = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    off = D2[~np.eye(len(Q), dtype=bool)]
    cands = []
    for q in (0.003, 0.01, 0.03, 0.06, 0.12):
        sig2 = np.quantile(off, q) + 1e-12
        Aff = np.exp(-D2 / sig2) * np.outer(wq, wq)
        d = Aff.sum(1)
        Dm = 1.0 / np.sqrt(d + 1e-12)
        L = np.eye(len(Q)) - (Dm[:, None] * Aff * Dm[None, :])
        _, evecs = np.linalg.eigh(L)
        for k in (1, 2):
            f = evecs[:, k] * Dm
            o = np.argsort(f)
            smooth = np.linalg.norm(np.diff(Q[o], axis=0), axis=1).max()
            cands.append((smooth, f))
    cands.sort(key=lambda c: c[0])
    return [f for _, f in cands]


def _arclen_param(Q, f):
    """Normalized cumulative arc-length parameter t in [0,1] along ordering f."""
    o = np.argsort(f)
    al = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q[o], axis=0), axis=1))]
    if al[-1] < 1e-9:
        return None
    al /= al[-1]
    t = np.empty(len(Q))
    t[o] = al
    return t


def _poly_res(Q, wq, t, M=14):
    """Cheap polynomial residual for ranking ordering candidates (model
    selection only -- the final curve is fit by REML)."""
    sw = np.sqrt(wq)
    Vd = np.vander(2 * t - 1, M + 1, increasing=True)
    cx, *_ = np.linalg.lstsq(Vd * sw[:, None], Q[:, 0] * sw, rcond=None)
    cy, *_ = np.linalg.lstsq(Vd * sw[:, None], Q[:, 1] * sw, rcond=None)
    return np.sqrt(np.average((Vd @ cx - Q[:, 0]) ** 2 + (Vd @ cy - Q[:, 1]) ** 2, weights=wq))


def _open_param(P2, w):
    """Estimate an open-curve arc-length parameter t for the in-block points:
    order them by a dense spectral (Laplacian) coordinate -- fold-robust -- and
    take cumulative arc length. Best of several ordering candidates by a cheap
    polynomial residual. Returns (mask, Q, wq, t, residual)."""
    m = w > 0.25
    if m.sum() < 40:
        m = w > 0.1 * w.max()
    Q, wq = P2[m], w[m]
    cands = _spectral_orders(Q, wq)
    ctr = np.average(Q, 0, weights=wq)
    _, _, Vt = np.linalg.svd((Q - ctr) * np.sqrt(wq)[:, None], full_matrices=False)
    cands = cands[:4] + [(Q - ctr) @ Vt[0]]
    best = None
    for f in cands:
        t = _arclen_param(Q, f)
        if t is None:
            continue
        res = _poly_res(Q, wq, t)
        if best is None or res < best[1]:
            best = (t, res)
    return m, Q, wq, best[0], best[1]


def _is_closed(P2, w):
    """Topology test: fit a closed model, then measure the largest angular gap
    in the *coverage of the model parameter* by data. A genuinely open curve
    leaves a large stretch of the closed model un-supported (the closing arc).
    Returns the support gap fraction (small => closed, large => open)."""
    t, _ = _closed_param(P2, w)
    z = P2[:, 0] + 1j * P2[:, 1]
    # closed Fourier model curve on a dense grid (cheap; for the gap test only)
    M = 10
    Phi = _basis_closed(t, M)
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(Phi * sw[:, None], z * sw, rcond=None)
    grid = np.linspace(0, 1, 256, endpoint=False)
    zg = _eval_closed(c, grid, M)
    d = np.abs(z[:, None] - zg[None, :])
    near = np.argmin(d, axis=1)
    G = len(zg)
    occ = np.zeros(G)
    np.add.at(occ, near, w)
    supp = occ > (w.sum() / G) * 0.04
    # largest run of unsupported grid cells (cyclically)
    ss = np.r_[supp, supp]
    maxgap = run = 0
    for v in ss:
        run = 0 if v else run + 1
        maxgap = max(maxgap, run)
    return min(maxgap, G) / G


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def recover(X, K, G=200):
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape

    # 1. whiten to the 2K-dim block span
    Z, V, lam, mu = _whiten(X, 2 * K)
    n = Z.shape[1]

    # 2. JADE rotation -> independent coordinates
    mats = _cumulant_matrices(Z)
    W = _joint_diag(mats)                  # (n,n) orthogonal
    S = Z @ W                              # independent-ish coords (N, n)

    # ambient basis for each whitened/rotated coordinate:
    #   x ~= sum_k S[:,k] * sqrt(lam) ... actually rebuild full transform.
    # Z = Xc @ V / sqrt(lam);  S = Z @ W.
    # To map a 2D point in S-space back to ambient:
    #   Xc = (S @ W.T) * sqrt(lam) @ V.T
    # so ambient column for coord k = (W.T scaling) -> build basis B (n, D):
    #   for unit vector e_k in S-space: Z_row = W @ e_k ; Xc = (Z_row*sqrt(lam)) @ V.T
    Bk = (W * np.sqrt(lam)[:, None]).T @ V.T   # wrong-shape guard below
    # Bk[k] should be ambient image of S-coord k. Derive cleanly:
    # S-coord axis k -> Z = W[:,k] -> ambient = (W[:,k]*sqrt(lam)) @ V.T
    Bk = ((W.T) * np.sqrt(lam)[None, :]) @ V.T   # (n, D); row k = ambient image of e_k

    # 3. group axes into K 2D subspaces
    pairs = _group_axes(S, K)

    curves = []
    for (i, j) in pairs:
        P2 = S[:, [i, j]]                  # (N, 2) coords in this subspace
        # per-point energy in this block vs total
        e_block = P2[:, 0] ** 2 + P2[:, 1] ** 2
        e_tot = (S ** 2).sum(1)
        frac = e_block / (e_tot + 1e-12)
        # soft gate: weight points whose energy is dominated by this block
        w = np.clip(frac - 0.5, 0, None) ** 2
        if w.sum() < 1e-6:
            w = frac ** 4
        w = w / w.max()

        B2 = Bk[[i, j]]                    # (2, D) ambient basis for this subspace

        # cheap parameter estimates + residuals for the closed/open decision
        gap = _is_closed(P2, w)            # small => closed topology
        tc, res_c = _closed_param(P2, w)
        m, Q, wq, to, res_o = _open_param(P2, w)
        # Closed only when the model both *closes* (small support gap) AND the
        # closed (Fourier) residual beats the open (poly) residual. A spiral
        # nearly tiles the angle circle (small gap) but the closed model cannot
        # capture its winding radius, so res_c >> res_o flags it as open.
        closed = (gap < 0.06) and (res_c < res_o)

        # final curve: gamfit Duchon REML (auto-smoothing) over arc length, then
        # resample to G points. Use the in-block (high-weight) points only.
        if closed:
            out2 = _reml_resample(tc[m], Q, wq, G, periodic=True, period=1.0)
        else:
            out2 = _reml_resample(to, Q, wq, G, periodic=False)

        curve = out2 @ B2 + mu             # lift to ambient (G, D)
        curves.append(curve)

    return curves
