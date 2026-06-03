"""Distribution-matching / optimal-transport recovery of a union of curves.

Model of the data.  X (N x D) is a mixture of K pushforwards.  Each ground-
truth curve g_k(t), t ~ Uniform[0,1], lifted into its own 2D subspace of R^D,
is the pushforward of the uniform law on [0,1]; the cloud is the equal-weight
mixture of these K distributions, and the K 2D subspaces are mutually
orthogonal.  Everything is recovered by minimising a SLICED-WASSERSTEIN
distance (random 1D projections, sort, L2) between the data cloud and a model
cloud, by gradient descent in torch.  No clustering, no nearest-curve
assignment, no k-NN graph, no encoder.

Pipeline (all distribution matching):

  1.  Reduce X to the 2K-dim PCA span the orthogonal subspaces occupy.

  2.  Greedy OT deflation of subspaces.  Fit a single parametric curve (its own
      2D frame + closed polyline) to the residual cloud by sliced-Wasserstein,
      modelling the other curves as a point mass at the origin (which is exact,
      because the subspaces are orthogonal so every other curve projects to ~0).
      The single curve locks onto the dominant un-explained 2D plane; project
      that plane out and repeat K times.  This hands us K orthogonal 2D frames.

  3.  Per-subspace 2D curve fit by OT.  Project the WHOLE cloud onto each
      recovered plane.  Orthogonality collapses every other curve to a tiny
      origin blob, leaving curve k at full extent.  A free 2D polyline is fit to
      that 2D distribution by sliced-Wasserstein, with the origin blob modelled
      explicitly so the curve only explains its own arm.  Closed/open topology
      and (for winding curves) the number of turns are chosen by lowest final
      sliced-Wasserstein over a small bank of initialisations.

  4.  REML smoothing.  The OT polyline positions are smoothed coordinate-wise
      against arc-length with gamfit's Duchon-spline Gaussian REML
      (``gaussian_reml_fit_positions``, periodic for closed curves); REML
      auto-selects the smoothness, removing residual OT jitter.

Exposes ``recover(X, K) -> list of K (G, D) arrays``.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import gamfit
    _HAS_GAMFIT = True
except Exception:                                   # pragma: no cover
    _HAS_GAMFIT = False


# --------------------------------------------------------------------------- #
#  Sliced-Wasserstein-2 between two clouds.
# --------------------------------------------------------------------------- #
def _sliced_w2(A, B, n_proj, gen):
    d = A.shape[1]
    dirs = torch.randn(d, n_proj, generator=gen, device=A.device, dtype=A.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
    pa = (A @ dirs).sort(dim=0).values
    pb = (B @ dirs).sort(dim=0).values
    if pa.shape[0] != pb.shape[0]:
        n = max(pa.shape[0], pb.shape[0])
        q = torch.linspace(0, 1, n, device=A.device, dtype=A.dtype)
        pa, pb = _qi(pa, q), _qi(pb, q)
    return ((pa - pb) ** 2).mean()


def _qi(sv, q):
    n = sv.shape[0]
    pos = q * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    w = (pos - lo.to(sv.dtype)).unsqueeze(1)
    return sv[lo] * (1 - w) + sv[hi] * w


def _interp1(cp, t, closed):
    """cp:(P,2), t:(G,) -> (G,2) piecewise-linear (periodic if closed)."""
    P = cp.shape[0]
    if closed:
        pos = t * P
        lo = pos.floor().long() % P
        hi = (lo + 1) % P
        w = (pos - pos.floor()).view(-1, 1)
    else:
        pos = t * (P - 1)
        lo = pos.floor().long().clamp(0, P - 1)
        hi = (lo + 1).clamp(0, P - 1)
        w = (pos - lo.to(cp.dtype)).view(-1, 1)
    return cp[lo] * (1 - w) + cp[hi] * w


# --------------------------------------------------------------------------- #
#  Stage 2: single-curve OT fit to a residual cloud -> one 2D plane.
# --------------------------------------------------------------------------- #
def _fit_one_plane(Xt, K, seed, n_steps=700):
    N, r = Xt.shape
    gen = torch.Generator().manual_seed(seed)
    ds = float(np.sqrt((Xt ** 2).sum(1).mean().item()) + 1e-12)
    frame = (0.1 * torch.randn(r, 2, generator=gen, dtype=Xt.dtype)).requires_grad_()
    P = 60
    a = torch.linspace(0, 2 * np.pi, P, dtype=Xt.dtype)
    cp = (0.5 * ds * torch.stack([torch.cos(a), torch.sin(a)], 1)).clone().requires_grad_()
    opt = torch.optim.Adam([frame, cp], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    for step in range(n_steps):
        Q, R = torch.linalg.qr(frame)
        Q = Q * torch.sign(torch.diagonal(R))
        idx = torch.randint(0, N, (1500,), generator=gen)
        Xb = Xt[idx]
        nc = 1500 // K
        t = torch.rand(nc, generator=gen, dtype=Xt.dtype)
        cl = _interp1(cp, t, True) @ Q.T
        bl = torch.zeros(1500 - nc, r, dtype=Xt.dtype)
        sw = _sliced_w2(Xb, torch.cat([cl, bl], 0), 192, gen)
        d2 = cp.roll(-1, 0) - 2 * cp + cp.roll(1, 0)
        loss = sw + (0.02 + 0.3 * (1 - step / n_steps)) * (d2 ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        Q, R = torch.linalg.qr(frame)
        Q = (Q * torch.sign(torch.diagonal(R)))
    return Q


def _greedy_subspaces(Xr, K, seed):
    """Greedy OT deflation: pull off one 2D plane at a time."""
    r = Xr.shape[1]
    Xcur = Xr.copy()
    planes = []
    for k in range(K):
        Xt = torch.tensor(Xcur, dtype=torch.float64)
        Q = _fit_one_plane(Xt, K, seed=seed * 131 + 7 * k + 1).numpy()  # (r,2)
        planes.append(Q)
        Xcur = Xcur - Xcur @ Q @ Q.T                # deflate this plane
    return planes


# --------------------------------------------------------------------------- #
#  Stage 3: per-subspace 2D curve fit (handles origin blob explicitly).
# --------------------------------------------------------------------------- #
def _detect_closed(P2):
    nrm = np.linalg.norm(P2 - np.median(P2, 0), axis=1)
    arm = P2[nrm > 0.4 * np.quantile(nrm, 0.95)]
    if len(arm) < 12:
        return False, 0.0
    c = arm.mean(0)
    rel = arm - c
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.linalg.norm(rel, axis=1)
    asort = np.sort(ang)
    maxgap = np.degrees(np.diff(np.r_[asort, asort[0] + 2 * np.pi]).max())
    bins = np.linspace(-np.pi, np.pi, 13)
    bi = np.digitize(ang, bins)
    cvs = [rad[bi == b].std() / (rad[bi == b].mean() + 1e-9) for b in range(1, 13) if (bi == b).sum() > 3]
    cv = float(np.median(cvs)) if cvs else 0.0
    closed = (maxgap < 60) and (cv < 0.15)
    return closed, cv


def _fit_curve_2d(P2t, closed, init, seed, K, n_steps=600):
    N = P2t.shape[0]
    gen = torch.Generator().manual_seed(seed)
    ctr = P2t.median(0).values
    sc = float((P2t - ctr).norm(dim=1).quantile(0.95))
    near = P2t[(P2t - ctr).norm(dim=1) < 0.3 * sc]
    bstd = near.std(0).detach() if near.shape[0] > 10 else torch.full((2,), 0.01, dtype=P2t.dtype)
    cp = init.clone().requires_grad_()
    opt = torch.optim.Adam([cp], lr=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    for step in range(n_steps):
        idx = torch.randint(0, N, (1800,), generator=gen)
        Xb = P2t[idx]
        nc = 1800 // K
        t = torch.rand(nc, generator=gen, dtype=P2t.dtype)
        cl = _interp1(cp, t, closed)
        bl = ctr + bstd * torch.randn(1800 - nc, 2, generator=gen, dtype=P2t.dtype)
        sw = _sliced_w2(Xb, torch.cat([cl, bl], 0), 256, gen)
        if closed:
            d2 = cp.roll(-1, 0) - 2 * cp + cp.roll(1, 0)
            seg = (cp.roll(-1, 0) - cp).pow(2).sum(-1)
        else:
            d2 = cp[2:] - 2 * cp[1:-1] + cp[:-2]
            seg = (cp[1:] - cp[:-1]).pow(2).sum(-1)
        loss = sw + (0.003 + 0.07 * (1 - step / n_steps)) * (d2 ** 2).sum(-1).mean() + 0.03 * seg.var()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(seed + 1)
        t = torch.rand(N // K, generator=g2, dtype=P2t.dtype)
        cl = _interp1(cp, t, closed)
        nb = N - N // K
        bl = ctr + bstd * torch.randn(nb, 2, generator=g2, dtype=P2t.dtype)
        fsw = _sliced_w2(P2t, torch.cat([cl, bl], 0), 512, torch.Generator().manual_seed(seed + 2)).item()
    return cp.detach(), fsw


def _stage_b(P2, seed, K, G):
    P2t = torch.tensor(P2, dtype=torch.float64)
    ctr = P2t.median(0).values
    sc = float((P2t - ctr).norm(dim=1).quantile(0.95))
    closed, cv = _detect_closed(P2)
    P = 80
    cands = []
    if closed:
        a = torch.linspace(0, 2 * np.pi, P + 1, dtype=torch.float64)[:-1]
        ring = ctr + 0.6 * sc * torch.stack([torch.cos(a), torch.sin(a)], 1)
        cands.append((True, ring))
    else:
        Pc = P2t - P2t.mean(0)
        _, _, Vt = torch.linalg.svd(Pc, full_matrices=False)
        ax = Vt[0]
        s = torch.linspace(-1, 1, P, dtype=torch.float64).unsqueeze(1)
        cands.append((False, ctr + s * ax * sc))
        if cv > 0.15:                                # winding open curve (spiral): add spiral inits
            cen = P2t.mean(0)
            tt = torch.linspace(0, 1, P, dtype=torch.float64)
            for turns in (1.5, 2.0, 2.5, 3.0):
                rr = 0.1 * sc + sc * tt
                th = 2 * np.pi * turns * tt
                cands.append((False, cen + torch.stack([rr * torch.cos(th), rr * torch.sin(th)], 1)))
    best = None
    for j, (cl_flag, init) in enumerate(cands):
        cp, fsw = _fit_curve_2d(P2t, cl_flag, init, seed * 17 + j, K)
        if best is None or fsw < best[0]:
            best = (fsw, cl_flag, cp)
    _, cl_flag, cp = best
    if cl_flag:
        tt = torch.linspace(0, 1, G + 1, dtype=torch.float64)[:-1]
    else:
        tt = torch.linspace(0, 1, G, dtype=torch.float64)
    return _interp1(cp, tt, cl_flag).numpy(), cl_flag


# --------------------------------------------------------------------------- #
#  Stage 4: REML smoothing of the recovered polyline.
# --------------------------------------------------------------------------- #
def _reml_smooth(curve, closed):
    if not _HAS_GAMFIT:
        return curve
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return curve
    if closed:
        t = d / (d[-1] + np.linalg.norm(curve[0] - curve[-1]) + 1e-12)
    else:
        t = d / d[-1]
    amp = np.ones_like(t)
    out = np.zeros_like(curve)
    for j in range(curve.shape[1]):
        try:
            res = gamfit.gaussian_reml_fit_positions(
                t, curve[:, j], basis="duchon", basis_order=2,
                by=amp, periodic=bool(closed), period=(1.0 if closed else None))
            out[:, j] = np.asarray(res["fitted"]).ravel()
        except Exception:
            out[:, j] = curve[:, j]
    return out


# --------------------------------------------------------------------------- #
def recover(X, K, seed=0, G=200, verbose=False):
    torch.manual_seed(seed)
    Xnp = np.asarray(X, dtype=np.float64)
    N, D = Xnp.shape
    mu = Xnp.mean(0)
    Xc = Xnp - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(2 * K, D)
    Vr = Vt[:r]
    Xr = Xc @ Vr.T

    planes = _greedy_subspaces(Xr, K, seed)         # K planes (r,2) in reduced coords

    out = []
    for k, Q in enumerate(planes):
        Qo = np.linalg.qr(Q)[0]                      # ensure orthonormal
        P2 = Xr @ Qo                                 # (N,2)
        c2, closed = _stage_b(P2, seed * 53 + k, K, G)
        c2 = _reml_smooth(c2, closed)
        out.append(c2 @ Qo.T @ Vr + mu)             # 2D -> reduced -> ambient
        if verbose:
            print(f"  curve {k}: closed={closed}", flush=True)
    return out


if __name__ == "__main__":
    from experiments.recover_bench import evaluate
    evaluate(recover, seeds=(0,))
