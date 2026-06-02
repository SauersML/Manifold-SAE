"""Joint, gauge-pinned manifold SAE — the architecture the session converged on.

x = sum_k gate_k * a_k * g_k(t_k), trained END-TO-END so the disentangling (which
manifolds, gate), the chart (where on each, t_k), and the shape (g_k) co-adapt —
because under superposition they are one coupled fixed point and cannot be solved
separately.

The naive version of this collapses (positions bunch, curves flatten) because of
the REPARAMETERIZATION GAUGE: you can slide t and reshape g and reconstruct
identically, so gradient wanders the null space. Two fixes make it well-posed:

  * GAUGE PINNING — an arc-length penalty (the curve should move at constant speed
    in t) removes the reparameterization freedom, making the chart identifiable
    AND a meaningful steering coordinate; a coverage term keeps positions spread.
  * DIVERSE INIT — each atom starts in its own random 2D subspace so atoms don't
    duplicate.

Decoder g_k is a Duchon thin-plate spline (differentiable in t), so it is shape-
flexible (any curve, not just a quadric). Free per-token latents (t, a, gate) are
optimized directly with the atoms — the transductive joint fit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


N_HARM = 3   # Fourier harmonics appended to the basis so CLOSED loops can close


def duchon_basis_torch(t, centers):
    """Duchon m=2 thin-plate basis ++ linear ++ Fourier harmonics.
    [|t-c|^3]_c  ++  [1, t]  ++  [sin2pi k t, cos2pi k t]_{k=1..N_HARM}.
    The RBF+linear part represents open curves; the Fourier part lets a curve
    close smoothly (g(0)=g(1)) without a discontinuity at the t=0/1 seam."""
    r = (t[..., None] - centers).abs()
    feats = [r ** 3, torch.ones_like(t)[..., None], t[..., None]]
    for k in range(1, N_HARM + 1):
        feats.append(torch.sin(2 * np.pi * k * t)[..., None])
        feats.append(torch.cos(2 * np.pi * k * t)[..., None])
    return torch.cat(feats, dim=-1)


def _basis_dim(M):
    return M + 2 + 2 * N_HARM


class ManifoldSAEJoint(torch.nn.Module):
    def __init__(self, N, K, D, M=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.K, self.D, self.M = K, D, M
        self.register_buffer("centers", torch.linspace(0, 1, M))
        # decoder: per-atom spline coeffs, init in DISTINCT random 2D subspaces
        B = torch.zeros(K, _basis_dim(M), D)
        grid = torch.linspace(0, 1, M)
        for k in range(K):
            Q, _ = torch.linalg.qr(torch.randn(D, 2, generator=g))     # random 2D plane
            ang = 2 * np.pi * grid
            loop = torch.stack([torch.cos(ang), torch.sin(ang)], 1) @ Q.T  # (M,D) seed curve
            phi = duchon_basis_torch(grid, self.centers)                   # (M, basis)
            B[k] = 0.3 * torch.linalg.lstsq(phi, loop).solution
        self.B = torch.nn.Parameter(B)
        # free per-token latents
        self.t_raw = torch.nn.Parameter(torch.randn(N, K, generator=g))    # -> sigmoid -> position
        self.c_raw = torch.nn.Parameter(-3.0 + 0.1 * torch.randn(N, K, generator=g))  # -> softplus -> code

    def curve(self, k, grid):
        return duchon_basis_torch(grid, self.centers) @ self.B[k]          # (G, D)

    def forward(self, topk=None):
        t = torch.sigmoid(self.t_raw)
        c = F.softplus(self.c_raw)                                         # ONE nonneg sparse code (gate=amp)
        if topk is not None and topk < self.K:
            # hard top-k mask (straight-through): keep the k largest codes per
            # token, zero the rest. With free per-atom positions, ANY atom can
            # fit ANY point, so reconstruction alone gives no pressure to choose
            # the right atom -- the hard selection supplies it.
            thr = torch.topk(c, topk, dim=1).values[:, -1:]
            c = c * (c >= thr).detach()
        phi = duchon_basis_torch(t, self.centers)                         # (N,K,basis)
        g = torch.einsum("nkm,kmd->nkd", phi, self.B)                     # atom curve at each t
        xhat = torch.einsum("nk,nkd->nd", c, g)
        return xhat, c, t


@torch.no_grad()
def _warm_start(sae, X, K, M):
    """Geometric warm-start: init each atom's shape/positions/codes from the
    moment + co-activation subspace recovery. Random init lands the joint fit in
    an entangled local minimum (any flexible atom can fit any point via its free
    position); a geometric basis breaks that symmetry so the refit only has to
    clean up the overlap that superposition introduced."""
    from experiments.method_moment_reml import recover_subspaces, _init_positions
    Xn = np.asarray(X)
    blocks = recover_subspaces(Xn, K)                       # K x (D,2)
    grid = torch.linspace(0, 1, 200)
    phi_grid = duchon_basis_torch(grid, sae.centers)        # (200, basis)
    for k, B in enumerate(blocks):
        Y = Xn @ B                                          # (N,2) projection onto block
        t, closed = _init_positions(Y)
        # build a smooth curve in R^D from sorted projected points, fit atom coeffs
        o = np.argsort(t); ts = t[o]; cu = (Y[o] @ B.T)     # (N,D) curve samples along t
        curve = torch.stack([torch.from_numpy(np.interp(grid.numpy(), ts, cu[:, j]))
                             for j in range(cu.shape[1])], 1)   # (200,D)
        sae.B[k] = torch.linalg.lstsq(phi_grid, curve).solution
        sae.t_raw[:, k] = torch.logit(torch.from_numpy(np.clip(t, 1e-3, 1 - 1e-3)))
        energy = np.linalg.norm(Y, axis=1)
        sae.c_raw[:, k] = torch.from_numpy(np.log(np.expm1(np.clip(energy, 1e-3, None))))
    return sae


def fit(X, K, M=16, n_steps=4000, lr=8e-3, sparsity=2e-2, smooth=1e-4,
        arclen=5e-2, topk=2, warm_start=True, seed=0, log_every=0):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    N, D = Xt.shape
    sae = ManifoldSAEJoint(N, K, D, M=M, seed=seed)
    if warm_start:
        _warm_start(sae, X, K, M)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    grid = torch.linspace(0, 1, 64)
    for step in range(n_steps):
        opt.zero_grad()
        xhat, c, t = sae(topk=topk)
        mse = ((xhat - Xt) ** 2).mean()
        sp = c.mean()                                              # L1 on the code -> sparse active set
        # arc-length GAUGE: curve should move at ~constant speed in t (pins the
        # reparameterization gauge -> identifiable, steerable chart). Also gives
        # per-atom speed -> normalize atoms to unit length (fix code<->scale gauge).
        crv = torch.stack([sae.curve(k, grid) for k in range(K)], 0)   # (K,64,D)
        seglen = torch.linalg.norm(torch.diff(crv, dim=1), dim=2)      # (K,63)
        # arc-length gauge as a SCALE-FREE coefficient of variation of speed (pins
        # reparameterization without dictating overall scale -> loops can grow).
        arc = (seglen.std(dim=1) / (seglen.mean(dim=1) + 1e-9)).mean()
        cur = (sae.B[:, :M, :] ** 2).mean()                          # smoothness on RBF coeffs
        loss = mse + sparsity * sp + smooth * cur + arclen * arc
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            act = (c > 0.1 * c.max()).float().sum(1).mean().item()
            print(f"  step {step:4d} mse={mse.item():.5f} code={c.mean().item():.3f} "
                  f"nact={act:.2f} arc={arc.item():.3f}", flush=True)
    sae.topk = topk
    return sae


def _reml_polish(samp, tk, grid):
    """gamfit REML fit of clean per-atom samples samp(=g_k(tk)) at positions tk.
    Detects closed (periodic) vs open from the 2D shape and fits accordingly."""
    import gamfit
    # work in the atom's own 2D plane (denoise + topology test), map back after
    U = np.linalg.svd(samp - samp.mean(0), full_matrices=False)[2][:2]   # (2,D)
    Y2 = (samp - samp.mean(0)) @ U.T
    phi = np.arctan2(Y2[:, 1], Y2[:, 0])
    sp = np.sort(phi)
    closed = np.diff(np.concatenate([sp, [sp[0] + 2 * np.pi]])).max() < 0.6
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(tk), torch.from_numpy(samp), basis="duchon", basis_order=2,
        by=torch.from_numpy(np.ones(len(tk))), periodic=bool(closed),
        period=(1.0 if closed else None))
    fitted = np.asarray(res["fitted"])
    o = np.argsort(tk); ts, fs = tk[o], fitted[o]
    if closed:
        ts = np.concatenate([ts - 1, ts, ts + 1]); fs = np.concatenate([fs, fs, fs], 0)
    return np.stack([np.interp(grid, ts, fs[:, j]) for j in range(samp.shape[1])], 1)


def recover(X, K, M=16, grid_size=200, polish=True, gate_thresh=0.25, **kw):
    sae = fit(X, K, M=M, **kw)
    return _extract(sae, X, K, grid_size, polish, gate_thresh)


@torch.no_grad()
def _extract(sae, X, K, grid_size, polish, gate_thresh):
    grid_t = torch.linspace(0, 1, grid_size)
    raw = [sae.curve(k, grid_t).cpu().numpy() for k in range(K)]
    if not polish:
        return raw
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    xhat, c, t = sae(topk=getattr(sae, "topk", None))
    phi = duchon_basis_torch(t, sae.centers)
    contrib = torch.einsum("nk,nkm,kmd->nkd", c, phi, sae.B)             # per-atom contribution
    thr = gate_thresh * c.max().item()
    grid = np.linspace(0, 1, grid_size, endpoint=False)
    out = []
    for k in range(K):
        active = c[:, k] > thr
        amp = c[:, k][active]
        if active.sum() < 12 or amp.min() < 1e-3:
            out.append(raw[k]); continue
        resid = Xt[active] - (contrib[active].sum(1) - contrib[active, k])  # backfit-subtract others
        samp = (resid / amp[:, None]).cpu().numpy()                        # clean samples of g_k(t_k)
        tk = t[active, k].cpu().numpy()
        try:
            out.append(_reml_polish(samp, tk, grid))
        except Exception:
            out.append(raw[k])
    return out
