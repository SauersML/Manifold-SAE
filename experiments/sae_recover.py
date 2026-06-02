"""Manifold-SAE by gradient descent — no clustering, no kNN, no graph.

The model IS the recovery: x = sum_k w_k(n) * a_k(n) * g_k(t_k(n)), where
  - g_k : [0,1] -> R^D is a learnable smooth curve (thin-plate / Duchon spline,
    differentiable in torch),
  - per token n, the latents (position t, amplitude a, soft assignment w over the
    K atoms) are FREE variables optimized directly (non-amortized),
and everything — curve coefficients and all per-token latents — is optimized
jointly by Adam to minimize reconstruction error + a sparsity prior on the
assignment. Positions are free per token, so they settle to each point's place on
its curve (no position collapse, the failure mode of the amortized encoder).

Open AND closed curves are handled by the same open spline basis: a closed curve
is simply one whose ends meet (g(0)=g(1)), which the fit discovers from the data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def duchon_basis_torch(t, centers):
    """1D Duchon m=2 thin-plate basis, differentiable in t.
    phi(t) = [ |t-c_j|^3 ]_j  ++  [1, t]   (RBF block + linear nullspace)."""
    r = (t[..., None] - centers).abs()                 # (..., M)
    rbf = r ** 3
    ones = torch.ones_like(t)[..., None]
    return torch.cat([rbf, ones, t[..., None]], dim=-1)  # (..., M+2)


class ManifoldSAE(torch.nn.Module):
    def __init__(self, N, K, D, M=18):
        super().__init__()
        self.K, self.D, self.M = K, D, M
        self.register_buffer("centers", torch.linspace(0, 1, M))
        # decoder: per-atom spline coefficients (K, M+2, D)
        self.B = torch.nn.Parameter(0.05 * torch.randn(K, M + 2, D))
        # free per-token latents
        self.t_raw = torch.nn.Parameter(torch.randn(N, K))          # -> sigmoid -> [0,1]
        self.a_raw = torch.nn.Parameter(torch.zeros(N, K))          # -> softplus -> amp
        self.w_raw = torch.nn.Parameter(0.01 * torch.randn(N, K))   # -> softmax gate

    def curves(self, grid):
        phi = duchon_basis_torch(grid, self.centers)               # (G, M+2)
        return torch.einsum("gm,kmd->kgd", phi, self.B)            # (K, G, D)

    def forward(self, tau):
        t = torch.sigmoid(self.t_raw)                              # (N,K)
        a = F.softplus(self.a_raw)
        w = F.softmax(self.w_raw / tau, dim=1)                     # soft assignment
        phi = duchon_basis_torch(t, self.centers)                 # (N,K,M+2)
        g = torch.einsum("nkm,kmd->nkd", phi, self.B)             # atom curve at each t
        xhat = torch.einsum("nk,nkd->nd", w * a, g)
        return xhat, w, a, t


def fit_sae(X, K, M=18, n_steps=3000, lr=1e-2, sparsity=1e-2, smooth=1e-4,
            ortho_w=0.0, tau0=1.0, tau1=0.15, seed=0, log_every=0):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    N, D = Xt.shape
    sae = ManifoldSAE(N, K, D, M=M)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rbf = M
    for step in range(n_steps):
        tau = tau0 * (tau1 / tau0) ** (step / max(n_steps - 1, 1))
        opt.zero_grad()
        xhat, w, a, t = sae(tau)
        mse = ((xhat - Xt) ** 2).mean()
        # sparsity: each token should use ~one atom (entropy of the gate)
        ent = -(w * (w + 1e-9).log()).sum(1).mean()
        # smoothness: penalize the RBF (curvature) coefficients
        sm = (sae.B[:, :rbf, :] ** 2).mean()
        # identifiability: atoms should occupy DISTINCT ambient subspaces. With
        # B_k (Mb,D), the ambient overlap between atoms k,j is ||B_k B_j^T||_F^2,
        # which is 0 iff their row-spaces are orthogonal. Penalizing it breaks
        # the reconstruction degeneracy (a circle re-expressed as two crossed
        # lines from different atoms) without assuming any shape.
        Bk = sae.B.reshape(sae.K, -1, D)
        Gram = torch.einsum("kmd,jnd->kjmn", Bk, Bk)         # (K,K,Mb,Mb)
        ov = (Gram ** 2).sum((2, 3))
        ortho = (ov.sum() - torch.diagonal(ov).sum()) / max(sae.K * (sae.K - 1), 1)
        loss = mse + sparsity * ent + smooth * sm + ortho_w * ortho
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step:4d} mse={mse.item():.5f} ent={ent.item():.3f} tau={tau:.2f}", flush=True)
    return sae


@torch.no_grad()
def extract_curves(sae, grid_size=200):
    grid = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    return sae.curves(grid).cpu().numpy()                          # (K, grid, D)
