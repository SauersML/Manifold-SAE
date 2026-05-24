"""Minimal entropic Sinkhorn + free-support Wasserstein barycenter on S^1.

Used by `manifold_sae.wasserstein_sae`. Falls back to POT (`import ot`) if
available for cross-validation in tests; the production path uses the pure
PyTorch implementation below so gradients flow cleanly under MPS.

Conventions
-----------
* All distributions are length-`M` probability simplex vectors (rows sum to 1).
* Cost matrices are (M, M) — for the circular hue support we use the squared
  circular distance ``min(|i-j|, M-|i-j|)^2 / (M/2)^2`` so values live in
  [0, 1].
* Entropic regularization ε is a scalar; smaller ε → sharper transport plan
  but slower convergence + numerically stiffer.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def circular_cost_matrix(M: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Squared circular distance on the M-point hue circle, normalized to [0, 1]."""
    idx = torch.arange(M, device=device, dtype=dtype)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    d = torch.minimum(d, M - d)
    C = (d / (M / 2.0)) ** 2
    return C


def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.01,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Log-domain Sinkhorn — returns (B, M, M) transport plan(s).

    `a`, `b` may be (M,) or (B, M); broadcasting follows the leading dim.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    Bsz = max(a.shape[0], b.shape[0])
    a = a.expand(Bsz, -1)
    b = b.expand(Bsz, -1)
    dtype = a.dtype
    b = b.to(dtype)
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    K = -C.to(dtype) / eps  # (M, M) log-kernel
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iter):
        # f_i = log_a_i - logsumexp_j ( K_ij + g_j )
        f_new = log_a - torch.logsumexp(K.unsqueeze(0) + g.unsqueeze(1), dim=2)
        g_new = log_b - torch.logsumexp(K.unsqueeze(0) + f_new.unsqueeze(2), dim=1)
        if torch.max((f_new - f).abs()).item() < tol and torch.max((g_new - g).abs()).item() < tol:
            f, g = f_new, g_new
            break
        f, g = f_new, g_new
    log_T = f.unsqueeze(2) + g.unsqueeze(1) + K.unsqueeze(0)
    return log_T.exp()


def sinkhorn_barycenter(
    atoms: torch.Tensor,           # (F, M)  — F dictionary distributions
    weights: torch.Tensor,         # (B, F)  — simplex weights per batch row
    C: torch.Tensor,               # (M, M)
    eps: float = 0.01,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Entropic Wasserstein barycenter (free-mass-balanced, fixed support).

    Implements the IBP iteration of Benamou et al. 2015 / Schmitz et al. 2018:
        u_k ← a_k / (K  v_k)
        b   ← Π_k (K^T u_k)^{w_k}
        v_k ← b / (K^T u_k)
    Returns (B, M).

    Implementation note: we use direct (exp-domain) matmuls against the kernel
    K = exp(-C / eps). The fully-log-domain alternative needs a (B, F, M, M)
    intermediate (one outer-sum per `logsumexp`) which OOMs on MPS at B=256
    F=128 M=64. We keep u and v in normalized form to control overflow when
    eps is small.
    """
    Bsz, F = weights.shape
    M = atoms.shape[1]
    assert atoms.shape == (F, M)
    # Unify dtype on `atoms.dtype` — under torch.set_default_dtype(float64)
    # (used by the test conftest) bare `torch.tensor(...)` produces float64
    # weights, which would clash with float32 atoms / C on the next matmul.
    dtype = atoms.dtype
    K = torch.exp(-C.to(dtype) / eps)                                        # (M, M)
    a_bf = atoms.unsqueeze(0).expand(Bsz, F, M).contiguous()                 # (B, F, M)
    w = weights.to(dtype)                                                    # (B, F)

    v = torch.ones(Bsz, F, M, device=atoms.device, dtype=atoms.dtype) / M
    bary = torch.ones(Bsz, M, device=atoms.device, dtype=atoms.dtype) / M
    prev = bary.clone()
    for _ in range(n_iter):
        # K @ v over the last dim — (B, F, M) = (B, F, M) @ (M, M) -- matmul
        Kv = v @ K                                                           # (B, F, M)
        u = a_bf / Kv.clamp(min=1e-30)                                       # (B, F, M)
        Ktu = u @ K.t()                                                      # (B, F, M)
        # b_j = Π_k Ktu[k, j]^{w_k}  ↔  log b_j = Σ_k w_k · log Ktu[k, j]
        log_Ktu = torch.log(Ktu.clamp(min=1e-30))
        log_bary = (w.unsqueeze(2) * log_Ktu).sum(dim=1)                     # (B, M)
        # Renormalize barycenter to a probability vector each iteration.
        # Without this, near-zero-weight atoms make log_Ktu drift unboundedly.
        log_bary = log_bary - torch.logsumexp(log_bary, dim=-1, keepdim=True)
        bary = torch.exp(log_bary)
        v = bary.unsqueeze(1) / Ktu.clamp(min=1e-30)                         # (B, F, M)
        if torch.max((bary - prev).abs()).item() < tol:
            break
        prev = bary
    return bary
