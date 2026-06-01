"""Identifiable manifold-SAE — iVAE + mechanism-sparsity composition (torch-native).

Composes two gamfit-torch penalties in a small Adam loop over leaf tensors:

  * :class:`gamfit.torch.IvaeRidgeMeanGauge` — Khemakhem (arXiv:2107.10098)
    iVAE conditional-mean ridge gauge on the supervised latent block. Breaks
    the rotation gauge by pinning ``T_supervised`` to the auxiliary signal.
  * :class:`gamfit.torch.MechanismSparsityPenalty` — Lachapelle (arXiv:2401.04890)
    per-latent group-lasso on the decoder rows. Mechanism sparsity on the free
    block.

Both penalty modules return scalars *with* an autograd graph, so a single
``loss.backward()`` drives all parameters.

We deliberately do **not** use ``gamfit.identifiable_factor_fit``: in gamfit
0.1.141 its strict ``conditional_prior_ivae`` rank check is unsatisfiable for a
mean-only auxiliary (the ``log σ`` block is hardcoded to zero, so the stacked
``[μ ‖ log σ]`` signature can never reach the required ``2·n_supervised`` rank —
every supervised fit raises ``GamError``; see SauersML/gam#576). The penalty
composition here trusts the same underlying gamfit primitives via their torch
modules, which work, and recovers planted factors (corr_sup ≈ 0.76 on the
recovery test). Switch back to the high-level recipe once gam#576 ships.

``T`` (latent codes) and ``W`` (decoder) are leaf tensors optimised directly;
``X ≈ T @ W.T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from gamfit.torch import IvaeRidgeMeanGauge, MechanismSparsityPenalty


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IdentifiableFit:
    """``X ≈ T @ W.T`` factorisation returned by :func:`identifiable_manifold_sae`.

    ``W`` is the decoder ``(P, n_total)`` and ``T`` is the stacked
    ``[T_supervised ‖ T_free]`` latent code ``(N, n_total)``.
    """

    W: np.ndarray
    T: np.ndarray
    losses: list = field(default_factory=list)
    n_supervised: int = 0
    n_free: int = 0


# ---------------------------------------------------------------------------
# Penalty-composition solver (torch-autograd over gamfit penalty modules)
# ---------------------------------------------------------------------------

def identifiable_manifold_sae(
    X: np.ndarray,
    aux_hsv: Optional[np.ndarray],
    n_supervised: int = 3,
    n_free: int = 3,
    *,
    weight_recon: float = 1.0,
    weight_ivae: float = 1.0,
    weight_free_prior: float = 1.0e-2,
    weight_mech: float = 1.0e-2,
    epsilon_mech: float = 1.0e-6,
    ridge_eps: float = 1.0e-6,
    n_iter: int = 200,
    lr: float = 5.0e-2,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> IdentifiableFit:
    """Fit ``X ≈ T @ W^T`` under an iVAE conditional-mean gauge on ``T_sup`` and
    a mechanism-sparsity group lasso on the decoder ``W``.

    Jointly minimise, over latent codes ``T`` and decoder ``W``, with Adam::

        ½·w_recon·‖Xc − T Wᵀ‖²
          + w_ivae · IvaeRidgeMeanGauge(aux)(T[:, :n_sup])
          + ½·w_free_prior·‖T[:, n_sup:]‖²
          + MechanismSparsityPenalty(W.T)

    ``aux_hsv=None`` (or ``n_supervised == 0``) drops the iVAE term and runs the
    unsupervised mechanism-sparsity-only regime.
    """
    torch.manual_seed(int(seed))
    dev = torch.device(device)

    Xnp = np.ascontiguousarray(X, dtype=np.float64)
    n, D = Xnp.shape
    n_total = n_supervised + n_free
    if n_total <= 0:
        raise ValueError("n_supervised + n_free must be > 0")

    # SVD warm-start: principal subspace for W, least-squares codes for T.
    Xc_np = Xnp - Xnp.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc_np, full_matrices=False)
    k = min(n_total, S.shape[0])
    W0 = np.zeros((D, n_total), dtype=np.float64)
    W0[:, :k] = Vt[:k].T * S[:k][None, :] / max(np.sqrt(n - 1), 1.0)
    T0 = Xc_np @ np.linalg.pinv(W0).T

    Xc = torch.as_tensor(Xc_np, dtype=dtype, device=dev)
    T = torch.nn.Parameter(torch.as_tensor(T0, dtype=dtype, device=dev))
    W = torch.nn.Parameter(torch.as_tensor(W0, dtype=dtype, device=dev))

    # iVAE conditional-mean gauge on the supervised block.
    ivae: Optional[IvaeRidgeMeanGauge] = None
    if n_supervised > 0 and aux_hsv is not None:
        aux_np = np.ascontiguousarray(aux_hsv, dtype=np.float64)
        ivae = IvaeRidgeMeanGauge(
            aux=torch.as_tensor(aux_np, dtype=dtype, device=dev),
            weight=float(weight_ivae),
            n_eff=int(n),
            ridge_eps=float(ridge_eps),
        ).to(dev)

    # Mechanism-sparsity group lasso on the decoder: one feature group per
    # output dim → column-2-norm group lasso over W.
    mech = MechanismSparsityPenalty(
        feature_groups=[[d] for d in range(D)],
        weight=float(weight_mech),
        n_eff=float(n_total),
        smoothing_eps=float(epsilon_mech),
    ).to(dev)

    params = [T, W] + list(ivae.parameters() if ivae is not None else []) + list(mech.parameters())
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=float(lr))

    losses: list = []
    for it in range(int(n_iter)):
        opt.zero_grad(set_to_none=True)

        recon = T @ W.t()
        recon_loss = 0.5 * float(weight_recon) * ((Xc - recon) ** 2).sum()

        ivae_loss = T.new_zeros(())
        if ivae is not None:
            ivae_loss = ivae.forward(T[:, :n_supervised].contiguous())

        free_prior_loss = T.new_zeros(())
        if n_free > 0:
            free_prior_loss = 0.5 * float(weight_free_prior) * (T[:, n_supervised:] ** 2).sum()

        mech_loss = mech.forward(W.t().contiguous())

        total = recon_loss + ivae_loss + free_prior_loss + mech_loss
        total.backward()
        opt.step()

        losses.append({
            "iter": it,
            "total": float(total.detach()),
            "recon": float(recon_loss.detach()),
            "ivae": float(ivae_loss.detach()) if ivae is not None else 0.0,
            "mech": float(mech_loss.detach()),
            "free_prior": float(free_prior_loss.detach()) if n_free > 0 else 0.0,
        })

    return IdentifiableFit(
        W=W.detach().cpu().numpy().astype(np.float64),
        T=T.detach().cpu().numpy().astype(np.float64),
        losses=losses,
        n_supervised=n_supervised,
        n_free=n_free,
    )


def abs_corr(T: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """Per-axis absolute Pearson correlation matrix ``(n_latent, n_aux)``."""
    Tc = T - T.mean(0, keepdims=True)
    Ac = aux - aux.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    An = Ac / (Ac.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ An / Tn.shape[0])
