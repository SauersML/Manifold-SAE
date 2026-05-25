"""Identifiable manifold-SAE: iVAE + mechanism-sparsity composition.

Routes value/grad through the gamfit 0.1.123 Rust primitives:

  * ``gamfit._rust.mechanism_sparsity_jacobian`` — Lachapelle 2401.04890
    column-2-norm group lasso on the decoder W.
  * ``gamfit._rust.conditional_prior_ivae``      — Khemakhem 2107.10098
    per-row Gaussian log-prior on the supervised latent block.

These are the value/grad-exposing siblings of the new dataclass-style
descriptors :class:`gamfit.AuxConditionalPriorPenalty`,
:class:`gamfit.MechanismSparsityPenalty`, and
:class:`gamfit.IvaeRidgeMeanGauge`. Those descriptors configure
``sae_manifold_fit`` (topology-constrained atoms over Gumbel/IBP); they do
NOT expose a callable suitable for the (T_sup | T_free) block-coordinate
solver this module needs. See the migration notes in
``project_ivae_mechsparsity_cogito`` for the API gap that would let us
collapse this whole file into a single ``sae_manifold_fit(..., penalty=[...])``
call.

Outer loop: block-coordinate alternation of

  * T (closed-form per-row weighted-LS on (T_sup | T_free))
  * W (diagonal-Newton gradient step with mechanism-sparsity prox-grad)
  * μ(u), σ(u) iVAE smooths refit by ridge LS every ``smooth_refit_every``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from gamfit import _rust as _RUST  # type: ignore[attr-defined]


def mechanism_sparsity_jacobian(weight: float, epsilon: float, w: np.ndarray):
    """`weight · Σ_k √(||W[:,k]||² + ε²) − ε` value+grad (gamfit Rust)."""
    return _RUST.mechanism_sparsity_jacobian(weight, epsilon, np.ascontiguousarray(w, dtype=np.float64))


def conditional_prior_ivae(weight: float, t: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    """Per-row Gaussian log-prior value+grad_t (gamfit Rust)."""
    return _RUST.conditional_prior_ivae(
        weight,
        np.ascontiguousarray(t, dtype=np.float64),
        np.ascontiguousarray(mean, dtype=np.float64),
        np.ascontiguousarray(scale, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Piecewise-linear smooth μ(u), σ(u)
# ---------------------------------------------------------------------------

@dataclass
class PiecewiseLinearSmooth:
    """Separable per-aux-column piecewise-linear additive map.

    gamfit's ``ParametricAuxConditionalPriorPenalty`` would replace this once
    a Python-callable evaluator is exposed; today it only runs inside the
    Rust ``sae_manifold_fit`` outer loop.
    """

    aux_min: np.ndarray
    aux_max: np.ndarray
    coeffs_per_aux: list
    bias: np.ndarray

    def evaluate(self, aux: np.ndarray) -> np.ndarray:
        n, aux_dim = aux.shape
        latent_dim = self.bias.shape[0]
        out = np.broadcast_to(self.bias, (n, latent_dim)).copy()
        for j in range(aux_dim):
            coeffs = self.coeffs_per_aux[j]
            k = coeffs.shape[0]
            step = (self.aux_max[j] - self.aux_min[j]) / max(k - 1, 1)
            if step <= 0:
                continue
            pos = np.clip((aux[:, j] - self.aux_min[j]) / step, 0.0, k - 1 - 1e-12)
            lo = np.floor(pos).astype(np.int64)
            frac = pos - lo
            hi = np.minimum(lo + 1, k - 1)
            out += coeffs[lo] * (1.0 - frac[:, None]) + coeffs[hi] * frac[:, None]
        return out

    @classmethod
    def fit_ls(cls, aux: np.ndarray, target: np.ndarray, n_centres: int = 6,
               ridge: float = 1.0e-3) -> "PiecewiseLinearSmooth":
        n, aux_dim = aux.shape
        aux_min = aux.min(axis=0)
        aux_max = aux.max(axis=0)
        design_blocks = [np.ones((n, 1))]
        for j in range(aux_dim):
            step = (aux_max[j] - aux_min[j]) / max(n_centres - 1, 1)
            block = np.zeros((n, n_centres))
            if step > 0:
                pos = np.clip((aux[:, j] - aux_min[j]) / step, 0.0, n_centres - 1 - 1e-12)
                lo = np.floor(pos).astype(np.int64)
                frac = pos - lo
                hi = np.minimum(lo + 1, n_centres - 1)
                rows = np.arange(n)
                block[rows, lo] += (1.0 - frac)
                block[rows, hi] += frac
            design_blocks.append(block)
        design = np.concatenate(design_blocks, axis=1)
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design.T @ target)
        bias = coef[0]
        coeffs_per_aux = []
        offset = 1
        for j in range(aux_dim):
            coeffs_per_aux.append(coef[offset:offset + n_centres])
            offset += n_centres
        return cls(aux_min=aux_min, aux_max=aux_max, coeffs_per_aux=coeffs_per_aux, bias=bias)


# ---------------------------------------------------------------------------
# Composition: one-shot identifiable manifold SAE
# ---------------------------------------------------------------------------

@dataclass
class IdentifiableFit:
    W: np.ndarray
    T: np.ndarray
    mean_smooth: Optional[PiecewiseLinearSmooth]
    scale_smooth: Optional[PiecewiseLinearSmooth]
    losses: list = field(default_factory=list)
    used_rust: bool = True  # always True post-0.1.123
    n_supervised: int = 0
    n_free: int = 0


def identifiable_manifold_sae(
    X: np.ndarray,
    aux_hsv: Optional[np.ndarray],
    n_supervised: int = 3,
    n_free: int = 3,
    weight_recon: float = 1.0,
    weight_ivae: float = 1.0,
    weight_free_prior: float = 1.0e-2,
    weight_mech: float = 1.0e-2,
    epsilon_mech: float = 1.0e-6,
    n_centres: int = 6,
    n_iter: int = 200,
    smooth_refit_every: int = 5,
    sigma_floor: float = 0.2,
    seed: int = 0,
) -> IdentifiableFit:
    """Fit ``X ≈ T @ W^⊤`` under iVAE prior on T_sup + mechanism-sparsity on W."""
    n, D = X.shape
    n_total = n_supervised + n_free
    if n_total <= 0:
        raise ValueError("n_supervised + n_free must be > 0")

    U, S, Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    W = Vt[:n_total].T.copy() * S[:n_total][None, :] / max(np.sqrt(n - 1), 1.0)
    Xc = X - X.mean(0, keepdims=True)
    T = Xc @ np.linalg.pinv(W).T

    mean_smooth = scale_smooth = None
    if n_supervised > 0 and aux_hsv is not None:
        mean_smooth = PiecewiseLinearSmooth.fit_ls(aux_hsv, T[:, :n_supervised], n_centres=n_centres)
        resid = T[:, :n_supervised] - mean_smooth.evaluate(aux_hsv)
        scale_init = np.maximum(resid.std(axis=0), sigma_floor)
        scale_smooth = PiecewiseLinearSmooth(
            aux_min=aux_hsv.min(0), aux_max=aux_hsv.max(0),
            coeffs_per_aux=[np.zeros((n_centres, n_supervised)) for _ in range(aux_hsv.shape[1])],
            bias=scale_init,
        )

    losses: list = []

    for it in range(n_iter):
        WTW = W.T @ W
        rhs = Xc @ W
        if n_supervised > 0 and aux_hsv is not None and mean_smooth is not None:
            mu = mean_smooth.evaluate(aux_hsv)
            sigma = np.maximum(scale_smooth.evaluate(aux_hsv), sigma_floor)
            inv_var = weight_ivae / (sigma * sigma)
            T_new = np.zeros_like(T)
            for r in range(n):
                A = weight_recon * WTW.copy()
                b = weight_recon * rhs[r].copy()
                for i in range(n_supervised):
                    A[i, i] += inv_var[r, i]
                    b[i] += inv_var[r, i] * mu[r, i]
                for i in range(n_supervised, n_total):
                    A[i, i] += weight_free_prior
                T_new[r] = np.linalg.solve(A, b)
            T = T_new
        else:
            A = weight_recon * WTW + weight_free_prior * np.eye(n_total)
            T = (rhs * weight_recon) @ np.linalg.inv(A)

        recon = T @ W.T
        grad_W_recon = weight_recon * (recon - Xc).T @ T
        _, grad_W_mech = mechanism_sparsity_jacobian(weight_mech, epsilon_mech, W)
        col_norm_sq = (W * W).sum(axis=0) + epsilon_mech * epsilon_mech
        L_recon = weight_recon * float(np.linalg.eigvalsh(T.T @ T).max())
        L_mech = weight_mech / np.sqrt(col_norm_sq).min()
        step = 1.0 / (L_recon + L_mech + 1e-8)
        W = W - step * (grad_W_recon + grad_W_mech)

        if n_supervised > 0 and aux_hsv is not None and (it % smooth_refit_every == 0):
            mean_smooth = PiecewiseLinearSmooth.fit_ls(aux_hsv, T[:, :n_supervised], n_centres=n_centres)
            resid = T[:, :n_supervised] - mean_smooth.evaluate(aux_hsv)
            sigma_emp = np.maximum(np.abs(resid).std(axis=0), sigma_floor)
            scale_smooth = PiecewiseLinearSmooth(
                aux_min=aux_hsv.min(0), aux_max=aux_hsv.max(0),
                coeffs_per_aux=[np.zeros((n_centres, n_supervised)) for _ in range(aux_hsv.shape[1])],
                bias=sigma_emp,
            )

        recon_loss = 0.5 * float(((Xc - T @ W.T) ** 2).sum())
        ivae_loss = 0.0
        if n_supervised > 0 and aux_hsv is not None and mean_smooth is not None:
            mu = mean_smooth.evaluate(aux_hsv)
            sigma = np.maximum(scale_smooth.evaluate(aux_hsv), sigma_floor)
            ivae_loss = conditional_prior_ivae(weight_ivae, T[:, :n_supervised], mu, sigma)[0]
        mech_loss = mechanism_sparsity_jacobian(weight_mech, epsilon_mech, W)[0]
        free_prior_loss = 0.5 * weight_free_prior * float((T[:, n_supervised:] ** 2).sum())
        total = recon_loss + ivae_loss + mech_loss + free_prior_loss
        losses.append({"iter": it, "total": total, "recon": recon_loss,
                       "ivae": ivae_loss, "mech": mech_loss, "free_prior": free_prior_loss})

    return IdentifiableFit(W=W, T=T, mean_smooth=mean_smooth, scale_smooth=scale_smooth,
                           losses=losses, used_rust=True,
                           n_supervised=n_supervised, n_free=n_free)


def abs_corr(T: np.ndarray, aux: np.ndarray) -> np.ndarray:
    Tc = T - T.mean(0, keepdims=True)
    Ac = aux - aux.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    An = Ac / (Ac.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ An / Tn.shape[0])
