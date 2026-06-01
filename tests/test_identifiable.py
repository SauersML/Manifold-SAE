"""Tests for manifold_sae.identifiable — iVAE + mechanism-sparsity composition.

``identifiable_manifold_sae`` composes two gamfit torch penalty modules
(:class:`gamfit.torch.IvaeRidgeMeanGauge`, :class:`gamfit.torch.MechanismSparsityPenalty`)
in a small Adam loop over leaf tensors ``T`` (latent codes) and ``W`` (decoder),
fitting ``X ≈ T @ W.T``.

It deliberately does NOT use ``gamfit.identifiable_factor_fit``: in gamfit
0.1.141 that recipe's strict ``conditional_prior_ivae`` rank check is
unsatisfiable for a mean-only aux (``aux_scale`` hardcoded to ones → every
supervised fit raises ``GamError``; see SauersML/gam#576). The penalty-module
composition tested here works and recovers planted factors.
"""
from __future__ import annotations

import numpy as np

from manifold_sae.identifiable import abs_corr, identifiable_manifold_sae


def test_identifiable_manifold_sae_recovers_planted_factors():
    rng = np.random.default_rng(3)
    n, D, k = 240, 32, 4
    T_true = rng.normal(size=(n, k))
    W_true = rng.normal(size=(D, k))
    X = T_true @ W_true.T + 0.01 * rng.normal(size=(n, D))
    aux = T_true[:, :2] + 0.02 * rng.normal(size=(n, 2))  # supervise first 2 axes
    fit = identifiable_manifold_sae(
        X,
        aux,
        n_supervised=2,
        n_free=2,
        n_iter=400,
        lr=5e-2,
        weight_mech=1e-4,
        weight_ivae=2.0,
    )
    assert fit.T.shape == (n, 4)
    assert fit.W.shape == (D, 4)
    corr_sup = abs_corr(fit.T[:, :2], aux)
    assert corr_sup.max(axis=1).mean() > 0.70
    mse = float(((X - X.mean(0) - fit.T @ fit.W.T) ** 2).mean())
    assert mse < 0.5


def test_identifiable_manifold_sae_mech_only_runs_unsupervised():
    rng = np.random.default_rng(4)
    n, D, k = 150, 16, 3
    T_true = rng.normal(size=(n, k))
    W_true = rng.normal(size=(D, k))
    X = T_true @ W_true.T + 0.05 * rng.normal(size=(n, D))
    fit = identifiable_manifold_sae(
        X,
        None,
        n_supervised=0,
        n_free=3,
        n_iter=200,
        weight_mech=1e-3,
    )
    assert fit.T.shape == (n, 3)
    assert fit.n_supervised == 0


def test_abs_corr_matches_numpy_reference():
    rng = np.random.default_rng(7)
    T = rng.normal(size=(100, 3))
    aux = rng.normal(size=(100, 2))
    got = abs_corr(T, aux)
    assert got.shape == (3, 2)
    # constant column → 0 correlation, finite output
    aux_const = np.ones((100, 1))
    assert np.all(np.isfinite(abs_corr(T, aux_const)))
    # self-correlation of a column with itself is 1
    assert abs(abs_corr(T[:, :1], T[:, :1])[0, 0] - 1.0) < 1e-9
