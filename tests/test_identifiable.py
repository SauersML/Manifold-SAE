"""Tests for manifold_sae.identifiable — primitives + composition."""
from __future__ import annotations

import numpy as np

from manifold_sae.identifiable import (
    PiecewiseLinearSmooth,
    abs_corr,
    conditional_prior_ivae,
    identifiable_manifold_sae,
    mechanism_sparsity_jacobian,
)


def test_mechanism_sparsity_value_matches_closed_form():
    W = np.array([[3.0, 0.0], [4.0, 0.0]])
    value, grad = mechanism_sparsity_jacobian(1.0, 1e-8, W)
    assert abs(value - 5.0) < 1e-6
    # grad on the zero column is ~0; grad on the non-zero column is unit
    assert grad.shape == W.shape
    assert abs(np.linalg.norm(grad[:, 0]) - 1.0) < 1e-6
    assert np.linalg.norm(grad[:, 1]) < 1e-4


def test_mechanism_sparsity_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 3))
    _, g = mechanism_sparsity_jacobian(0.7, 1e-6, W)
    h = 1e-5
    fd = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += h
            Wm = W.copy(); Wm[i, j] -= h
            fd[i, j] = (mechanism_sparsity_jacobian(0.7, 1e-6, Wp)[0]
                        - mechanism_sparsity_jacobian(0.7, 1e-6, Wm)[0]) / (2 * h)
    assert np.max(np.abs(g - fd)) < 1e-4


def test_conditional_prior_ivae_reduces_to_standard_gaussian():
    n, d = 5, 3
    t = np.full((n, d), 0.5)
    mean = np.zeros((n, d))
    scale = np.ones((n, d))
    value, grad = conditional_prior_ivae(1.0, t, mean, scale)
    expected_quad = 0.5 * float((t * t).sum())
    expected = expected_quad + 0.5 * n * d * float(np.log(2 * np.pi))
    assert abs(value - expected) < 1e-9
    assert np.allclose(grad, 0.5 * np.ones((n, d)))


def test_conditional_prior_ivae_grad_finite_diff():
    rng = np.random.default_rng(1)
    n, d = 3, 2
    t = rng.normal(size=(n, d))
    mean = rng.normal(size=(n, d))
    scale = np.abs(rng.normal(size=(n, d))) + 0.5
    _, g = conditional_prior_ivae(1.7, t, mean, scale)
    h = 1e-5
    fd = np.zeros_like(t)
    for i in range(n):
        for j in range(d):
            tp = t.copy(); tp[i, j] += h
            tm = t.copy(); tm[i, j] -= h
            fd[i, j] = (conditional_prior_ivae(1.7, tp, mean, scale)[0]
                        - conditional_prior_ivae(1.7, tm, mean, scale)[0]) / (2 * h)
    assert np.max(np.abs(g - fd)) < 1e-5


def test_piecewise_linear_smooth_fits_linear_function_exactly():
    rng = np.random.default_rng(2)
    aux = rng.uniform(0.0, 1.0, size=(200, 1))
    target = 3.0 * aux + 0.5  # one-dim aux, one-dim target
    smooth = PiecewiseLinearSmooth.fit_ls(aux, target, n_centres=4, ridge=1e-8)
    pred = smooth.evaluate(aux)
    assert np.max(np.abs(pred - target)) < 1e-3


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
        n_iter=40,
        weight_mech=1e-4,
        weight_ivae=2.0,
    )
    assert fit.T.shape == (n, 4)
    # Supervised axes should correlate strongly with aux
    corr_sup = abs_corr(fit.T[:, :2], aux)
    # diagonal best-match: take per-axis max correlation across aux columns
    assert corr_sup.max(axis=1).mean() > 0.70
    # Reconstruction MSE should be small
    mse = float(((X - fit.T @ fit.W.T) ** 2).mean())
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
        n_iter=30,
        weight_mech=1e-3,
    )
    assert fit.T.shape == (n, 3)
    assert fit.mean_smooth is None
