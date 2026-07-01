"""Crosscoder primitive tests (gamfit-native smoke).

The crosscoder is now a thin re-export of ``gamfit.crosscoder.Crosscoder`` (a
numpy-in, Adam-trained, Rust-backed primitive — not an ``nn.Module``). These
smoke tests exercise the gamfit surface: fit, per_layer_r2, atom_layer_affinity,
harmonic_atoms, and the not-fitted guard.

Run only this file:
    uv run pytest tests/test_crosscoder.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from manifold_sae.crosscoder import Crosscoder


def _toy_stack(seed: int = 0, N: int = 256) -> list[np.ndarray]:
    """A synthetic 3-layer stack with shared latent structure.

    True generative model: 8 latent atoms z* ~ Exp(1) gated by Bernoulli(0.2).
    Each layer maps z* through a random per-layer decoder + Gaussian noise.
    Rows are aligned across layers — that alignment is the cross-layer signal.
    """
    rng = np.random.default_rng(seed)
    F_true = 8
    z = rng.exponential(1.0, size=(N, F_true)) * (rng.random((N, F_true)) < 0.2)
    stack: list[np.ndarray] = []
    for D in (16, 24, 32):
        W = rng.standard_normal((F_true, D)) / np.sqrt(F_true)
        stack.append(z @ W + 0.05 * rng.standard_normal((N, D)))
    return stack


def test_fit_and_per_layer_r2() -> None:
    """Fit converges to high per-layer R^2 on clean shared-latent data."""
    stack = _toy_stack()
    cc = Crosscoder(
        layer_dims=[16, 24, 32],
        n_atoms=32,
        l1_weight=1e-4,
        shared_encoder="linear",
    )
    ret = cc.fit(stack, epochs=150, lr=5e-3, seed=0)
    assert ret is cc  # .fit returns self

    r2 = cc.per_layer_r2()
    assert isinstance(r2, np.ndarray)
    assert r2.shape == (3,)
    assert all(r > 0.3 for r in r2), f"R^2 too low across layers: {r2}"

    # Diagnostics record a per-epoch loss curve that decreases overall.
    diag = cc.diagnostics
    assert diag.losses.shape == (150,)
    assert diag.losses[-1] < diag.losses[0]


def test_atom_layer_affinity_and_harmonic_atoms() -> None:
    """Affinity is (F, L) row-max normalised; harmonic_atoms returns valid indices."""
    stack = _toy_stack(seed=1)
    cc = Crosscoder(layer_dims=[16, 24, 32], n_atoms=16, l1_weight=1e-4)
    cc.fit(stack, epochs=40, lr=5e-3, seed=1)

    aff = cc.atom_layer_affinity()
    assert aff.shape == (16, 3)
    row_max = aff.max(axis=1)
    active = row_max > 0.0
    # Row-max normalisation: every active atom's strongest layer has affinity 1.0.
    assert np.allclose(row_max[active], 1.0, atol=1e-6)
    assert (aff >= 0.0).all() and (aff <= 1.0 + 1e-6).all()

    harm = cc.harmonic_atoms(tol=0.05)
    assert harm.ndim == 1
    assert set(harm.tolist()).issubset(set(range(16)))
    # tol=1.0 keeps only atoms uniform (== max) in every layer; a subset of tol=0.
    assert set(cc.harmonic_atoms(tol=1.0).tolist()).issubset(set(harm.tolist()))


def test_requires_fit_before_diagnostics() -> None:
    """Diagnostic accessors raise before .fit() is called."""
    cc = Crosscoder(layer_dims=[8, 8], n_atoms=4, shared_encoder="linear")
    with pytest.raises(RuntimeError):
        cc.per_layer_r2()
    with pytest.raises(RuntimeError):
        cc.atom_layer_affinity()


def test_input_validation() -> None:
    """Constructor and fit reject malformed shapes."""
    with pytest.raises(ValueError):
        Crosscoder(layer_dims=[], n_atoms=4)
    with pytest.raises(ValueError):
        Crosscoder(layer_dims=[8, 8], n_atoms=0)
    cc = Crosscoder(layer_dims=[16, 24, 32], n_atoms=8, shared_encoder="linear")
    # Wrong number of layers in the stack.
    with pytest.raises(ValueError):
        cc.fit([np.zeros((10, 16)), np.zeros((10, 24))], epochs=1)
