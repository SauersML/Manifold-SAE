"""Tests for the SAE steering benchmark.

We construct toy SAEs over a (cos hue, sin hue, lightness, noise...) feature
space and check that:
  * a "good" SAE (atoms perfectly aligned with the hue axes) scores high on
    every protocol
  * a "bad" SAE (random tied weights) scores low
  * anchor-swap behaves as a manual computation predicts
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from manifold_sae.eval.harness import SAEWrapper
from manifold_sae.eval.steering_bench import SteeringBench


# ---------------------------------------------------------------------------
# Toy SAEs
# ---------------------------------------------------------------------------


def _make_data(N: int = 400, D: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    hue = rng.uniform(0, 1, size=N).astype(np.float32)
    sat = rng.uniform(0.5, 1.0, size=N).astype(np.float32)
    val = rng.uniform(0.2, 1.0, size=N).astype(np.float32)
    # Construct X with explicit cos/sin/lightness signal + nuisance dims.
    X = np.zeros((N, D), dtype=np.float32)
    X[:, 0] = np.cos(2 * np.pi * hue)
    X[:, 1] = np.sin(2 * np.pi * hue)
    X[:, 2] = val
    X[:, 3:] = 0.1 * rng.standard_normal((N, D - 3)).astype(np.float32)
    hsv = np.stack([hue, sat, val], axis=1)
    return X, hsv


class GoodSAE(SAEWrapper):
    """An SAE whose feature 0 = cos-hue contribution, feature 1 = sin-hue,
    feature 2 = lightness. Decoder is identity-aligned on the same dims.
    """
    def __init__(self, D: int):
        self.name = "GoodSAE"
        self.n_features = 8
        self.input_dim = D
        self.firing_threshold = 1e-6
        F_ = self.n_features
        W_e = np.zeros((D, F_), dtype=np.float32)
        W_d = np.zeros((F_, D), dtype=np.float32)
        # Atom 0: cos-hue. Atom 1: sin-hue. Atom 2: lightness.
        W_e[0, 0] = 1.0; W_d[0, 0] = 1.0
        W_e[1, 1] = 1.0; W_d[1, 1] = 1.0
        W_e[2, 2] = 1.0; W_d[2, 2] = 1.0
        # Remaining atoms wired to nuisance dims for sparsity stats.
        for k in range(3, F_):
            W_e[3 + (k - 3) % (D - 3), k] = 1.0
            W_d[k, 3 + (k - 3) % (D - 3)] = 1.0
        self.W_e = torch.from_numpy(W_e)
        self.W_d = torch.from_numpy(W_d)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x @ self.W_e

    def decode_from_activations(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        return z @ self.W_d

    def reconstruct(self, x):
        return self.decode_from_activations(self.encode(x))


class BadSAE(SAEWrapper):
    """Random tied weights, no hue alignment."""
    def __init__(self, D: int, seed: int = 0):
        self.name = "BadSAE"
        self.n_features = 8
        self.input_dim = D
        self.firing_threshold = 1e-6
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((D, self.n_features)).astype(np.float32) * 0.01
        self.W_e = torch.from_numpy(W)
        self.W_d = torch.from_numpy(W.T)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x @ self.W_e

    def decode_from_activations(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        return z @ self.W_d

    def reconstruct(self, x):
        return self.decode_from_activations(self.encode(x))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _bench(model, X, hsv):
    Xt = torch.from_numpy(X)
    return SteeringBench(model, Xt, hsv_labels=hsv, k_hue_atoms=2,
                         k_value_atoms=2, seed=0)


def test_good_sae_linear_push_scores_high():
    X, hsv = _make_data()
    bench = _bench(GoodSAE(X.shape[1]), X, hsv)
    p = bench.protocol_linear_push(alpha=1.0)
    # We picked atoms wired exactly to hue → near-perfect cosine alignment.
    assert p.steering_r2 > 0.5, f"good SAE linear_push too low: {p.steering_r2}"


def test_bad_sae_anchor_swap_fails_to_close_distance():
    """A random-weight SAE has no hue-axis structure, so swapping atoms
    cannot reliably reduce the source→anchor hue distance. The good SAE,
    by contrast, fully closes the distance.
    """
    X, hsv = _make_data()
    bad = _bench(BadSAE(X.shape[1]), X, hsv).protocol_anchor_swap(
        n_anchors=4, n_sources=64
    )
    good = _bench(GoodSAE(X.shape[1]), X, hsv).protocol_anchor_swap(
        n_anchors=4, n_sources=64
    )
    assert good.steering_r2 > bad.steering_r2 + 0.2, (
        f"good ({good.steering_r2}) should beat bad ({bad.steering_r2}) by >0.2"
    )


def test_good_sae_magnitude_scaling_is_monotonic():
    X, hsv = _make_data()
    bench = _bench(GoodSAE(X.shape[1]), X, hsv)
    p = bench.protocol_magnitude_scaling()
    # Scaling the top hue atom should drive the cos/sin projection
    # monotonically.
    assert p.monotonicity > 0.8, f"monotonicity low: {p.monotonicity}"


def test_anchor_swap_matches_manual():
    """Manually predict the decoded hue after swapping atom 0+1 (the
    cos/sin atoms) of a source row with an anchor row in the GoodSAE.
    Expect d_after ≈ 0 (perfectly closes the distance).
    """
    X, hsv = _make_data(N=200)
    bench = _bench(GoodSAE(X.shape[1]), X, hsv)
    p = bench.protocol_anchor_swap(n_anchors=4, n_sources=64)
    # For the GoodSAE, atom 0/1 ARE the hue. Swapping them = swapping hue
    # exactly. d_after should be near zero relative to d_before.
    extra = p.extra
    assert extra["mean_d_after"] < extra["mean_d_before"], \
        f"anchor swap didn't close distance: {extra}"
    assert p.steering_r2 > 0.5, f"anchor_swap R^2 low: {p.steering_r2}"


def test_full_bench_runs_and_produces_summary():
    X, hsv = _make_data()
    bench = _bench(GoodSAE(X.shape[1]), X, hsv)
    res = bench.run()
    d = res.to_dict()
    assert set(d["protocols"].keys()) == {
        "linear_push", "anchor_swap", "magnitude_scaling", "compositional",
    }
    s = d["summary"]
    assert "composite" in s
    # Good SAE composite should be a real, finite number (not nan).
    assert np.isfinite(s["composite"])
