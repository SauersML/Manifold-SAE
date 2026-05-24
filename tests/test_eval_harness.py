"""Tests for the SAE evaluation harness.

Run with:
    pytest tests/test_eval_harness.py -q
"""
from __future__ import annotations

import numpy as np
import torch

from manifold_sae.eval.harness import (
    Harness,
    HarnessLabels,
    SAEWrapper,
    _gini,
    _effective_rank,
    sparsity_stats,
    reconstruction_r2,
    feature_absorption,
)
from manifold_sae.eval import baselines as bl


# ---------------------------------------------------------------------------
# Synthetic models with known structure
# ---------------------------------------------------------------------------


class PerfectIdentityWrapper(SAEWrapper):
    """Encoder = decoder = identity, so reconstruction R^2 should be 1.0."""
    def __init__(self, d):
        self.name = "PerfectId"
        self.input_dim = d
        self.n_features = d
        self.firing_threshold = 1e-6

    def encode(self, x):
        return x.clone()

    def decode_from_activations(self, z):
        return z

    def reconstruct(self, x):
        return x.clone()


class ZeroWrapper(SAEWrapper):
    """Returns zero. R^2 should be 0 if data has been centered."""
    def __init__(self, d):
        self.name = "Zero"
        self.input_dim = d
        self.n_features = d
        self.firing_threshold = 1e-6

    def encode(self, x):
        return torch.zeros_like(x)

    def decode_from_activations(self, z):
        return torch.zeros_like(z)

    def reconstruct(self, x):
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reconstruction_r2_perfect_and_zero():
    torch.manual_seed(0)
    X = torch.randn(200, 16)
    X_c = X - X.mean(0, keepdim=True)
    r2 = reconstruction_r2(PerfectIdentityWrapper(16), X_c)
    assert r2 > 0.999
    r2_zero = reconstruction_r2(ZeroWrapper(16), X_c)
    # Zero against centered data -> sse == sst -> R^2 == 0.
    assert abs(r2_zero) < 1e-4


def test_gini_increases_with_sparsity():
    uniform = np.ones(100)
    g_uniform = _gini(uniform)
    sparse = np.zeros(100)
    sparse[0] = 100.0
    g_sparse = _gini(sparse)
    assert g_uniform < 0.05  # near 0 for fully uniform
    assert g_sparse > 0.95   # near 1 for one-hot


def test_effective_rank_distinguishes_curve_vs_scatter():
    # 1D curve: points along a line in 3D.
    t = np.linspace(0, 1, 50)
    line = np.stack([t, 2 * t, -t], 1)
    rank_line = _effective_rank(line)
    # Scattered random points in 3D.
    rng = np.random.default_rng(0)
    scatter = rng.standard_normal((50, 3))
    rank_scatter = _effective_rank(scatter)
    assert rank_line < 1.3, f"line should be ~1, got {rank_line}"
    assert rank_scatter > 2.0, f"scatter should be >2, got {rank_scatter}"


def test_sparsity_stats_count_correctness():
    acts = np.zeros((10, 5))
    acts[:, 0] = 1.0   # one always-on atom
    s = sparsity_stats(acts, threshold=0.1)
    assert abs(s["L0"] - 1.0) < 1e-6
    assert abs(s["mean_active_fraction"] - 0.2) < 1e-6
    # Gini high because 1 out of 5 atoms carries all activity.
    assert s["gini"] > 0.5


def test_pca_baseline_within_expected_range():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Data with rank ~ 8 in dim 20.
    U = rng.standard_normal((500, 8)); V = rng.standard_normal((8, 20))
    X = (U @ V).astype(np.float32)
    pca = bl.pca_baseline(X, n_features=8, name="PCA-8")
    X_t = torch.from_numpy(X - X.mean(0))
    r2 = reconstruction_r2(pca, X_t)
    assert r2 > 0.95, f"PCA-8 should explain ≥95% of rank-8 data, got {r2}"

    pca_low = bl.pca_baseline(X, n_features=2, name="PCA-2")
    r2_low = reconstruction_r2(pca_low, X_t)
    assert r2_low < r2, "PCA-2 should be worse than PCA-8"


def test_feature_absorption_detects_sibling_takeover():
    # Build acts where:
    #   atom 0 reliably fires for concept c=0 90% of the time
    #   atom 1 covers the other 10% AND is also precise for c
    #   On rows where c=1 but atom 0 fails, atom 1 fires → absorption is high.
    n = 200
    y = np.zeros(n, dtype=bool); y[:100] = True
    acts = np.zeros((n, 3))
    # Rows 0..89: atom 0 fires.
    acts[:90, 0] = 1.0
    # Rows 90..99 (still c=1): atom 0 silent, atom 1 fires.
    acts[90:100, 1] = 1.0
    # Atom 2 is unrelated noise.
    rng = np.random.default_rng(0)
    acts[:, 2] = rng.standard_normal(n) * 0.01
    res = feature_absorption(acts, y.astype(np.int64))
    # Mean absorption should be near 1.0 because the "missed" rows are
    # fully covered by a precise sibling.
    assert res["mean_absorption"] > 0.9, res


def test_harness_end_to_end_synthetic_labels():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d = 16; n = 200
    X = rng.standard_normal((n, d)).astype(np.float32)
    X -= X.mean(0)
    X_t = torch.from_numpy(X)
    # Minimal labels: pretend each row "is" a color out of 5.
    row_color = (np.arange(n) % 5).astype(np.int64)
    color_rgb = np.eye(5, 3, dtype=np.float32)
    color_hsv = color_rgb.copy()
    labels = HarnessLabels(
        row_color_idx=row_color,
        color_rgb=color_rgb,
        color_hsv=color_hsv,
        row_hue=np.linspace(0, 1, n).astype(np.float32),
    )
    wrapper = PerfectIdentityWrapper(d)
    res = Harness(wrapper, X_t, labels=labels, ablation_subset=4).run()
    assert res.metrics["val_r2"] > 0.99
    assert "sparsity" in res.metrics
    assert "ablation" in res.metrics
    assert "hsv_coherence" in res.metrics


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
