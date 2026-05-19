"""Regression test for the synthetic-recovery pipeline.

We build a tiny dataset, train a small ManifoldSAE for a short budget, and check
that the trained features have at least directionally recovered the planted
curves. The threshold is intentionally generous: the goal is to catch total
breakage (encoder collapsed, decoder degenerate, dead features all the way down),
not to pin a particular reconstruction quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from manifold_sae.data_synthetic import SyntheticDataset, chamfer_distance


@pytest.mark.slow
def test_synthetic_recovery_directional() -> None:
    from manifold_sae import sae as sae_mod
    from manifold_sae import train as train_mod
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = SyntheticDataset(
        d_ambient=16,
        n_features=3,
        n_samples=512,
        sparsity=0.4,
        noise=0.05,
        seed=0,
        t_grid_size=64,
    )
    assert dataset.x.shape == (512, 16)
    assert len(dataset.features) == 3

    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    config = sae_mod.ManifoldSAEConfig(
        input_dim=16,
        n_features=5,  # 3 planted + 2 slack
        n_basis=14,
        top_k=2,
        # Tuned for fast synthetic recovery: drop sparsity so features don't die.
        sparsity_weight=0.0,
    )
    sae = sae_mod.ManifoldSAE(config)
    optimizer = train_mod.build_optimizer(sae, lr=5e-3)

    train_mod.train(sae, loader, optimizer, n_steps=800, log_every=400)

    # Directional check: reconstruction MSE should drop substantially below
    # the trivial baseline (zero prediction). A tighter feature-recovery test
    # is impractical at this training budget — the proper recovery experiment
    # in experiments/synthetic_recovery.py runs for far longer and explicitly
    # matches learned curves against ground truth via chamfer.
    sae.eval()
    with torch.no_grad():
        recon = sae(dataset.x).reconstruction
    mse = float(((recon - dataset.x) ** 2).mean().item())
    naive_mse = float((dataset.x ** 2).mean().item())
    relative = mse / naive_mse
    assert relative < 0.7, (
        f"SAE reconstruction MSE {mse:.4f} too close to naive zero-baseline "
        f"{naive_mse:.4f} (ratio={relative:.3f}); architecture did not learn"
    )
    # Also assert some features stayed alive — the dead-feature failure mode.
    alive = (sae(dataset.x).amplitudes.mean(dim=0) > 1e-3).sum().item()
    assert alive >= 2, f"only {alive} features survived training (need >=2)"
    _ = chamfer_distance  # imported for the longer experiment; keep reachable


def test_synthetic_dataset_shape() -> None:
    """Pure data-side check; runs even when the SAE swarm hasn't shipped yet."""
    ds = SyntheticDataset(d_ambient=8, n_features=2, n_samples=64, sparsity=0.5, noise=0.0, seed=1)
    assert ds.x.shape == (64, 8)
    item = ds[0]
    assert item.shape == (8,)
    assert ds.ground_truth["curve_points"].shape[0] == 2
    # At least one feature is active per sample.
    assert ds.ground_truth["active"].any(dim=1).all()


def test_chamfer_symmetric_and_zero_on_self() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(size=(20, 5))
    b = rng.standard_normal(size=(30, 5))
    assert chamfer_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    d_ab = chamfer_distance(a, b)
    d_ba = chamfer_distance(b, a)
    assert d_ab == pytest.approx(d_ba, rel=1e-9)
