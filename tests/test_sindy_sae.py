"""Tests for SINDy-SAE."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from manifold_sae.sindy_sae import (
    SINDySAE,
    build_library,
    library_size,
    library_term_names,
)
from manifold_sae.sindy_sae_static import (
    fake_derivative_from_batch_order,
    smoke_fit,
)


def test_library_size_and_names_consistent() -> None:
    for terms in [
        ("identity",),
        ("identity", "square"),
        ("constant", "identity", "square", "cube", "sin", "cos", "product"),
        ("product",),
    ]:
        P = library_size(3, terms)
        names = library_term_names(3, terms)
        assert P == len(names)
        z = torch.randn(7, 3)
        phi = build_library(z, terms)
        assert phi.shape == (7, P)


def test_forward_shapes_and_loss_decreases() -> None:
    torch.manual_seed(0)
    sindy = SINDySAE(state_dim=3, library_terms=("identity", "square", "product"), sparsity=1e-3)
    z = torch.randn(64, 3)
    dz = torch.randn(64, 3)
    out0 = sindy.loss(z, dz)
    opt = torch.optim.Adam(sindy.parameters(), lr=1e-1)
    for _ in range(200):
        opt.zero_grad()
        o = sindy.loss(z, dz)
        o["total"].backward()
        opt.step()
    out1 = sindy.loss(z, dz)
    assert out1["recon"].item() < out0["recon"].item()
    assert sindy(z).shape == (64, 3)


def test_threshold_zeros_small_entries() -> None:
    sindy = SINDySAE(state_dim=2, library_terms=("identity", "square"))
    with torch.no_grad():
        sindy.Theta.data.copy_(torch.tensor([[0.001, 1.0, 0.0, 2.0], [3.0, 0.002, 0.5, 0.0]]))
    n_active = sindy.threshold(eps=0.1)
    Theta = sindy.effective_Theta().numpy()
    assert Theta[0, 0] == 0.0 and Theta[1, 1] == 0.0
    assert Theta[0, 1] == 1.0 and Theta[1, 0] == 3.0
    # surviving entries: Theta[0,1]=1.0, Theta[0,3]=2.0, Theta[1,0]=3.0, Theta[1,2]=0.5
    assert n_active == 4


def test_fake_derivative_smoke_runs_on_random_data() -> None:
    torch.manual_seed(1)
    z = torch.randn(500, 4)
    sindy = SINDySAE(state_dim=4, library_terms=("identity", "square"), sparsity=1e-3)
    result = smoke_fit(sindy, z, n_steps=30, lr=1e-2)
    assert "WARNING" in result
    assert np.isfinite(result["total"])
    z_in, dz_fake = fake_derivative_from_batch_order(z, dt=1.0)
    assert z_in.shape == (499, 4) and dz_fake.shape == (499, 4)


@pytest.mark.slow
def test_lorenz_recovery_within_tolerance() -> None:
    """End-to-end: train SINDy-SAE on Lorenz, check Θ relative error < 0.1."""
    from scripts.train_sindy_synthetic_lorenz import TrainConfig, train

    cfg = TrainConfig(
        n_steps=15000,
        n_iters=4000,
        n_stlsq_rounds=4,
        out_dir=__import__("pathlib").Path("runs/SINDY_LORENZ_TEST"),
    )
    rep = train(cfg)
    assert rep["rel_frobenius_error"] < 0.1, rep["rel_frobenius_error"]
