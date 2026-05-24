"""Tests for AdaptiveKSAE."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.adaptive_k import AdaptiveKSAE


def test_shapes():
    model = AdaptiveKSAE(input_dim=64, F=128, k_min=4, k_max=32)
    x = torch.randn(8, 64)
    recon, z, k = model.forward(x)
    assert recon.shape == (8, 64)
    assert z.shape == (8, 128)
    assert k.shape == (8,)


def test_k_in_range():
    model = AdaptiveKSAE(input_dim=32, F=64, k_min=4, k_max=16)
    x = torch.randn(16, 32)
    _, z, k = model.forward(x)
    assert (k >= 4 - 1e-5).all() and (k <= 16 + 1e-5).all()
    n_active = (z.abs() > 0).sum(-1)
    assert (n_active >= 4).all() and (n_active <= 16).all()


def test_loss_smoke_and_backward():
    model = AdaptiveKSAE(input_dim=16, F=32, k_min=2, k_max=8, sparsity_weight=1e-2)
    x = torch.randn(4, 16)
    out = model.loss(x)
    out["loss"].backward()
    # All trainable params should have nonzero grad somewhere
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert any_grad


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = AdaptiveKSAE(input_dim=24, F=64, k_min=4, k_max=24, sparsity_weight=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(64, 24)
    init_loss = model.loss(x)["recon"].item()
    for _ in range(50):
        opt.zero_grad()
        out = model.loss(x)
        out["loss"].backward()
        opt.step()
    final_loss = model.loss(x)["recon"].item()
    assert final_loss < init_loss, (init_loss, final_loss)
