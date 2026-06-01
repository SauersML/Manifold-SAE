"""Tests for AdaptiveKSAE / AdaptiveKv2SAE (gamfit-native AdaptiveTopK shell).

forward() now returns ``(recon, z_active, k_pred_eff)`` where ``k_pred_eff`` is
the gate's per-row effective K (soft-mask mass, shape ``(B,)``). The number of
nonzero entries per row equals ``round(k_head(z))`` ∈ ``[k_min, k_max]``.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.adaptive_k import AdaptiveKSAE
from manifold_sae.adaptive_k_v2 import AdaptiveKv2SAE


def test_shapes():
    model = AdaptiveKSAE(input_dim=64, F=128, k_min=4, k_max=32)
    x = torch.randn(8, 64)
    recon, z, k = model.forward(x)
    assert recon.shape == (8, 64)
    assert z.shape == (8, 128)
    assert k.shape == (8,)


def test_active_count_in_range():
    model = AdaptiveKSAE(input_dim=32, F=64, k_min=4, k_max=16)
    x = torch.randn(16, 32)
    _, z, k = model.forward(x)
    # k_pred_eff is finite and roughly in the bracket.
    assert torch.isfinite(k).all()
    # Hard top-K active counts are bounded by [k_min, k_max].
    n_active = (z.abs() > 0).sum(-1)
    assert (n_active >= 4).all() and (n_active <= 16).all()


def test_loss_smoke_and_backward():
    model = AdaptiveKSAE(input_dim=16, F=32, k_min=2, k_max=8, sparsity_weight=1e-2)
    x = torch.randn(4, 16)
    out = model.loss(x)
    assert set(out) >= {"loss", "recon", "sparsity", "z", "k_pred"}
    out["loss"].backward()
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert any_grad
    # The gate's learnable sparsity weight should be a trainable parameter.
    assert model.gate.log_weight.requires_grad


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


def test_v2_shapes_and_mlp_head():
    model = AdaptiveKv2SAE(input_dim=32, F=64, k_target=16, k_min=8, k_max=32)
    x = torch.randn(8, 32)
    recon, z, k = model.forward(x)
    assert recon.shape == (8, 32)
    assert z.shape == (8, 64)
    assert k.shape == (8,)
    # v2 uses the mlp head.
    assert model.gate.head_kind == "mlp"
    out = model.loss(x)
    assert "k_std" in out
    out["loss"].backward()


def test_v1_uses_linear_head():
    model = AdaptiveKSAE(input_dim=16, F=32, k_min=2, k_max=8)
    assert model.gate.head_kind == "linear"
