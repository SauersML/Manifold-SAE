"""Tests for HyperbolicSAE + Poincaré primitives."""
from __future__ import annotations
import torch

from manifold_sae.kernels.poincare import (
    exp_0, log_0, mobius_add, poincare_distance,
)
from manifold_sae.hyperbolic_sae import HyperbolicSAE


def test_exp_log_roundtrip():
    torch.manual_seed(0)
    v = torch.randn(50, 8) * 0.3  # small to stay well inside ball after exp
    x = exp_0(v, c=1.0)
    v_back = log_0(x, c=1.0)
    assert torch.allclose(v, v_back, atol=1e-4), \
        f"max err = {(v - v_back).abs().max()}"


def test_mobius_commutativity_at_origin():
    """0 ⊕ y = y  and  x ⊕ 0 = x"""
    torch.manual_seed(1)
    x = torch.randn(20, 8) * 0.1
    zero = torch.zeros_like(x)
    a = mobius_add(zero, x, c=1.0)
    b = mobius_add(x, zero, c=1.0)
    assert torch.allclose(a, x, atol=1e-5), f"0⊕x mismatch {(a-x).abs().max()}"
    assert torch.allclose(b, x, atol=1e-5), f"x⊕0 mismatch {(b-x).abs().max()}"


def test_distance_triangle_inequality():
    torch.manual_seed(2)
    # sample points well inside ball
    pts = torch.randn(30, 8) * 0.1
    pts = exp_0(pts * 2.0, c=1.0)  # ensures strictly inside
    for _ in range(20):
        i, j, k = torch.randint(0, 30, (3,)).tolist()
        d_ij = poincare_distance(pts[i:i+1], pts[j:j+1], c=1.0).item()
        d_jk = poincare_distance(pts[j:j+1], pts[k:k+1], c=1.0).item()
        d_ik = poincare_distance(pts[i:i+1], pts[k:k+1], c=1.0).item()
        # Triangle inequality (with small slack for fp).
        assert d_ik <= d_ij + d_jk + 1e-4, \
            f"triangle fail: {d_ik} > {d_ij}+{d_jk}"


def test_nan_safe_at_boundary():
    """Vectors with huge tangent norm must not produce NaN/Inf."""
    v = torch.randn(20, 8) * 1e6  # blow up
    x = exp_0(v, c=1.0)
    assert torch.isfinite(x).all(), "exp_0 produced NaN/Inf"
    # And the resulting points must stay inside the ball.
    assert (x.norm(dim=-1) < 1.0).all(), "exp_0 escaped the ball"
    # log should be finite after clip.
    lg = log_0(x, c=1.0)
    assert torch.isfinite(lg).all(), "log_0 produced NaN/Inf"
    # Distance to origin finite.
    zero = torch.zeros_like(x)
    d = poincare_distance(zero, x, c=1.0)
    assert torch.isfinite(d).all(), "distance produced NaN/Inf"


def test_loss_decreases_under_training():
    torch.manual_seed(3)
    D, F, d = 16, 8, 4
    sae = HyperbolicSAE(input_dim=D, n_features=F, ball_dim=d,
                        curvature=1.0, sparsity_weight=1e-4)
    x = torch.randn(64, D)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        loss, _ = sae.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    # First 5-mean should be strictly larger than last 5-mean.
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
    # And finite throughout.
    assert all(l == l for l in losses), "NaN loss"
