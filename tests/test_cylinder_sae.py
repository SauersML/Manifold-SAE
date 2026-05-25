"""Tests for CylinderSAE (small/fast — B<=16, F<=8).

Covers:
    - forward shape
    - theta wraps at 2π (cos/sin head guarantees S^1)
    - lightness gradient flows
    - loss decreases under a few SGD steps
"""
from __future__ import annotations

import math

import pytest
import torch

from manifold_sae.cylinder_sae import (
    CylinderSAE,
    fourier_basis,
    bspline_basis,
)


D = 32
F = 8
B = 16


def _make() -> CylinderSAE:
    torch.manual_seed(0)
    return CylinderSAE(
        input_dim=D, n_features=F, fourier_harm=3,
        lightness_basis_k=4, top_k=4,
        sparsity_weight=1e-3, ard_weight=1e-3,
        hidden_dim=32,
    )


def test_forward_shape() -> None:
    model = _make()
    x = torch.randn(B, D)
    out = model(x)
    assert out["x_hat"].shape == (B, D)
    assert out["theta"].shape == (B, F)
    assert out["ell"].shape == (B, F)
    assert out["amp"].shape == (B, F)
    # TopK fires exactly top_k atoms per row
    assert torch.allclose(
        (out["amp"] > 0.5).float().sum(dim=1),
        torch.full((B,), 4.0, dtype=torch.float32),
    )
    # basis size = (2H+1) * K_ell = 7 * 4 = 28
    assert model.M == 28
    assert model.B_dec.shape == (F, 28, D)


def test_theta_wraps_at_2pi() -> None:
    """Fourier basis must be 2π-periodic; encoder atan2 head must produce wrap-safe θ."""
    # Direct basis check
    theta_a = torch.tensor([0.1, 1.7, -2.3])
    theta_b = theta_a + 2 * math.pi
    fa = fourier_basis(theta_a, harmonics=3)
    fb = fourier_basis(theta_b, harmonics=3)
    assert torch.allclose(fa, fb, atol=1e-5), \
        f"Fourier basis not 2π-periodic: max diff {(fa - fb).abs().max()}"

    # Encoder θ should be inside [-π, π]
    model = _make()
    x = torch.randn(B, D)
    with torch.no_grad():
        theta, ell, amp_b, _ = model.encode(x)
    assert theta.min() >= -math.pi - 1e-5
    assert theta.max() <= math.pi + 1e-5

    # Decoder reconstruction must agree under θ → θ + 2π shift
    with torch.no_grad():
        recon1 = model.decode(theta, ell, amp_b)
        recon2 = model.decode(theta + 2 * math.pi, ell, amp_b)
    assert torch.allclose(recon1, recon2, atol=1e-4), \
        f"Decoder not 2π-periodic: max diff {(recon1 - recon2).abs().max()}"


def test_lightness_gradient_flows() -> None:
    """∂loss/∂ell must be non-trivial — ell must actually influence the recon."""
    model = _make()
    x = torch.randn(B, D)
    # Make theta/ell leaf so we can get explicit gradients
    theta, ell, amp_b, _ = model.encode(x)
    ell_leaf = ell.detach().clone().requires_grad_(True)
    recon = model.decode(theta.detach(), ell_leaf, amp_b.detach())
    loss = ((recon - x) ** 2).mean()
    loss.backward()
    assert ell_leaf.grad is not None
    g = ell_leaf.grad.abs()
    # Gradient on the active atoms must be non-zero somewhere
    assert g.max().item() > 1e-8, f"ell gradient vanished: max abs grad={g.max()}"

    # B-spline basis itself must have non-zero gradient wrt ell. The basis is
    # partition-of-unity (rows sum to 1) so use a non-uniform readout to expose
    # ell-sensitivity rather than .sum() which is constant.
    e = torch.tensor([0.0, 0.5, -1.0], requires_grad=True)
    phi = bspline_basis(e, n_basis=4)
    w = torch.tensor([0.1, 0.4, -0.7, 1.3])
    (phi * w).sum().backward()
    assert e.grad is not None
    assert e.grad.abs().max().item() > 1e-6


def test_loss_decreases_under_sgd() -> None:
    """A few SGD steps on the same batch should reduce loss."""
    model = _make()
    x = torch.randn(B, D) * 0.3
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(25):
        loss, _ = model.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    # Specifically, the recon-dominant term should drop materially
    assert losses[-1] < 0.9 * losses[0], \
        f"Loss decreased too little: {losses[0]:.4f} -> {losses[-1]:.4f}"


if __name__ == "__main__":
    test_forward_shape()
    test_theta_wraps_at_2pi()
    test_lightness_gradient_flows()
    test_loss_decreases_under_sgd()
    print("ok 4/4")
