"""Tests for WassersteinSAE + Sinkhorn kernel."""
from __future__ import annotations

import torch

from manifold_sae.kernels.sinkhorn import (
    circular_cost_matrix,
    sinkhorn_barycenter,
    sinkhorn_log,
)
from manifold_sae.wasserstein_sae import WassersteinSAE


def test_sinkhorn_convergence_marginals():
    """A converged Sinkhorn plan has correct marginals.

    Plan T must satisfy T 1 = a and T^T 1 = b within eps-dependent tolerance.
    """
    torch.manual_seed(0)
    M = 16
    C = circular_cost_matrix(M)
    a = torch.softmax(torch.randn(M), dim=-1)
    b = torch.softmax(torch.randn(M), dim=-1)
    T = sinkhorn_log(a, b, C, eps=0.05, n_iter=400, tol=1e-9)[0]
    row = T.sum(dim=1)
    col = T.sum(dim=0)
    assert torch.allclose(row, a, atol=1e-3), f"row marginal off: {(row - a).abs().max()}"
    assert torch.allclose(col, b, atol=1e-3), f"col marginal off: {(col - b).abs().max()}"


def test_barycenter_is_probability_and_in_support_hull():
    """Barycenter is a probability distribution; with weight on two opposite
    atoms it has mass between them on the hue circle (not outside)."""
    M = 32
    C = circular_cost_matrix(M)
    # Two sharp atoms at opposite hues (0 and M/2).
    def bump(center, width=1.0):
        idx = torch.arange(M).float()
        d = (idx - center).abs()
        d = torch.minimum(d, M - d)
        return torch.softmax(-(d / width) ** 2 * 8, dim=-1)
    atoms = torch.stack([bump(0), bump(M // 2), bump(M // 4), bump(3 * M // 4)])
    pi = torch.tensor([[0.25, 0.25, 0.25, 0.25],            # uniform
                       [0.95, 0.05 / 3, 0.05 / 3, 0.05 / 3]])  # concentrated on atom 0
    bary = sinkhorn_barycenter(atoms, pi, C, eps=0.02, n_iter=400, tol=1e-9)
    assert bary.shape == (2, M)
    # Probability simplex: sums to 1, non-negative.
    assert torch.allclose(bary.sum(-1), torch.ones(2, dtype=bary.dtype), atol=1e-3)
    assert (bary >= -1e-6).all()
    # Concentrated-weight barycenter is closer to atom 0 than to any other
    # atom (the only "convex-hull" check that survives entropic blurring).
    tv = (bary[1].unsqueeze(0) - atoms).abs().sum(-1) / 2.0
    assert tv.argmin().item() == 0, f"95%-on-0 bary not closest to atom 0: tv={tv}"


def test_gradient_flows_through_pi_and_atoms():
    """Gradient agreement: dL/dπ and dL/dθ are both nonzero & finite."""
    torch.manual_seed(2)
    model = WassersteinSAE(F=8, M=16, D=32, eps=0.05, n_sinkhorn_iter=20,
                           neighbor_weight=1e-3)
    x = torch.randn(4, 32)
    out = model.loss(x)
    out["total"].backward()
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()
    assert model.encoder.weight.grad.abs().sum() > 0
    assert model.atom_logits.grad is not None
    assert torch.isfinite(model.atom_logits.grad).all()
    assert model.atom_logits.grad.abs().sum() > 0
    assert model.readout.weight.grad is not None
    assert torch.isfinite(model.readout.weight.grad).all()


def test_one_training_step_reduces_loss():
    """A handful of Adam steps on the same batch must reduce MSE."""
    torch.manual_seed(3)
    model = WassersteinSAE(F=8, M=16, D=32, eps=0.05, n_sinkhorn_iter=20,
                           neighbor_weight=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    x = torch.randn(16, 32)
    losses = []
    for _ in range(20):
        out = model.loss(x)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        losses.append(out["mse"].item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} → {losses[-1]}"
