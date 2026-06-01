"""Tests for the amortized Manifold-SAE (gam-native successor to M-SAE).

Validates the feedforward spine against gamfit 0.1.134: shapes, S^1 coordinate
wrap, sparsity, gradient flow, and that it actually recovers circle-structured
data. Every test must genuinely pass — no xfail (see feedback_never_xfail).
"""
from __future__ import annotations

import math

import torch

from manifold_sae.amortized_manifold_sae import (
    AmortizedManifoldSAE,
    AmortizedManifoldSAEConfig,
    _circle_basis_deriv,
)


def _circle_data(N: int = 256, D: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(N, generator=g) * 2 * math.pi
    ring = torch.stack([torch.cos(t), torch.sin(t), torch.cos(2 * t), torch.sin(2 * t)], -1)
    W = torch.randn(4, D, generator=g)
    return ring @ W + 0.05 * torch.randn(N, D, generator=g)


def test_shapes_and_forward():
    cfg = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=32, fourier_harmonics=3)
    sae = AmortizedManifoldSAE(cfg)
    x = _circle_data(64, 16)
    out = sae(x)
    assert out.x_hat.shape == (64, 16)
    assert out.gate.shape == (64, 32)
    assert out.theta.shape == (64, 32)
    assert torch.isfinite(out.x_hat).all()


def test_theta_is_valid_s1_coordinate():
    cfg = AmortizedManifoldSAEConfig(input_dim=12, n_atoms=16, fourier_harmonics=2)
    sae = AmortizedManifoldSAE(cfg)
    theta = sae(_circle_data(128, 12)).theta
    assert (theta >= -math.pi - 1e-4).all() and (theta <= math.pi + 1e-4).all()


def test_gate_is_sparse():
    cfg = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=64, gate_threshold=0.1)
    sae = AmortizedManifoldSAE(cfg)
    out = sae(_circle_data(128, 16))
    # JumpReLU must zero at least some atoms (not a dense code).
    assert float(out.active_fraction) < 1.0


def test_loss_backward_grads_flow():
    cfg = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=32)
    sae = AmortizedManifoldSAE(cfg)
    d = sae.loss(_circle_data(64, 16))
    d["loss"].backward()
    grads = [p.grad for p in sae.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)
    # decoder blocks B and both encoder heads must receive gradient.
    assert sae.B.grad is not None and sae.B.grad.abs().sum() > 0


def test_isometry_penalty_zero_for_unit_circle_nonzero_otherwise():
    # M = 2H+1 basis = [1, cos t, sin t, cos 2t, sin 2t, ...]
    cfg = AmortizedManifoldSAEConfig(input_dim=2, n_atoms=2, fourier_harmonics=2,
                                     isometry_weight=1.0)
    sae = AmortizedManifoldSAE(cfg)
    with torch.no_grad():
        sae.B.zero_()
        # atom 0: g(t) = [cos t, sin t]  -> constant speed 1 -> isometric.
        sae.B[0, 1, 0] = 1.0  # cos t -> x
        sae.B[0, 2, 1] = 1.0  # sin t -> y
        # atom 1: g(t) = [cos t, sin 2t] -> speed varies -> NOT isometric.
        sae.B[1, 1, 0] = 1.0  # cos t  -> x
        sae.B[1, 4, 1] = 1.0  # sin 2t -> y
    pen_all = sae.isometry_penalty()
    assert torch.isfinite(pen_all) and pen_all >= 0
    grid = torch.linspace(-math.pi, math.pi, 33, dtype=sae.B.dtype)[:-1]
    dphi = _circle_basis_deriv(grid, 2)                       # (32, M)
    speed0 = torch.einsum("gm,md->gd", dphi, sae.B[0]).norm(dim=-1)  # unit circle
    speed1 = torch.einsum("gm,md->gd", dphi, sae.B[1]).norm(dim=-1)  # cos t, sin 2t
    assert speed0.var() < 1e-6, "unit-circle atom must be isometric (constant speed)"
    assert speed1.var() > 1e-3, "cos t / sin 2t atom must NOT be isometric"


def test_incoherence_penalty_zero_for_orthogonal_atoms_positive_for_shared():
    # Two atoms; M = 2H+1 = 5 rows each. Decoder rows are unit vectors in R^8.
    cfg = AmortizedManifoldSAEConfig(input_dim=12, n_atoms=2, fourier_harmonics=2)
    sae = AmortizedManifoldSAE(cfg)
    M = sae.M  # = 5; needs input_dim >= 2*M for two orthogonal atoms
    with torch.no_grad():
        sae.B.zero_()
        # Atom 0 lives in dims 0..M-1, atom 1 in dims M..2M-1 -> orthogonal subspaces.
        for r in range(M):
            sae.B[0, r, r] = 1.0
            sae.B[1, r, M + r] = 1.0
    assert sae.incoherence_penalty() < 1e-8, "orthogonal atoms must be incoherent-free"
    with torch.no_grad():
        # Make atom 1 reuse atom 0's directions -> maximal cross-coherence.
        sae.B[1].copy_(sae.B[0])
    assert sae.incoherence_penalty() > 0.1, "atoms sharing a subspace must be penalized"


def test_incoherence_pushes_atoms_apart_in_training():
    # Two well-separated rings in R^16; with incoherence on, the atoms that capture
    # them should end up less mutually coherent than with it off.
    cfg_off = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=8, fourier_harmonics=2,
                                         incoherence_weight=0.0)
    cfg_on = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=8, fourier_harmonics=2,
                                        incoherence_weight=1e-1)
    x = _circle_data(256, 16)

    def train(cfg):
        torch.manual_seed(0)
        m = AmortizedManifoldSAE(cfg)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(150):
            opt.zero_grad(); m.loss(x)["loss"].backward(); opt.step()
        return float(m.incoherence_penalty())

    coh_off, coh_on = train(cfg_off), train(cfg_on)
    assert coh_on < coh_off, f"incoherence penalty did not reduce coherence ({coh_on} !< {coh_off})"


def test_training_recovers_circle_structure():
    cfg = AmortizedManifoldSAEConfig(input_dim=16, n_atoms=32, fourier_harmonics=3,
                                     sparsity_weight=1e-3)
    sae = AmortizedManifoldSAE(cfg)
    x = _circle_data(256, 16)
    opt = torch.optim.Adam(sae.parameters(), lr=5e-3)
    l0 = sae.loss(x)["loss"].item()
    for _ in range(300):
        opt.zero_grad()
        d = sae.loss(x)
        d["loss"].backward()
        opt.step()
    final = sae.loss(x)
    assert final["loss"].item() < l0
    r2 = 1.0 - final["recon"].item() / x.var().item()
    assert r2 > 0.5, f"R2={r2:.3f}: failed to recover circle-structured data"
