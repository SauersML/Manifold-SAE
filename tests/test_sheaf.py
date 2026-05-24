"""Tests for the cellular sheaf cohomology primitive."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from manifold_sae.sheaf import (
    CellularSheaf,
    SheafConsistencyHead,
    harmonic_atoms,
    sheaf_consistency_loss,
)


def _make_stub_saes(L: int, F: int):
    return [type("_S", (), {"F": F})() for _ in range(L)]


def test_sheaf_laplacian_is_psd():
    rng = np.random.default_rng(0)
    F, L = 8, 4
    saes = _make_stub_saes(L, F)
    Ts = {(l, l + 1): rng.standard_normal((F, F)) * 0.3 + np.eye(F) for l in range(L - 1)}
    sheaf = CellularSheaf(saes, Ts)
    L_mat = sheaf.laplacian()
    eigvals = np.linalg.eigvalsh(L_mat)
    assert eigvals.min() > -1e-9, f"sheaf Laplacian not PSD: min eigval {eigvals.min()}"
    # symmetry
    assert np.allclose(L_mat, L_mat.T, atol=1e-9)


def test_harmonic_atoms_in_kernel():
    """When transcoders are identity, the harmonic subspace = diagonal {(v,v,v)}.

    Every basis atom should be flagged harmonic.
    """
    F, L = 6, 3
    saes = _make_stub_saes(L, F)
    Ts = {(l, l + 1): np.eye(F) for l in range(L - 1)}
    sheaf = CellularSheaf(saes, Ts)
    modes, atoms, eigvals = harmonic_atoms(sheaf, tol=1e-8, return_eigenvalues=True)
    # ker dim should be F; the F smallest eigenvalues are ~0
    assert (eigvals[:F] < 1e-8).all()
    assert (eigvals[F:] > 1e-6).all()
    assert len(atoms) == F  # all atoms are globally consistent

    # verify modes really lie in the kernel
    L_mat = sheaf.laplacian()
    for mode in modes:
        residual = L_mat @ mode
        assert np.linalg.norm(residual) < 1e-6


def test_restriction_map_composition():
    """δ(δ⁻¹ s) = 0 — applying restrictions then differencing should reconstruct.

    For a chain sheaf, T_{l→l+1} composed with T_{l+1→l+2} should land in the
    span of T_{l→l+2}'s natural action; we test the simpler invariant that a
    harmonic 0-cochain built FROM the composition is unchanged by either
    coboundary route.
    """
    rng = np.random.default_rng(7)
    F = 5
    T01 = rng.standard_normal((F, F)) * 0.2 + np.eye(F)
    T12 = rng.standard_normal((F, F)) * 0.2 + np.eye(F)
    saes = _make_stub_saes(3, F)
    sheaf = CellularSheaf(saes, {(0, 1): T01, (1, 2): T12})

    # build a 0-cochain that's globally consistent by construction:
    # s_0 = v, s_1 = T01 v, s_2 = T12 T01 v.  Its coboundary must be 0.
    v = rng.standard_normal(F)
    s = [v, T01 @ v, T12 @ T01 @ v]
    edges = sheaf.coboundary(s)
    for e in edges:
        assert np.linalg.norm(e) < 1e-10

    # and the composition-built section is in the kernel of L:
    L_mat = sheaf.laplacian()
    s_flat = np.concatenate(s)
    assert np.linalg.norm(L_mat @ s_flat) < 1e-9


def test_energy_decreases_under_training():
    """A few SGD steps on a SheafConsistencyHead should drive ‖δz‖² down.

    Toy: random fixed codes per layer at F=4, L=3, learnable transcoders
    that should converge to the best linear maps.
    """
    torch.manual_seed(0)
    F, L, B = 4, 3, 64
    codes = [torch.randn(B, F) for _ in range(L)]
    head = SheafConsistencyHead(n_layers=L, F=F)
    opt = torch.optim.Adam(head.parameters(), lr=5e-2)
    e0 = float(head.energy(codes).detach())
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = head.energy(codes)
        loss.backward()
        opt.step()
    e1 = float(head.energy(codes).detach())
    assert e1 < e0 * 0.5, f"energy did not decrease meaningfully: {e0:.4f} → {e1:.4f}"


def test_sheaf_consistency_loss_differentiable():
    """The full SAE-piped loss should produce gradients on SAE encoder weights."""
    torch.manual_seed(0)

    class _Enc(torch.nn.Module):
        def __init__(self, D, F):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(D, F) * 0.1)
            self.F = F

        def encode(self, x):
            return torch.relu(x @ self.W)

    saes = [_Enc(8, 4), _Enc(8, 4), _Enc(8, 4)]
    head = SheafConsistencyHead(n_layers=3, F=4)
    x = [torch.randn(16, 8) for _ in range(3)]
    loss = sheaf_consistency_loss(head, saes, x)
    loss.backward()
    for s in saes:
        assert s.W.grad is not None and torch.isfinite(s.W.grad).all()
