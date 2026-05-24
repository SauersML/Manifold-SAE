"""Tests for the sparse curve-atom decode kernel."""
from __future__ import annotations

import pytest
import torch

from manifold_sae.kernels.sparse_decode import (
    dense_curve_decode,
    sparse_curve_decode,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_inputs(
    B: int, F: int, P: int, D: int, K_active: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    # build sparse gate: K_active random atoms per row, with random positive weights
    gate = torch.zeros(B, F, dtype=torch.float32)
    for b in range(B):
        idx = torch.randperm(F, generator=g)[:K_active]
        gate[b, idx] = (torch.rand(K_active, generator=g) + 0.1).float()
    atoms = torch.randn(B, F, P, generator=g, dtype=torch.float32)
    basis = torch.randn(F, P, D, generator=g, dtype=torch.float32) * (1.0 / (P ** 0.5))
    dev = _device()
    return gate.to(dev), atoms.to(dev), basis.to(dev)


def test_numeric_match_dense_F512():
    """sparse vs dense at F=512 must agree within 1e-4 max-abs."""
    B, F, P, D, K = 16, 512, 5, 128, 8
    gate, atoms, basis = _make_inputs(B, F, P, D, K)
    y_sparse = sparse_curve_decode(gate, atoms, basis)
    y_dense = dense_curve_decode(gate, atoms, basis)
    max_abs = (y_sparse - y_dense).abs().max().item()
    assert max_abs < 1e-4, f"max_abs diff {max_abs:.3e} >= 1e-4"


def test_gradient_agreement():
    """Gradients of (sparse_out.sum()) wrt basis_coeffs must match dense."""
    B, F, P, D, K = 8, 64, 4, 32, 6
    gate, atoms, basis = _make_inputs(B, F, P, D, K, seed=1)

    basis_s = basis.detach().clone().requires_grad_(True)
    basis_d = basis.detach().clone().requires_grad_(True)
    atoms_s = atoms.detach().clone().requires_grad_(True)
    atoms_d = atoms.detach().clone().requires_grad_(True)

    y_s = sparse_curve_decode(gate, atoms_s, basis_s)
    y_d = dense_curve_decode(gate, atoms_d, basis_d)

    # arbitrary scalar to backprop
    w = torch.randn_like(y_s)
    (y_s * w).sum().backward()
    (y_d * w).sum().backward()

    g_basis_diff = (basis_s.grad - basis_d.grad).abs().max().item()
    g_atoms_diff = (atoms_s.grad - atoms_d.grad).abs().max().item()
    assert g_basis_diff < 1e-4, f"basis grad diff {g_basis_diff:.3e}"
    assert g_atoms_diff < 1e-4, f"atoms grad diff {g_atoms_diff:.3e}"


def test_gate_zero_yields_zero_contribution():
    """Atoms with gate=0 must contribute nothing — even with huge atom/basis."""
    B, F, P, D = 4, 32, 3, 16
    dev = _device()
    gate = torch.zeros(B, F, device=dev, dtype=torch.float32)
    # turn on only atom 0 for every row, with weight 1
    gate[:, 0] = 1.0
    atoms = torch.randn(B, F, P, device=dev, dtype=torch.float32) * 100.0
    basis = torch.randn(F, P, D, device=dev, dtype=torch.float32) * 100.0

    y = sparse_curve_decode(gate, atoms, basis)
    # Contribution from atom 0 only
    expected = (atoms[:, 0, :].unsqueeze(1) @ basis[0]).squeeze(1)  # (B, D)
    assert (y - expected).abs().max().item() < 1e-3


def test_batch_one_edge_case():
    """B=1 must not crash and must match dense."""
    B, F, P, D, K = 1, 256, 5, 64, 4
    gate, atoms, basis = _make_inputs(B, F, P, D, K, seed=2)
    y_s = sparse_curve_decode(gate, atoms, basis)
    y_d = dense_curve_decode(gate, atoms, basis)
    assert y_s.shape == (1, D)
    assert (y_s - y_d).abs().max().item() < 1e-4


def test_duplicate_index_scatter_correctness():
    """Multiple atoms active in the SAME row must scatter-add correctly
    (each active (b,f) writes to row b, and accumulations must sum)."""
    B, F, P, D = 3, 8, 2, 5
    dev = _device()
    # All atoms active for row 0 → 8 contributions accumulate to row 0
    gate = torch.zeros(B, F, device=dev, dtype=torch.float32)
    gate[0, :] = 1.0           # row 0: all 8 atoms
    gate[1, 0] = 1.0           # row 1: one atom
    # row 2: zero
    atoms = torch.randn(B, F, P, device=dev, dtype=torch.float32)
    basis = torch.randn(F, P, D, device=dev, dtype=torch.float32)

    y_s = sparse_curve_decode(gate, atoms, basis)
    y_d = dense_curve_decode(gate, atoms, basis)
    # full equivalence
    assert (y_s - y_d).abs().max().item() < 1e-4

    # spot-check row 0 == sum over all atoms of atoms[0,f] @ basis[f]
    expected_row0 = sum(atoms[0, f] @ basis[f] for f in range(F))
    assert (y_s[0] - expected_row0).abs().max().item() < 1e-4
    # row 2 must be exactly zero (no active atoms)
    assert y_s[2].abs().max().item() == 0.0
