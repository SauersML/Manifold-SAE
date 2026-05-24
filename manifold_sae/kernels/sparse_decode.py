"""Sparse curve-atom decode kernel (PyTorch / MPS-safe).

Manifold-SAE's "curve atom" decoder for a batch row b is

    x_hat[b] = b_dec + Σ_f  gate[b,f] · ( atoms[b,f,:] @ basis_coeffs[f,:,:] )

where
    gate           : (B, F)              per-row, per-atom gate / amplitude
    atoms          : (B, F, P)           per-row, per-atom basis evaluations
                     (e.g. Fourier features on θ_f, or Duchon basis at position t_f)
    basis_coeffs   : (F, P, D)           per-atom decoder coefficients (ambient)

The dense path forms either a (B, F, D) per-atom contribution tensor or a
(B, F·P) flattened design that contracts with (F·P, D). Both are O(B·F·D)
in compute AND in peak memory — infeasible at F=2^18 with D=7168 on MPS.

Sparse path
-----------
Under TopK / IBP-Gumbel gating, ``gate`` is K_active-sparse per row
(K_active ≪ F). We:

  1) ``nz = gate.nonzero(as_tuple=False)``  → (M, 2)  rows (b, f) where active
  2) gather only the active rows of ``atoms``:  z_act  = atoms[b, f, :]   (M, P)
  3) gather only the active atoms of ``basis_coeffs``:  C_act = basis_coeffs[f]  (M, P, D)
  4) per-active contributions:  g_act = (gate[b,f] · z_act) · C_act     (M, D)
  5) scatter_add by b back to (B, D).

Peak memory is O(M · max(P, D)) ≈ O(B · K_active · D), independent of F.

API
---
This file exposes two functions with the same signature so callers can
pick the path with a single ``if F > thresh`` test:

    sparse_curve_decode(gate, atoms, basis_coeffs) -> (B, D)
    dense_curve_decode (gate, atoms, basis_coeffs) -> (B, D)

The ``b_dec`` bias is left to the caller (it's a single broadcast add and
keeps this kernel concern-free).

Backward is provided by autograd (no custom backward needed — every op
used is differentiable). The active-row gather is a static index pattern
within a step, so gradients flow correctly through ``index_select`` and
``scatter_add``.

MPS notes
---------
- ``torch.nonzero`` on MPS works for 2-D bool / float inputs (torch>=2.1).
- ``Tensor.index_select`` is MPS-native and is preferred over fancy
  indexing because it doesn't allocate the gather-index expansion.
- ``Tensor.index_add_`` is the MPS-friendly equivalent of scatter_add
  along dim=0 for our use.
"""
from __future__ import annotations

import torch


def _active_indices(gate: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (b_idx, f_idx) flat int64 vectors of active (row, atom) pairs.

    ``threshold`` is strict ``>``; default 0.0 catches IBP-Gumbel hard gates
    and TopK ReLU outputs.  Pass a small positive threshold for soft gates.
    """
    if gate.dtype == torch.bool:
        nz = gate.nonzero(as_tuple=False)
    else:
        nz = (gate.abs() > threshold).nonzero(as_tuple=False)
    # nz: (M, 2) -> rows then cols
    return nz[:, 0].contiguous(), nz[:, 1].contiguous()


def sparse_curve_decode(
    gate: torch.Tensor,
    atoms: torch.Tensor,
    basis_coeffs: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Sparse curve-atom decode. See module docstring.

    Args
    ----
    gate          : (B, F)        float gate · amplitude (or hard 0/1)
    atoms         : (B, F, P)     per-atom basis features for each row
    basis_coeffs  : (F, P, D)     per-atom decoder
    threshold     : strict > gate magnitude treated as active

    Returns
    -------
    out : (B, D) reconstruction (without ``b_dec``)
    """
    if gate.dim() != 2:
        raise ValueError(f"gate must be (B,F); got {tuple(gate.shape)}")
    if atoms.dim() != 3:
        raise ValueError(f"atoms must be (B,F,P); got {tuple(atoms.shape)}")
    if basis_coeffs.dim() != 3:
        raise ValueError(f"basis_coeffs must be (F,P,D); got {tuple(basis_coeffs.shape)}")
    B, F = gate.shape
    _, F2, P = atoms.shape
    F3, P2, D = basis_coeffs.shape
    if not (F == F2 == F3 and P == P2):
        raise ValueError(
            f"shape mismatch: gate=(B={B},F={F}), atoms=(_,F={F2},P={P}), "
            f"basis_coeffs=(F={F3},P={P2},D={D})"
        )

    device = gate.device
    out_dtype = torch.promote_types(
        torch.promote_types(gate.dtype, atoms.dtype), basis_coeffs.dtype
    )

    b_idx, f_idx = _active_indices(gate, threshold=threshold)
    M = b_idx.numel()
    if M == 0:
        return torch.zeros(B, D, dtype=out_dtype, device=device)

    # (1) gather active gate scalars: (M,)
    g_act = gate[b_idx, f_idx].to(out_dtype)
    # (2) gather active basis features. atoms is (B,F,P) — flatten the
    #     first two dims so we can use index_select on dim 0.
    atoms_flat = atoms.reshape(B * F, P)
    flat_idx = b_idx * F + f_idx
    z_act = atoms_flat.index_select(0, flat_idx).to(out_dtype)  # (M, P)
    # (3) gather per-atom coefficients: (M, P, D)
    C_act = basis_coeffs.index_select(0, f_idx).to(out_dtype)
    # (4) per-active contribution: (M, D)
    #     bmm path: (M,1,P) @ (M,P,D) → (M,1,D) → squeeze
    contrib = torch.bmm(z_act.unsqueeze(1), C_act).squeeze(1) * g_act.unsqueeze(1)
    # (5) scatter-add into (B, D)
    out = torch.zeros(B, D, dtype=out_dtype, device=device)
    out.index_add_(0, b_idx, contrib)
    return out


def dense_curve_decode(
    gate: torch.Tensor,
    atoms: torch.Tensor,
    basis_coeffs: torch.Tensor,
    *,
    threshold: float = 0.0,  # noqa: ARG001 — accepted for API parity
) -> torch.Tensor:
    """Dense reference path. Returns the SAME (B, D) tensor as the sparse
    kernel up to floating-point round-off.

    Computed as ``w_phi @ D_flat`` where ``w_phi = (gate * atoms).reshape(B, F*P)``
    and ``D_flat = basis_coeffs.reshape(F*P, D)``. Peak memory is O(B·F·P + F·P·D).
    """
    B, F = gate.shape
    _, _, P = atoms.shape
    _, _, D = basis_coeffs.shape
    out_dtype = torch.promote_types(
        torch.promote_types(gate.dtype, atoms.dtype), basis_coeffs.dtype
    )
    w_phi = (gate.unsqueeze(-1).to(out_dtype) * atoms.to(out_dtype)).reshape(B, F * P)
    D_flat = basis_coeffs.to(out_dtype).reshape(F * P, D)
    return w_phi @ D_flat
