"""Threshold-gated dispatch: dense vs sparse curve-atom decode.

The Manifold-SAE Fourier-decoder trainer in ``scripts/train_sae_comparison.py``
materializes ``w_phi.reshape(B, F*P) @ D_flat.reshape(F*P, D)``. That O(B·F·P + F·P·D)
peak memory is fine up to ~F=8 K on MPS-Apple-Silicon but blows past
unified-memory budgets at F=2^16 and above.

This module exposes ``curve_decode_auto`` which dispatches to the dense
matmul below a configurable threshold (default F=8192) and to the sparse
gather/scatter kernel above it. The sparse kernel is bit-identical to
the dense one up to float32 round-off (verified to max-abs-diff < 1e-4
in tests/test_sparse_decode.py and at F=512 end-to-end).

Usage in a decoder forward (the Fourier ManifoldSAE in
``scripts/train_sae_comparison.py``):

    from manifold_sae.scale import curve_decode_auto
    # ...
    recon = curve_decode_auto(gate * amp_unused, atoms=phi, basis_coeffs=self.D_k)
    recon = recon + self.b_d
"""
from __future__ import annotations

import os

import torch

from .kernels.sparse_decode import dense_curve_decode, sparse_curve_decode

# Threshold can be overridden by env var for benchmarking / ablation.
_DEFAULT_F_SPARSE_THRESHOLD = int(os.environ.get("MANIFOLD_SAE_SPARSE_F", "8192"))


def curve_decode_auto(
    gate: torch.Tensor,
    atoms: torch.Tensor,
    basis_coeffs: torch.Tensor,
    *,
    f_sparse_threshold: int | None = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Dispatch to dense decode for F ≤ threshold, sparse decode above.

    Args
    ----
    gate            : (B, F) gating values (TopK/Gumbel hard or soft).
    atoms           : (B, F, P) per-atom basis features.
    basis_coeffs    : (F, P, D) per-atom decoder.
    f_sparse_threshold : F above which we use the sparse kernel; defaults to
                         env var MANIFOLD_SAE_SPARSE_F (8192).
    threshold       : strict > magnitude considered "active" in the sparse path.

    Returns
    -------
    (B, D) reconstruction (no b_dec — caller adds the bias).
    """
    thr = f_sparse_threshold if f_sparse_threshold is not None else _DEFAULT_F_SPARSE_THRESHOLD
    F = gate.shape[1]
    if F > thr:
        return sparse_curve_decode(gate, atoms, basis_coeffs, threshold=threshold)
    return dense_curve_decode(gate, atoms, basis_coeffs, threshold=threshold)


__all__ = ["curve_decode_auto", "sparse_curve_decode", "dense_curve_decode"]
