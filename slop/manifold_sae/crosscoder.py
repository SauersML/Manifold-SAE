"""Crosscoder SAE: one shared sparse dictionary across multiple layers.

Reference: Lindsey et al., "Sparse Crosscoders for Cross-Layer Features",
Anthropic, 2024.

This module is a thin re-export of gamfit's Rust-backed crosscoder primitive
(``gamfit.crosscoder.Crosscoder``). The hand-rolled ``nn.Module`` that used to
live here (shared encoder + per-layer decoders, decoder-norm-weighted L1,
manifold-curve decoders, top-k gating, tied encoder) has been deleted; all
numerics now live in gamfit.

New (gamfit-native) API
-----------------------
    cc = Crosscoder(layer_dims, n_atoms, *, decoder_weighted_l1=True,
                    shared_encoder="mlp[1024]", l1_weight=1e-3)
    cc.fit(X_stack, epochs=..., lr=..., batch_size=..., seed=...)   # X_stack: list[np.ndarray]
    cc.per_layer_r2()          # -> np.ndarray (L,)
    cc.atom_layer_affinity()   # -> np.ndarray (n_atoms, L), row-max normalised
    cc.harmonic_atoms(tol)     # -> np.ndarray of cross-layer ("shared") atom indices
    cc.diagnostics             # per-epoch loss curves

Behaviour has intentionally changed. The gamfit ``Crosscoder`` is NOT an
``nn.Module``: it takes numpy arrays, builds its torch module lazily inside
``.fit()``, and trains with Adam. See the module cutover report for the list of
former features with no gamfit equivalent (manifold mode, top-k gating, tied
encoder, ``cross_layer_atom_mask``, ``encode``/``decode_layer``/``forward``,
and the row-sum-normalised affinity convention).
"""
from __future__ import annotations

from gamfit.crosscoder import Crosscoder, CrosscoderFitDiagnostics

__all__ = ["Crosscoder", "CrosscoderFitDiagnostics"]
