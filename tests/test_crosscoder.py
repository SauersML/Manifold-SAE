"""Crosscoder primitive tests.

Run only this file:
    uv run pytest tests/test_crosscoder.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from manifold_sae.crosscoder import Crosscoder
from manifold_sae.circuit_trace import (
    build_circuit,
    per_layer_attribution,
    trace_and_save,
)


def _toy_data(seed: int = 0, N: int = 256) -> list[torch.Tensor]:
    """A synthetic 3-layer stack with shared latent structure.

    True generative model: 8 latent atoms z* ~ Exp(1) gated by Bernoulli(0.2).
    Each layer maps z* through a random per-layer decoder + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    F_true = 8
    z = (rng.exponential(1.0, size=(N, F_true)) * (rng.random((N, F_true)) < 0.2)).astype(np.float32)
    layers: list[torch.Tensor] = []
    for D in (16, 24, 32):
        D_l = D
        W = rng.standard_normal((F_true, D_l)).astype(np.float32) / np.sqrt(F_true)
        x = z @ W + 0.05 * rng.standard_normal((N, D_l)).astype(np.float32)
        layers.append(torch.from_numpy(x))
    return layers


def test_forward_shapes_and_loss() -> None:
    """Output shapes line up; loss is finite + differentiable."""
    layers = _toy_data()
    model = Crosscoder(layer_dims=[16, 24, 32], n_atoms=32, sparsity_weight=1e-3)
    out = model(layers)
    assert out["z"].shape == (256, 32)
    for l, D in enumerate([16, 24, 32]):
        assert out["recons"][l].shape == (256, D)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    # Every decoder must receive a gradient.
    for l in range(3):
        g = model.decoders[l].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0


def test_training_step_reduces_loss() -> None:
    """A few Adam steps drive loss down (sanity of optimization path)."""
    layers = _toy_data(seed=1)
    model = Crosscoder(layer_dims=[16, 24, 32], n_atoms=32, sparsity_weight=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    out0 = model(layers)
    l0 = float(out0["loss"].item())
    for _ in range(80):
        opt.zero_grad()
        model(layers)["loss"].backward()
        opt.step()
    l1 = float(model(layers)["loss"].item())
    assert l1 < 0.5 * l0, f"loss did not drop enough: {l0:.3f} -> {l1:.3f}"

    # Per-layer R² should be meaningfully positive on this clean data.
    r2 = model.per_layer_r2(layers)
    assert all(r > 0.3 for r in r2), f"R² too low across layers: {r2}"


def test_atom_layer_affinity_and_cross_mask() -> None:
    """Affinity rows sum to 1; cross-layer mask threshold behaves correctly."""
    model = Crosscoder(layer_dims=[10, 12, 14], n_atoms=16)
    # Hand-set decoder norms: atom 0 → only layer 0; atom 1 → uniform.
    with torch.no_grad():
        for l in range(3):
            model.decoders[l].zero_()
        # Set rows so each non-zero row has L2 norm = 1 — that way "uniform"
        # affinity actually means equal decoder norm, not equal coordinate value.
        for D_l, l in zip([10, 12, 14], range(3)):
            scale = 1.0 / (D_l ** 0.5)
            if l == 0:
                model.decoders[l][0, :] = scale  # atom 0 only in layer 0
            model.decoders[l][1, :] = scale       # atom 1 uniformly
    aff = model.atom_layer_affinity()
    assert aff.shape == (16, 3)
    sums = aff.sum(dim=-1)
    # Rows for inactive atoms (norm 0) are NaN-safe (0/0 clamp); skip them.
    nonzero = (sums > 1e-6)
    assert torch.allclose(sums[nonzero], torch.ones_like(sums[nonzero]), atol=1e-5)
    # atom 0: concentrated in layer 0.
    assert aff[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    # atom 1: uniform across 3 layers.
    assert torch.allclose(aff[1], torch.full((3,), 1 / 3.0), atol=1e-5)

    mask = model.cross_layer_atom_mask(threshold=0.15)
    assert bool(mask[1].item())
    assert not bool(mask[0].item())


def test_circuit_trace_dot_emission(tmp_path) -> None:
    """Circuit tracer produces a non-trivial DOT file with valid syntax."""
    layers = _toy_data(seed=2)
    model = Crosscoder(layer_dims=[16, 24, 32], n_atoms=32, sparsity_weight=1e-4)
    # Quick warmup so atoms are not all dead.
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(40):
        opt.zero_grad()
        model(layers)["loss"].backward()
        opt.step()

    z_tilde = per_layer_attribution(model, layers)
    assert z_tilde.shape == (256, 32, 3)

    edges = build_circuit(z_tilde, top_k_per_atom=2, min_weight=0.0)
    assert len(edges) > 0
    # Only forward edges (l -> l+1).
    for l_src, _, l_dst, _, _ in edges:
        assert l_dst == l_src + 1

    out_dot = tmp_path / "circuit.dot"
    path, edges2 = trace_and_save(
        model, layers, out_dot, top_k_per_atom=2, min_weight=0.0
    )
    text = path.read_text()
    assert text.startswith("digraph"), "DOT must start with `digraph`"
    assert text.rstrip().endswith("}"), "DOT must end with `}`"
    assert "->" in text, "DOT must contain at least one edge"
    assert "cluster_l0" in text and "cluster_l2" in text
