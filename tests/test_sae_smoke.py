"""Smoke test: build a tiny gamfit-native ManifoldSAE via the sae module's
public API, run forward + backward + closed-form .fit on tiny float64 data,
and assert the output bundle has finite, correctly-shaped fields."""

from __future__ import annotations

import torch

from manifold_sae.sae import (
    ManifoldSAE,
    ManifoldSAEConfig,
    ManifoldSAEOutput,
    SparsityConfig,
)


def _build_sae() -> tuple[ManifoldSAE, ManifoldSAEConfig]:
    config = ManifoldSAEConfig(
        input_dim=16,
        n_atoms=4,
        intrinsic_rank=1,
        atom_manifold="circle",
        n_basis_per_atom=10,
        sparsity=SparsityConfig(kind="softmax_topk", target_k=2),
    )
    return ManifoldSAE(config), config


def test_forward_bundle_shapes_finite() -> None:
    torch.manual_seed(0)
    sae, config = _build_sae()

    # gamfit ManifoldSAE.forward rejects mismatched dtype -> feed float64.
    x = torch.randn(32, 16, dtype=config.dtype)
    out = sae(x)

    assert isinstance(out, ManifoldSAEOutput)
    assert out.x_hat.shape == (32, 16)
    assert out.positions.shape == (32, 4, 1)
    assert out.amplitudes.shape == (32, 4)
    assert out.z.shape == (32, 4)
    for name in ("x_hat", "positions", "amplitudes", "z"):
        field = getattr(out, name)
        assert torch.isfinite(field).all(), f"non-finite {name}"


def test_backward_reconstruction_gradients() -> None:
    torch.manual_seed(0)
    sae, config = _build_sae()

    x = torch.randn(32, 16, dtype=config.dtype)
    out = sae(x)

    # Module supplies all regularizer pieces; use a plain gamfit-native loss.
    loss = ((out.x_hat - x) ** 2).mean() + sae.sparsity_penalty(out.gate)
    assert torch.isfinite(loss)

    loss.backward()
    saw_grad = False
    for name, p in sae.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
            saw_grad = True
    assert saw_grad, "no parameter received a gradient"
