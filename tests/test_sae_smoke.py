"""Smoke test: build a tiny gamfit-native ManifoldSAE, forward + backward."""

from __future__ import annotations

import torch

from manifold_sae.losses import total_loss
from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig, SparsityConfig


def test_sae_smoke() -> None:
    torch.manual_seed(0)
    config = ManifoldSAEConfig(
        input_dim=16,
        n_atoms=4,
        intrinsic_rank=1,
        atom_manifold="circle",
        n_basis_per_atom=10,
        sparsity=SparsityConfig(kind="softmax_topk", target_k=2),
    )
    sae = ManifoldSAE(config)

    x = torch.randn(32, 16, dtype=config.dtype)
    out = sae(x)

    # New output schema.
    assert out.x_hat.shape == (32, 16)
    assert out.positions.shape == (32, 4, 1)
    assert out.amplitudes.shape == (32, 4)
    assert out.z.shape == (32, 4)

    losses = total_loss(out, x, sae)
    loss = losses["total"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()
    saw_grad = False
    for name, p in sae.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
            saw_grad = True
    assert saw_grad, "no parameter received a gradient"
