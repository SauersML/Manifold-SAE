"""Smoke test: build a tiny ManifoldSAE, forward + backward on random data."""

from __future__ import annotations

import torch

from manifold_sae.losses import total_loss
from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig


def test_sae_smoke() -> None:
    torch.manual_seed(0)
    config = ManifoldSAEConfig(input_dim=16, n_features=4, n_basis=10, top_k=2)
    sae = ManifoldSAE(config)

    x = torch.randn(32, 16)
    out = sae(x)
    losses = total_loss(out, x, config)
    loss = losses["total"]

    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()
    for name, p in sae.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
