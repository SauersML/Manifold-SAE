"""Distributed Manifold-SAE.

K=1,000,000-atom Manifold-SAE training on cogito-L40 activations across
N GPUs via PyTorch DDP / FSDP, with per-atom Circle-topology latents
maintained by Riemannian retraction at the optimizer step.

Status: scaffolding. See README.md for which pieces are functional vs stubbed.
"""

from .model import ManifoldSAE, ManifoldSAEConfig
from .loss import ComposedLoss, ComposedLossConfig

__all__ = [
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ComposedLoss",
    "ComposedLossConfig",
]
