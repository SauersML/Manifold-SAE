"""LinearSAE: an LRH-style baseline for head-to-head comparison with ManifoldSAE.

This is the canonical sparse autoencoder: a linear encoder with ReLU + bias produces
per-feature amplitudes; a tied-norm linear decoder reconstructs the input. Loss is
reconstruction MSE plus an L1 amplitude penalty. An optional TopK gate replaces the
soft L1 with a hard top-k mask over amplitudes (Anthropic-style).

The training-budget knobs mirror what ManifoldSAE training accepts so the eval harness
can train both with the same compute. We deliberately do not depend on ``manifold_sae``
internals here -- this module is the LRH baseline, kept self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import torch
from torch import nn


@dataclass
class LinearSAEConfig:
    input_dim: int
    n_features: int
    sparsity_weight: float = 1e-3
    top_k: int | None = None  # if set, TopK gate replaces L1
    tie_decoder_norm: bool = True  # normalize decoder columns to unit norm
    lr: float = 1e-3
    batch_size: int = 256
    n_steps: int = 2000
    log_every: int = 100
    seed: int = 0


class LinearSAE(nn.Module):
    """Standard SAE: encoder Linear(D,F)+ReLU+bias, decoder Linear(F,D) with unit-norm columns.

    forward(x) returns (reconstruction, amplitudes); encode/decode are exposed so
    steering can mutate amplitudes between the two halves.
    """

    def __init__(self, config: LinearSAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.n_features, bias=True)
        # Decoder: F -> D. We treat columns of decoder.weight (shape (D,F)) as the
        # per-feature dictionary atoms. Tied-norm means each column is unit-L2.
        self.decoder = nn.Linear(config.n_features, config.input_dim, bias=True)
        # Pre-decoder bias subtracted from x before encoding ("centering bias"); this is
        # standard in Anthropic / OpenAI SAE recipes.
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        with torch.no_grad():
            if config.tie_decoder_norm:
                w = self.decoder.weight  # (D, F)
                w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))
            # Tie encoder init to decoder^T so initial features are well-aligned.
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    @torch.no_grad()
    def _renorm_decoder(self) -> None:
        if not self.config.tie_decoder_norm:
            return
        w = self.decoder.weight
        w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x - self.pre_bias)
        amps = torch.relu(h)
        if self.config.top_k is not None and self.config.top_k > 0:
            k = min(self.config.top_k, amps.shape[-1])
            topk_vals, topk_idx = torch.topk(amps, k=k, dim=-1)
            mask = torch.zeros_like(amps)
            mask.scatter_(-1, topk_idx, 1.0)
            amps = amps * mask
        return amps

    def decode(self, amplitudes: torch.Tensor) -> torch.Tensor:
        return self.decoder(amplitudes) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amps = self.encode(x)
        recon = self.decode(amps)
        return recon, amps


def linear_sae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    amplitudes: torch.Tensor,
    config: LinearSAEConfig,
) -> dict[str, torch.Tensor]:
    mse = torch.mean((recon - target) ** 2)
    # When TopK is active the L1 term is redundant but harmless at small weight.
    sparsity = amplitudes.abs().mean()
    total = mse + config.sparsity_weight * sparsity
    return {"mse": mse, "sparsity": sparsity, "total": total}


def _iter_batches(activations: torch.Tensor, batch_size: int, generator: torch.Generator) -> Iterable[torch.Tensor]:
    n = activations.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
        yield activations[idx]


def train_linear_sae(
    activations: torch.Tensor,
    config: LinearSAEConfig,
    device: torch.device | None = None,
) -> LinearSAE:
    """Train a LinearSAE on (N, D) activations and return the fitted module.

    Mirrors ManifoldSAE training in spirit: Adam, fixed n_steps, random minibatches.
    Decoder columns are re-normalized after each step when ``tie_decoder_norm`` is set.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be (N, D); got shape {tuple(activations.shape)}")
    if activations.shape[1] != config.input_dim:
        raise ValueError(
            f"activations dim {activations.shape[1]} != config.input_dim {config.input_dim}"
        )

    torch.manual_seed(config.seed)
    device = device or activations.device
    sae = LinearSAE(config).to(device)
    activations = activations.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    batch_iter = _iter_batches(activations.cpu(), config.batch_size, generator)
    for step in range(config.n_steps):
        batch = next(batch_iter).to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, amps = sae(batch)
        losses = linear_sae_loss(recon, batch, amps, config)
        losses["total"].backward()
        optimizer.step()
        sae._renorm_decoder()

        if step % config.log_every == 0:
            print(
                f"[linear-sae step {step:6d}] "
                f"mse={losses['mse'].item():.4e} "
                f"sparsity={losses['sparsity'].item():.4e} "
                f"l0={(amps > 0).float().sum(dim=-1).mean().item():.2f}"
            )

    return sae


__all__ = ["LinearSAE", "LinearSAEConfig", "linear_sae_loss", "train_linear_sae"]
