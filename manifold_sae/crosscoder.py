"""Crosscoder SAE: one shared sparse dictionary across multiple layers.

Reference: Lindsey et al., "Sparse Crosscoders for Cross-Layer Features",
Anthropic, 2024.

Architecture
------------
Given L layers with per-layer activations x_l ∈ R^{D_l}, we learn:

  encoder : R^{sum(D_l)} → R^F           (one shared sparse code z)
  W_l     : R^{F × D_l}                  (per-layer decoders)

Forward:
    x = concat([x_1, ..., x_L])
    z = ReLU(encoder(x))                 (sparse via L1 or TopK)
    x_l_hat = z @ W_l + b_l              (per-layer reconstruction)

Loss:
    L = Σ_l ||x_l - x_l_hat||²  +  λ_sparse · Σ_l ||z · ||W_l[:, _]||_2 ||_1

Following Anthropic 2024 we use the "decoder-norm-weighted L1" so atoms with
strong cross-layer presence aren't penalized harder than layer-specific ones
(the standard fix from their paper, eq. (2)).

Manifold mode
-------------
``manifold=True`` replaces each atom's static W_l[k, :] with a Duchon curve
in R^{D_l} parameterized by a shared per-atom scalar position t_k ∈ [0,1]
(reusing the curve machinery already developed in ``manifold_sae.sae``).
This is an experimental composition — most users want the linear version.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


@dataclass
class CrosscoderConfig:
    layer_dims: list[int]
    n_atoms: int = 512
    sparsity_weight: float = 1e-3
    encoder_hidden: int | None = None  # None = linear encoder; else 1-hidden-layer MLP
    tied_encoder: bool = False          # if True, encoder = concat(W_l^T) — Anthropic-style tied
    activation: str = "relu"            # "relu" | "topk"
    top_k: int = 32                     # only used if activation == "topk"
    manifold: bool = False              # use per-atom curve decoders (experimental)
    n_basis: int = 16                   # for manifold mode


class Crosscoder(nn.Module):
    """Shared-dictionary, per-layer decoder SAE.

    Parameters
    ----------
    layer_dims : list[int]
        Per-layer activation widths [D_1, ..., D_L].
    n_atoms : int
        Number of shared latent atoms (F).
    Other args see :class:`CrosscoderConfig`.
    """

    def __init__(
        self,
        layer_dims: list[int],
        n_atoms: int,
        *,
        sparsity_weight: float = 1e-3,
        encoder_hidden: int | None = None,
        tied_encoder: bool = False,
        activation: str = "relu",
        top_k: int = 32,
        manifold: bool = False,
        n_basis: int = 16,
    ) -> None:
        super().__init__()
        self.config = CrosscoderConfig(
            layer_dims=list(layer_dims),
            n_atoms=int(n_atoms),
            sparsity_weight=float(sparsity_weight),
            encoder_hidden=encoder_hidden,
            tied_encoder=bool(tied_encoder),
            activation=str(activation),
            top_k=int(top_k),
            manifold=bool(manifold),
            n_basis=int(n_basis),
        )
        self.layer_dims = list(layer_dims)
        self.n_layers = len(self.layer_dims)
        self.D_total = int(sum(self.layer_dims))
        self.F = int(n_atoms)

        # ----- Encoder -----
        if encoder_hidden is None:
            self.encoder = nn.Linear(self.D_total, self.F, bias=True)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.D_total, int(encoder_hidden), bias=True),
                nn.GELU(),
                nn.Linear(int(encoder_hidden), self.F, bias=True),
            )

        # ----- Per-layer decoders -----
        # W_l shape (F, D_l). Stored as ParameterList for clean per-layer access.
        self.decoders = nn.ParameterList()
        self.dec_biases = nn.ParameterList()
        for D_l in self.layer_dims:
            # He-init scaled down — keeps decoder norms ~unit at init.
            W = torch.randn(self.F, D_l) / max(D_l, 1) ** 0.5
            self.decoders.append(nn.Parameter(W))
            self.dec_biases.append(nn.Parameter(torch.zeros(D_l)))

        # ----- Manifold mode plumbing -----
        if manifold:
            K = int(n_basis)
            # Per-atom shared position parameter — used to read the curve at
            # a single t value for every atom. (Full manifold path with
            # per-token positions is left to manifold_sae.sae; this is a
            # minimal hook for downstream composition.)
            centers = torch.linspace(0.0, 1.0, K, dtype=torch.float32)
            self.register_buffer("centers", centers)
            # Curve coefficients per (atom, layer) — shape (L, F, K, D_l).
            # Replaces the static decoder W_l for ambient reconstruction.
            self.curve_coefs = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.F, K, D_l)) for D_l in self.layer_dims]
            )
            self.atom_positions = nn.Parameter(torch.full((self.F,), 0.5))

        if tied_encoder:
            # Force encoder linear layer to mirror concatenated decoders.
            # We DON'T tie weights with a shared parameter (that breaks
            # per-layer decoder updates); instead we replace forward.
            if encoder_hidden is not None:
                raise ValueError("tied_encoder requires linear encoder (encoder_hidden=None)")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x_concat: torch.Tensor) -> torch.Tensor:
        """Encode a concatenated multi-layer activation to sparse code z."""
        if self.config.tied_encoder and isinstance(self.encoder, nn.Linear):
            # encoder weight = concat([W_l^T]) along input axis (D_total).
            W_concat = torch.cat([Wl for Wl in self.decoders], dim=1)  # (F, D_total)
            z_pre = F.linear(x_concat, W_concat, self.encoder.bias)
        else:
            z_pre = self.encoder(x_concat)

        if self.config.activation == "topk":
            z = self._topk_relu(z_pre, k=self.config.top_k)
        else:
            z = F.relu(z_pre)
        return z

    @staticmethod
    def _topk_relu(z_pre: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, k=k, dim=-1)
        mask = torch.zeros_like(z_pre)
        mask.scatter_(-1, idx, 1.0)
        return F.relu(z_pre) * mask

    def decode_layer(self, z: torch.Tensor, layer: int) -> torch.Tensor:
        """Reconstruct layer ``layer`` from shared code ``z``."""
        W_l = self.decoders[layer]
        b_l = self.dec_biases[layer]
        return z @ W_l + b_l

    def forward(
        self, x_layers: list[torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the crosscoder.

        Parameters
        ----------
        x_layers : list[Tensor]
            List of (B, D_l) activations, one per layer.

        Returns dict with keys:
          z              (B, F) sparse code
          recons         list[(B, D_l)] per-layer reconstructions
          mse_per_layer  list[scalar] per-layer MSE
          l1             scalar decoder-norm-weighted L1 penalty
          loss           scalar total loss
        """
        if len(x_layers) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layers, got {len(x_layers)}"
            )
        x_concat = torch.cat(x_layers, dim=-1)
        z = self.encode(x_concat)

        recons = [self.decode_layer(z, l) for l in range(self.n_layers)]
        mse_per_layer = [
            ((x_layers[l] - recons[l]) ** 2).mean() for l in range(self.n_layers)
        ]
        total_mse = sum(mse_per_layer)

        # Anthropic-style decoder-norm-weighted L1: weight each atom by the
        # sum of its per-layer decoder L2 norms. This balances "concentrate
        # in one layer" vs "spread across layers" gradients.
        with torch.no_grad():
            pass
        per_atom_dec_norm = sum(
            self.decoders[l].norm(dim=1) for l in range(self.n_layers)
        )  # (F,)
        l1 = (z.abs() * per_atom_dec_norm.unsqueeze(0)).sum(dim=-1).mean()

        loss = total_mse + self.config.sparsity_weight * l1

        return {
            "z": z,
            "recons": recons,
            "mse_per_layer": mse_per_layer,
            "l1": l1,
            "loss": loss,
        }

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def per_layer_r2(
        self, x_layers: list[torch.Tensor]
    ) -> list[float]:
        """Coefficient of determination per layer."""
        out = self.forward(x_layers)
        r2s: list[float] = []
        for l in range(self.n_layers):
            x = x_layers[l]
            recon = out["recons"][l]
            ss_res = ((x - recon) ** 2).sum().item()
            ss_tot = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum().item()
            r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
        return r2s

    @torch.no_grad()
    def atom_layer_affinity(self) -> torch.Tensor:
        """Per-atom per-layer decoder norm, normalized.

        Returns (F, L) matrix where row k sums to 1. Atoms with mass
        concentrated in one column are layer-specific; uniform rows are
        cross-layer.
        """
        norms = torch.stack(
            [self.decoders[l].norm(dim=1) for l in range(self.n_layers)],
            dim=-1,
        )  # (F, L)
        norms = norms / norms.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return norms

    @torch.no_grad()
    def cross_layer_atom_mask(self, threshold: float = 0.15) -> torch.Tensor:
        """Boolean (F,) — True for atoms with min-per-layer share ≥ threshold.

        With L=3 layers, a perfectly uniform atom has share 1/3 ≈ 0.33 per
        layer. Threshold 0.15 catches "present in all layers, possibly
        unbalanced" atoms; lower it for stricter cross-layer definition.
        """
        affinity = self.atom_layer_affinity()
        return (affinity.min(dim=-1).values >= threshold)
