"""Complete Replacement Model (CRM) skeleton.

Chains per-layer SAEs and inter-layer transcoders in series:

    x_0 → SAE_0 → x̂_0 → Tcoder_{0→1} → x̂_1 → SAE_1 → x̂'_1 → ...

For an L-layer stack we have L SAEs and L−1 transcoders. The training loss
is the sum of (a) per-stage reconstruction at each SAE and (b) per-stage
activation match — predicted vs ground-truth layer activations.

This is a skeleton: the architecture, end-to-end loss, and training loop
work. Attribution-graph extraction (which SAE features mediate a specific
prediction) is a follow-up — the per-stage SAE.encode results already
expose the sparse latent dictionaries needed for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import math
import torch
from torch import nn
import torch.nn.functional as F


class SimpleTopKSAE(nn.Module):
    """Minimal TopK SAE (reused for each layer of the CRM)."""

    def __init__(self, d_in: int, n_feat: int, top_k: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_feat = n_feat
        self.top_k = top_k
        self.W_e = nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / math.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_feat))
        self.W_d = nn.Parameter(torch.randn(n_feat, d_in) * (1.0 / math.sqrt(n_feat)))
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.b_d) @ self.W_e + self.b_e
        topv, topi = z.topk(self.top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topi, F.relu(topv))
        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_d + self.b_d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class Transcoder(nn.Module):
    """Sparse MLP-replacement: maps layer-l SAE latents → layer-(l+1) activations.

    Architecture: latent → ReLU(W h + b) (overcomplete sparse mid) → linear out.
    """

    def __init__(self, d_in: int, d_out: int, n_mid: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.W_e = nn.Parameter(torch.randn(d_in, n_mid) * (1.0 / math.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_mid))
        self.W_d = nn.Parameter(torch.randn(n_mid, d_out) * (1.0 / math.sqrt(n_mid)))
        self.b_d = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.W_e + self.b_e
        topv, topi = z.topk(self.top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topi, F.relu(topv))
        out = z_sparse @ self.W_d + self.b_d
        return out, z_sparse


@dataclass
class CRMConfig:
    layer_dims: List[int]                       # [d_0, d_1, ..., d_{L-1}]
    n_features_per_sae: int = 512
    sae_top_k: int = 32
    transcoder_mid: int = 1024
    transcoder_top_k: int = 64


class CompleteReplacementModel(nn.Module):
    """Chains L SAEs and L−1 transcoders for full-model replacement."""

    def __init__(self, config: CRMConfig) -> None:
        super().__init__()
        self.config = config
        self.L = len(config.layer_dims)
        assert self.L >= 2, "Need at least 2 layers"

        self.saes = nn.ModuleList(
            [
                SimpleTopKSAE(d, config.n_features_per_sae, config.sae_top_k)
                for d in config.layer_dims
            ]
        )
        self.transcoders = nn.ModuleList(
            [
                Transcoder(
                    config.layer_dims[l],
                    config.layer_dims[l + 1],
                    config.transcoder_mid,
                    config.transcoder_top_k,
                )
                for l in range(self.L - 1)
            ]
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> dict:
        """Forward chained replacement.

        xs: list of L tensors, each (B, d_l) — the ground-truth activations
            at each layer for the same input batch.

        Returns dict with per-stage reconstructions and latents.
        """
        assert len(xs) == self.L, (len(xs), self.L)

        recons: list[torch.Tensor] = []
        latents_sae: list[torch.Tensor] = []
        latents_tc: list[torch.Tensor] = []

        # Chain: feed each SAE the *output of the previous transcoder*
        # (or the original x_0 at the first layer). Compare each
        # reconstruction against the ground-truth at that layer.
        prev_x = xs[0]
        for l in range(self.L):
            recon_l, z_l = self.saes[l](prev_x)
            recons.append(recon_l)
            latents_sae.append(z_l)
            if l < self.L - 1:
                tc_out, tc_z = self.transcoders[l](recon_l)
                latents_tc.append(tc_z)
                prev_x = tc_out
        return {
            "recons": recons,
            "latents_sae": latents_sae,
            "latents_tc": latents_tc,
        }

    def loss(
        self,
        xs: Sequence[torch.Tensor],
        recon_weight: float = 1.0,
        match_weight: float = 1.0,
    ) -> dict:
        out = self.forward(xs)
        per_stage = []
        total = xs[0].new_zeros(())
        for l in range(self.L):
            mse_l = (out["recons"][l] - xs[l]).pow(2).mean()
            per_stage.append(mse_l.detach())
            # Layer-0 recon is the "input reconstruction"; downstream are
            # activation-match terms.
            w = recon_weight if l == 0 else match_weight
            total = total + w * mse_l
        return {
            "loss": total,
            "per_stage_mse": per_stage,
            "out": out,
        }

    @torch.no_grad()
    def per_stage_r2(self, xs: Sequence[torch.Tensor]) -> List[float]:
        out = self.forward(xs)
        r2s: list[float] = []
        for l in range(self.L):
            res = (out["recons"][l] - xs[l]).pow(2).mean().item()
            var = xs[l].var().item()
            r2s.append(1.0 - res / max(var, 1e-12))
        return r2s
