"""Encoder mapping input activations to (position, amplitude) pairs per feature."""

from __future__ import annotations

import torch
from torch import nn


class ManifoldEncoder(nn.Module):
    """Per-feature attention encoder: each feature has its OWN small MLP
    mapping (LayerNorm input) -> (position, amplitude). Breaks the shared-
    representation bottleneck that made one-big-MLP encoders settle on
    feature-permutation-symmetric local optima.

    Each feature's MLP: Linear(D -> H) -> GELU -> Linear(H -> 2).
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        hidden_dim: int | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.top_k = top_k
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(2 * input_dim, 32)

        self.norm = nn.LayerNorm(input_dim)
        # Per-feature parallel MLPs implemented as a single (F, D, H) tensor
        # for vectorized forward.
        self.fc1_w = nn.Parameter(torch.randn(n_features, input_dim, self.hidden_dim) / max(input_dim, 1) ** 0.5)
        self.fc1_b = nn.Parameter(torch.zeros(n_features, self.hidden_dim))
        self.fc2_w = nn.Parameter(torch.randn(n_features, self.hidden_dim, 2) / max(self.hidden_dim, 1) ** 0.5)
        self.fc2_b = nn.Parameter(torch.zeros(n_features, 2))
        self.act = nn.GELU()

        self._init_position_bias()

    def _init_position_bias(self) -> None:
        with torch.no_grad():
            # Positions are now unbounded; the SAE's soft-rescaling guarantees
            # per-batch coverage of the basis domain. We just need each
            # feature head to produce distinct (and per-token varying) z_raw
            # at init. Different per-feature biases break feature symmetry.
            self.fc2_b.zero_()
            self.fc2_b[:, 0] = torch.linspace(-1.0, 1.0, self.n_features)
            self.fc2_b[:, 1] = 0.0
            # Position head weights: large enough that z_raw has substantial
            # per-token variance, so soft-rescale produces well-spread positions.
            scale = 3.0 / max(self.hidden_dim, 1) ** 0.5
            self.fc2_w.data[:, :, 0].normal_(0.0, scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (raw_position_logits, mask_soft, mask_binary).

        ``raw_position_logits`` are UNBOUNDED scalar position outputs — the
        caller (ManifoldSAE) does soft per-batch min-max rescaling to map
        them into the basis domain. Decoupling encoder output from gamfit
        domain prevents the position-clustering-at-init failure mode.

        ``mask_soft`` is sigmoid(amp_logit) — used for the cumulant
        identifiability loss (continuous probabilities). ``mask_binary`` is
        the straight-through TopK result — used as gamfit's by-gate.
        """
        x_n = self.norm(x)
        h = torch.einsum("bd,fdh->bfh", x_n, self.fc1_w) + self.fc1_b.unsqueeze(0)
        h = self.act(h)
        out = torch.einsum("bfh,fho->bfo", h, self.fc2_w) + self.fc2_b.unsqueeze(0)
        # Position is unbounded; SAE will soft-rescale.
        z_raw = torch.nan_to_num(out[:, :, 0], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(out[:, :, 1], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)  # (B, F)
        # Binary mask via straight-through TopK.
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(mask_soft)
            hard_mask.scatter_(1, idx, 1.0)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        else:
            hard_mask = torch.ones_like(mask_soft)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        return z_raw, mask_soft, mask_binary

