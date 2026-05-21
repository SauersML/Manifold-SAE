"""Shared MLP encoder for LLM-scale Manifold-SAE.

A single 2-layer MLP shared across features (NOT per-feature).
Same scaling as a vanilla SAE encoder × small constant — fully
feedforward, parameter-cost ~ D*H + H*F (linear in F).

Two heads on the shared trunk: one for position logits, one for
amplitude logits. The hidden width is chosen as 4*D by default (same
ratio as a TransformerLens-style "wide enough" encoder for vanilla SAEs).

The nonlinearity is necessary for curve features: reading "position
along feature k's curve" from x is a non-linear function of x in
general. Vanilla TopK SAEs don't need it because they only need to
read "which feature fired".
"""

from __future__ import annotations

import torch
from torch import nn


class ManifoldEncoderLinear(nn.Module):
    """Shared MLP encoder. Despite the class name (kept for back-compat),
    this is a 2-layer MLP with two heads.
    """

    def __init__(
        self,
        intrinsic_rank: int,
        n_features: int,
        input_dim: int,
        top_k: int | None = None,
        hidden_dim: int | None = None,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.intrinsic_rank = intrinsic_rank
        self.n_features = n_features
        self.input_dim = input_dim
        self.top_k = top_k

        D = input_dim
        F = n_features
        # H = 4·D by default (vanilla TopK SAE convention). DO NOT scale
        # with F — that makes the encoder O(F²) and infeasible at LLM
        # scale (F=100K → 20B params just for the heads). Override via
        # hidden_dim when you actually want a bigger trunk.
        H = hidden_dim if hidden_dim is not None else 4 * D
        self.hidden_dim = H

        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H, bias=True)
        self.act = nn.GELU()
        self.head_t = nn.Linear(H, F, bias=True)
        self.head_a = nn.Linear(H, F, bias=True)
        nn.init.normal_(self.fc1.weight, std=1.0 / D ** 0.5)
        nn.init.normal_(self.head_t.weight, std=1.0 / H ** 0.5)
        nn.init.normal_(self.head_a.weight, std=1.0 / H ** 0.5)

    def forward(self, x: torch.Tensor, y_proj: torch.Tensor | None = None):
        x_n = self.norm(x)
        h = self.act(self.fc1(x_n))
        z_raw = self.head_t(h)
        amp_logits = self.head_a(h)
        z_raw = torch.nan_to_num(z_raw, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(amp_logits, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)
        if getattr(self, "continuous_amp", False):
            # Continuous nonneg magnitude with TopK gating. Magnitude lives
            # in amp; curve carries shape only (proper LLM-style gauge).
            amp_cont = torch.nn.functional.softplus(amp_logits)
            if self.top_k is not None and self.top_k < self.n_features:
                _vals, idx = torch.topk(amp_cont, self.top_k, dim=1)
                gate = torch.zeros_like(amp_cont)
                gate.scatter_(1, idx, 1.0)
                amp_out = amp_cont * gate
            else:
                amp_out = amp_cont
            return z_raw, mask_soft, amp_out
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(mask_soft)
            hard_mask.scatter_(1, idx, 1.0)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        else:
            hard_mask = torch.ones_like(mask_soft)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        return z_raw, mask_soft, mask_binary
