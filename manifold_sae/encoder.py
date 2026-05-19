"""Per-feature parallel MLP encoder."""

from __future__ import annotations

import torch
from torch import nn


class ManifoldEncoder(nn.Module):
    def __init__(
        self,
        intrinsic_rank: int,
        n_features: int,
        input_dim: int,
        hidden_dim: int | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.intrinsic_rank = intrinsic_rank
        self.n_features = n_features
        self.input_dim = input_dim
        self.top_k = top_k
        H = hidden_dim if hidden_dim is not None else max(4 * input_dim, 8 * n_features)
        self.hidden_dim = H

        D = input_dim
        F = n_features
        self.norm = nn.LayerNorm(D) if D >= 4 else nn.Identity()
        self.fc1_w = nn.Parameter(torch.randn(F, D, H) / max(D, 1) ** 0.5)
        self.fc1_b = nn.Parameter(torch.zeros(F, H))
        self.fc2_w = nn.Parameter(torch.randn(F, H, 2) / max(H, 1) ** 0.5)
        self.fc2_b = nn.Parameter(torch.zeros(F, 2))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_n = self.norm(x)
        h = torch.einsum("bd,fdh->bfh", x_n, self.fc1_w) + self.fc1_b.unsqueeze(0)
        h = self.act(h)
        out = torch.einsum("bfh,fho->bfo", h, self.fc2_w) + self.fc2_b.unsqueeze(0)
        z_raw = torch.nan_to_num(out[:, :, 0], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(out[:, :, 1], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(mask_soft)
            hard_mask.scatter_(1, idx, 1.0)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        else:
            hard_mask = torch.ones_like(mask_soft)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        return z_raw, mask_soft, mask_binary
