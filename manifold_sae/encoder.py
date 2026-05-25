"""Per-feature parallel MLP encoder."""

from __future__ import annotations

import torch
from torch import nn


class ManifoldEncoder(nn.Module):
    """Per-feature parallel MLP. Each feature's MLP receives x augmented with
    the per-feature subspace energy ``||x @ W_k||`` — a strong scalar signal
    of "does this token live in feature k's ambient subspace?" Without this,
    multiple features that look similar to a black-box-on-raw-x encoder
    (e.g., all monotone-in-some-direction curves) end up multiplexed onto
    one SAE feature.
    """

    def __init__(
        self,
        intrinsic_rank: int,
        n_features: int,
        input_dim: int,
        hidden_dim: int | None = None,
        top_k: int | None = None,
        shared_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.intrinsic_rank = intrinsic_rank
        self.n_features = n_features
        self.input_dim = input_dim
        self.top_k = top_k
        self.shared_encoder = bool(shared_encoder)
        # Default H caps at 256: the per-feature `(F, in_dim, H)` weight is F× bigger
        # than a shared encoder. Old default max(4*D, 8*F) silently OOMs at D=7168, F=512.
        H = hidden_dim if hidden_dim is not None else min(256, max(64, 8 * n_features))
        self.hidden_dim = H

        D = input_dim
        F = n_features
        R = intrinsic_rank
        self.norm = nn.LayerNorm(D) if D >= 4 else nn.Identity()
        # Input is concat([x_normalized (D), y_proj_k (R)]) — per-feature
        # MLP sees its own R-dim subspace projection as extra signal.
        in_dim = D + R
        if self.shared_encoder:
            # Shared-trunk + F-head encoder. The trunk consumes only x (D),
            # the per-feature y_proj_k (R) is injected at the head as a
            # lightweight 1×R linear contribution. Memory: O(D·H + H·F·2 + F·R·2)
            # vs the per-feature O(F·D·H + F·H·2) — at D=7168, F=512, H=256 the
            # difference is 0.94 GB (shared) vs 7.3 GB (per-feature) for fc1
            # alone (the old "210 GB" trap was at H = max(4·D, 8·F) = 28672).
            self.trunk_fc1 = nn.Linear(D, H)
            self.head_w = nn.Parameter(torch.randn(F, H, 2) / max(H, 1) ** 0.5)
            self.head_b = nn.Parameter(torch.zeros(F, 2))
            self.head_y = nn.Parameter(torch.randn(F, R, 2) / max(R, 1) ** 0.5)
        else:
            self.fc1_w = nn.Parameter(torch.randn(F, in_dim, H) / max(in_dim, 1) ** 0.5)
            self.fc1_b = nn.Parameter(torch.zeros(F, H))
            self.fc2_w = nn.Parameter(torch.randn(F, H, 2) / max(H, 1) ** 0.5)
            self.fc2_b = nn.Parameter(torch.zeros(F, 2))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, y_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, D). y_proj: (B, F, R), per-feature subspace projection."""
        x_n = self.norm(x)
        B = x_n.shape[0]
        F = y_proj.shape[1]
        if self.shared_encoder:
            # Shared trunk: one (B, D) → (B, H) MLP, broadcast across F.
            h_shared = self.act(self.trunk_fc1(x_n))                 # (B, H)
            h_b = h_shared.unsqueeze(1).expand(B, F, h_shared.shape[-1])
            # Per-feature head + y_proj contribution.
            out = torch.einsum("bfh,fho->bfo", h_b, self.head_w) + self.head_b.unsqueeze(0)
            out = out + torch.einsum("bfr,fro->bfo", y_proj, self.head_y)
        else:
            x_n_b = x_n.unsqueeze(1).expand(B, F, x_n.shape[-1])     # (B, F, D)
            input_per_feature = torch.cat([x_n_b, y_proj], dim=-1)   # (B, F, D+R)
            h = torch.einsum("bfi,fih->bfh", input_per_feature, self.fc1_w) + self.fc1_b.unsqueeze(0)
            h = self.act(h)
            out = torch.einsum("bfh,fho->bfo", h, self.fc2_w) + self.fc2_b.unsqueeze(0)
        z_raw = torch.nan_to_num(out[:, :, 0], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(out[:, :, 1], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)
        if getattr(self, "continuous_amp", False):
            amp_cont = torch.nn.functional.softplus(amp_logits)
            if self.top_k is not None and self.top_k < self.n_features:
                _vals, idx = torch.topk(amp_cont, self.top_k, dim=1)
                gate = torch.zeros_like(amp_cont)
                gate.scatter_(1, idx, 1.0)
                amp_out = amp_cont * gate
            else:
                amp_out = amp_cont
            return z_raw, mask_soft, amp_out
        # Binary straight-through TopK: closes the (amp, curve) gauge by
        # forbidding amplitude from carrying magnitude. Curves absorb it.
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(mask_soft)
            hard_mask.scatter_(1, idx, 1.0)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        else:
            hard_mask = torch.ones_like(mask_soft)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        return z_raw, mask_soft, mask_binary
