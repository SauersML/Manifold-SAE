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
            # Spread per-feature MEAN positions across [0,1].
            targets = torch.linspace(0.25, 0.75, self.n_features)
            logits = torch.log(targets / (1.0 - targets))
            self.fc2_b.zero_()
            self.fc2_b[:, 0] = logits
            self.fc2_b[:, 1] = 0.0
            # Boost position-head weights so per-token positions actually vary
            # across the batch at init. Without this, position logits have
            # tiny variance, sigmoid saturates near the bias target, gamfit's
            # design is rank-deficient on near-constant positions, and the
            # inner solve returns zero coefficients.
            scale = 3.0 / max(self.hidden_dim, 1) ** 0.5
            self.fc2_w.data[:, :, 0].normal_(0.0, scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_n = self.norm(x)
        h = torch.einsum("bd,fdh->bfh", x_n, self.fc1_w) + self.fc1_b.unsqueeze(0)
        h = self.act(h)
        out = torch.einsum("bfh,fho->bfo", h, self.fc2_w) + self.fc2_b.unsqueeze(0)
        pos_logits = torch.nan_to_num(out[:, :, 0], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(out[:, :, 1], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        positions = torch.sigmoid(pos_logits).clamp(1e-4, 1.0 - 1e-4)
        # Amplitudes as BINARY gates via sigmoid + TopK + straight-through.
        # Standard "manifold features" deliberately do NOT use amp magnitude
        # to encode continuous information — position-on-the-curve is the
        # one-dim degree of freedom. If amp is unconstrained the encoder
        # uses it as a free gauge to leak position information, so amp is
        # forced to {0, 1}-ish via a hard gate.
        amp_soft = torch.sigmoid(amp_logits)  # (B, F) in (0, 1)
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(amp_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(amp_soft)
            hard_mask.scatter_(1, idx, 1.0)
            amplitudes = hard_mask + (amp_soft - amp_soft.detach())
        else:
            # All features always fire (top_k >= n_features). Hard amp = 1.
            hard_mask = torch.ones_like(amp_soft)
            amplitudes = hard_mask + (amp_soft - amp_soft.detach())
        return positions, amplitudes

