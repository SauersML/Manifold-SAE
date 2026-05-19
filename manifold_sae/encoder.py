"""Encoder mapping input activations to (position, amplitude) pairs per feature."""

from __future__ import annotations

import torch
from torch import nn


class ManifoldEncoder(nn.Module):
    """Linear -> GELU -> Linear(2F); split into positions in [0,1] and ReLU amplitudes."""

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(input_dim, 2 * n_features)

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, 2 * n_features)

        self._init_position_bias()

    def _init_position_bias(self) -> None:
        # Spread per-feature MEAN positions across [0, 1] via biased sigmoid logits,
        # but keep small random position-head weights so positions vary across the
        # batch from step 0 — without per-token variance the gamfit inner solve
        # sees a rank-deficient design and the SAE never starts learning. The
        # spec's "Initial curves near zero" failure mode kicks in if this is wrong.
        with torch.no_grad():
            # Keep targets in the steep part of sigmoid (away from saturation)
            # so weight noise translates to real per-token position variance.
            # Training can pull features toward [0, 1] endpoints later.
            targets = torch.linspace(0.25, 0.75, self.n_features)
            logits = torch.log(targets / (1.0 - targets))
            self.fc2.bias.zero_()
            self.fc2.bias[: self.n_features].copy_(logits)
            # Position-head weights are scaled so per-token positions vary
            # substantially across the batch at init. If position variance is too
            # small (~1e-5), gamfit's REML solver silently returns zeros for the
            # near-rank-1 design, the loss carries no signal, and the encoder
            # never learns — the spec's "Initial curves near zero" failure mode.
            # The std below targets ~0.1-0.2 std of post-sigmoid positions at
            # init, which is enough to give the inner solve a well-conditioned
            # design while staying near the per-feature target mean.
            std = 1.0 / max(self.hidden_dim, 1) ** 0.5
            self.fc2.weight.data[: self.n_features].normal_(mean=0.0, std=std)
            # Amplitude-head bias is set so softplus(bias) ≈ 1 at init: every
            # feature is gated ON from step 0. With the previous ReLU + zero
            # bias, ~half of features had amp=0 at init, and amp=0 makes
            # gamfit's `by`-gated design degenerate → grad_by returns 0 →
            # the feature is permanently dead. Softplus has no zero-gradient
            # region, so even small amplitudes can recover.
            self.fc2.bias[self.n_features :].fill_(0.5413)  # softplus(0.5413) ≈ 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        pos_logits, amp_raw = out[:, : self.n_features], out[:, self.n_features :]
        positions = torch.sigmoid(pos_logits)
        amplitudes = torch.nn.functional.softplus(amp_raw)
        return positions, amplitudes
