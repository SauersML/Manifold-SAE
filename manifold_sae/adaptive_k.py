"""AdaptiveK SAE — per-input dynamic top-K (arxiv 2508.17320).

The model predicts a continuous K_pred(x) per input via a learned gate head.
That K is rounded (with straight-through gradient) to an integer in
[k_min, k_max], and only the top-K_pred features (by encoder pre-activation
magnitude) are kept for reconstruction.

Sparsity is supervised in expectation: λ · E[K_pred] replaces a fixed top-K
in the loss. This lets the model spend more atoms on hard rows and fewer on
easy rows — matching or beating fixed-TopK at the same MEAN sparsity.

Design notes
------------
* Encoder: tied to a single linear (W_e, b_e). Decoder: W_d, b_d.
* K-head: shared linear from the encoder pre-activation (mean/maxpool over
  features) to a scalar logit → softplus + offset → continuous K_pred.
  We use the encoder's own statistics so the head co-adapts.
* Top-K_pred selection: rank features by ReLU(z); the top-⌈K_pred⌉ get their
  ReLU(z) value, the rest are zero. Gradient on K_pred flows through a
  straight-through bias term added to the kept activations (so reducing K
  has a smooth-ish cost during training).
* For row r with K_r = ⌈k_min + (k_max−k_min)·σ(k_logit_r)⌉ — sigmoid keeps
  K bounded; the loss term λ·K_pred (continuous) pushes mean K down.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from gamfit.torch import IBPAssignmentPenalty  # gamfit >= 0.1.123


@dataclass
class AdaptiveKSAEConfig:
    input_dim: int
    n_features: int = 512
    k_min: int = 8
    k_max: int = 128
    sparsity_weight: float = 1e-3


class AdaptiveKSAE(nn.Module):
    """Per-row dynamic top-K SAE."""

    def __init__(
        self,
        input_dim: int,
        F: int = 512,
        k_min: int = 8,
        k_max: int = 128,
        sparsity_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        assert 1 <= k_min < k_max <= F, (k_min, k_max, F)
        self.input_dim = int(input_dim)
        self.n_features = int(F)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sparsity_weight = float(sparsity_weight)

        self.W_e = nn.Parameter(torch.randn(input_dim, F) * (1.0 / math.sqrt(input_dim)))
        self.b_e = nn.Parameter(torch.zeros(F))
        self.W_d = nn.Parameter(torch.randn(F, input_dim) * (1.0 / math.sqrt(F)))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

        # K-head: input is feature-pooled stats (mean(|z|), max(|z|), std(z))
        # → scalar K logit per row.
        self.k_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Initialize bias so K_pred starts near k_max (so the model first
        # learns to reconstruct, then the λ pressure trims K down).
        with torch.no_grad():
            self.k_head[-1].bias.fill_(2.0)  # sigmoid(2) ≈ 0.88

        # gamfit IBP-assignment prior over the encoder pre-activation logits:
        # finite Indian-Buffet-Process penalty with expected K = k_max·alpha.
        # Acts as a richer Bayesian-nonparametric stand-in for the hand-rolled
        # population-mean K penalty (which has a degenerate fixed point at
        # k_min; see auto_exp_74 for why v2 added clipped-hinge to work around
        # that). The IBP prior pulls per-row K toward `alpha` atoms without
        # collapsing to zero.
        self.ibp_prior = IBPAssignmentPenalty(
            k_max=F, alpha=float(k_min) / float(F), tau=1.0,
        )

    # ------------------------------------------------------------------
    # K prediction
    # ------------------------------------------------------------------

    def predict_k(self, z: torch.Tensor) -> torch.Tensor:
        """Continuous K_pred per row, in [k_min, k_max]."""
        z_abs = z.abs()
        stats = torch.stack(
            [z_abs.mean(dim=-1), z_abs.amax(dim=-1), z.std(dim=-1)], dim=-1
        )
        logit = self.k_head(stats).squeeze(-1)
        k_norm = torch.sigmoid(logit)  # (B,)
        k_pred = self.k_min + (self.k_max - self.k_min) * k_norm
        return k_pred

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.b_d) @ self.W_e + self.b_e
        k_pred = self.predict_k(z)
        return z, k_pred

    def _apply_topk(self, z: torch.Tensor, k_pred: torch.Tensor) -> torch.Tensor:
        """Per-row top-⌈k_pred⌉. Straight-through on K through a small bias."""
        B, F = z.shape
        z_pos = F_relu(z)  # nonneg activations
        # Sort once descending; cheaper than per-row topk for the K-varying mask.
        sorted_vals, sorted_idx = z_pos.sort(dim=-1, descending=True)

        # Hard integer K per row (clamped, rounded). STE: gradient flows via k_pred.
        k_round = k_pred.detach().round().clamp(self.k_min, self.k_max)
        k_ste = k_round + (k_pred - k_pred.detach())  # straight-through

        # Mask: position i kept iff i < k_round.
        positions = torch.arange(F, device=z.device).unsqueeze(0)  # (1, F)
        keep_mask = (positions < k_round.unsqueeze(-1)).to(z.dtype)  # (B, F)

        # Apply mask in sorted space, scatter back.
        kept_sorted = sorted_vals * keep_mask
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, sorted_idx, kept_sorted)

        # Differentiable contribution from k_ste: scale by (k_ste / k_round) so
        # gradient on k_pred reaches the loss without changing forward values.
        scale = (k_ste / k_round.clamp(min=1.0)).unsqueeze(-1)
        z_sparse = z_sparse * scale
        return z_sparse

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, k_pred = self.encode(x)
        z_sparse = self._apply_topk(z, k_pred)
        recon = z_sparse @ self.W_d + self.b_d
        return recon, z_sparse, k_pred

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x: torch.Tensor) -> dict:
        recon, z_sparse, k_pred = self.forward(x)
        mse = (recon - x).pow(2).mean()
        # gamfit IBPAssignmentPenalty over the pre-activation logits — a
        # Bayesian-nonparametric prior over per-row K that subsumes the v1
        # population-mean penalty. Divide by N·F so the descriptor's sum
        # reduces to a mean and `sparsity_weight` keeps its scale.
        z_logits = (x - self.b_d) @ self.W_e + self.b_e
        ibp_val = self.ibp_prior(z_logits)
        sparsity = ibp_val / float(z_logits.numel())
        total = mse + self.sparsity_weight * sparsity
        n_active = (z_sparse.abs() > 0).float().sum(-1).mean()
        return {
            "loss": total,
            "recon": mse,
            "sparsity": sparsity,
            "mean_k_pred": k_pred.mean().detach(),
            "mean_k_actual": n_active.detach(),
            "recon_out": recon,
            "z": z_sparse,
            "k_pred": k_pred,
        }


# Module-level relu alias (avoid name shadowing of F=n_features arg).
def F_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)
