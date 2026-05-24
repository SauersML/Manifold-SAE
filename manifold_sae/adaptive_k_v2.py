"""AdaptiveK v2 — target-K losses (clipped hinge / squared).

The v1 head collapsed to k_min once the population-mean penalty (λ·E[K]) ramped
in: any reduction in K helped the loss, so the equilibrium was at k_min, not
at the intended target. v2 fixes this with TWO loss variants that have a
non-degenerate fixed point at K_pred ≈ k_target:

  A) clipped hinge:    λ · mean( max(0, (K_pred − k_target)/k_max) )
       — only penalizes K above the target. Below target it is free, so the
         only force pulling K down is the *reconstruction* cost saved by
         pruning a useless atom. Reconstruction pressure pulls K up; clip
         pressure pulls K down when above target. Equilibrium ≈ k_target.

  B) symmetric squared: λ · mean( ((K_pred − k_target)/k_max)² )
       — explicit two-sided target. Fastest equilibration; treats over-/under-
         shoot symmetrically. Per-row spread is suppressed harder than (A).

Both keep the rest of v1 (STE through round, top-K mask, k-head over row
stats). The k-head bias is initialized so K_pred starts near k_target rather
than k_max, removing the warm-up collapse seen in v1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@dataclass
class AdaptiveKv2Config:
    input_dim: int
    n_features: int = 512
    k_target: int = 32
    k_min: int = 4
    k_max: int = 128
    sparsity_weight: float = 1e-2
    loss_kind: str = "clipped"  # "clipped" | "squared"


class AdaptiveKv2SAE(nn.Module):
    """Per-row dynamic top-K SAE with target-K loss.

    Parameters
    ----------
    F : int
        Number of dictionary atoms.
    k_target : int
        Population *and* per-row target K. The loss has its minimum (or
        plateau) at K_pred ≈ k_target.
    k_max : int
        Hard upper bound on K_pred (also caps storage of the top-K mask).
    loss_kind : {"clipped", "squared"}
        Which target-K penalty to use. See module docstring.
    """

    def __init__(
        self,
        input_dim: int,
        F: int = 512,
        k_target: int = 32,
        k_min: int = 4,
        k_max: int = 128,
        sparsity_weight: float = 1e-2,
        loss_kind: str = "clipped",
    ) -> None:
        super().__init__()
        assert 1 <= k_min <= k_target <= k_max <= F, (k_min, k_target, k_max, F)
        assert loss_kind in ("clipped", "squared"), loss_kind
        self.input_dim = int(input_dim)
        self.n_features = int(F)
        self.k_target = int(k_target)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sparsity_weight = float(sparsity_weight)
        self.loss_kind = loss_kind

        self.W_e = nn.Parameter(torch.randn(input_dim, F) * (1.0 / math.sqrt(input_dim)))
        self.b_e = nn.Parameter(torch.zeros(F))
        self.W_d = nn.Parameter(torch.randn(F, input_dim) * (1.0 / math.sqrt(F)))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

        self.k_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Initialize bias so K_pred starts at k_target (not k_max). This
        # removes the v1 chicken-and-egg: with v1's init at k_max, the
        # only direction λ pushed was down, so the head ratchets to k_min.
        # With target-K loss + init at target, the head has no reason to
        # move on day 0 — it co-adapts to row-specific reconstruction.
        target_norm = (self.k_target - self.k_min) / max(self.k_max - self.k_min, 1)
        # sigmoid logit for target_norm; clamp for numerical safety.
        target_norm = float(min(max(target_norm, 1e-3), 1 - 1e-3))
        with torch.no_grad():
            self.k_head[-1].bias.fill_(math.log(target_norm / (1 - target_norm)))

    # ------------------------------------------------------------------

    def predict_k(self, z: torch.Tensor) -> torch.Tensor:
        z_abs = z.abs()
        stats = torch.stack(
            [z_abs.mean(dim=-1), z_abs.amax(dim=-1), z.std(dim=-1)], dim=-1
        )
        logit = self.k_head(stats).squeeze(-1)
        k_norm = torch.sigmoid(logit)
        k_pred = self.k_min + (self.k_max - self.k_min) * k_norm
        return k_pred

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.b_d) @ self.W_e + self.b_e
        k_pred = self.predict_k(z)
        return z, k_pred

    def _apply_topk(self, z: torch.Tensor, k_pred: torch.Tensor) -> torch.Tensor:
        B, Fdim = z.shape
        z_pos = _relu(z)
        sorted_vals, sorted_idx = z_pos.sort(dim=-1, descending=True)
        k_round = k_pred.detach().round().clamp(self.k_min, self.k_max)
        k_ste = k_round + (k_pred - k_pred.detach())
        positions = torch.arange(Fdim, device=z.device).unsqueeze(0)
        keep_mask = (positions < k_round.unsqueeze(-1)).to(z.dtype)
        kept_sorted = sorted_vals * keep_mask
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, sorted_idx, kept_sorted)
        scale = (k_ste / k_round.clamp(min=1.0)).unsqueeze(-1)
        z_sparse = z_sparse * scale
        return z_sparse

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, k_pred = self.encode(x)
        z_sparse = self._apply_topk(z, k_pred)
        recon = z_sparse @ self.W_d + self.b_d
        return recon, z_sparse, k_pred

    # ------------------------------------------------------------------
    # Target-K penalties
    # ------------------------------------------------------------------

    def clipped_k_penalty(self, k_pred: torch.Tensor) -> torch.Tensor:
        """λ-scaled mean of max(0, K_pred - k_target), normalized by k_max."""
        denom = max(self.k_max, 1)
        excess = _relu(k_pred - self.k_target) / denom
        return excess.mean()

    def target_l1_penalty(self, k_pred: torch.Tensor) -> torch.Tensor:
        """Misnamed for back-compat with the user spec: squared deviation."""
        denom = max(self.k_max, 1)
        dev = (k_pred - self.k_target) / denom
        return dev.pow(2).mean()

    def k_penalty(self, k_pred: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "clipped":
            return self.clipped_k_penalty(k_pred)
        return self.target_l1_penalty(k_pred)

    def loss(self, x: torch.Tensor) -> dict:
        recon, z_sparse, k_pred = self.forward(x)
        mse = (recon - x).pow(2).mean()
        sparsity = self.k_penalty(k_pred)
        total = mse + self.sparsity_weight * sparsity
        n_active = (z_sparse.abs() > 0).float().sum(-1).mean()
        return {
            "loss": total,
            "recon": mse,
            "sparsity": sparsity,
            "mean_k_pred": k_pred.mean().detach(),
            "mean_k_actual": n_active.detach(),
            "k_std": k_pred.std().detach(),
            "recon_out": recon,
            "z": z_sparse,
            "k_pred": k_pred,
        }
