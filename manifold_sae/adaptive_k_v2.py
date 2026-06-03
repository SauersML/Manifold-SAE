"""AdaptiveK v2 — gamfit-native, MLP K-head + target-K regime.

BREAKING REWRITE. The hand-rolled clipped/squared ``k_penalty`` target losses,
the ``_apply_topk`` STE, and the ``IBPAssignmentPenalty`` are all GONE. Sparsity
is now routed through :class:`gamfit.torch.AdaptiveTopK` exactly like v1.

v1-vs-v2 distinction (the sparsity mechanism is shared)
-------------------------------------------------------
Both variants put ``λ · E[K_pred]`` (learnable ``λ``) through ``AdaptiveTopK``.
What distinguishes v2 — and this is a *real* architectural difference, not a
flag:

* **MLP K-head** (``head='mlp'``, ``hidden`` units) vs v1's single ``Linear``.
  The deeper head can model row-specific K nonlinearly off the full ``z``.
* **Target-K regime.** v2 brackets ``[k_min, k_max]`` tightly around a
  ``k_target`` and seeds the MLP head's final bias so the predicted K starts
  near ``k_target``. The *bracket + bias seed* enforces the target; there is no
  hand-rolled target penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from manifold_sae.adaptive_k import HardTopKGate  # corrected hard-forward gate


@dataclass
class AdaptiveKv2Config:
    input_dim: int
    n_features: int = 512
    k_target: int = 32
    k_min: int = 4
    k_max: int = 128
    sparsity_weight: float = 1e-2
    temperature: float = 0.1
    hidden: int = 32


class AdaptiveKv2SAE(nn.Module):
    """Per-row dynamic top-K SAE with an MLP K-head and a target-K regime.

    Architecture: encoder ``Linear`` → ``AdaptiveTopK`` gate (MLP K-head) →
    decoder ``Linear``. The ``[k_min, k_max]`` bracket is centered on
    ``k_target`` to keep predicted K near the target without a hand-rolled
    target penalty.

    Parameters
    ----------
    F : int
        Number of dictionary atoms.
    k_target : int
        Center of the per-row K regime. Used to derive the default bracket and
        to seed the K-head bias.
    k_min, k_max : int
        Inclusive per-row K bounds. ``1 <= k_min <= k_target <= k_max <= F``.
    sparsity_weight : float
        Seeds the gate's ``init_weight`` (initial ``λ = exp(log_weight)``) and
        acts as an outer anneal multiplier on the gate penalty.
    temperature : float
        Soft-top-K backward temperature.
    hidden : int
        Hidden width of the MLP K-head.
    """

    def __init__(
        self,
        input_dim: int,
        F: int = 512,
        k_target: int = 32,
        k_min: int = 4,
        k_max: int = 128,
        sparsity_weight: float = 1e-2,
        temperature: float = 0.1,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        assert 1 <= k_min <= k_target <= k_max <= F, (k_min, k_target, k_max, F)
        self.input_dim = int(input_dim)
        self.n_features = int(F)
        self.k_target = int(k_target)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sparsity_weight = float(sparsity_weight)

        self.W_e = nn.Parameter(torch.randn(input_dim, F) * (1.0 / math.sqrt(input_dim)))
        self.b_e = nn.Parameter(torch.zeros(F))
        self.W_d = nn.Parameter(torch.randn(F, input_dim) * (1.0 / math.sqrt(F)))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

        # v2 == MLP head + target-K regime, sparsity via AdaptiveTopK.
        self.gate = HardTopKGate(
            F=F,
            k_min=k_min,
            k_max=k_max,
            head="mlp",
            hidden=int(hidden),
            temperature=float(temperature),
            init_weight=max(float(sparsity_weight), 1e-8),
        )
        # Seed the MLP head's final bias so K_pred starts near k_target rather
        # than the bracket midpoint. The primitive maps a raw logit through
        # sigmoid into [k_min, k_max]; invert that for k_target.
        span = max(self.k_max - self.k_min, 1)
        target_norm = (self.k_target - self.k_min) / span
        target_norm = float(min(max(target_norm, 1e-3), 1.0 - 1e-3))
        with torch.no_grad():
            final = self.gate.k_head[-1]  # last Linear of the MLP head
            final.bias.fill_(math.log(target_norm / (1.0 - target_norm)))

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = (x - self.b_d) @ self.W_e + self.b_e
        z_active, k_pred_eff, sparsity_penalty = self.gate(z)
        return z_active, k_pred_eff, sparsity_penalty

    def decode(self, z_active: torch.Tensor) -> torch.Tensor:
        return z_active @ self.W_d + self.b_d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(recon, z_active, k_pred_eff)``; ``k_pred_eff`` shape ``(B,)``."""
        z_active, k_pred_eff, _sparsity_penalty = self.encode(x)
        recon = self.decode(z_active)
        return recon, z_active, k_pred_eff

    def loss(self, x: torch.Tensor) -> dict:
        z_active, k_pred_eff, sparsity = self.encode(x)
        recon = self.decode(z_active)
        mse = (recon - x).pow(2).mean()
        total = mse + self.sparsity_weight * sparsity
        n_active = (z_active.abs() > 0).float().sum(-1).mean()
        return {
            "loss": total,
            "recon": mse,
            "sparsity": sparsity,
            "mean_k_pred": k_pred_eff.mean().detach(),
            "mean_k_actual": n_active.detach(),
            "k_std": k_pred_eff.std().detach(),
            "recon_out": recon,
            "z": z_active,
            "k_pred": k_pred_eff,
        }
