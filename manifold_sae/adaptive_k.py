"""AdaptiveK SAE — per-input dynamic top-K (arxiv 2508.17320), gamfit-native.

BREAKING REWRITE. The hand-rolled K-head, the ``_apply_topk`` STE, and the
``IBPAssignmentPenalty`` sparsity path are all GONE. The model is now a thin
SAE shell (encoder ``Linear`` → :class:`gamfit.torch.AdaptiveTopK` gate →
decoder ``Linear``) built directly on the gamfit primitive.

What changed vs the old internals
----------------------------------
* The K-head now consumes the **full** encoder pre-activation ``z`` (F dims),
  not 3 pooled stats. This is :class:`AdaptiveTopK`'s own learned head.
* Top-K selection uses :class:`AdaptiveTopK`'s analytic sigmoid-relaxed
  soft-top-K STE (hard mask forward, smooth backward). No ``k_ste``/``k_round``
  scaling hack.
* Sparsity is ``λ · E[K_pred]`` with a **learnable** ``log_weight`` exposed by
  the primitive (REML/LAML-selectable via :meth:`AdaptiveTopK.reml_descriptor`).
  The fixed ``sparsity_weight`` only seeds ``init_weight``; after construction
  the primitive owns λ as a trainable parameter.

v1-vs-v2 distinction
--------------------
The sparsity *mechanism* is now identical for both (``AdaptiveTopK.penalty()``).
We keep the variants meaningful by **head architecture + K regime**:

* **v1 (this file):** ``head='linear'`` — a single ``Linear(F, 1)`` K-head, the
  simplest predictor; a wide ``[k_min, k_max]`` exploration regime.
* **v2 (:mod:`adaptive_k_v2`):** ``head='mlp'`` — a two-layer ``Linear→GELU→
  Linear`` K-head with hidden units, concentrated around a ``k_target`` regime
  (tight ``[k_min, k_max]`` bracket) for sharper per-row K control.

Both are documented; neither re-implements sparsity by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from gamfit.torch import AdaptiveTopK  # gamfit (post-refactor wheel)


@dataclass
class AdaptiveKSAEConfig:
    input_dim: int
    n_features: int = 512
    k_min: int = 8
    k_max: int = 128
    sparsity_weight: float = 1e-3
    temperature: float = 0.1


class AdaptiveKSAE(nn.Module):
    """Per-row dynamic top-K SAE built on :class:`gamfit.torch.AdaptiveTopK`.

    Architecture: encoder ``Linear`` → ``AdaptiveTopK`` gate (linear K-head) →
    decoder ``Linear``. Sparsity is ``λ · E[K_pred]`` with ``λ`` a learnable
    parameter inside the gate.

    Parameters
    ----------
    input_dim : int
        Input activation width ``D``.
    F : int
        Number of dictionary atoms.
    k_min, k_max : int
        Inclusive per-row K bounds. ``1 <= k_min <= k_max <= F``.
    sparsity_weight : float
        Seeds the gate's ``init_weight`` (initial ``λ = exp(log_weight)``).
        After construction the gate owns ``λ`` as a trainable parameter.
    temperature : float
        Soft-top-K backward temperature for the gate STE.
    """

    def __init__(
        self,
        input_dim: int,
        F: int = 512,
        k_min: int = 8,
        k_max: int = 128,
        sparsity_weight: float = 1e-3,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        assert 1 <= k_min <= k_max <= F, (k_min, k_max, F)
        self.input_dim = int(input_dim)
        self.n_features = int(F)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sparsity_weight = float(sparsity_weight)

        # Keep these exact parameter names: integration.py / leaderboard_v2.py
        # read W_e / W_d / b_d directly off the state_dict.
        self.W_e = nn.Parameter(torch.randn(input_dim, F) * (1.0 / math.sqrt(input_dim)))
        self.b_e = nn.Parameter(torch.zeros(F))
        self.W_d = nn.Parameter(torch.randn(F, input_dim) * (1.0 / math.sqrt(F)))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

        # gamfit primitive owns the K-head + STE + learnable-λ sparsity.
        # v1 == linear head (simplest predictor), wide K regime.
        self.gate = AdaptiveTopK(
            F=F,
            k_min=k_min,
            k_max=k_max,
            head="linear",
            temperature=float(temperature),
            init_weight=max(float(sparsity_weight), 1e-8),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z_active, k_pred_eff, sparsity_penalty)`` from the gated encoder."""
        z = (x - self.b_d) @ self.W_e + self.b_e
        z_active, k_pred_eff, sparsity_penalty = self.gate(z)
        return z_active, k_pred_eff, sparsity_penalty

    def decode(self, z_active: torch.Tensor) -> torch.Tensor:
        return z_active @ self.W_d + self.b_d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(recon, z_active, k_pred_eff)``.

        ``k_pred_eff`` is the gate's per-row effective K (``≈`` soft-mask mass),
        shape ``(B,)`` — same shape as the old ``k_pred`` so callers that index
        ``out[2]`` per row keep working.
        """
        z_active, k_pred_eff, _sparsity_penalty = self.encode(x)
        recon = self.decode(z_active)
        return recon, z_active, k_pred_eff

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x: torch.Tensor) -> dict:
        z_active, k_pred_eff, sparsity = self.encode(x)
        recon = self.decode(z_active)
        mse = (recon - x).pow(2).mean()
        # Sparsity is λ·E[K_pred] from the primitive on this exact batch. λ is
        # learnable inside the gate, so the outer sparsity_weight only seeded it.
        # We still expose a multiplier so trainers can anneal an extra scale on
        # top of the learnable λ.
        total = mse + self.sparsity_weight * sparsity
        n_active = (z_active.abs() > 0).float().sum(-1).mean()
        return {
            "loss": total,
            "recon": mse,
            "sparsity": sparsity,
            "mean_k_pred": k_pred_eff.mean().detach(),
            "mean_k_actual": n_active.detach(),
            "recon_out": recon,
            "z": z_active,
            "k_pred": k_pred_eff,
        }
