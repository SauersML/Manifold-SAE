"""Composed loss for distributed Manifold-SAE training.

Total loss = MSE(recon, x)
           + w_ibp     * IBP-Gumbel KL (prior over per-row activation count)
           + w_iso     * isometry penalty on tangent frames T_k
           + w_ard     * ARD penalty on per-atom log-amplitude precision
           + w_mech    * mechanism-sparsity (encourages atom usage sparsity
                                              across the data distribution)
           + w_anchor  * anchor displacement penalty (per auto_exp_44/47)
           + w_tangent * tangent magnitude penalty

Per the gamfit composition-engine philosophy (MEMORY: project_gamfit_…)
each term is its own callable, weights are config-driven, and the
top-level ComposedLoss is just a sum-reduction harness. This makes it
straightforward to swap or extend individual penalties.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ComposedLossConfig:
    w_recon: float = 1.0
    w_ibp: float = 1e-2
    w_iso: float = 1e-3
    w_ard: float = 1e-4
    w_mech: float = 1e-3
    w_anchor: float = 1e-4
    w_tangent: float = 1e-5
    # IBP target rate: per-row expected K_eff. Default equals top_k.
    ibp_target_rate: float | None = None
    # ARD: per-atom amp variance prior shape (inverse-gamma alpha, beta).
    ard_alpha: float = 1.0
    ard_beta: float = 1.0


# ---------------------------------------------------------------------------
# Individual penalty terms.
# ---------------------------------------------------------------------------
def recon_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Standard MSE per element, summed over D then averaged over B."""
    return ((x - recon) ** 2).sum(dim=-1).mean()


def ibp_gumbel_kl(mask_soft: torch.Tensor, target_rate: float) -> torch.Tensor:
    """KL(q || Bern(p)) for the soft mask, with p = target_rate / K.

    Encourages per-row active count toward target_rate.
    """
    K = mask_soft.shape[-1]
    p = target_rate / K
    q = mask_soft.clamp(1e-6, 1 - 1e-6)
    kl = q * (q.log() - torch.log(torch.tensor(p, device=q.device))) + \
         (1 - q) * ((1 - q).log() - torch.log(torch.tensor(1 - p, device=q.device)))
    # Sum over atoms (KL is per-Bernoulli), mean over batch.
    return kl.sum(dim=-1).mean()


def isometry_penalty(tangent: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
    """‖T_k^T T_k − I_d‖²_F summed over active atoms.

    `active_idx` is a (B, k) tensor of atom indices used this step. Restricting
    the penalty to active atoms keeps the cost O(B*k*d^2) instead of O(K*d^2).
    """
    # Gather active tangent frames: (B*k, D, d)
    flat_idx = active_idx.flatten()
    T = tangent[flat_idx]                          # (N, D, d)
    d = T.shape[-1]
    gram = T.transpose(-1, -2) @ T                 # (N, d, d)
    I = torch.eye(d, device=T.device, dtype=T.dtype).expand_as(gram)
    return ((gram - I) ** 2).flatten(start_dim=-2).sum(dim=-1).mean()


def ard_penalty(amp: torch.Tensor, mask: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Per-atom inverse-gamma ARD on amplitude precision.

    Atoms that are rarely active accumulate large precision → driven to zero.
    Implemented as a closed-form negative log marginal under IG(alpha, beta)
    prior on per-atom amp variance.
    """
    # Per-atom sum of squared amps (weighted by mask), and active count.
    a2 = (amp * mask) ** 2                          # (B, K)
    s = a2.sum(dim=0)                               # (K,)
    n = mask.sum(dim=0).clamp_min(1.0)              # (K,)
    # IG-Gaussian marginal: ½(n+2α) log(β + s/2) - already normalized constants
    # dropped. Mean over atoms keeps the scale comparable across K.
    return ((n + 2 * alpha) * 0.5 * torch.log(beta + 0.5 * s)).mean()


def mechanism_sparsity(mask: torch.Tensor) -> torch.Tensor:
    """L1 on per-atom activation rate across the batch.

    Encourages a small fraction of K atoms to do the work — distinct from
    per-row sparsity, which IBP-Gumbel already handles.
    """
    rate = mask.mean(dim=0)                         # (K,)
    return rate.sum()


def anchor_penalty(anchor: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
    """L2 on anchor norm for active atoms (per auto_exp_44/47)."""
    flat_idx = active_idx.flatten()
    A = anchor[flat_idx]                            # (N, D)
    return (A ** 2).sum(dim=-1).mean()


def tangent_magnitude_penalty(tangent: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
    """L2 on tangent Frobenius norm for active atoms."""
    flat_idx = active_idx.flatten()
    T = tangent[flat_idx]                           # (N, D, d)
    return (T ** 2).flatten(start_dim=-2).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Composed loss.
# ---------------------------------------------------------------------------
class ComposedLoss:
    """Composition harness. Call __call__(x, model_out, model) → (total, dict)."""

    def __init__(self, cfg: ComposedLossConfig, top_k: int):
        self.cfg = cfg
        self.target_rate = cfg.ibp_target_rate if cfg.ibp_target_rate is not None else float(top_k)

    def __call__(self, x: torch.Tensor, out: dict, model) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        recon = out["recon"]
        mask_soft = out["mask_soft"]
        mask_hard = out["mask_hard"]
        amp = out["amp"]

        # active_idx: (B, top_k) — recompute from mask_hard for the penalty restrictions.
        active_idx = mask_hard.topk(model.cfg.top_k, dim=-1).indices

        terms: dict[str, torch.Tensor] = {}
        terms["recon"] = recon_mse(x, recon)
        terms["ibp"] = ibp_gumbel_kl(mask_soft, self.target_rate)
        terms["iso"] = isometry_penalty(model.tangent, active_idx)
        terms["ard"] = ard_penalty(amp, mask_hard, cfg.ard_alpha, cfg.ard_beta)
        terms["mech"] = mechanism_sparsity(mask_hard)
        terms["anchor"] = anchor_penalty(model.anchor, active_idx)
        terms["tangent"] = tangent_magnitude_penalty(model.tangent, active_idx)

        total = (
            cfg.w_recon * terms["recon"]
            + cfg.w_ibp * terms["ibp"]
            + cfg.w_iso * terms["iso"]
            + cfg.w_ard * terms["ard"]
            + cfg.w_mech * terms["mech"]
            + cfg.w_anchor * terms["anchor"]
            + cfg.w_tangent * terms["tangent"]
        )

        log = {k: float(v.detach()) for k, v in terms.items()}
        log["total"] = float(total.detach())
        return total, log
