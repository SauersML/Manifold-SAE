"""Matryoshka SAE — nested-prefix multi-resolution SAE (gamfit 0.1.123 wrapper).

Shared encoder over ``F`` atoms; per-shell decoder views slice
``W_dec[:s, :]`` for s ∈ shells. Loss = Σ_l (MSE_l + l1·‖z[:s_l]‖₁) —
the analytic kernel ``gamfit.AnalyticPenaltyKind.NESTED_PREFIX``.

gamfit 0.1.123 migration status
-------------------------------
``AnalyticPenaltyKind.NESTED_PREFIX`` enum + manifest entry
(``{"kind": "nested_prefix", "rust": "NestedPrefix:NestedPrefixPenalty",
"python": "NestedPrefixPenalty"}``) ship in 0.1.123. We tag the kind on
the module for REML introspection.

**Gap filed for gamfit**: the manifest names a Python wrapper
``NestedPrefixPenalty`` but it is NOT actually exported from ``gamfit``
(see ``gamfit/_penalties.py:122`` — only the enum value exists). When
gamfit ships the wrapper, replace the per-shell python loop in
``matryoshka_loss`` with ``NestedPrefixPenalty(shells=cfg.shells,
l1_weight=cfg.l1_weight).evaluate(z, recons, x)``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gamfit
    _NESTED_PREFIX = gamfit.AnalyticPenaltyKind.NESTED_PREFIX  # REML-side tag
except Exception:
    gamfit = None
    _NESTED_PREFIX = None


@dataclass
class MatryoshkaSAEConfig:
    input_dim: int
    n_features: int
    shells: Sequence[int] = (64, 128, 256, 512)
    l1_weight: float = 1e-3


class MatryoshkaSAE(nn.Module):
    """Shared encoder, per-shell decoder views (slices of ``W_dec[:s, :]``).

    Tagged with ``self.penalty_kind = AnalyticPenaltyKind.NESTED_PREFIX``.
    """

    def __init__(self, cfg: MatryoshkaSAEConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.shells[-1] == cfg.n_features
        for a, b in zip(cfg.shells, cfg.shells[1:]):
            assert a < b
        D, F_ = cfg.input_dim, cfg.n_features
        self.W_enc = nn.Parameter(torch.randn(D, F_) * (1.0 / D ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(F_))
        self.W_dec = nn.Parameter(torch.randn(F_, D) * (1.0 / F_ ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(D))
        self.penalty_kind = _NESTED_PREFIX

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        z = self.encode(x)
        recons = {s: z[:, :s] @ self.W_dec[:s, :] + self.b_dec for s in self.cfg.shells}
        return {"z": z, "recon_per_shell": recons}


def matryoshka_loss(out, x, cfg, shell_weights=None):
    """Mirrors ``AnalyticPenaltyKind.NESTED_PREFIX`` (Σ_l MSE_l + l1·L1_l)."""
    z = out["z"]
    shells = list(cfg.shells)
    if shell_weights is None:
        shell_weights = [1.0] * len(shells)
    total = x.new_zeros(()); log = {}
    for w, s in zip(shell_weights, shells):
        mse_s = F.mse_loss(out["recon_per_shell"][s], x)
        l1_s = cfg.l1_weight * z[:, :s].abs().mean()
        total = total + w * (mse_s + l1_s)
        log[f"mse_s{s}"] = mse_s.detach(); log[f"l1_s{s}"] = l1_s.detach()
    log["loss"] = total.detach()
    return total, log


@torch.no_grad()
def shell_r2_and_dead(model, X, var_t, bs: int = 256):
    """Per-shell val R² + dead-atom rate."""
    model.eval()
    device = next(model.parameters()).device
    shells = model.cfg.shells
    sse = [0.0] * len(shells); n_total = 0
    fired = torch.zeros(model.cfg.n_features, dtype=torch.bool, device=device)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs].to(device)
        out = model(xb)
        fired |= (out["z"] > 1e-6).any(dim=0)
        for li, s in enumerate(shells):
            sse[li] += F.mse_loss(out["recon_per_shell"][s], xb, reduction="sum").item()
        n_total += xb.numel()
    res = {}
    for li, s in enumerate(shells):
        res[f"r2_s{s}"] = 1.0 - sse[li] / n_total / var_t
        res[f"dead_s{s}"] = float((~fired[:s]).sum().item()) / s
    res["alive_total"] = int(fired.sum().item())
    return res
