"""Pure-Python Matryoshka SAE — nested-prefix multi-resolution SAE.

Nested shells of feature indices [0:s_1], [0:s_2], ..., [0:s_L] (s_1 < s_2 < ... < s_L = F).
ONE shared encoder produces F-dim activations. Each shell's reconstruction uses
ONLY the first s_l atoms; loss = sum over shells of (MSE_l + l1_w * L1(z_{:s_l})).

Coarsest shell (smallest s) gets trained-against EVERY step, so its atoms are
forced to be the most universally useful → coarse concepts. Outer shells specialize.

NO Rust / gamfit dependency. Built as a small linear SAE (no curve atoms — just
the classic encoder/decoder pair). For curve-atom Matryoshka, wrap ManifoldSAE
with the same nested-prefix loss instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MatryoshkaSAEConfig:
    input_dim: int
    n_features: int                          # outer-most F (e.g. 512)
    shells: Sequence[int] = (64, 128, 256, 512)
    l1_weight: float = 1e-3
    use_topk: bool = False                   # if True, hard top_k per shell
    top_k_per_shell: Sequence[int] | None = None  # required if use_topk


class MatryoshkaSAE(nn.Module):
    """Shared encoder, per-shell decoder views (slicing the same W_dec)."""

    def __init__(self, cfg: MatryoshkaSAEConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.shells[-1] == cfg.n_features, \
            f"last shell {cfg.shells[-1]} must equal n_features {cfg.n_features}"
        for a, b in zip(cfg.shells, cfg.shells[1:]):
            assert a < b, f"shells must be strictly increasing, got {cfg.shells}"
        D, F_ = cfg.input_dim, cfg.n_features

        # encoder: x → z (F,)
        self.W_enc = nn.Parameter(torch.randn(D, F_) * (1.0 / (D ** 0.5)))
        self.b_enc = nn.Parameter(torch.zeros(F_))
        # decoder: z → x_hat (D,). Per-shell reconstruction slices W_dec[:s, :].
        self.W_dec = nn.Parameter(torch.randn(F_, D) * (1.0 / (F_ ** 0.5)))
        self.b_dec = nn.Parameter(torch.zeros(D))

        if cfg.use_topk:
            assert cfg.top_k_per_shell is not None and len(cfg.top_k_per_shell) == len(cfg.shells)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (pre-activation-masked) z ∈ R^(B, F)."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode_shell(self, z: torch.Tensor, s: int, top_k: int | None = None) -> torch.Tensor:
        """Reconstruct using only the first s atoms."""
        z_s = z[:, :s]
        if top_k is not None and top_k < s:
            # keep top_k by magnitude, zero else
            topv, topi = z_s.topk(top_k, dim=-1)
            mask = torch.zeros_like(z_s)
            mask.scatter_(1, topi, 1.0)
            z_s = z_s * mask
        return z_s @ self.W_dec[:s, :] + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        """Returns dict with per-shell recons + the shared z. Loss is computed
        by ``matryoshka_loss`` outside so trainers can weight shells."""
        z = self.encode(x)                              # (B, F)
        per_shell: dict[int, torch.Tensor] = {}
        for li, s in enumerate(self.cfg.shells):
            tk = None
            if self.cfg.use_topk:
                tk = int(self.cfg.top_k_per_shell[li])
            per_shell[s] = self.decode_shell(z, s, top_k=tk)
        return {"z": z, "recon_per_shell": per_shell}


def matryoshka_loss(
    out: dict,
    x: torch.Tensor,
    cfg: MatryoshkaSAEConfig,
    shell_weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sum over shells of (MSE_l + l1 * mean(|z_{:s_l}|))."""
    z = out["z"]
    shells = list(cfg.shells)
    if shell_weights is None:
        shell_weights = [1.0] * len(shells)
    assert len(shell_weights) == len(shells)
    total = x.new_zeros(())
    log = {}
    for w, s in zip(shell_weights, shells):
        recon_s = out["recon_per_shell"][s]
        mse_s = F.mse_loss(recon_s, x)
        l1_s = cfg.l1_weight * z[:, :s].abs().mean()
        total = total + w * (mse_s + l1_s)
        log[f"mse_s{s}"] = mse_s.detach()
        log[f"l1_s{s}"] = l1_s.detach()
    log["loss"] = total.detach()
    return total, log


@torch.no_grad()
def shell_r2_and_dead(model: MatryoshkaSAE, X: torch.Tensor, var_t: float,
                      bs: int = 256) -> dict:
    """Per-shell val R² + per-shell dead-rate (frac of atoms never fires)."""
    model.eval()
    device = next(model.parameters()).device
    n_shells = len(model.cfg.shells)
    sse_per_shell = [0.0] * n_shells
    n_total = 0
    fired_per_atom = torch.zeros(model.cfg.n_features, dtype=torch.bool, device=device)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs].to(device)
        out = model(xb)
        fired_per_atom |= (out["z"] > 1e-6).any(dim=0)
        for li, s in enumerate(model.cfg.shells):
            r = out["recon_per_shell"][s]
            sse_per_shell[li] += F.mse_loss(r, xb, reduction="sum").item()
        n_total += xb.numel()
    res = {}
    for li, s in enumerate(model.cfg.shells):
        mse = sse_per_shell[li] / n_total
        res[f"r2_s{s}"] = 1.0 - mse / var_t
        res[f"dead_s{s}"] = float((~fired_per_atom[:s]).sum().item()) / s
    res["alive_total"] = int(fired_per_atom.sum().item())
    return res
