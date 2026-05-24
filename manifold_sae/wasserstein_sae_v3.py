"""WassersteinSAEv3 — NaN-hardened variant of v2.

Background
----------
v2 NaN'd at epoch ~8 of cogito-L40 training: as τ annealed from 4.0 → 1.0 the
encoder logits sharpened and the Sinkhorn barycenter underflowed when ε was
held fixed at 0.01 with a very-peaky π distribution. The IBP step computes

    log Kᵀu  with K = exp(-C/ε)

and once ε is small AND some π columns approach 0/1 the log-domain updates
overflow on V100 fp32. v3 applies four orthogonal fixes simultaneously:

1. ε FLOOR coupled to τ:  eps = max(eps_floor, eps_scale * tau)
   keeps the Sinkhorn cost well-conditioned as τ → τ_end.
2. Defensive ``nan_to_num`` on π after softmax (uniform 1/F fallback).
3. Clamp on encoder logits BEFORE the temperature divide so a single rogue
   logit can't dominate.
4. τ_end raised to 1.5 (was 1.0) — still sparse-ish but never pathological.

Gradient clipping at norm 1.0 is the trainer's responsibility (we keep that
unchanged from the v2 trainer script).

Does NOT modify wasserstein_sae_v2.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.sinkhorn import circular_cost_matrix, sinkhorn_barycenter


@dataclass
class WassersteinSAEv3Config:
    input_dim: int = 7168
    n_features: int = 128
    n_support: int = 64
    eps_floor: float = 0.01
    eps_scale: float = 0.05  # eps = max(eps_floor, eps_scale * tau)
    n_sinkhorn_iter: int = 20
    neighbor_weight: float = 1e-3
    tau_start: float = 4.0
    tau_end: float = 1.5
    logit_clamp: float = 12.0


class WassersteinSAEv3(nn.Module):
    def __init__(
        self,
        F: int = 128,
        M: int = 64,
        D: int = 7168,
        eps_floor: float = 0.01,
        eps_scale: float = 0.05,
        n_sinkhorn_iter: int = 20,
        neighbor_weight: float = 1e-3,
        tau_start: float = 4.0,
        tau_end: float = 1.5,
        logit_clamp: float = 12.0,
    ) -> None:
        super().__init__()
        self.F = int(F)
        self.M = int(M)
        self.D = int(D)
        self.eps_floor = float(eps_floor)
        self.eps_scale = float(eps_scale)
        self.n_sinkhorn_iter = int(n_sinkhorn_iter)
        self.neighbor_weight = float(neighbor_weight)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.logit_clamp = float(logit_clamp)

        self.encoder = nn.Linear(D, F)

        with torch.no_grad():
            theta = torch.randn(F, M) * 0.1
            centers = torch.linspace(0, M, F + 1)[:F]
            idx = torch.arange(M).float()
            for k in range(F):
                d = (idx - centers[k]).abs()
                d = torch.minimum(d, M - d)
                theta[k] += -((d / (M / 8.0)) ** 2)
        self.atom_logits = nn.Parameter(theta)

        self.readout = nn.Linear(M, D)
        self.b_dec = nn.Parameter(torch.zeros(D))

        self.register_buffer("C", circular_cost_matrix(M))
        self.register_buffer("tau", torch.tensor(float(tau_start)))

    def set_tau(self, value: float) -> None:
        with torch.no_grad():
            self.tau.fill_(float(value))

    def current_eps(self) -> float:
        return float(max(self.eps_floor, self.eps_scale * float(self.tau)))

    def atoms(self) -> torch.Tensor:
        return torch.softmax(self.atom_logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        # Fix #3: clamp logits BEFORE divide so a single huge logit can't
        # dominate at small tau.
        logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        tau = self.tau.clamp_min(1e-3)
        pi = torch.softmax(logits / tau, dim=-1)
        # Fix #2: defensive nan/inf scrub; replace with uniform.
        pi = torch.nan_to_num(pi, nan=1.0 / self.F, posinf=1.0 / self.F, neginf=1.0 / self.F)
        # Re-normalize in case nan_to_num produced a row that doesn't sum to 1.
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return pi

    def decode(self, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.atoms()
        eps = self.current_eps()  # Fix #1: ε coupled to τ.
        bary = sinkhorn_barycenter(
            atoms, pi, self.C,
            eps=eps, n_iter=self.n_sinkhorn_iter,
        )
        # Defensive scrub on barycenter too.
        bary = torch.nan_to_num(bary, nan=1.0 / self.M, posinf=1.0 / self.M, neginf=1.0 / self.M)
        bary = bary / bary.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        recon = self.readout(bary) + self.b_dec
        return recon, bary

    def forward(self, x: torch.Tensor) -> dict:
        pi = self.encode(x)
        recon, bary = self.decode(pi)
        return {"recon": recon, "pi": pi, "bary": bary}

    def neighbor_penalty(self, pi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            atoms = self.atoms()
            M = self.M
            angles = 2 * torch.pi * torch.arange(M, device=atoms.device) / M
            cx = (atoms * torch.cos(angles)).sum(-1)
            cy = (atoms * torch.sin(angles)).sum(-1)
            com_angle = torch.atan2(cy, cx)
            d = (com_angle.unsqueeze(0) - com_angle.unsqueeze(1)).abs()
            d = torch.minimum(d, 2 * torch.pi - d)
            D_atoms = (d / torch.pi) ** 2
        return torch.einsum("bf,bg,fg->", pi, pi, D_atoms) / pi.shape[0]

    def loss(self, x: torch.Tensor) -> dict:
        out = self.forward(x)
        mse = (out["recon"] - x).pow(2).mean()
        neigh = self.neighbor_penalty(out["pi"])
        total = mse + self.neighbor_weight * neigh
        return {
            "total": total, "mse": mse, "neighbor": neigh,
            "pi": out["pi"], "recon": out["recon"], "bary": out["bary"],
        }

    @torch.no_grad()
    def atom_compactness(self) -> torch.Tensor:
        atoms = self.atoms()
        ent = -(atoms * torch.log(atoms.clamp(min=1e-30))).sum(-1)
        return 1.0 - ent / torch.log(torch.tensor(self.M, dtype=atoms.dtype))

    @torch.no_grad()
    def pi_entropy(self, x: torch.Tensor) -> torch.Tensor:
        pi = self.encode(x)
        return -(pi * torch.log(pi.clamp_min(1e-30))).sum(-1)
