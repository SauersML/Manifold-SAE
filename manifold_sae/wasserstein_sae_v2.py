"""WassersteinSAEv2 — fixes encoder-π collapse to one-hot under raw softmax.

Problem (from v1)
-----------------
With `π = softmax(W_enc x)`, the encoder rapidly drives π_max → 1.0 within ~100
steps. The barycenter degenerates to "pick one atom," losing the multi-modal
interpolation that motivates the Wasserstein dictionary architecture in the
first place. Effectively the model becomes a hard-routing 1-of-F selector — no
geodesic blending happens.

Fix
---
Apply a temperature τ on the encoder logits:

    π = softmax(encoder(x) / τ)

with τ annealed from `tau_start` → `tau_end` across training. Large τ keeps π
diffuse (forces multi-atom blending early); annealing down lets a sparser code
emerge while still tying the loss to multiple atoms long enough for them to
specialize over the hue circle.

The trainer (`scripts/train_wasserstein_sae_v2.py`) sets `model.tau` at the start
of each epoch via a linear schedule. The class itself can also be used standalone
with a fixed tau or a registered buffer.

This file does NOT modify `wasserstein_sae.py`; it sits alongside.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.sinkhorn import circular_cost_matrix, sinkhorn_barycenter


@dataclass
class WassersteinSAEv2Config:
    input_dim: int = 7168
    n_features: int = 128
    n_support: int = 64
    eps: float = 0.01
    n_sinkhorn_iter: int = 20
    neighbor_weight: float = 1e-3
    tau_start: float = 4.0
    tau_end: float = 1.0


class WassersteinSAEv2(nn.Module):
    """Same pipeline as v1 with a temperature on the encoder softmax.

    Tau handling:
      - `model.tau` is a non-trainable buffer (so it moves to device with `.to`)
        that the trainer overwrites each epoch.
      - The two anchor values `tau_start`, `tau_end` are stored for reference.
    """

    def __init__(
        self,
        F: int = 128,
        M: int = 64,
        D: int = 7168,
        eps: float = 0.01,
        n_sinkhorn_iter: int = 20,
        neighbor_weight: float = 1e-3,
        tau_start: float = 4.0,
        tau_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.F = int(F)
        self.M = int(M)
        self.D = int(D)
        self.eps = float(eps)
        self.n_sinkhorn_iter = int(n_sinkhorn_iter)
        self.neighbor_weight = float(neighbor_weight)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)

        # Encoder logits.
        self.encoder = nn.Linear(D, F)

        # Atoms parameterized as logits over M-point S^1.
        with torch.no_grad():
            theta = torch.randn(F, M) * 0.1
            centers = torch.linspace(0, M, F + 1)[:F]
            idx = torch.arange(M).float()
            for k in range(F):
                d = (idx - centers[k]).abs()
                d = torch.minimum(d, M - d)
                theta[k] += -((d / (M / 8.0)) ** 2)
        self.atom_logits = nn.Parameter(theta)

        # Readout: hue distribution → ambient.
        self.readout = nn.Linear(M, D)
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Cost matrix and current temperature (mutable buffer).
        self.register_buffer("C", circular_cost_matrix(M))
        self.register_buffer("tau", torch.tensor(float(tau_start)))

    # ------------------------------------------------------------------
    def set_tau(self, value: float) -> None:
        """Update the encoder temperature in-place (called by the trainer)."""
        with torch.no_grad():
            self.tau.fill_(float(value))

    def atoms(self) -> torch.Tensor:
        return torch.softmax(self.atom_logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        # tau is a 0-d buffer; clamp to avoid division by zero.
        tau = self.tau.clamp_min(1e-3)
        return torch.softmax(logits / tau, dim=-1)

    def decode(self, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.atoms()
        bary = sinkhorn_barycenter(
            atoms, pi, self.C,
            eps=self.eps, n_iter=self.n_sinkhorn_iter,
        )
        recon = self.readout(bary) + self.b_dec
        return recon, bary

    def forward(self, x: torch.Tensor) -> dict:
        pi = self.encode(x)
        recon, bary = self.decode(pi)
        return {"recon": recon, "pi": pi, "bary": bary}

    # ------------------------------------------------------------------
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
        """Per-row entropy H(π_b) in nats. > log(3) means truly multi-atom blend."""
        pi = self.encode(x)
        return -(pi * torch.log(pi.clamp_min(1e-30))).sum(-1)
