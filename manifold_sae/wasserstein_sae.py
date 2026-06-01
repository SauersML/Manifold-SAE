"""Wasserstein Dictionary SAE — atoms are probability distributions on S^1.

Reference
---------
Schmitz et al. 2018, "Wasserstein Dictionary Learning" (arXiv:1708.01955)
Mishne et al. 2022, "Geometric Sparse Coding"      (arXiv:2210.12135)

Motivation
----------
The cogito-L40 hue axis is intrinsically circular. A linear sum of point
atoms can't span a circle without shattering it across many atoms ("feature
absorption"). OT barycenters of hue-supported distributions interpolate
along the manifold instead of through ambient space — a single atom is
multi-modal, and the barycenter of two atoms is a hue-arc move.

Pipeline
--------
    x ∈ R^D
      └─ encoder (linear → softmax/tau)     → π ∈ Δ^F          (B, F)
      └─ atoms[k] ∈ Δ^M  = softmax(θ_k)     (F, M) on hue circle
      └─ barycenter = Sinkhorn(atoms, π, ε)                    (B, M)
      └─ readout (linear M → D) + bias                         (B, D)

History note
------------
v1 used raw softmax, ε fixed → π collapsed to one-hot and Sinkhorn
underflowed under sharpening. v2 added τ-annealed softmax, v3 added ε-floor
coupling + nan_to_num + logit clamp. All three are folded into this single
class; legacy ``eps=`` constructor kw is preserved for tests, while the
hardened path activates whenever ``tau_schedule`` or ``logit_clamp`` is
set.

gamfit primitive integration
----------------------------
The Sinkhorn IBP barycenter lives in :mod:`manifold_sae.kernels.sinkhorn`.
gamfit 0.1.134 ships ``gamfit.kernels.sinkhorn_barycenter`` (+ ``_vjp``), but
it is a numpy->numpy primitive: routing the SAE decode through it would force
a host/numpy roundtrip per forward and a hand-rolled ``autograd.Function`` VJP
shim, breaking device residency and autograd-native gradients. The trainable
SAE therefore keeps the torch-native log-domain kernel deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.sinkhorn import circular_cost_matrix, sinkhorn_barycenter


@dataclass
class WassersteinSAEConfig:
    input_dim: int = 7168
    n_features: int = 128
    n_support: int = 64
    eps: float = 0.01
    n_sinkhorn_iter: int = 20
    neighbor_weight: float = 1e-3
    # NaN-hardening + temperature schedule (folded in from v2/v3)
    tau_start: float | None = None        # None → vanilla softmax
    tau_end: float = 1.5
    eps_scale: float = 0.05              # eps = max(eps_floor, eps_scale * tau)
    logit_clamp: float | None = None     # None → no clamp


class WassersteinSAE(nn.Module):
    def __init__(
        self,
        F: int = 128,
        M: int = 64,
        D: int = 7168,
        eps: float = 0.01,
        n_sinkhorn_iter: int = 20,
        neighbor_weight: float = 1e-3,
        *,
        tau_start: float | None = None,
        tau_end: float = 1.5,
        eps_scale: float = 0.05,
        logit_clamp: float | None = None,
    ) -> None:
        super().__init__()
        self.F = int(F)
        self.M = int(M)
        self.D = int(D)
        self.eps_floor = float(eps)
        self.eps_scale = float(eps_scale)
        self.n_sinkhorn_iter = int(n_sinkhorn_iter)
        self.neighbor_weight = float(neighbor_weight)
        self.tau_end = float(tau_end)
        self.logit_clamp = None if logit_clamp is None else float(logit_clamp)
        # Tau buffer always present (used when tau_start is set);
        # a value of 1.0 reduces softmax(logits / tau) to plain softmax.
        self.register_buffer("tau", torch.tensor(float(tau_start if tau_start is not None else 1.0)))
        self._use_tau_schedule = tau_start is not None

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

    # ------------------------------------------------------------------
    # NaN-hardening / scheduling API
    # ------------------------------------------------------------------
    def set_tau(self, value: float) -> None:
        with torch.no_grad():
            self.tau.fill_(float(value))

    def current_eps(self) -> float:
        if self._use_tau_schedule:
            return float(max(self.eps_floor, self.eps_scale * float(self.tau)))
        return self.eps_floor

    # ------------------------------------------------------------------
    def atoms(self) -> torch.Tensor:
        return torch.softmax(self.atom_logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        if self.logit_clamp is not None:
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        if self._use_tau_schedule:
            tau = self.tau.clamp_min(1e-3)
            pi = torch.softmax(logits / tau, dim=-1)
        else:
            pi = torch.softmax(logits, dim=-1)
        if self.logit_clamp is not None or self._use_tau_schedule:
            pi = torch.nan_to_num(pi, nan=1.0 / self.F, posinf=1.0 / self.F, neginf=1.0 / self.F)
            pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return pi

    def decode(self, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.atoms()
        bary = sinkhorn_barycenter(
            atoms, pi, self.C,
            eps=self.current_eps(), n_iter=self.n_sinkhorn_iter,
        )
        if self.logit_clamp is not None or self._use_tau_schedule:
            bary = torch.nan_to_num(bary, nan=1.0 / self.M, posinf=1.0 / self.M, neginf=1.0 / self.M)
            bary = bary / bary.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        recon = self.readout(bary) + self.b_dec
        return recon, bary

    def forward(self, x: torch.Tensor) -> dict:
        pi = self.encode(x)
        recon, bary = self.decode(pi)
        return {"recon": recon, "pi": pi, "bary": bary}

    # ------------------------------------------------------------------
    def neighbor_penalty(self, pi: torch.Tensor) -> torch.Tensor:
        """Mishne-style geometric sparsity: COM-angle weighted co-firing."""
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
        return {"total": total, "mse": mse, "neighbor": neigh,
                "pi": out["pi"], "recon": out["recon"], "bary": out["bary"]}

    @torch.no_grad()
    def atom_compactness(self) -> torch.Tensor:
        atoms = self.atoms()
        ent = -(atoms * torch.log(atoms.clamp(min=1e-30))).sum(-1)
        return 1.0 - ent / torch.log(torch.tensor(self.M, dtype=atoms.dtype))

    @torch.no_grad()
    def pi_entropy(self, x: torch.Tensor) -> torch.Tensor:
        pi = self.encode(x)
        return -(pi * torch.log(pi.clamp_min(1e-30))).sum(-1)
