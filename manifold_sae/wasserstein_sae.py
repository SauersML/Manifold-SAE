"""Wasserstein Dictionary SAE — atoms are probability distributions on S^1.

Reference
---------
Schmitz et al. 2018, "Wasserstein Dictionary Learning" (arXiv:1708.01955)
Mishne et al. 2022, "Geometric Sparse Coding" (arXiv:2210.12135)

Motivation
----------
The cogito-L40 hue axis is intrinsically circular (S^1). A linear sum of
point-atoms can't span a circle without shattering it across many atoms
("feature absorption"). OT barycenters of hue-supported distributions
interpolate along the manifold instead of through ambient space — a single
atom can be multi-modal, and the barycenter of two atoms is a hue-arc move.

Pipeline
--------
    x ∈ R^D
      └─ encoder (linear → softmax)        → π ∈ Δ^F          (B, F)
                                              simplex weights over F atoms
      └─ atoms[k] ∈ Δ^M     parameterized as softmax(θ_k)     (F, M)
      └─ barycenter = Sinkhorn(atoms, π, ε)                    (B, M)
                                              on the M-point hue circle
      └─ readout (linear M → D) + bias                         (B, D)

Loss
----
    MSE(x, x̂)
    + λ_neighbor · Σ_b Σ_{(k, l) far apart} π_b[k] π_b[l]
        (Mishne's neighborhood penalty: encourages weight to concentrate
         on geodesically-nearby atoms — measured by ½-Wasserstein dist
         between their hue distributions, computed once per epoch as a
         buffer.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.sinkhorn import circular_cost_matrix, sinkhorn_barycenter


@dataclass
class WassersteinSAEConfig:
    input_dim: int = 7168
    n_features: int = 128       # F atoms
    n_support: int = 64         # M support points on S^1
    eps: float = 0.01           # Sinkhorn regularization
    n_sinkhorn_iter: int = 20
    neighbor_weight: float = 1e-3
    readout_hidden: int | None = None


class WassersteinSAE(nn.Module):
    def __init__(self, F: int = 128, M: int = 64, D: int = 7168,
                 eps: float = 0.01, n_sinkhorn_iter: int = 20,
                 neighbor_weight: float = 1e-3) -> None:
        super().__init__()
        self.F = int(F)
        self.M = int(M)
        self.D = int(D)
        self.eps = float(eps)
        self.n_sinkhorn_iter = int(n_sinkhorn_iter)
        self.neighbor_weight = float(neighbor_weight)

        # Encoder: linear x → R^F, softmax → simplex Δ^F.
        self.encoder = nn.Linear(D, F)

        # Atoms: parameterized as logits θ ∈ R^(F, M), softmax → Δ^M.
        # Init each atom as a smooth bump centered at a unique hue angle.
        with torch.no_grad():
            theta = torch.randn(F, M) * 0.1
            centers = torch.linspace(0, M, F + 1)[:F]
            idx = torch.arange(M).float()
            for k in range(F):
                d = (idx - centers[k]).abs()
                d = torch.minimum(d, M - d)
                theta[k] += -((d / (M / 8.0)) ** 2)
        self.atom_logits = nn.Parameter(theta)

        # Readout: M-bin hue distribution → R^D ambient.
        self.readout = nn.Linear(M, D)
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Cost matrix on the hue circle — registered buffer so it moves with
        # `.to(device)`.
        C = circular_cost_matrix(M)
        self.register_buffer("C", C)

    # ------------------------------------------------------------------
    def atoms(self) -> torch.Tensor:
        return torch.softmax(self.atom_logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.encoder(x), dim=-1)

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
        """Mishne-style geometric sparsity.

        Computes Σ_b Σ_{k,l} π_b[k] π_b[l] · d(atom_k, atom_l)^2
        where d is a cheap proxy for ½-Wasserstein distance between atoms:
        the circular distance between atom centers of mass.

        Atoms that fire together should be NEARBY on the hue circle, not
        scattered across it.
        """
        with torch.no_grad():
            atoms = self.atoms()                                       # (F, M)
            M = self.M
            # Embed M-point circle in R^2, compute COM, recover angle.
            angles = 2 * torch.pi * torch.arange(M, device=atoms.device) / M
            cx = (atoms * torch.cos(angles)).sum(-1)
            cy = (atoms * torch.sin(angles)).sum(-1)
            com_angle = torch.atan2(cy, cx)                            # (F,)
            d = (com_angle.unsqueeze(0) - com_angle.unsqueeze(1)).abs()
            d = torch.minimum(d, 2 * torch.pi - d)
            D_atoms = (d / torch.pi) ** 2                              # (F, F) in [0,1]
        # gradient flows through pi only — D_atoms is a slowly-changing buffer
        # (Mishne treats it as a fixed neighborhood graph).
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
        """Per-atom hue-arc compactness ∈ [0, 1]; 1 = single bin, 0 = uniform.

        Returns 1 − H(atom) / log(M).
        """
        atoms = self.atoms()
        ent = -(atoms * torch.log(atoms.clamp(min=1e-30))).sum(-1)
        return 1.0 - ent / torch.log(torch.tensor(self.M, dtype=atoms.dtype))
