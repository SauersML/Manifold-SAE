"""SINDy-SAE: atoms as governing-equation rows, not static feature directions.

Reference: Brunton et al. 2016 (SINDy), arXiv 2507.18220 (SINDy-LOM, 2025).

Standard SAEs learn  z = enc(x)  where atoms are static directions in activation
space. SINDy-SAE instead learns

    dz/dt ≈ Θ φ(z)

where φ(z) is a fixed library of nonlinear basis functions of the state z
(identity, square, cube, sin, cos, pairwise products) and Θ ∈ R^{state_dim × P}
is a SPARSE coefficient matrix. Each row Θ[i] is a "governing-equation atom"
that specifies a sparse ODE governing the i-th state coordinate.

Sparsity in Θ is enforced via an L1 penalty (soft) and optional STLSQ-style
hard thresholding at inference time (Brunton 2016 §III).

Cogito caveat
-------------
Cogito-L40 is harvested at a single token position per prompt; there is NO
time/token trajectory in the existing harvest. Applying SINDy-SAE to cogito
therefore requires a fresh MULTI-TOKEN harvest where activations across token
positions form a trajectory. This file ships the architecture; cogito
application is future work. See `scripts/sindy_smoke_cogito.py` for a
clearly-broken smoke test that exercises the pipeline only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Library φ(z)
# ---------------------------------------------------------------------------


_AVAILABLE_TERMS = {"constant", "identity", "square", "cube", "sin", "cos", "product"}


def library_size(state_dim: int, terms: Sequence[str]) -> int:
    """Number of library columns P given state_dim and term list."""
    P = 0
    for t in terms:
        if t == "constant":
            P += 1
        elif t in ("identity", "square", "cube", "sin", "cos"):
            P += state_dim
        elif t == "product":
            # unordered pairs i<j
            P += state_dim * (state_dim - 1) // 2
        else:
            raise ValueError(f"Unknown library term: {t}")
    return P


def library_term_names(state_dim: int, terms: Sequence[str]) -> list[str]:
    names: list[str] = []
    for t in terms:
        if t == "constant":
            names.append("1")
        elif t == "identity":
            names += [f"z{i}" for i in range(state_dim)]
        elif t == "square":
            names += [f"z{i}^2" for i in range(state_dim)]
        elif t == "cube":
            names += [f"z{i}^3" for i in range(state_dim)]
        elif t == "sin":
            names += [f"sin(z{i})" for i in range(state_dim)]
        elif t == "cos":
            names += [f"cos(z{i})" for i in range(state_dim)]
        elif t == "product":
            for i, j in combinations(range(state_dim), 2):
                names.append(f"z{i}*z{j}")
    return names


def build_library(z: torch.Tensor, terms: Sequence[str]) -> torch.Tensor:
    """Evaluate φ(z): (B, state_dim) -> (B, P)."""
    cols: list[torch.Tensor] = []
    B, D = z.shape
    for t in terms:
        if t == "constant":
            cols.append(torch.ones(B, 1, dtype=z.dtype, device=z.device))
        elif t == "identity":
            cols.append(z)
        elif t == "square":
            cols.append(z * z)
        elif t == "cube":
            cols.append(z * z * z)
        elif t == "sin":
            cols.append(torch.sin(z))
        elif t == "cos":
            cols.append(torch.cos(z))
        elif t == "product":
            pair_cols = [z[:, i : i + 1] * z[:, j : j + 1] for i, j in combinations(range(D), 2)]
            if pair_cols:
                cols.append(torch.cat(pair_cols, dim=1))
        else:
            raise ValueError(f"Unknown library term: {t}")
    return torch.cat(cols, dim=1)


# ---------------------------------------------------------------------------
# SINDy-SAE
# ---------------------------------------------------------------------------


@dataclass
class SINDyConfig:
    state_dim: int
    library_terms: tuple[str, ...] = ("identity", "square", "product")
    sparsity: float = 0.01  # L1 weight on Θ
    init_scale: float = 0.01


class SINDySAE(nn.Module):
    """SINDy-style SAE.

    Forward maps a state z to a predicted time derivative dz/dt via a sparse
    linear combination of a fixed nonlinear library. The sparse coefficient
    matrix Θ plays the role of "atoms": each row is a governing equation.

    Parameters
    ----------
    state_dim : int
        Dimension of the state z (e.g. number of latent coords or
        activation-stream dims).
    library_terms : sequence of str
        Subset of {"constant", "identity", "square", "cube", "sin", "cos",
        "product"}. "product" adds all unordered pairwise products z_i z_j.
    sparsity : float
        L1 penalty on Θ used by `loss`.
    """

    def __init__(
        self,
        state_dim: int,
        library_terms: Sequence[str] = ("identity", "square", "cube", "sin", "cos", "product"),
        sparsity: float = 0.01,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        bad = set(library_terms) - _AVAILABLE_TERMS
        if bad:
            raise ValueError(f"Unknown library terms: {bad}")
        self.state_dim = state_dim
        self.library_terms: tuple[str, ...] = tuple(library_terms)
        self.sparsity = float(sparsity)
        P = library_size(state_dim, self.library_terms)
        self.num_library_terms = P
        # Θ : (state_dim, P) — rows are governing-equation atoms.
        self.Theta = nn.Parameter(torch.randn(state_dim, P) * init_scale)
        # Hard-threshold mask used for STLSQ-style inference; learned externally.
        self.register_buffer("mask", torch.ones(state_dim, P))

    # ----- library --------------------------------------------------------
    def phi(self, z: torch.Tensor) -> torch.Tensor:
        return build_library(z, self.library_terms)

    def term_names(self) -> list[str]:
        return library_term_names(self.state_dim, self.library_terms)

    # ----- forward --------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict dz/dt at state z. z : (B, state_dim) -> (B, state_dim)."""
        phi_z = self.phi(z)  # (B, P)
        Theta_eff = self.Theta * self.mask
        return phi_z @ Theta_eff.T

    # ----- loss -----------------------------------------------------------
    def loss(
        self,
        z: torch.Tensor,
        dz_true: torch.Tensor,
        sparsity: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """SINDy regression loss + L1 on Θ.

        Returns dict with 'recon', 'l1', 'total'.
        """
        lam = self.sparsity if sparsity is None else float(sparsity)
        dz_pred = self.forward(z)
        recon = ((dz_pred - dz_true) ** 2).mean()
        l1 = (self.Theta * self.mask).abs().mean()
        total = recon + lam * l1
        return {"recon": recon, "l1": l1, "total": total}

    # ----- STLSQ-style hard thresholding ---------------------------------
    @torch.no_grad()
    def threshold(self, eps: float) -> int:
        """Zero-out (mask) coefficients with |Θ_ij| < eps (Brunton 2016).

        Returns number of currently-active terms.
        """
        keep = (self.Theta.abs() >= eps).float()
        self.mask.copy_(self.mask * keep)
        self.Theta.data.mul_(self.mask)
        return int(self.mask.sum().item())

    @torch.no_grad()
    def reset_mask(self) -> None:
        self.mask.fill_(1.0)

    @torch.no_grad()
    def effective_Theta(self) -> torch.Tensor:
        return self.Theta * self.mask
