"""Shadow cone — presence/amplitude decoupling for Block-Sparse Featurizers.

Goodfire's BSF "shadow" result shows a block's norm ‖z_g‖ tracks *luminance*: it
is an INTENSITY coordinate that conflates "present but weak" with "absent". This
module tests that head-to-head on the paper's own ground and asks whether a
GATED variant — presence carried by a *separate binary gate*, amplitude+identity
by the signed in-block code — recovers presence where the raw block norm cannot
(mirroring the gate/code split BT1 uses in gam: ``a_k = σ((ℓ_k − θ_k)/τ)``).

Three readings of a block, kept explicitly separate here:
  * **amplitude / intensity** = ‖z_g‖               (the block norm; "luminance")
  * **identity**             = z_g / ‖z_g‖          (in-block direction; "hue")
  * **presence**             = a gate a_g on a learned logit ℓ_g = w_gᵀx − θ_g

The claim under test: block-norm presence-detection AUC collapses in the weak
regime (weak-present vs absent overlap in amplitude, especially under distractor
leakage), while a presence gate separates them; and the identity axis (in-block
direction) is orthogonal to intensity (norm) — steerable independently.

Companion driver: ``run.py`` (synthetic presence ROC, real weekday/month
intensity-vs-identity, steering figure). Grassmannian blocks are reused from
``experiments/bsf_baseline/bsf.py``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bsf_baseline"))
from bsf import BSF, BSFConfig, TrainConfig, train_bsf  # noqa: E402


# ==========================================================================
# metrics
# ==========================================================================
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the Mann–Whitney U statistic (rank-based; ties handled)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ==========================================================================
# Gated BSF: presence gate (separate) + signed in-block code (amplitude/identity)
# ==========================================================================
@dataclass
class GatedConfig:
    d_model: int
    n_blocks: int
    block_size: int
    l0_coef: float = 1.0e-2       # sparsity weight on the presence gates
    gate_tau: float = 0.2          # JumpReLU temperature (softness of the gate)
    presence_supervision: float = 0.0  # optional aux BCE weight toward known presence
    seed: int = 0


class GatedBSF(nn.Module):
    """Grassmannian blocks with a decoupled presence gate per block.

    ``z_g = γ·x D_gᵀ`` is the signed in-block code (identity = direction,
    amplitude = ‖z_g‖). A SEPARATE presence pathway produces a logit
    ``ℓ_g = w_gᵀ(x − b_dec) − θ_g`` and a JumpReLU gate ``a_g = σ(ℓ_g/τ)·1[ℓ_g>0]``
    (the gam ``jumprelu_row`` gate). Reconstruction is ``Σ_g a_g · z_g D_g``, so
    the gate — not the code norm — decides which blocks contribute. The gate can
    thus fire for a weak-but-real feature and stay dark for large-norm distractor
    leakage, decoupling presence from intensity.
    """

    def __init__(self, cfg: GatedConfig):
        super().__init__()
        self.cfg = cfg
        g, b, d = cfg.n_blocks, cfg.block_size, cfg.d_model
        gen = torch.Generator().manual_seed(cfg.seed)
        dec = torch.empty(g, b, d, dtype=torch.float64)
        for gi in range(g):
            q, _ = torch.linalg.qr(torch.randn(d, b, generator=gen, dtype=torch.float64))
            dec[gi] = q.T
        self.decoder = nn.Parameter(dec)
        self.b_dec = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        self.log_gamma = nn.Parameter(torch.zeros((), dtype=torch.float64))
        # separate presence pathway (a learned readout direction + threshold / block)
        self.w_pres = nn.Parameter(0.1 * torch.randn(g, d, generator=gen, dtype=torch.float64))
        self.theta = nn.Parameter(torch.zeros(g, dtype=torch.float64))

    def code(self, x: torch.Tensor) -> torch.Tensor:
        g, b = self.cfg.n_blocks, self.cfg.block_size
        dflat = self.decoder.reshape(g * b, self.cfg.d_model)
        z = torch.exp(self.log_gamma) * ((x - self.b_dec) @ dflat.T)
        return z.reshape(-1, g, b)

    def presence_logit(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.w_pres.T - self.theta  # (N, G)

    def gate(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        ell = self.presence_logit(x)
        if hard:
            return (ell > 0).double()
        return torch.sigmoid(ell / self.cfg.gate_tau) * (ell > 0).double()

    def forward(self, x: torch.Tensor):
        z = self.code(x)                       # (N, G, b)
        a = self.gate(x)                       # (N, G)
        g, b, d = self.cfg.n_blocks, self.cfg.block_size, self.cfg.d_model
        contrib = (a.unsqueeze(-1) * z).reshape(-1, g * b) @ self.decoder.reshape(g * b, d)
        x_hat = contrib + self.b_dec
        return x_hat, z, a

    @torch.no_grad()
    def reproject_stiefel(self):
        d = self.decoder.data
        for gi in range(self.cfg.n_blocks):
            q, r = torch.linalg.qr(d[gi].T)
            sign = torch.sign(torch.diagonal(r))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            d[gi] = (q * sign).T


def train_gated(model: GatedBSF, X: torch.Tensor, steps: int = 3000, lr: float = 3e-3,
                reproj_every: int = 20, presence_labels: torch.Tensor | None = None,
                seed: int = 0) -> None:
    """Reconstruction + L0(gate) [+ optional presence BCE]. Blocks retracted to
    the Stiefel manifold every ``reproj_every`` steps."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    n = X.shape[0]
    gen = torch.Generator().manual_seed(seed)
    bs = min(512, n)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=gen)
        xb = X[idx]
        x_hat, _, a = model(xb)
        recon = ((x_hat - xb) ** 2).mean()
        l0 = a.abs().mean()
        loss = recon + model.cfg.l0_coef * l0
        if presence_labels is not None and model.cfg.presence_supervision > 0:
            loss = loss + model.cfg.presence_supervision * bce(
                model.presence_logit(xb), presence_labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % reproj_every == 0:
            model.reproject_stiefel()
    model.eval()


# ==========================================================================
# block matching (which model block is the planted target feature?)
# ==========================================================================
def orthonormal_basis(dec_g: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(dec_g.T)
    return q


def subspace_r2(a: np.ndarray, b: np.ndarray) -> float:
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return float((np.clip(s, 0, 1) ** 2).mean())


def match_target_block(decoder: np.ndarray, target_basis: np.ndarray) -> int:
    """Index of the model block whose subspace best aligns with the planted target."""
    rec = [orthonormal_basis(decoder[g]) for g in range(decoder.shape[0])]
    return int(np.argmax([subspace_r2(target_basis, q) for q in rec]))
