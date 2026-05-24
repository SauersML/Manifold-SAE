"""Universal-SAE: one shared sparse dictionary across multiple LLMs.

Reference: Bricken et al., "Universal Sparse Autoencoders for Concept-Level
Alignment Across Models", arXiv:2502.03714 (2025).

Closely related to ``manifold_sae.crosscoder`` (cross-layer in the same
model), but here the "layers" are *different models with potentially
different residual widths*. The shared encoder learns a single sparse code
z ∈ R^F from a concatenated multi-model activation; per-model decoders
W_m ∈ R^{F × D_m} reconstruct each model independently.

Universality
------------
After training, atom k is "universal" if every model's decoder retains a
non-trivial column norm for atom k (i.e. atom k is actually used to
reconstruct every model, not just one). Falsifiable: a non-universal
feature manifests as a decoder whose mass concentrates in one model's
columns.

This is the cross-model analogue of the Anthropic crosscoder per-layer
mask in ``manifold_sae.crosscoder.cross_layer_atom_mask``.

Loss
----
    L = Σ_m  ||x_m - z @ W_m||²  +  λ_sparse · Σ_k |z_k| · Σ_m ||W_m[k]||_2

The decoder-norm-weighted L1 (per Anthropic 2024) prevents the optimizer
from pushing all weight into a single decoder to dodge the sparsity
penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


@dataclass
class UniversalSAEConfig:
    model_dims: dict[str, int]
    n_atoms: int = 512
    sparsity_weight: float = 1e-3
    activation: str = "relu"  # "relu" | "topk"
    top_k: int = 32
    encoder_hidden: int | None = None  # if int, 1-hidden-layer MLP encoder


class UniversalSAE(nn.Module):
    """Shared-encoder, per-model-decoder SAE for cross-LLM alignment.

    Parameters
    ----------
    F : int
        Number of shared latent atoms.
    model_dims : dict[str, int]
        Mapping ``{model_name: hidden_dim}``. Determines decoder shapes and
        the input dimension of the shared encoder (= sum of dims).
    """

    def __init__(
        self,
        F: int,
        model_dims: dict[str, int],
        *,
        sparsity_weight: float = 1e-3,
        activation: str = "relu",
        top_k: int = 32,
        encoder_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.config = UniversalSAEConfig(
            model_dims=dict(model_dims),
            n_atoms=int(F),
            sparsity_weight=float(sparsity_weight),
            activation=str(activation),
            top_k=int(top_k),
            encoder_hidden=encoder_hidden,
        )
        # Stable ordering of model names — used everywhere downstream.
        self.model_names: list[str] = list(model_dims.keys())
        self.dims: list[int] = [int(model_dims[m]) for m in self.model_names]
        self.D_total = int(sum(self.dims))
        self.F = int(F)

        # Per-model input centering (mean-shift) — important when concatenated
        # widths have very different scales. Stored as buffers; can be set via
        # `fit_centers`.
        for m, d in zip(self.model_names, self.dims):
            self.register_buffer(f"mu_{m}", torch.zeros(d))
            self.register_buffer(f"scale_{m}", torch.ones(1))

        # ----- Encoder -----
        if encoder_hidden is None:
            self.encoder = nn.Linear(self.D_total, self.F, bias=True)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.D_total, int(encoder_hidden), bias=True),
                nn.GELU(),
                nn.Linear(int(encoder_hidden), self.F, bias=True),
            )

        # ----- Per-model decoders -----
        self.decoders = nn.ParameterDict()
        self.dec_biases = nn.ParameterDict()
        for m, d in zip(self.model_names, self.dims):
            W = torch.randn(self.F, d) / max(d, 1) ** 0.5
            self.decoders[m] = nn.Parameter(W)
            self.dec_biases[m] = nn.Parameter(torch.zeros(d))

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_centers(self, X_by_model: dict[str, torch.Tensor]) -> None:
        """Set per-model mean and a single global RMS scale.

        Centers each model's data to zero mean (per-dim) and rescales the
        whole stack so the concatenated input has unit RMS — without this,
        a 7168-d model dominates the encoder vs a 896-d model.
        """
        total_sq = 0.0
        total_n = 0
        for m in self.model_names:
            x = X_by_model[m]
            mu = x.mean(dim=0)
            getattr(self, f"mu_{m}").copy_(mu)
            xc = x - mu
            total_sq += float((xc ** 2).sum().item())
            total_n += xc.numel()
        rms = (total_sq / max(total_n, 1)) ** 0.5
        for m in self.model_names:
            getattr(self, f"scale_{m}").fill_(1.0 / max(rms, 1e-8))

    def _normalize(self, X_by_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for m in self.model_names:
            mu = getattr(self, f"mu_{m}")
            s = getattr(self, f"scale_{m}")
            out[m] = (X_by_model[m] - mu) * s
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def encode(self, x_concat: torch.Tensor) -> torch.Tensor:
        z_pre = self.encoder(x_concat)
        if self.config.activation == "topk":
            return self._topk_relu(z_pre, k=self.config.top_k)
        return F.relu(z_pre)

    @staticmethod
    def _topk_relu(z_pre: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, k=k, dim=-1)
        mask = torch.zeros_like(z_pre)
        mask.scatter_(-1, idx, 1.0)
        return F.relu(z_pre) * mask

    def decode(self, z: torch.Tensor, model_name: str) -> torch.Tensor:
        return z @ self.decoders[model_name] + self.dec_biases[model_name]

    def forward(self, X_by_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        Xn = self._normalize(X_by_model)
        x_concat = torch.cat([Xn[m] for m in self.model_names], dim=-1)
        z = self.encode(x_concat)

        recons = {m: self.decode(z, m) for m in self.model_names}
        mse_per_model = {
            m: ((Xn[m] - recons[m]) ** 2).mean() for m in self.model_names
        }
        total_mse = sum(mse_per_model.values())

        # Decoder-norm-weighted L1.
        per_atom_dec_norm = sum(
            self.decoders[m].norm(dim=1) for m in self.model_names
        )  # (F,)
        l1 = (z.abs() * per_atom_dec_norm.unsqueeze(0)).sum(dim=-1).mean()

        loss = total_mse + self.config.sparsity_weight * l1
        return {
            "z": z,
            "recons": recons,
            "mse_per_model": mse_per_model,
            "l1": l1,
            "loss": loss,
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    @torch.no_grad()
    def per_model_r2(self, X_by_model: dict[str, torch.Tensor]) -> dict[str, float]:
        out = self.forward(X_by_model)
        Xn = self._normalize(X_by_model)
        r2 = {}
        for m in self.model_names:
            x = Xn[m]
            rec = out["recons"][m]
            ss_res = ((x - rec) ** 2).sum().item()
            ss_tot = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum().item()
            r2[m] = 1.0 - ss_res / max(ss_tot, 1e-12)
        return r2

    @torch.no_grad()
    def atom_model_affinity(self) -> torch.Tensor:
        """Per-atom per-model normalized decoder norm.

        Returns (F, M) where row k sums to 1. Atoms concentrated in one
        column are model-specific; near-uniform rows are universal.
        """
        norms = torch.stack(
            [self.decoders[m].norm(dim=1) for m in self.model_names], dim=-1
        )  # (F, M)
        s = norms.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return norms / s

    @torch.no_grad()
    def universal_atom_mask(self, threshold: float = 0.15) -> torch.Tensor:
        """Boolean (F,) — True for atoms whose minimum per-model share ≥ threshold.

        With M=2 models, a perfectly-shared atom has share 0.5 / model;
        0.15 catches "present in all models, possibly unbalanced". For M=3,
        uniform share is 0.33 — keep threshold below that.
        """
        affinity = self.atom_model_affinity()
        return affinity.min(dim=-1).values >= threshold

    @torch.no_grad()
    def universality_score(self, threshold: float = 1e-3) -> torch.Tensor:
        """Per-atom universality = fraction of models with non-trivial mass.

        Normalizes decoder column norms by max-per-atom, then counts the
        fraction of models above ``threshold``. Returns shape (F,) in [0, 1].
        """
        norms = torch.stack(
            [self.decoders[m].norm(dim=1) for m in self.model_names], dim=-1
        )  # (F, M)
        peak = norms.max(dim=-1, keepdim=True).values.clamp(min=1e-12)
        rel = norms / peak
        return (rel > threshold).float().mean(dim=-1)

    @torch.no_grad()
    def alive_mask(self, X_by_model: dict[str, torch.Tensor]) -> torch.Tensor:
        """Atoms that fire on at least one sample."""
        out = self.forward(X_by_model)
        return (out["z"].abs().max(dim=0).values > 0)
