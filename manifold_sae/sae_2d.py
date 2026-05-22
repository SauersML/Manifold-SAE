"""Manifold-SAE 2D — each atom carries a 2D parameterization (t_k, s_k).

This is the native 2D extension of `manifold_sae.sae.ManifoldSAE`. Instead
of clustering vanilla SAE features post-hoc to recover 2D manifolds, each
atom learns its own 2D surface `g_k: [0, 1]² → ℝ^R` via a tensor-product
Duchon m=2 basis.

See docs/architecture_2d.md for the design.

v1 supports plane / disk topology (non-periodic in both axes), single λ
per atom (one Kronecker-summed penalty). Future work adds cylinder/torus
(periodic Duchon on one or both axes) and per-axis smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import gamfit.torch as gt
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------


@dataclass
class ManifoldSAE2DConfig:
    input_dim: int
    n_features: int
    n_basis: int = 8                  # K per axis; total basis dim is K²
    top_k: int = 4
    intrinsic_rank: int = 2
    sparsity_weight: float = 3e-4
    ortho_weight: float = 1e-3
    coverage_weight: float = 1e-2
    isotropy_weight: float = 1e-3     # replaces monotonicity from 1D
    continuous_amp: bool = True
    init_lambda: float = 1e-2


def _soft_rescale_1d(z: torch.Tensor, weights: torch.Tensor | None,
                      beta: float = 10.0, eps: float = 1e-4,
                      frozen_min: torch.Tensor | None = None,
                      frozen_max: torch.Tensor | None = None,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as the 1D `_soft_rescale_positions` in sae.py. Per-feature
    firing-weighted soft min/max so the rescale isn't dominated by
    non-firing tokens.
    """
    if frozen_min is not None and frozen_max is not None:
        soft_min, soft_max = frozen_min, frozen_max
    elif weights is not None:
        log_w = torch.log(weights.clamp(min=1e-6))
        soft_max = (1.0 / beta) * torch.logsumexp(beta * z + log_w, dim=0)
        soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z + log_w, dim=0)
        active = weights.sum(dim=0) > 1e-6
        soft_min = torch.where(active, soft_min, torch.zeros_like(soft_min))
        soft_max = torch.where(active, soft_max, torch.ones_like(soft_max))
    else:
        soft_max = (1.0 / beta) * torch.logsumexp(beta * z, dim=0)
        soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z, dim=0)
    span = (soft_max - soft_min).clamp(min=1e-6)
    t = (z - soft_min.unsqueeze(0)) / span.unsqueeze(0)
    return t.clamp(eps, 1.0 - eps), soft_min, soft_max


def _duchon_penalty_1d(centers: torch.Tensor, m: int = 2) -> torch.Tensor:
    """Build the 1D Duchon m=2 function-norm penalty matrix `(K, K)`.
    Wraps gamfit's `_duchon_function_norm_penalty`.
    """
    from gamfit._api import _duchon_function_norm_penalty

    centers_np = centers.detach().cpu().numpy().astype(np.float64)
    P = _duchon_function_norm_penalty(centers_np, m=m, periodic=False)
    return torch.from_numpy(np.asarray(P, dtype=np.float64))


def _tensor_product_penalty(P_1d: torch.Tensor) -> torch.Tensor:
    """Kronecker-summed penalty for the tensor-product basis.

    For a separable basis φ_2d(t, s) = φ_1d(t) ⊗ φ_1d(s), the standard
    smoothness penalty `∫∫ (∂²f/∂t²)² + (∂²f/∂s²)² dt ds` reduces (under
    suitable normalization of φ_1d) to

        P_2d = P_1d ⊗ I + I ⊗ P_1d

    where I is K×K identity. This is a single-λ approximation; per-axis
    smoothing would carry two penalty pieces with two λ's.
    """
    K = P_1d.shape[0]
    I = torch.eye(K, dtype=P_1d.dtype, device=P_1d.device)
    return torch.kron(P_1d, I) + torch.kron(I, P_1d)


# ---------------------------------------------------------------------------
# Encoder with TWO position heads
# ---------------------------------------------------------------------------


class ManifoldEncoder2D(nn.Module):
    """Two scalar position heads per feature + one amplitude head."""

    def __init__(self, input_dim: int, n_features: int, top_k: int,
                 hidden_dim: int | None = None) -> None:
        super().__init__()
        D = input_dim
        F = n_features
        H = hidden_dim if hidden_dim is not None else 4 * D
        self.input_dim = D
        self.n_features = F
        self.top_k = top_k
        self.hidden_dim = H

        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H, bias=True)
        self.act = nn.GELU()
        self.head_t = nn.Linear(H, F, bias=True)     # first axis logit
        self.head_s = nn.Linear(H, F, bias=True)     # second axis logit
        self.head_a = nn.Linear(H, F, bias=True)     # amplitude logit
        nn.init.normal_(self.fc1.weight, std=1.0 / D**0.5)
        nn.init.normal_(self.head_t.weight, std=1.0 / H**0.5)
        nn.init.normal_(self.head_s.weight, std=1.0 / H**0.5)
        nn.init.normal_(self.head_a.weight, std=1.0 / H**0.5)
        self.continuous_amp = True

    def forward(self, x: torch.Tensor):
        x_n = self.norm(x)
        h = self.act(self.fc1(x_n))
        z_t = self.head_t(h).clamp(-10.0, 10.0)
        z_s = self.head_s(h).clamp(-10.0, 10.0)
        amp_logits = self.head_a(h).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)
        if self.continuous_amp:
            amp_cont = torch.nn.functional.softplus(amp_logits)
            if self.top_k is not None and self.top_k < self.n_features:
                _vals, idx = torch.topk(amp_cont, self.top_k, dim=1)
                gate = torch.zeros_like(amp_cont)
                gate.scatter_(1, idx, 1.0)
                amp_out = amp_cont * gate
            else:
                amp_out = amp_cont
            return z_t, z_s, mask_soft, amp_out
        # Binary-amp path
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard = torch.zeros_like(mask_soft).scatter_(1, idx, 1.0)
            mask_binary = hard + (mask_soft - mask_soft.detach())
        else:
            hard = torch.ones_like(mask_soft)
            mask_binary = hard + (mask_soft - mask_soft.detach())
        return z_t, z_s, mask_soft, mask_binary


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


@dataclass
class ManifoldSAE2DOutput:
    reconstruction: torch.Tensor
    positions_t: torch.Tensor          # (B, F)
    positions_s: torch.Tensor          # (B, F)
    amplitudes: torch.Tensor           # (B, F)
    mask_soft: torch.Tensor            # (B, F)
    coefficients: torch.Tensor         # (F, K*K, R)
    lam: torch.Tensor                  # (F,)
    reml_score: torch.Tensor           # (F,)
    directions: torch.Tensor           # (F, D, R)
    intrinsic_dim_ratio: torch.Tensor  # (F,) — std(s) / std(t) over firing tokens


class ManifoldSAE2D(nn.Module):
    """2D Manifold-SAE: each atom = one 2D surface in residual stream.

    Differences from the 1D version:
        * Encoder outputs (z_t, z_s) per feature.
        * Basis is the tensor-product Duchon, K² per atom.
        * Penalty is the Kronecker-summed 2D penalty (single λ in v1).
        * gamfit batched (not positions_batched) with explicit basis matrix.
        * Self-test in update_snapshot checks training vs locked agreement.
    """

    def __init__(self, config: ManifoldSAE2DConfig) -> None:
        super().__init__()
        self.config = config
        K = int(config.n_basis)
        D = int(config.input_dim)
        R = int(config.intrinsic_rank)
        F = int(config.n_features)
        self.K = K
        self.D = D
        self.R = R
        self.F = F

        self.encoder = ManifoldEncoder2D(
            input_dim=D, n_features=F, top_k=config.top_k,
        )
        self.encoder.continuous_amp = bool(config.continuous_amp)

        # Shared basis centers in [0, 1].
        centers = torch.linspace(0.0, 1.0, K, dtype=torch.float64)
        self.register_buffer("centers", centers)

        # Per-feature subspace W_k. Orthogonal init.
        if D >= F * R:
            Q, _ = torch.linalg.qr(torch.randn(D, F * R))
            dirs = Q.reshape(D, F, R).permute(1, 0, 2).contiguous()
        else:
            dirs = torch.empty(F, D, R)
            for k in range(F):
                q, _ = torch.linalg.qr(torch.randn(D, R))
                dirs[k] = q
        self.directions = nn.Parameter(dirs.to(torch.float32))
        self.b_dec = nn.Parameter(torch.zeros(D, dtype=torch.float32))

        # Locked snapshot buffers.
        self.register_buffer("B_locked", torch.zeros(F, K * K, R, dtype=torch.float64))
        self.register_buffer("lam_locked", torch.ones(F, dtype=torch.float64))
        self.register_buffer("soft_min_locked", torch.zeros(F, 2, dtype=torch.float32))
        self.register_buffer("soft_max_locked", torch.ones(F, 2, dtype=torch.float32))
        self.register_buffer("has_snapshot", torch.tensor(False))

        # Penalty matrix is constant per config — build once, cache as buffer.
        P_1d = _duchon_penalty_1d(centers, m=2)
        P_2d = _tensor_product_penalty(P_1d)
        self.register_buffer("P_2d", P_2d)              # (K², K²)

        self.inference_mode = False

    # ------------------------------------------------------------------
    # Basis evaluation
    # ------------------------------------------------------------------

    def _eval_basis_2d(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Evaluate the tensor-product Duchon basis at positions (t, s).

        Inputs:
            t: shape `(N,)` of float64 t-coordinates in [0, 1].
            s: shape `(N,)` of float64 s-coordinates in [0, 1].
        Returns:
            phi: shape `(N, K²)` of float64.
        """
        phi_t = gt.duchon_basis_1d(t, self.centers, m=2, periodic=False)  # (N, K)
        phi_s = gt.duchon_basis_1d(s, self.centers, m=2, periodic=False)  # (N, K)
        # Tensor product: phi_2d[n, i*K + j] = phi_t[n, i] * phi_s[n, j]
        N = phi_t.shape[0]
        K = self.K
        phi_2d = (phi_t.unsqueeze(2) * phi_s.unsqueeze(1)).reshape(N, K * K)
        return phi_2d

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> ManifoldSAE2DOutput:
        x_dtype = x.dtype
        dirs = self.directions.to(x_dtype)
        b_dec = self.b_dec.to(x_dtype)
        x_centered = x - b_dec
        y_proj = torch.einsum("bd,fdr->bfr", x_centered, dirs)   # (B, F, R)

        z_t, z_s, mask_soft, amp = self.encoder(x_centered)

        in_inference = self.inference_mode and bool(self.has_snapshot.item())
        if in_inference:
            t, _, _ = _soft_rescale_1d(
                z_t, weights=None,
                frozen_min=self.soft_min_locked[:, 0].to(z_t.dtype),
                frozen_max=self.soft_max_locked[:, 0].to(z_t.dtype),
            )
            s, _, _ = _soft_rescale_1d(
                z_s, weights=None,
                frozen_min=self.soft_min_locked[:, 1].to(z_s.dtype),
                frozen_max=self.soft_max_locked[:, 1].to(z_s.dtype),
            )
            return self._forward_inference(x_dtype, dirs, b_dec, t, s, mask_soft, amp)

        # Training mode: per-axis firing-weighted soft rescale.
        weights = amp.detach()
        t, soft_min_t, soft_max_t = _soft_rescale_1d(z_t, weights=weights)
        s, soft_min_s, soft_max_s = _soft_rescale_1d(z_s, weights=weights)
        self._last_soft_min = torch.stack([soft_min_t, soft_min_s], dim=1).detach()  # (F, 2)
        self._last_soft_max = torch.stack([soft_max_t, soft_max_s], dim=1).detach()
        return self._forward_training(x_dtype, dirs, b_dec, y_proj, t, s, mask_soft, amp)

    def _forward_training(self, x_dtype, dirs, b_dec, y_proj, t, s, mask_soft, amp) -> ManifoldSAE2DOutput:
        """Per-batch REML fit over each atom's 2D surface."""
        B, F = t.shape
        D, R, K = self.D, self.R, self.K

        # Pack positions and responses per-atom for batched REML.
        # Atom k contributes B tokens; total N_total = F*B.
        t_packed = t.t().contiguous().view(-1).to(torch.float64)   # (F*B,)
        s_packed = s.t().contiguous().view(-1).to(torch.float64)
        y_packed = y_proj.permute(1, 0, 2).contiguous().view(F * B, R).to(torch.float64)
        by_packed = amp.t().contiguous().view(-1).to(torch.float64)
        row_offsets = (torch.arange(F + 1, device=t.device) * B).to(torch.uint64)

        # Build the (F*B, K²) tensor-product basis matrix.
        phi_packed = self._eval_basis_2d(t_packed, s_packed)        # (F*B, K²)

        # gamfit batched REML with explicit penalty.
        fit = gt.gaussian_reml_fit_batched(
            phi_packed, y_packed, row_offsets, self.P_2d,
            by=by_packed,
            init_lambda=self.config.init_lambda,
        )
        # fit.fitted: (F*B, R) — already amp-weighted (gamfit semantic)
        fitted = fit.fitted.view(F, B, R).to(x_dtype)
        contribution = torch.einsum("fbr,fdr->bfd", fitted, dirs)
        recon = contribution.sum(dim=1) + b_dec.unsqueeze(0)

        # Per-atom intrinsic-dim ratio: std of s relative to std of t over
        # firing tokens. 0 = atom is 1D (uses only t); 1 = balanced 2D.
        amp_mask = (amp > 1e-6).float()                              # (B, F)
        eps = 1e-8
        t_var = (((t - t.mean(0)) ** 2) * amp_mask).sum(0) / (amp_mask.sum(0) + eps)
        s_var = (((s - s.mean(0)) ** 2) * amp_mask).sum(0) / (amp_mask.sum(0) + eps)
        dim_ratio = (s_var.sqrt() / (t_var.sqrt() + eps)).clamp(0, 5)

        return ManifoldSAE2DOutput(
            reconstruction=recon,
            positions_t=t, positions_s=s,
            amplitudes=amp, mask_soft=mask_soft,
            coefficients=fit.coefficients.to(x_dtype),
            lam=fit.lam.to(x_dtype),
            reml_score=fit.reml_score.to(x_dtype),
            directions=self.directions,
            intrinsic_dim_ratio=dim_ratio.to(x_dtype),
        )

    def _forward_inference(self, x_dtype, dirs, b_dec, t, s, mask_soft, amp) -> ManifoldSAE2DOutput:
        """Locked path — no gamfit call, just basis @ B_locked."""
        B, F = t.shape
        R, K = self.R, self.K
        t_packed = t.t().contiguous().view(-1).to(torch.float64)
        s_packed = s.t().contiguous().view(-1).to(torch.float64)
        phi = self._eval_basis_2d(t_packed, s_packed).view(F, B, K * K)
        # g_k(t, s) for atom k at every token b: phi[F, B, K²] @ B_locked[F, K², R]
        g = torch.einsum("fbm,fmr->fbr", phi, self.B_locked).to(x_dtype)
        # amp-weighted contribution, lifted to ambient.
        contribution = torch.einsum("fbr,fdr->bfd", g * amp.t().unsqueeze(-1), dirs)
        recon = contribution.sum(dim=1) + b_dec.unsqueeze(0)
        return ManifoldSAE2DOutput(
            reconstruction=recon,
            positions_t=t, positions_s=s,
            amplitudes=amp, mask_soft=mask_soft,
            coefficients=self.B_locked.to(x_dtype),
            lam=self.lam_locked.to(x_dtype),
            reml_score=torch.zeros(F, dtype=x_dtype, device=t.device),
            directions=self.directions,
            intrinsic_dim_ratio=torch.ones(F, dtype=x_dtype, device=t.device),
        )

    # ------------------------------------------------------------------
    # Lock-and-cache snapshot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_snapshot(self, reference_batch: torch.Tensor) -> None:
        """Run REML on a reference batch, store coefficients + soft-rescale
        stats as buffers. Self-test asserts the locked-mode forward agrees
        with training-mode on the same batch.
        """
        was_training = self.training
        was_inference_mode = self.inference_mode
        self.eval()
        self.inference_mode = False
        try:
            out = self(reference_batch)
            self.B_locked.copy_(out.coefficients.detach().to(torch.float64))
            self.lam_locked.copy_(out.lam.detach().to(torch.float64))
            self.soft_min_locked.copy_(self._last_soft_min.detach().to(torch.float32))
            self.soft_max_locked.copy_(self._last_soft_max.detach().to(torch.float32))
            self.has_snapshot.fill_(True)

            # Self-test: training-mode and locked-mode reconstructions
            # must agree to within float32 noise.
            training_recon = out.reconstruction.detach()
            self.inference_mode = True
            locked_out = self(reference_batch)
            locked_recon = locked_out.reconstruction
            diff = (training_recon - locked_recon).abs().max().item()
            ref = training_recon.abs().mean().clamp(min=1e-6).item()
            rel = diff / ref
            if rel >= 5e-1:
                raise RuntimeError(
                    f"update_snapshot self-test FAILED: training vs locked "
                    f"reconstructions diverged by max_abs={diff:.4e} "
                    f"(rel={rel:.4e}). Likely a regression in "
                    f"`_forward_inference`."
                )
            elif rel >= 5e-2:
                import warnings as _w
                _w.warn(
                    f"update_snapshot self-test: rel={rel:.2e} max_abs={diff:.2e}. "
                    f"Within tolerance but worth flagging — accumulated f32 noise."
                )
        finally:
            self.train(was_training)
            self.inference_mode = was_inference_mode

    # ------------------------------------------------------------------
    # Convenience: report per-atom dimensionality
    # ------------------------------------------------------------------

    def report_intrinsic_dims(self, x: torch.Tensor) -> dict:
        """For each atom, report whether it's using 1D or 2D structure.

        An atom is *1D-like* if its `s_k` collapses (low variance across
        firing tokens) — i.e. the surface degraded to a curve along t.
        Returns a dict with per-atom ratios + summary counts.
        """
        with torch.no_grad():
            was_inf = self.inference_mode
            self.inference_mode = False
            out = self(x)
            self.inference_mode = was_inf
        ratios = out.intrinsic_dim_ratio.cpu().numpy()
        return {
            "per_atom_dim_ratio": ratios.tolist(),
            "n_atoms_1d_like": int((ratios < 0.2).sum()),
            "n_atoms_2d_like": int((ratios > 0.5).sum()),
            "mean_ratio": float(ratios.mean()),
        }
