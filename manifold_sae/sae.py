"""Manifold-SAE: SAE with curve-valued atoms, REML training, lock-and-cache deploy.

Architecture (Path B with lock-and-cache):

  Training forward:
    encoder(x)                       → (positions, amplitudes)        [Adam-owned weights]
    y_proj = x @ W                   → (B, F, R)                      [Adam-owned W per feature]
    fit = gamfit.gaussian_reml_fit_positions_batched(
            positions, y_proj, ..., by=amplitudes)
                                     → (B, λ, fitted, reml_score)     [REML each batch, autograd]
    recon = Σ_k amp_k · fit.fitted_k @ W_k^T

  Loss:
    MSE(recon, x)                    — data fit through encoder + W
    − reml_score.sum()              — REML log-likelihood (smoothness selected by gamfit)
    + sparsity, identification priors

  Adam optimizes:
    encoder weights, W, b_dec

  gamfit owns each batch:
    coefficients B_k, smoothing λ_k

  Lock-and-cache at end of training:
    one big REML fit on a held-out reference batch → frozen B, λ, basis_state
    as nn.Module buffers (not parameters).

  Inference forward:
    encoder(x)                       → (positions, amplitudes)
    φ = duchon_basis_1d(positions)   → (B, F, K)
    g = φ @ B_locked                 → (B, F, R)
    recon = Σ_k amp_k · g_k @ W_k^T

    Single-token feedforward. No gamfit call at inference.

Methodological claim
--------------------
Each feature is a smooth 1D curve in residual stream parameterized as the
penalized maximum-likelihood estimate of a Gaussian GAM given the encoder's
positions. Smoothness λ_k is selected automatically by REML (gamfit owns
the math). At inference the curve coefficients are cached, giving a
feedforward decoder identical in shape to a standard SAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import gamfit.torch as gt
import numpy as np
import torch
from torch import nn

from .encoder import ManifoldEncoder
from .encoder_linear import ManifoldEncoderLinear


@dataclass
class ManifoldSAEConfig:
    input_dim: int
    n_features: int
    n_basis: int
    top_k: int
    intrinsic_rank: int = 2
    sparsity_weight: float = 1e-3
    ortho_weight: float = 1e-2
    reml_weight: float = 1.0           # weight on −REML log-likelihood term
    encoder_type: str = "mlp"          # "mlp" | "linear"
    continuous_amp: bool = False
    periodic: bool = False             # use periodic Duchon basis (cyclic features)
    init_lambda: float | None = None   # init for gamfit's REML λ-optimization;
                                       # higher = biased toward smoother fits.
                                       # None lets gamfit pick (default ~1e-4).


@dataclass
class ManifoldSAEOutput:
    reconstruction: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor
    mask_soft: torch.Tensor
    coefficients: torch.Tensor         # B_k from gamfit (per batch during training,
                                       # locked snapshot at inference)
    lam: torch.Tensor                  # λ_k from gamfit (per batch during training,
                                       # locked snapshot at inference)
    reml_score: torch.Tensor           # per-feature REML log-likelihood; 0 at inference
    fitted: torch.Tensor               # per-feature subspace prediction at this batch's positions
    directions: torch.Tensor
    ortho_loss: torch.Tensor
    monotonicity_loss: torch.Tensor


def _soft_rescale_positions(z_raw: torch.Tensor, beta: float = 10.0, eps: float = 1e-4) -> torch.Tensor:
    """Smooth per-batch min-max normalization of unbounded scalar logits to [0, 1].

    Gauge-fixes the position so atoms see positions spanning the basis domain —
    no init-clustering pathology. λ_k and B_k are reparam-invariant under
    monotone rescaling of t, so this is lossless. O(B·F) work, no parameters.
    """
    soft_max = (1.0 / beta) * torch.logsumexp(beta * z_raw, dim=0)
    soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z_raw, dim=0)
    span = (soft_max - soft_min).clamp(min=1e-6)
    t = (z_raw - soft_min.unsqueeze(0)) / span.unsqueeze(0)
    return t.clamp(eps, 1.0 - eps)


class ManifoldSAE(nn.Module):
    def __init__(self, config: ManifoldSAEConfig) -> None:
        super().__init__()
        self.config = config
        EncoderCls = ManifoldEncoderLinear if getattr(config, "encoder_type", "mlp") == "linear" else ManifoldEncoder
        self.encoder = EncoderCls(
            intrinsic_rank=config.intrinsic_rank,
            n_features=config.n_features,
            input_dim=config.input_dim,
            top_k=config.top_k,
        )
        self.encoder.continuous_amp = bool(getattr(config, "continuous_amp", False))

        K = int(config.n_basis)
        D = int(config.input_dim)
        R = int(config.intrinsic_rank)
        F = int(config.n_features)

        # Centers in [0, 1] — float64 because gamfit's REML requires it.
        # We share centers across features (standard GAM setup).
        centers = torch.linspace(0.0, 1.0, K, dtype=torch.float64)
        self.register_buffer("centers", centers)

        # Persistent per-feature W_k in R^(D, R). Adam-owned.
        if D >= F * R:
            Q, _ = torch.linalg.qr(torch.randn(D, F * R))
            directions = Q.reshape(D, F, R).permute(1, 0, 2).contiguous()
        else:
            directions = torch.empty(F, D, R)
            for k in range(F):
                q, _ = torch.linalg.qr(torch.randn(D, R))
                directions[k] = q
        self.directions = nn.Parameter(directions.to(torch.float32))

        # Decoder pre-bias.
        self.b_dec = nn.Parameter(torch.zeros(D, dtype=torch.float32))

        # Locked snapshot buffers. Filled by `update_snapshot`; consulted in
        # eval-mode if `inference_mode=True`.
        self.register_buffer("B_locked", torch.zeros(F, K, R, dtype=torch.float64))
        self.register_buffer("lam_locked", torch.ones(F, dtype=torch.float64))
        self.register_buffer("has_snapshot", torch.tensor(False))
        self.inference_mode = False  # set True after lock_and_cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        x_dtype = x.dtype
        dirs = self.directions.to(x_dtype)
        b_dec = self.b_dec.to(x_dtype)
        x_centered = x - b_dec
        y_proj = torch.einsum("bd,fdr->bfr", x_centered, dirs)  # (B, F, R)
        z_raw, mask_soft, mask_binary = self.encoder(x_centered, y_proj)
        positions = _soft_rescale_positions(z_raw)
        B, F = positions.shape

        if self.inference_mode and bool(self.has_snapshot):
            return self._forward_inference(x_dtype, dirs, b_dec, y_proj, positions, mask_soft, mask_binary)
        return self._forward_training(x_dtype, dirs, b_dec, y_proj, positions, mask_soft, mask_binary)

    def _forward_training(
        self,
        x_dtype: torch.dtype,
        dirs: torch.Tensor,
        b_dec: torch.Tensor,
        y_proj: torch.Tensor,
        positions: torch.Tensor,
        mask_soft: torch.Tensor,
        mask_binary: torch.Tensor,
    ) -> ManifoldSAEOutput:
        """gamfit REML fit per batch; B and λ flow from gamfit, not from parameters."""
        B, F = positions.shape
        R = dirs.shape[-1]

        # Pack for gamfit: (F*B,) positions, (F*B, R) targets, (F*B,) amplitude weights.
        # gamfit expects float64 throughout.
        t_packed = positions.t().contiguous().view(-1).to(torch.float64)
        y_packed = y_proj.permute(1, 0, 2).contiguous().view(F * B, R).to(torch.float64)
        by_packed = mask_binary.t().contiguous().view(-1).to(torch.float64)
        offsets = (torch.arange(F + 1, device=positions.device) * B).to(torch.uint64)

        fit = gt.gaussian_reml_fit_positions_batched(
            t_packed, y_packed, offsets,
            "duchon",
            self.centers,        # explicit centers (shared across features)
            None,                # penalty: gamfit auto-derives the Duchon m=2 penalty matrix
            basis_order=2,
            periodic=self.config.periodic,
            period=1.0 if self.config.periodic else None,
            by=by_packed,
            init_lambda=self.config.init_lambda,
        )

        # gamfit.fitted: (F*B, R) — per-feature subspace prediction at this batch's positions.
        fitted = fit.fitted.view(F, B, R).to(x_dtype)
        # SAE reconstruction: lift each feature's fit to ambient, gated by amplitude.
        contribution = torch.einsum("fbr,fdr->bfd", fitted * mask_binary.t().unsqueeze(-1), dirs)
        recon = contribution.sum(dim=1) + b_dec.unsqueeze(0)

        # Identification: per-feature column ortho + cross-feature off-block ortho.
        ortho_loss = self._ortho_loss(dirs)
        monotonicity_loss = self._monotonicity_loss(positions, y_proj, mask_binary)

        return ManifoldSAEOutput(
            reconstruction=recon,
            positions=positions,
            amplitudes=mask_binary,
            mask_soft=mask_soft,
            coefficients=fit.coefficients.to(x_dtype),  # (F, K, R) — autograd-aware
            lam=fit.lam.to(x_dtype),
            reml_score=fit.reml_score.to(x_dtype),
            fitted=fitted,
            directions=self.directions,
            ortho_loss=ortho_loss,
            monotonicity_loss=monotonicity_loss,
        )

    def _forward_inference(
        self,
        x_dtype: torch.dtype,
        dirs: torch.Tensor,
        b_dec: torch.Tensor,
        y_proj: torch.Tensor,
        positions: torch.Tensor,
        mask_soft: torch.Tensor,
        mask_binary: torch.Tensor,
    ) -> ManifoldSAEOutput:
        """Use locked snapshot — no gamfit call. Single-token-evaluable."""
        B, F = positions.shape
        R = dirs.shape[-1]

        # Evaluate the basis at this batch's positions (gamfit's basis evaluator).
        t_flat = positions.t().contiguous().view(-1).to(torch.float64)
        phi = gt.duchon_basis_1d(t_flat, self.centers, m=2, periodic=self.config.periodic)
        K = phi.shape[-1]
        phi = phi.view(F, B, K)
        # g_k(t_b) = φ_b @ B_locked_k.
        g = torch.einsum("fbk,fkr->fbr", phi, self.B_locked).to(x_dtype)
        contribution = torch.einsum("fbr,fdr->bfd", g * mask_binary.t().unsqueeze(-1), dirs)
        recon = contribution.sum(dim=1) + b_dec.unsqueeze(0)

        ortho_loss = self._ortho_loss(dirs)
        monotonicity_loss = self._monotonicity_loss(positions, y_proj, mask_binary)

        return ManifoldSAEOutput(
            reconstruction=recon,
            positions=positions,
            amplitudes=mask_binary,
            mask_soft=mask_soft,
            coefficients=self.B_locked.to(x_dtype),
            lam=self.lam_locked.to(x_dtype),
            reml_score=torch.zeros(F, dtype=x_dtype, device=positions.device),
            fitted=g,
            directions=self.directions,
            ortho_loss=ortho_loss,
            monotonicity_loss=monotonicity_loss,
        )

    # ------------------------------------------------------------------
    # Identification (gauge / parameterization tiebreakers — these stay
    # because REML doesn't speak to them)
    # ------------------------------------------------------------------

    def _ortho_loss(self, dirs: torch.Tensor) -> torch.Tensor:
        """Per-feature column ortho + cross-feature off-block diversity."""
        F = dirs.shape[0]
        R = dirs.shape[-1]
        I_R = torch.eye(R, dtype=dirs.dtype, device=dirs.device).unsqueeze(0)
        WtW = torch.einsum("fdr,fds->frs", dirs, dirs)
        per_feature_ortho = ((WtW - I_R) ** 2).mean()
        M = dirs.permute(0, 2, 1).reshape(F * R, dirs.shape[1])
        gram = M @ M.t()
        block_eye = torch.kron(
            torch.eye(F, dtype=dirs.dtype, device=dirs.device),
            torch.ones(R, R, dtype=dirs.dtype, device=dirs.device),
        )
        off_block = gram * (1.0 - block_eye)
        cross_ortho = (off_block ** 2).mean()
        return per_feature_ortho + 0.1 * cross_ortho

    def _monotonicity_loss(
        self,
        positions: torch.Tensor,
        y_proj: torch.Tensor,
        mask_binary: torch.Tensor,
    ) -> torch.Tensor:
        """Position should track the principal-axis projection (loose prior).

        Identification: when multiple parameterizations explain the data
        equally well (monotone vs U-shape), prefer monotone. Doesn't bind
        when data demands non-monotone (e.g. parabola).
        """
        principal = y_proj[..., 0]                                 # (B, F)
        mask_f = (mask_binary.detach() > 0.5).to(positions.dtype)
        mass = mask_f.sum(dim=0).clamp(min=1.0)
        p_mean = (positions * mask_f).sum(dim=0) / mass
        q_mean = (principal * mask_f).sum(dim=0) / mass
        p_c = (positions - p_mean.unsqueeze(0)) * mask_f
        q_c = (principal - q_mean.unsqueeze(0)) * mask_f
        num = (p_c * q_c).sum(dim=0).abs()
        den = (p_c.pow(2).sum(dim=0) * q_c.pow(2).sum(dim=0)).clamp(min=1e-12).sqrt()
        per_feat = 1.0 - num / den
        active = (mass >= 5.0).to(positions.dtype)
        return (per_feat * active).sum() / active.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # Lock-and-cache: snapshot B and λ for feedforward inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_snapshot(self, reference_batch: torch.Tensor) -> None:
        """Run one REML fit on a (large) representative batch; freeze (B, λ).

        Call at end of training. After this, ``self.inference_mode = True``
        switches the forward path to use the cached snapshot — feedforward,
        single-token-evaluable, no gamfit call.
        """
        was_training = self.training
        was_inference_mode = self.inference_mode
        self.eval()
        self.inference_mode = False
        try:
            out = self(reference_batch)
            self.B_locked.copy_(out.coefficients.detach().to(torch.float64))
            self.lam_locked.copy_(out.lam.detach().to(torch.float64))
            self.has_snapshot.fill_(True)
        finally:
            self.train(was_training)
            self.inference_mode = was_inference_mode


@torch.no_grad()
def extract_feature_curves(
    sae: ManifoldSAE,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Per-feature learned curves on ``t_grid`` in ambient space.

    Uses the locked snapshot (preferred) or computes a fresh REML fit on
    ``activations`` if no snapshot exists. Returns (F, T, D).
    """
    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid_f64 = t_grid.to(device=device, dtype=torch.float64)

    sae.eval()
    if not bool(sae.has_snapshot):
        sae.update_snapshot(activations)

    out = sae(activations)
    pos = out.positions
    amp = out.amplitudes
    firing = amp > 1e-3
    F = sae.B_locked.shape[0]
    T = t_grid_f64.shape[0]
    D = sae.directions.shape[1]
    dirs = sae.directions.to(torch.float64)
    curves = torch.zeros(F, T, D, dtype=torch.float64, device=device)
    for k in range(F):
        m = firing[:, k]
        if m.sum() < 2:
            t_lo, t_hi = 0.0, 1.0
        else:
            pos_k = pos[m, k].to(torch.float64)
            t_lo = float(pos_k.quantile(0.02).item())
            t_hi = float(pos_k.quantile(0.98).item())
            if t_hi - t_lo < 1e-3:
                t_lo, t_hi = 0.0, 1.0
        t_k = t_lo + (t_hi - t_lo) * t_grid_f64
        phi_k = gt.duchon_basis_1d(t_k, sae.centers, m=2, periodic=sae.config.periodic)
        intrinsic_curve = phi_k @ sae.B_locked[k]
        ambient_curve = intrinsic_curve @ dirs[k].T
        curves[k] = ambient_curve
    return curves.to(activations.dtype)
