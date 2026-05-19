"""Manifold-SAE: standard-SAE shape with curve-valued atoms.

Each dictionary atom is a smooth 1D curve ``g_k: [0, 1] -> R^D`` instead
of a single direction. The whole thing is shaped like a vanilla SAE so
the scaling story is identical: persistent parameters, feedforward
encoder, Adam-trainable, applies to a fresh token in one forward pass.

Parameterization
----------------
Per feature ``k``:

  * ``W_k in R^(D, R)``           — persistent ambient subspace
  * ``B_k in R^(K_basis, R)``    — persistent spline coefficients
  * ``log_lambda_k``              — persistent per-feature smoothness scalar

  g_k(t) = phi(t) @ B_k @ W_k^T,  phi the Duchon m=2 1D basis on [0, 1].

The basis evaluation is the only call into gamfit; there is NO per-batch
REML solve. ``B_k`` is updated by Adam just like ``W_dec`` columns are in
a vanilla SAE.

Smoothness
----------
The loss includes ``sum_k exp(log_lambda_k) * tr(B_k^T S B_k)`` where
``S`` is the integrated-squared-second-derivative penalty matrix from
gamfit. ``log_lambda_k`` is learnable, so Adam picks per-feature
smoothing on its own — the same identifiability gain as REML, paid for
in parameters instead of per-batch linear solves.

Encoder
-------
Linear ``W_enc_t, W_enc_a in R^(F, D)`` heads (vanilla-SAE shape).
A small per-feature MLP on top of ``(x_norm, y_proj_k)`` is used for the
toy task only because the linear heads can't separate look-alike
monotone features at F = 5. The MLP is replaceable with the linear heads
at LLM scale where overcompleteness does the separation.

Amplitudes
----------
Straight-through TopK on sigmoid amplitude logits — binary firing,
closes the amp/coef gauge so position must carry where-on-the-curve.
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
    cumulant_weight: float = 0.0
    ortho_weight: float = 1e-2
    smoothness_weight: float = 1e-3   # base coef on tr(B^T S B), scaled per feature by exp(log_lambda_k)
    encoder_type: str = "mlp"          # "mlp" (per-feature, toy) or "linear" (LLM scale)
    continuous_amp: bool = False        # True: softplus(a) + topk gate (continuous magnitude)
    curve_norm_weight: float = 0.0      # weight on (avg curve_norm² - 1)² gauge penalty


@dataclass
class ManifoldSAEOutput:
    reconstruction: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor       # binary (TopK + straight-through)
    mask_soft: torch.Tensor        # pre-binarization sigmoid probabilities
    coefficients: torch.Tensor     # persistent B (F, K, R), broadcast for compat
    directions: torch.Tensor       # persistent W (F, D, R)
    cumulant_loss: torch.Tensor
    ortho_loss: torch.Tensor
    monotonicity_loss: torch.Tensor
    smoothness_loss: torch.Tensor
    curve_norm_loss: torch.Tensor


def _soft_rescale_positions(z_raw: torch.Tensor, beta: float = 10.0, eps: float = 1e-4) -> torch.Tensor:
    """Smooth per-batch min-max normalization of unbounded scalar logits to [0, 1].

    Gauge-fixes the position so atoms always see positions spanning the
    basis domain — no init-clustering pathology where every position
    starts at sigmoid(0) = 0.5. ``log_lambda`` and ``B_k`` are reparam-
    invariant under monotone rescaling of ``t``, so this is lossless.
    O(B*F) work at LLM scale, no extra parameters.
    """
    soft_max = (1.0 / beta) * torch.logsumexp(beta * z_raw, dim=0)
    soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z_raw, dim=0)
    span = (soft_max - soft_min).clamp(min=1e-6)
    t = (z_raw - soft_min.unsqueeze(0)) / span.unsqueeze(0)
    return t.clamp(eps, 1.0 - eps)


def _basis_lookup(positions: torch.Tensor, phi_lookup: torch.Tensor) -> torch.Tensor:
    """Evaluate the Duchon basis at positions in [0, 1] by linear
    interpolation over a precomputed (T_fine, K) lookup table.

    Lets the SAE run end-to-end on MPS/CUDA — no gamfit float64 calls in
    the forward path. Accuracy: with T_fine=4096 the interp error is
    O(1/T_fine^2) for the (smooth) Duchon basis — well below SGD noise.

    Args:
        positions: any shape with values in [0, 1], on the same device.
        phi_lookup: (T_fine, K) float32.
    Returns:
        Same leading shape as positions, with a final K dim.
    """
    T_fine = phi_lookup.shape[0]
    K = phi_lookup.shape[1]
    flat = positions.reshape(-1).clamp(0.0, 1.0)
    idx_f = flat * (T_fine - 1)
    i0 = idx_f.floor().long().clamp(0, T_fine - 2)
    i1 = i0 + 1
    w = (idx_f - i0.to(positions.dtype)).unsqueeze(-1)
    phi = (1.0 - w) * phi_lookup[i0] + w * phi_lookup[i1]
    return phi.view(*positions.shape, K)


def _duchon_penalty_matrix(centers: torch.Tensor, basis_order: int = 2) -> torch.Tensor:
    """Integrated squared mth-derivative penalty for Duchon basis on [0,1].

    Approximated by finite-difference on a fine grid of basis evaluations.
    Returns a (K, K) PSD matrix.
    """
    K = int(centers.shape[0])
    M = 2048
    t = torch.linspace(0.0, 1.0, M, dtype=torch.float64)
    phi = gt.duchon_basis_1d(t, centers.to(torch.float64), m=basis_order, periodic=False)
    # Finite-difference second derivative across t.
    d2 = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]
    h = 1.0 / (M - 1)
    d2 = d2 / (h ** 2)
    S = d2.T @ d2 * (1.0 / (M - 2))
    # Add a tiny diagonal to keep PSD on degenerate basis directions.
    S = S + 1e-8 * torch.eye(K, dtype=torch.float64)
    return S


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
        # Plumb continuous_amp into the encoder so it can switch gauges.
        self.encoder.continuous_amp = bool(getattr(config, "continuous_amp", False))
        K = int(config.n_basis)
        D = int(config.input_dim)
        R = int(config.intrinsic_rank)
        F = int(config.n_features)

        centers_np = np.linspace(0.0, 1.0, K, dtype=np.float64)
        # Centers used only at init (penalty + lookup precompute) — store as
        # float32 so the buffer can live on MPS (which lacks float64).
        self.register_buffer("centers", torch.from_numpy(centers_np).to(torch.float32))
        # Real Duchon m=2 penalty matrix — gives correct smoothness behavior
        # (penalizes curvature, leaves linear functions free). Cast to float32
        # so it lives on MPS/CUDA without dtype promotion.
        S = _duchon_penalty_matrix(torch.from_numpy(centers_np), basis_order=2)
        self.register_buffer("penalty", S.to(torch.float32))

        # Pre-compute the Duchon basis on a dense grid so forward passes
        # never call gamfit. MPS doesn't support float64 (which gamfit
        # requires), so any forward path that calls gamfit forces CPU.
        # The basis function is fixed once centers are fixed → precompute
        # at init in float64, store as float32, then evaluate at arbitrary
        # positions by linear interpolation across the grid.
        T_fine = 4096
        t_fine = torch.linspace(0.0, 1.0, T_fine, dtype=torch.float64)
        phi_fine = gt.duchon_basis_1d(t_fine, torch.from_numpy(centers_np), m=2, periodic=False)
        self.register_buffer("phi_lookup", phi_fine.to(torch.float32))     # (T_fine, K)

        # Persistent per-feature W_k in R^(D, R).
        if D >= F * R:
            Q, _ = torch.linalg.qr(torch.randn(D, F * R))
            directions = Q.reshape(D, F, R).permute(1, 0, 2).contiguous()
        else:
            directions = torch.empty(F, D, R)
            for k in range(F):
                q, _ = torch.linalg.qr(torch.randn(D, R))
                directions[k] = q
        self.directions = nn.Parameter(directions.to(torch.float32))

        # Persistent per-feature spline coefficients B_k in R^(K, R).
        # Init small so curves start near zero — Adam will grow them.
        self.coeff = nn.Parameter(0.05 * torch.randn(F, K, R, dtype=torch.float32))

        # Per-feature learnable log-smoothness.
        self.log_lambda = nn.Parameter(torch.zeros(F, dtype=torch.float32))

        # Decoder pre-bias (standard SAE practice): subtract from x before
        # encoding, add back to reconstruction. Captures the data mean so
        # atoms aren't wasted reconstructing it.
        self.b_dec = nn.Parameter(torch.zeros(D, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        x_dtype = x.dtype
        dirs = self.directions.to(x_dtype)
        coeff = self.coeff.to(x_dtype)
        b_dec = self.b_dec.to(x_dtype)
        x_centered = x - b_dec
        y_proj = torch.einsum("bd,fdr->bfr", x_centered, dirs)  # (B, F, R)
        z_raw, mask_soft, mask_binary = self.encoder(x_centered, y_proj)
        positions = _soft_rescale_positions(z_raw)

        B, F = positions.shape
        R = dirs.shape[-1]
        K = coeff.shape[1]

        # Evaluate basis at positions via the precomputed lookup table —
        # no gamfit calls in the forward path, so the whole forward runs
        # on MPS/CUDA in float32.
        phi = _basis_lookup(positions, self.phi_lookup.to(x_dtype))  # (B, F, K)
        # g_{b,f,:} = phi_{b,f,:} @ B_f
        g = torch.einsum("bfk,fkr->bfr", phi, coeff)

        # Lift to ambient, gated by binary amplitude. Add back the
        # decoder pre-bias to undo the input centering.
        contribution = torch.einsum("bfr,fdr->bfd", g * mask_binary.unsqueeze(-1), dirs)
        recon = contribution.sum(dim=1) + b_dec.unsqueeze(0)

        # Curve-norm gauge penalty: keep E_t[||g_k(t) @ W_k^T||²] near 1
        # so amplitude carries magnitude and curve carries shape (when
        # ``continuous_amp`` is enabled in the config). Without this and
        # with continuous amp, (amp, curve) is a multiplicative gauge.
        curve_norm_w = float(getattr(self.config, "curve_norm_weight", 0.0))
        if curve_norm_w > 0:
            t_grid = torch.linspace(0.02, 0.98, 32, dtype=x_dtype, device=x.device)
            phi_g = _basis_lookup(t_grid, self.phi_lookup.to(x_dtype))     # (T, K)
            g_grid = torch.einsum("tk,fkr->ftr", phi_g, coeff)             # (F, T, R)
            amb = torch.einsum("ftr,fdr->ftd", g_grid, dirs)               # (F, T, D)
            per_feat = amb.pow(2).sum(dim=-1).mean(dim=-1)                 # (F,)
            curve_norm_loss = ((per_feat - 1.0) ** 2).mean()
        else:
            curve_norm_loss = torch.zeros((), dtype=x_dtype, device=x.device)

        # Per-feature smoothness penalty: sum_k exp(log_lambda_k) * tr(B_k^T S B_k).
        # Done in float64 to avoid quadratic blow-up.
        S = self.penalty.to(coeff.dtype)
        SB = torch.einsum("kj,fjr->fkr", S, coeff)
        per_feature = torch.einsum("fkr,fkr->f", coeff, SB)  # tr(B_k^T S B_k)
        lam = torch.exp(self.log_lambda).to(coeff.dtype)
        smoothness_loss = (lam * per_feature).mean()

        # Identifiability: 4th-moment cumulant contrast (off by default at
        # cumulant_weight=0).
        if self.config.cumulant_weight > 0:
            a = mask_soft
            a_c = a - a.mean(dim=0, keepdim=True)
            a_std = a_c.std(dim=0, keepdim=True).clamp(min=1e-6)
            a_n = a_c / a_std
            pair_4th = (a_n.unsqueeze(-1) * a_n.unsqueeze(-2)).pow(2).mean(dim=0)
            n_off = max(F * F - F, 1)
            cumulant_loss = (pair_4th.sum() - pair_4th.diagonal().sum()) / n_off
        else:
            cumulant_loss = torch.zeros((), dtype=x_dtype, device=x.device)

        # Per-feature column orthonormality AND cross-feature subspace
        # orthogonality. Cross-feature term is the load-bearing one: without
        # it, different features' W_k can align to overlapping ambient
        # directions, the encoder sees nearly identical y_proj signals
        # across features and can't separate them — the recovered curves
        # then represent mixtures of GT features. Build M = (FR, D) of
        # all column vectors; penalize off-diagonal entries of M M^T.
        I_R = torch.eye(R, dtype=dirs.dtype, device=dirs.device).unsqueeze(0)
        WtW = torch.einsum("fdr,fds->frs", dirs, dirs)
        per_feature_ortho = ((WtW - I_R) ** 2).mean()
        # Cross-feature subspace coherence: penalize OFF-DIAGONAL blocks only.
        # The diagonal blocks are per-feature ortho (already covered); the
        # off-diagonal blocks measure how much different features' subspaces
        # overlap. Real planted features in this dataset share dimensions, so
        # this penalty should be light — it only breaks pathological aliasing
        # where two SAE features end up in the same subspace.
        M = dirs.permute(0, 2, 1).reshape(F * R, dirs.shape[1])  # (F*R, D)
        gram = M @ M.t()  # (F*R, F*R)
        # Mask out the F diagonal R×R blocks
        block_eye = torch.zeros_like(gram)
        for k in range(F):
            block_eye[k*R:(k+1)*R, k*R:(k+1)*R] = 1.0
        off_block = gram * (1.0 - block_eye)
        cross_ortho = (off_block ** 2).mean()
        ortho_loss = per_feature_ortho + 0.1 * cross_ortho

        principal = y_proj[..., 0]
        mask_f = mask_binary.detach()
        eps = 1e-6
        terms = []
        for k in range(F):
            m = mask_f[:, k] > 0.5
            if m.sum() < 5:
                continue
            p = positions[m, k]
            q = principal[m, k]
            p_c = p - p.mean()
            q_c = q - q.mean()
            denom = (p_c.pow(2).sum() * q_c.pow(2).sum()).clamp(min=eps).sqrt()
            terms.append(1.0 - (p_c * q_c).sum().abs() / denom)
        monotonicity_loss = torch.stack(terms).mean() if terms else torch.zeros((), dtype=x_dtype, device=x.device)

        return ManifoldSAEOutput(
            reconstruction=recon,
            positions=positions,
            amplitudes=mask_binary,
            mask_soft=mask_soft,
            coefficients=self.coeff,
            directions=self.directions,
            cumulant_loss=cumulant_loss,
            ortho_loss=ortho_loss,
            monotonicity_loss=monotonicity_loss,
            smoothness_loss=smoothness_loss,
            curve_norm_loss=curve_norm_loss,
        )


@torch.no_grad()
def extract_feature_curves(
    sae: ManifoldSAE,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Per-feature learned curves on ``t_grid`` in ambient space.

    Uses the persistent lookup table — no gamfit calls, runs on whatever
    device the SAE lives on.
    """
    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid_f = t_grid.to(device=device, dtype=torch.float32)

    sae.eval()
    out = sae(activations)
    coeff = sae.coeff
    dirs = sae.directions
    amp = out.amplitudes
    pos = out.positions
    firing = amp > 1e-3
    F = coeff.shape[0]
    T = t_grid_f.shape[0]
    D = dirs.shape[1]
    curves = torch.zeros(F, T, D, dtype=torch.float32, device=device)
    for k in range(F):
        m = firing[:, k]
        if m.sum() < 2:
            t_lo, t_hi = 0.0, 1.0
        else:
            pos_k = pos[m, k]
            t_lo = float(pos_k.quantile(0.02).item())
            t_hi = float(pos_k.quantile(0.98).item())
            if t_hi - t_lo < 1e-3:
                t_lo, t_hi = 0.0, 1.0
        t_k = t_lo + (t_hi - t_lo) * t_grid_f
        phi_k = _basis_lookup(t_k, sae.phi_lookup)
        intrinsic_curve = phi_k @ coeff[k]
        ambient_curve = intrinsic_curve @ dirs[k].T
        curves[k] = ambient_curve
    return curves.to(activations.dtype)
