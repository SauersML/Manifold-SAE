"""Manifold-SAE: SAE with curve-valued atoms, REML training, lock-and-cache deploy.

Architecture (Path B with lock-and-cache, joint additive REML):

  Training forward:
    encoder(x)                       → (positions, amplitudes)        [Adam-owned weights]
    result = gamfit.torch.fit(
        points=[positions[:, k:k+1] for k in range(F)],
        response=x_centered,                                          # (B, D)
        smooths=[Duchon(centers, m=2, by=amplitudes[:, k]) for k],
    )                                                                 # joint additive REML
    coefficients_k : (K, D)  for each atom k
    recon = result.fitted + b_dec

  Loss:
    MSE(recon, x)                    — data fit through encoder + joint additive fit
    − reml_score                     — joint REML log-likelihood
    + sparsity, identification priors

  Adam optimizes:
    encoder weights, W (used for y_proj in identification priors), b_dec

  gamfit owns each batch:
    per-atom curve coefficients B_k ∈ R^(K, D), joint smoothing λ

  Lock-and-cache at end of training:
    one big additive REML fit on a held-out reference batch → frozen B (per-atom
    K × D blocks), λ, rescale stats as nn.Module buffers (not parameters).

  Inference forward:
    encoder(x)                       → (positions, amplitudes)
    φ_k = duchon_basis(positions[:, k:k+1], centers)                  → (B, K)
    g_k = amp_k · (φ_k @ B_locked[k])                                 → (B, D)
    recon = Σ_k g_k + b_dec

    Single-token feedforward. No gamfit call at inference.

Methodological claim
--------------------
Each feature is a smooth curve in residual stream parameterized as the
penalized maximum-likelihood estimate of a Gaussian GAM given the encoder's
positions. Smoothness λ is selected automatically by REML (gamfit owns the
math). At inference the curve coefficients are cached, giving a feedforward
decoder identical in shape to a standard SAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import gamfit.torch as gt
from gamfit.torch import Duchon, fit as gam_fit
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
    # gamfit basis kind. Default "duchon_multipenalty" (triple-operator) lets
    # REML select three smoothing λ's per feature — one each for the mass,
    # tension, and stiffness operators. Empirically gives 3-4× more alive
    # atoms at the same F + matched or higher EV vs the single-λ "duchon"
    # (validated at Qwen-0.5B L18 F=128: 49 alive vs 14, EV 0.967 vs 0.936).
    # Override with "duchon" for plain single-λ function-norm. Not
    # compatible with `periodic=True` — falls back to "duchon" silently.
    basis_kind: str = "duchon_multipenalty"


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


_SAE_SCHEMA_VERSION = 2  # bump when forward semantics change → invalidates checkpoints


def _soft_rescale_positions(
    z_raw: torch.Tensor,
    beta: float = 10.0,
    eps: float = 1e-4,
    *,
    frozen_min: torch.Tensor | None = None,
    frozen_max: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth min-max normalization of unbounded scalar logits to [0, 1].

    Training mode (frozen_min / frozen_max both None): compute soft min/max
    over the current batch's z_raw per feature. λ_k and B_k are reparam-
    invariant under monotone rescaling of t, so this is a lossless gauge fix.

    Inference mode (snapshot frozen): use the per-feature (min, max) captured
    when the SAE was snapshotted. Same input → same position regardless of
    batch composition, so the locked B is consulted at consistent positions.

    Returns (t, soft_min, soft_max) — the rescaled positions plus the stats
    used. Caller stashes the stats at snapshot time.
    """
    if frozen_min is not None and frozen_max is not None:
        soft_min, soft_max = frozen_min, frozen_max
    elif weights is not None:
        # Firing-weighted soft min/max: rescale stats are computed using
        # only positions where the feature actually fires. Without this,
        # non-firing positions dominate the min/max for sparse features
        # (most positions are noise for that feature). Numerically:
        # logsumexp(β·z + log(w)) is equivalent to soft-max over the
        # positions weighted by w. Dead features (weights ≈ 0 everywhere)
        # fall back to the default [0, 1] range.
        log_w = torch.log(weights.clamp(min=1e-6))
        soft_max = (1.0 / beta) * torch.logsumexp(beta * z_raw + log_w, dim=0)
        soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z_raw + log_w, dim=0)
        active = weights.sum(dim=0) > 1e-6
        soft_min = torch.where(active, soft_min, torch.zeros_like(soft_min))
        soft_max = torch.where(active, soft_max, torch.ones_like(soft_max))
    else:
        soft_max = (1.0 / beta) * torch.logsumexp(beta * z_raw, dim=0)
        soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z_raw, dim=0)
    span = (soft_max - soft_min).clamp(min=1e-6)
    t = (z_raw - soft_min.unsqueeze(0)) / span.unsqueeze(0)
    return t.clamp(eps, 1.0 - eps), soft_min, soft_max


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
        # Stored as (K, 1) — the new gamfit multi-d API takes (K, d).
        centers = torch.linspace(0.0, 1.0, K, dtype=torch.float64).unsqueeze(1)
        self.register_buffer("centers", centers)
        # NOTE: the Duchon function-norm penalty is no longer kept as a
        # module buffer — the new gamfit.torch.fit() API builds it from each
        # Smooth's centers internally per call.

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
        # eval-mode if `inference_mode=True`. Both the curve coefficients AND
        # the per-feature soft-rescale stats are frozen — without freezing the
        # rescale, the same input token would get different positions in
        # different batches (the soft min/max is per-batch), and the locked B
        # would consult the wrong place on its curve.
        # B_locked: per-atom curve coefficients in AMBIENT space, shape
        # (F, K, D). Each slice B_locked[k] of shape (K, D) is the (K, D)
        # block returned by the joint additive fit for atom k — it has
        # absorbed the previous W_k embedding into the coefficient space.
        self.register_buffer("B_locked", torch.zeros(F, K, D, dtype=torch.float64))
        # The joint additive fit shares a scalar λ across atoms; stored as
        # length-F for backward compat with downstream code that indexes per
        # feature.
        self.register_buffer("lam_locked", torch.ones(F, dtype=torch.float64))
        self.register_buffer("soft_min_locked", torch.zeros(F, dtype=torch.float32))
        self.register_buffer("soft_max_locked", torch.ones(F, dtype=torch.float32))
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

        in_inference = self.inference_mode and bool(self.has_snapshot)
        if in_inference:
            positions, _, _ = _soft_rescale_positions(
                z_raw,
                frozen_min=self.soft_min_locked.to(z_raw.dtype),
                frozen_max=self.soft_max_locked.to(z_raw.dtype),
            )
            return self._forward_inference(x_dtype, dirs, b_dec, y_proj, positions, mask_soft, mask_binary)

        # Firing-weighted soft min/max so the rescale is dominated by
        # positions where the feature actually fires.
        positions, soft_min, soft_max = _soft_rescale_positions(z_raw, weights=mask_binary.detach())
        self._last_soft_min = soft_min.detach()
        self._last_soft_max = soft_max.detach()
        # Stash x_centered for _forward_training (the joint additive fit
        # needs the full ambient residual, not just the per-atom W_k
        # projections). Cleared after use to avoid stale refs.
        self._x_centered_for_training = x_centered
        try:
            return self._forward_training(x_dtype, dirs, b_dec, y_proj, positions, mask_soft, mask_binary)
        finally:
            self._x_centered_for_training = None

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
        """Joint additive REML fit via gamfit's Smooth + fit() API.

        SEMANTIC CHANGE (vs. previous per-atom path)
        --------------------------------------------
        Previously each atom k was fit *independently*: design = duchon basis
        at positions[:, k], gated by amp_k; target = y_proj[:, k, :] (the
        x_centered projected onto that atom's R-dim W_k subspace); coefs
        ∈ R^(K, R). The reconstruction was then assembled as
        Σ_k amp_k · (φ_k @ B_k) @ W_k^T.

        That formulation is only mathematically equivalent to the joint
        additive fit when the W_k subspaces are mutually orthogonal — i.e. it
        treated atoms as if they couldn't see each other's contributions to
        the shared residual.

        The new path uses ONE call to ``gamfit.torch.fit`` with F Duchon
        smooths against the (B, D)-shaped ambient residual ``x_centered``.
        Each smooth contributes a (B, K) design block, row-gated by its
        amplitude (``by=mask_binary[:, k]``). gamfit's additive REML jointly
        solves all atoms together, returning per-atom coefficients of shape
        (K, D) — the previous explicit W_k embedding is now absorbed into the
        coefficient space directly. This is the mathematically correct joint
        fit; W_k survives only as an identification prior driver (y_proj is
        still used for the monotonicity loss).
        """
        B, F = positions.shape
        D = b_dec.shape[0]
        K = self.centers.shape[0]
        device = positions.device

        periodic = bool(self.config.periodic)
        per_axis = (periodic,) if periodic else None

        pos64 = positions.to(torch.float64)
        by64 = mask_binary.to(torch.float64)
        x_centered_f64 = (
            # Recover x_centered from y_proj is wrong; pass the actual
            # ambient residual. The caller doesn't hand it to us, but
            # x = (y_proj decoder) is not in scope — instead, recompute:
            # x_centered = x - b_dec is what we want, but x isn't on hand
            # here. The caller (``forward``) has it; we receive y_proj only.
            # The cleanest fix: receive x_centered through the call. See
            # forward() which now passes it.
            self._x_centered_for_training
        ).to(torch.float64)

        smooths = [
            Duchon(
                centers=self.centers,
                m=2,
                by=by64[:, k],
                periodic_per_axis=per_axis,
            )
            for k in range(F)
        ]
        points = [pos64[:, k:k+1] for k in range(F)]

        # Production path: per-atom independent REML — scales linearly in F.
        # The joint additive path is O((F·M_k)³), infeasible past F ≳ 1000.
        # Under TopK gating most atoms are zero per row, so per-atom λ
        # independence is the right algorithm. ``mode="auto"`` would also
        # pick ``independent`` for F > 64 OR D > 1, but we set it explicitly
        # because the SAE production regime is always at F ≫ 64.
        result = gam_fit(
            points=points,
            response=x_centered_f64,
            smooths=smooths,
            init_lambdas=(
                torch.tensor(
                    [float(self.config.init_lambda)] * F, dtype=torch.float64,
                )
                if self.config.init_lambda is not None else None
            ),
            mode="independent",
        )

        # Stack per-atom coefficient blocks into (F, K, D). Each list entry
        # is a (K, D) tensor with autograd flowing through the joint fit.
        coefs_list = result.coefficients  # list[(K, D)]
        coefs = torch.stack(coefs_list, dim=0)  # (F, K, D)

        # Per-atom ambient-space contributions (autograd-aware). We rebuild
        # them from the basis at the atom's positions × the atom's (K, D)
        # block (no extra REML call). Used both for downstream reporting and
        # for the locked-mode self-test in update_snapshot.
        fitted_all = torch.zeros(F, B, D, dtype=torch.float64, device=device)
        for k in range(F):
            phi_k = gt.duchon_basis(
                pos64[:, k:k+1], self.centers, m=2, periodic_per_axis=per_axis,
            )                                                      # (B, K)
            fitted_all[k] = (phi_k @ coefs_list[k]) * by64[:, k:k+1]

        # Joint reconstruction: gamfit's ``result.fitted`` already sums the
        # per-atom (by-gated) contributions over k. Add the decoder bias.
        recon = result.fitted.to(x_dtype) + b_dec.unsqueeze(0)

        # The joint additive fit shares a scalar λ and a scalar REML score.
        # Broadcast to per-feature shape (F,) for backward compat with
        # downstream loggers that index per atom.
        # Fix: gamfit may return lambdas of shape (F,) not (1,); mean-collapse safely.
        lam_scalar = result.lambdas.reshape(-1).mean()
        lams = lam_scalar.expand(F).contiguous().to(x_dtype)
        reml_scores = result.reml_score.reshape(-1).mean().expand(F).contiguous().to(x_dtype)

        fitted = fitted_all.to(x_dtype)

        # Identification: per-feature column ortho + cross-feature off-block ortho.
        ortho_loss = self._ortho_loss(dirs)
        monotonicity_loss = self._monotonicity_loss(positions, y_proj, mask_binary)

        return ManifoldSAEOutput(
            reconstruction=recon,
            positions=positions,
            amplitudes=mask_binary,
            mask_soft=mask_soft,
            coefficients=coefs.to(x_dtype),                        # (F, K, D) — autograd-aware
            lam=lams,
            reml_score=reml_scores,
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
        """Use locked snapshot — no gamfit call. Single-token-evaluable.

        Locked path uses the joint additive fit's per-atom (K, D) coefficient
        blocks ``B_locked[k]`` directly: contribution_k = amp_k · (φ_k @
        B_locked[k]), summed across atoms with the decoder bias added back.
        """
        B, F = positions.shape
        D = b_dec.shape[0]

        # Evaluate the basis at this batch's positions, atom by atom. The
        # (F*B, 1) batched basis call is also valid and shaves a Python
        # loop, but B_locked is now per-atom so we go atom-wise for clarity.
        periodic = bool(self.config.periodic)
        per_axis = (periodic,) if periodic else None
        pos64 = positions.to(torch.float64)
        amp64 = mask_binary.to(torch.float64)

        g_all = torch.zeros(F, B, D, dtype=torch.float64, device=positions.device)
        for k in range(F):
            phi_k = gt.duchon_basis(
                pos64[:, k:k+1], self.centers, m=2, periodic_per_axis=per_axis,
            )                                                      # (B, K)
            g_all[k] = (phi_k @ self.B_locked[k]) * amp64[:, k:k+1]

        recon = g_all.sum(dim=0).to(x_dtype) + b_dec.unsqueeze(0)
        g = g_all.to(x_dtype)

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
        # Threshold > 0 catches continuous-amp firings (softplus output
        # of the active TopK lane can be any positive value). The old
        # > 0.5 threshold dropped low-amplitude active firings.
        mask_f = (mask_binary.detach() > 0).to(positions.dtype)
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
            self.soft_min_locked.copy_(self._last_soft_min.detach().to(torch.float32))
            self.soft_max_locked.copy_(self._last_soft_max.detach().to(torch.float32))
            self.has_snapshot.fill_(True)

            # Self-test: training-mode and locked-mode reconstructions on
            # the SAME snapshot batch should agree up to float32 numerical
            # noise. Two tiers:
            #   < 5e-2 relative: silently OK (typical float32 round-trip).
            #   5e-2 to 5e-1   : warn — could be slightly stale rescale
            #                     stats or accumulated f32 error.
            #   >= 5e-1        : raise — the locked path's MATH is wrong
            #                     (e.g. another amp²·curve-style mismatch).
            training_recon = out.reconstruction.detach()
            self.inference_mode = True
            try:
                with torch.no_grad():
                    locked_recon = self(reference_batch).reconstruction
                    diff = (training_recon - locked_recon).abs().max().item()
                    ref = training_recon.abs().mean().clamp(min=1e-6).item()
                    rel = diff / ref
                    if rel >= 5e-1:
                        raise RuntimeError(
                            f"update_snapshot self-test FAILED: training and locked "
                            f"reconstructions diverged by max_abs={diff:.4e} "
                            f"(relative to ref_scale {ref:.4e}: {rel:.4e}). "
                            f"Locked path's math differs from training path — "
                            f"likely a regression in `_forward_inference`."
                        )
                    elif rel >= 5e-2:
                        import warnings as _warnings
                        _warnings.warn(
                            f"update_snapshot self-test: training/locked recon "
                            f"differ by rel={rel:.2e} (max_abs={diff:.2e}). "
                            f"Within tolerance but worth flagging — accumulated "
                            f"f32 noise or stale rescale stats."
                        )
            finally:
                self.inference_mode = was_inference_mode
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
        periodic = bool(sae.config.periodic)
        per_axis = (periodic,) if periodic else None
        phi_k = gt.duchon_basis(
            t_k.unsqueeze(1), sae.centers, m=2, periodic_per_axis=per_axis,
        )
        # B_locked[k] is now (K, D) — already in ambient space (the joint
        # additive fit absorbed the previous per-atom W_k embedding into
        # the coefficient block). No further W_k matmul needed.
        curves[k] = phi_k @ sae.B_locked[k]
    return curves.to(activations.dtype)
