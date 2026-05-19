"""Differentiable torch glue around ``gamfit.gaussian_reml_fit_positions_batched``.

This module wraps the gamfit Rust primitive ``gaussian_reml_fit_positions_batched``
(and its analytic VJP ``gaussian_reml_fit_positions_batched_backward``) in a
``torch.autograd.Function`` so the rest of the Manifold-SAE project can use it
as a normal differentiable component inside a torch graph.

Contract
--------
The downstream-facing interface is the dataclass :class:`BasisSpec`, the
autograd primitive :class:`ManifoldFit`, and the functional wrapper
:func:`manifold_fit`. Inputs are an ``(B, F)`` matrix of positions in ``[0, 1]``,
an ``(B, F)`` matrix of non-negative amplitudes (used as gamfit's ``by`` gate),
and an ``(B, D)`` matrix of reconstruction targets. Outputs are a reconstruction
``(B, D)`` formed by summing the F per-feature smooth fits, a scalar REML score
formed by summing the F per-feature REML scores, plus per-feature ``lambdas``
``(F,)`` and effective-degrees-of-freedom ``edf`` ``(F,)``.

Per-feature semantics
---------------------
Each of the F SAE features is fit as one independent closed-form Gaussian REML
problem with its own 1D smooth basis (n_basis = K, periodic flag from
:class:`BasisSpec`) evaluated at ``positions[:, f]``. Amplitudes act as the
``by``-gate that multiplies each design row, so a token with zero amplitude
contributes a zero design row to that feature's fit. The K=F independent fits
share the response matrix ``targets`` — each feature independently tries to
explain ``targets``, and the reconstruction is the SUM over features. This is
the SAE-decoder pattern: every feature contributes additively and the encoder
side is responsible for choosing which features fire.

Packing
-------
The F per-feature problems are packed into a single batched call by
concatenating positions in feature-major order, tiling targets F times along
axis 0, and packing amplitudes the same way. ``row_offsets`` then marks each
feature's slice of length B. This lets us hit the gamfit primitive once per
forward (and once per backward) instead of looping in Python.

Upstream gap (still open as of gamfit 0.1.67)
---------------------------------------------
**No analytic backward for ``edf``.** ``gaussian_reml_fit_positions_batched_backward``
accepts ``grad_lambda``, ``grad_coefficients``, ``grad_fitted``, ``grad_reml_score``
but no ``grad_edf``. If a caller requests gradients via ``edf`` we treat that
upstream gradient as zero (silently dropped). Downstream code that wants
``edf``-driven regularization needs a different path until gamfit ships an
``edf`` backward.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Imports of gamfit happen lazily inside the autograd Function so that
# ``import manifold_sae.gamfit_glue`` does not fail when gamfit is missing in
# environments that only need the dataclass / type definitions.


@dataclass(frozen=True)
class BasisSpec:
    """Specification of the 1D smooth basis used per feature.

    v1 uses 1D Duchon order-2 splines on ``[0, 1]`` for every feature. Cyclic
    concepts are fit as approximately-closed open curves with a seam at the
    endpoints — the v1 spec accepts this, and periodic Duchon lacks
    end-to-end REML support in gamfit anyway.

    Parameters
    ----------
    n_basis:
        Number of basis functions ``K`` per feature. Penalty matrix is ``(K, K)``.
    init_lambda:
        Seed for the REML smoothing-parameter optimiser. Fixing this makes
        forward + backward numerically deterministic across small input
        perturbations, which is critical for gradcheck and stable training.
        Default ``1.0``.
    """

    n_basis: int
    init_lambda: float | None = 1.0


# --------------------------------------------------------------------------- #
# Basis / penalty construction
# --------------------------------------------------------------------------- #


def _build_basis_materials(basis_spec: BasisSpec) -> tuple[str, np.ndarray, np.ndarray, int]:
    """Return ``(basis_kind, knots_or_centers, penalty, basis_order)``.

    Every feature uses 1D Duchon order-2 on ``[0, 1]``.
    """
    K = int(basis_spec.n_basis)
    if K < 3:
        raise ValueError(f"BasisSpec.n_basis must be >= 3 for Duchon order 2, got {K}")
    centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
    penalty = np.eye(K, dtype=np.float64)
    return "duchon", centers, penalty, 2


def build_penalty(basis_spec: BasisSpec) -> torch.Tensor:
    """Return the ``(K, K)`` penalty matrix used internally by :class:`ManifoldFit`.

    This is forward-only and exists so callers can inspect or pre-compute the
    penalty without going through ``ManifoldFit.apply``. The returned tensor is
    f64 on CPU; callers free to move it.
    """
    _, _, penalty, _ = _build_basis_materials(basis_spec)
    return torch.from_numpy(np.ascontiguousarray(penalty, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Packing helpers
# --------------------------------------------------------------------------- #


def _pack_inputs(
    positions: np.ndarray,
    amplitudes: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Pack (B, F) positions / amplitudes / (B, D) targets into the batched layout.

    Returns
    -------
    t_packed: ``(F*B,) float64``
    y_packed: ``(F*B, D) float64``  — targets tiled F times along axis 0
    by_packed: ``(F*B,) float64`` — amplitudes packed feature-major
    offsets: ``(F+1,) uintp``  — ``[0, B, 2B, ..., FB]``
    B, F, D: ints
    """
    B, F = positions.shape
    if amplitudes.shape != (B, F):
        raise ValueError(
            f"amplitudes shape {amplitudes.shape} does not match positions {positions.shape}"
        )
    if targets.ndim != 2 or targets.shape[0] != B:
        raise ValueError(
            f"targets must have shape (B, D) with B={B}; got {targets.shape}"
        )
    D = int(targets.shape[1])

    # Feature-major packing: t_packed[f*B : (f+1)*B] = positions[:, f].
    t_packed = np.ascontiguousarray(positions.T.reshape(-1).astype(np.float64, copy=False))
    by_packed = np.ascontiguousarray(amplitudes.T.reshape(-1).astype(np.float64, copy=False))
    y_packed = np.ascontiguousarray(np.tile(targets.astype(np.float64, copy=False), (F, 1)))
    offsets = (np.arange(F + 1, dtype=np.uintp) * np.uintp(B))
    return t_packed, y_packed, by_packed, offsets, B, F, D


# --------------------------------------------------------------------------- #
# Autograd Function
# --------------------------------------------------------------------------- #


class ManifoldFit(torch.autograd.Function):
    """Differentiable wrapper around the packed position-based Gaussian REML fit.

    See module docstring for the contract.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        positions: torch.Tensor,
        amplitudes: torch.Tensor,
        targets: torch.Tensor,
        basis_spec_bytes: bytes,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import gamfit  # local import so module-level import does not require gamfit

        basis_spec: BasisSpec = pickle.loads(basis_spec_bytes)
        basis_kind, knots_or_centers, penalty, basis_order = _build_basis_materials(basis_spec)

        # Detach inputs into contiguous f64 numpy arrays on CPU. The Rust
        # backend runs strictly on CPU f64; we will lift outputs back to
        # ``positions``' device/dtype at the end.
        positions_np = positions.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        amplitudes_np = amplitudes.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        targets_np = targets.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()

        t_packed, y_packed, by_packed, offsets, B, F, D = _pack_inputs(
            positions_np, amplitudes_np, targets_np
        )

        out = gamfit.gaussian_reml_fit_positions_batched(
            t_packed,
            y_packed,
            offsets,
            basis_kind,
            knots_or_centers,
            penalty,
            basis_order=basis_order,
            periodic=False,
            period=None,
            by=by_packed,
            init_lambda=basis_spec.init_lambda,
        )

        # ``fitted`` is (F*B, D). Reshape to (F, B, D) and sum over F to get the
        # additive reconstruction.
        fitted = np.asarray(out["fitted"], dtype=np.float64)  # (F*B, D)
        per_feature = fitted.reshape(F, B, D)
        reconstruction_np = per_feature.sum(axis=0)
        reml_per_feat = np.asarray(out["reml_score"], dtype=np.float64)  # (F,)
        lambdas = np.asarray(out["lambda"], dtype=np.float64)  # (F,)
        edf = np.asarray(out["edf"], dtype=np.float64)  # (F,)
        coefficients = np.asarray(out["coefficients"], dtype=np.float64)  # (F, K, D)
        # Degenerate features (fully gated off via by=0) yield NaN reml/lambda
        # because the design matrix is rank-deficient. Replace with 0 so the
        # reduced scalar REML score and per-feature lambdas remain finite.
        # gamfit signals these via ``status[f] == 'degenerate'``.
        statuses = list(out.get("status", []))
        for f_idx, st in enumerate(statuses):
            if st != "ok":
                reml_per_feat[f_idx] = 0.0
                if not np.isfinite(lambdas[f_idx]):
                    lambdas[f_idx] = 0.0
                if not np.isfinite(edf[f_idx]):
                    edf[f_idx] = 0.0
        reml_total = float(reml_per_feat.sum())

        # Stash context for backward. Tensors that flow gradient back to the
        # caller go through ``save_for_backward`` to honour autograd's version
        # counter machinery. Everything else lives as attributes.
        ctx.save_for_backward(positions, amplitudes, targets)
        ctx.basis_kind = basis_kind
        ctx.knots_or_centers = knots_or_centers
        ctx.penalty = penalty
        ctx.basis_order = basis_order
        ctx.positions_np = positions_np
        ctx.amplitudes_np = amplitudes_np
        ctx.targets_np = targets_np
        ctx.t_packed = t_packed
        ctx.y_packed = y_packed
        ctx.by_packed = by_packed
        ctx.offsets = offsets
        ctx.B = B
        ctx.F = F
        ctx.D = D
        ctx.init_lambda = basis_spec.init_lambda
        # Per-feature lambdas are stashed so backward can mask out gradients on
        # features whose smoothing parameter has saturated. gamfit's analytic
        # backward divides by a near-zero quantity at the lambda upper bound and
        # returns spurious O(1e6) gradients; since those features contribute
        # zero to the reconstruction anyway, their input gradients should also
        # be zero. Threshold chosen well above any meaningful smoothing param
        # the SAE would learn (REML drives lambda monotonically into [0, 1e10]
        # for well-conditioned features).
        ctx.lambdas_np = lambdas
        ctx.feature_statuses = statuses

        ref = positions
        reconstruction = torch.as_tensor(reconstruction_np, dtype=torch.float64, device="cpu").to(
            device=ref.device, dtype=ref.dtype
        )
        reml_score = torch.as_tensor(reml_total, dtype=torch.float64, device="cpu").to(
            device=ref.device, dtype=ref.dtype
        )
        lambdas_t = torch.as_tensor(lambdas, dtype=torch.float64, device="cpu").to(
            device=ref.device, dtype=ref.dtype
        )
        edf_t = torch.as_tensor(edf, dtype=torch.float64, device="cpu").to(
            device=ref.device, dtype=ref.dtype
        )
        coefficients_t = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu").to(
            device=ref.device, dtype=ref.dtype
        )
        return reconstruction, reml_score, lambdas_t, edf_t, coefficients_t

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        g_recon: torch.Tensor | None,
        g_reml: torch.Tensor | None,
        g_lambdas: torch.Tensor | None,
        g_edf: torch.Tensor | None,
        g_coefficients: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        import gamfit

        F = ctx.F
        B = ctx.B
        D = ctx.D

        # gamfit's batched-positions backward consumes upstream gradients on
        # three outputs: fitted (B, D per feature, tiled F times along axis 0
        # because the reconstruction is the SUM of per-feature fits), reml_score
        # (F,), and lambda (F,). It has no VJP for edf; g_edf is dropped.
        if g_recon is None:
            grad_fitted_np = None
        else:
            g_recon_np = g_recon.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
            grad_fitted_np = np.ascontiguousarray(np.tile(g_recon_np, (F, 1)))

        if g_reml is None:
            grad_reml_np = None
        else:
            g_reml_val = float(g_reml.detach().to(device="cpu", dtype=torch.float64).item())
            grad_reml_np = np.full(F, g_reml_val, dtype=np.float64)

        if g_lambdas is None:
            grad_lambda_np = None
        else:
            grad_lambda_np = np.ascontiguousarray(
                g_lambdas.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy(),
                dtype=np.float64,
            )

        if grad_fitted_np is None and grad_reml_np is None and grad_lambda_np is None:
            return None, None, None, None

        out = gamfit.gaussian_reml_fit_positions_batched_backward(
            ctx.t_packed,
            ctx.y_packed,
            ctx.offsets,
            ctx.basis_kind,
            ctx.knots_or_centers,
            ctx.penalty,
            grad_lambda=grad_lambda_np,
            grad_coefficients=None,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_np,
            basis_order=int(ctx.basis_order),
            periodic=False,
            period=None,
            weights=None,
            init_lambda=None if ctx.init_lambda is None else float(ctx.init_lambda),
            by=ctx.by_packed,
            by_start_col=0,
        )

        grad_t_packed = np.asarray(out["grad_t"], dtype=np.float64)  # (F*B,)
        grad_y_packed = np.asarray(out["grad_y"], dtype=np.float64)  # (F*B, D)
        grad_by_packed_raw = out.get("grad_by")
        if grad_by_packed_raw is None:
            grad_by_packed = np.zeros(F * B, dtype=np.float64)
        else:
            grad_by_packed = np.asarray(grad_by_packed_raw, dtype=np.float64)

        # Unpack feature-major (F, B) layout back to (B, F) for positions and
        # amplitudes. y was tiled F times so contributions to the same target
        # row are summed across features.
        grad_positions_np = grad_t_packed.reshape(F, B).T  # (B, F)
        grad_amplitudes_np = grad_by_packed.reshape(F, B).T  # (B, F)
        grad_targets_np = grad_y_packed.reshape(F, B, D).sum(axis=0)  # (B, D)

        # Mask gradients for features whose lambda saturated or whose forward
        # status was degenerate. gamfit's analytic VJP divides by an
        # ill-conditioned quantity at the lambda upper bound and emits O(1e6)
        # gradients that disagree with finite differences. Since the
        # reconstruction contribution of a saturated feature is essentially
        # zero (the smooth is fully penalised), its input gradients should be
        # zero too — masking is the locally correct fix until upstream patches
        # the boundary case.
        lam = ctx.lambdas_np
        statuses = getattr(ctx, "feature_statuses", []) or [""] * F
        SATURATED_LAMBDA = 1e10
        for f_idx in range(F):
            is_sat = (lam.size > f_idx and not np.isfinite(lam[f_idx])) or (
                lam.size > f_idx and lam[f_idx] >= SATURATED_LAMBDA
            )
            is_degen = (f_idx < len(statuses)) and (statuses[f_idx] not in ("", "ok"))
            if is_sat or is_degen:
                grad_positions_np[:, f_idx] = 0.0
                grad_amplitudes_np[:, f_idx] = 0.0

        positions, amplitudes, targets = ctx.saved_tensors

        def _wrap(arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
            t = torch.as_tensor(arr, dtype=torch.float64, device="cpu")
            return t.to(device=ref.device, dtype=ref.dtype)

        grad_positions = _wrap(grad_positions_np, positions) if positions.requires_grad else None
        grad_amplitudes = _wrap(grad_amplitudes_np, amplitudes) if amplitudes.requires_grad else None
        grad_targets = _wrap(grad_targets_np, targets) if targets.requires_grad else None

        return grad_positions, grad_amplitudes, grad_targets, None


def manifold_fit(
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    basis_spec: BasisSpec,
) -> dict[str, torch.Tensor]:
    """Functional wrapper around :class:`ManifoldFit`.

    Parameters
    ----------
    positions:
        ``(B, F)`` float tensor in ``[0, 1]``.
    amplitudes:
        ``(B, F)`` non-negative float tensor. Acts as gamfit's ``by`` gate.
    targets:
        ``(B, D)`` float tensor of reconstruction targets.
    basis_spec:
        :class:`BasisSpec` describing the per-feature smooth basis.

    Returns
    -------
    dict with keys ``reconstruction`` ``(B, D)``, ``reml_score`` scalar,
    ``lambdas`` ``(F,)``, ``edf`` ``(F,)``.
    """
    basis_bytes = pickle.dumps(basis_spec)
    recon, reml, lambdas, edf, coefficients = ManifoldFit.apply(
        positions, amplitudes, targets, basis_bytes
    )
    return {
        "reconstruction": recon,
        "reml_score": reml,
        "lambdas": lambdas,
        "edf": edf,
        "coefficients": coefficients,
    }
