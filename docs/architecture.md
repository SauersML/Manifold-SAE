# Architecture

This document describes the Manifold-SAE architecture as implemented in `manifold_sae/sae.py`. The companion `README.md` summarizes; this document drills in.

## Overview

A Manifold-SAE has F features. Each feature is a smooth 1D curve in residual stream:

```
g_k : [0, 1] → ℝ^D
```

The encoder receives a token x ∈ ℝ^D and produces, for each feature k:

- a non-negative amplitude `a_k(x)` (sparse via TopK on the amplitude logits)
- a position `t_k(x) ∈ [0, 1]` along feature k's curve

The reconstruction is

```
x̂ = b_dec + Σ_k a_k(x) · g_k(t_k(x))
```

`b_dec` is a learned decoder pre-bias (standard SAE practice — absorbs the activation mean so atoms don't waste capacity on it).

## The curve

For a given feature k, the curve is factored:

```
g_k(t) = (φ(t) B_k) W_kᵀ
```

- `φ(t) ∈ ℝ^K` is the value of a Duchon m=2 thin-plate basis with K knots at t. The basis is shared across all features.
- `B_k ∈ ℝ^{K × R}` is the per-feature spline coefficient matrix in the feature's R-dimensional intrinsic subspace.
- `W_k ∈ ℝ^{D × R}` is the per-feature ambient subspace — embeds the R-dim curve back into ℝ^D.

R is the *per-feature intrinsic rank*. Default R=2 (enough capacity for any 2D embedded curve like a parabola; matching R to the GT max rank avoids noise-dimension wiggle — see `docs/known_issues.md`).

## Adam-owned vs gamfit-owned

| Quantity | Shape | Owner | When updated |
| --- | --- | --- | --- |
| Encoder weights | varies | Adam | Every batch via backprop |
| `W_k` ambient subspace | `(F, D, R)` | Adam | Every batch via backprop |
| `b_dec` decoder pre-bias | `(D,)` | Adam | Every batch via backprop |
| `B_k` spline coefficients | `(F, K, R)` | gamfit per batch | Returned from gamfit's REML solve given the encoder's current positions |
| `λ_k` smoothing scalar | `(F,)` | gamfit per batch | Selected by REML each batch |
| `centers` knots | `(K,)` | constant | Linspace `[0, 1]` at init |

The split is what makes the methodological claim — "smoothness selected by REML" — actually true. Adam descends the SAE reconstruction loss, which backpropagates through gamfit's analytic backward, so the encoder learns positions that admit a smooth, well-fitting curve.

## Training step

```python
# 1. Encoder
positions, amplitudes = encoder(x_centered)        # (B, F), (B, F)

# 2. Project x onto each feature's R-dim subspace
y_proj = einsum("bd,fdr->bfr", x_centered, W)      # (B, F, R)

# 3. gamfit REML — per batch, autograd-aware
fit = gamfit.gaussian_reml_fit_positions_batched(
    positions, y_proj, by=amplitudes,
    basis_kind="duchon", basis_order=2, periodic=False,
    knots_or_centers=centers, penalty=None,         # auto-derived
)
# fit.coefficients: (F, K, R) — autograd-aware via implicit function theorem
# fit.lam        : (F,)       — REML-selected per feature
# fit.fitted     : (F·B, R)   — model prediction at this batch's positions
# fit.reml_score : (F,)       — REML log-marginal-likelihood

# 4. Reconstruct
fitted = fit.fitted.view(F, B, R)
recon = einsum("fbr,fdr->bfd", fitted * amplitudes.t().unsqueeze(-1), W).sum(dim=1) + b_dec

# 5. Loss (sparsity + identification priors; REML score available for diagnostics)
loss = mse(recon, x) + λ_sparse·|amps|.mean() + λ_ortho·ortho_loss(W) + ...

# 6. Backward + step
loss.backward()  # backprops through gamfit into encoder + W
optimizer.step()
```

REML score is *not* added to the training loss as an explicit term. Doing so creates a degenerate maximum at all-zero amplitudes (gamfit's REML score on zero-weight features is unbounded above because there's no residual to fit and no penalty paid). The MSE on the SAE reconstruction is sufficient — gamfit is descending REML internally to select λ; the encoder learns positions that yield good reconstructions, which is the gradient signal we want.

## Soft position rescale

The encoder outputs unbounded position logits `z_raw`. We rescale these to `[0, 1]` per batch via a smooth min-max:

```python
def soft_rescale(z_raw, beta=10.0, eps=1e-4):
    soft_max =  (1/beta) · logsumexp( beta·z_raw, dim=batch)
    soft_min = -(1/beta) · logsumexp(-beta·z_raw, dim=batch)
    span = max(soft_max - soft_min, 1e-6)
    t = (z_raw - soft_min) / span
    return t.clamp(eps, 1 - eps)
```

This gauge-fixes the position so atoms see positions spanning the basis domain — no init-clustering pathology. `λ_k` and `B_k` are reparameterization-invariant under monotone rescaling of `t`, so this is lossless. Zero parameters, O(B·F) work.

## Lock-and-cache

After training, we freeze the curve coefficients and smoothing parameters:

```python
sae.update_snapshot(reference_batch)  # one big REML fit on a held-out batch
sae.inference_mode = True
```

`update_snapshot` runs the SAE in training mode on the reference batch (so gamfit returns fresh B and λ), then copies them into `B_locked` and `lam_locked` buffers. After setting `inference_mode = True`, the forward path no longer calls gamfit:

```python
phi = duchon_basis_1d(positions, centers, m=2, periodic=False)  # (F·B, K)
g = einsum("fbk,fkr->fbr", phi, B_locked)                       # (F, B, R)
recon = einsum("fbr,fdr->bfd", g * amplitudes.t().unsqueeze(-1), W).sum(dim=1) + b_dec
```

Single-token-evaluable. Same shape and cost as a vanilla feedforward SAE.

## Loss components

| Term | Weight | Purpose |
| --- | --- | --- |
| `mse(recon, x)` | 1 | Reconstruction (the only term whose gradient drives encoder + W through REML) |
| `sparsity · mask_soft.mean()` | `sparsity_weight` (~1e-3) | Standard SAE sparsity |
| `ortho_loss(W)` | `ortho_weight` (~1e-3) | Per-feature column ortho + cross-feature off-block diversity |
| `coverage(positions, mask)` | `1e-2` | KL from uniform on soft-binned firing-position histogram |
| `monotonicity(positions, y_proj_principal)` | `1e-2` | `1 − \|Pearson(position, principal projection)\|` (parameterization tiebreaker) |

What's *not* here: smoothness penalty (gamfit owns it via REML), curve-norm gauge (REML's likelihood pins amp-curve scale), cumulant penalty (sparsity + REML are enough).

## Encoder

Two options, both implemented:

- **Per-feature MLP** (`encoder.py`). Toy scale; one tiny MLP per feature. Quadratic in F.
- **Shared 2-layer MLP** (`encoder_linear.py`, `encoder_type="linear"`). Shared trunk D → H=4·D, two heads to F (position + amplitude). Linear in F. Use this for LLM scale.

Hidden dim is **fixed at 4·D** (not scaled with F). Earlier the default was `max(4·D, 2·F)` — that made the encoder quadratic in F and infeasible at LLM scale.

Both encoders support a `continuous_amp` mode where the amplitude lane is `softplus(logit) · binary_topk_gate` instead of `sigmoid · binary_topk_gate`. Continuous amp lets the amplitude carry magnitude (proper LLM-style gauge) instead of relying on the curve to absorb amplitude variation. Default on for the LLM driver; off for the toy.

## Why not a persistent B as `nn.Parameter`?

That was the architecture before commit `74c307a`. It worked locally on the toy but doesn't actually do REML — `B_k` was Adam-owned, and the "smoothness loss" was a hand-rolled `λ · tr(BᵀSB)` quadratic penalty. The log-determinant term required for the REML criterion was missing, so Adam pulled λ → 0 without resistance. The methodological claim that smoothness was selected by REML wasn't true under that architecture.

Moving B to gamfit per batch + lock-and-cache for inference restores the claim while preserving single-token feedforward deployment. See commit message of `74c307a` for the migration details.

## Identifiability and gauge

The model has natural gauge ambiguities REML doesn't resolve:

- **Amplitude / curve gauge.** `(α · a_k, g_k / α)` produces the same `a_k · g_k`. REML's data fit pins this up to a scalar — `by = amplitude` weights the gamfit fit, so a smaller amplitude with a bigger curve fits the same data. We add a light curve-norm prior in `continuous_amp` mode that targets `‖g_k‖_avg ≈ 1`.
- **Position parameterization.** `t ↦ φ(t)` and `t ↦ φ(1-t)` produce reflection-equivalent curves. The soft monotonicity prior breaks this for features whose data admits a monotone parameterization.
- **Per-feature reflection.** `W_k → -W_k`, `B_k → -B_k` is a symmetry. The data fit constrains it up to sign at convergence.
- **Cross-feature subspace overlap.** Without an ortho prior, two features can land in the same `W_k`. The cross-feature off-block ortho penalty prevents this.

These are *identification* constraints, not smoothness constraints, so they stay even under correct REML.
