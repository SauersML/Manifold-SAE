# Manifold-SAE 2D — native multi-dim manifold atoms

The 1D architecture (`manifold_sae/sae.py`) has each atom k carry a
single scalar `t_k ∈ [0, 1]` and a smooth curve `g_k: [0, 1] → ℝ^D`.
That captures 1D manifolds (magnitudes, polarities, weekdays,
positions). It cannot capture 2D manifolds (grids, cylinders, tori,
day×hour, hue×saturation, position-on-a-game-board) in a single atom.

`manifold_sae/sae_2d.py` extends each atom to a 2D parameterization
`(t_k, s_k) ∈ [0, 1]²` and a smooth surface `g_k: [0, 1]² → ℝ^R`
embedded in `ℝ^D` via the per-atom direction matrix `W_k`.

This replaces the post-hoc clustering pipeline (Bhalla et al. 2026)
with native architectural manifold discovery: one atom = one
manifold, of dimensionality up to 2.

## Architectural changes

### Atom representation

| Quantity | 1D shape | 2D shape | Role |
| --- | --- | --- | --- |
| Position | `(B, F)` — `t_k` | `(B, F, 2)` — `(t_k, s_k)` | Encoder output |
| Amplitude | `(B, F)` | `(B, F)` | Sparse via TopK |
| Spline coefficients `B_k` | `(K, R)` | `(K², R)` | gamfit-owned per batch / snapshot |
| Smoothing `λ_k` | scalar | `(λ_t_k, λ_s_k)` — two per atom | REML-selected |
| Ambient subspace `W_k` | `(D, R)` | `(D, R)` | Adam-owned |
| Knot centers | `(K,)` shared | `(K,)` shared, used in tensor product | Constant |

`R` (intrinsic rank) defaults to 2 — same as 1D. The 2D manifold has
its own 2D coordinate system embedded into an R-dim subspace of ℝ^D.

### Encoder

Two scalar heads per feature instead of one:

```
(z_t, z_s, amp_logits) = encoder(x_centered, y_proj)
positions_t = soft_rescale(z_t)     # (B, F) → [0, 1]
positions_s = soft_rescale(z_s)     # (B, F) → [0, 1]
mask_binary = topk_gate(amp_logits) # (B, F)
```

Each atom's 2D coordinate is independent — separate soft-rescale per
axis. The Adam-owned linear encoder gets `2F` output heads instead of
`F` plus the existing amp head.

### Basis: tensor-product Duchon m=2

The 1D basis `φ_1d: [0, 1] → ℝ^K` is shared across atoms via the
shared `centers` buffer. The 2D basis is the tensor product:

```
φ_2d(t, s) = φ_1d(t) ⊗ φ_1d(s)  ∈ ℝ^{K²}
```

The atom's contribution at `(t, s)` is

```
g_k(t, s) = (φ_2d(t, s) · B_k) @ W_kᵀ   ∈ ℝ^D
```

with `B_k ∈ ℝ^{K² × R}`.

### REML with two smoothing parameters per atom

For a 2D Duchon m=2 spline the natural penalty is

```
J(f) = ∫∫ (∂²f/∂t²)² + 2(∂²f/∂t∂s)² + (∂²f/∂s²)² dt ds
```

For the tensor-product basis this becomes a sum of three Kronecker
products of the 1D penalty `P_1d`:

```
J(B) = trace(Bᵀ (P_1d ⊗ I + I ⊗ P_1d) B)
```

(The cross-derivative term drops to zero under tensor product because
∂²(φ_1d(t) φ_1d(s))/∂t∂s = φ_1d'(t) φ_1d'(s) has zero L² inner
product against ∂²(φ_1d(t') φ_1d(s'))/∂t'∂s' integrated over the
square, when expanded — this is the standard tensor-product penalty
simplification, see Wood 2017 §5.6.)

With **separable smoothing**:

```
J(B) = λ_t · trace(Bᵀ (P_1d ⊗ I) B) + λ_s · trace(Bᵀ (I ⊗ P_1d) B)
```

REML jointly selects both `λ_t` and `λ_s`. gamfit's
`gaussian_reml_fit_batched` accepts arbitrary penalty matrices; we
build the Kronecker-product penalty and pass it.

Pragmatic v1 falls back to a single λ via

```
P_2d = P_1d ⊗ I + I ⊗ P_1d
J(B) = λ · trace(Bᵀ P_2d B)
```

which is one λ-search per atom (cheaper, slightly less principled).
The per-axis smoothing can come back later.

### Identification priors

| Prior (1D) | Adapted to 2D |
| --- | --- |
| `sparsity` on amplitudes | unchanged |
| `subspace_ortho` per-feature W ortho | unchanged |
| `position_coverage` over `[0, 1]` | now a 2D Gaussian-bin coverage over `[0, 1]²` (n_bins × n_bins grid) |
| `monotonicity` | **dropped** — doesn't generalize to 2D; replaced with **isotropy** prior: penalize anisotropy between t-span and s-span of (positions × amplitudes) so neither axis collapses |

### Lock-and-cache

`update_snapshot` runs the per-batch 2D REML fit on a reference
batch, then writes `B_locked: (F, K², R)`, `lam_t_locked: (F,)`,
`lam_s_locked: (F,)`, and the 2D rescale stats
`soft_min_locked: (F, 2)`, `soft_max_locked: (F, 2)`.

Inference path: feedforward, no gamfit call. Same shape as a 1D
locked atom plus a slightly larger basis evaluation
(`K²` instead of `K` per atom-token).

### Self-test in `update_snapshot`

Same as 1D: training-mode and locked-mode forward on the snapshot
batch must agree within 5% relative tolerance.

## When does an atom degrade to 1D?

The encoder's `(z_t, z_s)` are independent — nothing forces the atom
to use both dimensions. If the data only varies along one axis, the
encoder learns to either:

1. Set `s_k ≈ constant` across all firing tokens (the atom uses only
   `t_k`). REML still selects a large `λ_s` to keep `g_k(t, s)` flat
   along `s`. The atom is effectively 1D.
2. Allocate signal to one axis. The other axis collapses.

So a 2D atom can degenerate gracefully to 1D when 1D suffices. We
report a per-atom *intrinsic dimensionality* score: the ratio
`std(positions_s) / std(positions_t)` across firing tokens (after
soft-rescale). Atoms with the ratio near 0 are using one axis;
atoms with the ratio near 1 are genuinely 2D.

## Topology

This v1 covers **plane** and **disk** (non-periodic in both axes).
For:

- **Cylinder** (one axis periodic): swap one of the basis dimensions
  to periodic Duchon. Plumbing change only.
- **Torus** (both periodic): both axes periodic Duchon. Same.
- **Sphere** (no boundary, non-square): not supported; requires a
  different parameterization (e.g. spherical harmonics).

For day-of-week × hour-of-day or position-on-grid, the plane
parameterization with periodic boundaries handles it. Goodfire's ICL
grid task is non-periodic; Goodfire's weekday cylinder task is
one-periodic.

## Open questions

- **REML joint search over two λ per atom**: gamfit batched REML
  supports multi-smoothing-parameter selection but the autograd
  backward gets more involved. v1 ships single-λ; v2 adds joint.
- **Encoder cost**: doubles the per-feature head count (two position
  heads). For F=10K this matters; for F=128-4096 (current scale)
  it's free.
- **Coverage prior shape**: 2D grid binning is `O(B·F·n_bins²)`.
  At F=128, n_bins=10 → 12.8M ops per batch, fine.

## Status

`manifold_sae/sae_2d.py` implements the v1 (single λ, non-periodic).
Synthetic 2D recovery test in
`experiments/synthetic_2d_recovery.py` plants a known 2D structure
(grid + helix on a cylinder) and measures per-atom recovery.
