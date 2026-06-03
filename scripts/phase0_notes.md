# Phase 0 — gamfit substrate reference

Quick-reference for the gamfit primitive that the Manifold-SAE decoder wraps.

> **Historical (gamfit 0.1.64).** These notes record the early
> reconnaissance against gamfit 0.1.64, including its broken-backward
> wrapper and the 0.1.64/0.1.66 wrapper-vs-binding version skew. The
> repo is now on gamfit 0.1.145 (standing rule: always newest gamfit); the
> version-skew and "broken in 0.1.64" notes below no longer describe the
> installed wheel. Kept as the phase-0 record.

## Versions tested

- Python wrapper: `gamfit` **0.1.64** (pip)
- Rust binding inside that wheel (`gamfit._rust`): crate **0.1.66**
- Verified on macOS arm64, Python 3.13, in `/Users/user/Manifold-SAE/.venv`.

## Imports

```python
import gamfit
from gamfit import _rust  # used only for backward — see WARNING
gamfit.gaussian_reml_fit_positions_batched
gamfit.gaussian_reml_fit_positions_batched_backward   # broken in 0.1.64 wrapper
_rust.gaussian_reml_fit_positions_batched_backward    # works
```

Source: `gamfit/_api.py` lines 969–1064. Rust binding:
`crates/gam-pyffi/src/lib.rs` (`gaussian_reml_fit_positions_batched` /
`..._backward`).

## Forward — `gaussian_reml_fit_positions_batched`

```
(t, y, row_offsets, basis_kind, knots_or_centers, penalty,
 *, basis_order=None, periodic=False, period=None,
 weights=None, init_lambda=None, by=None, by_start_col=0) -> dict
```

All array inputs are **dtype-strict** and validated by `_numeric_*` helpers
(zero-copy FFI). No automatic promotion — pass exactly these dtypes:

| arg                | shape          | dtype       | meaning |
|--------------------|----------------|-------------|---------|
| `t`                | `(N,)`         | `float64`   | packed 1-D positions, all `F` problems concatenated |
| `y`                | `(N, D)`       | `float64`   | packed responses, **matrix Y supported natively** |
| `row_offsets`      | `(F+1,)`       | `uintp`     | CSR-style offsets, must be non-decreasing, start at 0, end at N |
| `basis_kind`       | str            |             | `"bspline"` / `"spline"` or `"duchon"` / `"duchon_spline"`; case- / sep-insensitive |
| `knots_or_centers` | `(K,)`         | `float64`   | B-spline knots or Duchon centers |
| `penalty`          | `(P, P)`       | `float64`   | smoothing penalty matrix; rank-deficiency allowed |
| `basis_order`      | int            |             | B-spline degree (default 3) or Duchon `m` (default 2) |
| `periodic`         | bool           |             | enables periodic B-spline; **periodic Duchon is not supported on the position API** (requires `period=None`) |
| `period`           | float          |             | required when `periodic=True` for B-spline |
| `weights`          | `(N,)`         | `float64`   | optional row weights |
| `init_lambda`      | float          |             | warm-start the smoothing parameter |
| `by`               | `(N,)`         | `float64`   | per-row "by" gate, applied to columns `[by_start_col:]` of the design matrix |
| `by_start_col`     | int            |             | first design column the `by` gate multiplies |

For Manifold-SAE: the per-feature **amplitude / gate** in the spec maps to
`by`, with `by_start_col=0` so the gate multiplies the whole basis.

### B-spline column count

For non-periodic B-spline with `K` knots and degree `d`, the basis has
`P = K - d - 1` columns. Use `gamfit.smoothness_penalty(knots, degree=d,
order=2)` to get a compatible `(P, P)` penalty.

### Returned dict (verified at runtime)

Per-batch (F = number of problems = `len(row_offsets) - 1`):

| key                              | shape         | notes |
|----------------------------------|---------------|-------|
| `status`                         | list[str]     | `"ok"` or e.g. `"degenerate"` per fit |
| `lambda`                         | `(F,)`        | estimated smoothing parameter (positive) |
| `rho`                            | `(F,)`        | `log(lambda)` |
| `reml_score`                     | `(F,)`        | finite, may be very negative |
| `reml_grad_lambda`, `reml_hess_lambda` | `(F,)`  | REML score derivatives at the optimum |
| `reml_grad_rho`, `reml_hess_rho` | `(F,)`        | same in `rho` parameterisation |
| `edf`                            | `(F,)`        | effective degrees of freedom; in `(0, P]` |
| `coefficients`                   | `(F, P, D)`   | per-feature spline coefficients per ambient dim |
| `fitted`                         | `(N, D)`      | fitted values, packed in same order as `t` / `y` |
| `sigma2`                         | `(F, D)`      | residual variance per dim |
| `cache_penalty_eigenvalues`      | `(F, P)`      | optional warm-cache for repeated fits |
| `cache_eigenvectors`             | `(F, P, P)`   | "                                   |
| `cache_coefficient_basis`        | `(F, P, P)`   | "                                   |
| `cache_xtwx_fingerprints`        | `(F,)` u64    | "                                   |
| `cache_penalty_fingerprints`     | `(F,)` u64    | "                                   |
| `cache_logdet_xtwx`              | `(F,)`        | "                                   |
| `cache_logdet_penalty_positive`  | `(F,)`        | "                                   |
| `cache_penalty_ranks`            | `(F,)` i64    | "                                   |
| `cache_nullities`                | `(F,)` i64    | "                                   |

`λ̂` lambda is estimated internally (REML); pass `init_lambda` to warm-start.

## Backward — `gaussian_reml_fit_positions_batched_backward`

Separate function. Same forward arg list, plus four optional output cotangents
and an optional cache pointer:

```
(t, y, row_offsets, basis_kind, knots_or_centers, penalty,
 grad_lambda=None, grad_coefficients=None, grad_fitted=None,
 grad_reml_score=None, forward_state=None,
 basis_order=3, periodic=False, period=None,
 weights=None, init_lambda=None, by=None, by_start_col=0)
```

- `grad_lambda`     : `(F,)` float64 cotangent on `lambda`
- `grad_coefficients`: `(F, P, D)` float64
- `grad_fitted`     : `(N, D)` float64
- `grad_reml_score` : `(F,)` float64
- `forward_state`   : optional dict carrying the per-fit cache from forward to
  skip redundant work (the reference torch wrapper passes `None`).

Returns dict with keys `status`, `grad_t` `(N,)`, `grad_y` `(N, D)`,
`grad_weights` `(N,)`, `grad_by` `(N,)` or `None`. No gradient flows through
`knots_or_centers`, `penalty`, or any of the basis-config args — they are
structural.

Smoke-test confirmed: with `grad_fitted = ones` and zero gradient elsewhere,
`grad_y` is propagated (≡ ones in this test), `grad_t` is well-defined and
finite (≈1e−13 because the fit nearly reproduces `y` and `d(fitted)/dt` is
small at the optimum on random positions), and `grad_by` is `None` since no
`by` was passed.

## Phase-1 `torch.autograd.Function` shape

Saved-for-backward state: `t_np`, `y_np`, `row_offsets_np`, `knots_np`,
`penalty_np`, `weights_np`, `by_np`, plus the (non-tensor) `basis_kind`,
`basis_order`, `periodic`, `period`, `init_lambda`, `by_start_col`. The
reference upstream `gamfit/torch/_reml.py` already implements this and is a
good template (it doesn't pass `forward_state`, matching what we do here).

Forward returns four torch tensors `(coefficients, fitted, lambda,
reml_score)`. Backward receives four cotangents in the same order and routes
them to the corresponding `grad_*` kwargs. The autograd.Function returns
`grad_t`, `grad_y`, and optionally `grad_weights`, `grad_by`; everything else
is `None` (structural).

## Multi-response

Y as a `(N, D)` matrix is supported natively. Output `coefficients` is `(F,
P, D)`, `fitted` is `(N, D)`. No D-dim loop needed.

## Periodic vs non-periodic

- **B-spline**: both supported. Periodic requires `periodic=True` and an
  explicit positive finite `period`. Periodic B-spline has full REML support
  (uses `create_periodic_bspline_basis_dense` and matching derivative routine
  for the backward `grad_t` contraction).
- **Duchon**: non-periodic only on the position API. The Rust code calls
  `validate_position_period("duchon", ..., periodic, period)` which **rejects
  `periodic=True`** (and accepts only `period=None` when `periodic=False`).
  Matches the Manifold-SAE spec note that periodic Duchon does not have full
  REML support.

## Sanity numbers (this smoke test)

`F=4, B=256, D=8, K=12 (knots), degree=3 → P=8`, targets `amp_f · sin(2π t +
phase_d) + 0.01·N(0,1)`. Got per-fit RMS residual ≈ 0.054, `λ̂ ∼ 1e−6`,
`edf ≈ 7.9 / 8`, REML score finite. All four fits status `"ok"`.

## WARNING — gotchas Phase 1 must know

1. **Wrapper / binding version skew (PyPI 0.1.64)**.
   `gamfit.gaussian_reml_fit_positions_batched_backward` in the published
   wheel does **not** pass the `forward_state` positional argument that the
   bundled Rust binding (crate 0.1.66) requires. As a result calling the
   wrapper raises:
   `TypeError: argument 'forward_state': 'int' object is not an instance of
   'dict'` (`int(by_start_col)` lands in the `forward_state` slot).
   **Workaround used here**: call
   `gamfit._rust.gaussian_reml_fit_positions_batched_backward` positionally
   with `forward_state=None`. Phase 1's `torch.autograd.Function.backward`
   should do the same, or bump `gamfit` to a release where `_api.py` is
   re-synced — needs upstream confirmation. **The forward wrapper is fine.**

2. **dtype is strict and zero-copy**: float64 for every numeric arg, `uintp`
   for `row_offsets`. A wrong dtype raises `TypeError` from `_numeric_*`
   helpers — there is no implicit cast.

3. **`row_offsets` must start at 0 and end at `t.size`**, and be
   non-decreasing. The Rust binding checks this explicitly and errors out
   otherwise. Empty segments are allowed (offsets can repeat).

4. **Penalty rank deficiency is expected**: for `order=2` difference penalty
   `cache_nullity = 2`. Don't assume `penalty` is positive definite.

5. **`status` is per-fit**: with random / pathological data some fits may
   come back `"degenerate"`. Phase 1 should treat non-`"ok"` fits as
   recoverable (e.g. mask them out of the loss) rather than crashing.

6. **`init_lambda` is a single scalar** for the whole batch (not per-fit) in
   the forward API, but `grad_lambda` in backward is a per-batch vector
   `(F,)`. Don't mix them up.

7. **Periodic Duchon is rejected**. Use periodic B-spline if you need a
   periodic basis with full REML support.

8. **Gradient through structural args is `None`**: `knots_or_centers`,
   `penalty`, `basis_kind`, `basis_order`, `periodic`, `period`,
   `init_lambda`, `by_start_col`. Do not try to optimize these via autograd.
