# gamfit — what the API should look like

## Audience

Three personas drive the design:

1. **Statisticians** porting R / mgcv code. Want familiar formula syntax,
   sensible defaults, REML by default. Don't care about autograd.
2. **Classical applied scientists** doing GAMs on tabular data — survival,
   epidemiology, ecology. Want robust families, predictions, intervals.
3. **ML researchers** embedding GAM-shaped layers inside larger neural
   networks. Need differentiable primitives, multi-input smooths, gated
   atoms, fast inference. Don't want a formula parser; want composable
   building blocks.

Persona 3 is currently under-served — the torch path exposes a thin slice
of what the Rust engine can do. This document focuses on the API needed
to fully serve persona 3 without compromising 1 or 2.

## Architectural layering

```
Layer 5  torch.nn.Module       ← drop-in for embedding in neural nets
Layer 4  torch primitives      ← autograd-differentiable basis + REML
Layer 3  Python high-level API ← formula parser, fit(), predict()
Layer 2  Python primitives     ← numpy wrappers around Rust functions
Layer 1  Rust engine           ← basis, penalty, REML, LAML math
```

The Rust engine (Layer 1) is mostly fine — it has multi-dim Duchon, joint
multi-smooth REML, all the math. Layers 2 and 4 only expose a 1D-friendly
subset. The fix is API surface, not math.

## What gamfit/torch SHOULD have

### Naming

Group by concern, drop dim-specific suffixes (dimensionality is inferred
from input shape).

```
gamfit.torch.basis              # basis function families
gamfit.torch.penalty            # penalty matrix constructors
gamfit.torch.reml               # REML / LAML fitters
gamfit.torch.module             # nn.Module wrappers
```

### Layer 4 — autograd primitives

```python
# gamfit/torch/basis.py

def duchon(
    points: torch.Tensor,    # (N, d)
    centers: torch.Tensor,   # (K, d)
    *,
    m: int = 2,
    periodic_per_axis: tuple[bool, ...] | None = None,
) -> torch.Tensor:           # (N, K)
    """Differentiable Duchon m-spline basis evaluation in d-dim.
    Backward routes through engine VJP w.r.t. points and centers."""

def bspline(
    t: torch.Tensor,         # (N,)
    knots: torch.Tensor,     # (K,)
    *,
    degree: int = 3,
    periodic: bool = False,
) -> torch.Tensor:           # (N, K)
    """1D B-spline basis. Tensor-product is built explicitly when needed
    via tensor_product()."""

def matern(
    points: torch.Tensor,    # (N, d)
    centers: torch.Tensor,   # (K, d)
    *,
    nu: float = 1.5,
    length_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:           # (N, K)
    """Matérn kernel basis."""

def sphere_harmonic(
    lat: torch.Tensor,
    lon: torch.Tensor,
    *,
    max_degree: int = 8,
) -> torch.Tensor:
    """Spherical-harmonic basis on the unit sphere."""

def tensor_product(*bases: torch.Tensor) -> torch.Tensor:
    """Tensor product of marginal bases — for axes with DIFFERENT UNITS
    (the te() case in mgcv). For axes with same units, use a multi-d
    radial basis like duchon() directly."""
```

```python
# gamfit/torch/penalty.py

def duchon_function_norm(
    centers: torch.Tensor,   # (K, d)
    *,
    m: int = 2,
    periodic_per_axis: tuple[bool, ...] | None = None,
) -> torch.Tensor:           # (K, K) SPD

def duchon_operator_triplet(
    centers: torch.Tensor,   # (K, d)
    *,
    m: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mass, tension, stiffness) — useful for adaptive smoothing
    with multiple λ pieces."""

def bspline_difference(
    n_knots: int,
    *,
    order: int = 2,
) -> torch.Tensor:           # (n_knots, n_knots)

def block_diagonal(penalties: list[torch.Tensor]) -> torch.Tensor:
    """Compose a block-diagonal penalty from per-smooth pieces.
    Used when multiple smooths share a joint REML fit."""
```

```python
# gamfit/torch/reml.py

@dataclass
class RemlOutput:
    coefficients: torch.Tensor   # (M, D) — multi-output supported
    fitted: torch.Tensor         # (N, D)
    lam: torch.Tensor            # scalar (or per-smooth if additive)
    reml_score: torch.Tensor     # scalar
    edf: torch.Tensor            # scalar (or per-smooth)

def gaussian_fit(
    design: torch.Tensor,       # (N, M)
    response: torch.Tensor,     # (N, D)
    penalty: torch.Tensor,      # (M, M)
    *,
    by: torch.Tensor | None = None,    # (N,) row-multiplier
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
    smoothing: str = "reml",    # "reml" | "fixed" | "adam"
    log_lambda: torch.Tensor | None = None,
) -> RemlOutput:
    """Gaussian REML for a single smooth (or joint smooth with
    block-diagonal penalty)."""

@dataclass
class AdditiveRemlOutput:
    coefficients: list[torch.Tensor]    # per-smooth (M_i, D)
    fitted: torch.Tensor                # (N, D)
    lams: torch.Tensor                  # (F,) per-smooth λ
    reml_score: torch.Tensor            # scalar
    edf: torch.Tensor                   # (F,) per-smooth

def gaussian_fit_additive(
    smooths: list[Smooth],              # F smooths
    response: torch.Tensor,             # (N, D)
    *,
    weights: torch.Tensor | None = None,
    smoothing: str = "reml",            # "reml" (per-smooth λ via REML)
                                        # "fixed" (provided λ)
                                        # "adam" (λ as trainable)
    init_log_lambdas: torch.Tensor | None = None,
) -> AdditiveRemlOutput:
    """Joint multi-smooth additive Gaussian REML.
    Per-smooth λ jointly selected (REML mode) or provided (fixed/adam).
    Equivalent to formula-API `y ~ s(x1) + s(x2) + ...` but
    autograd-aware and bypassing the formula parser."""


@dataclass
class Smooth:
    design: torch.Tensor                # (N, M_i)
    penalty: torch.Tensor               # (M_i, M_i)
    by: torch.Tensor | None = None      # (N,) per-row gating
    name: str | None = None             # for diagnostics
```

```python
# gamfit/torch/independent.py  (renamed from "batched" — clearer name)

def gaussian_fit_independent(
    designs: list[torch.Tensor],        # K independent problems
    responses: list[torch.Tensor],
    penalty: torch.Tensor,              # shared penalty
    *,
    bys: list[torch.Tensor] | None = None,
) -> list[RemlOutput]:
    """K independent univariate REML fits.
    This is what 'batched' meant historically. Renamed for clarity:
    'batched additive' (which is what people often want) is gaussian_fit_additive."""
```

### Layer 5 — nn.Module

```python
# gamfit/torch/module.py

class GAMLayer(nn.Module):
    """A multi-smooth additive GAM as a nn.Module — for embedding inside
    neural-net architectures.

    Smooths defined declaratively. Bases evaluated each forward, joint
    REML fit each batch (training) or coefficients frozen and feedforward
    (inference, after .freeze()).
    """

    def __init__(
        self,
        smooths: list[SmoothSpec],
        out_features: int,
        *,
        smoothing: str = "reml",
        sparsity_top_k: int | None = None,   # TopK gating across smooths
    ): ...

    def forward(
        self,
        positions: torch.Tensor | dict[str, torch.Tensor],
        amplitudes: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def freeze(self, reference_batch: torch.Tensor) -> None:
        """Snapshot β at the current REML fit; switch to feedforward inference."""


@dataclass
class SmoothSpec:
    name: str
    kind: str                # "duchon" | "bspline" | "matern" | "sphere" | "tensor"
    n_centers: int
    in_features: int = 1
    m: int = 2
    periodic_per_axis: tuple[bool, ...] | None = None
```

This is the layer that the SAE community + cross-coder community + steering-vector community would actually USE. It eliminates the need for them to re-invent the wheel.

## What should be DELETED

User said "no users, delete legacy". Drop:

```
gamfit/_api.py
  - _duchon_function_norm_penalty (1D-only, leading underscore "internal")
  - _duchon_operator_penalties (1D-only)

gamfit/torch/_reml.py
  - gaussian_reml_fit_positions               → use basis.duchon() + reml.gaussian_fit()
  - gaussian_reml_fit_positions_batched       → use reml.gaussian_fit_additive() with bys for the "many independent atoms" pattern
  - The basis_kind="duchon_multipenalty" string-dispatch — make it explicit via penalty.duchon_operator_triplet() composition

gamfit/torch/_basis.py
  - duchon_basis_1d                           → basis.duchon() with (N, 1) shape
  - duchon_basis_1d_derivative                → basis.duchon_derivative()
  - gaussian_weighted_ridge                   → make internal (not part of public API)
  - gaussian_weighted_ridge_batch             → make internal

crates/gam-pyffi/src/lib.rs
  - duchon_basis_1d, duchon_basis_1d_derivative (replace with duchon, duchon_derivative)
  - The PyReadonlyArray1 signatures (replace with PyReadonlyArray2)

manifold_sae/
  - sae_2d.py (hand-rolled tensor-product, wrong math) — already deleted
  - The basis_kind="duchon_multipenalty" config knob (move to penalty composition)
  - Any direct use of _duchon_function_norm_penalty, _duchon_operator_penalties
```

## What gamfit ALREADY has (Layer 1, don't touch)

```
src/terms/basis.rs::
  build_duchon_basis(data: ArrayView2, spec: &DuchonBasisSpec)  -- multi-d ✓
  build_duchon_operator_penalty_matrices(centers: ArrayView2)   -- multi-d ✓
  build_duchon_function_norm_penalty(centers: ArrayView2, ...)  -- multi-d ✓
  build_thin_plate_penalty_matrix(centers: ArrayView2)          -- multi-d ✓
  build_bspline_basis_1d, build_matern_basis, ...
  build_sphere_basis, build_spherical_spline_basis

src/solver/pirls.rs, src/solver/outer_strategy.rs::
  P-IRLS, REML / LAML outer-loop, multi-smoothing-parameter optimization
  Already used by formula-API for joint multi-smooth fits.
```

## Migration plan

### Stage 1 — Python-only (no Rust rebuild)
- New top-level modules: `gamfit/torch/basis.py`, `gamfit/torch/penalty.py`,
  `gamfit/torch/reml.py`, `gamfit/torch/module.py`.
- `duchon()`, `bspline()`, etc. for d=1 compose existing 1D Rust bindings.
- `gaussian_fit_additive()` for single-λ implemented via block-diagonal
  penalty over existing `gaussian_reml_fit`.
- For d > 1: raise NotImplementedError with clear pointer to Stage 2.

This stage ships TODAY. ~600 lines of Python.

### Stage 2 — Rust binding updates
Add multi-d versions of PyO3 functions in `crates/gam-pyffi/src/lib.rs`:

```rust
#[pyfunction(signature = (points, centers, m = 2, periodic_per_axis = None))]
fn duchon_basis(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    centers: PyReadonlyArray2<f64>,
    m: usize,
    periodic_per_axis: Option<Vec<bool>>,
) -> PyResult<Py<PyArray2<f64>>> { ... }

#[pyfunction(signature = (centers, m = 2, periodic_per_axis = None))]
fn duchon_function_norm_penalty(
    py: Python<'_>,
    centers: PyReadonlyArray2<f64>,
    m: usize,
    periodic_per_axis: Option<Vec<bool>>,
) -> PyResult<Py<PyArray2<f64>>> { ... }

#[pyfunction(signature = (smooths_spec, response, smoothing = "reml", weights = None,
                          init_lambdas = None))]
fn gaussian_reml_fit_additive_programmatic(
    py: Python<'_>,
    smooths_spec: Vec<&PyDict>,         // list of {design, penalty, by, name}
    response: PyReadonlyArray2<f64>,
    smoothing: &str,
    weights: Option<PyReadonlyArray1<f64>>,
    init_lambdas: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyDict>> { ... }
```

The third is the most important — exposes the multi-smooth REML algorithm
that `fit_from_formula` already uses internally.

### Stage 3 — torch nn.Module + autograd-aware multi-d bases
Build `gamfit.torch.module.GAMLayer` on top of Stage 2. Ship the
embed-in-neural-net workflow.

### Stage 4 — full deprecation
Remove the 1D-only legacy entries. Document the migration. Bump major
version.

## Testing strategy

- **Existing tests** for formula-API must continue passing — Layer 3
  unchanged.
- **New tests** in `tests/torch/`:
  - `test_basis_duchon.py` — gradcheck at d=1, 2, 3
  - `test_penalty_duchon.py` — SPD check, eigenvalue spectrum sanity
  - `test_reml_additive.py` — single-λ joint fit equivalent to single
    smooth with block-diag penalty; per-λ joint fit recovers known
    structure on synthetic data
  - `test_module_gamlayer.py` — round-trip a known function, autograd
    through positions

## What this enables (concrete)

Past papers / libraries that would have used this if it existed:

- **Manifold-SAE** (this repo) — sum of smooth atoms in residual stream
- **Bilinear autoencoders** (Costa et al. 2025) — bases discovered jointly
- **Sparse crosscoders** (Lindsey et al. 2024) — smooth coupling across layers
- **InContextModel kernels** (Wurgaft et al. 2026) — smooth manifolds fit
  through centroids; would replace cubic-spline post-hoc with first-class
  GAM fit during training
- **Steering-vector regularization** — penalty-controlled smoothness
  along learned steering directions

## TL;DR

Layer 1 (Rust) is fine. Layer 4 (torch) is what needs the most work. Add
`gamfit.torch.basis.duchon`, `gamfit.torch.reml.gaussian_fit_additive`,
and `gamfit.torch.module.GAMLayer`. Delete the 1D-only legacy. Anyone
building a sum-of-smooth-atoms architecture (us; future authors) gets
the right primitives.
