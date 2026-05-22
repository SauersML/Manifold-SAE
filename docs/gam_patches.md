# gam / gamfit patches needed for full multi-dim Duchon support

These patches expose the multi-dim Duchon machinery that already exists
in the Rust crate (`src/terms/basis.rs::build_duchon_basis`,
`build_duchon_operator_penalty_matrices`, etc.) but is hidden behind 1D-only
Python bindings.

Status:
- Stage 1 (Python-only, no Rust rebuild): shipped in this repo, see
  `manifold_sae/_gam.py`. Builds joint additive Gaussian REML on top
  of the existing `gaussian_reml_fit` (single-λ).
- Stage 2 (Rust rebuild required): patches below. Adds multi-dim
  Duchon bindings + multi-λ joint REML programmatic entry.

---

## Stage 2 patch — `crates/gam-pyffi/src/lib.rs`

### REPLACE — `duchon_basis_1d` → `duchon_basis`

```diff
-#[pyfunction(signature = (t, centers, m = 2, periodic = false))]
-fn duchon_basis_1d<'py>(
-    py: Python<'py>,
-    t: PyReadonlyArray1<'py, f64>,
-    centers: PyReadonlyArray1<'py, f64>,
-    m: usize,
-    periodic: bool,
-) -> PyResult<Py<PyArray2<f64>>> {
-    let basis = duchon_basis_1d_impl(t.as_array(), centers.as_array(), m, periodic)
-        .map_err(py_value_error)?;
-    Ok(basis.into_pyarray(py).unbind())
-}
+#[pyfunction(signature = (points, centers, m = 2, periodic_per_axis = None))]
+fn duchon_basis<'py>(
+    py: Python<'py>,
+    points: PyReadonlyArray2<'py, f64>,    // (N, d)
+    centers: PyReadonlyArray2<'py, f64>,   // (K, d)
+    m: usize,
+    periodic_per_axis: Option<Vec<bool>>,  // length d
+) -> PyResult<Py<PyArray2<f64>>> {
+    let pts = points.as_array();
+    let ctrs = centers.as_array();
+    if pts.ncols() != ctrs.ncols() {
+        return Err(py_value_error(format!(
+            "points has d={} but centers has d={}", pts.ncols(), ctrs.ncols()
+        )));
+    }
+    let spec = DuchonBasisSpec {
+        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
+        length_scale: None,
+        power: 0.0,
+        nullspace_order: duchon_nullspace_from_m(m),
+        identifiability: SpatialIdentifiability::None,
+        aniso_log_scales: None,
+        operator_penalties: Default::default(),
+        periodic: periodic_per_axis.as_ref().map(|p| p.iter().any(|&x| x)).unwrap_or(false),
+        period: None,
+    };
+    let res = build_duchon_basis(pts, &spec).map_err(py_value_error)?;
+    Ok(res.design.into_pyarray(py).unbind())
+}
```

### REPLACE — `duchon_function_norm_penalty`

```diff
-#[pyfunction(signature = (centers, m = 2, periodic = false, period = None))]
-fn duchon_function_norm_penalty<'py>(
-    py: Python<'py>,
-    centers: PyReadonlyArray1<'py, f64>,
-    ...
+#[pyfunction(signature = (centers, m = 2, periodic_per_axis = None))]
+fn duchon_function_norm_penalty<'py>(
+    py: Python<'py>,
+    centers: PyReadonlyArray2<'py, f64>,   // (K, d)
+    m: usize,
+    periodic_per_axis: Option<Vec<bool>>,
+) -> PyResult<Py<PyArray2<f64>>> {
+    let ctrs = centers.as_array();
+    let res = build_duchon_function_norm_penalty(ctrs, m, periodic_per_axis)
+        .map_err(py_value_error)?;
+    Ok(res.into_pyarray(py).unbind())
+}
```

(The Rust function `build_duchon_function_norm_penalty(centers: ArrayView2, ...)` already exists in `src/terms/basis.rs`; this is just a binding update.)

### REPLACE — `duchon_operator_penalties`

```diff
-#[pyfunction(signature = (centers, m = 2, periodic = false, period = None))]
-fn duchon_operator_penalties<'py>(
-    py: Python<'py>,
-    centers: PyReadonlyArray1<'py, f64>,
-    m: usize, periodic: bool, period: Option<f64>,
-) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)>
+#[pyfunction(signature = (centers, m = 2))]
+fn duchon_operator_penalties<'py>(
+    py: Python<'py>,
+    centers: PyReadonlyArray2<'py, f64>,   // (K, d)
+    m: usize,
+) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)>
```

(Rust function `build_duchon_operator_penalty_matrices(centers: ArrayView2, ...)` is already multi-dim.)

### DELETE

- `duchon_basis_1d` (replaced by `duchon_basis`).
- `duchon_basis_1d_derivative` (replaced by `duchon_basis_derivative` if/when needed; not used by anything in our code).
- The 1D-coercion code paths.

### ADD — new programmatic multi-smooth REML entry

```rust
#[pyfunction(signature = (designs, y, penalties, bys = None, weights = None,
                          init_lambdas = None, smoothing = "reml"))]
fn gaussian_reml_fit_additive<'py>(
    py: Python<'py>,
    designs: Vec<PyReadonlyArray2<'py, f64>>,    // F design matrices
    y: PyReadonlyArray2<'py, f64>,               // (N, D) shared response
    penalties: Vec<PyReadonlyArray2<'py, f64>>,  // F penalty matrices
    bys: Option<Vec<PyReadonlyArray1<'py, f64>>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambdas: Option<PyReadonlyArray1<'py, f64>>,
    smoothing: &str,                             // "reml" | "fixed" | "adam"
) -> PyResult<Py<PyDict>>
```

Backs onto the same `TermCollection`-based joint REML that `fit_from_formula` already uses internally. Just need a programmatic constructor that takes (design, penalty, by) tuples instead of formula-parsed terms.

---

## Stage 2 patch — `gamfit/_api.py`

```diff
-def _duchon_function_norm_penalty(centers, *, m=2, periodic=False, period=None):
-    centers_array = _numeric_vector(centers, "centers")
-    return rust_module().duchon_function_norm_penalty(centers_array, m, periodic, period)
+def duchon_function_norm_penalty(centers, *, m=2, periodic_per_axis=None):
+    """Duchon m-spline function-norm penalty for K control points in d-dim.
+
+    Parameters
+    ----------
+    centers : ndarray of shape (K, d)
+    m : int, default 2
+    periodic_per_axis : sequence of bool of length d, optional
+    """
+    centers_array = _numeric_matrix_2d(centers, "centers")
+    return rust_module().duchon_function_norm_penalty(
+        centers_array, m, periodic_per_axis
+    )
```

Same for `duchon_basis`, `duchon_operator_penalties`. Drop the `_numeric_vector` coercion.

---

## Stage 2 patch — `gamfit/torch/_basis.py`

Drop `duchon_basis_1d`, replace with:

```python
def duchon_basis(
    points: torch.Tensor,    # (N, d)
    centers: torch.Tensor,   # (K, d)
    *,
    m: int = 2,
    periodic_per_axis: tuple[bool, ...] | None = None,
) -> torch.Tensor:           # (N, K)
    """Differentiable Duchon m-spline basis evaluation."""
    return _DuchonBasisFn.apply(points, centers, m, periodic_per_axis)
```

Backward routes through the existing engine VJP (autograd through `points`, optionally `centers`).

---

## Stage 2 patch — `gamfit/torch/_reml.py`

Add `gaussian_reml_fit_additive` Function + wrapper. Drop
`gaussian_reml_fit_positions{,_batched}` (replaced by additive +
explicit position-to-design composition).

---

## Why these patches matter

Without them:
- 1D-only Python entry points force callers to do tensor-product hacks
  for 2D Duchon (wrong math).
- No way to fit a single multi-smooth additive GAM from torch with
  per-smooth λ — only N independent univariate fits OR one shared λ.
- SAE-style architectures that decompose a response as a sum of
  smooth atoms cannot use the right primitive; they fight the API.

After them:
- 2D / 3D / d-dim Duchon is a one-liner: `duchon_basis(points_Nxd, centers_Kxd)`.
- Joint additive REML is the natural API call.
- Anyone building a sum-of-smooths architecture (us; bilinear
  autoencoders; cross-coders; future architectures) gets the right
  primitive without reinventing tensor products.

Estimated work: ~400 lines Rust binding edits + ~300 lines Python.
Half a day to implement + test + republish gamfit.
