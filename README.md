# Manifold-SAE

Sparse autoencoders whose features are **smooth 1D curves** in
residual stream — and, optionally, **smooth 2D surfaces**
(`manifold_sae/sae_2d.py`). Research code; expect rough edges.

> "Native manifold discovery, not post-hoc clustering." Where
> Bhalla et al. (2026) recover curved-manifold structure by
> clustering features of a vanilla SAE, Manifold-SAE atoms IS
> the manifold by construction — one atom = one 1D curve (or 2D
> surface), parameterized by a coordinate `t_k` (or `(t_k, s_k)`)
> emitted directly by the encoder. See `docs/architecture_2d.md`
> for the 2D extension.

## What this tries to fix

Standard SAEs assume the Linear Representation Hypothesis: each interpretable concept is a direction, and the decoder is a matrix `W_dec` whose columns are those directions. That works, but it is a lossy story:

- Many features that look "monosemantic" under LRH are actually **1D manifolds** — magnitude, position, quantity, time-of-day, polarity, hue. They live on smooth curves, not single rays.
- LRH-based SAEs paper over this by splitting one curve into many near-redundant point-features, inflating the dictionary and fragmenting interpretability.
- Recent work (Wurgaft et al. 2026, *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior*) shows that interpretable structure in residual-stream activations is often genuinely manifold-valued, and steering along the manifold transfers cleanly to behavior.

**Manifold-SAE** keeps the SAE training loop but replaces the linear decoder with a vector-output Generalized Additive Model (GAM). Each active feature `k` carries a scalar parameter `t_k` (its position on the manifold), and the decoder reconstructs the residual stream as

```
x̂ = Σ_k a_k · g_k(t_k)
```

where `g_k: [0, 1] → ℝ^D` is a smooth 1D curve in residual stream and `a_k` is the encoder's amplitude. Smoothness per feature is selected automatically by **REML** (restricted maximum likelihood) — not a hand-tuned hyperparameter.

[`gamfit`](https://pypi.org/project/gamfit/) owns the GAM math (penalized least squares, REML score, derivative-tracked smoothing-parameter optimization, batched ridge solves). Manifold-SAE owns the SAE-shaped wrapping (encoder, sparsity, identification priors, per-feature ambient subspaces, snapshot for feedforward deployment).

## How it actually works

Each feature `k` carries three quantities:

| Quantity | Shape | Owner | Purpose |
| --- | --- | --- | --- |
| Ambient subspace `W_k` | `(D, R)` | Adam | Embeds the curve into residual stream |
| Spline coefficients `B_k` | `(K, R)` | gamfit (per batch) → snapshot at end | The curve shape in subspace |
| Smoothing `λ_k` | scalar | gamfit (REML per batch) → snapshot | Penalizes wiggliness |

`K` is the basis size (default 10). `R` is the per-feature intrinsic rank (default 2 — enough for any planar curve). The curve in ambient is `g_k(t) = (φ(t) B_k) W_kᵀ` where `φ(t) ∈ ℝ^K` is the Duchon m=2 basis on `[0, 1]`.

### Training step

```
encoder(x)                        → (positions, amplitudes)        # Adam-owned weights
y_proj = x @ W                    → (B, F, R)                      # Adam-owned W per feature
fit = gamfit.gaussian_reml_fit_positions_batched(
        positions, y_proj, by=amplitudes, basis_kind="duchon", ...)
                                  → (B_k, λ_k, fitted, reml_score)  # REML each batch, autograd-aware
recon = Σ_k amp_k · fit.fitted_k @ W_kᵀ
loss = MSE(recon, x) + sparsity + identification priors
loss.backward()                   # backprops through gamfit into (encoder, W)
```

Adam updates `encoder` and `W`. Gamfit returns `B_k` and `λ_k` each batch — these are *not* `nn.Parameter`s. The autograd path goes through gamfit's analytic backward, so the encoder learns positions that admit a smooth, well-fitting curve and `W` learns subspaces REML's likelihood rewards.

### Lock-and-cache for inference

At end of training:

```
sae.update_snapshot(reference_batch)   # one big REML fit; freeze (B, λ) as buffers
sae.inference_mode = True
```

After this the SAE is feedforward: encoder → basis-eval(positions) @ B_snapshot → ambient. No gamfit call at inference. Same shape and cost as a vanilla feedforward SAE.

### Identification priors

REML selects smoothness but doesn't speak to gauge or parameterization. Four light priors stay:

- `sparsity` — standard SAE sparsity on amplitudes.
- `subspace_ortho` — per-feature column ortho + cross-feature off-block diversity so `W_k ≠ W_j`.
- `position_coverage` — positions should span `[0, 1]` so the curve isn't free between firing points.
- `monotonicity` — soft prior that position tracks principal projection. Tiebreaker when monotone and U-shaped fits are equally good for the data; doesn't force monotone where data demands U (parabola etc.).

What's *not* in the loss: smoothness penalty (REML owns it), curve-norm gauge (REML's likelihood pins the amp-curve scale), cumulant penalty.

## Methodological claim

> Each Manifold-SAE atom is a smooth 1D curve in residual stream parameterized as the penalized maximum-likelihood estimate of a Gaussian GAM given the encoder's positions for the current batch. Smoothness `λ_k` is selected automatically by Restricted Maximum Likelihood (REML). At inference the coefficients are cached, giving a feedforward decoder identical in shape to a standard SAE.

This is what the code does. The earlier persistent-`B`-as-`nn.Parameter` version did *not* support this claim — there REML wasn't actually running, just a hand-rolled quadratic penalty — and was migrated back to the spec'd architecture (commit `74c307a`).

## Joint manifold-SAE recovery objective (gamfit `sae_manifold_fit`)

The design session converged on building joint multi-atom recovery directly on
gamfit's first-class joint solve, `gamfit.sae_manifold_fit`, rather than a
hand-rolled torch loop. There is no separate encoder net and no student: the
**canonical answer *is* the joint fit**, and the encode direction (activation →
coordinate) is the same solver run with the dictionary frozen
(`ManifoldSAE.encode`).

- **Canonical assignment = IBP.** The assignment prior is `assignment="ibp"` — an
  Indian-Buffet-Process prior giving an *adaptive* atom count and *true zeros* in
  the assignment — not `softmax` + `top_k` (a fixed-count soft relaxation). IBP is
  the gam default and the design-decided answer; every recovery fit uses it.
- **Cross-atom decoder incoherence (the separability lever, gamfit #671).** When
  two superposed atoms share a coherent plane, the per-token split between them is
  ill-posed. `decoder_incoherence_weight` pushes superposed atoms' decoder column
  spaces apart so the split is identifiable. This is the headline knob the
  verification harness gates on (ON vs OFF).
- **Nuclear-norm embedding-rank selection (#672).** The per-atom ambient embedding
  rank is selected by a nuclear-norm (trace-norm) penalty rather than a hand-set
  `R`, letting the fit choose how many ambient directions each atom needs.
- **ScadMcp non-convex sparsity.** Assignment/amplitude sparsity uses SCAD / MCP
  non-convex penalties (near-unbiased on large coefficients) instead of plain L1.
- **Isometry gauge + gauge-conditional topology evidence (#673).** The isometry
  gauge weight (`isometry_weight > 0`, verified load-bearing — at 0 the periodic
  parameterization collapses) pins each atom's coordinate to be a near-isometry of
  its manifold; #673 makes the topology model-evidence *conditional on that gauge*,
  so topology selection is comparable across candidates rather than confounded by
  gauge freedom.
- **Per-atom uncertainty + typical coordinate range.** The fit result now exposes,
  per atom, topology/manifold **uncertainty** (posterior shape bands, the curve
  mean ± sd) and a **typical coordinate range** (where the coordinate actually
  concentrates). These are read off the gamfit result, not estimated downstream.

### Verification harness

Two files in `experiments/` form the recovery verification gate. Both build
directly on `gamfit.sae_manifold_fit` with the canonical IBP assignment and
self-gate (report **BLOCKED**, never a false PASS) when a fit diverges or the
installed gamfit lacks a needed knob:

- **`experiments/manifold_recovery.py`** — the gate. Three checks: (1) K=2
  superposed-circle recovery under IBP (PASS if reconstruction R² > 0.9); (2) the
  headline **incoherence ON vs OFF** gate across a coherence sweep — ON must raise
  the recovered-tangent `σ_min` *and* lower the cross-atom decoder cross-Gram
  ‖B₀B₁ᵀ‖_F (or improve coordinate recovery); (3) a **single-atom out-of-class
  specification check** (K=1, runs today) that flags a 2D blob mis-fit as a circle
  via an absolute out-of-class margin calibrated against a true-circle null.
- **`experiments/manifold_falsifier.py`** — the keystone falsifier + shared scoring
  primitives. Plants two circles with two orthogonal knobs (coherence → colinear
  planes; coverage → co-active fraction) and scores per-token coordinate recovery
  up to each circle's isometry (`circ_procrustes_r2`), plus the **`σ_min`
  identifiability metric** (smallest singular value of the stacked active-atom
  tangent frame — 0 iff the split is underdetermined). `--selftest` proves the
  scoring is trustworthy (isometry-invariant, split-sensitive, σ_min decreasing as
  planes go colinear) *before* the fit unblocks.

Status: the multi-atom (K ≥ 2) joint solve currently diverges upstream (fix in
progress), so checks 1–2 self-gate BLOCKED and the single-atom check (3) runs
today; the harness is correct and goes green the moment the solver fix and the
incoherence knob land.

## Repository layout

```
manifold_sae/                 Library code
  sae.py                      ManifoldSAE module + ManifoldSAEConfig + extract_feature_curves
                              Per-batch gamfit REML in forward; lock-and-cache via
                              update_snapshot(); inference_mode for feedforward decode.
  encoder.py                  Per-feature MLP encoder (toy scale)
  losses.py                   MSE + sparsity + identification priors; REML score
                              available in output struct for diagnostics
  data_synthetic.py           Planted-curve synthetic dataset
  train.py                    Adam training loop with device handling
  diagnostics.py              Position-collapse / dead-feature / grad-ratio probes

experiments/                  Runnable experiment drivers (Config dataclass at top)
  synthetic_recovery.py       Planted-curve toy + Procrustes-aligned plot
  pure_curve_benchmark.py     Curve SAE vs vanilla SAE, pure-curve synthetic data
  realistic_scaling.py        Same comparison across D ∈ {128, 256, 512}
  llm_real.py                 LM activations + head-to-head + lock-and-cache
  llm_like_stress_test.py     Mixed point + curve atoms stress test
  llm_curve_sae.py            Standalone curve SAE on captured LM activations
  baselines_linear_sae.py     Bare-bones TopK SAE for sanity-check baselines

tests/                        pytest suite — gradcheck, smoke, synthetic recovery

colab_run.txt                 One-cell Colab paste: clone + install + run experiments/llm_real.py
```

## Install

Uses [uv](https://docs.astral.sh/uv/). No conda.

```bash
uv venv
uv pip install -e ".[llm]"     # transformers / datasets / accelerate / safetensors
uv pip install -e ".[dev]"     # pytest, ruff
```

**Always newest gamfit.** The standing rule for this repo is to track the newest
gamfit. The joint manifold-SAE objective and its verification harness (below) depend
on features that land in the *upcoming* gamfit beyond the 0.1.145 currently installed:
cross-atom decoder incoherence (`decoder_incoherence_weight`, gamfit #671),
nuclear-norm embedding-rank selection (#672), ScadMcp non-convex sparsity, the
gauge-conditional topology evidence on top of the isometry gauge (#673), and the new
per-atom topology/manifold **uncertainty** (posterior shape bands, mean ± sd) and
**typical coordinate range** exposed on the fit result. The verification harness
self-gates (reports BLOCKED, never green-washes) when the installed gamfit does not yet
expose a knob it needs, so it is correct to run against any version and goes green once
the upcoming release lands. The base GAM/REML primitives (multi-dim Duchon + additive
REML API, autograd-aware backward, auto-derived knots/penalty) have been available since
gamfit 0.1.141.

## Running things

### Local

Each experiment is a Python module with a `Config` dataclass at the top. Edit defaults inline, or import and call programmatically:

```python
from experiments.synthetic_recovery import Config, main
main(Config(d_ambient=64, n_features=5, n_steps=8000, output_dir="runs/syn"))
```

Same pattern for `pure_curve_benchmark`, `realistic_scaling`, `llm_real`.

### Colab (T4 GPU)

Paste `colab_run.txt` into one Colab cell. It clones, installs, and runs `experiments.llm_real` end-to-end. Warm-starts from `/content/runs/LLM_REAL/` across re-runs — activations and SAE weights are cached.

To force a fresh run: `!rm -rf /content/runs/LLM_REAL/` before the cell.

## Warm-start and checkpointing

`experiments/llm_real.py` caches both activations and per-SAE state to `/content/runs/LLM_REAL/`:

- **Activation cache** is keyed on `(model_name, layer, dataset, subset, seq_len)` and reused if the cached `n_tokens >= requested`. Asking for fewer tokens slices the cache; only changing model/layer/dataset or asking for more tokens triggers re-harvest.

- **SAE checkpoints** are keyed on *structural* fields only. Vanilla SAE signature: `(model_name, layer, n_features, top_k)`. Curve SAE signature: above + `(sae_n_basis, sae_intrinsic_rank)`. Changing `lr`, `batch_size`, `n_steps`, or the other architecture's hyperparameters does **not** invalidate a checkpoint — training simply continues from the saved step.

- Subset-match resume: legacy checkpoints with extra signature fields are still loadable. Incompatible optimizer state is gracefully reinitialized; weights still resume.

## Known limitations

### gamfit dual-cuBLAS bridge

Cluster nodes and many cloud images (Colab, hosted images) map both
the system CUDA `/usr/local/cuda-*` and torch's bundled
`nvidia/cublas-cu12` via `dlopen`. gamfit's safety check refuses to
load Rust on this dual mapping by default. The actual catastrophe
condition (cublas-destroy across libraries) cannot trigger because
cudarc's `culib()` is a process-wide `OnceLock<Library>` — all
symbols resolve through one handle regardless of how many files are
mapped.

The Rust-side fix (downgrade to warn-once) shipped upstream in gam main
(`SauersML/gam`, commits `ff0f5380` + `233672b6`) and is in the installed
gamfit 0.1.141.

**Workaround in this repo**:
`manifold_sae/_cluster_bridge.py::bypass_gamfit_cuda_check()`
monkey-patches the Python-side assert to a no-op. All LLM
experiment drivers call this at import time.

### gamfit REML stays on CPU at small K

gamfit's CUDA policy thresholds (`gemm ≥ 109.54 Mflop`,
`xtwx_rows ≥ 512` on B200) are measurement-calibrated for the
hardware. At our default basis K=10, intrinsic rank R=2, batch ≤ 8192,
per-fit FLOPs sit below the GPU-launch crossover, so REML runs on
faer + Rayon (CPU). This is *correct behavior* — CPU is genuinely
faster than launching a CUDA kernel for these shapes.

For workloads that would benefit from GPU REML (much larger K or
batched-many-feature solves), a batched X^TWX kernel would push the
crossover lower. Tracked upstream as a feature.

## v1 scope and roadmap

In scope:

- 1D smooths per feature (curves, not surfaces)
- Duchon m=2 basis on `[0, 1]`
- Single residual-stream layer, single SAE
- Per-batch gamfit REML in training; lock-and-cache for inference
- Synthetic-first: validate recovery on toy manifolds before real activations

Now landing via the joint `sae_manifold_fit` objective (see "Joint manifold-SAE
recovery objective" above):

- Topology discovery — gauge-conditional topology model-evidence (#673) selects
  per-atom topology rather than requiring the user to declare cyclic vs non-cyclic.
- Per-atom uncertainty — posterior shape bands (curve mean ± sd) and a typical
  coordinate range exposed directly on the fit result, replacing the deferred
  manifold-CLT-style `t_k` interval estimate.

Still deferred:

- 2D feature manifolds (tensor-product smooths)
- Periodic features through lock-and-cache (gamfit has periodic-Duchon; not yet
  plumbed in to `update_snapshot`)
- Multi-layer / cross-layer SAEs
- Steering-along-curve evaluation on AxBench-style benchmarks

## License

AGPL-3.0-or-later (matches gamfit).

## References

- Bhalla, U. et al. (2026). *Can SAEs Capture Neural Geometry?* —
  shattering / dilution / compact-capture taxonomy; Manifold-SAE
  targets compact-capture directly via the architecture.
- Wurgaft, N. et al. (2026). *Manifold Steering Reveals the Shared
  Geometry of Neural Network Representation and Behavior* —
  cubic-spline post-hoc fit of activation manifolds for steering.
  Manifold-SAE's `g_k(t)` is the same kind of object as their fitted
  spline, but emitted natively from the SAE rather than fitted
  post-hoc through centroids.
- Engels, J. et al. (2024). *Not All Language Model Features Are
  Linear* — cyclic representations in LM residuals (days of the
  week, months). Direct motivation for the curve-atom architecture.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction
  with R* (2nd ed.) — the GAM/REML math gamfit implements.
- Wahba, G. (1990). *Spline Models for Observational Data* — Duchon
  m=2 basis and the function-norm penalty.
