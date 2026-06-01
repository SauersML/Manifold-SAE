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

## Empirical results

### Synthetic curves, head-to-head with vanilla TopK SAE at matched dictionary size

`experiments/realistic_scaling.py`. Three scenarios, F=#GT curves, TopK matched:

| Scenario | D | GT curves | Anchors/curve | Active/token | Vanilla expl | Curve expl | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 32 | 3 | 0.494 | **0.768** | **+0.274** |
| mid   | 256 | 32 | 64 | 5 | 0.513 | **0.760** | **+0.247** |
| large | 512 | 64 | 64 | 8 | 0.452 | **0.643** | **+0.191** |

Curve SAE wins on reconstruction at matched F across all scales by
+19 to +27 percentage points. Hungarian-matched Chamfer (shape recovery)
is also 30-35% lower for the curve SAE on every scenario.

### Real LM activations — see `docs/findings.md`

The honest, post-fix assessment of Manifold-SAE on real LM residuals
lives in `docs/findings.md`. The short version: after fixing a
late-discovered per-dim normalization bug and measuring the intrinsic
dimensionality of real concept manifolds (correlation dim 2.4–3.4 for
most concept×layer pairs), the architecture's core 1D assumption does
*not* hold at scale. With proper normalization, Manifold-SAE wins only
at the smallest model (Qwen-0.5B, small F) and loses to vanilla TopK
SAE at every model with D ≥ 1536. See `docs/findings.md` for the full
post-fix EV / alive-atom tables, the intrinsic-dimension measurement,
and the 2D-atom ablation.

An earlier round of real-LM results (holdout concept-encoding transfer,
concept-localization counts, L18 atom-utilization, the `n_basis` sweep)
was measured on a contaminated/rank-1 preprocessing path and has been
retracted; do not cite those numbers.

### Connection to Goodfire's neural-geometry series

Bhalla et al. (2026) (*Can SAEs Capture Neural Geometry?*) identify
three regimes by which SAEs represent curved manifolds: *shattering*
(one feature per point), *dilution* (many overlapping features), and
*compact capture* (small set of shared features acts as a coordinate
system). Their pipeline reaches compact capture via post-hoc clustering
of standard SAE features.

**Manifold-SAE is an architecture that targets the compact-capture
regime directly.** Each curve atom IS a compact-capture unit: one
atom spans one 1D manifold via its `g_k(t)` curve, parameterized
natively by the encoder's `t_k`. On pure-synthetic 1D-manifold data the
architecture does land in compact-capture; whether that transfers to
real LM residuals is addressed (negatively, at scale) in
`docs/findings.md`.

## Repository layout

```
manifold_sae/                 Library code
  sae.py                      ManifoldSAE module + ManifoldSAEConfig + extract_feature_curves
                              Per-batch gamfit REML in forward; lock-and-cache via
                              update_snapshot(); inference_mode for feedforward decode.
  encoder.py                  Per-feature MLP encoder (toy scale)
  encoder_linear.py           Shared 2-layer MLP encoder (LLM scale; H=4·D fixed)
  losses.py                   MSE + sparsity + identification priors; REML score
                              available in output struct for diagnostics
  metrics.py                  Hungarian-matched per-feature Chamfer + diagnostics
  data_synthetic.py           Planted-curve synthetic dataset
  data_activations.py         LLM activation harvest helpers
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

`gamfit >= 0.1.141` is required (multi-dim Duchon + additive REML API, autograd-aware backward, auto-derived knots/penalty).

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

Deferred:

- 2D feature manifolds (tensor-product smooths)
- Periodic features (gamfit has periodic-Duchon; not yet plumbed in to lock-and-cache)
- Manifold-CLT-style uncertainty quantification on `t_k`
- Topology discovery (currently the user declares cyclic vs non-cyclic per feature)
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
