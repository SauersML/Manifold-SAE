# Manifold-SAE

Sparse autoencoders whose features are **smooth 1D curves** in residual stream, not single directions. Research code; expect rough edges.

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
| small | 128 | 16 | 32 | 3 | 0.494 | 0.668 | **+0.174** |
| mid | 256 | 32 | 64 | 5 | 0.513 | 0.672 | **+0.159** |
| large | 512 | 64 | 64 | 8 | 0.456 | 0.664 | **+0.207** |

Curve SAE wins on reconstruction at matched F across all scales. Zero dead features in the curve SAE; about half of vanilla's features are dead at matched parameter budget.

### Toy synthetic recovery (`experiments/synthetic_recovery.py`)

Five planted curves in `ℝ^64`: line, parabola, ramp_exp, logmap, sqrt. Parabola is the load-bearing case — non-monotone, so a single vanilla atom (one direction × scalar) cannot represent it; it would need two atoms minimum. Hungarian-matched per-feature Chamfer (centered + Frobenius-normalized, modulo orientation via Procrustes alignment):

Under the corrected REML-based architecture, results are sensitive to per-feature intrinsic rank vs GT rank (R=4 with mostly rank-1 GT noise dims causes spline wiggle; R matched to GT rank works). Retuning under the current architecture is the outstanding work — see *Known limitations* below.

### Real LM activations (`experiments/llm_real.py`)

End-to-end pipeline: harvest residual-stream activations from a HuggingFace causal LM on a text corpus, train vanilla TopK + Manifold-SAE side by side at matched dictionary size, lock-and-cache, evaluate. Default `Qwen/Qwen2.5-0.5B` mid layer; configurable.

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
  llm_activations.py          Older activation-pipeline scaffold
  baselines_linear_sae.py     Bare-bones TopK SAE for sanity-check baselines
  steering_eval.py            Skeleton for steering-along-curve evaluation

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

`gamfit >= 0.1.81` is required (autograd-aware `grad_penalty`, auto-derived knots/penalty).

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

### gamfit CUDA dual-stack conflict on Colab and several cloud images

gamfit 0.1.98 refuses to load its Rust extension if it detects two cuBLAS files mapped in the process (`/proc/self/maps`). On Colab and similar, both the system CUDA `/usr/local/cuda-12.8/...libcublas.so.12.8.4.1` and the torch-bundled `nvidia/cublas-cu12/...libcublas.so.12` are mapped via `dlopen`, triggering this check.

In practice this dual mapping is benign — `dlopen` resolves a SONAME to exactly one file, so all CUDA calls route through one handle even with both files in the address space. The "double-free" pathology only triggers if code crosses handles by absolute path.

**Upstream fix** is committed at `SauersML/gam@7efd17eb`: downgrade the assert to a once-per-process warning. Awaiting publication of a new gamfit wheel.

**Transitional bridge** in `experiments/llm_real.py` monkey-patches `cuda_diagnostics` to return an empty conflict set, neutralizing the check at the source. Removed once a new gamfit wheel lands.

A second, lower-level Rust check inside gamfit independently refuses CUDA dispatch when it sees the dual mapping. Until that's also addressed upstream, gamfit's inner REML solve runs on CPU even after the Python bridge — the PyTorch encoder / W / Adam path stays on GPU, but each batch incurs a CPU-GPU round-trip.

### Toy 5/5 visual recovery not yet reproduced under Path B

The toy `experiments/synthetic_recovery.py` is currently a hyperparameter-tuning problem under the corrected architecture (commit `74c307a` onward). Earlier versions hit 5/5 visually clean overlays via a hand-rolled smoothness penalty + persistent `B` as `nn.Parameter`; that architecture didn't actually do REML.

Diagnosis: gamfit selects one `λ_k` shared across all R output dimensions of a feature's subspace. If most of those dimensions are noise rather than GT signal (e.g. R=4 with rank-1 GT line), `λ_k` compromises and the noise dimensions wiggle. Fix candidates (not yet executed): match R to GT max rank (R=2 here), shrink basis K, larger training batches so REML has more data per feature per batch.

Scale benchmarks (`realistic_scaling`) already demonstrate the architectural advantage; the toy retune is a smaller cleanup.

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

- Wurgaft, N. et al. (2026). *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior.*
- Wu, Z. et al. (2025). *AxBench: Benchmarking Representation Steering.*
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.).
- Wahba, G. (1990). *Spline Models for Observational Data.*
