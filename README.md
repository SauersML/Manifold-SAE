# Manifold-SAE

Sparse autoencoders whose features are **smooth 1D curves** in the residual stream, instead of single directions.

This is research code. Expect rough edges.

## What this tries to fix

Standard SAEs assume the **Linear Representation Hypothesis (LRH)**: each interpretable concept is a direction, and the decoder is a single matrix `W_dec` whose columns are those directions. That works, but it is a lossy story:

- Many features that look "monosemantic" under LRH are actually **1D manifolds** — e.g. magnitude / position / quantity / time-of-day. They live on smooth curves, not single rays.
- LRH-based SAEs paper over this by splitting one curve into many near-redundant point-features, inflating the dictionary and fragmenting interpretability.
- Recent work (Wurgaft et al. 2026, *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior*) shows that interpretable structure in residual-stream activations is often genuinely manifold-valued, and steering along the manifold transfers cleanly to behavior.

**Manifold-SAE** keeps the SAE training loop but replaces the linear decoder with a **vector-output Generalized Additive Model (GAM)**. Each active feature `k` carries a scalar parameter `t_k` (its position on the manifold), and the decoder reconstructs the residual stream as a sum of smooth vector-valued functions `f_k(t_k)`. Smoothness is selected automatically by **REML** (restricted maximum likelihood), not hand-tuned.

We wrap the Gaussian REML primitive from the [`gamfit`](https://pypi.org/project/gamfit/) package — gamfit owns the numerics (penalized least squares, REML score, derivative-tracked smoothing-parameter optimization); Manifold-SAE owns the SAE-shaped wrapping (vector outputs, joint Adam training with an encoder, sparsity).

## Decoder, concretely

For each feature `k`:

- A **1D Duchon (thin-plate-style) basis** on `[0, 1]`. Uniform centers, order 2. One basis type for every feature, cyclic or not. Cyclic concepts are fit as **approximately-closed open curves** — the v1 spec explicitly accepts a small seam at the endpoints, since gamfit's periodic-Duchon path lacks end-to-end REML support.
- A **penalty matrix** `S_k` driving smoothness.
- A **smoothing parameter** `λ_k`, selected by REML — gamfit does this for us.
- A **coefficient matrix** `B_k ∈ R^{d_basis × d_model}` solved fresh per batch via the inner ridge — there are no learned decoder weights; only the encoder weights and the per-feature smoothing parameters are trained.

The full reconstruction is `x̂ = Σ_k a_k · B_k^T φ(t_k)` where `a_k` is the encoder's amplitude gate (acts as gamfit's `by` weighting) and `t_k` is the encoder's manifold coordinate for feature `k`.

## v1 scope

What's in:

- **1D smooths only.** Each feature is a curve, not a surface.
- **Duchon basis everywhere.** Order-2 1D Duchon on `[0, 1]`, identity penalty matrix, REML-selected `λ_k`. Cyclic concepts are fit as open curves with a seam at the endpoints — gamfit's periodic Duchon path lacks REML support, and the spec accepts the seam. Whether the *steering operator* wraps modulo 1 is a caller-level choice (declare `cyclic=True` per task), independent of the basis.
- **Single layer.** One residual-stream site, one SAE.
- **Joint Adam training.** Encoder weights and per-feature `log λ` go through one Adam; decoder coefficients are solved analytically per batch.
- **Synthetic-first.** Validate recovery on toy manifolds before touching real activations.

What's deferred to v2+:

- **2D feature manifolds** (tensor-product / additive bivariate smooths).
- **Manifold-CLT** style uncertainty quantification on `t_k`.
- **Topology discovery** — currently the user declares cyclic vs. non-cyclic per feature; learning this is future work.
- Multi-layer / cross-layer SAEs.

## Experiment program

1. **Synthetic recovery.** Plant known 1D manifolds (sinusoids, ramps, sawtooths) in a fake residual stream; verify Manifold-SAE recovers them with fewer features than a vanilla SAE and that REML picks sensible `λ_k`.
2. **LLM activations.** Train on a mid-layer residual stream of a small open model (Pythia / Gemma-2-2B class). Compare reconstruction MSE and feature count vs. a TopK / JumpReLU baseline at matched sparsity.
3. **Steering benchmark.** Evaluate on the AxBench steering / concept-editing suite (Wu et al. 2025, *AxBench: Benchmarking Representation Steering*) — does steering *along the curve* (varying `t_k`) outperform direction-only steering?

## Install

Uses [uv](https://docs.astral.sh/uv/). No conda.

```bash
uv venv
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e ".[llm]"     # transformers / datasets / accelerate / safetensors
uv pip install -e ".[dev]"     # pytest, ruff
```

## Running things

No CLI. Each experiment is a Python module with a frozen `Config` dataclass at the top; edit the dataclass instantiation (or import the module and call programmatically) to override defaults:

```python
from experiments.synthetic_recovery import main, Config
main(Config(d_ambient=64, n_features=5, n_steps=2000, output_dir="runs/syn"))
```

Same pattern for `experiments.llm_activations` (`HarvestConfig`, `TrainConfig`, `AnalyzeConfig`) and `experiments.steering_eval` (`Config`).

## Layout

```
manifold_sae/      # library code
  gamfit_glue.py   # BasisSpec, ManifoldFit, manifold_fit — thin wrapper over gamfit's Gaussian REML
  encoder.py       # ManifoldEncoder — produces (a_k, t_k) per feature
  decoder.py       # decode() + extract_feature_curves() for post-hoc topology probes
  sae.py           # ManifoldSAE, ManifoldSAEConfig — full module, joint training step
  losses.py        # MSE + L1 sparsity + REML + position-spread entropy
  train.py         # Adam training loop
  diagnostics.py   # position-collapse / dead-feature / grad-ratio probes
  steering.py      # manifold-position / linear / diff-of-means steering operators
  data_synthetic.py        # planted-curve synthetic dataset
  data_activations.py      # LLM activation harvest + dataset wrapper
experiments/       # runnable experiment drivers (synthetic recovery, LLM activations, steering)
scripts/           # phase 0 gamfit substrate check + notes
tests/             # pytest suite — gradcheck, smoke, synthetic recovery
```

## License

AGPL-3.0-or-later, matching `gamfit`.

## References

- Wurgaft, N. et al. (2026). *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior.*
- Wu, Z. et al. (2025). *AxBench: Benchmarking Representation Steering.*
