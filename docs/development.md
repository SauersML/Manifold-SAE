# Development

## Repository conventions

- **No CLI.** Each experiment is a `Config` dataclass + `main(cfg)` function. Edit the Config inline or import and call programmatically.
- **No `argparse` proliferation.** If you need to override settings, do it in Python.
- **Output goes to `runs/...`.** Either `runs/<experiment_name>/` (relative; in-repo) for synthetic experiments, or `/content/runs/LLM_REAL/` (absolute; outside repo) for the Colab-friendly LLM driver. Plots, results.json, and checkpoints live there.
- **uv for environment.** No conda.

## Running tests

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -x
```

`tests/test_sae_smoke.py` covers forward/backward + lock-and-cache on tiny dims. `tests/test_synthetic.py` runs an abbreviated planted-curve recovery and checks the chamfer threshold.

## Running experiments locally

```python
from experiments.synthetic_recovery import Config, main
main(Config(d_ambient=64, n_features=5, n_steps=8000, output_dir="runs/toy"))
```

For comparisons with vanilla SAE:

```python
from experiments.realistic_scaling import main
main()  # runs small + mid + large scenarios; writes runs/REALISTIC/summary.json
```

For real LM activations:

```python
from experiments.llm_real import Config, main
main(Config(
    model_name="Qwen/Qwen2.5-0.5B",
    layer=12,
    n_tokens=80_000,
    n_features=2048,
    top_k=24,
    output_dir="/content/runs/LLM_REAL",  # outside repo for Colab survival
))
```

## Running on Colab

Open `colab_run.txt`, paste contents into one Colab cell with a T4 runtime selected, run.

The cell:
1. clones the repo fresh from GitHub `main`
2. installs `gamfit>=0.1.141` (standing rule: always newest gamfit — the local
   venv is on 0.1.145, and the joint manifold-recovery objective needs the
   upcoming release beyond it), `transformers>=4.50`, `datasets`, `scipy`
3. prints gamfit + transformers versions for debugging
4. runs `python -m experiments.llm_real`

Iteration workflow: edit code locally, push to `main`, re-run the cell. Activations and SAE checkpoints persist at `/content/runs/LLM_REAL/` across `rm -rf /content/Manifold-SAE` cycles. Warm-start signatures are structural-only, so changing `lr`, `batch_size`, or `n_steps` continues training instead of restarting.

To force a clean run anytime: `!rm -rf /content/runs/LLM_REAL/` before the cell.

## Architecture changes

The most consequential migration in this repo's history was commit `74c307a` ("Use gamfit fully: per-batch REML training, lock-and-cache for inference"), which moved the curve coefficients `B_k` and smoothing `λ_k` from `nn.Parameter` (Adam-trained, with a hand-rolled quadratic smoothness penalty) to gamfit-output (per-batch REML, autograd-aware backward). The hand-rolled penalty wasn't actually REML — λ collapsed to ~0 because the log-determinant term that gives REML its Occam mechanism was missing.

If you're considering reverting that decision: see `docs/architecture.md#why-not-a-persistent-b-as-nnparameter` and `docs/known_issues.md#toy-5-5-visual-recovery-not-yet-reproduced-under-path-b`. The principled architecture is more constrained on hyperparameters but supports the methodological claim that "smoothness is selected by REML."

## Adding a new experiment

1. Create `experiments/<name>.py` with a `Config` dataclass and `main(cfg)` function.
2. Default `output_dir` to `runs/<NAME>` (relative).
3. If you need real LM activations, depend on `experiments.llm_real.harvest_activations` (use the forward-hook pattern, not `output_hidden_states=True`).
4. Add a smoke test if non-trivial: small dims, short training, assert basic invariants.

## Adding a loss term

Loss components live in `manifold_sae/losses.py`. Each is a pure function of `ManifoldSAEOutput` and (optionally) the config; `total_loss` aggregates them with weights from the config. Identification priors (gauge / parameterization) belong here; smoothness-related terms should not — gamfit owns smoothness.

## Adding a basis or smoothness alternative

Currently fixed to Duchon m=2 on `[0, 1]`. If you want a different basis:

1. Verify gamfit supports it via `gamfit.torch.gaussian_reml_fit_positions_batched(basis_kind="...")`.
2. Plumb a `basis_kind` field through `ManifoldSAEConfig`.
3. Update the `inference_mode` path in `ManifoldSAE.forward` to call the matching `gamfit.torch.*_basis_1d` evaluator with locked coefficients.

## Why not put REML score directly in the loss?

It's tempting to write `loss = mse + (-fit.reml_score.sum())` so the REML log-likelihood enters explicitly. Don't. With `mask_binary` as the `by` weighting, REML score on a zero-amplitude feature is unbounded above (no residual, no penalty). Adding `-reml_score` to the loss creates a degenerate maximum at all-zero amplitudes — the model collapses.

gamfit is *already* running REML internally to select λ given the encoder's positions. The MSE on the SAE reconstruction is sufficient signal — Adam backprops through gamfit's analytic backward into the encoder + W, learning positions that admit a high-likelihood fit. REML score is exposed in the output struct for diagnostics; don't add it to the training loss.

## Common pitfalls

- **dtype mismatch at the gamfit boundary.** gamfit needs float64. The current SAE casts inputs to f64 before the call and casts outputs back to whatever `x.dtype` is. If you add new tensors that flow into the gamfit call, cast them.
- **Encoder hidden dim quadratic in F.** Don't override `hidden_dim` to anything that scales with F. The default `4·D` is what you want.
- **Identification priors disabled.** All four (sparsity, ortho, coverage, monotonicity) are doing real work. Disabling them does not "let REML handle it"; REML doesn't address gauge or parameterization.
- **Activation cache invalidated unexpectedly.** The cache signature is `(model_name, layer, dataset, subset, seq_len)`. If you change tokenizer settings or the seq_len, you'll re-harvest.

## License

AGPL-3.0-or-later. Matches gamfit's license.
