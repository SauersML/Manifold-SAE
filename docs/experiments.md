# Experiments

Each driver in `experiments/` is a self-contained Python module with a `Config` dataclass at the top. Edit the dataclass, or import and call `main(Config(...))` programmatically.

## Synthetic recovery — `experiments/synthetic_recovery.py`

Plant five known curves (line, parabola, ramp_exp, logmap, sqrt) in `ℝ^64` via random orthogonal projection. Train Manifold-SAE on the mix. Hungarian-match learned to planted curves; report per-feature Chamfer (centered + Frobenius-normalized, modulo orientation via Procrustes alignment in the plot).

This is the smallest reproducible test of "did the architecture recover the planted curves." Parabola is the load-bearing case — non-monotone, cannot be represented by a single vanilla SAE direction × scalar.

Run:
```python
from experiments.synthetic_recovery import Config, main
main(Config(d_ambient=64, n_features=5, n_steps=8000, output_dir="runs/toy"))
```

## Pure-curve benchmark — `experiments/pure_curve_benchmark.py`

Manifold-SAE vs vanilla TopK SAE head-to-head on synthetic data where every GT feature is a smooth random curve. Tests:

- **Matched dictionary size** (both SAEs get F = #GT features): does the curve SAE explain more variance than vanilla?
- **Overcomplete vanilla** (vanilla gets F = 10× #GT): how many vanilla atoms does it take to match Manifold-SAE's reconstruction with one atom per family?

Reports per-feature Hungarian-Chamfer between learned curves and planted curves.

## Realistic scaling — `experiments/realistic_scaling.py`

Same head-to-head, three scenarios:

| name | D | GT curves | Anchors/curve | Active/token |
| --- | --- | --- | --- | --- |
| small | 128 | 16 | 32 | 3 |
| mid | 256 | 32 | 64 | 5 |
| large | 512 | 64 | 64 | 8 |

Latest committed results (commit `def0cdb` / `realistic_scaling.py`):

```
small : vanilla expl 0.494, curve expl 0.668 (Δ +0.174)
mid   : vanilla expl 0.513, curve expl 0.672 (Δ +0.159)
large : vanilla expl 0.456, curve expl 0.664 (Δ +0.207)
```

Curve SAE wins reconstruction at all three scales; vanilla chamfer is fundamentally bounded above zero because a vanilla atom is a constant point in residual space while the GT signal is a curve cloud.

## LM activations — `experiments/llm_real.py`

End-to-end pipeline on real residual-stream activations. Defaults:

- `model_name`: `Qwen/Qwen2.5-0.5B` (Apache 2.0, pure causal text LM, 24 layers, hidden=896)
- `layer`: 12 (mid)
- `text_dataset`: `wikitext/wikitext-2-raw-v1/train`
- `n_tokens`: 80,000 (cap on harvest)
- `n_features`: 2048
- `top_k`: 24
- `n_steps`: 3000 (vanilla) / 1500 (curve)
- `batch_size`: 1024 (vanilla) / 128 (curve — keep gamfit's densified design under ~300 MiB)

Activation harvest uses a forward hook on the target block (cheaper than `output_hidden_states=True` which materializes every layer). Resilient to model families: hook attaches to `model.h[i]` (gpt2), `model.layers[i]` (Llama/Qwen/Mistral), or `model.model.layers[i]` (newer HF conventions).

Outputs head-to-head vanilla vs curve at matched (F, top_k), plus a lock-and-cache verification that single-token inference produces the same reconstruction as the trained model.

### Warm-start

Activations and SAE checkpoints persist at `/content/runs/LLM_REAL/` (absolute path so it survives `rm -rf /content/Manifold-SAE` in the Colab cell).

- **Activation cache** is keyed on `(model_name, layer, dataset, subset, seq_len)`. Reused when cached_n ≥ requested_n; only changing model/layer/dataset or asking for more tokens triggers re-harvest.
- **SAE checkpoint** is keyed on *structural* fields only — fields that change weight shapes. Vanilla: `(model, layer, F, top_k)`. Curve: above + `(n_basis, intrinsic_rank)`. Changing `lr`, `batch_size`, or `n_steps` does **not** invalidate a checkpoint; training continues from the saved step.
- Subset match: legacy checkpoints with extra signature fields are still loadable. Incompatible optimizer state is gracefully reinitialized.

To force a clean run: `rm -rf /content/runs/LLM_REAL/`.

### Colab one-cell

`colab_run.txt` is the paste-into-one-cell wrapper:

```
%cd /content
!nvidia-smi | head -5
!rm -rf /content/Manifold-SAE
!git clone -q https://github.com/SauersML/Manifold-SAE.git /content/Manifold-SAE
!pip install -q -U gamfit "transformers>=4.50" datasets scipy
!python -c "import gamfit, transformers; print('gamfit', gamfit.__version__, '| transformers', transformers.__version__)"
!cd /content/Manifold-SAE && python -m experiments.llm_real
```

The script handles the gamfit dual-cuBLAS conflict transitionally (see `docs/known_issues.md`).

## LLM-like synthetic stress test — `experiments/llm_like_stress_test.py`

Mix of point atoms (vanilla-SAE-flavored: a direction with continuous activation) and curve atoms in high-dimensional ambient. Tests whether Manifold-SAE handles a realistic distribution of feature types — some genuinely 1D, some manifold-valued.

## Steering eval — `experiments/steering_eval.py`

Skeleton for evaluating steering along a learned curve (varying `t_k`) vs direction-only steering. AxBench-style task. Not yet wired to a benchmark.

## Linear baselines — `experiments/baselines_linear_sae.py`

Bare-bones TopK SAE without all the training machinery — useful as a sanity check that the comparison numbers in `realistic_scaling` aren't an artifact of the more elaborate vanilla SAE implementation.

## Standalone curve SAE on captured activations — `experiments/llm_curve_sae.py`

Older driver for training just the curve SAE on a corpus of pre-harvested activation tensors. Use this if you've already saved activations from another pipeline and just want to train the SAE on them.

## What's not yet wired

- 2D feature manifolds (tensor-product smooths)
- Periodic / cyclic features through lock-and-cache (gamfit has periodic-Duchon; we haven't tested the snapshot path with periodic curves)
- Manifold-CLT-style posterior intervals on `t_k`
- Multi-layer cross-layer SAEs
- Steering along curve on actual AxBench tasks
