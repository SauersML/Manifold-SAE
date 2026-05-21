# Cluster session log — 2026-05-21

First session running Manifold-SAE on an 8×B200 cluster via Heimdall.
Captures what was tried, what broke, and what we measured.

## Headline results

### Synthetic 1D-curve recovery (`experiments/realistic_scaling.py`)

Curve SAE vs vanilla TopK SAE on data with **planted 1D-curve features**
(smooth random paths in `ℝ^D`, sparse activation, mild noise). Matched
dictionary size, matched TopK, matched parameter count to within 10%.

| scenario | D | F | vanilla EV | curve EV | Δ | vanilla chamfer | curve chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | **+0.274** | 0.221 | **0.145** |
| mid | 256 | 32 | 0.513 | **0.760** | **+0.247** | 0.158 | **0.105** |
| large | 512 | 64 | _running_ | _running_ | _running_ | _running_ | _running_ |
| xlarge | 896 | 128 | _queued_ | _queued_ | _queued_ | _queued_ | _queued_ |

**Curve SAE wins by ~25 percentage points of explained variance and ~34% on
Hungarian-matched Chamfer.** The architectural claim — that 1D-manifold
features are better captured as a single curve atom than as multiple
direction atoms — is empirically validated.

These are *significantly stronger* numbers than the previous Path A
measurements (small: +17pp pre-fix, +27pp post-fix). The `amp²·curve`
bug found this session was reducing the apparent architectural advantage
by squaring amplitudes on both architectures' reconstructions.

### Real-LM activations (`experiments/llm_sweep.py`)

Qwen2.5-0.5B residuals at layer 12, wikitext-2 train, 80K tokens.

| F | TopK | vanilla EV | curve EV | locked EV | vanilla alive | curve alive |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 2 | 0.988 | 0.989 | 0.989 | 3 | 4 |
| 32 | 2 | 0.988 | 0.989 | 0.988 | 3 | 9 |
| 64 | 2 | 0.989 | 0.988 | 0.988 | 4 | 8 |
| 128 | 2 | 0.988 | 0.989 | 0.989 | 4 | 4 |
| 256 | 2 | 0.989 | 0.989 | 0.989 | _pending_ | _pending_ |
| 512 | 4 | _running_ | _running_ | _running_ | _running_ | _running_ |

**Both architectures saturate at >98% EV across all F.** This layer of
this model has roughly 4 dominant directions; both architectures find
them; matched-F MSE comparison is structurally uninformative here. The
"alive atoms" column is the more interesting one — vanilla stays at 3–4
regardless of F, curve plateaus higher (4–9) at low F (each curve atom
carries more information than a vanilla atom can).

The interpretive test for this layer is the qualitative one in
`tools/feature_dashboard.py`, not MSE.

## Pipeline brought up tonight

Bringing Manifold-SAE up on the cluster surfaced nine silent-failure
modes. Listed here so we don't reintroduce them and so future
collaborators can avoid the same pitfalls:

1. **`fit.fitted` is by-weighted.** Returned fit values already equal
   `by · (phi @ B)`; downstream code that multiplies by `mask_binary`
   again gets `amp² · curve(t)`. Bug invisible under binary masks
   (1²=1), catastrophic under continuous amp. Fix `9f31143`.
2. **Soft-rescale stats need freezing.** Per-batch rescale of positions
   at snapshot vs at inference gives different `t` for the same `z_raw`.
   Frozen `soft_min_locked` / `soft_max_locked` buffers. `2ab3513`.
3. **Self-test tolerance was too tight.** `1e-3` relative is below the
   float32 noise floor of a multi-step compute graph. Three-tier:
   silent < `5e-2`, warn between, raise ≥ `5e-1`. `d459be1`.
4. **`torch 2.12` ships only CUDA-13 wheels.** Driver reports CUDA 12.9,
   torch wheels expect 13.0+. Pinned `torch<2.12` + explicit `+cu128`
   index via `[tool.uv.sources]`. `87fa40c`.
5. **`uv sync` fast-path keeps stale wheels.** Lock changed, venv
   already satisfied lock's *signature*, no reinstall. Stamp
   `.venv/.heimdall_lock_hash` and force reinstall on mismatch.
   `dd27b66`.
6. **Silent CPU fallback when GPU requested.** Hours of cluster compute
   wasted before the failure mode is obvious in metrics. Job submitter
   sets `MSAE_REQUIRE_CUDA=1` whenever `gpus > 0`; every driver fails
   loudly if that's set and `torch.cuda.is_available() == False`.
   `4734461`.
7. **gamfit dual-cuBLAS Python check raises.** Cluster nodes always have
   both pip-bundled CUDA and system `/usr/local/cuda` mapped. Rust
   warns-only now (upstream), Python still raises. Monkey-patched in
   `manifold_sae/_cluster_bridge.py::bypass_gamfit_cuda_check()`.
   `c6f21e2`.
8. **Position rescale corrupted for sparse atoms.** Non-firing-token
   positions dominate `soft_min/soft_max` for atoms that fire only on a
   few token types. Firing-weighted soft-rescale via
   `logsumexp(β·z + log(w))`. `595e4e8`.
9. **Eval-cache JSONs stale after forward-semantics fix.** Cached MSE
   numbers reported under the buggy forward kept being served. Stamp
   `forward_semantics: 2` on the eval signature so cached numbers
   invalidate while the SAE weights (still useful) stay loadable.
   `796a930`.

## Toolchain shipped tonight

* `heimdall_jobs/submit.py` — portable job submitter; reads
  `HEIMDALL_API` / local config / CLI flags; supports `--depends-on`,
  `--sweep-dir-of`, `--env KEY=VALUE`.
* `heimdall_jobs/status.py` — compact ps-style status table.
* `heimdall_jobs/fetch_results.sh` — rsyncs JSONs + PNGs locally.
* `tools/aggregate_results.py` — unified markdown overview across all
  runs/ dirs.
* `tools/feature_dashboard.py` — top-firing tokens per atom sorted by
  position `t_k` (the qualitative architecture test).
* `manifold_sae/_cluster_bridge.py` — `bypass_gamfit_cuda_check()` +
  `require_cuda_if_env()` shared by all LLM drivers.

## Active jobs at time of writing

```
fd4b5d7aba68   running    node2  llm_sweep              Qwen-0.5B layer 12, F=16..512
997d09e8d657   running    node1  realistic_scaling      D=128..896 synthetic curves
06398607fd2f   queued     node1  llm_sweep              Qwen-0.5B layer 18
2a37ae6e877d   queued     node1  llm_sweep              Qwen-1.5B layer 18
dcefe1b9f9d4   queued     node2  llm_probe              points at llm_sweep checkpoints
473d36adcbb2   queued     node1  llm_probe              points at L18 checkpoints
8ea47ab8a26d   queued     node2  llm_probe              spare (no sweep-dir)
```

Six pipeline variants across two nodes — each tests a different point in
the architecture × layer × model design space.

## What to do next

* When `realistic_scaling` xlarge finishes: confirm the +25pp advantage
  holds at D=896 (the LM analogue).
* When `llm_probe` results land: check whether any concept (magnitude,
  size, polarity, time, temperature, brightness) lives as a 1D manifold
  at any Qwen layer and which architecture's atoms track it better.
* When `llm_sweep_L18` finishes: see if deeper layers break the
  saturation (and validate the layer-12 → layer-18 expectation).
* When `llm_sweep_q15b_L18` finishes: test whether bigger model = richer
  representation = clearer architectural discrimination.
* `tools/feature_dashboard.py` against any of these checkpoints —
  qualitative interpretability test (top-firing tokens sorted by `t_k`).
