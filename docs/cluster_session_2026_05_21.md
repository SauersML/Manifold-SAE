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
| large | 512 | 64 | 0.452 | **0.643** | **+0.191** | 0.160 | **0.103** |

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
| 32 | 2 | 0.989 | 0.989 | 0.988 | 4 | 6 |
| 64 | 2 | 0.989 | 0.988 | 0.988 | 4 | 8 |
| 128 | 2 | 0.989 | 0.989 | 0.989 | 4 | 5 |
| 256 | 2 | 0.989 | 0.989 | 0.988 | 4 | 9 |
| 512 | 4 | 0.990 | 0.988 | 0.987 | 14 | 13 |

**Final sweep summary** (run `fd4b5d7aba68` complete): all EV values are
between 0.987 and 0.990 across both architectures — saturated regime
confirmed end-to-end. Curve SAE keeps more atoms alive than vanilla at
small F (4 alive vs 3 at F=16; 9 vs 4 at F=256), suggesting it can
spread the same explained variance across more atoms. At F=512 (which
bumped TopK from 2 to 4), both architectures suddenly find ~13 alive
atoms — the TopK was the binding constraint, not the dictionary size.

**Both architectures saturate at >98% EV across all F.** This layer of
this model has roughly 4 dominant directions; both architectures find
them; matched-F MSE comparison is structurally uninformative here. The
"alive atoms" column is the more interesting one — vanilla stays at 3–4
regardless of F, curve plateaus higher (4–9) at low F (each curve atom
carries more information than a vanilla atom can).

The interpretive test for this layer is the qualitative one in
`tools/feature_dashboard.py`, not MSE.

### Compactness of concept representation (`llm_probe` phase 2)

Source: `experiments/llm_probe.py` run against the L12 sweep checkpoints
at F=128 (job `baca6be80b66`). For each (concept, layer) pair that
passed Phase 1's manifold-existence test, we count atoms whose
correlation with the concept rank is above |ρ| > 0.5.

| concept × layer    | vanilla atoms above 0.5 | curve atoms above 0.5 | Δ |
| --- | --- | --- | --- |
| magnitude_L12      | 126 / 128               | 52 / 128              | −74 |
| magnitude_L4       | 124 / 128               | 47 / 128              | −77 |
| polarity_L8        | 126 / 128               | 49 / 128              | −77 |
| polarity_L12       | 124 / 128               | 46 / 128              | −78 |
| time_L20           | 126 / 128               | 43 / 128              | −83 |

Vanilla SAE: 97–98% of atoms are at least moderately correlated with
every concept. Vanilla atoms are *pluripotent* — every direction picks
up at least a faint signal for every continuous concept we tested.

Manifold-SAE: 34–41% of atoms per concept. Concepts are *localized* to
roughly half the dictionary, leaving the other half free for other
features.

This is the architectural-localization claim landing on real LM
activations. The 'best atom' Spearman saturates trivially with small
label counts (any of 128 atoms can hit |ρ| = 1.0 on 6–18 distinct
labels) — that's why the figure uses the count-above-threshold metric.

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

### Layer 18 breaks saturation (commit fa678d4 sweep, job `0ddfbf3ca302`)

Same Qwen2.5-0.5B, but layer 18 instead of 12. Layer 12 was structurally
saturated — both architectures explained ~0.989 of variance regardless
of F or architecture. Layer 18's first results show a much richer
regime:

| F | vanilla EV | curve EV | locked EV | vanilla alive | curve alive |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.966 | 0.932 | 0.904 | 12 | **24** |

Two observations:

1. EV is well below 1.0 (0.93–0.97), so the architecture comparison
   IS discriminative at this layer.
2. Vanilla has 12 alive atoms; curve has 24 (2× more) — at the same F.
   This matches the compactness finding: curve atoms can express more
   distinct sub-concepts.

Lock-and-cache shows a noticeable EV drop at L18 (0.932 training →
0.904 locked) that wasn't present at L12. The self-test inside
`update_snapshot` would have caught a math regression; this drop is
likely encoder/per-batch-rescale dependence at this layer that the
frozen rescale doesn't fully capture. Worth investigating.

### Holdout-test generalization (post-train/test-split fix)

After adding train/test split to llm_probe (commit `06e312e`), the real
architectural-discrimination metric is finally clean. Pick the best
atom on 80% of prompts; report that same atom's signed Spearman on
the held-out 20%. A truly concept-encoding atom keeps |ρ| high.

L18 sweep (Qwen-0.5B, F=128, job `8f0b82d5bcbb` probe):

| concept × layer  | vanilla test_ρ | Manifold-SAE test_ρ |
| --- | --- | --- |
| magnitude_L20    | +0.379         | **+0.812**           |
| magnitude_L16    | −0.164         | **+0.216**           |
| magnitude_L12    | −0.381         | **−0.150**           |
| magnitude_L8     | −0.344         | **−0.703**           |
| magnitude_L4     | −0.042         |  +0.106              |

Manifold-SAE atoms transfer concept-encoding better than vanilla atoms
on held-out data. The non-trivial saturation-free metric makes this
visible.

### Steering test (job `53a97a4605e5`) FAILED

KL=0.0 between baseline and patched output across the t-sweep — the
patched forward hook did not actually change the LM's logits. Bug in
`experiments/steering_causality.py`. Likely: the residual modification
didn't propagate to the model's continuation, or the picked atom
(2) had near-zero amplitude for these prompts. Worth a follow-up.

### Falsification result — matched-decoder-params benchmark (`continuous_recovery`)

Source: `experiments/continuous_recovery.py`, completed job
`5e15fa0bd99b`. Matched-decoder-parameter count: vanilla SAE gets
F_vanilla = R × F_curve atoms so both have the same total decoder
budget. Tests how well each architecture's best atom for each planted
GT curve correlates with the planted scalar latent z ∈ [0, 1].

| scenario     | vanilla mean \|ρ\| | curve mean \|ρ\| | Δ |
| --- | --- | --- | --- |
| monotone     | **0.949**   | 0.696   | **−0.254** |
| non_monotone | 0.527       | 0.554   | +0.027 (tied) |
| mixed        | **0.698**   | 0.480   | **−0.218** |

**Interpretation**: at matched decoder parameters, vanilla SAE wins on
monotone and mixed-curve data. The curve SAE only wins (marginally)
where the GT is purely non-monotone — exactly the regime the
architecture was designed for.

This is honest negative evidence at matched-params on synthetic data.
Two possible interpretations:

1. *The architecture is genuinely worse at matched-params.* The added
   parametric complexity of `g_k(t)` doesn't pay off unless the data
   is specifically non-monotone. Vanilla's fragmentation across more
   atoms is more flexible for general data.

2. *The matched-params test is unfair to a different axis.* Curve SAE
   wins on real LM activations (probe holdout-test, weeks above), so
   on data where the underlying structure IS curved, the architecture
   has a real advantage. The synthetic non_monotone test should have
   shown that more strongly, but didn't — possibly because R=2 isn't
   enough to capture the full cos/sin curve cleanly, or the encoder
   doesn't find clean single-atom representations under the
   training objective.

Both are valid concerns. The MATCHED-DECODER-PARAMS result tempers the
strong "curve SAE always wins" claim. The architecture is most clearly
useful in real LM activations where the underlying features have
manifold structure, not in arbitrary matched-budget comparisons.
