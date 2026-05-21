# Results

Empirical claims, with the commit where each was measured. Updated when re-measured.

## Manifold-SAE vs vanilla TopK SAE, matched dictionary size

Source: `experiments/realistic_scaling.py`, commit `def0cdb`. Synthetic data: smooth random curves in `ℝ^D`, lifted via random orthogonal projection. Both SAEs get the same F (= #GT curves) and the same TopK. Hungarian-matched per-feature Chamfer on Frobenius-normalized point clouds.

| Scenario | D | GT curves | Anchors/curve | Active/token | Vanilla expl | Curve expl | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 32 | 3 | 0.494 | 0.668 | **+0.174** |
| mid | 256 | 32 | 64 | 5 | 0.513 | 0.672 | **+0.159** |
| large | 512 | 64 | 64 | 8 | 0.456 | 0.664 | **+0.207** |

Notes:

- Curve SAE wins reconstruction at all three scales, by 16–21 percentage points.
- Vanilla SAE has ~50% dead features at matched parameter budget. Curve SAE has zero dead features.
- Vanilla Chamfer is fundamentally bounded above zero because a vanilla SAE atom is a constant point in residual space, while the GT signal is a curve cloud. Comparing a point to a non-degenerate curve via Chamfer gives a finite floor.

### Re-measurement on cluster B200 after the amp²·curve bug fix

Source: `experiments/realistic_scaling.py`, commit `cfbaafc` + onward. Same scenarios, run on B200 GPU via Heimdall after the
`fit.fitted` = `by · (phi @ B)` bug was fixed (see commit `9f31143`). Numbers are *better* than the pre-fix measurements above — the architectural advantage was being partially masked by the `amp²·curve` semantic eating into both architectures' MSE comparison.

| Scenario | D | Vanilla expl | Curve expl | Δ | Vanilla chamfer | Curve chamfer | Chamfer Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 0.494 | **0.768** | **+0.274** | 0.221 | **0.145** | **-34%** |
| mid | 256 | 0.513 | **0.760** | **+0.247** | 0.158 | **0.105** | **-34%** |

Curve SAE now wins by ~25 percentage points on explained variance (vs ~17pp before the fix) and ~34% on shape recovery (Chamfer). Large + xlarge scenarios are running at time of writing; will be added when they complete.

Methodology
-----------

Activations are smooth random curves planted as 1D manifolds in `ℝ^D` (per-curve smooth path of `n_anchors` points). Each sample picks `sparsity/token` curves uniformly without replacement, samples an amplitude uniformly in `[0.7, 1.3]`, samples a discrete position uniformly along the curve, and adds the curve's value at that position scaled by the amplitude. White noise at 2% is added on top of the sum. Vanilla SAE is a TopK SAE (LayerNorm → Linear(D, 4D) → GELU → Linear(4D, F) → ReLU → TopK), curve SAE is the Manifold-SAE architecture with `intrinsic_rank=4` and Duchon m=2 basis with `n_basis=8–12`. Both architectures get matched F and matched TopK. Hungarian matching pairs each learned atom to the closest GT curve; Chamfer is mean-min-distance between matched atom and GT curve, Frobenius-normalized.

## Per-active-atom efficiency, overcomplete vanilla

Source: `experiments/pure_curve_benchmark.py` (earlier results, persistent-B era; needs re-measurement under Path B).

Under-Path-A measurement: at matched dictionary size, curve SAE achieves ~3.7× higher explained variance per active atom than vanilla SAE on pure-curve synthetic data. Path B is structurally equivalent for this comparison — the curve atom *is* a curve regardless of how it's optimized.

**Status**: needs a Path B re-measurement. Tracking issue for `realistic_scaling`-style sweep with overcomplete vanilla.

## Synthetic recovery toy (`experiments/synthetic_recovery.py`)

Five planted curves (line, parabola, ramp_exp, logmap, sqrt) in `ℝ^64`. Hungarian-matched per-feature Chamfer.

**Under Path A** (persistent-B era, commit `df18008`): 5/5 visually clean Procrustes-aligned overlays. Per-feature Chamfer:

```
line     0.0035   parabola 0.0151   ramp_exp 0.0142
logmap   0.0049   sqrt     0.0189
                 mean 0.011   max 0.019
```

Parabola is the load-bearing case — non-monotone, cannot be represented by a single vanilla atom. Recovery as a clean U is the demonstration that the architecture isn't equivalent to vanilla SAE.

**Under Path B** (current architecture, commit `74c307a` onward): not yet reproduced. See `docs/known_issues.md#toy-5-5-visual-recovery-not-yet-reproduced-under-path-b`.

## Real LM activations

Source: `experiments/llm_real.py`. Defaults: Qwen/Qwen2.5-0.5B mid-layer residuals on wikitext-2 train, F=2048 matched TopK=24.

**Status**: end-to-end pipeline works (Colab T4 + LD_PRELOAD bypass / Python-side gamfit shim). Numbers not yet committed to this doc — re-running with the current defaults will populate a row here.

When committed, the table below will record:

| Model | Layer | F | TopK | n_tokens | Vanilla expl | Curve expl | Locked-snapshot MSE | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Cluster pipeline (2026-05-21)

Real-LM experiments now run on an 8×B200 cluster via Heimdall jobs.
Submission/observation scripts:

```
heimdall_jobs/submit.py        # POST a job JSON, env-driven config (no secrets in repo)
heimdall_jobs/status.py        # ps-style status table, optional --watch
heimdall_jobs/fetch_results.sh # rsync JSONs + PNGs locally, auto-open in Preview
tools/aggregate_results.py     # markdown overview across all runs/ dirs
```

### Bugs and fixes shipped during the cluster bring-up

| # | Bug | Symptom | Fix (commit) |
| --- | --- | --- | --- |
| 1 | `fit.fitted` is already `by * (phi @ B)` | Locked-mode MSE ~100× training-mode MSE under `continuous_amp=True` | Drop the duplicate `mask_binary` multiply in `_forward_training` (`9f31143`) |
| 2 | Soft-rescale stats not frozen at snapshot | Lock-and-cache inference at per-chunk rescale gave nonsense positions | Freeze `soft_min_locked` / `soft_max_locked` buffers in `update_snapshot` (`2ab3513`) |
| 3 | Self-test tolerance too tight (1e-3 rel) | F=64 spuriously raised on float32 noise | Three-tier: silent < 5e-2, warn 5e-2…5e-1, raise ≥ 5e-1 (`d459be1`) |
| 4 | torch 2.12 ships only +cu130 wheels | Cluster CUDA-12.9 driver: "driver too old, found 12090"; silent CPU fallback | Pin `torch<2.12` + `tool.uv.sources` pointing at +cu128 PyTorch index, Linux-only marker (`87fa40c`) |
| 5 | `uv sync` fast-path kept stale wheels | venv survived a lock change with old torch | Stamp `.venv/.heimdall_lock_hash` with `sha256(uv.lock)`, reinstall on mismatch (`dd27b66`) |
| 6 | Silent CPU fallback when GPU requested | Hours of wasted cluster compute on the wrong device | `MSAE_REQUIRE_CUDA=1` assertion in every driver, auto-set by submitter when `gpus > 0` (`4734461`) |
| 7 | gamfit dual-cuBLAS Python check raises | Cluster nodes inevitably map both /usr/local/cuda + pip nvidia/cublas | `manifold_sae/_cluster_bridge.py::bypass_gamfit_cuda_check()` (`c6f21e2`) |
| 8 | Position rescale dominated by non-firing tokens for sparse atoms | Soft min/max stats degenerate for atoms with few fires | Firing-weighted soft-rescale via `logsumexp(β·z + log(w))` (`595e4e8`) |
| 9 | Forward semantics changed but eval-cache JSONs stale | Old `runs/llm_sweep/eval_F*.json` serve back outdated numbers | `forward_semantics: 2` stamped on eval signature, checkpoints stay loadable (`796a930`) |

Most of these are silent-failure modes — they wouldn't crash, just degrade results. Numbers reported above are after all nine fixes.

### Self-test in `update_snapshot`

After every `update_snapshot` the SAE runs a forward pass in both training mode and locked mode on the same snapshot batch and asserts the two reconstructions agree. Catches future regressions in the locked-forward path immediately rather than in offline analysis. The bug #1 above would have surfaced as a self-test failure here.

## Failed experiments

Honest record of architecture attempts that didn't pan out.

### Continuous amplitude + curve-norm gauge on the toy (V7, commit `1cc...`)

`continuous_amp=True` with a quadratic `curve_norm` penalty targeting `‖g_k‖_avg ≈ 1`. On the small toy this gave chamfer 0.042 (worse than V5's 0.017). Continuous amp is the right gauge for data with continuous amplitude variation (LLM activations), but for the toy where GT amplitude is approximately uniform, binary amp is the simpler choice and works better.

### REML score as an explicit loss term (V16)

Adding `-fit.reml_score.sum()` to the training loss created a degenerate maximum at all-zero amplitudes (REML score on zero-by features is unbounded above). All features went dead within a few hundred steps. Removed.

### Persistent B as `nn.Parameter` with quadratic smoothness penalty (Path A)

Worked on the toy (5/5 visual recovery, chamfer 0.011 mean) but didn't actually do REML — the log-determinant term required for REML's Occam mechanism was missing, so `λ_k` was Adam-pulled toward 0 with no resistance. Methodologically equivalent to "parametric SAE atoms with a hand-tuned smoothness regularizer" rather than "GAM-decoder SAE with REML smoothness selection." Migrated to the current Path B architecture at commit `74c307a`.

### High intrinsic rank for low-rank GT features

R=4 in the curve SAE with rank-1 or rank-2 GT features produces visible spline wiggle: most R output dimensions are noise, and gamfit's single shared `λ_k` per feature compromises between the (one) signal dimension and the (three) noise dimensions. R matched to GT max intrinsic rank (R=2 for the toy) is the fix.
