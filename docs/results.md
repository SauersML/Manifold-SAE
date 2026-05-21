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

These three scenarios were the original scale validation for the architecture. They pass robustly under the corrected REML-based path B (commit `74c307a` onward).

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
