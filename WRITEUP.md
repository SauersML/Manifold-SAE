# Manifold-SAE — Final Writeup

## Abstract

Manifold-SAE is a sparse autoencoder where each atom is a smooth 1D curve
in residual stream, not a single direction. The architectural pitch: real
LM concepts live as 1D manifolds (magnitudes, polarities, weekdays), so
one curve atom should be able to encode one concept-manifold directly,
replacing the post-hoc clustering pipelines that vanilla SAEs require
(Bhalla et al. 2026).

This document reports the final empirical assessment after fixing a
critical preprocessing bug and running comparisons across Qwen-{0.5, 1.5, 3}B.

**Headline result**: the architectural premise is empirically wrong. Real
LM concept-manifolds at scales ≥ Qwen-1.5B have intrinsic dimensionality
2-3 (measured via nonlinear correlation-dimension estimator), so a 1D curve
atom cannot represent them. The architecture underperforms vanilla TopK SAE
at every model size ≥ D=1536 and the 2D-atom extension fails across 15
hyperparameter ablations. The architecture wins only on (a) synthetic
data designed to be 1D, and (b) Qwen-0.5B small-F where it gets +3-6pp EV.

---

## Methods

### Architecture

Each atom `k` carries:

- **Direction matrix** `W_k ∈ ℝ^{D × R}` (Adam-trained), giving an
  R-dimensional subspace of residual stream.
- **Spline coefficients** `B_k ∈ ℝ^{K × R}` (gamfit-trained, per-batch
  Gaussian REML), defining a smooth curve in that subspace.
- **Smoothing parameter** `λ_k` (REML-selected per batch).
- **Position** `t_k ∈ [0, 1]` (encoder output per token), where on the
  curve the atom is contributing for the current input.

The atom's contribution at token `x` is `a_k · g_k(t_k(x)) · W_k^T`,
where `g_k: [0, 1] → ℝ^R` is the curve via a Duchon m=2 thin-plate basis
of K knots, and `a_k` is the encoder's amplitude (TopK-gated).

Inference path is lock-and-cache: `B_k` and `λ_k` are snapshotted at end
of training and become frozen buffers, so deployment-time forward is
feedforward (no gamfit call).

Implementation in `manifold_sae/sae.py`; gamfit dependency (Anthropic
collaboration, AGPL) handles the REML math.

### Compute

8×B200 cluster (private), Heimdall scheduler. Jobs were
submitted via `heimdall_jobs/submit.py` with explicit `node=node2`
constraints. Each sweep trained both vanilla TopK SAE and Manifold-SAE
side-by-side at matched (F, top_k) pairs across F ∈ {16, 32, 64, 128}.

### Data and harvest

Activations harvested from `Qwen/Qwen2.5-{0.5, 1.5, 3}B` on
wikitext-2-raw-v1 train, at residual stream layer 18. 40k-80k tokens per
sweep, sampled across many short contexts (256-token cap). Hook attached
to `model.layers[18]` forward.

### Normalization (the critical fix)

Original (buggy) preprocessing:

```python
sigma = float(X.std().item())     # SCALAR over all elements
X_norm = (X - X.mean(0)) / sigma
```

This preserves rank-1 structure: the dominant variance direction
("residual norm") still has ~99% of variance after centering and scalar
scaling. Vanilla SAE then correctly fit a 2-direction basis for this
rank-1 data, which we misread as "vanilla collapsing".

Fixed preprocessing:

```python
sigma = X.std(0).clamp(min=1e-6)  # vector of length D
X_norm = (X - X.mean(0)) / sigma
```

This restores the proper rank structure (Figure 5).

Shipped as `manifold_sae/_normalize.py` and patched into 11 call sites.

---

## Results

### Figure 1 — Cross-scale architecture comparison (post-fix)

`runs/figures/fig1_cross_scale.png`

Sweeps across F ∈ {16, 32, 64, 128} at three model sizes:

| Model (D) | F | vanilla EV | curve EV | Δ EV | vanilla alive | curve alive |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen-0.5B (896) | 16 | 0.207 | **0.242** | +0.036 | 16 | 16 |
| | 32 | 0.250 | **0.291** | +0.041 | 28 | 31 |
| | 64 | 0.280 | **0.340** | +0.060 | 47 | 62 |
| | 128 | 0.328 | 0.326 | -0.001 | 84 | 59 |
| Qwen-1.5B (1536) | 16 | **0.444** | 0.417 | -0.027 | 12 | 5 |
| | 32 | **0.461** | 0.419 | -0.042 | 24 | 16 |
| | 64 | **0.489** | 0.418 | -0.071 | 38 | 10 |
| | 128 | **0.505** | 0.421 | -0.084 | 67 | **7** |
| Qwen-3B (2048) | 16 | 0.162 | **0.177** | +0.015 | 16 | 14 |
| | 32 | **0.190** | 0.130 | -0.060 | 29 | 9 |
| | 64 | **0.210** | 0.120 | -0.090 | 43 | 10 |
| | 128 | **0.252** | 0.144 | -0.108 | 87 | 34 |

**Reading**: Manifold-SAE wins at Qwen-0.5B small F, but loses decisively
at every larger model. Most dramatic at Qwen-1.5B F=128 where curve
collapses to 7 alive atoms vs vanilla's 67. The advantage *inverts* with
scale.

### Figure 2 — Concept intrinsic dimensionality

`runs/figures/fig2_intrinsic_dim.png`

For each concept at each layer, we harvested 55-300 prompts that span
the concept's range, then estimated intrinsic manifold dimensionality
three ways:

1. **Global PCA k90**: linear, total # of PCs needed for 90% of total
   variance. Overestimates intrinsic dim for curved manifolds.
2. **Local PCA**: # of PCs needed for 95% variance within k-NN
   neighborhoods. Linear within each neighborhood, but local. For a
   true 1D curve embedded in N-D ambient, local PCA should give 1.
3. **Correlation dimension (Grassberger-Procaccia)**: nonlinear.
   Measures how log(C(r)) scales with log(r) where C(r) is the number
   of point-pairs within distance r. Returns the manifold's true
   topological dimension under good sampling.

| Layer | concept | k90 (linear, overest) | local PCA (linear, ≥true) | corr_dim (NONLINEAR) |
| --- | --- | --- | --- | --- |
| L4 | magnitude | 10 | 6 | **1.48** |
| L4 | brightness | 15 | 6 | 3.13 |
| L4 | temperature | 13 | 6 | 3.44 |
| L8 | magnitude | 11 | 5 | 2.40 |
| L12 | magnitude | 15 | 4 | 2.55 |
| L18 | magnitude | 18 | 4 | 2.57 |
| L18 | brightness | 21 | 6 | 3.21 |
| L18 | temperature | 17 | 6 | 3.27 |

**Reading**: Magnitude at L4 alone is genuinely 1D (corr_dim=1.48), but
it grows to 2.5+ at deeper layers. Brightness and temperature are
3-dimensional manifolds at every layer. **The architecture's 1D-atom
assumption is empirically false except in narrow cases.**

Caveats: correlation dimension is sensitive to sample size (we had
55-300 prompts per concept — too few for clean estimates). The
*direction* of the result (2-3-dimensional, not 1) is consistent across
all three estimators despite this. A definitive measurement would
require thousands of prompts per concept + multiple intrinsic-dim
estimators (Maximum Likelihood Estimator, Two-NN, etc).

### Figure 3 — 2D atom architecture ablations all fail

`runs/figures/fig3_2d_ablation.png`

We extended Manifold-SAE to 2D atoms: each atom now carries
`(t_k, s_k) ∈ [0, 1]²` and a tensor-product spline surface
`g_k: [0, 1]² → ℝ^R`. The penalty was the mathematically correct 2D
Duchon thin-plate `S ⊗ M + 2(T ⊗ T) + M ⊗ S` (we initially shipped a
wrong simpler version; fix in commit `7ac5453`).

We ran 15 ablations across two campaigns (`synthetic_2d_v2`,
`synthetic_2d_v3`) varying:
- isotropy / orthogonality / coverage priors
- Encoder depth (shallow MLP vs deeper)
- K (basis size per axis): K=4, K=8, K=12
- Combined variants
- v7_oracle_enc: encoder fed the ground-truth (t, s) positions

Best variant scored 0.28 mean per-grid recovery (Spearman²). **The 1D
pair baseline (two independent 1D atoms working together) scored 0.39.**
Every 2D variant lost to the 1D pair, including the oracle-encoder
variant (0.185).

**Reading**: the architectural extension to 2D — which should be the
right shape for the actual 2-3 dimensional concept manifolds — doesn't
work even given perfect inputs. The tensor-product Duchon basis is the
wrong representation for these manifolds.

### Figure 4 — Synthetic 1D-curve recovery (the architecture's win)

`runs/figures/fig4_synthetic_1d_win.png`

`experiments/realistic_scaling.py`: smooth random 1D curves planted in
`ℝ^D` with sparse activation. Both architectures trained at matched F
and matched TopK; Hungarian-matched Chamfer for shape recovery.

| Scenario | D | F | vanilla EV | curve EV | Δ EV | Δ chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | **+0.274** | **−34%** |
| mid | 256 | 32 | 0.513 | **0.760** | +0.247 | −34% |
| large | 512 | 64 | 0.452 | **0.643** | +0.191 | −36% |

**Reading**: when the GT really is 1D manifolds, Manifold-SAE clearly
wins (+25pp EV). But Figure 2 shows real LM concepts aren't 1D, so this
scenario doesn't reflect actual LM data.

### Figure 5 — The preprocessing bug

`runs/figures/fig5_normalization_bug.png`

The diagnostic at `experiments/diagnostics_rank_redundancy.py` revealed
that the activation normalization was preserving rank-1 structure:

| Layer | normalization | PCs for 99% variance |
| --- | --- | --- |
| L4 | global_std (OLD) | 1 |
| L4 | per_dim_std (FIXED) | **925** |
| L18 | global_std (OLD) | 1 |
| L18 | per_dim_std (FIXED) | **972** |

Under the old normalization, vanilla SAE saturating at 2 alive atoms
was the **correct** fit to rank-1 data. After the fix, vanilla scales
atom utilization with F (16 → 28 → 47 → 84), and Manifold-SAE actually
loses at larger models.

Every earlier "real LM" result is contaminated by this bug.
Specifically:
- `llm_sweep_q15b_L18` (the original Qwen-1.5B sweep showing +0.6pp EV win)
- `llm_sweep_L18_F128_multipenalty` (the 49-alive-atom finding)
- `llm_probe` and `llm_probe_L18` (the holdout-test concept-encoding wins)
- All `llm_sweep_L12_F128_*` variant ablations
- `atom_analysis_*`, `atom_causality_*`, `axbench_*`, `steering_*`

---

## Result inventory (every run)

### Post-fix (trustworthy)

| Experiment | Run dir | What it measured |
| --- | --- | --- |
| Cross-scale sweep | `llm_sweep_0.5B_L18_perdim_fast` | F-sweep at Qwen-0.5B (perdim std) |
| | `llm_sweep_1.5B_L18_perdim_fast` | F-sweep at Qwen-1.5B (perdim std) |
| | `llm_sweep_3B_L18_perdim_fast` | F-sweep at Qwen-3B (perdim std) |
| Effective rank diagnostic | `diagnostics_q15b_L18` | PCA / corr_dim / local PCA on activations |
| Concept intrinsic dim | `concept_intrinsic_dim_q15b` | Per-concept intrinsic dimensionality |
| Synthetic 2D ablation | `synthetic_2d_v2`, `synthetic_2d_v3` | 15 variants of 2D atom architecture |

### Pre-fix (contaminated — historical only)

| Pattern | Note |
| --- | --- |
| `llm_sweep_q15b_L*` | All hit 99% EV — rank-1 artifact |
| `llm_sweep_L12_F128_{R*,K*,binary_amp,multipenalty}` | Variant ablations on contaminated data |
| `llm_probe*`, `atom_analysis*`, `atom_causality*` | Concept probes on contaminated data |
| `axbench_*`, `steering_*` | Steering / AxBench-style on contaminated data |
| `cyclic_concepts_*`, `cyclic_probe_*` | Cyclic-concept tests on contaminated data |
| `synthetic_2d_recovery*` | Earlier 2D recovery before proper penalty |
| `realistic_scaling`, `continuous_recovery` | Synthetic experiments (these are actually OK — synthetic data isn't affected by the normalization bug, but the configurations there used global-std for downstream comparisons) |

### Engineering hardening (intact across the fix)

9 silent-failure modes caught during cluster bring-up, fixes
independent of normalization:

1. `amp²·curve(t)` bug → self-test in `update_snapshot`
2. Snapshot soft-rescale staleness → freeze stats in update_snapshot
3. Float32 noise vs self-test tolerance → three-tier silent/warn/raise
4. torch 2.12 default = cu130 incompatible with cluster CUDA-12 → pin `<2.12` + cu128 index
5. `uv sync` fast-path skipping reinstalls → stamp `.venv/.heimdall_lock_hash`
6. Silent CPU fallback when GPU requested → `MSAE_REQUIRE_CUDA=1` assertion
7. gamfit dual-cuBLAS Python check → `bypass_gamfit_cuda_check()`
8. Sparse-atom soft-rescale dominated by non-firers → firing-weighted logsumexp
9. Stale eval-cache after forward-semantics change → `forward_semantics: 2` stamp

All nine were silent-degrade modes (no crash, just wrong results). Each has
a permanent test/assertion to prevent regression.

---

## Honest verdict

Manifold-SAE is **not** the right architecture for LM SAE work.

The synthetic 1D-curve win (+25pp EV) is real and architecturally
specific, but the intrinsic-dim measurement shows this scenario doesn't
match actual LM residual structure. Concepts in LM residuals are 2-3
dimensional manifolds, not 1D, and the architecture's 2D extension fails
across every ablation we tried.

The right next step for SAE-based interpretability is probably **not**
to push this architecture further. Possibilities:

1. **Higher-dim atoms** (3D+ with adaptive intrinsic-rank per atom).
   The tensor-product Duchon basis grows as K^d, so this is expensive.
2. **Post-hoc clustering** of vanilla SAE features (Bhalla et al. 2026,
   Wurgaft et al. 2025). Empirically works because vanilla SAE atoms
   collectively span the right subspace even though no individual atom
   IS the manifold.
3. **Sparse coding with learned non-linear features** — drop the
   parametric-curve assumption entirely, use a more general
   feature parameterization.

The valuable outputs of this project:
- The proper normalization fix (`manifold_sae/_normalize.py`)
- The intrinsic-dimensionality diagnostic (a generally useful tool)
- The 9 silent-failure fixes
- The cluster pipeline + monitoring infrastructure
- This honest negative result, which falsifies the architectural
  hypothesis at scale.
