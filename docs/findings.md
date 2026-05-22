# Findings — final honest assessment

## TL;DR

After fixing a critical normalization bug, retraining on properly-distributed
data, and measuring the intrinsic dimensionality of real LM concept manifolds:

**Manifold-SAE's core architectural assumption — that LM concepts live as
1D smooth manifolds — is empirically false at the scales we care about.**
Real concept manifolds at Qwen-1.5B have correlation dimension 2.4-3.4.
A 1D curve atom cannot represent a 3D manifold, and the 2D atom extension
also failed across 7 hyperparameter variants.

Manifold-SAE underperforms vanilla TopK SAE at every model scale ≥ D=1536.

---

## The decisive measurement: concept intrinsic dimensionality

`experiments/concept_intrinsic_dim.py` measured the intrinsic dimensionality
of concept-firing token clouds in Qwen-1.5B residuals via three methods
(PCA, Grassberger-Procaccia correlation dimension, local PCA).

| Layer | concept | k90 (PCA) | corr_dim | local PCA |
| --- | --- | --- | --- | --- |
| L4 | magnitude | 10 | **1.48** | 6.0 |
| L4 | brightness | 15 | 3.13 | 6.0 |
| L4 | temperature | 13 | 3.44 | 6.0 |
| L8 | magnitude | 11 | 2.40 | 5.0 |
| L8 | brightness | 19 | 3.36 | 6.0 |
| L8 | temperature | 15 | 3.12 | 6.0 |
| L12 | magnitude | 15 | 2.55 | 4.0 |
| L18 | magnitude | 18 | **2.57** | 4.0 |
| L18 | brightness | 21 | **3.21** | 6.0 |
| L18 | temperature | 17 | **3.27** | 6.0 |

Only `magnitude` at L4 is close to 1D. Every other concept at every other
layer has intrinsic dim 2.4-3.4. **The architecture's `g_k : [0,1] → ℝ^D`
parameterization is mathematically the wrong shape.**

---

## Real-LM headline results (proper per-dim normalization)

| Model (D) | F | vanilla EV | curve EV | Δ EV | van alive | crv alive |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen-0.5B (896) | 16 | 0.207 | **0.242** | +0.036 | 16 | 16 |
| Qwen-0.5B | 32 | 0.250 | **0.291** | +0.041 | 28 | 31 |
| Qwen-0.5B | 64 | 0.280 | **0.340** | **+0.060** | 47 | **62** |
| Qwen-0.5B | 128 | **0.328** | 0.326 | -0.001 | **84** | 59 |
| **Qwen-1.5B (1536)** | 16 | **0.444** | 0.417 | -0.027 | 12 | 5 |
| Qwen-1.5B | 32 | **0.461** | 0.419 | -0.042 | 24 | 16 |
| Qwen-1.5B | 64 | **0.489** | 0.418 | **-0.071** | 38 | 10 |
| Qwen-1.5B | 128 | **0.505** | 0.421 | **-0.084** | **67** | **7** |
| **Qwen-3B (2048)** | 32 | **0.190** | 0.130 | -0.060 | 29 | 9 |
| Qwen-3B | 64 | **0.210** | 0.120 | **-0.090** | 43 | 10 |
| Qwen-3B | 128 | **0.252** | 0.144 | **-0.108** | **87** | 34 |

Manifold-SAE wins only at Qwen-0.5B small F. Vanilla wins decisively
at every larger model. Curve atoms COLLAPSE at the larger models (7 alive
at Qwen-1.5B F=128 vs vanilla's 67).

---

## 2D atom architecture also fails

`experiments/synthetic_2d_v3.py` ran 7 ablation variants of the 2D atom
on planted 2D-grid synthetic data:

| variant | mean per-grid recovery (Spearman²) | alive |
| --- | --- | --- |
| v0_baseline | 0.253 | 10 |
| v1_isotropy_safe | 0.222 | 8 |
| v2_ortho | 0.236 | 6 |
| v3_coverage | 0.175 | 16 |
| v4_deeper_enc | 0.205 | 7 |
| **v5_lower_K** | **0.279** | 10 |
| v6_combined | 0.146 | 16 |
| **1D pair (baseline)** | **~0.39** | 16 |

Best 2D variant (v5_lower_K) gets 0.279, still worse than 2 independent
1D atoms working in parallel (0.39).

The 2D atom architecture works as designed (`ratio_frac_2d` ≈ 0.7-1.0
means atoms use both axes), but cannot beat the simpler "two coordinated
1D atoms" approach. The architectural extension doesn't pay off.

---

## What's still defensible

**Synthetic 1D-curve recovery at matched F** still works:

| Scenario | D | F | vanilla EV | curve EV | Δ EV |
| --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | +0.274 |
| mid   | 256 | 32 | 0.513 | **0.760** | +0.247 |
| large | 512 | 64 | 0.452 | **0.643** | +0.191 |

When the GT is a mix of pure 1D manifolds, curve SAE clearly wins.
But the intrinsic-dim measurement shows this scenario isn't realistic
for actual LM residuals.

---

## What we now know was wrong

The previous (PRE-FIX) results that claimed Manifold-SAE wins on real LM:

* "Vanilla collapses to 2 alive atoms" — preprocessing artifact, vanilla
  correctly fit the rank-1 data we accidentally fed it.
* "Compactness: vanilla 124-126/128, curve 43-52/128" — both numbers
  measured on rank-1 contaminated data, no longer trustworthy.
* "Holdout-test concept-encoding: curve 0.81 vs vanilla 0.38" —
  measured on contaminated data, hasn't been reproduced post-fix.
* "Architectural advantage scales with model size" — exactly inverted.
  Architecture works at D=896, fails at D≥1536.

---

## Engineering hardening (intact across the fix)

The 9 silent-failure modes caught during cluster bring-up are real and
their fixes are independent of the normalization issue:

* `amp²·curve(t)` bug → self-test in `update_snapshot`
* torch ≥ 2.12 / CUDA-13 incompatibility → pin <2.12
* `MSAE_REQUIRE_CUDA=1` assertion against silent CPU fallback
* gamfit dual-cuBLAS Python check bypass
* firing-weighted soft-rescale for sparse atoms
* eval-cache forward-semantics stamp
* tensor-product penalty math fix
* scalar→per-dim std normalization (the big one — late discovery)
* `LogisticRegression(multi_class=...)` deprecated arg

---

## Honest verdict

**Manifold-SAE is not the right architecture for LM SAE work.**

The core assumption (1D smooth atoms = concept manifolds) is empirically
false. Real LM concepts have intrinsic dim 2-3+. Even when the
architecture is extended to 2D, it doesn't beat coordinated 1D atoms.

What the architecture is good at:
* Pure-synthetic 1D-curve recovery (toy benchmark, doesn't reflect real LM)
* The math + cluster pipeline + tooling (all of which is reusable)

What we learned:
* Real LM concepts need higher-dim representations than 1D curves.
* Vanilla TopK SAE remains the right default for this scale.
* The proper way to use curves in interpretability is probably post-hoc
  (Bhalla et al., Wurgaft et al.) — cluster vanilla features, then fit
  smooth manifolds through the clusters. Not as an atom architecture.

The valuable outputs of this project are:
1. The diagnostic experiments (intrinsic_dim, rank/redundancy)
2. The 9 silent-failure fixes
3. The cluster pipeline + monitoring infrastructure
4. The corrected normalization utility
5. This honest negative result, which falsifies the architectural
   hypothesis at scale.
