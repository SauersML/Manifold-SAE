# Findings — what we now know

**MAJOR CAVEAT (May 2026)**: A diagnostic experiment
(`experiments/diagnostics_rank_redundancy.py`) discovered that the
previous activation normalization used a scalar `X.std().item()` which
preserved rank-1 structure in LM residuals. Under this normalization,
99% of variance lived in 1 PC at Qwen-1.5B L4/L8/L12/L18 (participation
ratio ≈ 1.0). Under proper per-dimension std, 99% of variance needs
925-972 PCs (participation ratio ≈ 2.3).

**Every real-LM result in this document is contaminated by this bug.**
Vanilla SAE saturating at 2 alive atoms wasn't a vanilla failure — it
was the correct fit to rank-1 data. Manifold-SAE's higher alive-atom
counts may have been interchangeable atoms in the same low-rank
subspace rather than feature discovery.

The fix landed in commit `408c36c` (manifold_sae/_normalize.py + 11
patched call sites). All headline experiments are being re-run with
per-dimension normalization. Numbers below are PRE-FIX and should
not be cited until reproduced.

---

## Diagnostic results (the only post-fix numbers we trust)

### Q1: per-layer effective rank under three normalizations

| Layer | normalization | PCs for 50% | for 90% | for 99% | participation ratio |
| --- | --- | --- | --- | --- | --- |
| L4 | raw_centered | 1 | 1 | 1 | 1.0 |
| L4 | global_std (OLD) | 1 | 1 | 1 | 1.0 |
| L4 | **per_dim_std (NEW)** | 1 | **295** | **925** | **2.3** |
| L8 | raw_centered | 1 | 1 | 1 | 1.0 |
| L8 | global_std (OLD) | 1 | 1 | 1 | 1.0 |
| L8 | **per_dim_std (NEW)** | 1 | **267** | **902** | **2.3** |
| L12 | raw_centered | 1 | 1 | 1 | 1.0 |
| L12 | global_std (OLD) | 1 | 1 | 1 | 1.0 |
| L12 | **per_dim_std (NEW)** | 1 | **237** | **866** | **2.3** |
| L18 | raw_centered | 1 | 1 | 1 | 1.0 |
| L18 | global_std (OLD) | 1 | 1 | 1 | 1.0 |
| L18 | **per_dim_std (NEW)** | 1 | **324** | **972** | **2.6** |

Qwen-1.5B residual stream is high-rank under proper normalization.

### Q2: intra-SAE atom redundancy (PRE-FIX data, untrusted)

| arch | alive | mean \|cos\| | median \|cos\| | pairs > 0.5 | > 0.8 | > 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| vanilla | 2 | 0.820 | 0.820 | 1 | 1 | 0 |
| curve | 4 | 0.533 | 0.410 | 3 | 1 | 0 |

Both architectures' atoms have high inter-atom cosine — consistent
with both populating a small subspace of the (collapsed) residual
stream.

### Q3: curve-span vs direction-magnitude (PRE-FIX, untrusted)

Median along-curve / direction-magnitude ratio: **0.179**.
**0 atoms** where the curve dominates the direction.
**2 atoms** where the direction dominates (curve trivial).

Reading: under our normalization, the curve parameterization barely
did work — atoms behaved like vanilla atoms with a small wiggle term.

---

## Synthetic experiments — UNAFFECTED by the normalization bug

Synthetic data was constructed with proper per-feature distributions,
so these results are honest.

### 1D-curve recovery, matched F (`realistic_scaling`)

| Scenario | D | F | Vanilla EV | Curve EV | Δ EV | Δ chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | **+0.274** | **−34%** |
| mid   | 256 | 32 | 0.513 | **0.760** | **+0.247** | **−34%** |
| large | 512 | 64 | 0.452 | **0.643** | **+0.191** | **−36%** |

When the GT really is 1D manifolds, curve SAE wins +25pp EV.

### Continuous-recovery matched-decoder-params

| scenario     | vanilla mean \|ρ\| | curve mean \|ρ\| | Δ |
| --- | --- | --- | --- |
| monotone     | **0.949**   | 0.696   | **−0.254** |
| non_monotone | 0.527       | 0.554   | +0.027 (tied) |
| mixed        | **0.698**   | 0.480   | **−0.218** |

At matched-decoder-params on arbitrary synthetic data, vanilla often
wins on monotone & mixed. Curve only ties on pure non-monotone.

### Synthetic cyclic recovery — non-periodic basis DOES recover cycles

`experiments/synthetic_cyclic_recovery.py` plants 4 independent
circles (θ_n in [0, 2π) → cos(θ)·v1 + sin(θ)·v2) in ℝ²⁵⁶ and trains
the 1D Manifold-SAE on the planted data.

| variant | EV | alive | max ρ_circ | per-cycle ρ_circ |
| --- | --- | --- | --- | --- |
| non-periodic (`periodic=False`) | 0.662 | 15/16 | **0.899** | 0.75, 0.88, 0.84, 0.90 |
| periodic-default (K=8) | — | — | — | FAILED (gamfit bug) |
| periodic-K12 | — | — | — | FAILED (gamfit bug) |

**Two findings**:

1. **Non-periodic 1D SAE recovers planted cycles at ρ_circ ≥ 0.75
   on every cycle**. The Q1.5B L18 weekday probe failure
   (max ρ_circ = 0.305) was not an architecture failure — the
   architecture DOES recover cyclic structure when the signal is
   present. The failure on weekday prompts is signal-strength
   (likely: weekday isn't a strong-enough axis at Qwen-1.5B L18 with
   F=128 to allocate a dedicated atom).

2. **The periodic Duchon path in gamfit crashes**:
   `GamError: Duchon function-norm penalty (stiffness) was not built;
   ensure spec.operator_penalties.stiffness is Active`. The crash is
   at `gam-pyffi/src/lib.rs:782` (`duchon_function_norm_penalty`)
   when `periodic=true`: the periodic build pipeline returns a
   `penaltyinfo` that doesn't include an `OperatorStiffness` slot, so
   the FFI lookup fails. Bug to file in gamfit. Until fixed,
   `ManifoldSAEConfig(periodic=True)` cannot train.

### Synthetic 2D recovery — architecture FAILS

Both single-λ and proper-penalty 2D Manifold-SAE:

| | EV | alive | per-grid recovery |
| --- | --- | --- | --- |
| 2D atom arch | 0.921-0.937 | 8-12 | 0.17-0.25 per grid |
| 1D pair (2 atoms per grid) | 0.909-0.918 | 16 | 0.30-0.49 per grid |

**The 1D atom pair beats the single 2D atom on every grid**, even
with the mathematically correct Duchon penalty (S⊗M + 2·T⊗T + M⊗S).
The 2D architecture is not delivering on the "one atom = one 2D
manifold" claim.

---

## Engineering hardening (mostly intact across the fix)

9+ silent-failure modes found during cluster bring-up, each with a
permanent fix (self-test, assertion, env-knob, or stamp). See
docs/known_issues.md. Critically:

* `amp²·curve(t)` bug in continuous-amp forward (commit `9f31143`)
* `update_snapshot` self-test catches training/locked divergence
* torch <2.12 + cu128 wheels (commit `87fa40c`)
* `MSAE_REQUIRE_CUDA=1` assertion against silent CPU fallback
* gamfit dual-cuBLAS Python check bypass
* firing-weighted soft-rescale for sparse atoms
* eval-cache forward-semantics stamp

The 9 silent failures are real and the fixes are independent of the
normalization issue.

---

## Post-fix Q1.5B L18 sweep — the architectural claim is INVERTED

`llm_sweep_1.5B_L18_perdim_fast` (job `a8ad9d79a4a4`):

| F   | top_k | van EV | crv EV | Δ EV | van alive | crv alive |
| --- | --- | --- | --- | --- | --- | --- |
| 16  | 2 | 0.444 | 0.417 | **−0.027** | 12/16 | 5/16 |
| 32  | 2 | 0.461 | 0.419 | **−0.042** | 24/32 | 16/32 |
| 64  | 2 | 0.489 | 0.418 | **−0.071** | 38/64 | 10/64 |
| 128 | 2 | 0.505 | 0.421 | **−0.084** | 67/128 | **7/128** |

Under per-dim normalization:
* Vanilla wins on EV at every F. The gap widens with F (Δ=-0.08 at F=128).
* Vanilla uses many more alive atoms (67 vs 7 at F=128). The pre-fix
  headline "curve has 4-5× MORE alive atoms" is the OPPOSITE of the
  truth under proper normalization.
* Curve EV is roughly flat with F (0.417-0.421) — the architecture's
  expressivity isn't scaling with the dictionary. Atoms are wasted.

This is a real falsification of the central architectural pitch.

## Concept intrinsic dimensionality (post-fix, Q1.5B)

`concept_intrinsic_dim` (job `56c8258ae65f`):

| layer | concept | k50 | k90 | corr_dim | local_pca | PC1↔concept ρ |
| --- | --- | --- | --- | --- | --- | --- |
| L4  | magnitude | 4 | 10 | 1.48 | 6.0 | +0.13 |
| L8  | magnitude | 4 | 11 | 2.40 | 5.0 | +0.28 |
| L12 | magnitude | 5 | 15 | 2.55 | 4.0 | +0.15 |
| L18 | magnitude | 5 | 18 | 2.57 | 4.0 | +0.15 |
| L18 | brightness | 4 | 21 | 3.21 | 6.0 | +0.04 |
| L18 | temperature | 3 | 17 | 3.27 | 6.0 | −0.08 |

* Local intrinsic dimensionality of every measured concept is ≈ 4-6,
  not 1.
* PC1↔concept Spearman is +0.04 to +0.28 — the prior "Phase 1" claim
  of 15/30 (concept × layer) pairs with |ρ|>0.7 was a normalization
  artifact. Concepts are NOT first-PC aligned in the residual stream.

A 1D curve atom is structurally mis-shaped for these concept clouds.
At minimum the architecture should be run at `intrinsic_rank=4-8`
not 2; even then a one-curve-per-atom inductive bias may not fit.

## Synthetic 2D recovery v3 — ablations don't rescue the 2D arch

`synthetic_2d_v3` (job `0d7abddebe3d`):

| variant | EV_train | EV_lock | alive | mean_per_grid | frac_2d |
| --- | --- | --- | --- | --- | --- |
| v5_lower_K (K=4) | 0.949 | 0.946 | 10 | **0.279** | 0.94 |
| v0_baseline | 0.948 | 0.903 | 10 | 0.253 | 0.75 |
| v2_ortho | 0.948 | 0.948 | 6 | 0.236 | 0.75 |
| v1_isotropy_safe | 0.944 | 0.944 | 8 | 0.222 | 0.69 |
| v4_deeper_enc | 0.928 | 0.928 | 7 | 0.205 | 0.38 |
| v3_coverage | 0.908 | 0.907 | 16 | 0.175 | 1.00 |
| v6_combined | 0.904 | 0.904 | 16 | 0.146 | 1.00 |

None of the proposed fixes (isotropy, ortho, coverage, deeper encoder,
combined) lift mean_per_grid past ~0.28. The 2D atom architecture
genuinely can't pack a planted 2D grid into a single atom in this
configuration. Lower K (4 vs 8) slightly helps — matches the 1D
finding that smaller per-atom capacity → more useful atoms. Deeper
encoder hurts. Coverage on both axes hurts.

The lock-vs-train EV gap (0.903 vs 0.948 in baseline) is real — the
2048-token snapshot doesn't generalize the per-axis soft-rescale to
the full population. Atoms outside the locked range get clamped.

---

## Honest verdict — post-fix

* Synthetic 1D-curve recovery at matched F: **architecture still wins**
  (+25pp EV per old realistic_scaling). This claim survives.
* Synthetic 2D recovery: **architecture loses** even with ablations.
* Synthetic cyclic recovery: **non-periodic 1D arch recovers planted
  cycles at ρ_circ ≥ 0.75** — falsifies the "architecture failed on
  weekdays" claim; the weekday signal at Q1.5B L18 was too weak.
* Real-LM EV: **vanilla wins, Δ widens with F**. Concepts aren't 1D.
* Real-LM concept compactness, holdout transfer, "more alive atoms":
  **all the previous "wins" are inverted under per-dim norm**.

The architecture as currently shipped is **not** a universal upgrade
on real LM residuals. It is a useful specialized tool when the GT
structure is genuinely 1D (the synthetic curve scenario), and an
investigative lens (curve diagnostics, lock-and-cache decoder) — not
a feature-extraction win on Qwen residuals.

Open questions worth chasing:
* Does `intrinsic_rank=4-8` (matching measured local intrinsic dim)
  let curve atoms compete with vanilla on EV?
* Does scaling F to 256-512 close the alive-atom gap, or does curve
  stay capped at ~7 alive regardless of F?
* Is the holdout-test concept-encoding transfer (the metric vanilla
  failed PRE-FIX) also inverted, or does curve still win on transfer
  even if it loses on EV?

The cluster bring-up + diagnostic work caught a critical normalization
bug. Without that, every "real LM" claim in this project was sand.
The architecture's defensible scope is narrower than the pre-fix
findings suggested, but the post-fix scope is honest.
