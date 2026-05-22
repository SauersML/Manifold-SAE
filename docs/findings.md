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

## Pending (per-dim std reruns)

| Experiment | Status |
| --- | --- |
| `llm_sweep_q15b_L18_perdim` | queued |
| `llm_sweep_q05b_L18_perdim` | queued |

Once these complete, the headline real-LM numbers will be honestly
established for the first time.

## Honest verdict (interim)

After tonight's diagnostic:

* Synthetic-only claims: **architecture wins on 1D-manifold data,
  fails on 2D-manifold data, loses to vanilla on matched-decoder-params
  on arbitrary data**. These hold.
* Real-LM claims: **untrusted — pending per-dim rerun**. Likely most
  apparent wins (compactness, holdout transfer, multipenalty alive count)
  will weaken or invert.
* Architecture's strongest defensible value: **synthetic 1D-curve
  recovery at matched F**. The +25pp EV gain there is real and the
  scenario maps onto the strongest version of the architectural pitch.

The cluster bring-up + diagnostic work caught a critical normalization
bug. Without that, every "real LM" claim in this project was sand.
