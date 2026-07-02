# Pre-registration — First manifold dictionary on a 35B residual stream

**Status: PRE-REGISTERED / RE-FROZEN 2026-07-02 eve (expanded to the full 6-axis
scorecard).** By EVAL (Lane 6). Thresholds below are fixed *before* the numbers land;
the filled results doc `REPORT_35B.md` is a generated build output of
`experiments/report_35b_figures.py`. The original A1–A6 are unchanged — the expansion
only *adds* cells.

Target: on ~1.4M residual-stream tokens of **Qwen3.6-35B-A3B** (SuperGPQA reasoning
rollouts), layer **L17**, ambient **d=2048** — do activations decompose into the right
parts? The competitive baseline is **our own** `gamfit.sparse_dictionary_fit` (it IS a
TopK SAE) at matched K / L0. No external software is vendored (SPEC).

---

## Meta-rules (bake into every reading of this scorecard)

1. **Goodhart — no single metric is the objective.** This is a *portfolio*; thresholds
   are set before the run and no cell is optimized against. Our own parable: the
   **affine-PCA shortcut** that once "greened" the OLMo gate *by being its baseline* —
   a metric optimized becomes a metric that lies. Read the whole card, not one number.
2. **Effect size > significance.** At millions of tokens everything real is
   statistically detectable, so *truth* (evidence — is it there) and *salience*
   (min-effect floors on Θ, ΔEV, dose) are **separate dials**. Every geometry/causal
   claim carries a pre-set min-effect floor, distinct from its significance.
3. **Hierarchy: descriptive < predictive < causal.** Fidelity/parsimony/geometry
   describe; steering necessity/sufficiency and dose calibration are *causal* (top of
   the hierarchy). A method that only describes is a demo; the crown is the climb.

---

## The scorecard — 6 axes. A run is reported cell-by-cell; misses are informative.

Legend: **GATE** = hard pass/fail that licenses its axis; **HEADLINE** = one of the two
print figures; **STRETCH** = needs a harness not guaranteed tonight (reported if it lands,
else PENDING, never faked).

### Axis 1 — FIDELITY (preserves what the model USES)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **A1** | held-out EV vs TopK @ matched **actives** | within **0.02** below/above | T1 + COMPOSE |
| **F2** | **loss-recovered** `(L_ablate−L_recon)/(L_ablate−L_clean)` @ floor | hybrid ≥ TopK − **0.02** | CONTROL |
| **F3** | **KL-patched** `KL(clean ‖ recon-patched)` @ floor | hybrid ≤ TopK × **1.05** | CONTROL |
| — | **distortion floor** `R²*` (quantize sweep, BSF) | *reported*; ALL fidelity read AT it | CONTROL |

Fidelity in EV alone is Euclidean and prices all directions equally; the model reads them
unequally (rogue dims = huge variance, ~0 leverage). F2/F3 are fidelity in the *model's
currency* and are the deciding fidelity cells — hence Tier-0 removes the top rogue dims.

### Axis 2 — PARSIMONY (simpler than the thing)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **P1** | **L0** (mean active latents/token) | *reported*; matched-actives named (a d-chart = d actives + gate) | T1 + COMPOSE |
| **P2** | **MDL bits @ distortion-floor δ** (#2085 surface) | ≥ **5** curved atoms with finite `f*` AND actual firings ≥ `f*` (chart pays) | COMPOSE mdl |

MDL = support bits + value bits (water-filled ½log₂(var/δ)) + residual + amortized
dictionary; our REML evidence IS this up to constants. A necessary *filter*, never the
objective (compression ≠ comprehension). Read at the floor — precision above δ is wasted.

### Axis 3 — IDENTITY (one atom = one thing)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **I1** | **shatter count** — # linear latents to match ONE curved atom at fixed fidelity | analytic **median ≥ 2** over accepted curved atoms (ε=0.1), analytic≈empirical (≤2×) where empirical lands | Θ (analytic) + COMPOSE/T1 (empirical, STRETCH) |
| **I2** | absorption rate + one of SCR/TPP + sparse-probing | **STRETCH** (SAEBench harness) — PENDING | — |
| **I3** | chart-interp: is the coordinate ordering *nameable* (Mon→Sun, dim→bright) | qualitative; nameable iff A4 ordering > 0.9 on a named probe | DOSE |

Shatter law: a linear SAE needs ~`Θ/(2√(2ε))` atoms to match a curve at rel-err ε. I1
converts "identity" into a number a linear SAE scores on too — the honest cross-method axis.

### Axis 4 — GEOMETRIC CORRECTNESS (our axis; licensed by the null gate)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **G0** | **HALLUCINATED-STRUCTURE CONTROL** — full pipeline on (i) Gaussian noise matched to real mean+cov, (ii) shuffled real data | **GATE**: accepted curved atoms ≤ **1** (target 0) AND mean Θ < **0.5** on BOTH nulls; harmonic matched-null shows no spurious higher modes | CONTROL |
| **A2** | **(Θ, ΔEV) scatter** — the discriminating figure | ≥ **5** atoms Θ>1 rad & ΔEV>min_effect | COMPOSE |
| **A4** | coordinate fidelity: circular corr(fitted t, true cyclic) / ordering | > **0.9** | DOSE |
| **G_wrap** | **wraparound**: Sun adjacent to Mon on the chart (a line's ends are maximally far) | **pass** (cyclic first/last-probe adjacency) | DOSE |
| **G_band** | **band coverage**: 95% band contains fraction of held-out on-atom points | coverage ∈ **[0.90, 0.98]** (target 0.95) | COMPOSE |
| **G_util** | **stable rank** `(Σλ)²/Σλ²` of within-atom code cov ≈ d (ARD prunes idle) | median ≤ **d + 0.5** (d=1 chart ⇒ ≤ 1.5) | COMPOSE |

**G0 is the most important negative control we owe** precisely because we add a richer
prior: a method that finds circles in noise is DISQUALIFIED regardless of every other
score. Passing G0 *licenses* the whole axis. Θ = ∫|κ|ds (reparam-invariant; line 0,
circle 2π); high-ΔEV@Θ≈0 = linear in a curved costume, high-ΔEV@high-Θ = genuine curved
minority. G_band is calibration (decoration vs real UQ): far above = vague, far below =
overconfident.

### Axis 5 — CAUSAL VALIDITY (top of the hierarchy)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **A5** | **dose slope** — regress measured-KL on predicted-nats (Fisher-integrated along chart) | slope ∈ **[0.5, 2]** | DOSE |
| **A6** | **dose R²** | > **0.7** | DOSE |
| **C_steer** | on-target effect at **matched coherence** (fair vs direction-steering) | **STRETCH** (steering_bench) — reported if it lands | DOSE/steer |

Dose calibration is **the crown, ours alone**: slope≈1 + high R² ⇒ the coordinate system
is *metrically* true — the reliability diagram of interpretability. No direction/block
system can produce it (no coordinate to integrate along).

### Axis 6 — RELIABILITY (believe us twice)
| ID | Metric | Threshold | Source |
|----|--------|-----------|--------|
| **R1** | **seed stability, two resolutions** | subspace **principal-angle overlap > 0.9**; Hungarian latent-match reported *honestly* alongside (SAEs famously 15–50%) | STABILITY |
| **R2** | **cross-distribution replication** — creditscope Qwen3.5-35B L30 | ≥ **3** curved atoms recur (report which) — PENDING if arm not run | DATA/COMPOSE |
| **R3** | **split hygiene + matched budget** stated in advance | **pass**: chunk-level split + Tier-0 train-only + matched-currency (actives) named | DATA manifest |

---

## THE TWO HEADLINE FIGURES (print these)
1. **Pareto frontier** — held-out EV (and F2 loss-recovered on a twin panel) vs L0:
   hybrid vs pure-linear T1 vs our TopK, at matched actives. Tie/beat the field on ITS
   own axis. (Fig 1)
2. **Dose-calibration scatter** — predicted nats vs measured KL, slope + R². The field
   has NO axis here. (Fig 8)

Supporting figures: (Θ,ΔEV) scatter, curved-atom gallery, MDL bits/token, stable-rank +
utilization, curved-tier EV lift, probe ordering, and the **hallucination-null control bar**
(accepted curved atoms: real vs Gaussian-null vs shuffled-null — G0 made visible).

---

## Data hygiene (non-negotiable, pre-registered) — R3
- **Split by chunk / rollout, NEVER by row.** SuperGPQA rollouts are contiguous; a row
  split leaks adjacent tokens and inflates held-out EV for *every* method equally — the
  *first* way to fake tonight. Manifest states the chunk-level split + held-out chunk ids.
- **EV baseline = TRAIN column mean applied to held-out rows, NEVER the held-out column
  mean.** The held-out EV denominator (TSS) is taken about the train mean (equivalently
  the origin after subtracting the train Tier-0 mean). Using the held-out column mean
  leaks the first moment and inflates every absolute EV number identically — the *second*
  silent way to fake tonight. Held-out EV = 1 − SSE_recon/TSS on the disjoint whole-shard
  held-out split. When ingesting T1's frontier and COMPOSE's composed EV (they proved
  bit-exact parity with each other), I verify BOTH declare `ev_baseline: train_mean`, so
  the headline "matches TopK at 0.X" is an honest *absolute* number, not just an honest
  ranking. Both artifacts must carry an `ev_baseline` field; unstated → flagged UNVERIFIED.
- **Tier-0** (mean, top-1..3 rogue dims, global scale) on **TRAIN chunks only**, stored;
  held-out transformed with the frozen Tier-0.
- **Held-out EV / geometry** on a **50k held-out subsample** from held-out chunks (seed
  recorded). Fidelity currency (F2/F3) read AT the distortion floor.
- **Matched budget names its currency**: actives (compute/token) is the headline;
  bits@δ (information) and params (capacity) are reported as separate panels — mixing
  them is misleading.

---

## Producing-lane artifacts (drop into `results/run_35b/`)
| Lane | File | Feeds |
|------|------|-------|
| T1 | `l17_t1_frontier.json` | A1, P1 frontier + baseline |
| COMPOSE | `compose_per_atom.json` | A2, I1(empirical), G_band, G_util, P2(firings), curved gallery |
| COMPOSE | `compose_mdl.json` | P2 (MDL bits @ δ) |
| **CONTROL** | `fidelity_currency.json` | **F2, F3, distortion floor** |
| **CONTROL** | `null_control.json` | **G0 gate** |
| DOSE | `dose_calibration.json` | A4, A5, A6, G_wrap, I3 |
| STABILITY | `stability.json` | R1 |
| DATA | `manifest.json` | R3 attestation |

---

## Pre-registered failure branches (all informative, all reported)
- Full-width p=2048 births stall past the 30-min/birth guard → PCA-512 twin carries the
  headline, stall reported as the perf datum.
- 35B won't load for DOSE → Qwen3-8B carries the crown (A4–A6, wraparound on 8B), stated.
- A joint-fit arm collapses where stagewise does not → strongest architecture evidence.
- **G0 fails** (machinery finds curved structure in matched noise/shuffle) → axis-4
  claims are VOID and reported as such; this is the single disqualifying outcome.
- A1/F2 miss parity → reported honestly: curved minority resolves structure linear cannot
  (A2, I1) but does not yet reach TopK parity at matched actives; ΔEV/Θ evidence stands.
