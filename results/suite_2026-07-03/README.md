# Results suite — 2026-07-03 (Qwen3-8B / Qwen3.6-35B, manifold-SAE program)

Everything needed to analyze or hand to another agent. Each subdirectory is one
experiment with its raw data, logs, and (where finished) figures. Repro details for
the flagship are in `crown_8b/README.md`; all fits use gamfit (SauersML/gam) with
SHAs noted per experiment.

## Index & one-line verdicts

| dir | experiment | verdict |
|---|---|---|
| `crown_8b/` | dose-calibration crown, Qwen3-8B L18 weekday circle | **PASS, headline**: held-out slope 0.945, R²=0.999, median ratio 1.10; 97% of edits within 2× of forecast. Full reproducible bundle (figures, 504-row raw data, harvest npz, driver code, analyses incl. honest negatives). |
| `crown_35b/` | same recipe, Qwen3.6-35B-A3B L17, bf16 2×A40 | **FAIL v1 (honest miss)**: held-out slope 0.559, R²=0.69, median ratio 5.8×. n=42 harvest rows (fewer than 8B's 70), bf16 numerics in the Fisher harvest, and L17-by-depth-analogy are the suspect list. Retry job (12486012) running. Raw JSON + PNG + full log included. |
| `safety/` | Track S: sycophancy & refusal charts + sycophancy dose loop | **Sycophancy = graded open-arc intensity dial** (circle r²=0.693, Spearman(coord, designed grade)=0.910, partial 294° arc, no wraparound); **refusal = switch-like** (circle boosts r² 0.83→0.87 but collapses ordering 0.63→0.32 — honest negative). **Dose loop (n=336): slope 0.934, median ratio 1.01, bite rate 1.0, R²=0.59** — absolutely calibrated on average, noisier per-edit than the calendar crown (fit-quality caveat r²=0.69 carried). Behavioral effect real: on the circle arm the agreement-pole shifts +0.125 (reduce-agreement dose) / +0.235 (increase-agreement dose); the metric-free linear arm gives the cleaner ± split (+0.083 / −0.059) — verified against `safety/dose_safety_sycophancy.json`. |
| `scale_evidence/` | gam-side scale runs on real 35B L17 activations | **Block lane at K=32,000 explained variance 0.9895** with 0 dead blocks, orthonormality dev ≤4e-8, 55 min (t1_frame_health.json — rows for K=4k EV 0.707, K=16k EV 0.906, K=32k EV 0.990). The old K≥64 co-collapse ceiling (EV 0.199) is dead. qwen_kscale.log: dense-width joint fit at p=2048 hits the #1995 width-wall grind (K=2 timeout) while PCA-128 arm is healthy — corroborates the grind class. real_l17_ab.log: 150k×2048 stagewise rank-charge A/B in flight. |
| `nulls/` | hallucinated-structure controls (matched Gaussian + shuffled real) | IN FLIGHT (jobs 12484519/20, ~1h in at snapshot). PASS criterion: ~0 accepted curved atoms on noise, Θ mass at 0. Verdict JSONs will be added when scored. |
| `rung3/` | splice-KL / loss-recovered driver (behavioral-Fisher rung 3) | PLACEHOLDER — results land from the rung3 lane. |
| `transport/` | per-layer chart fits + carried-vs-rotated metric transport (JVP arm) | PLACEHOLDER — results land from the transport lane. |
| `premise/` | premise / assumption-audit experiments | PLACEHOLDER — results land from the premise lane. |
| `norm_audit/` | normalization / dose-mode audit | PLACEHOLDER — results land from the norm_audit lane. |
| `crown_35b/` (v2) | 35B recipe with floor protocol ported from 8B | PLACEHOLDER — v2 retry rows land alongside the v1 FAIL above. |

## The one-paragraph story

An unsupervised curved atom fitted to a real LLM's residual stream carries a
behavioral metric good enough to *forecast* interventions: on Qwen3-8B, "move this
token N days along the weekday circle" comes with a predicted output shift in nats
that the model then matches to ~10% (median) across four orders of magnitude — while
the field-standard metric-free steering is 6× miscalibrated. The same instrument
pointed at sycophancy finds a graded intensity dial (and honestly reports that
refusal is a switch, not a dial), and its dose loop is calibrated in scale on the
first try. The first 35B attempt failed calibration — recorded as such. Meanwhile the
block-sparse dictionary lane scales to K=32k on real 35B activations at EV 0.99 with
zero dead atoms, which is the substrate the full Atlas run composes on. The forecast
also holds *globally*: on 8B the chart's calibration error does not grow as the edit
sweeps the full arc — the signed log-ratio's slope against |Δt| is ≈0.0015 out to
|Δt|≈3.05 rad (essentially flat), so a large edit that wraps most of the circle is
predicted as well as a tiny one. Controls (noise nulls) and the composed run are in
flight.

## Definitions (used precisely throughout)

- **chart** — an explicit parametric curve g(t) fitted (unsupervised) to a feature's
  activations; t is the *coordinate* along it. A K=1 `circle` chart is g(t) tracing a
  circle in the residual stream. The chart supplies *where* an edit lands (which token
  it moves toward, in what order) — its ordering/targeting content.
- **metric** — the pulled-back output-Fisher G at a point: δᵀGδ is the KL the output
  distribution moves for a residual-stream displacement δ. Attached to the chart via
  VJP probes (δᵀGδ ≈ Σᵢ(vᵢᵀδ)²) so `steer.predicted_nats` is a path integral of the
  metric along the chart arc. The metric supplies *how far* the output moves — the
  scale of an edit, in nats.
- **nats** — natural-log units of KL divergence between the clean and patched
  next-token distributions; the currency both the forecast and the measurement are in.
- **dose** — the edit magnitude, expressed as a fraction of ‖h‖ (the residual-stream
  norm) and converted to an arc length Δt along the chart. `amplitude-normalized dose`
  divides out the K=1 presence weight so the displacement is the true chart move
  g(t₁)−g(t₀), not a ~1e-6-scaled shadow of it.

## Attribution: what the metric buys vs what the chart buys

Two components combine in a forecast, and the suite separates their contributions.

**Metric ⇒ calibration (scale).** Hold the *coordinate* fixed (a linear SAE latent)
and toggle only the metric. On 8B the metric-free arm (`linear_norm`, displacement
scaled to a target ‖δ‖ with no metric) sits at median measured/predicted **0.157** —
i.e. ~6.4× miscalibrated, and only 10.7% of edits land within 2× of forecast. Hand the
*same* linear coordinate the exact base-point Fisher (`linear_fisher`) and the ratio
snaps to **1.099** with 92.9% within 2×. Same move, same coordinate; the only change is
the metric, and it is what makes the forecast right in scale. The Track-S sycophancy
dose loop shows the same signature behaviorally: the metric-free linear arm calibrates
in *shape* (log-log slope 1.01) but carries a **6.2× constant offset** (median
measured/predicted 6.198 on biting edits) because a bare latent has no metric-correct
scale; the circle arm, which path-integrates the metric, is scale-accurate (median
ratio **1.015**, bite rate 1.0).

The cleanest proof that the metric is doing this — and not some incidental scaling —
comes from NORM's audit (`norm_audit/NORM_AUDIT.md`, arms (c)+(d)). The *same* isotropic
baseline (`½·c̄·‖δ‖²`, where `c̄ = trace(G)/p` is the mean Fisher eigenvalue) miscalibrates
in **opposite directions** depending on where the feature sits in the output-Fisher
spectrum: on **high-Fisher** sycophancy it *under*-scores by 4–6× (linear median 6.198),
while on **low-Fisher** calendar it *over*-scores by ~6× (linear_norm median 0.157). A
real multiplicative units bug in the shared `predicted_nats` assembly would have pushed
both the *same* way — they go opposite ways because the isotropic scalar averages over a
spectrum with opposite tails. Meanwhile every arm that carries the *real* metric
calibrates regardless of feature (circle 1.015, manifold 1.081, linear_fisher 1.099).
Direction-dependent failure of the isotropic baseline, direction-independent success of
the metric arms: that is the metric mattering, demonstrated by construction.

**Chart ⇒ targeting (ordering).** The metric alone does not tell you *which* token an
edit moves toward — that is the chart coordinate. The weekday chart recovers calendar
order with order-correlation **0.995** (unsupervised, wraparound TRUE). On sycophancy
the fitted coordinate orders the seven *designed* intensity grades at Spearman **0.910**
(chart-topology fit). This is what "lands on the feature" means: the coordinate is a
graded dial you can address monotonically.

**The safety table is the cleanest demonstration** — same instrument, three features,
each isolating a different regime (numbers verified against `safety/` JSONs except
hedging, which is MSI-only and marked):

| feature | regime | scale (median meas/pred) | ordering (Spearman coord↔grade) | verdict |
|---|---|---:|---:|---|
| sycophancy (chart fit) | graded open arc (294°, no wrap) | — | **0.910** circle / 0.901 linear | chart orders the dial |
| sycophancy (dose loop) | scale test | **1.015** circle / 6.198 linear (biting) | 0.413 circle / 0.907 linear | metric fixes scale; circle ordering unstable across seeds |
| refusal (chart fit) | switch-like | — | 0.626 linear → **0.316** circle | **honest negative**: forcing a circle collapses ordering (0.63→0.32); linear is the honest pick |
| hedging (dose loop, MSI-only) | graded | 1.00 slope / R² 0.83 / ratio 0.81 | — | MSI-only; not yet verifiable against a local JSON |

Read together: the metric is what makes the *nats* right (linear+Fisher and the circle
both calibrate; the metric-free linear is 6.2× off), and the chart coordinate is what
makes an edit *land in order* on features that are actually graded — while honestly
reporting (refusal) that a switch-like feature has no dial to land on.

## Ledger of arbitrariness

Every knob a skeptic could call arbitrary, and for each: why it is principled, or the
sensitivity check that discharges it. Culture note (Scott): *geometry is never used
where a price will do* — i.e. we do not reach for a curved chart when a scalar penalty
or a linear coordinate already explains the data; the chart has to earn its complexity
penalty.

| choice | value here | why it's principled / the check |
|---|---|---|
| **layer** | 8B L18 / 35B L17 | Chosen once by depth analogy (mid-stack, where calendar structure is legible), not tuned per result. Sensitivity check: the transport lane fits charts across L11–L23 and reports carried-vs-rotated metric — if the result only exists at one layer, that lane exposes it. |
| **last-token convention** | feature token = last position | Makes the measured KL a *clean* next-token-distribution shift (no averaging over positions to launder). It is a definition of the readout, not a fit degree of freedom. |
| **per-template demeaning (W7)** | subtract per-template mean before geometry | The raw signal is dominated by template identity (template-0 Frobenius norm 2.9× the others); demeaning isolates the *within-template* calendar move. Documented, and the raw-data views (fig7/8) show the week is visible in calendar order *after* correct labeling, so the structure is not manufactured by the demeaning. |
| **dose grid** | {0.005…0.4}‖h‖ (crown); {0.02…0.5} (sycophancy) | Spans four decades of KL so slope/R² are not read off one scale. Both signs, multiple base tokens. The grid is reported in every JSON `config`. |
| **bite gate** | KL > 1e-3 | A floor to declare an edit "behaviorally real". In the crown it fired FAIL at 4.92e-4 where *prediction and measurement agreed* — i.e. the gate floor was miscalibrated, not the physics; disclosed and being moved to abort-or-annotate. The spec lane's floor is currently bugged (see honest-negatives) and being rerun. |
| **validity radius** | formula-based (crown v1) | Passed only n=7 rows and is superseded by per-atom *empirical* validity radii in the next rev. We report both the within-validity (n=7, ratio 1.057) and held-out (n=84) arms so the claim does not hinge on the formula. |
| **amplitude-normalized dose mode** | δ = steer.delta/amplitude | Not a free scale: the K=1 presence weight (~2.26e-1 here, ~1e-6 in the pre-fix run) multiplies the whole displacement; dividing it out consistently (and predicted_nats by amplitude²) recovers the *true* chart move. The `dose_mode_note` in each JSON derives it. The norm_audit lane exists to stress this exact choice. |
| **topology (circle vs linear)** | chosen by penalized loss, not by hand | The fit *selects* topology under a complexity penalty; circle is only preferred when it beats linear net of the penalty (sycophancy: circle−linear r² +0.089 and survives; refusal: circle wins r² but we *keep linear* because it collapses ordering). This is the price mechanism doing its job. |
| **rank-8 working subspace** | rdim 8–32 | A compute economy for the fit, not the readout (measurement is full-dimensional forward passes). Note the honest counter-finding that PCA-compressed fits condition *worse* than raw-ambient here (#813/#821 class). |

## Honest-negatives index

Every registered miss, in one place. This is the ledger the writeup is judged against.

| miss | where | status |
|---|---|---|
| **35B v1 calibration FAIL** | `crown_35b/` | Held-out slope 0.559, R² 0.691, median ratio 5.786 (vs 8B's 0.945/0.999/1.098). Suspects: bf16 numerics in the Fisher harvest, n=42 harvest rows, L17-by-analogy. Recorded as a fail; v2 with the 8B floor protocol is queued. |
| **35B v2 calibration** (MSI-only) | placeholder | Reported still failing on the second attempt per the fleet; not yet landed as a local JSON — MSI-only. |
| **refusal ordering collapse** | `safety/safety_charts.json` | Forcing a circle on a switch-like feature raises r² 0.834→0.871 but collapses coordinate↔grade ordering 0.626→0.316. The chart honestly has no dial to offer here; linear is the honest pick. |
| **anisotropy regression null** (MSI-only) | placeholder | R² 0.048 / 0.004 — no measurable anisotropy signal. MSI-only, not locally verifiable. |
| **linear+Fisher NOT breaking on tangent arcs — theory tier-3 unproven** | `crown_8b/writeup/ANALYSES.md` | The pre-registered curvature test failed to fire: base-point Fisher stays calibrated (does not degrade quadratically) up to 40% ‖h‖; it is *modestly better* than the chart's path integral at large fractions. What curvature buys over base-point Fisher is not yet demonstrated in nats — only ordering/wraparound/validity. Larger arcs (dt→π) are queued. |
| **month certified radius only 0.005‖h‖** (MSI-only) | placeholder | The month upgrade's certified validity radius is tiny (~0.005‖h‖); held-out calibration reported 0.963/R²0.894/ratio 1.054 but over a narrow trusted band. MSI-only. |
| **color weak ordering 0.30** (MSI-only) | placeholder | The color feature charts with weak coordinate↔grade ordering (~0.30). MSI-only. |
| **misdiagnosed "tangent-column units bug" — retracted** | `crown_8b/writeup/ANALYSES.md`, `norm_audit/NORM_AUDIT.md` | We originally published a normalization bug in `predicted_nats_tangent` (a "constant ≈4.2× deficit at dt→0"). **That was wrong** and is retracted: NORM's audit found the "dt→0" rows were `clamped:true` at dt≈3.05 (116/168 rows), so dt was pinned, not small; at *true* small dt the pathint/tangent ratio is 1.000 exactly. The 4.2× is real clamp-driven curvature of a base-point quadratic over a near-half-circle, not a units bug — the tangent column is correct physics. No published calibration number was affected (tangent is a reference column). Kept visible as our own corrected error. |
| **spec collateral floor bugged** (MSI-only) | placeholder | Spec-specificity collateral ratios 0.003–0.005 vs linear 0.02, but the zero-floor is bugged and being rerun; treat as provisional. MSI-only. |
| **fit-seed fragility** | `crown_8b/` | 2 of 3 REML seeds failed to converge (guards aborted them correctly); only seed 1093 converged. The result is real but the fit is not push-button robust yet. |

## Provenance

- Models: Qwen3-8B (`$ROOT/models/qwen3-8b`), Qwen3.6-35B-A3B (bf16, 2×A40).
- gamfit: 0.1.247 (SHA 67735d1f4) for crown_8b; 0.1.248 (6f89ebf84) for scale runs.
- Cluster: MSI, partitions msismall (CPU) + preempt-gpu (A40/L40S); $ROOT =
  /projects/standard/hsiehph/sauer354.
- All prompts/templates are generated in code (see crown_8b/code/ and
  safety/safety_probes_DESIGN.md) — no external datasets.
- Repos: SauersML/gam (engine), SauersML/Manifold-SAE (this bundle).
