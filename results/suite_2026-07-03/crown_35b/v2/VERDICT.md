# crown_35b v2 — floor protocol + rank sweep (Qwen3.6-35B-A3B L17, bf16 2×A40)

**Verdict: PARTIAL / HONEST-MISS with a diagnosis.** v2 improved the median calibration
(median meas/pred 1.85 at rank 8 vs v1's 5.1×) but the held-out log-slope stayed ≈0.57 —
the intervention effect is dose-DEPENDENT, not a constant-units offset. The rank sweep
localizes the cause.

## Floor protocol — the honest finding
The empty-edit control (a zero patch, run as a SECOND forward vs the cached base) measured
**noise_floor = 0** on every prompt, both ranks. The 35B forward is deterministic under
`device_map=auto` + eval, so a zero-delta patch is bitwise-identical to the base and the KL
is exactly 0. Consequences:
- The `max(1e-3, 30×floor)` gate degraded to the fixed 1e-3, so the **>30×-floor censoring
  was inert** — that's why the censored slope ≈ the raw slope.
- This does NOT reproduce Scott's offline "~7e-4 floor → slope 1.003". That ~7e-4 must come
  from a DIFFERENT control (a random-direction / non-null perturbation), not the zero-patch
  empty-edit. A zero-patch floor is structurally 0 on a deterministic model. **Recorded as a
  finding: the zero-patch floor mechanism is inert here; a random-direction floor is the fix.**

## Rank sweep (Fisher sketch rank s), held-out edits
| s (rank) | raw slope | raw med | censored slope | censored med | gated | router-flip |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | 0.625 | 1.85 | 0.569 | 1.49 | 25/300 | 88% |
| 32 | 0.619 | 1.53 | 0.557 | 1.29 | 29/300 | 88% |
| 64 | (pending) | | | | | |

## Discriminator (NORM's test): CURVATURE/CLAMP, not sketch-rank truncation
Raising s from 8→32 tightened the median (1.49→1.29, less systematic under-prediction) but
left the SLOPE flat (0.569→0.557). Per NORM's pre-registered test, pure rank-8 Fisher
under-capture would have lifted slope AND median toward 1 together; the slope staying ≪1 says
the residual dose-dependence is large-arc CURVATURE / chord-clamp geometry, not Fisher rank.
Corroborated: **88% of patched edits flip ≥1 MoE router expert** vs the base forward — discrete
routing jumps the smooth output-Fisher metric cannot forecast — a mechanism for a
dose-dependent residual that rank cannot fix. (rank64 will confirm the slope-vs-s trend.)

## Censoring-threshold sweep (post-hoc, held-out) — no threshold recovers slope→1
Raising the censoring threshold T on measured KL does NOT push the slope toward 1 — it pushes
it toward 0, because higher T keeps only the LARGE-dose edits where curvature bites hardest:
| T (nats) | rank8 slope / med / n | rank32 slope / med / n |
|---:|---|---|
| 1e-3 | 0.569 / 1.49 / 134 | 0.557 / 1.29 / 133 |
| 3e-3 | 0.494 / 1.13 / 124 | 0.489 / 1.22 / 124 |
| 1e-2 | 0.436 / 1.08 / 117 | 0.432 / 1.19 / 117 |
| 3e-2 | 0.351 / 0.86 / 107 | 0.340 / 0.72 / 106 |
| 1e-1 | 0.154 / 0.68 /  87 | 0.138 / 0.58 /  86 |
The median ratio crosses 1 and drops below (under-predict at small dose, OVER-predict at large
dose) — a saturating/clamped response: the chart's predicted_nats keeps growing while measured
KL saturates. This DECISIVELY refutes, on this 35B data, "censor to 30×floor → slope 1.003":
no censoring threshold recovers a slope-1 calibration; censoring out small edits keeps the
worst-behaved large arcs. The calibration is good in MEDIAN in a small-dose band (T≈3e-3: med
1.1) but the SLOPE (dose-dependence) is a genuine large-arc curvature defect, not a rank or
units or floor artifact.

## v3 protocol change (named)
1. FLOOR: replace the zero-patch empty-edit (identically 0 on a deterministic forward — it only
   measures nondeterminism) with a MATCHED-NORM RANDOM-DIRECTION control: patch a random unit
   vector scaled to the edit's ‖δ‖, measure its KL = the honest per-magnitude measurement floor,
   and censor real edits at 30× THAT. This is the control Scott's ~7e-4 floor came from.
2. CURVATURE: restrict the calibration claim to the small-dose validity band (ratio≈1) OR model
   the large-arc saturation explicitly; raising Fisher rank does not help (slope flat in s).

Data: rank{8,32,64}/dose_calibration_real.{json,png,report.md}; ranksweep.json; threshold_sweep.json.
