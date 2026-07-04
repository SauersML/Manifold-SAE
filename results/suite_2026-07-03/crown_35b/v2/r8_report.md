# Real-model dose calibration — predicting an intervention's effect in nats

**Model:** `REAL model qwen3.6-35b-a3b (layer 17); measured output KL = patched forward pass, exact next-token distribution`

**Claim tested:** a curved manifold-SAE atom is an explicit parametric chart `g(t)` carrying a downstream output-Fisher metric, so `steer` reports `predicted_nats` — how far the model's output token distribution will move — *before* the edit. We plot that prediction against the **measured** output KL from actually patching the edit into the forward pass and re-reading the logits.

**Setup:** layer-17 residual-stream activations at calendar-token sites (weekday, month, color); one K=1 `circle` chart per feature with the downstream output-Fisher metric attached (`harvest_downstream_output_fisher_factors`, the exact real-model call). Feature token is the last position, so the measured KL is the clean next-token-distribution shift. Per-template demeaning before geometry (W7 recipe).

- mean chart reconstruction R² = 0.8017 over 1 atoms.

- **dose mode = `on_chart_amplitude_normalized`.** ON-CHART, amplitude-normalized. The demeaned calendar signal is O(30) on a ~96-norm residual, so the fitted circle has a genuine radius; the prior run's sub-measurable ~1e-7 move was caused by steer's presence weight `amplitude` (~1e-6 for this K=1 atom) scaling the whole displacement. We divide it out consistently: patched move = steer.delta/amplitude = the real chart displacement g(t1)-g(t0); predicted_nats = steer.predicted_nats/amplitude^2 = the Fisher path integral along that SAME arc. Doses target a fraction of ||h|| by inverting the chord for the chart radius R (clamped to the diameter 2R). predicted_nats_tangent (1/2 c_tan m^2) is recorded for the same move as the local-quadratic reference.


## Headline (ideal = slope 1.0, R² 1.0, ratio 1.0)

| method | n | slope (log-log) | R² | median meas/pred | mean|log ratio| |
|---|---:|---:|---:|---:|---:|
| **manifold chart — HELD-OUT, censored >floor** | 134 | 0.569 | 0.622 | 1.488 | 1.877 |
| manifold chart — HELD-OUT edits (raw, incl. sub-floor) | 150 | 0.625 | 0.748 | 1.852 | 1.997 |
| manifold chart — all edits, censored >floor | 275 | 0.617 | 0.672 | 2.490 | 2.167 |
| manifold chart — all edits (raw) | 300 | 0.652 | 0.744 | 2.981 | 2.251 |
| manifold chart — within empirical validity radius | nan | nan | nan | nan | nan |
| manifold — LARGE ARCs on-chart (registered ~flat) | 80 | 0.355 | 0.097 | 16.387 | 2.560 |
| manifold — LARGE ARCs TANGENT extrapolation (curvature probe) | 80 | 0.005 | 0.000 | 0.325 | 1.442 |
| linear latent, norm dose (no metric) — *task baseline* | 300 | 0.716 | 0.765 | 1.985 | 1.875 |
| linear latent + base-point Fisher (fairness ref) | 300 | 0.663 | 0.732 | 5.099 | 2.573 |
| linear+Fisher — TANGENT large arcs (where it breaks) | 80 | -0.034 | 0.045 | 0.488 | 1.285 |


**Per-atom empirical validity certificate** (largest dose with |log(meas/pred)|<0.2, on-chart, as a fraction of ‖h‖):

- `weekday`: certified radius = 0.000 ‖h‖ (11/195 edits calibrated)

**Curvature anisotropy** (regress log(meas/pred) on θ²; slope×4 = c⊥/c∥). Registered: on-chart arc ~flat, tangent extrapolation positive:

- pooled `chart_arc`: slope=-0.0195 → c⊥/c∥=-0.0778 (R²=0.001, n=80)
- pooled `tangent_arc`: slope=-0.0978 → c⊥/c∥=-0.3913 (R²=0.034, n=80)

**Empty-edit noise floor:** median zero-patch KL = 0.00e+00 nats over 10 controls (the measurement floor; edits below it are flagged `gated=true`).


**Gate audit:** 25/300 manifold edits fell below the per-prompt gate = max(1e-03, 30x measured floor) and are flagged `gated=true` (excluded from the certificate, the curvature regression, and the censored headline; never silently shipped). The **censored** headline rows above keep only edits whose measured KL clears 30x the per-prompt measurement floor (Scott's 35B protocol: floor-dominated small edits otherwise sit above y=x and drag the raw slope down).


**Router-flip diagnostic (MoE):** 87.7% of 300 patched edits flip at least one expert vs the base forward (mean 12.96 flips/edit, max 22); discrete routing jumps the smooth Fisher metric cannot predict — the honest suspect for any residual multiplicative calibration constant.

![dose calibration real](dose_calibration_real.png)


Left: predicted nats (x) vs measured output KL (y), one point per (atom, base, frac, sign), with y=x. Right: calibration ratio vs move magnitude.


Data: `dose_calibration_real.json`
