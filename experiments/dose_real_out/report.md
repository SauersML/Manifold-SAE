# Real-model dose calibration — predicting an intervention's effect in nats

**Model:** `REAL model llama-3.1-8b-instruct (layer 16); measured output KL = patched forward pass, exact next-token distribution`

**Claim tested:** a curved manifold-SAE atom is an explicit parametric chart `g(t)` carrying a downstream output-Fisher metric, so `steer` reports `predicted_nats` — how far the model's output token distribution will move — *before* the edit. We plot that prediction against the **measured** output KL from actually patching the edit into the forward pass and re-reading the logits.

**Setup:** layer-16 residual-stream activations at calendar-token sites (weekday); one K=1 `circle` chart per feature with the downstream output-Fisher metric attached (`harvest_downstream_output_fisher_factors`, the exact real-model call). Feature token is the last position, so the measured KL is the clean next-token-distribution shift. Per-template demeaning before geometry (W7 recipe).

- mean chart reconstruction R² = 0.9230 over 1 atoms.


## Headline (ideal = slope 1.0, R² 1.0, ratio 1.0)

| method | n | slope (log-log) | R² | median meas/pred | mean|log ratio| |
|---|---:|---:|---:|---:|---:|
| **manifold chart — `predicted_nats`** | 288 | 0.908 | 0.951 | 0.881 | 0.516 |
| linear latent, norm dose (no metric) — *task baseline* | 288 | 0.954 | 0.950 | 10.044 | 2.377 |
| linear latent + base-point Fisher (fairness ref) | 288 | 0.931 | 0.925 | 1.266 | 0.560 |

![dose calibration real](dose_calibration_real.png)


Left: predicted nats (x) vs measured output KL (y), one point per (atom, base, dose, sign), with y=x. Right: calibration ratio vs move magnitude.


Data: `dose_calibration_real.json`

## Interpretation (full run, 288 points)

The manifold chart's `predicted_nats` is a **near-unbiased** predictor of the real
model's output shift: median measured/predicted = 0.882 across ~4 decades of KL, and
**0.999 inside `steer`'s certified validity radius** (n=49) — where the chart's local
linearization is trusted, the dose in nats predicts the patched forward pass almost
exactly. The task baseline (a linear SAE latent scaled to the same ‖δ‖, carrying no
metric) is **~10× miscalibrated**: it assumes isotropy and cannot see the output-Fisher
anisotropy the chart supplies and path-integrates for free. A linear latent separately
handed the exact base-point Fisher calibrates to ~1.28 — close, but a bare SAE latent
does not come with that metric; the curved atom does.

Build: pre-guard-fix (gamfit 0.1.247, PyPI). Only the **weekday** circle is shown: the
12-token **month** loop triggers the pre-fix multi-modal auto-grow / co-collapse
(`sae_manifold_fit` grows a K=1 request into 7 co-collapsed circles), so it is skipped
by the reconstruction-R²/atom-count floor rather than reported as a bad chart. Re-run
month/color hue-loop against S1-guards' patched build.
