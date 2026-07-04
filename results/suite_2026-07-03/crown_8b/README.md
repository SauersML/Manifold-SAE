# Crown result: calibrated dose–response forecasting on a real LLM (Qwen3-8B)

**Date:** 2026-07-03 · **MSI job:** 12471979 (COMPLETED, 18m16s, 1×A40, preempt-gpu)
**Headline:** an unsupervised curved manifold-SAE atom (a weekday *circle* fitted on
layer-18 residual-stream activations) carries a pulled-back output-Fisher metric that
**forecasts the behavioral effect of activation edits in nats, before making them** —
and the model then does what was forecast:

| arm | n | log-log slope | R² | median measured/predicted |
|---|---:|---:|---:|---:|
| **chart, HELD-OUT edits** | 84 | **0.945** | **0.999** | 1.098 |
| chart, all edits | 168 | 0.940 | 0.994 | 1.081 |
| linear latent, norm dose (field standard) | 168 | 0.951 | 0.800 | **0.157** |
| linear latent + base-point Fisher (fair ref) | 168 | 1.057 | 0.986 | 1.099 |

97.0% of edits land within 2× of the forecast (91.1% within 1.5×); the metric-free
baseline manages 10.7%. Measured KL spans 3.3e-05 … 0.36 nats (four orders of magnitude).
Pre-registered acceptance was slope ∈ [0.5, 2] and R² > 0.7 on held-out edits.

## What was done, end to end

1. **Prompts.** `code/safety_features.py`-style generation via `build_prompts` in
   `code/dose_calibration_real.py`: 10 templates × 7 weekday words, calendar word in the
   **last token position** (template-major order: prompt i has template i//7, day i%7).
2. **Harvest.** One forward pass per prompt through Qwen3-8B (36 layers, hidden 4096),
   hook on **layer 18** residual-stream output, keep the last-token vector →
   `X_last (70, 4096)`. Per-template mean `tmpl_mean (70, 4096)` stored; geometry is fit
   on `X_last − tmpl_mean` (the W7 demeaning recipe). Also harvested:
   `U_last (70, 4096, 8)` — 8 output-Fisher factor vectors per token, i.e. VJP probes
   vᵢ = Jᵀ F^{1/2} uᵢ through layers 19–36 + unembedding
   (`harvest_downstream_output_fisher_factors`); these give
   δᵀGδ ≈ Σᵢ (vᵢᵀδ)² without materializing the 4096×4096 pulled-back Fisher G.
   Cache: `data/harvest_cache_weekday_L18_n70.npz`.
3. **Fit.** `gamfit.sae_manifold_fit` — K=1, `d_atom=1`, `atom_topology="circle"`,
   `assignment="ibp_map"`, rank-8 working subspace, REML/evidence-selected smoothness.
   No labels used. Converged on the 3rd seed (n_iter=80, random_state=1093;
   the first two seeds hit the outer probe-refusal guard and were correctly aborted —
   see the job log `data/crown_job_12471979.log`). Fit r² = 0.9970;
   cyclic ordering vs true calendar: order_corr 0.995, wraparound TRUE
   (fitted order Wed→Thu→Fri→Sat→Sun→Mon→Tue), gap uniformity 0.733.
4. **Dose sweep.** Mode `on_chart_amplitude_normalized`: displacement =
   `steer.delta / amplitude` (the true chart displacement g(t₁)−g(t₀); `amplitude`
   ≈ 2.26e-1 here is the K=1 presence weight and must be divided out consistently),
   prediction = `steer.predicted_nats / amplitude²` (the Fisher path integral along the
   same arc). Dose fractions {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4} of ‖h‖, both signs,
   12 base tokens, Δt from inverting the chord for the fitted radius R=2.04 (clamped at
   the diameter). 168 manifold cells + 168 per baseline arm = 504 patched forward passes.
5. **Measurement.** Overwrite the layer-18 last-token activation with the moved point
   (re-assembled with the template mean), run layers 19–36, measure
   KL(clean next-token distribution ‖ patched) exactly. Half the edit cells held out of
   any calibration.

## Reproduction (bit-for-bit)

- Model: Qwen3-8B at `/projects/standard/hsiehph/sauer354/models/qwen3-8b` (HF snapshot),
  layer 18, fp32 harvest math on cuda:0 (A40), torch 2.12.0+cu130.
- gamfit **0.1.247**, wheel built from SauersML/gam SHA `67735d1f4`
  (`$ROOT/wheels_head2/gamfit-0.1.247-cp310-abi3-manylinux_2_28_x86_64.whl`,
  installed in `$ROOT/saevenv`, python 3.11). Config echoed in the JSON: rank=8,
  features=['weekday'], n_iter=40 (escalated to 80 on seed retry), n_bases=12, seed base 891.
- GPU runtime libs: `LD_LIBRARY_PATH=$ROOT/cudart_shim:$ROOT/saevenv/lib/python3.11/
  site-packages/nvidia/cu13/lib:<torch>/lib` (gamfit dlopens .so.12 names; torch cu130
  ships .so.13 — the shim + nvidia/cu13 dir supply them).
- Driver: `code/dose_calibration_real.py` (the exact deployed script). Run shape:
  `DOSE_FEATURES=weekday python dose_calibration_real.py` under the sbatch in the job log.
- Everything the sweep produced: `data/dose_calibration_real.json` (config + fit +
  per-row records) and `data/dose_calibration_rows.csv` (504 rows; per row: method, mode,
  atom, base, heldout, frac, dose, dt, clamped, amplitude, radius, c_tan, delta_norm,
  h_norm, delta_frac, off_manifold, validity_radius, within_validity, predicted_nats,
  predicted_nats_pathint, predicted_nats_tangent, measured_kl).
- Figures regenerate from the JSON with `code/crown_plots.py`; raw-data views with
  `code/raw_views.py` (needs the npz). Python deps: numpy, matplotlib, plotly (uv run).

## Figures

- `fig1_crown_hero.png` — predicted vs measured, log-log, y=x, held-out highlighted.
- `fig2_three_methods.png` — same edits under the three forecast arms.
- `fig3_ratio_vs_dose.png` — calibration ratio vs dose size (flat ≈1.1 for chart and
  linear+Fisher; flat ≈0.16 for the metric-free baseline).
- `fig4_weekday_circle.png` — the fitted circle with the 7 days at their fitted angles.
- `fig5_dose_response.png` — one edit site: the forecast is non-monotone (large doses
  wrap the circle and nearly cancel) and the measured points follow the fold.
- `fig6/7/8 + weekday_pca3d.html` — RAW activation views: PCA of the day-centroids
  (fig7: the week is a visible cycle in calendar order with no model involved),
  raw heatmap of the 40 most day-informative dims (fig8), rotatable 3D (html).

## Honest caveats & corrections (see writeup/ANALYSES.md for numbers)

- One feature family (weekdays), one model (8B). 35B replication + month/color and a
  sycophancy chart are in flight (results land as sibling directories).
- **[CORRECTED]** An earlier rev of this file called `predicted_nats_tangent` a
  "normalization inconsistency" (constant ≈4.2× deficit at dt→0). That was a
  misdiagnosis — retracted per NORM's audit (`../norm_audit/NORM_AUDIT.md`). The
  apparent "dt→0" rows are `clamped:true` at dt≈3.05 (116/168 rows: the chord for the
  fitted radius R≈2.04 inverts past the diameter and pins to 2R), so dt was never small.
  At *true* small dt the pathint/tangent ratio is 1.000 exactly, rising monotonically
  with dt as a base-point quadratic must. The 4.2× is genuine curvature over a
  near-half-circle arc — `predicted_nats_tangent` is correct physics, a valid
  local-quadratic reference. It is a reference column (never used for the fig5 nats), so
  no calibration number changes.
- **Linear + base-point Fisher is a strong baseline**: its error does *not* grow with
  dose up to 40% ‖h‖ on this feature. At these scales the metric is the calibration
  constant and the chart's distinctive contributions are the move itself (ordering,
  wraparound, the non-monotone wrap forecast) and on-manifold validity — not yet a
  measurable curvature-correction to the nats. Larger arcs (dt→π) are queued to find
  where base-point Fisher breaks.
- The KL>1e-3 first-edit bite gate printed FAIL and continued (prediction and
  measurement *agreed* below the gate at 4.92e-4 vs 4.92e-4 — the gate floor was
  miscalibrated, not the physics). Fixed to abort-or-annotate in the next driver rev.
- The formula-based `validity_radius` passed only n=7 rows; superseded by per-atom
  empirical validity radii in the next rev.
- Fit fragility is real: 2 of 3 REML seeds failed before convergence (guards aborted
  them correctly). Related solver-grind issues (#813/#821 class) are being fixed in gam.

## Bonus result in data/safety_charts.json (Track S, same day)

Sycophancy charts as a graded open-arc intensity dial (circle r²=0.693,
Spearman(coord, designed grade)=0.910, partial 294° arc, no wraparound — circle
survives the complexity penalty); refusal does NOT chart as a dial (circle boosts r²
but collapses ordering 0.63→0.32; linear is the honest pick). Design:
`writeup/safety_probes_DESIGN.md`; the sycophancy dose calibration is running.
