# F4 — LN-sphere ambient + behavior-first pullback (2026-07-04)

Reviewer-F4 lane (AMBIENT). Two deliverables: (D1) fit atoms as submanifolds of
the LayerNorm sphere instead of flat ℝᵖ; (D2) the "boldest inversion" — fit the
manifold in behavioral (output) geometry first and pull it back to activations.

## Headline

**The task-invariant object is the pullback from output geometry, not the
embedded activation set.** Two probes of the SAME weekday feature, opposite
regimes:

| probe | what the position encodes | activation-first circle | behavior-first circle |
|---|---|---|---|
| **input-encoding** ("Today is Monday") | the weekday just read | **clean** (perfect weekday sort) | fails — next-token readout is template-dominated (same-template KL 0.19 vs same-weekday 1.48 nats) |
| **output-computation** ("The day after Monday is ___", pred-acc 0.71) | the arithmetic (base+offset), not the answer | **fails** (order-purity 0.41) | **clean** (order-purity 1.00; weekday-KL calibration r² 0.37 all-rows / **0.63 high-confidence** vs activation-first 0.007 / 0.017) |

The activation-first circle is tied to whatever a given layer/position happens to
encode — clean in one regime, junk in the other. The behavioral (output)
structure carries the feature whenever the task makes the model express it, and
its coordinate is **layer-independent by construction**. Precondition, now stated
and TESTED (same-day-vs-same-template KL diagnostic): behavior-first needs a
readout that actually carries the feature.

**Transport without refit** (`data/transport_figure.png`): the behavior-first
coordinate is fit ONCE and holds at L11/L18/L23 with order-purity 1.00 and a
stable pullback image (~14% activation EV each layer); the activation-first circle
must be refit per layer, its coordinates drift (cross-layer circular corr
0.28–0.42), and its order-purity decays 0.47→0.41→0.36. This is the atom-level
image of XPORT's metric result: the L18 output-Fisher metric propagates by the
model's Jacobian to L11 within 0.1%, and the pullback atom is a preimage of that
metric.

**First direct evidence for the pullback architecture** (D2 synthetic, where
behavior carries the feature by construction): behavior-first calibrates KL
**tighter** than activation-first (r² 0.94 vs 0.86), the two circles agree
(circular corr 0.95), and the pullback atom explains 99% of activation variance.

**D1 verdict:** a genuine LN-sphere fit IS needed — structured-residual whitening
cannot cover LN structure (the radial nuisance direction rotates with the latent,
outside the shared-factor model). `ln_sphere_project` (gam behavior.rs) + a green
Rust acceptance test. Flat-fit spurious curvature (fitted-norm CV) climbs
0.008→0.504 with injected norm variation; the LN-sphere fit stays ~0.003–0.008.
On real Qwen3-8B L18 activations the sphere decoder is machine-precision invariant
to injected norm (rel-change 1.9e-14) while the flat radial residual explodes
1.6→4.5.

## Files

`data/` — `spurious_curvature_result.json`, `real_norm_invariance_result.json`
(D1); `pullback_synthetic_result.json`, `pullback_pilot_result.json` (input
probe), `predict_weekday_pullback_result.json` (output probe + high-conf calib),
`transport_predict_weekday_result.json`, `transport_figure.png` (transport).

`code/` — `spurious_curvature.py`, `real_norm_invariance.py` (D1);
`pullback_synthetic.py`, `pullback_pilot.py`, `harvest_behavior.py` (input probe);
`harvest_predict_weekday.py` / `_ml.py`, `predict_weekday_pullback.py`,
`transport_predict_weekday.py`, `transport_figure.py` (output probe + transport).

In-tree (SauersML/gam): `crates/gam-sae/src/manifold/behavior.rs::ln_sphere_project`
+ `tests_ln_sphere_ambient_f4.rs` (green; confirmed by TOPO's full-suite run at
origin/main HEAD e4234b21f). Reusable MSI harvest:
`$ROOT/dose_qwen8b_out/predict_weekday_multilayer.npz` (probs7 on √7 sphere +
X_last at L11/L18/L23).
