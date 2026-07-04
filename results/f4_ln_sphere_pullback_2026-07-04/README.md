# F4 — LN-sphere ambient + behavior-first pullback (2026-07-04)

Reviewer-F4 lane (AMBIENT). Two deliverables: (D1) fit atoms as submanifolds of
the LayerNorm sphere instead of flat ℝᵖ; (D2) the "boldest inversion" — fit the
manifold in behavioral (output) geometry first and pull it back to activations.

> **SPEC-compliance status (2026-07-04, in progress).** The D2 circle-fit numbers
> below were originally produced by a numpy circle fit, which SPEC.md forbids as a
> *source* of published numbers (fitting logic must run in the Rust gamfit engine).
> Re-sourcing the fits through `gamfit.sae_manifold_fit` **changed a conclusion**:
> the "behavior-first calibrates KL tighter" claim is a numpy artifact and does NOT
> survive the production fitter (synthetic re-fit: cal r² 0.63 behavior vs 0.64
> activation — tied — where numpy gave 0.94 vs 0.86). **That claim is RETRACTED**
> pending a newest-gamfit re-derivation. The robust, surviving claims are the
> ORDER/geometry ones (mirror pair, order-purity, latent recovery, pullback EV);
> the calibration r² values are marked *[numpy, provisional]* until re-sourced.

## Headline

**The task-invariant object is the pullback from output geometry, not the
embedded activation set.** Two probes of the SAME weekday feature, opposite
regimes:

| probe | what the position encodes | activation-first circle | behavior-first circle |
|---|---|---|---|
| **input-encoding** ("Today is Monday") | the weekday just read | **clean** (perfect weekday sort) | fails — next-token readout is template-dominated (same-template KL 0.19 vs same-weekday 1.48 nats) |
| **output-computation** ("The day after Monday is ___", pred-acc 0.71) | the arithmetic (base+offset), not the answer | **fails** (order-purity 0.41) | **clean** (order-purity 1.00) |

Order-purity is a measurement on the fitted coordinate (token adjacency by target
weekday), not a fit, so it is robust to the fitter. The weekday-KL calibration r²
comparison *[numpy, provisional]* was 0.37/0.63 (behavior) vs 0.007/0.017
(activation), but the analogous synthetic comparison does not survive gamfit, so
these await re-sourcing before they can be cited.

The activation-first circle is tied to whatever a given layer/position happens to
encode — clean in one regime, junk in the other. The behavioral (output)
structure carries the feature whenever the task makes the model express it, and
its coordinate is **layer-independent by construction**. Precondition, now stated
and TESTED (same-day-vs-same-template KL diagnostic): behavior-first needs a
readout that actually carries the feature.

**Transport without refit** (`data/transport_figure.png`): the behavior-first
coordinate is fit ONCE and holds at L11/L18/L23 with order-purity 1.00 (a
coordinate-level, fitter-robust claim); the activation-first circle must be refit
per layer, its coordinates drift (cross-layer circular corr 0.28–0.42), and its
order-purity decays 0.47→0.41→0.36. **Honest limit:** a direct JVP test of the
stronger algebraic claim — that the behavioral atom's tangent *plane* pushes
forward by the model's Jacobian onto the target-layer behavioral plane — is a
NEGATIVE: with exact forward-AD (wiring error 0.0) J·plane(L11) sits 73–80° from
plane(L18), barely reduced from the raw 82–84°. So the invariance that holds is
the coordinate ORDERING, not the tangent subspace. The metric/length invariance
(the correct reading of "composes with J") is XPORT's already-established 0.1%
result, which the layer-independent behavioral coordinate inherits.

**D2 synthetic** (behavior carries the feature by construction): under gamfit the
two circles still agree (circular corr 0.75), behavior-first recovers the true
latent better (|corr| 0.81 vs 0.68), and the pullback atom explains 92% of
activation variance — but the calibration advantage is gone (see banner).

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
