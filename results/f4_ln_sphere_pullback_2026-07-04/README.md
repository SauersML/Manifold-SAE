# F4 — LN-sphere ambient + behavior-first pullback (2026-07-04)

Artifact manifest (verdict lives in the lane hand-off, not here).

## code/
- `spurious_curvature.py` — synthetic sweep: flat vs LN-sphere circle fit under norm variation; measures decoder higher-harmonic (spurious curvature) fraction and radial residual RMS.
- `real_norm_invariance.py` — real Qwen3-8B L18 weekday activations (`X_last`): inject norm variation, show the LN-sphere decoder is invariant while the flat radial residual grows. Runs on MSI.
- `harvest_behavior.py` — CPU forward pass harvesting the next-token distribution (restricted top-union vocab) for the 70 weekday prompts. Submitted via sbatch on MSI (`msismall`).
- `pullback_pilot.py` — real-data behavior-first pullback vs activation-first circle: agreement, pullback activation EV, within/across-weekday KL diagnostic, same-template KL calibration.
- `pullback_synthetic.py` — method check: shared-latent synthetic where behavior expresses the feature; behavior-first pullback vs activation-first vs truth.
- `harvest_predict_weekday.py` / `harvest_predict_weekday_ml.py` — predict-the-weekday probe ("The day after Monday is ___") whose next-token readout CARRIES the weekday; single-layer (L18) and multi-layer (L11/L18/L23) harvests.
- `predict_weekday_pullback.py` — behavior-first vs activation-first on the weekday-carrying readout: order purity, pullback EV, and P7-KL calibration.

## data/
- `spurious_curvature_result.json` — flat/sphere hh-fraction + radial RMS vs norm-CV sweep.
- `real_norm_invariance_result.json` — flat/sphere fits across injected norm scales + sphere-decoder rel-change.
- `pullback_pilot_result.json` — real weekday pullback numbers + KL diagnostics + calibration.
- `pullback_synthetic_result.json` — synthetic pullback agreement/EV/calibration.

## In-tree code (SauersML/gam @ 6b9e4e17d)
- `crates/gam-sae/src/manifold/behavior.rs` — `ln_sphere_project` (the LN-sphere ambient path).
- `crates/gam-sae/src/manifold/tests_ln_sphere_ambient_f4.rs` — Rust acceptance reproducing the synthetic spurious-curvature result with the real circle fit.
