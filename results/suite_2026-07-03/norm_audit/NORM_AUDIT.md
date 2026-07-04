# NORM audit — predicted_nats normalization / gauge

Auditor: NORM lane. Scope: the `predicted_nats` pipeline end-to-end
(`crates/gam-sae/src/inference/steering.rs` — `steer_delta`, `path_integrated_dose`,
`validity_radius`, amplitude² handling; the per-arm python drivers; every raw
`dose_calibration_real.json` / `dose_safety_sycophancy.json`).

Method (falsification-first): for each flagged constant, derive what `predicted_nats`
SHOULD be, then reproduce the observed constant from the raw JSON. A genuine
multiplicative units bug in the shared assembly must (i) appear in EVERY arm including
the curved/manifold arms, and (ii) push all arms the SAME direction. Neither holds.

## Verdict table

| arm | observed | cause | bug or physics | published numbers affected |
|---|---|---|---|---|
| (a) 8B crown `predicted_nats_tangent` "4.2× deficit at dt→0" | ~3.9–4.9× | **misdiagnosis**: the report's "dt→0" rows are `frac`→small but `dt` **clamped to the diameter (≈3.05)**. At TRUE small dt (unclamped, dt≈0.02) `pathint/tangent = 1.000`. The ~4.2× is the real curvature deficit of a base-point quadratic over a half-circle arc. | **physics** (curvature), not a units bug. The tangent column is correct. | none — `predicted_nats_tangent` is a reference column, never used in fig5 nats. But the *writeup diagnosis text* (ANALYSES.md §"Tangent-column units bug", README.md line 132, crown_8b/README.md lines 91-93) is **wrong** and should be corrected. |
| (b) 35B residual | manifold median 10.0×, **slope 0.56** | dose-DEPENDENT (slope≠1), diverges at high dose. NOT a constant. Consistent with rank-8 Fisher sketch under-capture on MoE and/or genuine large-arc curvature. | deferred — **OPS35's rank-sweep lane** (s∈{8,32,64}). Not a constant-units bug. | TBD by OPS35. |
| (c) sycophancy LINEAR arm | 4.12× (all) / 6.20× (biting), slope 1.01 | the "linear" topology's `predicted_nats` is **isotropic** `½·c_bar·‖δ‖²` (`c_eff = 2·pred/‖δ‖²` is constant to 1.00 across all 336 rows — NO direction dependence). `c_bar = trace(G)/p` = mean Fisher eigenvalue. The sycophancy steering direction ("agree"/"false" tokens) is HIGH-Fisher, so the mean-eigenvalue scalar under-scores it 4-6×. | **physics** — the isotropic baseline cannot see direction. Proof: the SAME-data `circle` arm using the real Fisher metric calibrates to **1.015**; crown's `linear_fisher` (base-point Fisher) calibrates to **1.099**. | none — this is a labelled baseline behaving as designed. |
| (d) month/calendar LINEAR baseline | 0.157 (≈6× the OTHER way) | SAME isotropic `½·c_bar·‖δ‖²` baseline (`linear_norm`). Calendar/color directions are LOW-Fisher, so the mean-eigenvalue scalar OVER-scores them ~6×. | **physics** — same baseline, opposite spectral tail. | none — labelled baseline. |

## Key evidence

**(c)+(d) are the same baseline in opposite directions** — the decisive falsification.
A real multiplicative units bug in `predicted_nats` would push (c) and (d) the *same*
way. They go opposite ways (sycophancy under 4-6×, calendar over 6×) because the two
features live in opposite tails of the output-Fisher spectrum, and the isotropic scalar
`c_bar = trace(G)/p` (dose_calibration_real.py:657) is the mean. The arms that use the
*real* metric — `circle` (1.015), `manifold` (1.081), `linear_fisher` (1.099) — are all
calibrated. So the shared `predicted_nats` assembly is units-correct.

**Amplitude² division (`on_chart_amplitude_normalized`) is self-consistent.** Rust
returns `steer.delta = a·(g_to−g_from)` and `steer.predicted_nats = ½a²∫Δtᵀ g Δt`. The
driver divides the patched move by `a` and the nats by `a²` — the SAME `a`, removing its
dependence consistently (delta linear in a, nats quadratic). The measured KL patches the
a-divided delta and is compared to nats/a². Manifold calibration 1.081 confirms.

**(a) reproduction.** `crown_8b/data/dose_calibration_real.json`, manifold rows:
- 116/168 rows are `clamped:true` at `dt≈3.052` (chord for the fitted radius R≈2.04
  inverts past the diameter → clamped to 2R).
- Per-`frac` median `pathint/tangent` is identically 129051 — dominated by the clamped
  dt=3.052 rows. There is no dt variation among them, which is why the report's
  "(ratio−1)-vs-dt exponent" came out ≈0 ("constant"): dt was pinned, not →0.
- Unclamped rows (dt genuinely 0.019 → 2.59): `pathint/tangent = 1.000` at dt≈0.02,
  rising monotonically with dt (77× at dt=1.83). Base-point tangent quadratic converges
  to the path integral at dt→0 exactly as it must. No inconsistent constant.

## Bottom line

No units/gauge bug in the `predicted_nats` assembly. Arms (c) and (d) are the honest
isotropic baseline demonstrating why the metric is needed; arm (a) is misdiagnosed
clamp-driven curvature (tangent column is correct); arm (b) is dose-dependent
(OPS35 rank sweep, not a constant). The only fix owed is a **text correction** to the
crown_8b tangent-column diagnosis, deferred to the crown/OPS35 lane owner.
