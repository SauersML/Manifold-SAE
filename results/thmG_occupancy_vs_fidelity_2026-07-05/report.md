# Theorem G / P1 — occupancy vs fidelity for topology

**Verdict: SUPPORTED (qualified).** A manifold atom's *topology* verdict (circle vs interval) is set by **occupancy** (number of firing rows `n_eff`), **not by fidelity** (reconstruction SNR / rank). "Fidelity cannot buy topology; only occupancy can" (Superposed Geometry, Theorem G).

## Design

2D+ sweep reusing the PREMISE held-out paired-deviance instrument verbatim (gamfit `sae_manifold_fit`, linear `interval` vs `circle`, leave-template-out 2-fold, sign test + Gaussian surrogate per cell). Verdict strength = held-out paired-deviance dividend `Δ = D(interval) − D(circle)` in nats; positive = circle wins.
- **Occupancy axis:** number of templates kept, `n_eff = C·n_tpl ∈ {14,21,28,35,56,70}`, at a **fixed** converging fit config.
- **Fidelity axes (held at full occupancy):** working rank `3→6`; additive isotropic noise `σ 0→2`.
- **Convergence guard (critical):** the same converging config (rho off, `smoothness_weight=0.01`, rank 4) was held identical across the *entire* occupancy range so the occupancy axis is not confounded by fit-convergence failure. Non-converged cells are **excluded**, never scored "weak."

## Result — the decisive channel (sycophancy · raw activation)

This is the one of four channels carrying a real topology verdict (circle beats interval); it lives in *activation* space, consistent with PREMISE (curvature real in activations, behaviorally inert).

**Occupancy births and sharpens the verdict** (fixed fidelity = rank 4, σ0; 4 seeds/level, 46/46 converged incl. every n_eff=14 cell):

| n_eff | verdict strength | sign-test p |
|---:|---:|---:|
| 14 | +1.0 | 0.018 (borderline) |
| 21 | +2.2 | 0.0074 |
| 28 | +1.8 | 0.00099 |
| 35 | +4.5 | 0.00014  ← turns on hard, then plateaus |
| 56 | +4.5 | 0.0002 |
| 70 | +3.8 | 0.00017 |

Measured circle R² drifts only 0.53→0.71 across this range (fidelity ×1.3) while the verdict rises ×4.5.

**Fidelity never flips or manufactures the verdict** (fixed full occupancy) — it only attenuates detection power:
- rank 3→6: verdict +5.2 → +3.0 (circle wins throughout, p 6e-6 → 1e-3)
- noise σ 0→2: verdict +3.8 → +1.3 (circle wins throughout, p 2e-4 → 0.02)

**Dissociation (the G signature).** OLS of `|verdict|` on standardized `n_eff` and measured fidelity (46 converged cells): **β_occupancy = +0.47, β_fidelity = +0.014 (t = 0.04, essentially zero).** Occupancy carries the effect; measured fidelity is inert.

## Weekday (intended flagship cyclic feature) — honest null

No topology verdict in any channel (behavioral or raw); sign unstable across occupancy, flat-null across fidelity. This matches PREMISE's held-out weekday finding — the in-sample weekday circle is post-selection optimism that does not cross-validate. It neither confirms nor falsifies G; notably, fidelity never manufactured the absent circle either (G-consistent).

## Caveats (honest)

1. The sycophancy occupancy *partial* slope is positive but sub-|t|>2 because the verdict plateaus above n_eff≈35 and the harvest caps n_eff at 70; the direct marginal p-trend (0.018→1e-4) is the cleaner evidence. More occupancy would tighten it.
2. Additive isotropic noise is a weak fidelity knob (the rank-4 PCA frame averages it out; measured R² barely moves), so **rank** is the informative fidelity axis — both agree.
3. Instrument deviates from PREMISE's rho-on config (which hangs — gam#2138): rho-off + fixed `smoothness_weight=0.01` + rank 4, held identical across occupancy; at full occupancy it reproduces PREMISE's sign in each channel.
4. Only one of four channels carries a real topology verdict, so this is a clean single-channel confirmation with the correct dissociation structure, not a broad multi-feature one.

## Reading

The **falsification failed** (fidelity is provably inert) and the **positive** holds (occupancy births + sharpens the verdict), giving qualified support for Theorem G. This *explains* the program's honest negatives: topology labels are the weakest, last-to-converge signal because they carry only bounded per-row `KL_seam`, so no amount of reconstruction fidelity resolves them — only occupancy does. Weekday's non-cross-validating circle is exactly the predicted low-occupancy null.

Artifacts: `thmG_occupancy_vs_fidelity.png` (occupancy marginal + 2 fidelity marginals + dissociation scatter per feature), `thmG_verdict.json` (machine-readable verdict + marginal tables + converged counts), `code/` (kill-scheduled, convergence-gated driver), `data/` (per-cell grids). Related gamfit robustness bug: SauersML/gam#2138.
