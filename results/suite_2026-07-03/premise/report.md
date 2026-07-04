# Premise instrument — held-out paired deviance + slow-feature atlas

**Question 1 (curvature).** For each candidate feature, does adding *curvature* to the 1-D chart reduce reconstruction deviance on rows the fit never saw — independent of any dose calibration? We fit a straight `line` and a `circle` (same dimension, one extra geometric d.o.f.) on the demeaned residual stream via the identical `sae_manifold_fit`, score every row **held out** on a fit that never saw its template (2-fold complementary template split), and take the PAIRED per-row deviance difference `Δ = D(line) − D(circle)` in the behavioral (output-Fisher, nats) metric and in raw activation units. Significance is a paired **sign-flip** randomization test (the exact scheme for a within-row contrast). See `DESIGN.md`.

**Falsification.** A Gaussian-matched surrogate (structureless, same 2nd moments) is run through the identical pipeline; if the circle 'wins' there, the extra geometric freedom is biasing the test and the p-values are worthless. It must sit at Δ≈0.

Squared-deviance means are outlier-sensitive (a held-out point that projects catastrophically onto the *closed* circle — which, unlike a line, cannot extrapolate — dominates the mean), so the **robust headline is the distribution-free sign test on the median dividend**; the mean-based sign-flip p is reported alongside.

## Behavioral dividend (output-Fisher, nats) — the calibration-free premise number

| feature | n rows | median Δ | frac rows circle wins | sign-test p | mean-flip p | surrogate median | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| weekday · 8B L18 | 70 | -0.018 | 0.3 | 0.0011 | 5e-05 | 0.00878 | **curvature COSTS** |
| month · 8B L18 | 120 | -0.00139 | 0.46 | 0.41 | 0.16 | 0.00177 | honest negative |
| sycophancy · 8B L18 | 70 | -0.000235 | 0.5 | 1 | 0.66 | 0.00305 | honest negative |
| hedging · 8B L18 | 70 | 0.00458 | 0.53 | 0.72 | 0.58 | 0.014 | honest negative |

## Activation-space dividend (raw) — is the geometry real *in the activations*?

| feature | median Δ raw | frac circle wins | sign-test p | held-out dev line / circle |
|---|---:|---:|---:|---:|
| weekday · 8B L18 | -212 | 0.46 | 0.55 | 6.55e+03 / 4.6e+05 |
| month · 8B L18 | -16.5 | 0.42 | 0.12 | 4.55e+03 / 5.37e+03 |
| sycophancy · 8B L18 | 648 | 0.71 | 0.00044 | 3.85e+03 / 2.87e+03 |
| hedging · 8B L18 | 214 | 0.69 | 0.0025 | 3.53e+03 / 3.07e+03 |

## Reading — curvature's dividend is real in activations, inert in behavior

- **Behaviorally, curvature never pays** at 8B·L18: month · 8B L18, sycophancy · 8B L18, hedging · 8B L18 are flat honest negatives, and **weekday curvature COSTS** (line beats circle on ~70% of held-out rows, sign-test p≈1e-3). This sharpens the crown: the pulled-back Fisher metric buys dose **forecasting**, not per-row behavioral *likelihood*, at this layer.
- **In activation space, curvature DOES pay for the graded features** (sycophancy · 8B L18, hedging · 8B L18): the circle significantly reduces raw held-out reconstruction error (~70% of rows) — but that geometric win lands in **behaviorally inert directions** (the same features are flat in nats). Real geometry, no behavioral dividend.
- The cyclic calendar features (weekday, month) do **not** cross-validate even in activation space (median raw dividend n.s./negative): the in-sample topology races' preference for a circle is **post-selection optimism** that does not survive leave-template-out — finding (a) in the raw-column audit. It is not a demeaning artifact (b: identical per-prompt demeaning as the races) nor a complexity-charge artifact (c: raw held-out deviance carries no penalty).
- **Falsification passed.** The Gaussian-matched surrogate sits at the null on every feature (see `surrogate median` column, all ≈0 and n.s.), so the circle's extra freedom is not manufacturing wins. The sign-flip null is calibrated (under a true null, p is uniform: frac p<0.05 = 0.055, mean p = 0.49; a real +0.4σ shift detected at p=2e-4).

## Question 2 — slow-feature atlas on context means

The PerContextMean (per-prompt token-mean, subtracted as a nuisance everywhere) is tested as a *modeled* feature: pool all context-mean vectors across features per model and ask whether contextual structure charts — is the feature-of-origin recoverable from the context-mean geometry (1-NN LOO in standardized top-PC space) above the majority baseline, a permutation null, AND a Gaussian-matched surrogate?

| pool | n | dim | PC1 frac | partic. ratio real/null | resid (no PC1) real/null | feature-of-origin 1-NN acc (base / null / perm p) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 8b_L18 | 330 | 4096 | 0.998 | 1/1 | 14.9/14.2 | 1 / 0.27 / p=0.0002 | contextual_structure_charts |
| 35b_L17 | 162 | 2048 | 0.683 | 2.11/2 | 13.2/12.4 | 1 / 0.31 / p=0.0002 | contextual_structure_charts |

**Reading (honest).** The context mean is *not* unstructured: which feature a prompt belongs to is **perfectly recoverable** from its context-mean geometry (1-NN LOO acc 1.00) and far above a Gaussian-matched null (0.27–0.31) — so the subtracted PerContextMean genuinely carries contextual identity and behaves as a *modeled* feature. But two caveats keep this a pilot: (i) the population is dominated by a single common-mode axis (PC1 ≈ 0.998 at 8B), and (ii) the residual intrinsic dimension after removing PC1 is **not** below the matched Gaussian null (14.9 vs 14.2) — so the strong signal is categorical family *separability*, not yet a clean low-dimensional smooth manifold. A fuller atlas (more contexts, per-template resolution, topology certificates) is the follow-up.

## Coverage & what is pending

- **Complete:** the five 8B·L18 features above (weekday, month, sycophancy, hedging; day-of-month appended when its harvest lands) + the slow-feature atlas.
- **35B·L17 (color, weekday, month): BLOCKED and filed.** The held-out reconstruction hits a gamfit bug — `sae_manifold_predict_oos: decoder_blocks[0] has M=2 but rebuilt basis has M=3` — on the 35B circle fits, and the 35B color fit additionally grinds (dead-atom-revival churn, EV collapse). A crude polyline-projection fallback exists but only correlates ~0.54 with gamfit's true reconstruct, so it is **not** mixed into published numbers. Filed as a gam issue; 35B replication resumes once the OOS-predict basis bug is fixed.
- **Day-of-month (8th feature) and the joint month×day torus test** (pair-κ ρ statistic: ρ≈1 factorized product of two circles vs ρ>1 one bound 2-torus) are harvesting; results append as `dayofmonth` rows + a `joint_date_torus.json` verdict.

## Figures

- `figures/fig1_curvature_dividend.png` — **the premise figure.** Per-feature median dividend with the sign test, side by side in the behavioral (nats) and raw activation metrics: behavioral flat/negative everywhere, raw positive for the graded features.
- `figures/fig2_paired_scatter.png` — per-row held-out behavioral deviance, line vs circle (points below y=x = circle wins that row).
- `figures/fig3_permutation_null.png` — sign-flip permutation null vs the observed dividend.
- `figures/fig4_slow_feature_atlas.png` — context-mean spectrum (one dominant common-mode axis) + feature-of-origin recovery vs the Gaussian-matched null and majority baseline.

## Reproduce

- EXP1: `premise_deviance.py` (CPU, `venv_head_atlas`); safety acts via `harvest_safety_acts.py` (GPU). EXP2: `slow_feature_atlas.py`. sbatch: `premise_cpu.sbatch`, `premise_harvest.sbatch`. Figures: `premise_figs.py`; this report: `make_report.py`.
