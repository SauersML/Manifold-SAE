# Post-hoc analyses of the crown data (all reproducible from data/dose_calibration_real.json)

## Summary statistics (as computed by code/crown_plots.py)

```
manifold                   n= 168 slope=0.940 R2=0.994 median_ratio=1.081 mean|log ratio|=0.195
linear_norm                n= 168 slope=0.951 R2=0.800 median_ratio=0.157 mean|log ratio|=1.952
linear_fisher              n= 168 slope=1.057 R2=0.986 median_ratio=1.099 mean|log ratio|=0.205
manifold_within_validity   n=   7 slope=0.934 R2=0.919 median_ratio=1.057 mean|log ratio|=0.204
manifold_heldout           n=  84 slope=0.945 R2=0.999 median_ratio=1.098 mean|log ratio|=0.143

manifold: KL spans 3.31e-05 .. 0.36 nats (10821x range)
within factor-2 of prediction: 97.0% of edits ; within factor-1.5: 91.1%
linear_norm within factor-2:   10.7%
linear_fisher within factor-2: 92.9%
```

## Forecast-hierarchy check (theory falsification)

Prediction under test: linear+Fisher (base-point metric, no curvature) should degrade
quadratically with dose while the chart's path integral stays calibrated.

Median |log(measured/predicted)| per dose fraction:

```
 frac  | chart  | linear+Fisher | ratio(fisher/chart)
 0.005 | 0.089  | 0.090         | 1.01x
 0.01  | 0.125  | 0.106         | 0.84x
 0.02  | 0.154  | 0.094         | 0.61x
 0.05  | 0.185  | 0.106         | 0.57x
 0.1   | 0.157  | 0.100         | 0.64x
 0.2   | 0.132  | 0.070         | 0.53x
 0.4   | 0.154  | 0.080         | 0.52x
```

VERDICT: falsified at these dose scales — linear+Fisher does NOT degrade up to 40% ‖h‖
(it is modestly better than the chart's path integral at large fractions). What the data
does establish: metric ≫ no-metric (6× absolute calibration gap), and the chart's
path-integral error does not grow with dt (0.163 small-dt → 0.132 at dt>1.5 rad),
i.e. it stays calibrated through the wrap. The curvature-error regime needs larger arcs.

## Tangent-column units bug

At dt→0 the tangent and path-integral forecasts must coincide; the data shows
ratio pathint/tangent ≈ 3.9–4.9 CONSTANT (fitted scaling exponent of (ratio−1) vs dt:
b = 0.026, not the theoretical 2). Also, rescaled by its small-dose constant, the
tangent column lies exactly ON the path-integral curve, including the wrap fold —
because it is computed on the true chord (m = chord of the on-chart move), inheriting
the wrap geometry. Conclusion: `predicted_nats_tangent` is a chord-quadratic with an
inconsistent constant, NOT a flat-space forecast. The honest flat-space arm in this
dataset is `linear_fisher`.

## Raw-data views: bugs found & fixed while building them

1. **Label inference:** prompts are template-major (`build_prompts`: for template →
   for word), so day labels are `tile(arange(7), 10)`, NOT `repeat`. A block-label
   mistake groups by TEMPLATE and manufactures a fake "Monday outlier" cluster.
2. **Rogue dims:** in the demeaned data, dims 2276 and 233 have day-range 1900 and 375
   vs median 0.48 — classic rogue dimensions; naive full PCA gives PC1 = 99.7% variance
   and hides the week. Views drop dims with range > 50× median (none remain after
   correct labeling + template exclusion at the centroid level; the two above only
   dominate under wrong labels).
3. **Outlier template:** template 0's demeaned Frobenius norm is 2.9× the median of the
   other nine (11562 vs ~4100) and is excluded from the raw-plane view (disclosed).
4. After fixes: the top-2 PCs of the 7 day-centroids carry 57.9% of day-to-day
   structure and the raw scatter shows the week as a cycle in calendar order (fig7).

## Fit-seed fragility (from the job log)

Seeds 891 (n_iter=40) and 992 (n_iter=60) failed REML convergence — the outer
probe-refusal non-termination guard fired after 25 consecutive infeasible cost probes
(|g|≈1.3e3 vs ceiling 5.0) and correctly aborted; seed 1093 (n_iter=80) converged to
r²=0.997 in 183 s. Related: PCA-compressed fits (rank 48) of the same data grind in the
#813/#821 outer cost-stall class while raw-ambient fits converge — compression worsens
conditioning here, contrary to the usual intuition.
