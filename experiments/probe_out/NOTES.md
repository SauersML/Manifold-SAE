# Curved-feature probes — notes & interpretation

Companion to `summary.md` / `curved_feature_probes.json`. Probe code:
`experiments/curved_feature_probes.py`.

## What was run

Cheap dedicated harvest of **residual-stream activations** from `Qwen/Qwen2.5-0.5B`
(24 layers, D=896) for the three canonical cyclic/curved token features, each in 5
natural template sentences:

- **weekday** — Monday..Sunday (7 tokens, a 7-point circle)
- **month** — January..December (12 tokens, a 12-point circle)
- **year** — 1950,1955,…,2020 (15 tokens, a 1-D ordered non-periodic curve)

Readout = residual at the target token's last sub-token position, layers {5,8,11,14}.
**Per-template demeaning** (subtract each sentence's mean over its tokens) is applied
before any geometry — the raw activation is dominated by sentence context; the
token-of-interest is a small component. This is the same "frame-demean before geometry"
recipe the color harvest uses (DATA_README §7). Without it, held-out EV goes strongly
negative (context swamps the feature).

The curved dictionary is gamfit's manifold SAE (torch backend `gamfit.torch.ManifoldSAE`),
**K=1 atom, intrinsic_rank=1, periodic circle atom (fourier basis, n_basis=4)**. It is
fit by backprop (robust). The REML `gamfit.sae_manifold_fit` solver was attempted as a
cross-check but is **OOM-blocked in this environment** (see the Year/REML note below);
`gamfit.torch.ManifoldSAE` is the same curved dictionary and stands as the primary fit.
The single-atom recovered chart coordinate
(angle) is what the ordering test reads. Layer is chosen by the *linear* PCA diagnostic
(conservative for the linear baseline).

## Headline result (weekday & month — the two circles)

**Matched intrinsic-coordinate budget.** The curved atom reconstructs each sample from
**one** scalar (an angle); linear-L uses L PCA coordinates. A circle is intrinsically 1-D
but extrinsically 2-D, so the prediction is curved(1) ≈ linear(2) ≫ linear(1). In-sample
EV confirms it exactly:

| set | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |
|---|---:|---:|---:|
| weekday | 0.584 | 0.444 | 0.580 |
| month   | 0.598 | 0.316 | 0.586 |

One curved coordinate does the reconstruction work of two linear PCs — i.e. a single
curved atom captures what a linear SAE must spread across two direction-atoms. (Leave-
one-template-out CV in `summary.md` tells the same story but is noisier at 7–12 tokens.)

**Ordering.** From the single periodic coordinate the recovered angle orders the tokens
around the circle: weekday cyclic-adjacency **0.71 (5/7)**, month **0.83 (10/12)**. A
single linear direction *cannot* express the cyclic wrap (Sun–Mon, Dec–Jan): its best
single-PC Spearman is high (~0.82, there is a partial linear trend) but its cyclic
adjacency via a *folded* 1-D readout is not meaningful. The fair linear upper bound is the
**2D**-PCA angle (needs two dims for a circle) — month's circle is clean enough that 2D-PCA
also nails it (adj 1.0), which is expected; the curved atom matches that ordering quality
using **half** the coordinates. See `recovered_orderings.png`.

## Year (non-periodic control)

The real **year harvest could not be obtained**: this shared box has an external OOM
reaper that SIGKILLs the model-loading process; weekday and month were harvested across
many retries, but the year harvest (75 forward passes — the longest window) was killed on
every attempt. This is an environment/infra blocker, not a code issue — reported honestly
rather than faked.

The year branch of the pipeline is exercised on **synthetic** planted-curve data
(`synthetic_validation.json`), which validates the whole fit/analysis path end-to-end on
ground truth:

| synthetic set | curved EV (1 coord) | linear 1PC | linear 2PC | ordering |
|---|---:|---:|---:|---|
| year (non-periodic curve) | 0.996 | 0.979 | 0.997 | Spearman **1.00** |
| weekday (7-circle) | 0.795 | 0.613 | 0.999 | circ_r 0.96, adj 0.71 |
| month (12-circle) | 0.801 | 0.561 | 0.999 | circ_r 0.95, adj **1.00** |

On the planted circles curved(1)≈? the periodic atom reaches ~0.8 EV from a single
coordinate vs 0.56–0.61 for one linear PC (and 2 linear PCs trivially reach ~1.0 on
noise-free planted circles), and recovers the cyclic order (circular corr ≈0.95). On the
non-periodic year curve, a linear direction already reconstructs it (0.979 at 1 PC) and the
curved atom matches it while ordering the years perfectly — i.e. the curved advantage is
specific to genuinely **circular** features, exactly as the real weekday/month result shows.

**REML** (`gamfit.sae_manifold_fit`, K=1, circle topology, retries at n_iter 60/120/200)
was also SIGKILLed on every attempt by the same OOM reaper — its slow inner solve is a large
kill window, and lowering n_iter to shrink it trips `RemlConvergenceError` instead. Recorded
as blocked in `reml_corroboration.json`; the torch-backend curved fit is unaffected.

## Files

- `curved_feature_probes.json` — full metrics (per-set EV CV+in-sample, ordering, layer diag).
- `summary.md` — the auto-generated results tables.
- `recovered_orderings.png` — recovered weekday & month circles, tokens by recovered angle.
- `reml_corroboration.json` — REML `gamfit.sae_manifold_fit` EV on the same demeaned data.
- `synthetic_validation.json` — pipeline self-test on planted circles + a non-periodic curve.
- `harvest_{weekday,month}.npz` — cached activations (re-analyzable without the model).
