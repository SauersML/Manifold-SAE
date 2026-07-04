# Compute-matched frontiers — curved refinement vs TopK-linear

**Reviewer condition 1**: publish *both* frontiers ourselves before the fight gets
framed as EV-per-FLOP only. This directory is that harness and its results.

Two frontiers, priced at **matched compute** (multiply-accumulates, analytic, training
+ inference — see `FLOP_ACCOUNTING.md`), across three lanes:

- **Curved chart (ours)** — `sae_manifold_fit`, circle atoms, `d_atom=1`, TopK gate.
- **Same-lane linear (control)** — `sae_manifold_fit`, `atom_topology="linear"` (affine
  `{1,t}`). Isolates *curvature* under one optimizer: the cleanest matched contrast.
- **Block/TopK linear** — `sparse_dictionary_fit`, the production block lane in TopK mode.

## The two claims under test

1. **EV vs FLOPs** — *curved refinement never loses explained variance at matched
   compute.* (Establish or refute.)
2. **bits/token vs FLOPs at matched distortion** — *curved wins description length.*
   Selection cost reported in both currencies: combinatorial `log₂ C(K, L0)` and
   empirical support entropy `H(S)` (the currency the MDL lane is migrating to).

Plus the honest cost line:

3. **Pure-linear DGP overhead** — on data with no curvature, curved refinement can only
   *lose* the selection/dictionary overhead. We report that number so the win is not
   oversold.

## Reproduce

```bash
# synthetic massive-K frontier (curved DGP), heavy-tailed firing, p up to 1024
python -m experiments.frontier_bench \
    --dgp curved --p 1024 --n 40000 --concepts 24 --firing-tail zipf --zipf-s 1.2 \
    --k 8 16 24 32 48 --active 4 --lanes linear curved manifold_linear \
    --out results/suite_2026-07-03/frontiers/synth_p1024_curved.json

# pure-linear overhead row
python -m experiments.frontier_bench --dgp linear ... \
    --out results/suite_2026-07-03/frontiers/synth_p1024_linear.json

# plots
python -m experiments.frontier_plots \
    --in results/suite_2026-07-03/frontiers/synth_p1024_curved.json \
    --out-dir results/suite_2026-07-03/frontiers/ --tag p1024_curved
```

Harness: `experiments/frontier_bench.py` (fits, FLOP model, MDL bits) and
`experiments/frontier_plots.py` (palette-matched frontier figures). Every fit runs in
an isolated subprocess; non-convergence/timeout is recorded as a miss, never crashes
the sweep.

## Files

| file | what |
|---|---|
| `FLOP_ACCOUNTING.md` | the MAC accounting, honestly (training-MAC is a charitable lower bound on the manifold lane) |
| `synth_*.json` | raw frontier results (per-lane, per-K: held-out EV, measured L0, realized decoder widths, FLOPs, MDL bits) |
| `frontier_*.png` | EV-vs-FLOPs, bits-vs-FLOPs, EV-vs-K panels |
| `real_l17_*.json` | real 35B L17 activations at K∈{4k,16k,32k} (block-lane EV from `../scale_evidence`; curved where feasible) |

## Results

**Real 35B L17 (block lane, DONE — EVs lifted from frame-health, see caveats).**
`real_l17_block.json`, `frontier_real_l17_block.png`. Block-sparse EV rises
0.707 → 0.906 → 0.990 across K∈{4k,16k,32k} capacity (2k/8k/16k blocks), 0 dead blocks,
L0 = 2 blocks (4 coords) / token. At matched distortion the **dictionary term dominates
description length** (~7,000 bits/token at 16k blocks, `K·block·p·16 / 150k` tokens): the
massive-K block dictionary buys EV with enormous per-token bits — exactly the regime a
*fewer-atom* curved code would win. But the curved (manifold REML) lane is a **small-K,
small-p tool** — ~70 s at p=12, but timing out (>15 min/fit) at p≥256 and OOM at p=1024
— so it does not reach K in the thousands with the current solver. Honest verdict:
**block and curved lanes are complementary** (block owns massive-K coverage; curved owns
small-K EV-per-atom + bits).

> **Caveats (do not bury; flagged by ATLAS2).** These EVs are *lifted* from the
> frame-health run, not re-fit here, so two comparability risks ride along: (1) the
> leading ~100k rows of `L17_train.f32.npy` are a **biased/ordered slice** (colmean ~15.2
> vs ~0.1 random) — if frame-health trained on leading-N, the EVs sit on a biased slice;
> an unbiased seeded-random-subsample recompute is owed (coordinate with ATLAS2), and
> whether the EVs replicate is a *finding*, not an assumption. (2) `tier0.json`'s mean is
> **stale** (~22% of energy is a spurious constant offset); tier0-zero EVs are inflated
> and NOT comparable — use `tier0_recentered.json` for any recompute and never mix
> conventions on one axis. Absolute EVs are convention-dependent until the unbiased
> recompute confirms replication.

**Synthetic frontier — the honest road, including a diagnosed dead end.**

Getting a *fair* curved-vs-linear synthetic frontier took two corrections, both recorded
here because they are the kind of thing a reviewer should see us catch ourselves:

1. **Multi-active DGP broke the fit (not curvature).** A first-round DGP with independent
   Bernoulli concept firing (`active_mean=4`, and even `active_mean≈1`) produces ~30%
   pure-noise (0-active) and ~27% multi-active tokens. A top-1 manifold SAE cannot fit
   that mixture: held-out EV collapsed to ~0.2–0.25 and went *negative* at larger K
   (`synth_p48_*.json`, `verdicts_p48_multiactive.md`). Proof this was the fit and not the
   data: on the identical data a **linear PCA rank-3 reconstruction reaches EV 0.73**
   (rank-6 → 0.995). The fix is `--exactly 1` (draw exactly one concept per token by
   heavy-tailed weight — matches the SAE's top-1 assumption; WHICH concept fires stays
   heavy-tailed). This is the regime where the manifold fit converges (standalone EV 0.99).

2. **Scale wall.** The manifold REML lane (`sae_manifold_fit`) is a small-p / small-K
   tool: raw p≥256 times out (>1200 s/fit) and p=1024 OOMs. Fix = `--pca-dim` (train-PCA-
   whiten to the informative subspace — the scale runs' PCA-128 recipe), so *ambient* p
   can be ≥1024 (data richness) while the SAE operates on a feasible `fit_dim`, priced at
   that dimension in the FLOP model.

### Synthetic verdict: the curved lane does NOT clear the bar here (falsified, gam#2132)

Even in the cleanest regime (single-active planted circles), the multi-atom manifold
dictionary **co-collapses** — it cannot be placed on a fair compute-matched frontier
against linear because it does not reach the linear reconstruction ceiling, let alone beat
it. Measured held-out EV (train `reconstruction_r2` in parens), `exact_p24_curved.json`,
`manifold_cocollapse_p24.png`:

| regime | linear PCA @rank | curved chart | same-lane linear |
|---|---|---|---|
| C=3, p=12, K=6 | ~0.73 | **0.23** (train 0.54) | — |
| C=6, p=24, K=6 | ~0.55 | **0.26** (train 0.37) | 0.25 (train 0.57) |
| C=6, p=24, K=8 | ~0.56 | **0.11** (train 0.46) | 0.25 (train 0.57) |

Three things, all reported not buried: (1) curved held-out EV **decreases** with K
(0.26→0.11) — dictionary co-collapse (`[#1026] restoring incumbent` in the inner-fit
logs); (2) a large train→held-out gap for **both** lanes (~0.55→~0.25) — the
out-of-sample encode path generalizes poorly; (3) curved atoms underperform even the
same-lane **linear** atoms on curved data, and both fall below a trivial linear-PCA
baseline that recovers 0.55–0.73 on the identical data. So the structure is trivially
recoverable; the gap is solver quality, filed as **gam#2132**.

**Consequence for the mission.** The compute-matched *curved-wins* frontier cannot be
demonstrated on synthetic with the current `sae_manifold_fit` dictionary solver — an
honest negative, not a claim that curvature has no value (the `K=1` single-chart fits
behind the dose-calibration crown are unaffected; the failure is specific to the
multi-atom dictionary sweep). The harness, FLOP accounting, and MDL-bits currency are
complete and correct, and will produce the frontier the moment the solver reaches the
linear ceiling. The **real-35B block-lane frontier above stands on its own** as the
delivered compute/bits frontier.

Jobs (checkpointed; `exact_p48`=12515810 raw p=48, `exact_p1024`=12515811 ambient
p=1024/PCA-24 heavy-tailed — both reproduce the same co-collapse as they land). First-round
`synth_p*` multi-active artifacts are kept as the record of the earlier diagnosed dead end.
