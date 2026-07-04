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

**Synthetic massive-p frontier (in flight on MSI, curved vs same-lane linear):**

| job | what | out |
|---|---|---|
| `12501065` | calibration p=256 N=6k K={8,24} | `scratch/fr_calib_p256.json` |
| `12501080` | p=48  N=8k  K={6,12,18,24}, curved+linear DGP | `synth_p48_{curved,linear}.json` |
| `12501209` | p=256 N=4k  K={8,16,24,32}, curved+linear DGP | `synth_p256_{curved,linear}.json` |
| `12501212` | p=1024 N=2.5k K={8,16,24,32}, curved+linear DGP | `synth_p1024_{curved,linear}.json` |

Verdicts (EV-at-matched-FLOP, bits crossover, pure-linear overhead) are computed by
`experiments/frontier_analyze.py` and land here as each job completes. A small local
`p=16` curved-vs-linear frontier is included as an immediately-reproducible smoke.
