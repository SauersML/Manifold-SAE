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

<!-- FILLED AFTER RUNS: headline EV-at-matched-FLOP verdict, pure-linear overhead number,
     bits crossover vs firing frequency, real-35B feasibility envelope, job IDs -->
_Runs in flight — verdicts and job IDs land here._
