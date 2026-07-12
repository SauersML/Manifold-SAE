# AMM zoo — all-topology Manifold-SAE benchmark

This benchmark plants 28 sparse additive manifold factors in `R^128`, fits five
featurizers at a common routing budget, Hungarian-matches recovered additive
contributions to ground truth, and scores held-out geometry as well as
reconstruction.

| topology | factors | intrinsic dim | embedding span |
|---|---:|---:|---:|
| circle | 8 | 1 | 2 |
| torus | 4 | 2 | 4 |
| sphere | 4 | 2 | 3 |
| arc | 4 | 1 | 2 |
| helix | 2 | 1 | 3 |
| Möbius strip | 2 | 2 | 3 |
| linear control | 4 | 2 | 2 |

Every token activates exactly three factors. Train and test splits are generated
independently; planted frames, memberships, and intrinsic coordinates are used
only by the scorer.

## Arms

- `topk_sae`: signed scalar TopK SAE.
- `bsf_vanilla`: free-encoder block-sparse featurizer, block width 2.
- `bsf_grassmann`: tied Grassmann/Stiefel BSF, block width 2.
- `sasa`: free encoder plus genuinely Stiefel-projected decoder blocks.
- `ours`: production `gamfit.sae_manifold_fit`, with `K=28`,
  `atom_topology="auto"`, `assignment="topk"`, and `top_k=3`.

`ours` is not a ring heuristic. Its topology, coordinates, assignments, decoder,
and per-atom held-out reconstructions all come from the native Manifold-SAE fit
and frozen-decoder OOS solve. The scorer also honors the model's persisted hybrid
curved/linear verdicts.

## Metrics

Recovered factors are matched to planted factors by maximum total contribution
R². The report includes:

- held-out contribution R²;
- circular correlation and manifold-distance Spearman correlation;
- topology, intrinsic-dimension, and embedding-span accuracy;
- a matched token-permutation null for each non-linear structure claim;
- description length in the sibling `mdl_ladder` scorer.

The R² kernel uses sparse planted support sufficient statistics. It does not
materialize the former 1.43-GB list of full true-factor tensors at the full
setting. The permutation null ranks each pairwise-distance graph once and then
relabels it in batches.

## Warm/resumable execution

One subprocess owns all requested arms for a `(seed, noise)` group. It warms
Torch and optimizer kernels once, generates the dataset once, checkpoints after
every arm, and retries only missing/failed arms. This removes 80 of the former
100 cold Torch startups in the full sweep. An existing results file is resumed
only when its complete scientific config matches; `--fresh` is the explicit
destructive start-over switch.

Quick validation:

```bash
PY=/path/to/current/gamfit-venv/bin/python
$PY run.py --quick --fresh --profile --out ./quick_out
```

Full run:

```bash
PY=/path/to/current/gamfit-venv/bin/python
$PY run.py --full --profile \
  --out /projects/standard/hsiehph/sauer354/amm_zoo/all_topologies \
  --scratch "${SLURM_TMPDIR}/amm_zoo"
```

Rerun the same full command without `--fresh` to resume. Focused baseline checks
can omit the native arm explicitly, for example:

```bash
$PY run.py --quick --fresh --no-figures \
  --arms topk_sae,bsf_vanilla,bsf_grassmann,sasa \
  --out ./baseline_check
```

`--profile` writes one `.prof` file per warm `(seed, noise)` worker under
`OUT/profiles/`. Each result cell also records generation, fit, OOS extraction,
scoring, null, and total wall times in JSON.

## MSI

The checked-in [`msi.sbatch`](msi.sbatch) uses the project standard root
`/projects/standard/hsiehph/sauer354`, requires the manifest-pinned gamfit wheel
to match the source checkout commit, and runs against node-local scratch. Submit
it through the wrapper:

```bash
~/msi-node/msi sub msi.sbatch /projects/standard/hsiehph/sauer354/Manifold-SAE/experiments/amm_zoo
```

Artifacts are `results*.json`, `REPORT.md`, `r2_vs_sigma.png`,
`topology_id.png`, and `profiles/*.prof`.
