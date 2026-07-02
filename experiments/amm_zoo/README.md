# amm_zoo — the Appendix-H replication/beat benchmark

Goodfire's *Block-Sparse Featurizers* validates on **Additive Manifold Mixtures**
(AMMs, their Def 2.1) and reports per-factor reconstruction R². This benchmark
**replicates that number and then goes one rung further**: on a *zoo* of planted
topologies it asks not just "how much variance does the featurizer reconstruct?"
but "did it recover the RIGHT geometry?" — coordinate fidelity, topology
identification, intrinsic dimension, and description length. The block featurizer
answers *"subspace"* for every factor; a **chart** featurizer reads the topology
off the manifold and **denoises curved factors onto their manifold** as noise
grows.

## The corpus (`amm.py`)

AMM Def 2.1 in `R^128`: each token is a sparse additive sum of `k=3` per-factor
manifold contributions plus isotropic noise, over a 24-factor **zoo**:

| topology | count | intrinsic `d_i` | ambient `b` | note |
|---|---|---|---|---|
| circle | 8 | 1 | 2 | `r[cosθ, sinθ]` |
| torus | 4 | 2 | 4 | `2-in-4` |
| sphere | 4 | 2 | 3 | `2-in-3` |
| arc | 4 | 1 | 2 | open (boundary), never wraps |
| helix | 2 | 1 | 3 | **non-closed** curved 1-D (`cos,sin,pitch·t`) |
| mobius | 2 | 2 | 3 | orientation-**reversing** strip (half-twist `θ/2`) |
| linear | 4 | 2 | 2 | Gaussian **control — curvature must NOT win here** |

Helix and Möbius are the cases where topology typing matters most: a non-closed
curved 1-D and an orientation-reversing 2-D surface both look like a "subspace" to
a block featurizer. **Intrinsic** (topological) and **embedding-span** (ambient
`b`) dimension are scored separately per factor — e.g. helix is intrinsic-1 but
embedding-3.

- Noise `σ ∈ {0.02, 0.05, 0.1, 0.2} × signal-RMS` (matched absolute floor across
  splits); `n = 200k / 50k` train/test; **5 seeds**.
- A **subspace-coherence** knob mixes each factor frame `V_g` toward a shared
  subspace; the achieved **minimum principal angle** between factor subspaces is
  measured and stored (67° orthogonal → 14° entangled).
- Ground truth (`V_g`, per-token intrinsic coords, active membership, radii,
  topology) is saved **for scoring only**; dense contributions are recomputed on
  demand, never stored.

## The arms (`arms.py`), matched budget + matched L0

All arms share the true generative decoder-scalar budget (`Σ_g b_g·d`) and the
same active-code count `L0`. bsf.py (review-verified) is imported, never
reimplemented.

- `topk_sae` — TopK-SAE, `b=1` directions.
- `bsf_vanilla` — vanilla Block-Sparse Featurizer, `b=2`.
- `bsf_grassmann` — Grassmannian BSF, `b=2` (tied `γ`, Stiefel decoder).
- `sasa` — **Subspace-Aware SAE** (arXiv 2606.06333), the LLM-side cousin: free
  encoder + **learned decoder subspaces** (per-block Stiefel reprojection every
  step) + block-level sparsity. Subspace-aware but no tied encoder, no curvature.
- `ours` — Grassmannian **block T1** (== `bsf_grassmann`) **→ per-block K=1 circle
  chart**: classify each block's code cloud (ring vs blob by relative radial
  spread), and on ring blocks **denoise onto the fitted circle** (radius fixed,
  angle kept) — but only when a held-in guard shows it lowers train MSE, so the
  chart can never hurt the linear control. The direction ⊂ block ⊂ chart ladder.

## The metrics (`metrics.py`), Hungarian-matched, held-out

Recovered → true factors matched by maximum total contribution R² (compact
no-SciPy Kuhn–Munkres). Per matched pair: **contribution R²** (the replication
number), **coordinate circular-corr** (circle/arc), **geodesic-Spearman**
(recovered-code Euclidean vs true geodesic distance), **topology-ID accuracy**,
**dimension-estimation accuracy**, and **MDL bits/token** (`mdl_ladder.score_json`
at the task noise floor `δ² = d·σ²`).

Every recovered structural factor (circle/arc/torus/sphere) is gated by a
**matched permutation null** on its geodesic-Spearman (permute the recovered
coordinate's token alignment — preserves both marginals, destroys the
correspondence, the `matched_null.py` discipline): a factor is only "recovered"
at `p < 0.05`.

## Headlines

1. **R²-vs-σ crossing curves** per topology (`r2_vs_sigma.png`): the chart denoises
   curved factors onto the manifold, so `ours` pulls ahead of the block arms on
   circles/arcs as σ grows, while staying tied on the **linear control** (no false
   win — the guard).
2. **Topology-ID table** (`topology_id.png`): the BSF baselines score chance
   ("subspace" for everything); `ours` identifies circles/arcs.

## Run

```bash
VENV=/Users/user/gam/.venv/bin/python
# full node run: 200k/50k, 5 seeds, 4 sigmas, 4 arms (each cell isolated + retried)
$VENV run.py --full     # -> results.json, r2_vs_sigma.png, topology_id.png, REPORT.md
# quick local validation:
$VENV run.py --quick    # -> results_quick.json (smaller n/steps)
```

Each `(seed, σ, arm)` cell runs in its own subprocess with a wall-clock timeout
and retries (the box has an OOM reaper); `results*.json` is written incrementally
after every cell, so an interrupted sweep resumes with all completed cells intact.

## Extending with new arms (schema stability)

The output schema is **arm-additive**: each `(seed, σ, arm)` is an independent
cell, and every arm is just a name in `arms.ARMS` that returns a list of
`metrics.RecoveredFactor` (contribution, coord, active, topology + intrinsic +
embedding dim, `n_params`). A future arm — e.g. gamfit's forthcoming
`atom_topology="linear_block"` (γ_g(t)=t·D_g, BSF as a literal special case of the
manifold SAE) — slots in by adding its name and a `RecoveredFactor` producer;
existing cells need **no rescoring**, and it is compared on the same Hungarian +
MDL + null battery as every other arm.
