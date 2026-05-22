# Experiments

Each driver in `experiments/` is a self-contained Python module. Most
have a `Config` dataclass at the top with env-var overrides for cluster
runs. Edit defaults, or import and call `main(Config(...))`
programmatically, or set env vars and `python -m experiments.<name>`.

## Real-LM concept probes

### `experiments/llm_probe.py` — manifold detection + atom interpretability

The primary architectural test. Two phases:

**Phase 1** — Plant 6 continuous concepts (magnitude, size, polarity,
time, temperature, brightness) via prompt templates. For each
(concept × layer) pair, harvest the LM's residual at the answer-position
token, PCA to 64-dim, compute Spearman correlations between the top-8
PCs and the concept rank. If `|ρ| > 0.7` the concept lives as a 1D
manifold at that layer.

**Phase 2** — Load trained SAE checkpoints. For each architecture and
each (concept × layer) pair that passed Phase 1, identify atoms whose
signal (activation magnitude for vanilla; position `t_k` for curve)
correlates with the concept rank. Reports several metrics including
80/20 train/test holdout (best train-atom's |ρ| on held-out 20%) —
this is the saturation-free metric that discriminates architectures.

Reads `MSAE_MODEL`, `MSAE_LAYER`, `MSAE_SWEEP_DIR` (where to find the
SAE checkpoints), `MSAE_PROBE_RESULTS` (optional path to a prior probe
run's results.json for atom selection). Output: `phase1_heatmap.png`
plus per-(concept × layer) `phase2_*.png`.

**Result (May 2026)**: At Qwen-0.5B layer 18, curve atoms' holdout |ρ|
for magnitude beats vanilla by 2.1× (0.81 vs 0.38). 15 of 30
(concept × layer) pairs pass Phase 1 — Qwen really does encode these
concepts as 1D manifolds.

### `experiments/llm_sweep.py` — head-to-head sweep with both architectures

Train vanilla TopK SAE + Manifold-SAE side-by-side at matched F across
a range of dictionary sizes. Includes lock-and-cache verification
(self-test in `update_snapshot` asserts locked-mode = training-mode on
the snapshot batch).

Env knobs (all optional):
- `MSAE_MODEL` (default `Qwen/Qwen2.5-0.5B`)
- `MSAE_LAYER` (default 12)
- `MSAE_F_VALUES` (comma-separated, default `16,32,64,128,256,512`)
- `MSAE_TOPK_MIN`, `MSAE_TOPK_RATIO` (sparsity tier)
- `MSAE_N_BASIS`, `MSAE_INTRINSIC_RANK` — curve architecture
- `MSAE_CONTINUOUS_AMP` (default 1; set 0 for binary amp)

Per-F outputs: `eval_F{F}.json` (cached result), checkpoints
`vanilla_F{F}.pt` + `curve_F{F}.pt`. Final outputs: `results.json`
plus `pareto.png`, `alive.png`, `curves.png`, `positions.png`,
`intrinsic_dim.png`, `per_pc.png`.

### `experiments/cyclic_probe.py` — Engels-style weekday probe on a pre-trained SAE

Replaces the earlier `cyclic_concepts.py` (which tried to train an SAE
from scratch on 49 weekday prompts — far too little data). Loads a
pre-trained SAE (one trained on wikitext at the target layer) and
probes whether any of its atoms encode the cyclic weekday/month
structure that Engels et al. 2024 and Wurgaft et al. 2026 recover
post-hoc via cubic-spline fit through centroids.

Reports Mardia circular Spearman correlation between atom positions
and the cyclic GT result-class index. Plot: per-class centroids in
top-3 PCs alongside the best curve atom's `g_k(t)` lifted into the
same basis.

### `experiments/steering_causality.py` — causal-intervention test

Pick a curve atom with high holdout |ρ| for a concept (e.g.
magnitude). For prompts that prime that concept, modify the atom's
`t_k` to a swept value, patch the modified reconstruction back into
the LM's residual at the chosen layer, run the forward pass through,
compare top-token probabilities and KL divergence to the baseline.

A causally effective atom produces ordered output shifts as `t_k` is
swept. A merely correlative atom produces noise.

## Synthetic-data benchmarks

### `experiments/realistic_scaling.py` — matched-F head-to-head

Smooth random curves planted in `ℝ^D` with sparse activation. Three
core scenarios (small/mid/large) plus xlarge and points_only
steelman scenarios. Both SAEs get matched F = #GT_curves and matched
TopK. Hungarian-matched per-feature Chamfer (Frobenius-normalized).

**Result (cluster B200, post-fix)**:

| Scenario | D | F | Vanilla EV | Curve EV | Δ EV | Vanilla chamfer | Curve chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | **+0.274** | 0.221 | **0.145** |
| mid   | 256 | 32 | 0.513 | **0.760** | **+0.247** | 0.158 | **0.105** |
| large | 512 | 64 | 0.452 | **0.643** | **+0.191** | 0.160 | **0.103** |

Curve wins by ~25pp EV and ~35% Chamfer at matched dictionary size.

### `experiments/continuous_recovery.py` — matched-decoder-params test

Falsifiability test: instead of matched F, match TOTAL DECODER
PARAMETER COUNT (vanilla gets R × F_curve atoms). For each GT 1D-curve
with a planted scalar latent z ∈ [0, 1], compute |Spearman(atom_signal, z)|
where atom_signal is TopK activation (vanilla) or t_k position (curve).

Three regimes:
- `monotone`: C(z) = z · v
- `non_monotone`: C(z) = cos(2πz) · v₁ + sin(2πz) · v₂ (1D loop in 2D)
- `mixed`: half each

The non-monotone case is the sharpest test of the architecture's
claim — a scalar activation can't encode a non-monotone function, so
vanilla SAE atoms can only approximate the loop with multiple atoms.

### `experiments/synthetic_recovery.py` — five planted curves in ℝ^64

Smaller, faster sanity check. Line, parabola, ramp_exp, logmap, sqrt
planted via random orthogonal projection. Procrustes-aligned
visualization of learned vs planted curves.

### `experiments/llm_like_stress_test.py` — point + curve mixture

Mix of GT point atoms and GT curve atoms in high-D ambient. Tests
whether Manifold-SAE handles a realistic distribution of feature types
— some genuinely 1D-manifold, some single-direction.

## Tools and analysis

### `tools/feature_dashboard.py` — top-firing tokens sorted by t_k

The qualitative interpretability test. For each alive curve atom in a
trained checkpoint, finds the top-50 firing tokens in a corpus and
sorts them by `t_k`. If the atom encodes a continuous feature, the
sorted token list reads as a semantic gradient (small → large for
magnitude; negative → positive for polarity).

Reads `MSAE_CHECKPOINT`, `MSAE_MODEL`, `MSAE_LAYER`. Emits markdown
table per alive atom + JSON sidecar.

### `tools/plot_atom_compactness.py` — concept-localization figure

Reads any `llm_probe` `results.json` and renders the count-of-atoms-
above-threshold bar chart used in the compactness finding.

### `tools/plot_variant_sweep.py` — cross-variant grid plot

Aggregates all `runs/llm_sweep*/results.json` files into one figure
comparing vanilla EV, curve EV, locked EV, and alive-atom counts
across F for each architectural-variant run.

### `tools/aggregate_results.py` — unified markdown overview

Walks a `runs/` tree, reads every recognized JSON schema, emits one
markdown table per experiment type.

## Cluster submission

`heimdall_jobs/submit.py` POSTs job JSON to a Heimdall-style scheduler.
Reads `HEIMDALL_API`, `MSAE_WORKING_DIR`, `MSAE_NODE`, `MSAE_GPUS` from
env or `~/.config/manifold-sae/heimdall.json`. Supports `--depends-on`
(chain jobs), `--sweep-dir-of` (point a probe at a sweep's
checkpoints), `--env KEY=VALUE` (per-job env overrides for variant
sweeps).

`heimdall_jobs/status.py` — ps-style table; `--watch <seconds>` for
live refresh. `heimdall_jobs/fetch_results.sh <run-name>` — rsync
JSONs + PNGs locally (skips multi-GB checkpoints).

## What's not yet wired

- 2D feature manifolds (tensor-product smooths) — Manifold-SAE atoms
  are fundamentally 1D per atom; multi-dimensional manifolds (grids,
  cylinders from the Bhalla et al. 2026 ICLR tasks) currently require
  multiple atoms.
- Periodic Duchon basis through lock-and-cache (gamfit supports
  periodic, not yet plumbed through `update_snapshot`).
- Manifold-CLT-style posterior intervals on `t_k`.
- Multi-layer cross-layer SAEs.
- Steering on AxBench tasks at scale.
