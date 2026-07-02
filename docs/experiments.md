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

**Phase 1 result**: 15 of 30 (concept × layer) pairs have a top-PC
projection that correlates with the concept rank (|ρ| > 0.7). Note that
a 1D projection does not imply a 1D manifold — the intrinsic-dimension
follow-up in `docs/findings.md` shows these concepts are 2–3D.

The earlier Phase 2 holdout claim (curve beats vanilla 0.81 vs 0.38 on
magnitude) was measured on a contaminated preprocessing path and has
been retracted; see `docs/findings.md`.

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
probes whether any of its atoms encode cyclic weekday/month structure
that Engels et al. 2024 and Wurgaft et al. 2026 recover post-hoc via
cubic-spline fit through centroids. In this repo's current reporting,
weekday should be treated as a fragile probe; month/null-robust claims
need the template-transfer and null checks documented in the chart
transfer report.

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

## Joint manifold-recovery verification gate

The canonical multi-atom recovery objective is the first-class joint solve
`gamfit.sae_manifold_fit` (canonical `assignment="ibp"` — adaptive count, true
zeros — not `softmax`+`top_k`). Its verification harness is two files:

### `experiments/manifold_recovery.py` — the gate

Three checks, each wrapped so a crash/non-convergence reports cleanly:

1. **K=2 superposed-circle recovery under IBP** — PASS if reconstruction R² > 0.9.
2. **Incoherence ON vs OFF (headline)** — sweep coherence; at each level fit with
   `decoder_incoherence_weight` ON (1.0) vs OFF (0.0). ON must *raise* the
   recovered-tangent σ_min (better-conditioned per-token split) **and** *lower* the
   cross-atom decoder cross-Gram ‖B₀B₁ᵀ‖_F (more incoherent decoders) or improve
   coordinate recovery. The incoherence knob is the separability lever (gamfit
   #671); the harness resolves its name against the live signature and self-gates
   BLOCKED if absent.
3. **Single-atom out-of-class specification margin (K=1, runs today)** — let model
   evidence pick a topology from the menu, then flag a 2D blob mis-fit as a circle
   via an absolute out-of-class margin (R²_patch − R²_circle) calibrated against a
   true-circle null.

### `experiments/manifold_falsifier.py` — keystone falsifier + shared scoring

Plants two circles with two orthogonal knobs — **coherence** (planes share a tangent
direction; → colinear makes the split ill-posed) and **coverage** (co-active fraction
+ disambiguating-only tokens) — and scores per-token coordinate recovery up to each
circle's isometry (`circ_procrustes_r2`), plus the **σ_min identifiability metric**
(`tangent_sigma_min`: smallest singular value of the stacked active-atom tangent
frame; 0 iff the split is underdetermined). `--selftest` proves the scoring is
isometry-invariant, split-sensitive, and that σ_min decreases monotonically as planes
go colinear — i.e. the scoring is trustworthy *before* the fit unblocks.

**Status.** The multi-atom (K ≥ 2) joint fit currently diverges upstream (the
cold-start assignment logits init to a uniform symmetric saddle; fix in progress), so
checks 1–2 self-gate BLOCKED and the single-atom check (3) runs today. The harness is
correct and goes green once the solver fix and the incoherence knob land. The broader
objective also includes nuclear-norm embedding-rank selection (#672), ScadMcp
non-convex sparsity, and the isometry gauge + gauge-conditional topology evidence
(#673), with per-atom uncertainty (posterior shape bands, mean ± sd) and a typical
coordinate range on the fit result.

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
- Multi-layer cross-layer SAEs.
- Steering on AxBench tasks at scale.

Now landing via the joint `sae_manifold_fit` objective (no longer deferred):

- Per-atom uncertainty — posterior shape bands (curve mean ± sd) and a typical
  coordinate range exposed on the fit result, superseding the deferred
  Manifold-CLT-style `t_k` interval estimate.
- Topology discovery — gauge-conditional topology model-evidence (#673) selects
  per-atom topology from a menu rather than requiring a declared cyclic/non-cyclic.
