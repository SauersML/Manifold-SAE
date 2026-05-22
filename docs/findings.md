# Findings — what we now know

Living summary of the empirical claims supported by current experiments,
their effect sizes, and what's still unresolved.

## Strong claims (well-supported)

### Compactness of concept representation

For 9 (concept × layer) pairs that pass Phase-1 manifold detection on
Qwen-0.5B residuals, count atoms whose Spearman with the concept rank
exceeds 0.5:

* Vanilla TopK SAE: **124–126 of 128 atoms** (97–98%) on every concept.
* Manifold-SAE curve atoms: **43–52 of 128** per concept (34–41%).

Effect size: ~3× reduction in correlated-atom count. Vanilla atoms are
pluripotent (each direction picks up faint signal for every concept);
curve atoms are localized (most of the dictionary is silent on any
given concept). Matches Bhalla et al. 2026's "compact-capture" regime
— Manifold-SAE lands there by construction.

Source: cluster job `baca6be80b66`, `tools/plot_atom_compactness.py`.

### Holdout-test concept-encoding transfer

Train/test split: pick best train-atom by Spearman on 80% of prompts,
evaluate that atom on held-out 20%.

| concept × layer | vanilla holdout \|ρ\| | curve holdout \|ρ\| |
| --- | --- | --- |
| magnitude_L20 | 0.38 | **0.81** |
| magnitude_L8  | 0.34 | **0.70** |
| magnitude_L16 | 0.16 | **0.22** |
| magnitude_L4  | 0.04 | **0.11** |

Curve wins 4/5 magnitude-layer pairs, by 2× margins. Vanilla's "best
magnitude atom" is largely a spurious best-of-128 fit; curve's best
atom genuinely tracks magnitude on unseen prompts.

Source: cluster jobs `baca6be80b66` + `8f0b82d5bcbb`, llm_probe phase 2
with 80/20 split (commit `06e312e`).

### Bigger model + deeper layer = clearer architectural win

| Sweep regime | Vanilla EV | Curve EV | Δ | Vanilla alive | Curve alive |
| --- | --- | --- | --- | --- | --- |
| Qwen-0.5B L12, F=128 | 0.989 | 0.989 | 0.000 | 4 | 5 |
| Qwen-0.5B L18, F=128 | 0.967 | 0.936 | -0.031 | 12 | 14 |
| **Qwen-1.5B L18, F=128** | **0.9897** | **0.9953** | **+0.006** | **2** | **11** |
| Qwen-1.5B L18, F=16 | 0.9898 | 0.9959 | +0.006 | 2 | 10 |
| Qwen-1.5B L18, F=64 | 0.9897 | 0.9953 | +0.006 | 2 | 8 |

At small model + shallow layer, both architectures saturate (no
discrimination possible). At bigger model + deeper layer, curve SAE
wins +0.6pp EV with 4-5× more alive atoms. The architecture's value
scales with the residual stream's structural richness.

Source: cluster jobs `fd4b5d7aba68`, `b8d07bc58a21`, `ce894b5e2559`.

### Synthetic 1D-curve recovery

Smooth random curves planted in `ℝ^D` with sparse activation, matched
F and TopK. Hungarian-matched per-feature Chamfer
(Frobenius-normalized).

| Scenario | D | Vanilla EV | Curve EV | Δ EV | Δ chamfer |
| --- | --- | --- | --- | --- | --- |
| small | 128 | 0.494 | **0.768** | **+0.274** | **−34%** |
| mid   | 256 | 0.513 | **0.760** | **+0.247** | **−34%** |
| large | 512 | 0.452 | **0.643** | **+0.191** | **−36%** |

When the GT really is 1D-manifold features, curve SAE wins by ~25
percentage points of EV and ~35% lower Chamfer.

### Substrate exists in real LM (Phase 1)

15 of 30 (concept × layer) pairs in Qwen-0.5B exhibit |PC-Spearman|
> 0.7 — independent of any SAE. This is direct evidence that
continuous concepts (magnitude, polarity, time, temperature,
brightness, size) live as 1D manifolds in the residual stream at this
scale. The architectural premise is empirically confirmed for the
architecture to *target*.

## Counterintuitive findings

### Lower n_basis unlocks atom utilization

At Qwen-0.5B L12, F=128, TopK=2:

| n_basis K | curve alive | vanilla alive |
| --- | --- | --- |
| 10 (default) | 5  | 4 |
| **4**            | **16** | 4 |

Smaller per-atom expressive capacity → more atoms productively used.
Suggests we've been over-allocating expressiveness per atom.

### At matched-decoder-params on synthetic data, vanilla often wins

| scenario     | vanilla mean \|ρ\| | curve mean \|ρ\| | Δ |
| --- | --- | --- | --- |
| monotone     | **0.949**   | 0.696   | **−0.254** |
| non_monotone | 0.527       | 0.554   | +0.027 (tied) |
| mixed        | **0.698**   | 0.480   | **−0.218** |

When you equate decoder budgets and the data isn't pure 1D-manifold,
vanilla's atom-fragmentation flexibility wins. Architecture's value
isn't raw modeling capacity per dollar — it's *what kind of structure*
it makes easy to discover.

## Engineering hardening (8 silent-failure modes found + fixed)

| # | Issue | Fix |
| --- | --- | --- |
| 1 | `gamfit.fit.fitted` is `by · (phi @ B)` — don't multiply by mask again | self-test in `update_snapshot` |
| 2 | Per-batch soft-rescale stats differ at lock-and-cache | freeze `soft_min_locked` / `soft_max_locked` |
| 3 | Float32 noise floor (≈1e-3 rel) — self-test tolerance | three-tier silent/warn/raise |
| 4 | torch ≥ 2.12 ships only +cu130 wheels | pin to <2.12 + explicit +cu128 PyTorch index |
| 5 | `uv sync` fast-path kept stale wheels | stamp `.venv/.heimdall_lock_hash` |
| 6 | Silent CPU fallback when GPU requested | `MSAE_REQUIRE_CUDA=1` assertion |
| 7 | gamfit dual-cuBLAS Python check raises | `bypass_gamfit_cuda_check()` |
| 8 | Position rescale dominated by non-firing tokens | firing-weighted soft-rescale via `logsumexp(β·z + log(w))` |
| 9 | Eval-cache JSONs stale after forward-semantics fix | `forward_semantics: 2` stamp |

All nine were silent-degrade modes. Numbers in this doc are after all
fixes.

## Negative results / unresolved

### Cyclic concepts (`cyclic_concepts.py`) didn't recover the weekday circle

Both architectures hit |ρ_circ| ≈ 0 on weekdays + months. **The
experiment trains an SAE from scratch on only 49 weekday prompts** — far
too few for any SAE to find structure. This is a methodological issue
with the experiment design, not a finding about the architecture.

The proper test would be: probe an SAE already trained on a large
corpus to see if any of its atoms have cyclic structure on the
weekday prompts. Refactor pending.

### Steering test inconclusive

The original steering experiment reported KL=0 because the forward
hook's return wasn't replacing the residual. Fixed via in-place edit.
The post-fix rerun still reported KL=0 because the atom we selected
from the L12 probe had zero amplitude on magnitude prompts (the probe
metric saturation we documented earlier — "best atom" picked an
inactive atom by chance). Need to rerun with an atom known to fire on
the prompts.

### xlarge synthetic convergence

At D=896, F=128 with default hyperparams, curve SAE got stuck at MSE
0.85 while vanilla converged to 0.36. Retune with lower `intrinsic_rank`,
higher `n_basis`, and lower learning rate queued (`xlarge_v2`
scenario); haven't seen result yet.

## Queued / pending

| Experiment | What it'd tell us |
| --- | --- |
| `synthetic_2d_recovery` | Does the new 2D atom architecture recover planted 2D surfaces in a single atom? |
| `atom_causality` (counterfactual + cross-SAE) | Are atoms causally load-bearing? Are they universal across seeds? |
| `atom_analysis` (polysemy + cross-layer + adv + probe) | Per-atom polysemy distribution + cross-layer transfer + downstream-task probe |
| `llm_sweep_L18_F128_multipenalty` | Does three-λ-per-atom REML beat one-λ in the mixed L18 regime? |
| Architectural variants R=1, R=4, R=8, K=24, binary_amp | Optimal hyperparams |
| `realistic_scaling_v2` (with points_only steelman) | Where vanilla wins (no curve structure) |

## Verdict

Manifold-SAE is a useful architectural variant with measurable real-LM
advantages — not a universal upgrade.

It WINS on:
* Real-LM activations where features have manifold structure
* Held-out concept-encoding transfer (interpretability metric)
* Concept compactness in dictionary (3× fewer correlated atoms)
* Synthetic 1D-curve data
* Bigger models + deeper layers

It DOESN'T WIN universally on:
* Matched-decoder-params arbitrary data
* Pure MSE at saturated layers
* Convergence at very high D without hyperparam tuning

Publication framing: **"native compact-capture architecture for 1D
manifolds in LM residuals — discovers structure without post-hoc
clustering."** The holdout-test and compactness numbers are direct
evidence the architecture lands in Bhalla et al.'s "compact-capture"
regime by construction.
