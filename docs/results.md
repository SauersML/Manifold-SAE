# Results

Empirical claims, with the commit / job where each was measured.
Updated when re-measured.

## Headline — real LM concept-encoding transfers (Qwen-0.5B, May 2026)

`experiments/llm_probe.py` Phase 2 with 80/20 train/test split. Pick each
architecture's best atom for a planted concept on 80% of prompts, evaluate
that atom on the held-out 20%. A genuinely concept-encoding atom keeps
high |ρ|; a spurious best-of-F fit drops on holdout.

**Result**: Manifold-SAE curve atoms transfer concept encoding 2× better
than vanilla atoms.

| concept × layer | vanilla holdout \|ρ\| | curve holdout \|ρ\| | curve advantage |
| --- | --- | --- | --- |
| magnitude_L20 | 0.38 | **0.81** | 2.1× |
| magnitude_L8  | 0.34 | **0.70** | 2.1× |
| magnitude_L16 | 0.16 | **0.22** | 1.4× |
| magnitude_L4  | 0.04 | **0.11** | 2.8× |

This is the architecture's qualitative test. At saturated layers (Qwen
L12, EV ≥ 0.989 for both architectures) matched-F MSE is uninformative,
but the train/test holdout shows real architectural discrimination —
curve atoms genuinely encode magnitude continuously, vanilla's best
"magnitude atom" is mostly a spurious fit.

Source: cluster jobs `baca6be80b66` (L12 probe) + `8f0b82d5bcbb` (L18
probe). Probe metric implemented in commit `06e312e`.

## Compactness of concept representation (concept localization)

For each (concept × layer) pair where Phase 1 detected 1D-manifold
structure, we count atoms whose Spearman with the concept rank exceeds
0.5.

| concept × layer    | vanilla atoms ≥ 0.5 | curve atoms ≥ 0.5 |
| --- | --- | --- |
| magnitude_L12      | 126 / 128           | **52 / 128**       |
| magnitude_L4       | 124 / 128           | **47 / 128**       |
| polarity_L8        | 126 / 128           | **49 / 128**       |
| polarity_L12       | 124 / 128           | **46 / 128**       |
| polarity_L16       | 126 / 128           | **48 / 128**       |
| time_L20           | 126 / 128           | **43 / 128**       |
| temperature_L8     | 126 / 128           | **49 / 128**       |
| temperature_L16    | 125 / 128           | **48 / 128**       |
| brightness_L8      | 124 / 128           | **48 / 128**       |

Vanilla atoms are *pluripotent*: 97-98% pick up at least faint signal
for every continuous concept tested. Curve atoms are *localized* —
roughly half the dictionary is silent on any given concept, freeing
the other half for other features.

This is the architecture landing in Bhalla et al. (2026)'s
"compact-capture" regime by construction — a small set of atoms acts
as a coordinate system for each manifold, instead of every direction
carrying weak alignment.

Renders via `tools/plot_atom_compactness.py`.

## Synthetic 1D-curve recovery, matched dictionary size

Source: `experiments/realistic_scaling.py`, run on cluster B200 after
the amp²·curve fix (commit `9f31143`). Smooth random curves planted in
`ℝ^D` with sparse activation. Matched F, matched TopK, matched
parameter count. Hungarian-matched per-feature Chamfer
(Frobenius-normalized).

| Scenario | D | F | Vanilla EV | Curve EV | Δ EV | Vanilla chamfer | Curve chamfer | Δ chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128 | 16 | 0.494 | **0.768** | **+0.274** | 0.221 | **0.145** | **−34%** |
| mid   | 256 | 32 | 0.513 | **0.760** | **+0.247** | 0.158 | **0.105** | **−34%** |
| large | 512 | 64 | 0.452 | **0.643** | **+0.191** | 0.160 | **0.103** | **−36%** |

When the GT really is a mixture of 1D manifolds, curve SAE wins by ~25
percentage points of EV and ~35% lower Chamfer (better shape recovery).

These numbers replaced earlier Path-A measurements (small: +0.174 →
+0.274 post-fix); the `amp²·curve` semantic in the pre-fix forward
was eating into both architectures' MSE comparison and partially
masking the architectural advantage.

## Layer-18 of Qwen-0.5B — breaks saturation

Layer 12 saturates at EV ≥ 0.989 for both architectures (the layer has
~4 effective directions, so matched-F MSE can't discriminate). Layer 18
of the same model is richer:

| F | top_k | vanilla EV | curve EV | locked EV | vanilla alive | curve alive |
| --- | --- | --- | --- | --- | --- | --- |
| 16  | 2 | 0.965 | 0.950 | 0.943 | 6  | **12** |
| 32  | 2 | 0.966 | 0.886 | 0.863 | 9  | 9      |
| 64  | 2 | 0.966 | 0.932 | 0.904 | 12 | **24** |
| 128 | 2 | 0.967 | 0.936 | 0.935 | 12 | **14** |
| 256 | 2 | 0.968 | 0.963 | 0.962 | 17 | 11     |

Curve SAE uses 2× more atoms at small-F (12 vs 6 at F=16; 24 vs 12 at
F=64). The architecture distributes structure across more atoms while
vanilla collapses to fewer features.

Source: cluster job `b8d07bc58a21`.

## Hyperparameter insight — lower n_basis unlocks more atoms

Sweep at Qwen-0.5B layer 12, F=128, TopK=2, varying `n_basis`:

| n_basis K | curve alive atoms | vanilla alive atoms |
| --- | --- | --- |
| 10 (default) | 5  | 4 |
| **4**            | **16** | 4 |

Dropping `n_basis` from 10 → 4 pulls curve alive from 5 to 16 with no
EV loss. Lower per-atom expressive capacity → more atoms get
productively utilized. Vanilla is unchanged (it has no `n_basis`).

This suggests the default `n_basis=10` was over-allocating capacity
per atom; future runs should default lower.

Source: cluster job `574bed523d7b`.

## Phase-1 manifold-existence test on Qwen-0.5B

`experiments/llm_probe.py` Phase 1: for each (concept × layer), check
if any of the top-8 PCs of the activation centroids has Spearman > 0.7
with the concept rank. Threshold 0.7 = "manifold lives here."

Across 6 concepts × 5 layers = 30 candidate pairs, **15 pass the
threshold** — Qwen-0.5B genuinely encodes these continuous concepts as
1D manifolds at the layers indicated:

```
magnitude: passes at layers 4, 8, 12, 16, 20  (best 0.916 at L12)
size:      passes at layers 8, 12               (best 0.750 at L12)
polarity:  passes at layers 8, 12, 16, 20       (best 0.818 at L16)
time:      passes at layer 20                   (best 0.849)
temperature: passes at layers 8, 16             (best 0.808 at L8)
brightness: passes at layer 8                   (best 0.808)
```

Phase 1 is independent of the SAE — pure PCA + Spearman on the raw
centroid activations. It establishes that the *substrate* for the
architectural claim (continuous concepts as 1D manifolds) exists in
this LM. Phase 2's holdout test then asks which SAE architecture
recovers those manifolds best.

## Cluster pipeline (May 2026)

Real-LM experiments run on an 8×B200 cluster via Heimdall. Submission
and observation scripts:

```
heimdall_jobs/submit.py             # POST job JSON, env-driven config
heimdall_jobs/status.py             # ps-style status table
heimdall_jobs/fetch_results.sh      # rsync JSONs + PNGs locally
tools/aggregate_results.py          # markdown overview across runs/
tools/plot_variant_sweep.py         # cross-variant comparison grid
tools/plot_atom_compactness.py      # the localization figure
tools/feature_dashboard.py          # top-firing tokens per atom by t_k
```

### Engineering hardening shipped during bring-up

Bringing the SAE up on cluster GPUs surfaced nine silent-failure modes;
each has a permanent fix (assertion + test) committed:

| # | Issue | Fix |
| --- | --- | --- |
| 1 | `gamfit.fit.fitted` is `by · (phi @ B)` (already amp-weighted) — don't multiply by mask_binary a second time | commit `9f31143`; self-test in `update_snapshot` |
| 2 | Per-batch soft-rescale stats differ at lock-and-cache | freeze `soft_min_locked` / `soft_max_locked` (`2ab3513`) |
| 3 | Float32 noise floor (≈1e-3 rel) — self-test tolerance | three-tier silent/warn/raise (`d459be1`) |
| 4 | torch ≥ 2.12 ships only +cu130 wheels | pin to <2.12 + explicit +cu128 PyTorch index, Linux-only marker (`87fa40c`) |
| 5 | `uv sync` fast-path kept stale wheels across lock changes | stamp `.venv/.heimdall_lock_hash` (`dd27b66`) |
| 6 | Silent CPU fallback when GPU requested | `MSAE_REQUIRE_CUDA=1` assertion (`4734461`) |
| 7 | gamfit dual-cuBLAS Python check raises by default | `manifold_sae/_cluster_bridge.py::bypass_gamfit_cuda_check()` (`c6f21e2`) |
| 8 | Position rescale dominated by non-firing tokens | firing-weighted soft-rescale via `logsumexp(β·z + log(w))` (`595e4e8`) |
| 9 | Eval-cache JSONs stale after forward-semantics fix | `forward_semantics: 2` stamped on signature (`796a930`) |

All nine are silent-failure modes — they degraded results without
crashing. Numbers reported above are after all nine fixes.

### Self-test in `update_snapshot`

After every snapshot, the SAE runs a forward pass in both training-mode
and locked-mode on the same batch and asserts they agree to within 5%
relative tolerance. Catches future regressions in the locked-forward
path immediately rather than in offline analysis. Issue #1 above would
have surfaced as a self-test failure with this check in place.
