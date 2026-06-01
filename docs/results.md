# Results

Empirical claims, with the commit / job where each was measured.
Updated when re-measured.

## Real-LM headline — see `docs/findings.md`

The current, honest real-LM assessment is in `docs/findings.md`: after
fixing a late-discovered per-dim normalization bug, Manifold-SAE wins
only at Qwen-0.5B small F and loses to vanilla TopK SAE at every model
with D ≥ 1536, and real concept manifolds have intrinsic dim 2.4–3.4
(too high for a 1D curve atom).

> **Retracted.** An earlier "headline" here reported curve atoms
> transferring concept-encoding ~2× better than vanilla (holdout |ρ|
> 0.81 vs 0.38) plus a concept-localization / compactness table
> (vanilla 124–126 / 128 vs curve 43–52 / 128). Both were measured on
> a contaminated rank-1 preprocessing path (cluster jobs `baca6be80b66`
> / `8f0b82d5bcbb`) and were **not reproduced after the normalization
> fix**. Do not cite those numbers; see `docs/findings.md`.

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
centroid activations — so these numbers are unaffected by the
normalization bug and stand. But a single PC correlating with the
concept rank only shows the concept has a 1D *projection*, not that the
manifold is 1D. The later intrinsic-dimension measurement in
`docs/findings.md` (Grassberger-Procaccia correlation dim 2.4–3.4 for
most of these concept×layer pairs) shows the underlying manifolds are
2–3D, not 1D — which is why the 1D curve-atom architecture underperforms
on real LM residuals at scale.

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
crashing. (A tenth, later-discovered issue — scalar→per-dim std
normalization — is the one that invalidated the retracted real-LM
numbers; see `docs/findings.md`.)

### Self-test in `update_snapshot`

After every snapshot, the SAE runs a forward pass in both training-mode
and locked-mode on the same batch and asserts they agree to within 5%
relative tolerance. Catches future regressions in the locked-forward
path immediately rather than in offline analysis. Issue #1 above would
have surfaced as a self-test failure with this check in place.
