# SAC scoreboard — continuous report

_Assembled continuously by WS-J from the committed results under `experiments/`.
Six headline figures (SAC_PLAN Part 4) plus throughput. Every number is pulled
from a committed result file; pending cells name the workstream that fills them.
Last refreshed: 2026-07-02._

**Program revision (STAGE1_DIAGNOSIS.md, supersedes SAC_PLAN Part 1).** A deep
code read + numeric computation of the guard thresholds on the repo's own OLMo
fixtures re-diagnosed the K>=2 failure: the joint optimizer is probably fine — it
is being executed by its own **miscalibrated supervision**. The collapse bar
(`collapse_ev_bar = 0.5 x pca_ev_ceiling`, dense rank-q reference) is a category
error against a k-active-sparse dictionary, evaluated even at iteration 0 on the
cold seed, and it flattens the outer objective via the wall cost. So:

- **Stage 1 (CRITICAL PATH, ~100-line diff) — guard surgery**: restore the
  collapse trigger to absolute degeneracy (EV<=eps AND co-vanished decoders); any
  surviving data bar must be **sparsity-matched** (`fraction * EV(sparse_dictionary_fit)`);
  never evaluate a bar at iter 0 / without stalled-progress; restrict the outer
  WALL to non-finite probes + absolute degeneracy; replace the seed-startup 0.10
  floor with best-of-candidates + finiteness. Acceptance = **program ignition**:
  exact W6 repro (OLMo-3-7B, top-128, K=8) completes with a real EV in minutes;
  W5 compose emits a composed EV; the K=3 planted coin-flip becomes deterministic.
- **Stage 2 (revises SAC) — growth as production mode**: the stagewise engine
  already exists (`structure_harvest`); the change is an **inversion of mode**, not
  a new subsystem — evidence-raced per-atom births from the whitened residual
  factor, Sigma refreshed between births, warm-started backfits, one terminal
  joint pass for evidence/certs/`dictionary_artifact` hash. Growth makes EV
  monotone in K by construction even if Stage 1 fully revives the joint fit.

Every "live SAC run" pending note below is therefore now **gated on Stage 1
guard surgery landing** (program ignition), then filled by the Stage-2 growth-mode
fit. The pending cells are the program's actual deliverable; they are named.

## Scoreboard

| # | figure | headline (committed) | source | pending -> filled by |
|---|--------|----------------------|--------|----------------------|
| 1 | **parity** — composed held-out EV >= external TopK at matched K/L0 | curved ceiling **0.994** at Theta=108 vs linear-dict saturation **0.974**; ~0.020 EV inaccessible to any linear budget under top-1 routing | `frontier_out/report.md` s2-3 | live composed held-out EV vs external TopK on real shards — **Stage 1 ignition -> WS-A growth fit + WS-C** |
| 2 | **structure** — typed atoms; dEV monotone in births | evidence-limited curved cutoff **K=3** recovers planted curved count 3; noise-floor global-dEV **5e-4**; per-atom circle-vs-line gap 0.047-0.057 vs 4-5e-4 | `frontier_out/results.json` s3 | dEV monotone in K *by construction* (Stage-2 growth) + shipped type/Theta/band/id-report per atom — **WS-A / WS-J T2 tier** |
| 3 | **semantics** — calendar recovered unsupervised, correct cyclic order | **weekday** cyclic-adjacency **0.71 (5/7)**, **month 0.83 (10/12)**; one curved coord = two linear PCs (weekday EV 0.584 vs lin-1PC 0.444 / lin-2PC 0.580; month 0.598 vs 0.316 / 0.586). Real Qwen2.5-0.5B | `probe_out/NOTES.md`, `curved_feature_probes.json` | frontier-model (Qwen3-32B L24/32/40 harvests on node2) calendar + year + REML cross-check — **WS-F #7** (probe harvests done, #12) |
| 4 | **control** — dose slope ~ 1, tight R^2 | chart `predicted_nats` **unbiased**: slope **0.847**, R^2 **0.807**, median meas/pred **1.103**, over ~4 decades of KL. Teacher head | `dose_out/report.md` | real-model dose on OLMo-3 (single most important figure) — **WS-F #6**, node2 GPU |
| 5 | **stability** — subspace agreement >> latent, hashed | deterministic linear tier: latent cos **1.000**, union-span cos **1.000**, **hashes identical** across seeds 0/1/2; aux random-tiling latent 0.845 / subspace 0.833 (byte-unstable). `dictionary_artifact` v1 hash port validated | `stability_out/seed_stability_table.md` | curved-SAE subspace>>latent gap on SAC output (manifold fit non-convergent pre-Stage-1) — **WS-A + WS-F #8** |
| 6 | **disentanglement** — absorption/SCR beat matched linear SAE | — | — | SAEBench subset + absorption/SCR, composed vs T1-only ablation (*the disentanglement delta is the thesis*) — **WS-F #10** |
| 7 | **throughput** — T1 hours; T2 <= day; encode >= 1e5 rows/s | — | — | T1 K=32k x 50M wall-clock — **WS-C #3**; encode rows/s + fallback by freq decile — **WS-E #15**; harvest done (Qwen3-32B, 667k+ tok/layer) — **WS-D #11-13** |

## The load-bearing "before" evidence (why Stage 1 exists)

The whitened-convergence probe on **real OLMo-3-7B-Instruct** activations
(`whitened_convergence_results.json`, layers 15-17, top-48 PCA, anisotropy 6.8x)
is the failure in one file: every K>=2 joint fit returns `NONCONVERGENCE_1784` or
`OOM_KILLED`. STAGE1_DIAGNOSIS attributes this to the guard stack executing the
cold seed against an unreachable dense-EV bar at iteration 0 and walling the outer
objective — not to the joint-fit architecture. Seed-stability and the frontier
independently record the same non-convergence. So figures 1, 2, 5 currently show
the linear / K=1 / accounting half; **Stage 1 guard surgery is what promotes them
to live composed measurements** (acceptance = W6 repro completes in minutes).

## Detail per figure

**1 - Parity.** `frontier_out` (planted real-shaped synthetic, p=9, 3 curved + 3
linear): linear/sparse dictionaries measured live; the curved side is an honest
parameter-accounting **upper bound** against measured per-atom geometry (pre-Stage-1
the joint solver OOM'd). Durable point survives the caveat: under top-1 routing a
straight atom cannot trace a circle, so ~0.020 EV is inaccessible to linear at any
budget — exactly what curved atoms recover. Live parity vs external TopK is gated
on Stage 1.

**2 - Structure.** Evidence-limited curved cutoff falls exactly at the circle/line
boundary (global dEV 0.047-0.057 circles, 4-5e-4 lines), recovering the planted
curved count unsupervised. Stage-2 growth makes dEV monotone in births by
construction; the artifact T2 tier (WS-J) ships each atom's topology, curvature
Theta, shape band, and identifiability report — the typed-atom contract in
`ARTIFACT_SCHEMA.md`.

**3 - Semantics.** Cleanest real-data headline today. On real Qwen2.5-0.5B a
single curved coordinate reconstructs a weekday/month token as well as two linear
PCs and, unlike any single linear direction, expresses the cyclic wrap (Sun-Mon,
Dec-Jan): adjacency 0.71 / 0.83. Year control validated on synthetic (Spearman
1.00) after the real year harvest was OOM-reaped. Frontier repro (Qwen3-32B probe
harvests weekday/month/year/color at L24/32/40 already on node2) is WS-F #7.

**4 - Control.** The chart carries an output-Fisher metric and `steer`
path-integrates it to predict an intervention's output shift in nats before the
edit; unbiased across ~4 decades (slope 0.847, R^2 0.807, median ratio 1.10). A
bare linear SAE latent carries no metric and is mis-calibrated ~3x. Teacher head;
one-line real-model swap queued as WS-F #6.

**5 - Stability.** The `dictionary_artifact` v1 content hash (ported to Python,
byte-validated against the Rust hash — the same port WS-J's artifact layer reuses)
makes seed agreement provable: the deterministic linear tier is byte-identical
across seeds. The curved-SAE subspace>>latent figure needs convergent multi-seed
manifold fits, which arrive with Stage 1 + Stage-2 growth — WS-F #8.

**6 - Disentanglement / 7 - Throughput.** Not yet measured; downstream of the
composed dictionary + scale harvest. The WS-D scale harvest exists (Qwen3-32B,
fineweb, layers 24/32/40, 667k+ tokens/layer on node2:/dev/shm/sauers_gpu/harvest/).

## T0 data-plane contract (WS-D, now matched)

WS-D's harvest manifests (node2 `/dev/shm/sauers_gpu/harvest/`, Qwen3-32B,
`d_model=5120`, layers 24/32/40) are the T0 source. Finalized manifests carry a
`t0` block (`compute_t0`): per-dim `mean`/`std`/`rms`, `scale_median_std` (robust
whitening reference), and a nested `rogue_dims` = **massive-activation dims**
(`{"index":[...], "rms":..., "rms_over_median":..., "mad_z":...}`, rule
`rms>5*median_rms OR MAD-z>8`; ~376 dims at L24). WS-J's `load_t0_from_manifest`
passes this block through verbatim and folds `rogue_dims.index` + per-dim scale
stats into the content hash; provisional manifests (`stats:{mean,norm}` only) fall
back cleanly. Verified against a real node2 manifest. See `ARTIFACT_SCHEMA.md`.

## Artifact / verbs status (WS-J)

| deliverable | state |
|---|---|
| `ARTIFACT_SCHEMA.md` (contract other WS emit into) | **done** — `/Users/user/gam/ARTIFACT_SCHEMA.md` (T0 matches WS-D finalized format) |
| `TieredArtifact` (T0 u T1 u T2 u Sigma u encoder u hash), save/load, content hash | **done** — `examples/tiered_artifact.py`; save==reload hash verified; T2 hash mirrors `dictionary_artifact.rs` (order/scale/reflection invariant) |
| loaders for WS-A / WS-C / WS-E / WS-D formats | **done** — `load_sac_result`, `load_tier1_artifact`, `load_encoder`, `load_t0_from_manifest` (defensive; absent tier recorded, never faked) |
| `fit` verb | **done** — drives T1 + SAC/growth T2 -> artifact dir (`fit_artifact` / CLI) |
| `encode` verb | **done** — amortized encoder + exact-solve certificate fallback, hash-bound; fallback rate by freq decile |
| `steer` verb | **done** — pass-through to `ManifoldSAE.steer` (W8 dose machinery) |
| `diff` verb | **done** — per-tier hash deltas + Hungarian latent match + principal-angle subspace agreement (W9 harness) |

Sigma (structured-residual whitening) is stubbed `pending` until WS-A exposes it
as a standalone Python object; encoder tier loads WS-E's bundle when present.
Nothing above requires a Rust build.

## Environment note (local fit blocker)

The curved SAC `fit` OOM-kills locally (exit 137) even at n=150/p=8 — the same
memory blocker every workstream hit with `sae_manifold_fit`, and consistent with
STAGE1_DIAGNOSIS (the guard/wall loop burns to timeout/OOM on real-shaped data).
The `fit` plumbing is verified correct (drives the real fitter through live
per-birth EV traces; T1 + serialization + diff fully exercised with a real
`sparse_dictionary_fit`: deterministic decoder, hashes identical across seeds,
reload-stable). Live curved composition belongs on node2, post Stage 1.
