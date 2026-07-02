# SAC scoreboard — continuous report

_Assembled continuously by WS-J from the committed results under `experiments/`.
Six headline figures (SAC_PLAN Part 4) plus throughput. Every number is pulled
from a committed result file; pending cells name the workstream that fills them.
Last refreshed: 2026-07-02 (S1 guard surgery landed; W8 real-model dose + W7
frontier probes committed)._

**Program status (STAGE1_DIAGNOSIS.md, supersedes SAC_PLAN Part 1).** The K>=2
failure was re-diagnosed as **miscalibrated supervision**, not joint-fit
architecture: the collapse bar (`0.5 x pca_ev_ceiling`, a dense rank-q reference)
is a category error against a k-active-sparse dictionary, evaluated even at
iteration 0 on the cold seed, walling the outer objective.

- **Stage 1 (guard surgery) — LANDED** (gam `8f14b403d`): absolute-degeneracy
  collapse floor, no iteration-0 reseed. Early corroboration: W7's REML
  `sae_manifold_fit` now returns **finite in-sample EVs** (weekday 0.525, month
  0.243, year 0.335, color 0.627 at n_iter=60) where the pre-S1 build recorded
  `NONCONVERGENCE_1784` / `OOM_KILLED`. The live-SAC scoreboard cells are now
  gated on the growth-mode **fit run on the post-S1 build**, not on Stage 1 itself.
- **Stage 2 (growth as production mode)** — Rust `fit_stagewise` driver +
  guards-off K=1 lane landed (gam `4b1aa0a6b`); evidence-raced per-atom births
  make EV monotone in K by construction. WS-A owns the live run.

**Pending metric-meaning sweep (MATH_REVIEW.md Section E, owner M-bench).** Several
eval relabels are queued and will touch cells 1/3 when M-bench lands: "matched-K"
-> matched effective-width/L0 (E2), train-PCA "upper bound" -> `train_pca_reference`
+ a test-PCA oracle ceiling (E7), probing F1 -> `oracle_best_f1` (E5). This
report already uses the honest framings (parity is labeled a *parameter-accounting
upper bound*, W7 compares curved(1) vs linear(1)/linear(2) at matched intrinsic
width); the cells will be re-worded to M-bench's final vocabulary on landing.

## Scoreboard

| # | figure | headline (committed) | source | pending -> filled by |
|---|--------|----------------------|--------|----------------------|
| 1 | **parity** — composed held-out EV >= external TopK at matched K/L0 | curved ceiling **0.994** at Theta=108 vs linear-dict saturation **0.974**; ~0.020 EV inaccessible to any linear budget under top-1 routing (parameter-accounting bound) | `frontier_out/report.md` s2-3 | live composed held-out EV vs external TopK on real shards — **WS-A growth fit (post-S1) + WS-C**; label sweep per MATH_REVIEW E2/E7 |
| 2 | **structure** — typed atoms; dEV monotone in births | evidence-limited curved cutoff **K=3** recovers planted curved count 3; noise-floor global-dEV **5e-4**; per-atom circle-vs-line gap 0.047-0.057 vs 4-5e-4 | `frontier_out/results.json` s3 | dEV monotone in K *by construction* (Stage-2 growth run) + shipped type/Theta/band/id-report per atom — **WS-A / WS-J T2 tier** |
| 3 | **semantics** — calendar recovered unsupervised, correct cyclic order | clean: real **Qwen2.5-0.5B** weekday adj **0.71 (5/7)**, month **0.83 (10/12)**, curved(1)=linear(2). Frontier **Qwen3-32B** (committed): **color cyclic ordering adj=1.000** recovered unsupervised; weekday/month noisier at 7-12 tokens; REML EV 0.24-0.63 | `probe_out/`, `frontier_probe_out/summary.md` | larger per-feature token counts to de-noise frontier CV; year open-interval atom — **WS-D bigger probe harvest / WS-A** |
| 4 | **control** — dose slope ~ 1, tight R^2 | **REAL Llama-3.1-8B (layer 16)**: chart `predicted_nats` slope **0.908**, R^2 **0.951**, median meas/pred **0.881** (unbiased); the bare-linear-latent baseline is mis-calibrated **~10x** (median ratio 10.0, mean\|log\| 2.38 vs chart 0.52) | `dose_real_out/report.md` | (done — real model; teacher stand-in slope 0.847/R^2 0.807 was the prior placeholder) |
| 5 | **stability** — subspace agreement >> latent, hashed | deterministic linear tier: latent cos **1.000**, union-span cos **1.000**, **hashes identical** across seeds 0/1/2; aux random-tiling latent 0.845 / subspace 0.833 (byte-unstable) | `stability_out/seed_stability_table.md` | curved-SAE subspace>>latent gap on post-S1 SAC output — **WS-A + WS-F #8** |
| 6 | **disentanglement** — absorption/SCR beat matched linear SAE | — | — | SAEBench subset + absorption/SCR, composed vs T1-only ablation (*the disentanglement delta is the thesis*) — **WS-F #10** |
| 7 | **throughput** — T1 hours; T2 <= day; encode >= 1e5 rows/s | — | — | T1 K=32k x 50M wall-clock — **WS-C #3**; encode rows/s + fallback by freq decile — **WS-E #15**; harvest live (Qwen3-32B, 667k+ tok/layer) — **WS-D** |

## Headline: control is now real (the single most important figure)

SAC_PLAN calls real-model dose calibration "the single most important figure in
the program." It landed (WS-F W8, `dose_real_out/`): on **Llama-3.1-8B-Instruct**
layer-16 weekday-token activations, a K=1 circle chart's `predicted_nats` —
computed from the chart's attached downstream output-Fisher metric *before* the
edit — predicts the **measured** output KL from actually patching the forward pass
with slope **0.908** and R^2 **0.951** over n=288 (atom, base, dose, sign) points.
It is unbiased (median measured/predicted 0.881). The task baseline — a bare
linear SAE latent scaled by matched push-norm, carrying no metric — is
mis-calibrated by ~10x (median ratio 10.0). This is the curved atom's payoff made
concrete: calibrated dosing falls out of the chart itself.

## The load-bearing "before" evidence (why Stage 1 was needed)

The whitened-convergence probe on **real OLMo-3-7B-Instruct**
(`whitened_convergence_results.json`, layers 15-17, anisotropy 6.8x): every K>=2
joint fit returned `NONCONVERGENCE_1784` or `OOM_KILLED`, attributed to the guard
stack executing the cold seed against an unreachable dense-EV bar at iteration 0.
Stage 1 guard surgery (landed) targets exactly this; W7's now-finite REML EVs are
the first post-surgery evidence the fault line was supervision, not architecture.
Figures 1, 2, 5 still show the linear / K=1 / accounting half until the post-S1
growth fit runs.

## Detail per figure

**1 - Parity.** `frontier_out` (planted real-shaped synthetic, p=9, 3 curved + 3
linear): linear/sparse measured live; curved side is an honest parameter-accounting
**upper bound** against measured per-atom geometry (pre-S1 the joint solver OOM'd).
Durable point survives: under top-1 routing a straight atom cannot trace a circle,
so ~0.020 EV is inaccessible to linear at any budget. Live parity vs external TopK
is the post-S1 WS-A/WS-C deliverable. (MATH_REVIEW E2/E7 will relabel the bound.)

**2 - Structure.** Evidence-limited curved cutoff falls exactly at the circle/line
boundary (global dEV 0.047-0.057 circles, 4-5e-4 lines), recovering the planted
curved count unsupervised. Stage-2 growth makes dEV monotone in births by
construction; the artifact T2 tier (WS-J) ships each atom's topology, curvature
Theta, shape band, and identifiability report.

**3 - Semantics.** Two committed views. Clean (real **Qwen2.5-0.5B**): a single
curved coordinate reconstructs weekday/month as well as two linear PCs and
expresses the cyclic wrap (adjacency 0.71 / 0.83). Frontier (real **Qwen3-32B**,
WS-D harvest, `frontier_probe_out/`): the win is uneven at these tiny per-feature
token counts (7-15) — **color recovers perfect cyclic ordering (adjacency 1.000)**
where a single linear PC scores 0.000, and in-sample month curved(1) 0.192 beats
linear-1PC 0.125; weekday/month held-out CV is noisy and does not cleanly beat
linear-1PC. REML `sae_manifold_fit` now returns finite EVs (0.24-0.63). Honest
read: the curved advantage is real and feature-specific (cyclic color/month), and
de-noising the frontier CV needs larger probe harvests. Year uses a circle atom on
a monotone arc (no rank-1 open-interval manifold in the torch backend yet).

**4 - Control.** See headline above. Real Llama-3.1-8B; slope 0.908, R^2 0.951,
unbiased; bare-linear baseline off ~10x. `dose_real_out/report.md`.

**5 - Stability.** The `dictionary_artifact` v1 content hash (ported to Python,
byte-validated against the Rust hash — the same port WS-J's artifact layer reuses)
makes seed agreement provable: the deterministic linear tier is byte-identical
across seeds. The curved-SAE subspace>>latent figure needs convergent multi-seed
manifold fits on the post-S1 build — WS-F #8.

**6 - Disentanglement / 7 - Throughput.** Not yet measured; downstream of the
composed dictionary + scale harvest (Qwen3-32B, fineweb, layers 24/32/40, 667k+
tokens/layer, live on node2).

## T0 data-plane contract (WS-D, matched)

WS-D harvest manifests (node2 `/dev/shm/sauers_gpu/harvest/`, Qwen3-32B,
`d_model=5120`, layers 24/32/40). Finalized manifests carry a `t0` block
(`compute_t0`): per-dim `mean`/`std`/`rms`, `scale_median_std`, and a nested
`rogue_dims` massive-activation block (`{"index":[...],"rms":...,"rms_over_median":
...,"mad_z":...}`, rule `rms>5*median_rms OR MAD-z>8`; ~376 dims at L24). WS-J's
`load_t0_from_manifest` passes it through verbatim (provisional manifests now also
carry T0) and folds `rogue_dims.index` + per-dim scale stats into the content
hash. Verified against a real node2 manifest. See `ARTIFACT_SCHEMA.md`.

## Artifact / verbs status (WS-J)

| deliverable | state |
|---|---|
| `ARTIFACT_SCHEMA.md` (contract other WS emit into) | **done** — `/Users/user/gam/ARTIFACT_SCHEMA.md` (T0 matches WS-D finalized format) |
| `TieredArtifact` (T0 u T1 u T2 u Sigma u encoder u hash), save/load, content hash | **done** — `examples/tiered_artifact.py`; save==reload verified; T2 hash mirrors `dictionary_artifact.rs`; 8-check selftest green |
| loaders for WS-A / WS-C / WS-E / WS-D formats | **done** — `load_sac_result`, `load_tier1_artifact`, `load_encoder`, `load_t0_from_manifest` (defensive; absent tier recorded, never faked) |
| `fit` / `encode` / `steer` / `diff` verbs | **done** — fit (T1 + SAC/growth T2), encode (amortized + exact-solve certificate fallback, hash-bound), steer (ManifoldSAE.steer / W8 dose), diff (per-tier hash + Hungarian latent + principal-angle subspace) |

Sigma (structured-residual whitening) stays stubbed `pending` until WS-A exposes it
as a standalone Python object; encoder tier loads WS-E's bundle when present.
Nothing in WS-J requires a Rust build.
