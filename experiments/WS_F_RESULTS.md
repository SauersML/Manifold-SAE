# WS-F — Science battery: results index

Status as of the pre-guard-fix build (gamfit 0.1.247 PyPI / gam_fable @ 6350c1d).
Reruns against S1-guards' patched build are staged and monitored.

## W8 — Real-model dose calibration  ✅ DONE (the headline figure)
`dose_real_out/` — `dose_calibration_real.{png,json}`, `report.md`; code
`dose_calibration_real.py`.

Llama-3.1-8B, layer 16, weekday calendar **circle** chart; predicted path-integrated
output-Fisher nats (`steer.predicted_nats`) vs **measured** next-token output KL from a
patched forward pass. 288 points:

| method | R² | log-slope | median meas/pred |
|---|---:|---:|---:|
| **manifold chart** | **0.952** | 0.911 | **0.882**  (**0.999** within validity radius) |
| linear-norm (no metric) | 0.952 | 0.964 | **10.24** |
| linear + base-point Fisher (fair ref) | 0.928 | 0.940 | 1.28 |

The chart supplies and path-integrates an output-Fisher metric, so calibrated dosing in
nats falls out of the SAE atom itself; a bare linear latent is ~10× miscalibrated.
Impl: downstream output-Fisher harvested at the last token position only (identical
G_n, ~5× cheaper); fit+steer in a PCA-reduced subspace, δ mapped back to full space for
the patch (exact under an orthonormal basis). Month/color pending the patched build
(pre-fix, the 12-token loop auto-grows a K=1 request into co-collapsed circles).

## W7 — Frontier calendar/color probes  ✅ DONE
`frontier_probe_out/` — code `curved_probes_frontier.py`. Qwen3-32B (d=5120), layers
{24,32,40}, WS-D harvest.

REML curved **1-coordinate** EV ≈ linear **2-PC** EV on all four features — the
intrinsic-dimension thesis at frontier scale:

| feature | REML curved(1) | linear 1-PC | linear 2-PC | ordering |
|---|---:|---:|---:|---|
| weekday | 0.525 | 0.407 | 0.527 | circ_r 0.64 / adj 0.29 |
| month   | 0.243 | 0.125 | 0.240 | circ_r 0.48 / adj 0.42 |
| year    | 0.335 | 0.179 | 0.333 | spearman 0.26 |
| color   | 0.627 | 0.536 | 0.627 | **adj 1.00** |

Honest: unsupervised ordering weaker than the 0.5B model except the color hue-loop
(perfect cyclic adjacency); the torch-backprop curved fit underperforms REML at scale.

## W9 — Manifold seed-stability  ⏸ BLOCKED (harness ready)
`manifold_stability_sac.py` — two seeds → Hungarian latent match vs principal-angle
union-subspace vs canonical content-hash. Blocked on the SAC engine; see
`W9_W10_SAC_BLOCKED.md`. Runner pins BLAS/OMP=1 (deadlock fix). Rerun post-guard-fix.

## W10 — EV-vs-budget from SAC births  ⏸ BLOCKED (harness ready)
`ev_budget_sac.py` — per-atom (Θ, ΔEV) birth frontier vs the linear/sparse reference at
matched Θ. Same SAC-engine blocker. Rerun post-guard-fix.

## SAEBench subset + absorption/SCR  ⏸ GATED
On T1 (WS-C) + composed T2 (WS-A). Deferred until a composed dictionary artifact exists.
