# RESULTS — Track A5 (EVAL + figures) running log

Pre-registration frozen at `Manifold-SAE/experiments/prereg_35b.md` (committed 849b5d5 →
75cc072 → f171b53, before any real number landed). Report generator:
`experiments/report_35b_figures.py` (`--selftest` = synthetic wiring proof; default reads
`results/run_35b/`). Scorecard bakes into `REPORT_35B.md`. EV baseline pinned = TRAIN-mean
TSS on the disjoint held-out split (held-out colmean is the silent-fake, rejected).

## Pipeline state
- Full 6-axis generator + 9 figures + G0 gate: BUILT, self-tested green (figures_35b/_selftest).
- `results/run_35b/` = real artifacts as they land; everything else PENDING (never faked).

## Landed real artifacts (results/run_35b/)
| when | lane | file | verdict |
|------|------|------|---------|
| 2026-07-03 | DOSE | dose_calibration.json (from dose_qwen8b_out, Qwen3-8B L18) | **DEGENERATE / null crown** — see below |

## Cell log
### A4/A5/A6/G_wrap/I3 — DOSE crown, Qwen3-8B (8B carries crown per pre-reg failure branch)
- **A5 dose slope: PENDING** — `stats.manifold.n == 0`. No calibration point survived the fit.
- **A6 dose R²: PENDING** — same cause.
- **A4 ordering: MISS** — spacing-robust ordering_corr: month 0.570, weekday 0.394 (raw circ),
  color 0.120. All < 0.9 threshold.
- **G_wrap: partial** — weekday wraparound_in_order = True; month/color = False.
- ROOT CAUSE (flagged to DOSE): dose steps produce Δnorm ~1e-7 in activation space
  (rel ~1e-9 of the residual) → measured KL ~1e-11 = machine noise. At doses small enough
  to stay inside the chart validity radius (~1.6e-4) the KL is unmeasurable; at doses big
  enough to register you leave validity. Only 21/480 rows within_validity, none usable →
  n=0 regression. Perturbation almost certainly mis-scaled (chart tangent applied in a
  whitened/normalized space, not mapped back to activation-norm units). Needs a rerun with
  dose scaled so Δnorm spans a real fraction of the residual norm.

### R3 — split hygiene / matched budget: **PASS** (real DATA attestation)
- data/l17/split_manifest.json (synced from MSI, byte-match): split_policy "whole-file
  (rollout/chunk-safe); no row-level split", 200 files, 1,204,602 train / 201,169 held-out
  tokens, nan_counts {train:0, heldout:0}, tier0.json sidecar present, matched_currency=actives.
  Chunk-level split + tier0-train-only both satisfied → the first way-to-fake (row split) closed.

## Blockers (fleet, per STATE.md 2026-07-03 14:15)
- BLOCKER-1: K=2 planted-circle smoke HANGS on gamfit 0.1.247 → stagewise gate RED →
  COMPOSE cells (A2/I1/G_band/G_util/P2/gallery) stay PENDING.
- BLOCKER-2: T1 GPU never engages (0% util, CPU fallback) → A1/P1 stay PENDING.
- DOSE crown: whitened-frame unit bug CONFIRMED by two lanes; DOSE re-running with Δt→raw
  activation-unit fix (sanity gate: one Δt=1 patch must give KL>1e-3 before any sweep).
- msi master: was briefly ABSENT mid-session, back UP now (get verified 14:13). Not a
  standing blocker; report if it drops again (never `msi up` myself).

## Generator robustness (verified this session)
- fig2 (Θ,ΔEV) returns MISS not crash on any atom missing theta/ΔEV; enforces held-out-LOAO
  ΔEV provenance (Goodhart guard) — A2 auto-MISSes unless COMPOSE sets delta_ev_source to a
  held-out/LOAO value. _accepted_curved None-safe. All threshold constants match prereg_35b.md
  (verified vs doc for CONTROL's drift diff).

## Awaiting
- T1 `l17_t1_frontier.json` (A1, P1) — t1_run in flight on acn116.
- COMPOSE `compose_per_atom.json` + `compose_mdl.json` (A2, I1, G_band, G_util, P2, gallery)
  — only smoke/pca512 checkpoints so far.
- CONTROL `fidelity_currency.json` (F2/F3/floor) + `null_control.json` (G0 GATE) — null data
  generated, 35B/8B verify jobs in flight.
- STABILITY `stability.json` (R1); DATA `manifest.json` (R3 — split_manifest+tier0 ready on MSI).
