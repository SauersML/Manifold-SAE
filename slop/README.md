# slop/

Repo-wide throwaway / non-load-bearing scratch, swept out of the live tree to
declutter. All moves are `git mv` — nothing deleted, history preserved, paths
mirror the original location (`slop/scripts/...`, `slop/runs/...`, etc.) so
anything here can be moved straight back.

This is the sibling of `experiments/slop/` (which holds experiment-script
scratch specifically). This top-level drawer holds slop swept from the rest of
the repo.

## Sweep — 2026-06-30

Nothing here is imported by the shipped `manifold_sae` package, exercised by
`tests/`, or referenced by CI (`steering_server`) — verified by grep before each
move.

- **`scripts/`** — orphaned one-off / superseded training & plotting scripts and
  Azure launcher shell scripts that nothing imports or invokes
  (`train_matryoshka`, `train_sae_f65k*`, `train_equivariant`,
  `train_cylinder_shared`, `train_manifold_f_sweep`, `train_behavioral_probes`,
  `crosscoder_hsv_corr_only`, `gamfit_periodic_repro`, `run_llm_autointerp`,
  `plot_ground_truth_fit`, `plot_duchon_diagnostics`, `plot_residuals_vs_t`,
  `phase0_substrate` + `phase0_notes.md`, the three `run_self_qualia_*_azure*.sh`).
  The ~23 scripts still imported by experiments/tests stayed in `scripts/`.
- **`distributed_manifold_sae/`** — the K=1M distributed-training scaffold. Its
  own `__init__.py`/README mark it "scaffolding … not run end-to-end"; no tests,
  nothing imports it.
- **`runs/`** — the force-added tracked artifacts from a `.gitignore`d output
  dir (diagnostic PNGs, run logs, metrics JSON, cached `.npy`). All regenerable.
- **`results/a100_runs/`** — cluster (A100) job outputs: run logs, cached
  activations, metrics dumps, and one-off in-run analysis scripts. The recent
  `results/color_gamfit_*` / `color_orderfree_gallery` showcase galleries were
  KEPT in `results/`.
- **`docs/`** — dated / superseded session logs and one-off reports:
  `cluster_session_2026_05_21.md` (self-marked historical/retracted),
  `a100_results_2026_06_03.md`, `MANIFOLD_SAE_UPGRADE_REPORT.md`, and
  `results.md` (superseded by `docs/findings.md`, which it points to).
- **`manifold_sae/`** — four documented-but-unused package modules that nothing
  imports (`_normalize.py`, `encoder_linear.py`, `data_activations.py`,
  `metrics.py`). The README repository-layout block was updated to drop them.

Kept in the live tree (NOT slop): the `manifold_sae` package proper,
`steering_server/`, the `dashboard/` / `cross_llm_platform/` /
`concept_manifold_steering/` subsystems (finished, tested), `tools/`,
`heimdall_jobs/`, `benchmarks/`, the ~23 imported `scripts/`, the recent
`results/color_*` galleries, and the 12 core `docs/`.
