# slop/

Throwaway / non-load-bearing experiment scratch, moved out of `experiments/`
to declutter. All git-tracked moves — nothing deleted, history preserved.

Contents:

- **`auto_NN.py`** — the unstructured precursor brainstorm log (`auto_01`…`auto_95`).
- **`auto_exp_NN*.py`** — the structured composition-engine experiment series.
- **Named one-off scripts** — `plot_*`, `atom_*`, `build_*_probe`, `llm_*`,
  `*_recovery`, ablations, and other standalone analyses that nothing imports.
- **`artifacts/`** — committed generated outputs (PNG plots + JSON/text dumps)
  that were checked into source: the `gam_mech_interp` data/plots tree and the
  steerability offline report. These regenerate from their producing scripts.

Notes:
- `manifold_sae/diffusion/cross_modality_atlas.py` still imports two helpers
  from `slop/auto_exp_77_diffusion_sae.py` (path repointed when moved).
- Intra-`auto_exp` cross-imports that used `from experiments.auto_exp_NN ...`
  will no longer resolve after the move; these are scratch and not maintained.

Kept in `experiments/` (NOT slop): shared infra (`_pca_basis`,
`color_filter_list`, `color_geometry`, `color_manifold_gam`,
`plot_color_geometry`), imported scripts (`llm_probe`, `llm_sweep`,
`plot_gam_results`, `pure_curve_benchmark`), and the load-bearing
`synthetic_recovery.py` demo.
