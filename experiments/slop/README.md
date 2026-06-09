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
- **Second sweep (2026-06-09)** — the Jun 5–8 batch that post-dated the first
  cleanup: the `self_qualia_*` / OLMo-introspection cluster, the `color_loop_*`
  / `color_gamfit_*` / `color_ring_*` / `color_orderfree_gallery` / Duchon
  media-generation scripts, the `color_media/`, `gam_mech_interp/`, and
  `gamfit_manifold/` subtrees, the `demo_*`/`real_llm_*`/`trust_*`/`expand_*`/
  `multi_axis_*`/`analyze_*` one-offs, `manifold_recovery`/`manifold_falsifier`,
  and their `.csv`/`.jsonl`/`.json` data sidecars. Bulk media render outputs
  (the Duchon training video frames + logs) live under `artifacts/`.

Notes:
- `manifold_sae/diffusion/cross_modality_atlas.py` still imports two helpers
  from `slop/auto_exp_77_diffusion_sae.py` (path repointed when moved).
- Intra-`auto_exp` cross-imports that used `from experiments.auto_exp_NN ...`
  will no longer resolve after the move; these are scratch and not maintained.

Kept in `experiments/` (NOT slop): shared infra (`_pca_basis`,
`color_filter_list`, `color_geometry` — the only experiment module imported by
real `manifold_sae`/`scripts` code — `color_manifold_gam`, `xkcd_colors.txt`),
imported scripts (`llm_probe`, `llm_sweep`), the `pure_curve_benchmark` demo,
and the `AUTO_EXP_PLAN.md` / `DATA_README.md` docs.

Two self-contained subsystems were left in `experiments/` as a judgment call
rather than swept: `steerability/` (a steering benchmark with harvested `.npz`
axes + `run_all.sh`) and `bank_additions/` (the self-qualia bank `.jsonl` data
store + `SCHEMA.md`). Neither is imported by real code; move them too if you
consider them slop.
