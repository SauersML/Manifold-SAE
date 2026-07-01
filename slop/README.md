# slop/

Repo-wide throwaway / non-core scratch, swept out of the live tree to keep the
shipped `manifold_sae` package focused. All moves are `git mv` (history
preserved); paths mirror the original location so anything here can be moved
straight back. Sibling of `experiments/slop/` (experiment scripts); this drawer
holds package-internal slop.

## SAE-variant sweep — 2026-06-30

Goal: a **focused, gamfit-0.1.241-native manifold SAE that beats the linear
baseline** — not a drawer of 17 half-finished SAE variants that gamfit already
implements natively (and better). Swept the experimental variant zoo out of
`manifold_sae/`. Verified before moving: every remaining reference to these from
the kept package was a comment/docstring — **no live imports**, so the core
package still imports clean (19/19 modules + subpackages) and the core tests pass.

Moved to `slop/manifold_sae/` (19 variant modules + the everything-leaderboard):
- SAE variants: `das_sae`, `crosscoder`, `transcoder`, `hyperbolic_sae`,
  `cylinder_sae`(+`_shared`), `sindy_sae`(+`_static`), `wasserstein_sae`,
  `equivariant`, `sheaf`, `matryoshka`, `crm`, `identifiable`,
  `adaptive_k`(+`_v2`), `amortized_manifold_sae`, `circuit_trace`, `integration`.
  (gamfit 0.1.241 provides native equivalents for most: `InterchangeSwapDecoder`,
  `crosscoder.Crosscoder`, `skip_transcoder`, `PoincareAtoms`, `SheafConsistencyPenalty`,
  `AdaptiveTopK`, etc. `matryoshka`/`sindy_sae` have no public gamfit primitive.)
- `eval/leaderboard_v2.py` — the "run every variant" leaderboard.
- Their `tests/test_*.py` (12) and `scripts/` drivers (14).

Kept in `manifold_sae/` (the core + applied mech-interp tooling): `sae` (thin
wrapper over `gamfit.torch.ManifoldSAE`), `encoder`, `losses`, `train`,
`diagnostics`, `data_synthetic`, `scale`, `_cluster_bridge`, and the subpackages
`eval` (harness/registry/baselines — the linear-vs-manifold "beats linear"
evidence path), `atlas`, `autointerp`, `behavioral`, `cross_llm`, `diffusion`,
`kernels`.

Note: also in this sweep the gamfit pin moved to `>=0.1.241` and the four core
hubs (`sae`/`encoder`/`losses`/`diagnostics`) were cut over to gamfit-native.
</content>
