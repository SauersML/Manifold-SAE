# R-review — adversarial review notes

Reviewer for the block-sparse + curved-chart featurizer fleet. Repos polled:
`/Users/user/gam` (main) and `/Users/user/Manifold-SAE` (main). I edit nothing
but this file.

## Baseline (established before lanes committed)

Reusable pieces BT1-block MUST build on rather than reimplement:

- `crates/gam-sae/src/frames.rs`
  - `GrassmannFrame` (l.500) with `polar_update(cross_moment)` (l.586) — polar
    factor of a p×r cross-moment via thin SVD → orthonormal frame U (the
    Grassmann/Stiefel retraction). `reconstruct_decoder`, `project_decoder`,
    `max_principal_angle`, `gauge_singular_values`.
  - `GrassmannCrossMoment` (l.709): `accumulate` + `polar_frame()`.
- `crates/gam-sae/src/sparse_dict/`
  - Current per-atom selection (`scoring.rs`): top-`s` by `|xᵀd|`, atoms
    unit-norm. `TopSSelector.offer` selects by `score.abs()`. This is the
    PER-ATOM analog; block-sparse must select by `‖z_g‖₂` over a b-dim block
    code, NOT by any single coordinate.
  - `codes.rs::solve_row_codes`: active-set ridge LS. Signed codes already
    (no ReLU). Good — block version must also keep signed codes.
  - `SparseDictFit` decoder rows unit-norm; scale identified by projection step.

Gauge-invariance target for BT1: any O(b) rotation R of block g's basis
(D_g → R D_g, z_g → R z_g) must leave BOTH selection (‖z_g‖₂ = ‖R z_g‖₂) and
the loss identical. Test must BITE: construct a fit, apply a random O(b)
rotation to one block, assert selection indices + loss unchanged; and a NEGATIVE
control that a non-orthogonal (norm-changing) map DOES change them.

## Lane status
- BT1-block: no commits yet (crates/gam-sae/src/sparse_dict/ block-* not present)
- N-nursery: experiments/block_nursery/ not present
- G-bsf: experiments/bsf_baseline/ not present
- M-mdl: experiments/mdl_ladder/ not present
- P-null: experiments/matched_null.py not present

## Prep notes (for fast review when lanes land)

N-nursery — likely forks `manifold_stability_sac.py` / `seed_stability.py`.
Checks to run when it lands:
- Planted-data leak: `data_planted` RETURNS the true `planes`. The nursery arm's
  chart seeds must come from residual-PCA / data (as SAC birth does), NOT from
  `planes`. Grep the nursery init for any use of the ground-truth planes/assign.
- Control arm (joint K≥2 `sae_manifold_fit`): must actually be invoked, bounded
  (timeout/iter cap), and its outcome recorded even on non-convergence — not
  silently skipped so the nursery arm "wins" by default. Memory says joint K≥2
  co-collapses / OOMs, so a subprocess wrapper with a recorded timeout is the
  honest control.
- Held-out EV: both arms must score EV on the SAME held-out split with the SAME
  demean/scale (per-template demean is mandatory here — memory
  [[manifold-sae-harvest-gotchas]]). Watch for train-EV in one arm vs test-EV in
  the other.
- Subprocess isolation: every `sae_manifold_fit` must be in a subprocess (memory
  [[gam-sae-manifold-fit-broken-build]] — stale-ext/non-PD/OOM in-proc). Confirm
  the wrapper exists around EVERY call, incl. the control.

P-null — `experiments/matched_null.py` (new). W7 cyclic claims.
- "Matched" = the null must preserve the stated invariants of the real statistic
  (e.g. phase-shuffle / rotation that keeps the marginal spectrum, radius) so the
  only thing destroyed is the cyclic structure. A null that also changes variance
  is not matched.
- Permutation count: p = (1+#{null ≥ obs})/(1+B); need B large enough that the
  smallest reportable p is below the claim (e.g. B≥999 for p<.001 headline).
- No peeking: null draws must not reuse the test statistic's fitted params.

## Findings
(none yet — polling)
