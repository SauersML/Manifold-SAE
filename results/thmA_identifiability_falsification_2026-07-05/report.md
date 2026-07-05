# Theorem A — real-category identifiability, falsification search

**Verdict: Theorem A SUPPORTED for d=1 and d=2.** No real-category counterexample found. The sparse-manifold-atom decomposition is uniquely identifiable exactly at/above the codimension boundary `Σ_k(d_k+1) ≤ p−1`; double parses live on a measure-zero set, not a positive-measure one. The specific failure mode the proof-schema was vulnerable to — real points of complex contact loci carrying positive measure — did **not** appear.

## Method

Atoms are curved cones in ℝ^p; a "parse" of a target `z = Σ_k a_k g_k(t_k)` is recovered by global multistart nonlinear least squares. Multiplicity is counted by clustering distinct parses **after quotienting each atom's ray/reparameterization gauge** (cluster by the reconstructed atom points `a_k·g_k(t_k) ∈ ℝ^p`). Genericity = random generic decoders + random ambient rotation + off-center.

**Positive controls (prove the search detects multiplicity, so nulls are real):** underdetermined `p` (params > equations) → ~100% of targets flagged multi-parse. So a null ("no second parse") means one genuinely does not exist, not that the solver missed it.

## Result — d=1 (ellipse atoms)

Sharp uniqueness cliff exactly at the hypothesis boundary `Σ(d_k+1)=4 ≤ p−1`:

| p | slack | mean frac 2nd parse |
|---:|---:|---:|
| 4 | −1 | 1.00 |
| 5 |  0 | 0.00 |
| 6 |  1 | 0.00 |
| 8 |  3 | 0.00 |

Resolution-invariant: 0.00 stable at n_z∈{50,100,200}×n_starts∈{50,100} (a genuine empty set, not sparse sampling). Centering (Prop 1) confirmed orthogonally: centering flattens a single atom's cone but does not break two-atom separation (still 0.00).

## Result — d=2 (curved sphere/quadric patch + ellipse)

Dictionary {d=2, d=1}, parse = 5 params, boundary `Σ(d+1)=3+2=5 ≤ p−1`:

| p | slack | mean frac 2nd parse |
|---:|---:|---:|
| 4 | −2 | 0.994 |
| 5 | −1 | 1.00 |
| 6 |  0 | 0.00 |
| 8 |  2 | 0.00 |

Cliff exactly at `Σ(d+1)=5 ⇒ p≥6`, identical structure to d=1. Measure-zero confirmed at the boundary p=6 (resolution-invariant to n_z=300/n_starts=150). **A generalizes past ellipses; the codim law is the right threshold across dimension.**

## Theorem B — quantitative margin (a correction to the theory)

In-regime the parse is unique, so "margin" = conditioning of the unique parse = `σ_min` of the parse Jacobian (`→0` = linearized Davis–Kahan co-collapse).
- **Inter-atom coincidence** (drive atom2→atom1, coincidence α): `σ_min ∝ separation (1−α)`, slope-1 log-log, decaying continuously to numerical zero only at exact coincidence.
- **Per-atom flatness** (flatten each atom, keep them separated): `σ_min` **plateaus at ~0.29, bounded away from 0** — still identifiable.

**Correction:** co-collapse is governed by **inter-atom separation/coincidence, NOT per-atom flatness.** Two near-flat but well-separated atoms stay identifiable; two curved atoms that coincide collapse. This sharpens the engine's co-collapse guard: it should key on inter-atom separation (`σ_min` of the *joint* parse Jacobian), not a single-residual eigengap.

## Caveats

d=1 ellipses and a d=2 sphere/quadric patch tested. A measure-zero exceptional locus (an isolated self-intersection) is not sampled by random-z — but that is *permitted* by Theorem A; only a positive-measure locus would falsify, and none was found at/above the codim boundary in either dimension.

Artifacts: `search.py`, `sweep.py`, `margin.py`, `d2_search.py`, `sweep_results.json`, `margin_results.json`, `d2_sweep_results.json`, `thmA_thmB_summary.png`.
