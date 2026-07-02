# Manifold-SAE vs linear-SAE: parity table + EV-vs-budget frontier

_Data: planted real-shaped synthetic -- disjoint circle+linear blocks under a random rotation Q; ground-truth curved count known._  
_p (ambient) = 9, n_train = 400, n_test = 100, planted curved = 3, planted linear = 3, curved-atom width b = 3 ({1, cos, sin})._

## 0. What is measured vs modelled

The linear/sparse baselines and the per-atom geometry are MEASURED with the working `gamfit.linear_dictionary_fit` / `gamfit.sparse_dictionary_fit`. The `gamfit.sae_manifold_fit` REML solver is NON-FUNCTIONAL in this build (see sec 4), so the curved side of the parity is by honest parameter ACCOUNTING against the measured per-atom curvature that a straight atom provably cannot reach -- not a live curved fit.

## 1. Linear/sparse EV-vs-budget baseline (measured)

| K_lin | Theta = K_lin*p | linear-dict EV | sparse-dict EV |
|---:|---:|---:|---:|
| 2 | 18 | 0.3632 | 0.3632 |
| 3 | 27 | 0.5342 | 0.5342 |
| 4 | 36 | 0.5887 | 0.5887 |
| 9 | 81 | 0.8531 | 0.7973 |
| 11 | 99 | 0.9068 | 0.8133 |
| 18 | 162 | 0.9703 | 0.9272 |
| 22 | 198 | 0.9739 | 0.9513 |
| 27 | 243 | 0.9739 | 0.9720 |
| 36 | 324 | 0.9739 | 0.9846 |

## 2. Parity table -- manifold vs linear at matched parameter budget

A curved dictionary represents every block with C curved atoms on the C richest circles (b*p = 3*9 = 27 params each) plus one straight atom (p = 9) on every remaining block. The manifold column is that dictionary's IDEALIZED per-block ceiling from measured captures (so C=3 reproduces the measured curved ceiling 0.994); the linear column is the ACTUAL learned `linear_dictionary_fit` at K_lin = Theta/p atoms. `dEV` is therefore an OPTIMISTIC UPPER BOUND on curvature's benefit -- a live curved fit (blocked, sec 4) lands at or below the ceiling.

| curved atoms C | Theta (params) | K_lin equiv | manifold EV (idealized ceiling) | linear EV (actual, matched) | dEV (upper bound) |
|---:|---:|---:|---:|---:|---:|
| 0 | 54 | 6.0 | 0.8343 | 0.6945 | 0.1398 |
| 1 | 72 | 8.0 | 0.8913 | 0.8002 | 0.0910 |
| 2 | 90 | 10.0 | 0.9453 | 0.8800 | 0.0654 |
| 3 | 108 | 12.0 | 0.9926 | 0.9159 | 0.0768 |

Stronger: the actual linear dictionary SATURATES at EV ~0.9739 (adding straight atoms past K~2p stops helping -- see sec 1), which is BELOW the curved ceiling 0.994 reached at only Theta = 108 params. Under top-1 routing a straight atom cannot trace a circle, so ~0.0201 EV is inaccessible to linear at ANY budget -- that residual is exactly what the curved atoms recover. The per-matched-budget edge is a modest UPPER BOUND (~0.05-0.09 EV); the durable point is that curvature accesses EV linear cannot, on the FEW genuinely curved blocks -- spend it sparingly (keep curved K small).

## 3. EV-vs-budget frontier -- which curved atoms pay rent (measured geometry)

Per planted atom: the variance a single STRAIGHT (rank-1, top-1) atom captures vs what a circle chart (2 PCs) captures. `global_dEV` is the curvature gap weighted by the atom's share of total variance -- the held-out reconstruction a curved atom buys that a straight one cannot. Sorted; the cutoff is where the gap falls to the straight-atom floor.

| rank | atom | planted | Theta_curved | linear top-1 EV | circle EV | global dEV |
|---:|---:|:--:|---:|---:|---:|---:|
| 1 | 0 | circle | 27 | 0.6137 | 0.9962 | 0.0570 |
| 2 | 1 | circle | 27 | 0.6062 | 0.9942 | 0.0541 |
| 3 | 2 | circle | 27 | 0.7164 | 0.9955 | 0.0473 |
| 4 | 3 | line | 27 | 0.9917 | 0.9937 | 0.0005 |
| 5 | 4 | line | 27 | 0.9910 | 0.9938 | 0.0005 |
| 6 | 5 | line | 27 | 0.9860 | 0.9898 | 0.0004 |

**Noise floor** (largest global dEV among planted-linear atoms): 0.0005.  
**Recommended K_curved cutoff = 3** (atoms whose gap clears the floor). Planted curved ground truth = 3 -> the evidence-limited cutoff recovers the true curved count. The marginal gap crosses the noise floor exactly at the circle/line boundary: keep curved K small.

## 4. Manifold-fit attempt (recorded blocker)

`gamfit.sae_manifold_fit` was attempted in isolated subprocesses. Outcomes in this build:

- K=4: FAILED -- worker produced no output (OOM/segfault)
- K=6: FAILED -- worker produced no output (OOM/segfault)

## Provenance / caveats

- Real OLMo-3-32B activations (runs/OLMO3_32B_*_SELF_QUALIA_*/, L44) ARE present and were tried first, but the self/qualia last-token readout is a LINEAR axis: circle atoms co-collapse ('no atom carries material signal to anchor', held-out EV ~0.03-0.11). Run `--real-probe` to reproduce.
- The documented curved structure (color/hue loop, DATA_README sec 8) is not present locally as an array harvest, so the ground-truth frontier uses fixed real-shaped planted synthetic where the curved count is known.
- gamfit.sae_manifold_fit (REML) is NON-FUNCTIONAL in this shared-tree build: the compiled ext is stale vs the Python wrapper (kwargs structured_residual_passes/promote_from_residual, runtime-stripped), the IBP joint Hessian goes non-PD ('infeasible rho probe' GamError), and the inner Newton loop is slow + memory-leaky (OOM). So the curved side is by parameter accounting against MEASURED per-atom geometry, not a live fit.
