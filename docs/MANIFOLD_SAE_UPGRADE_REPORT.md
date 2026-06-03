# Manifold-SAE gamfit 0.1.151 Upgrade Report

Run date: 2026-06-03

Logs:
- Local: `/tmp/manifold_sae_runner/`
- A100: `/home/azuser/exp_logs/manifold_sae_runner/`

## Installed gamfit and API

Local `/Users/user/Manifold-SAE/.venv` imports `gamfit 0.1.151` from
`.venv/lib/python3.12/site-packages/gamfit`.

A100 default `python3` does not import `gamfit`. The usable A100 setup is:

```bash
cd ~/Manifold-SAE
PYTHONPATH=$HOME/gam:$PWD .venv/bin/python
```

That imports `gamfit 0.1.151` from `/home/azuser/gam/gamfit` and has SciPy/Torch
available. `~/gam/.venv/bin/python` imports the editable gamfit source but lacks
SciPy and Torch, so it cannot run the Manifold-SAE harnesses directly.

The installed `sae_manifold_fit` signature exposes the expected current knobs,
including `assignment`, `isometry_weight`, `block_orthogonality_weight`,
`nuclear_norm_weight`, `nuclear_norm_max_rank`, `decoder_incoherence_weight`,
`top_k`, `t_init`, and `a_init`.

## Solver-independent checks

`experiments.manifold_falsifier --selftest` passed locally and on the A100. The
coordinate metric is isometry-invariant and split-sensitive:

| check | result |
| --- | ---: |
| self `t` vs `t` | 1.0000 |
| gauge flip+shift | 1.0000 |
| shuffled per-token split | -0.9244 |

The fit-independent `sigma_min` coherence sweep was identical locally and on the
A100:

| coherence | theta deg | median sigma_min | p10 sigma_min |
| ---: | ---: | ---: | ---: |
| 0.00 | 90.0 | 0.8572 | 0.6624 |
| 0.25 | 67.5 | 0.8367 | 0.6781 |
| 0.50 | 45.0 | 0.8152 | 0.5489 |
| 0.75 | 22.5 | 0.7968 | 0.4514 |
| 0.90 | 9.0 | 0.7940 | 0.4099 |
| 0.99 | 0.9 | 0.7768 | 0.3835 |

Local `experiments/py_recover.py` completed the pure-torch Run 0/1 coherence
sweep:

| coh | sigmin | coordR2 OFF | reconR2 OFF | coordR2 ON | reconR2 ON |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.955 | 0.544 | 0.910 | 0.994 | 0.909 |
| 0.30 | 0.895 | 0.539 | 0.910 | 0.983 | 0.901 |
| 0.60 | 0.828 | 0.547 | 0.910 | 0.940 | 0.876 |
| 0.80 | 0.799 | 0.987 | 0.910 | 0.879 | 0.853 |
| 0.90 | 0.790 | 0.951 | 0.910 | 0.834 | 0.840 |
| 0.95 | 0.787 | 0.830 | 0.910 | 0.776 | 0.831 |

This supports the incoherence penalty in the lower-coherence cells but not in the
high-coherence cells in this run; OFF is better at `coh >= 0.80`. Do not overclaim
the pure-torch result as a monotone ON-over-OFF win.

The A100 checkout does not currently contain `experiments/py_recover.py`; the
remote log records that file as unavailable.

## Method comparison smoke

Minimal one-seed archival `method_*` benchmark, `D=20`, `N=480`, `G=80` where
available. Metric is mean relative RMS from `recover_bench.evaluate`.

| method | local status | local mean rel RMS | A100 status | A100 mean rel RMS |
| --- | --- | ---: | --- | ---: |
| `method_moment_reml` | OK | 0.2131464918763321 | OK | 0.20344305437325422 |
| `method_fourier` | `GamError`: REML penalty not PSD, eigenvalue `-1.912e-6` | n/a | same error | n/a |
| `method_gpca` | OK | 0.012878359897158144 | OK | 0.012878359897842171 |
| `method_isa` | OK | 0.0806743066679528 | OK | 0.08284347948361381 |
| `method_scms` | OK | 0.05453817209385312 | OK | 0.054538170295369615 |
| `method_mp_reml` | OK | 0.20305579117028594 | OK | 0.24206112299595176 |

## Released gamfit SAE smoke

Local smoke used `n=96`, `D=6`, `n_iter=12`.

| cell | status | reconstruction R2 | REML score | error snippet |
| --- | --- | ---: | ---: | --- |
| `K=1`, circle, iso off | OK | 0.25242511491007114 | -38.78068651529497 | 1 atom |
| `K=1`, circle, iso on | ERROR | n/a | n/a | `RemlConvergenceError`: adaptive proximal correction failed after 16 attempts; Armijo rejected trial objective |
| `K=2`, circle, iso on | ERROR | n/a | n/a | `RemlConvergenceError`: per-row `H_tt^(14)` Cholesky failed, non-PD pivot |

A100 CPU-only smoke used `CUDA_VISIBLE_DEVICES=-1`, `n=24`, `D=4`, `n_iter=2`.
The larger `n=96`, `n_iter=12` CPU-only remote smoke exceeded 2 minutes on the
first cell and was stopped; the reduced run is the recorded A100 smoke.

| cell | status | reconstruction R2 | REML score | error snippet |
| --- | --- | ---: | ---: | --- |
| `K=1`, circle, iso off | OK | 0.34329779423914075 | 12.991598577585453 | 1 atom |
| `K=1`, circle, iso on | ERROR | n/a | n/a | `RemlConvergenceError`: Schur complement Cholesky failed, non-PD pivot `-6.670243943651345` |
| `K=2`, circle, iso on | ERROR | n/a | n/a | `RemlConvergenceError`: per-row `H_tt^(15)` Cholesky failed, non-PD pivot `-4.548369904054394` |

Conclusion: released/source-path `gamfit 0.1.151` can execute the K=1 iso-off
call but does not demonstrate useful circle recovery in this tiny smoke. Iso-on
K=1 and K=2 remain blocked by Arrow-Schur/Cholesky convergence failures.

## Blocked follow-up

No `/home/azuser/exp_logs/SUMMARY.md` or other `SUMMARY.md` indicating "matrix
converges" was present under `~/exp_logs` at run time. I therefore did not rerun
`experiments/manifold_recovery.py` or the Run 0/1 coherence sweep against a newer
source build. Those should be rerun only after the A100 convergence supervisor
publishes a source-build summary showing the matrix convergence fix.
