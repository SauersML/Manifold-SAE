# Known issues

Honest record of the codebase's current performance gaps and engineering
workarounds. Each item ends with status: **CLOSED**, **WORKAROUND**, or
**OPEN**.

## gamfit dual-cuBLAS conflict on cluster + cloud images

**Symptom.** gamfit (PyPI ≤ 0.1.101) refuses to load its Rust extension
when two cuBLAS files are mapped in the process:

```
CUDA library conflict before using gamfit._rust. Multiple distinct
shared objects for the same CUDA SONAME family are already mapped
in this Python process.
```

The two files are typically `/usr/local/cuda-12.*/...libcublas.so` and the
torch-bundled `nvidia/cublas-cu12/...libcublas.so.12`.

**Cause.** Both files appear in `/proc/self/maps` because the cluster
loads system CUDA via the driver and torch pulls in its bundled
nvidia-cublas wheel. The catastrophic case (double-free from crossing
handles) only triggers if code uses `dlopen(absolute_path)` on both
files. cudarc's `culib()` is a process-wide `OnceLock<Library>` —
all CUDA calls then route through one handle in practice, so the
detection is overcautious in this configuration.

**Upstream fix.** `SauersML/gam@ff0f5380` + `@233672b6`: downgrade the
assert to a once-per-process warning, lock the contract in a test.
Awaiting publication of a new gamfit wheel (already at 0.1.102 in
upstream main).

**Workaround in this repo.**
`manifold_sae/_cluster_bridge.py::bypass_gamfit_cuda_check()`
neutralizes the Python-side check defensively across
`gamfit._cuda`, `gamfit._binding`, `gamfit.torch._reml`, and
`gamfit._api`. All LLM experiment drivers call this at import.

Status: **WORKAROUND** (Python bypass shipped; upstream fix committed
and pending PyPI release).

## gamfit REML stays on CPU at small K

**Symptom.** With `K = 10` (default basis size), `R = 2`, and batches up
to 8192, gamfit's per-fit FLOPs sit below its CUDA dispatch policy
threshold. The PyTorch encoder / W / Adam path runs on GPU; the inner
REML solve runs on CPU (faer + Rayon).

**Cause.** The policy thresholds are measurement-calibrated. On B200
(measured fp64 ≈ 25 TFLOPS), CPU is genuinely faster than launching a
CUDA kernel for these shapes. The behavior is correct.

**Mitigation.** For workloads that would benefit from GPU REML (much
larger K or batched-many-feature solves), a batched X^TWX kernel in
the streaming-positions path would push the crossover lower. Tracked
upstream as a feature, not a bug.

Status: **OPEN by design** (CPU is the right choice for current K/R).

## Position-coverage and cross-feature ortho memory at large F

**Symptom.** Two loss components allocate `O(B·F·K)` and `O(F²·R²)`
intermediates respectively. For F = 100,000 these become hundreds of
MB per step.

`_position_coverage_loss` builds a `(B, F, n_bins)` Gaussian-binning
tensor. F=4096, B=1024, n_bins=10 → ~40M floats ≈ 160 MB per call.

`_ortho_loss` cross-feature term builds an `(F·R, F·R)` Gram matrix.
F=4096, R=2 → (8192, 8192) ≈ 256 MB per call.

Both fit in cluster B200 memory at current F=2048-4096 but scale
poorly. Both would need a reformulation for F=100K+ — likely chunked
or randomized off-diagonal estimation.

Status: **OPEN** but not blocking research-scale runs.

## Auto-derived knots may not be optimal

**Note, not a bug.** `experiments/llm_real.py` passes
`knots_or_centers=None` to gamfit, asking it to auto-derive knots
from position data. For the soft-rescaled positions in `[0, 1]`,
gamfit's auto-derivation is reasonable. If you find pathological
behavior, pass explicit `centers = torch.linspace(0.0, 1.0, K)` (the
convention used in `manifold_sae/sae.py`).

Status: **OPEN** (only matters at edge cases; default works in
practice).

## Hardcoded float64 for gamfit

gamfit's REML primitive requires float64. The current SAE forward
does dtype casts around the gamfit call:

```python
t_packed = positions.t().contiguous().view(-1).to(torch.float64)
y_packed = y_proj.permute(1, 0, 2).contiguous().view(F * B, R).to(torch.float64)
```

Adds ~4 MB allocation + copy per forward at F=2048, B=128, R=2. Small
but real overhead on GPU. Status: **OPEN**, awaiting gamfit float32
support.

## Resolved

### Encoder hidden dim defaulted to `max(4·D, 2·F)` (commit `1f7edf4`)

The encoder default was `H = max(4·D, 2·F)`, which made the encoder
O(F²) and infeasible at LLM scale (F=100K → 20B params just for the
heads). Now `H = 4·D`, scaling linearly with F. **CLOSED**.

### amp²·curve(t) bug in training-mode forward (commit `9f31143`)

gamfit's `fit.fitted` is `by · (phi @ B)` — already amplitude-weighted.
The pre-fix training-mode forward multiplied `fitted` by
`mask_binary` a second time, producing `amp²·curve(t)` per atom under
`continuous_amp=True`. Inference-mode (locked B) used the correct
`amp·curve(t)`, so locked MSE was ~100× training MSE. Fix: drop the
duplicate multiplication. Bug was invisible under binary masks
(mask²=mask). **CLOSED**.

### Soft-rescale stats not frozen at snapshot (commit `2ab3513`)

Per-batch rescale of positions at snapshot vs at inference gave
different `t` for the same `z_raw`. Now frozen as buffers
(`soft_min_locked`, `soft_max_locked`) in `update_snapshot`.
**CLOSED**.

### torch 2.12 ships only +cu130 wheels (commit `87fa40c`)

PyPI's default torch 2.12 wheel requires a CUDA-13 driver. Cluster
nodes report CUDA 12.9. Pinned `torch<2.12` + explicit +cu128
PyTorch index via `[tool.uv.sources]`, Linux-only marker so macOS
local development still works. **CLOSED**.

### `uv sync` fast-path kept stale wheels (commit `dd27b66`)

Bringing torch <2.12 down to actually install needed a forced
reinstall when the lock changed. `heimdall_jobs/submit.py` now
stamps `.venv/.heimdall_lock_hash` with `sha256(uv.lock)` and rebuilds
the venv on mismatch. **CLOSED**.

### Silent CPU fallback when GPU requested (commit `4734461`)

Cluster jobs requesting `gpus > 0` would silently fall back to CPU
on torch/CUDA mismatches, wasting hours of compute. `MSAE_REQUIRE_CUDA=1`
assertion in every driver (auto-set by submitter when `gpus > 0`)
turns the silent failure into a fast crash with a precise
diagnostic. **CLOSED**.

### Position rescale corrupted for sparse atoms (commit `595e4e8`)

Non-firing-token positions dominated `soft_min/soft_max` for atoms
that fire only on a few token types. Now uses firing-weighted
soft-rescale via `logsumexp(β·z + log(w))`. **CLOSED**.

### Eval-cache JSONs stale after forward-semantics fix (commit `796a930`)

After the amp²·curve fix, old cached MSE numbers were still being
served from `eval_F*.json`. Added `forward_semantics: 2` stamp on
eval-cache signature so cached numbers invalidate while SAE
checkpoints (still useful) stay loadable. **CLOSED**.
