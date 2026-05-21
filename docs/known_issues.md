# Known issues

Honest current state of the codebase. Each item ends with status: **OPEN**, **WORKAROUND**, or **CLOSED**.

## gamfit dual-cuBLAS conflict on Colab + several cloud images

**Symptom.** gamfit 0.1.81–0.1.98 refuses to load its Rust extension with:

```
CUDA library conflict before using gamfit._rust. Multiple distinct shared objects
for the same CUDA SONAME family are already mapped in this Python process.
```

The two files are `/usr/local/cuda-12.8/...libcublas.so.12.8.4.1` (system) and `/usr/local/lib/python3.12/.../nvidia/cublas/lib/libcublas.so.12` (torch-bundled wheel).

**Cause.** Both files appear in `/proc/self/maps` because Colab loads system CUDA via the driver and torch pulls in its bundled nvidia-cublas wheel. gamfit's safety check is *correct that two files exist* but the catastrophic case (double-free from crossing handles) only triggers if code uses `dlopen(absolute_path)` on both files. Standard library code uses `dlopen(SONAME)`, which glibc resolves to one file deterministically — all CUDA calls then route through one handle in practice.

**Upstream fix.** Committed at `SauersML/gam@7efd17eb`: downgrades the assert to a once-per-process warning. Awaiting a new gamfit wheel on PyPI.

**Workaround in this repo.** `experiments/llm_real.py` monkey-patches `gamfit._cuda.cuda_diagnostics` to return an empty conflict set, neutralizing the check at the source. Also patches the assert function in `gamfit._cuda`, `gamfit._binding`, `gamfit.torch._reml`, and `gamfit._api` defensively. Comes out when a new gamfit wheel containing the upstream fix lands.

**Secondary issue.** Inside gamfit's Rust there's an independent CUDA-dispatch refusal that also keys on the dual-stack diagnostic. Even with the Python bypass, gamfit falls back to CPU for its inner REML solve. The PyTorch encoder / W / Adam path stays on GPU, but every batch incurs a CPU-GPU round-trip. Status: **WORKAROUND** (Python bypass) + **OPEN** (Rust-side check still forces CPU REML).

## Curve SAE per-batch wall time on CPU

**Symptom.** With gamfit forced to CPU (above), each curve-SAE training step takes longer than vanilla SAE on GPU. At F=2048, B=128, K=10, expect roughly 1–3 sec/step on a T4 + one CPU core.

**Mitigations** in current defaults of `experiments/llm_real.py`:
- `n_steps_curve` = 1500 (separate from vanilla 3000)
- `batch_size_curve` = 128 (vanilla stays at 1024)
- `n_features` = 2048

Total estimated curve-SAE wall time: 5–15 min on T4.

**Status.** **WORKAROUND** (smaller defaults). Resolves to *fast* once gamfit's Rust-side dual-stack issue is also fixed and gamfit uses GPU REML.

## Toy 5/5 visual recovery not yet reproduced under Path B

**Symptom.** `experiments/synthetic_recovery.py` under the current REML-based architecture (commit `74c307a` onward) doesn't produce 5/5 visually clean Procrustes-aligned overlays for the planted curves (line, parabola, ramp_exp, logmap, sqrt). Earlier persistent-`B`-as-`nn.Parameter` architecture hit 5/5 at chamfer 0.011 mean / 0.019 max — but that version didn't actually do REML (the smoothness term was a hand-rolled quadratic penalty).

**Diagnosis (theoretical).** Each feature gets one scalar `λ_k` shared across all R output dimensions of its subspace. When the planted feature has intrinsic rank 1 (e.g. line) but R=4, three of the four output dimensions are noise. `λ_k` picks a compromise — under-smooths the noise dimensions, which then wiggle to fit batch noise. Visible as spline oscillation in the Procrustes plot.

**Fix candidates** (not yet executed):
- Match R to GT max intrinsic rank (R=2 for the parabola; line/ramp_exp/sqrt are rank ≤ 2 too).
- Smaller basis K to constrain spline freedom (K=8 instead of K=20).
- Larger batches so each per-feature REML fit sees more data.

Note that scale benchmarks (`experiments/realistic_scaling.py` at D up to 512, F up to 64) already pass under Path B with the curve SAE winning explained variance by 16–21 percentage points over vanilla. The toy retune is a smaller cleanup, not evidence of a structural issue.

**Status.** **OPEN.**

## Position-coverage and cross-feature ortho memory at large F

**Symptom.** Two loss components allocate `O(B·F·K)` and `O(F²·R²)` intermediates respectively. For F = 100,000 at LLM scale, these become hundreds of MB per step.

`_position_coverage_loss` builds a `(B, F, n_bins)` Gaussian-binning tensor. F=4096, B=1024, n_bins=10 → 40M floats ≈ 160 MB per call.

`_ortho_loss` cross-feature term builds an `(F·R, F·R)` Gram matrix. F=4096, R=2 → (8192, 8192) ≈ 256 MB per call.

Both fit in T4 memory at current F=2048–4096 but scale poorly. Both would need to be reformulated for F=100K+.

**Status.** **OPEN** but not blocking research-scale runs.

## Auto-derived knots may not be optimal

**Note, not a bug.** `experiments/llm_real.py` passes `knots_or_centers=None` to gamfit, asking it to auto-derive knots from position data. For the soft-rescaled positions in `[0, 1]`, gamfit's auto-derivation is reasonable. If you find pathological behavior, pass explicit `centers = torch.linspace(0.0, 1.0, K)` (the convention used in `manifold_sae/sae.py`).

## Hardcoded float64 for gamfit

gamfit's REML primitive requires float64. The current SAE forward does dtype casts around the gamfit call:

```python
t_packed = positions.t().contiguous().view(-1).to(torch.float64)
y_packed = y_proj.permute(1, 0, 2).contiguous().view(F * B, R).to(torch.float64)
```

Adds ~4 MB allocation + copy per forward at F=2048, B=128, R=2. Small but real overhead on GPU. Status: **OPEN**, awaiting gamfit float32 support.

## Encoder hidden dim defaulted to `max(4·D, 2·F)` (fixed)

Pre-commit `1f7edf4` the encoder default was `H = max(4·D, 2·F)`, which made the encoder O(F²) and infeasible at LLM scale (F=100K → 20B params just for the heads). Now `H = 4·D`, scaling linearly with F. Status: **CLOSED**.
