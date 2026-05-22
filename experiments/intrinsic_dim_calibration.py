"""Calibrate the intrinsic-dim measurement on known-1D synthetic data.

The post-fix `concept_intrinsic_dim` reported local PCA dim 4-6 for
Qwen-1.5B concepts, which was used to argue concepts aren't 1D. But:

* Local PCA at k=√N≈13 neighbors in ℝ¹⁵³⁶ is *extremely* noise-sensitive.
* A known 1D curve x(t) = direction · scalar_warp(t) plus ambient
  noise σ may also read as dim 4-6 locally if σ is non-trivial relative
  to the curve's local span.

This script plants a clean 1D smooth curve in ℝ¹⁵³⁶, adds noise σ
on a sweep {0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5}, runs the same
intrinsic-dim measurements (PCA k90, correlation dim, local PCA k95)
that were applied to LM concepts. The output is a calibration curve:
"what local PCA dim does the measurement report for *known* 1D data
at noise level σ?"

If known-1D at realistic σ (≈ 0.05-0.1 in normalized residual space)
reads as local dim 4-6, the LM concept measurement is consistent
with concepts being 1D plus realistic noise — and the negative
"concepts aren't 1D" verdict is overstated.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from manifold_sae._cluster_bridge import require_cuda_if_env


@dataclass
class Config:
    d_ambient: int = 1536
    n_samples: int = 170                    # match Qwen concept-prompt counts
    noise_levels: tuple[float, ...] = field(
        default_factory=lambda: (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5)
    )
    seed: int = 0
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/INTRINSIC_DIM_CALIB")


def plant_1d_curve(cfg: Config, sigma: float) -> torch.Tensor:
    rng = np.random.default_rng(cfg.seed)
    D = cfg.d_ambient
    # Single random direction in D
    v = rng.standard_normal(D); v = v / np.linalg.norm(v)
    # smooth scalar warp f(t) = sum_k a_k sin(k pi t)  for t in [0, 1]
    K = 6
    a = rng.standard_normal(K)
    t = rng.uniform(0.0, 1.0, cfg.n_samples)
    warp = sum(a[k] * np.sin((k + 1) * np.pi * t) for k in range(K))
    X = warp[:, None] * v[None, :]
    X = X + sigma * rng.standard_normal(X.shape)
    # Per-dim normalize to match the LM measurement convention
    Xn = X - X.mean(axis=0, keepdims=True)
    s = Xn.std(axis=0, keepdims=True).clip(min=1e-6)
    Xn = Xn / s
    return torch.from_numpy(Xn.astype(np.float32))


def k90(X: torch.Tensor) -> tuple[int, int, int]:
    X = X - X.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(X.float(), full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum().clamp(min=1e-12)
    def first_ge(t): return int((cum >= t).float().argmax().item()) + 1
    return first_ge(0.5), first_ge(0.9), first_ge(0.99)


def correlation_dim(X: torch.Tensor, n_r: int = 12) -> float:
    Xn = X.numpy().astype(np.float64)
    N = Xn.shape[0]
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    d = np.sqrt(sq[np.triu_indices(N, k=1)])
    d_med = np.median(d)
    rs = np.geomspace(0.05 * d_med, 2 * d_med, n_r)
    cs = np.array([(d < r).mean() for r in rs])
    mask = (cs > 0.05) & (cs < 0.5) & (cs > 0)
    if mask.sum() < 3:
        mask = cs > 0
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(rs[mask]), np.log(cs[mask]), 1)[0])


def local_pca_dim(X: torch.Tensor, thresh: float = 0.95) -> float:
    Xn = X.numpy().astype(np.float64)
    N, D = Xn.shape
    k = max(8, int(math.sqrt(N)))
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    nn = np.argsort(sq, axis=1)[:, 1:k+1]
    dims = []
    for i in range(N):
        nbr = Xn[nn[i]] - Xn[i]
        _, s, _ = np.linalg.svd(nbr, full_matrices=False)
        cum = np.cumsum(s**2) / (s**2).sum()
        dims.append(int((cum >= thresh).argmax()) + 1)
    return float(np.median(dims))


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] D={cfg.d_ambient} N={cfg.n_samples}  noise levels={cfg.noise_levels}", flush=True)

    print(f"\n  {'sigma':>6}  {'k50':>4}  {'k90':>4}  {'k99':>4}  {'corr_d':>7}  {'local_p':>8}", flush=True)
    results = {}
    for sigma in cfg.noise_levels:
        X = plant_1d_curve(cfg, sigma)
        k_50, k_90, k_99 = k90(X)
        cdim = correlation_dim(X)
        lpca = local_pca_dim(X)
        print(f"  {sigma:6.3f}  {k_50:4d}  {k_90:4d}  {k_99:4d}  {cdim:7.2f}  {lpca:8.2f}", flush=True)
        results[f"sigma={sigma}"] = {"k50": k_50, "k90": k_90, "k99": k_99,
                                       "corr_dim": cdim, "local_pca_dim": lpca}

    summary = {"config": asdict(cfg), "results": results}
    (out_dir / "calibration.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] {out_dir / 'calibration.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
