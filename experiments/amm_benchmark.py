"""Appendix-H AMM benchmark with BSF baselines and chart ceilings.

Purpose
-------
Faithfully exercise the Block-Sparse Featurizer paper's synthetic claim:

    x = sum_{g in S} m_g + eps,  m_g in M_g subset R^d, |S| = k.

The benchmark trains the local Goodfire-style BSF reimplementation on a mixed
manifold zoo and scores the paper's own recovery metric: per-factor contribution
R2 after Hungarian matching. It also includes two true-frame ceilings:

* ``linear_oracle``: project onto the true factor subspace.
* ``chart_oracle``: project onto the true factor subspace, then nearest-neighbor
  denoise onto the true manifold chart.

Those oracle rows are not a deployable method. They are an honesty guard for the
predicted noise-regime crossing: when sigma grows, the chart ceiling should pull
away from a subspace-only reconstruction by denoising off-manifold directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BSF_DIR = ROOT / "experiments" / "bsf_baseline"
if str(BSF_DIR) not in sys.path:
    sys.path.insert(0, str(BSF_DIR))

from bsf import BSF, BSFConfig, TrainConfig, ev, train_bsf  # noqa: E402


@dataclass(frozen=True)
class Primitive:
    name: str
    intrinsic_dim: int
    span_dim: int
    topology: str
    sampler: Callable[[np.random.Generator, int], np.ndarray]


def _center_scale(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    c = a - a.mean(0, keepdims=True)
    rms = math.sqrt(float(np.mean(np.sum(c * c, axis=1))))
    return c / max(rms, 1.0e-12)


def _circle(rng: np.random.Generator, n: int) -> np.ndarray:
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    return _center_scale(np.c_[np.cos(t), np.sin(t)])


def _arc(rng: np.random.Generator, n: int) -> np.ndarray:
    t = rng.uniform(-0.8 * np.pi, 0.8 * np.pi, n)
    return _center_scale(np.c_[np.cos(t), np.sin(t)])


def _torus4(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(0.0, 2.0 * np.pi, n)
    return _center_scale(np.c_[np.cos(u), np.sin(u), np.cos(v), np.sin(v)])


def _sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.uniform(-1.0, 1.0, n)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return _center_scale(np.c_[r * np.cos(t), r * np.sin(t), z])


def _disk(rng: np.random.Generator, n: int) -> np.ndarray:
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    return _center_scale(np.c_[r * np.cos(t), r * np.sin(t)])


def _linear_gaussian(rng: np.random.Generator, n: int) -> np.ndarray:
    return _center_scale(rng.standard_normal((n, 3)))


ZOO: tuple[Primitive, ...] = (
    Primitive("circle", 1, 2, "circle", _circle),
    Primitive("torus", 2, 4, "torus", _torus4),
    Primitive("sphere", 2, 3, "sphere", _sphere),
    Primitive("arc", 1, 2, "arc", _arc),
    Primitive("disk", 2, 2, "disk", _disk),
    Primitive("linear", 3, 3, "linear", _linear_gaussian),
)


@dataclass(frozen=True)
class Config:
    out_dir: Path = ROOT / "runs" / "AMM_BENCHMARK"
    n_factors: int = 24
    d_ambient: int = 128
    active_count: int = 3
    n_train: int = 20000
    n_test: int = 5000
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    noises: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    coherences: tuple[float, ...] = (0.0,)
    bsf_steps: int = 1200
    bsf_batch_size: int = 1024
    bsf_lr: float = 4.0e-3
    skip_bsf: bool = False


@dataclass
class Data:
    x_train: np.ndarray
    x_test: np.ndarray
    active_test: np.ndarray
    contrib_test: np.ndarray
    frames: list[np.ndarray]
    primitive_names: list[str]
    intrinsic_dims: list[int]
    span_dims: list[int]
    topologies: list[str]
    libraries: list[np.ndarray]


def _primitive_sequence(n: int) -> list[Primitive]:
    return [ZOO[i % len(ZOO)] for i in range(n)]


def _frame(rng: np.random.Generator, span_dim: int, d_ambient: int, anchor: np.ndarray, coherence: float) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d_ambient, span_dim)))
    if coherence > 0.0:
        q[:, 0] = (1.0 - coherence) * q[:, 0] + coherence * anchor
        q, _ = np.linalg.qr(q)
    return q[:, :span_dim].T.copy()


def make_data(cfg: Config, *, seed: int, noise: float, coherence: float) -> Data:
    rng = np.random.default_rng(seed)
    primitives = _primitive_sequence(cfg.n_factors)
    if sum(p.span_dim for p in primitives) > cfg.d_ambient and coherence == 0.0:
        print("[warn] total span exceeds ambient dimension; exact orthogonality impossible", flush=True)
    anchor = np.linalg.qr(rng.standard_normal((cfg.d_ambient, 1)))[0][:, 0]
    frames = [_frame(rng, p.span_dim, cfg.d_ambient, anchor, coherence) for p in primitives]
    libraries = [(p.sampler(rng, 4096) @ frames[i]) for i, p in enumerate(primitives)]

    def sample(n: int, keep: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = np.zeros((n, cfg.n_factors), dtype=bool)
        contrib = np.zeros((n, cfg.n_factors, cfg.d_ambient), dtype=np.float64)
        x = np.zeros((n, cfg.d_ambient), dtype=np.float64)
        for row in range(n):
            active[row, rng.choice(cfg.n_factors, cfg.active_count, replace=False)] = True
        for g, primitive in enumerate(primitives):
            rows = np.flatnonzero(active[:, g])
            if rows.size == 0:
                continue
            pts = primitive.sampler(rng, rows.size) @ frames[g]
            x[rows] += pts
            if keep:
                contrib[rows, g] = pts
        return x, active, contrib

    x_train, _, _ = sample(cfg.n_train, keep=False)
    x_test_clean, active_test, contrib_test = sample(cfg.n_test, keep=True)
    signal_rms = math.sqrt(float(np.mean(np.sum(x_train * x_train, axis=1))))
    eps_scale = noise * signal_rms / math.sqrt(cfg.d_ambient)
    x_train_noisy = x_train + eps_scale * rng.standard_normal(x_train.shape)
    x_test_noisy = x_test_clean + eps_scale * rng.standard_normal(x_test_clean.shape)
    scale = max(signal_rms, 1.0e-12)
    return Data(
        x_train=(x_train_noisy / scale).astype(np.float64),
        x_test=(x_test_noisy / scale).astype(np.float64),
        active_test=active_test,
        contrib_test=(contrib_test / scale).astype(np.float64),
        frames=frames,
        primitive_names=[p.name for p in primitives],
        intrinsic_dims=[p.intrinsic_dim for p in primitives],
        span_dims=[p.span_dim for p in primitives],
        topologies=[p.topology for p in primitives],
        libraries=[lib / scale for lib in libraries],
    )


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean(0, keepdims=True)) ** 2))
    if sst <= 1.0e-12:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / sst)


def _score_contrib(data: Data, block_contrib: np.ndarray, block_scores: np.ndarray) -> dict[str, object]:
    g_true = data.contrib_test.shape[1]
    g_hat = block_contrib.shape[1]
    r2 = np.full((g_true, g_hat), np.nan)
    auc = np.full((g_true, g_hat), np.nan)
    for i in range(g_true):
        rows = data.active_test[:, i]
        y = data.active_test[:, i].astype(int)
        for j in range(g_hat):
            if rows.sum() > 2:
                r2[i, j] = _r2(data.contrib_test[rows, i], block_contrib[rows, j])
            if np.unique(y).size == 2:
                auc[i, j] = roc_auc_score(y, block_scores[:, j])
    rr, cc = linear_sum_assignment(-np.nan_to_num(r2, nan=-1.0e9))
    ar, ac = linear_sum_assignment(-np.nan_to_num(auc, nan=-1.0e9))
    return {
        "contribution_r2": float(np.nanmean([r2[i, j] for i, j in zip(rr, cc)])),
        "presence_auc": float(np.nanmean([auc[i, j] for i, j in zip(ar, ac)])),
    }


def _decoder_subspaces(model: BSF) -> list[np.ndarray]:
    dec = model.decoder.detach().cpu().numpy()
    out = []
    for block in dec:
        q, _ = np.linalg.qr(block.T)
        out.append(q[:, : block.shape[0]])
    return out


def _principal_angle_score(data: Data, model: BSF) -> float:
    learned = _decoder_subspaces(model)
    score = np.zeros((len(data.frames), len(learned)))
    for i, frame in enumerate(data.frames):
        qt = frame.T
        for j, ql in enumerate(learned):
            s = np.linalg.svd(qt.T @ ql, compute_uv=False)
            score[i, j] = float(np.mean(np.clip(s, 0.0, 1.0) ** 2))
    rr, cc = linear_sum_assignment(-score)
    return float(np.mean([score[i, j] for i, j in zip(rr, cc)]))


def _score_bsf(data: Data, *, name: str, mode: str, block_size: int, n_blocks: int, k_blocks: int, cfg: Config, seed: int) -> dict[str, object]:
    model = BSF(BSFConfig(
        d_model=cfg.d_ambient,
        n_blocks=n_blocks,
        block_size=block_size,
        k_blocks=k_blocks,
        mode=mode,
        aux_k_blocks=1,
        seed=seed,
    ))
    xtr = torch.tensor(data.x_train, dtype=torch.float64)
    xte = torch.tensor(data.x_test, dtype=torch.float64)
    train_bsf(
        model,
        xtr,
        TrainConfig(steps=cfg.bsf_steps, batch_size=cfg.bsf_batch_size, lr=cfg.bsf_lr, seed=seed),
        verbose=False,
    )
    with torch.no_grad():
        out = model(xte, update_util=False)
    z = out.z_sparse.cpu().numpy()
    dec = model.decoder.detach().cpu().numpy()
    contrib = np.einsum("ngb,gbd->ngd", z, dec)
    scores = np.linalg.norm(z, axis=2)
    result = _score_contrib(data, contrib, scores)
    result.update({
        "method": name,
        "global_ev": ev(model, xte),
        "subspace_cos2": _principal_angle_score(data, model) if block_size > 1 else float("nan"),
        "topology_acc": float("nan"),
        "dimension_acc": float("nan"),
        "note": "subspace-only method; topology/dimension not identifiable from the model",
    })
    return result


def _linear_oracle(data: Data) -> dict[str, object]:
    contrib = np.zeros_like(data.contrib_test)
    for g, frame in enumerate(data.frames):
        rows = data.active_test[:, g]
        local = data.x_test[rows] @ frame.T
        contrib[rows, g] = local @ frame
    scores = np.linalg.norm(contrib, axis=2)
    result = _score_contrib(data, contrib, scores)
    result.update({
        "method": "linear_oracle_true_support",
        "global_ev": _r2(data.x_test, contrib.sum(axis=1)),
        "subspace_cos2": 1.0,
        "topology_acc": float("nan"),
        "dimension_acc": float("nan"),
        "note": "ceiling: true active set and true subspaces",
    })
    return result


def _chart_oracle(data: Data) -> dict[str, object]:
    contrib = np.zeros_like(data.contrib_test)
    for g, frame in enumerate(data.frames):
        rows = np.flatnonzero(data.active_test[:, g])
        if rows.size == 0:
            continue
        projected = (data.x_test[rows] @ frame.T) @ frame
        lib = data.libraries[g]
        # Chunked nearest-neighbor denoising onto the true manifold sample.
        out = np.zeros_like(projected)
        for start in range(0, rows.size, 512):
            p = projected[start:start + 512]
            d2 = np.sum((p[:, None, :] - lib[None, :, :]) ** 2, axis=2)
            out[start:start + 512] = lib[np.argmin(d2, axis=1)]
        contrib[rows, g] = out
    scores = np.linalg.norm(contrib, axis=2)
    result = _score_contrib(data, contrib, scores)
    result.update({
        "method": "chart_oracle_true_support",
        "global_ev": _r2(data.x_test, contrib.sum(axis=1)),
        "subspace_cos2": 1.0,
        "topology_acc": 1.0,
        "dimension_acc": 1.0,
        "note": "ceiling: true active set, true subspaces, true topology libraries",
    })
    return result


def run_one(cfg: Config, *, seed: int, noise: float, coherence: float) -> list[dict[str, object]]:
    data = make_data(cfg, seed=seed, noise=noise, coherence=coherence)
    max_b = max(data.span_dims)
    rows = []
    rows.append(_linear_oracle(data))
    rows.append(_chart_oracle(data))
    if not cfg.skip_bsf:
        rows.append(_score_bsf(data, name="topk_sae", mode="vanilla", block_size=1,
                               n_blocks=cfg.n_factors * max_b, k_blocks=cfg.active_count * max_b,
                               cfg=cfg, seed=seed))
        rows.append(_score_bsf(data, name="vanilla_bsf", mode="vanilla", block_size=max_b,
                               n_blocks=cfg.n_factors, k_blocks=cfg.active_count,
                               cfg=cfg, seed=seed))
        rows.append(_score_bsf(data, name="grassmann_bsf", mode="grassmann", block_size=max_b,
                               n_blocks=cfg.n_factors, k_blocks=cfg.active_count,
                               cfg=cfg, seed=seed))
    for row in rows:
        row.update({
            "seed": seed,
            "noise": noise,
            "coherence": coherence,
            "n_factors": cfg.n_factors,
            "d_ambient": cfg.d_ambient,
            "active_count": cfg.active_count,
            "n_train": cfg.n_train,
            "n_test": cfg.n_test,
        })
    return rows


def _parse_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x)


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Config.out_dir)
    p.add_argument("--n-factors", type=int, default=Config.n_factors)
    p.add_argument("--d-ambient", type=int, default=Config.d_ambient)
    p.add_argument("--active-count", type=int, default=Config.active_count)
    p.add_argument("--n-train", type=int, default=Config.n_train)
    p.add_argument("--n-test", type=int, default=Config.n_test)
    p.add_argument("--seeds", type=_parse_ints, default=Config.seeds)
    p.add_argument("--noises", type=_parse_floats, default=Config.noises)
    p.add_argument("--coherences", type=_parse_floats, default=Config.coherences)
    p.add_argument("--bsf-steps", type=int, default=Config.bsf_steps)
    p.add_argument("--bsf-batch-size", type=int, default=Config.bsf_batch_size)
    p.add_argument("--bsf-lr", type=float, default=Config.bsf_lr)
    p.add_argument("--skip-bsf", action="store_true")
    return Config(**vars(p.parse_args()))


def main() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for coherence in cfg.coherences:
        for noise in cfg.noises:
            for seed in cfg.seeds:
                print(f"[run] seed={seed} noise={noise} coherence={coherence}", flush=True)
                rows.extend(run_one(cfg, seed=seed, noise=noise, coherence=coherence))
                with (cfg.out_dir / "results.csv").open("w", newline="") as f:
                    fieldnames = sorted({k for row in rows for k in row})
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
    summary = {}
    for method in sorted({str(r["method"]) for r in rows}):
        mr = [r for r in rows if r["method"] == method]
        summary[method] = {
            key: float(np.nanmean([float(r[key]) for r in mr]))
            for key in ("global_ev", "contribution_r2", "presence_auc", "subspace_cos2", "topology_acc", "dimension_acc")
            if key in mr[0]
        }
    (cfg.out_dir / "summary.json").write_text(json.dumps({
        "config": {**asdict(cfg), "out_dir": str(cfg.out_dir)},
        "summary": summary,
    }, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
