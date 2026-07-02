"""Sample-complexity sweep for block discovery in the block nursery.

This is intentionally lighter than experiments/block_nursery.py's full Arm A/B
benchmark.  The default path measures only the discovery step over an n grid and
seeds:

  * plane overlap after Hungarian matching to planted circle planes
  * label-free discovery stability across bootstrap resamples
  * discovered linear subspace EV, plus optional composed chart EV

Smoke:
  python experiments/block_nursery_sample_complexity.py --n-grid 120,210 --seeds 0,1 \
    --bootstrap 3 --out experiments/block_nursery/sample_complexity_smoke.json

Optional chart EV, slower:
  python experiments/block_nursery_sample_complexity.py --n-grid 120,210 --seeds 0 \
    --fit-charts --chart-steps 80 --chart-timeout 90
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from block_nursery import (
    discover_blocks,
    ev,
    fit_curved_isolated,
    make_synthetic,
    oracle_blocks,
    subspace_ev,
    train_test_split,
)


DEFAULT_N_GRID = (120, 210, 300, 480, 960, 2000)
OUT_DIR = Path(__file__).resolve().parent / "block_nursery"


def _orthonormal_union(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        raise ValueError("cannot build a subspace union from zero blocks")
    B = np.concatenate(blocks, axis=1)
    Q, R = np.linalg.qr(B)
    keep = np.abs(np.diag(R)) > 1e-8
    return Q[:, : len(keep)][:, keep]


def _subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the smaller subspace captured by the larger one, in [0, 1]."""
    if a.shape[1] == 0 or b.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.clip((s**2).sum() / min(a.shape[1], b.shape[1]), 0.0, 1.0))


def _match_score(left: list[np.ndarray], right: list[np.ndarray]) -> dict:
    if not left or not right:
        return {
            "mean": 0.0,
            "min": 0.0,
            "scores": [],
            "pairs": [],
            "n_matched": 0,
        }
    scores = np.array([[_subspace_overlap(a, b) for b in right] for a in left])
    n_match = min(len(left), len(right))
    best_rows: tuple[int, ...] = ()
    best_cols: tuple[int, ...] = ()
    best_total = -1.0
    for rows in itertools.combinations(range(len(left)), n_match):
        for cols in itertools.permutations(range(len(right)), n_match):
            total = float(sum(scores[i, j] for i, j in zip(rows, cols)))
            if total > best_total:
                best_rows, best_cols, best_total = rows, cols, total
    matched = [float(scores[i, j]) for i, j in zip(best_rows, best_cols)]
    return {
        "mean": round(float(np.mean(matched)), 4),
        "min": round(float(np.min(matched)), 4),
        "scores": [round(x, 4) for x in matched],
        "pairs": [[int(i), int(j)] for i, j in zip(best_rows, best_cols)],
        "n_matched": int(len(matched)),
    }


def _bootstrap_stability(
    X: np.ndarray,
    reference_blocks: list[np.ndarray],
    *,
    n_dict: int,
    block_size: int,
    affinity_thresh: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if n_bootstrap <= 0 or not reference_blocks:
        return {"mean": None, "min": None, "scores": []}

    rng = np.random.default_rng(seed)
    scores = []
    dims = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, X.shape[0], size=X.shape[0])
        blocks, _, _ = discover_blocks(
            X[idx],
            n_dict=n_dict,
            block_size=block_size,
            affinity_thresh=affinity_thresh,
        )
        match = _match_score(reference_blocks, blocks)
        scores.append(match["mean"])
        dims.append([int(b.shape[1]) for b in blocks])

    return {
        "mean": round(float(np.mean(scores)), 4),
        "min": round(float(np.min(scores)), 4),
        "scores": [round(float(x), 4) for x in scores],
        "bootstrap_block_dims": dims,
    }


def _chart_ev_if_requested(
    X: np.ndarray,
    blocks: list[np.ndarray],
    *,
    theta: np.ndarray,
    seed: int,
    n: int,
    enabled: bool,
    chart_steps: int,
    chart_timeout: int,
) -> dict | None:
    if not enabled:
        return None

    train_idx, test_idx = train_test_split(X.shape[0], frac=0.7, seed=seed)
    mu = X[train_idx].mean(0)
    Xc = X - mu
    composed = np.zeros_like(X)
    per_block = []
    for bi, Q in enumerate(blocks):
        Z = Xc @ Q
        fit = fit_curved_isolated(
            Z,
            n_atoms=1,
            tag=f"sample_complexity_n{n}_seed{seed}_b{bi}",
            train_idx=train_idx,
            test_idx=test_idx,
            steps=chart_steps,
            timeout=chart_timeout,
            target_k=1,
        )
        rec = {
            "block": int(bi),
            "block_dim": int(Q.shape[1]),
            "chart_status": fit["status"],
            "chart_wall_s": fit.get("wall_s"),
        }
        if fit["status"] == "CONVERGED":
            z = np.load(fit["out_path"])
            z_hat = z["x_hat"]
            rec["chart_ev_block_coords_test"] = round(ev(Z[test_idx], z_hat[test_idx]), 4)
            composed += z_hat @ Q.T
        per_block.append(rec)

    converged = [b for b in per_block if b["chart_status"] == "CONVERGED"]
    if not converged:
        return {"status": "NO_CONVERGED_BLOCKS", "per_block": per_block}

    composed_full = mu + composed
    return {
        "status": "CONVERGED",
        "composed_ambient_ev_test": round(ev(X[test_idx], composed_full[test_idx]), 4),
        "composed_ambient_ev_train": round(ev(X[train_idx], composed_full[train_idx]), 4),
        "n_blocks": len(blocks),
        "n_converged_blocks": len(converged),
        "total_fit_dim": int(sum(b.shape[1] for b in blocks)),
        "per_block": per_block,
    }


def run_one(args: argparse.Namespace, n: int, seed: int) -> dict:
    X, planes, theta, meta = make_synthetic(
        n=n,
        p=args.p,
        ncirc=args.ncirc,
        amp=args.amp,
        noise=args.noise,
        seed=seed,
    )
    truth = oracle_blocks(planes)
    blocks, _, diag = discover_blocks(
        X,
        n_dict=args.n_dict,
        block_size=args.block_size,
        affinity_thresh=args.affinity_thresh,
    )

    discovered_union = _orthonormal_union(blocks)
    truth_union = _orthonormal_union(truth)
    plane_match = _match_score(truth, blocks)
    stability = _bootstrap_stability(
        X,
        blocks,
        n_dict=args.n_dict,
        block_size=args.block_size,
        affinity_thresh=args.affinity_thresh,
        n_bootstrap=args.bootstrap,
        seed=10_000 + seed,
    )
    chart_ev = _chart_ev_if_requested(
        X,
        blocks,
        theta=theta,
        seed=seed,
        n=n,
        enabled=args.fit_charts,
        chart_steps=args.chart_steps,
        chart_timeout=args.chart_timeout,
    )

    discovered_projection = (X - X.mean(0)) @ discovered_union @ discovered_union.T + X.mean(0)
    return {
        "n": int(n),
        "seed": int(seed),
        "data": meta,
        "discovery": diag,
        "plane_overlap": plane_match,
        "union_overlap": round(_subspace_overlap(truth_union, discovered_union), 4),
        "stability": stability,
        "ev": {
            "truth_union_linear_ev": round(subspace_ev(X, truth_union), 4),
            "discovered_union_linear_ev": round(ev(X, discovered_projection), 4),
            "chart_ev": chart_ev,
        },
    }


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for n in sorted({r["n"] for r in rows}):
        group = [r for r in rows if r["n"] == n]
        out.append(
            {
                "n": int(n),
                "n_seeds": len(group),
                "plane_overlap_mean": round(
                    float(np.mean([r["plane_overlap"]["mean"] for r in group])), 4
                ),
                "plane_overlap_min": round(
                    float(np.min([r["plane_overlap"]["min"] for r in group])), 4
                ),
                "union_overlap_mean": round(float(np.mean([r["union_overlap"] for r in group])), 4),
                "stability_mean": round(
                    float(np.mean([r["stability"]["mean"] for r in group])), 4
                )
                if group[0]["stability"]["mean"] is not None
                else None,
                "discovered_union_linear_ev_mean": round(
                    float(np.mean([r["ev"]["discovered_union_linear_ev"] for r in group])), 4
                ),
                "block_dims": [r["discovery"]["block_dims"] for r in group],
            }
        )
    return out


def _parse_ints(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=_parse_ints, default=list(DEFAULT_N_GRID))
    parser.add_argument("--seeds", type=_parse_ints, default=[0, 1, 2])
    parser.add_argument("--bootstrap", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "sample_complexity_results.json")
    parser.add_argument("--p", type=int, default=96)
    parser.add_argument("--ncirc", type=int, default=3)
    parser.add_argument("--amp", type=float, default=2.0)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--n-dict", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--affinity-thresh", type=float, default=0.35)
    parser.add_argument("--fit-charts", action="store_true")
    parser.add_argument("--chart-steps", type=int, default=600)
    parser.add_argument("--chart-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n in args.n_grid:
        for seed in args.seeds:
            print(f"[run] n={n} seed={seed}", flush=True)
            rows.append(run_one(args, n, seed))
            partial = {
                "config": vars(args) | {"out": str(args.out)},
                "summary": _summarize(rows),
                "rows": rows,
                "wall_s": round(time.time() - t0, 1),
            }
            args.out.write_text(json.dumps(partial, indent=2, default=float))
            print(f"[saved] {args.out}", flush=True)

    result = {
        "config": vars(args) | {"out": str(args.out)},
        "summary": _summarize(rows),
        "rows": rows,
        "wall_s": round(time.time() - t0, 1),
    }
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({"out": str(args.out), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
