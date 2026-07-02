"""WS-E on a REAL WS-D activation harvest (K=1 chart), end to end.

A complete real-data validation that needs neither SAC nor the big corpus: fit
the proven K=1 manifold chart on a WS-D residual harvest (a probe set such as
``qwen3_32b_probe_weekday_l24``, or a slice of ``qwen3_32b_fineweb_l24``),
distil the amortized encoder from the certified exact solves, and report
agreement / certificate-fallback / throughput.

For a CYCLIC probe (``manifest["cyclic"] == True`` with per-row ``order``), it
additionally checks that the recovered 1-D circle coordinate reproduces the
ground-truth cyclic ordering of the labels (the W7 pattern) â€” a real semantics
check on top of the encode-machinery check.

    python run_probe.py --dir /dev/shm/sauers_gpu/harvest/qwen3_32b_probe_weekday_l24 \
        --topology circle --d-atom 1 --out /dev/shm/sauers_gpu/encoder/weekday_l24.json
    python run_probe.py --dir /dev/shm/sauers_gpu/harvest/qwen3_32b_fineweb_l24 \
        --train-rows 20000 --eval-rows 20000 --out /dev/shm/sauers_gpu/encoder/fineweb_l24.json

Reads shards through the canonical ``residual_shard_io.ShardReader`` (no
bespoke format code). No env vars, no wall-clock budgets (SPEC.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_harness as dh          # noqa: E402


def _log(*a: object) -> None:
    print(*a)
    sys.stdout.flush()


def _import_shard_reader():
    for cand in ("/models/sauers_build/gam_fable/examples", "/Users/user/gam/examples"):
        if (Path(cand) / "residual_shard_io.py").is_file():
            sys.path.insert(0, cand)
            break
    from residual_shard_io import ShardReader  # noqa: E402

    return ShardReader


def load_probe(dir_path: str, *, max_rows: int | None = None) -> tuple[np.ndarray, dict]:
    """Load a residual-shard harvest dir as (X float32 (N,d), manifest)."""
    ShardReader = _import_shard_reader()
    reader = ShardReader(dir_path)
    parts: list[np.ndarray] = []
    got = 0
    for batch in reader.batches(8192):
        b = np.ascontiguousarray(np.asarray(batch, dtype=np.float32))
        parts.append(b)
        got += b.shape[0]
        if max_rows is not None and got >= max_rows:
            break
    X = np.concatenate(parts, axis=0)
    if max_rows is not None:
        X = X[:max_rows]
    return np.ascontiguousarray(X), dict(reader.manifest)


def _cyclic_order_agreement(coord: np.ndarray, order: np.ndarray) -> dict[str, float]:
    """How well a recovered 1-D circle coordinate reproduces the ground-truth
    cyclic order. Maps each row's coordinate to an angle, sorts, and measures the
    fraction of adjacent label-pairs whose angular order matches the cyclic
    ground truth up to a global rotation + reflection (the circle's gauge)."""
    theta = np.mod(np.asarray(coord, dtype=np.float64).ravel(), 2.0 * np.pi)
    order = np.asarray(order, dtype=np.int64).ravel()
    uniq = np.unique(order)
    n_lab = uniq.shape[0]
    # mean angle per label (circular mean), then check the label sequence.
    mean_ang = np.array([
        np.angle(np.mean(np.exp(1j * theta[order == u]))) for u in uniq
    ])
    ranks = np.argsort(np.mod(mean_ang, 2.0 * np.pi))
    seq = uniq[ranks]
    # best cyclic-rotation + direction match against the natural order 0..n-1.
    best = 0
    for direction in (seq, seq[::-1]):
        arr = np.concatenate([direction, direction])
        for s in range(n_lab):
            window = arr[s : s + n_lab]
            match = int(np.sum(window == np.arange(n_lab)))
            best = max(best, match)
    return {
        "n_labels": float(n_lab),
        "cyclic_order_matches": float(best),
        "cyclic_order_fraction": float(best / n_lab) if n_lab else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WS-E real WS-D harvest K=1 validation")
    ap.add_argument("--dir", required=True, help="residual_shard harvest dir")
    ap.add_argument("--topology", default="circle")
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--assignment", default="ibp_map")
    ap.add_argument("--train-rows", type=int, default=None, help="teacher rows (default: all but eval)")
    ap.add_argument("--eval-rows", type=int, default=None, help="held-out rows (default: min(train, N/2))")
    ap.add_argument("--fit-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--throughput-rows", type=int, default=1_000_000)
    ap.add_argument("--gate-rows-per-s", type=float, default=1.0e5)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    import gamfit

    max_rows = None
    if args.train_rows is not None and args.eval_rows is not None:
        max_rows = args.train_rows + args.eval_rows
    t0 = time.perf_counter()
    X, manifest = load_probe(args.dir, max_rows=max_rows)
    n, d = X.shape
    _log(f"[WS-E] loaded {n} rows x {d} from {args.dir} "
         f"(model={manifest.get('model_name')} layer={manifest.get('layer')} "
         f"probe={manifest.get('probe_set', manifest.get('text_dataset'))}) "
         f"({time.perf_counter()-t0:.1f}s)")

    # split teacher / held-out
    if args.train_rows is None:
        n_eval = args.eval_rows if args.eval_rows is not None else max(1, n // 2)
        n_train = n - n_eval
    else:
        n_train = args.train_rows
        n_eval = args.eval_rows if args.eval_rows is not None else (n - n_train)
    n_train = min(n_train, n - 1)
    n_eval = min(n_eval, n - n_train)
    X_train = np.ascontiguousarray(X[:n_train])
    X_eval = np.ascontiguousarray(X[n_train : n_train + n_eval])
    _log(f"[WS-E] split: teacher={X_train.shape} eval={X_eval.shape}")

    t0 = time.perf_counter()
    model = gamfit.sae_manifold_fit(
        X_train, K=1, d_atom=args.d_atom, atom_topology=args.topology,
        assignment=args.assignment, n_iter=args.fit_iter, random_state=args.random_state,
    )
    _log(f"[WS-E] K=1 {args.topology} fit R2={getattr(model,'reconstruction_r2',float('nan')):.4f} "
         f"({time.perf_counter()-t0:.1f}s)")

    _enc, report = dh.distill_and_gate(
        model, X_train, X_eval,
        dictionary_source=(
            f"REAL {manifest.get('model_name')} L{manifest.get('layer')} "
            f"{manifest.get('probe_set', manifest.get('text_dataset'))} "
            f"K=1 {args.topology}"
        ),
        token_freq=None,  # probe sets carry labels, not corpus frequencies
        hidden=tuple(args.hidden), epochs=args.epochs, random_state=args.random_state,
        throughput_target_rows=args.throughput_rows,
        throughput_gate_rows_per_s=args.gate_rows_per_s,
        notes=[f"real WS-D harvest {Path(args.dir).name}"],
    )

    # ground-truth cyclic-order science (weekday/month/year/color probes)
    extra: dict = {}
    if manifest.get("cyclic") and "order" in manifest:
        order_full = np.asarray(manifest["order"], dtype=np.int64)
        # recover the exact teacher coordinate for ALL rows and score ordering.
        coords_all = model.converged_latents(np.ascontiguousarray(X))["coords"][0]
        extra = _cyclic_order_agreement(coords_all[:, 0], order_full[: coords_all.shape[0]])
        _log(f"[WS-E] cyclic-order recovery: {extra}")

    _log("\n===== WS-E ENCODER REPORT (REAL PROBE) =====")
    _log(report.summary())
    if extra:
        _log(f"cyclic-order recovery: matches={extra['cyclic_order_matches']:.0f}/"
             f"{extra['n_labels']:.0f} (fraction={extra['cyclic_order_fraction']:.3f})")
    _log("============================================")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["cyclic_order_recovery"] = extra
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
        _log(f"[WS-E] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
