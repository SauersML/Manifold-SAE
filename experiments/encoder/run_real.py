"""WS-E on a REAL composed dictionary + REAL corpus sweep.

Consumes:
  * a SAC-composed dictionary artifact from WS-A (``--dictionary``), loaded as a
    ``gamfit.ManifoldSAE`` (:func:`distill_harness.load_dictionary`);
  * corpus activation shards + per-row token-frequency metadata from the WS-D
    manifest (``--manifest``). Rows are the layer activations the dictionary was
    fit on; token frequencies come from the manifest's T0 / token-count stats.

Emits the same :class:`distill_harness.EncoderReport` as the synthetic path:
agreement, certificate-fallback (overall + by token-frequency decile), and
amortized-encode throughput.

    python run_real.py --dictionary /dev/shm/sauers_gpu/sac_w6/composed.json \
        --manifest /dev/shm/sauers_gpu/harvest/MANIFEST.json \
        --layer 0 --train-rows 200000 --eval-rows 200000 \
        --out /dev/shm/sauers_gpu/encoder/real_report.json

Until the WS-D manifest exists this script is the WIRED-AND-READY real path; the
decile analysis activates automatically once ``token_freq`` is present. No env
vars, no wall-clock budgets (SPEC.md).
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


def _load_shard(path: str) -> np.ndarray:
    """Load one activation shard as an (N, p) float array.

    Supports ``.npy`` and ``.npz`` (first array); the WS-D ShardWriter format is
    resolved from the manifest's ``format`` field when present.
    """
    p = Path(path)
    if p.suffix == ".npy":
        return np.ascontiguousarray(np.load(p), dtype=np.float32)
    if p.suffix == ".npz":
        with np.load(p) as z:
            return np.ascontiguousarray(z[list(z.keys())[0]], dtype=np.float32)
    raise ValueError(f"unsupported shard format: {path}")


def load_corpus_from_manifest(
    manifest_path: str,
    *,
    layer: int,
    train_rows: int,
    eval_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Assemble (X_train, X_eval, token_freq_eval) from the WS-D manifest.

    The manifest is expected to enumerate per-layer activation shards and, per
    the WS-D task, T0 stats plus token metadata. This reader is tolerant to the
    exact schema: it looks for a ``layers`` map (or ``shards`` list) pointing at
    activation files, and for a per-row ``token_freq`` / ``token_id`` +
    ``token_counts`` vocabulary to derive frequencies. Missing token metadata
    yields ``token_freq_eval = None`` (overall fallback only; decile wired).
    """
    manifest = json.loads(Path(manifest_path).read_text())
    base = Path(manifest_path).parent

    # ---- locate this layer's activation shards -------------------------------
    shard_paths: list[str] = []
    layers = manifest.get("layers")
    if isinstance(layers, dict):
        entry = layers.get(str(layer)) or layers.get(layer)
        if entry is None:
            raise KeyError(f"layer {layer} not in manifest layers {list(layers)}")
        shard_paths = list(entry.get("shards", entry if isinstance(entry, list) else []))
    elif isinstance(manifest.get("shards"), list):
        shard_paths = [
            s["path"] if isinstance(s, dict) else s
            for s in manifest["shards"]
            if (not isinstance(s, dict)) or s.get("layer", layer) == layer
        ]
    if not shard_paths:
        raise ValueError("no activation shards found in manifest for this layer")
    shard_paths = [str((base / p) if not Path(p).is_absolute() else p) for p in shard_paths]

    need = train_rows + eval_rows
    chunks: list[np.ndarray] = []
    got = 0
    for sp in shard_paths:
        arr = _load_shard(sp)
        chunks.append(arr)
        got += arr.shape[0]
        if got >= need:
            break
    X = np.concatenate(chunks, axis=0)[:need]
    if X.shape[0] < need:
        raise ValueError(f"manifest shards supplied {X.shape[0]} rows < requested {need}")
    X_train = np.ascontiguousarray(X[:train_rows])
    X_eval = np.ascontiguousarray(X[train_rows : train_rows + eval_rows])

    # ---- per-row token frequency for the eval rows (optional) ----------------
    token_freq_eval: np.ndarray | None = None
    tok = manifest.get("token_freq") or manifest.get("token_frequencies")
    if tok is not None:
        freq = np.asarray(tok, dtype=np.float64)
        if freq.shape[0] >= need:
            token_freq_eval = freq[train_rows : train_rows + eval_rows]
    else:
        # derive from per-row token ids + a vocabulary count vector
        ids = manifest.get("token_id") or manifest.get("token_ids")
        counts = manifest.get("token_counts") or manifest.get("vocab_counts")
        if ids is not None and counts is not None:
            ids_arr = np.asarray(ids, dtype=np.int64)
            counts_arr = np.asarray(counts, dtype=np.float64)
            total = counts_arr.sum()
            if ids_arr.shape[0] >= need and total > 0:
                freq_all = counts_arr[ids_arr] / total
                token_freq_eval = freq_all[train_rows : train_rows + eval_rows]
    return X_train, X_eval, token_freq_eval


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WS-E real dictionary + corpus sweep")
    ap.add_argument("--dictionary", required=True, help="SAC composed-dictionary artifact (gamfit ManifoldSAE)")
    ap.add_argument("--manifest", required=True, help="WS-D MANIFEST.json")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--train-rows", type=int, default=200_000)
    ap.add_argument("--eval-rows", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--throughput-rows", type=int, default=2_000_000)
    ap.add_argument("--gate-rows-per-s", type=float, default=1.0e5)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    _log(f"[WS-E] loading dictionary {args.dictionary}")
    model = dh.load_dictionary(args.dictionary)
    _log(f"[WS-E] dictionary K={len(model.atoms)} assignment={model.assignment}")

    t0 = time.perf_counter()
    X_train, X_eval, token_freq = load_corpus_from_manifest(
        args.manifest, layer=args.layer,
        train_rows=args.train_rows, eval_rows=args.eval_rows,
    )
    _log(f"[WS-E] corpus: train={X_train.shape} eval={X_eval.shape} "
         f"token_freq={'present' if token_freq is not None else 'absent'} "
         f"({time.perf_counter()-t0:.1f}s)")

    _enc, report = dh.distill_and_gate(
        model, X_train, X_eval,
        dictionary_source=f"SAC composed: {args.dictionary} (layer {args.layer})",
        token_freq=token_freq,
        hidden=tuple(args.hidden), epochs=args.epochs,
        random_state=args.random_state,
        throughput_target_rows=args.throughput_rows,
        throughput_gate_rows_per_s=args.gate_rows_per_s,
    )
    _log("\n===== WS-E ENCODER REPORT (REAL) =====")
    _log(report.summary())
    _log("======================================")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dh.write_report(report, args.out)
        _log(f"[WS-E] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
