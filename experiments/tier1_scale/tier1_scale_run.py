#!/usr/bin/env python3
"""WS-C Tier-1 at scale: streaming SparseDictStream fit over a sharded harvest.

Consumes a WS-D manifest (JSON) describing bf16 activation shards and T0 stats,
runs the collapsed-linear-lane streaming trainer (gamfit.SparseDictStream) at
large K with a small active budget over multiple epochs, checkpoints the decoder
at every epoch boundary so a killed job resumes, evaluates held-out EV, and
exports a content-hashed dictionary artifact with T0 stats baked in.

Math lives in the Rust core (SparseDictStream); this is thin orchestration.
CLI-flag driven (no env-var toggles).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(manifest_path):
    """Return (train_shards, heldout_shards, P, t0) from a WS-D manifest.

    Manifest schema is read defensively: we accept a list of shard entries under
    any of a few likely keys, each entry a path (str) or {path, tokens, split}.
    T0 (mean / rogue dims / scale) is taken from the manifest if present.
    """
    with open(manifest_path) as fh:
        man = json.load(fh)
    base = Path(manifest_path).parent
    shard_entries = None
    for key in ("shards", "files", "train_shards", "chunks"):
        if key in man:
            shard_entries = man[key]
            break
    if shard_entries is None:
        raise ValueError(f"manifest has no shard list (keys={list(man)})")

    def entry_path(e):
        p = e if isinstance(e, str) else e.get("path") or e.get("file")
        p = Path(p)
        return p if p.is_absolute() else (base / p)

    def entry_split(e):
        if isinstance(e, dict):
            return e.get("split")
        return None

    train, held = [], []
    for e in shard_entries:
        sp = entry_split(e)
        (held if sp in ("heldout", "val", "test") else train).append(entry_path(e))
    # If the manifest declared no explicit split, hold out the last shard.
    if not held and len(train) > 1:
        held = [train.pop()]

    P = man.get("P") or man.get("d_model") or man.get("hidden")
    t0 = man.get("t0") or man.get("T0") or {}
    dtype = man.get("dtype", "bfloat16")
    layer = man.get("layer", man.get("layers"))
    log(f"manifest: {len(train)} train + {len(held)} heldout shards, P={P}, "
        f"dtype={dtype}, layer={layer}, t0_keys={list(t0)}")
    return train, held, P, t0, dtype


def load_shard(path, dtype, P):
    """Load one activation shard as float32 (N x P). bf16 is upcast on load."""
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path)
    else:
        raw = np.fromfile(path, dtype=_np_dtype(dtype))
        if P is None:
            raise ValueError("raw shard needs P from manifest")
        a = raw.reshape(-1, P)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def _np_dtype(name):
    name = (name or "float32").lower()
    if name in ("bf16", "bfloat16"):
        # numpy has no native bf16; harvest writes bf16 as uint16 payloads with a
        # sibling .npy when float. If we ever hit raw bf16 we widen via ml_dtypes.
        try:
            import ml_dtypes  # noqa
            return ml_dtypes.bfloat16
        except Exception:
            raise ValueError("raw bf16 shard requires ml_dtypes; ask WS-D for .npy fp shards")
    return {"float16": np.float16, "fp16": np.float16, "float32": np.float32,
            "fp32": np.float32}.get(name, np.float32)


def apply_t0(x, t0, center):
    """Center/scale rows by T0 stats (baked so encode can reverse)."""
    if not center:
        return x
    mean = t0.get("mean")
    scale = t0.get("scale")
    if mean is not None:
        x = x - np.asarray(mean, dtype=np.float32)
    if scale is not None:
        s = np.asarray(scale, dtype=np.float32)
        s = np.where(s > 0, s, 1.0).astype(np.float32)
        x = x / s
    return np.ascontiguousarray(x, dtype=np.float32)


def heldout_ev(artifact, shards, dtype, P, t0, center):
    """Centered explained variance of the frozen decoder on held-out shards."""
    sse = 0.0
    sst = 0.0
    # Two-pass would need the held-out grand mean; accumulate against per-shard
    # mean-removed totals via running sums (sufficient: EV is scale-free here).
    tot_sum = None
    tot_n = 0
    tot_sq = 0.0
    recon_sse = 0.0
    # First pass: grand mean over held-out.
    per_shard = []
    for sp in shards:
        x = apply_t0(load_shard(sp, dtype, P), t0, center)
        per_shard.append(x)
        if tot_sum is None:
            tot_sum = x.sum(0)
        else:
            tot_sum += x.sum(0)
        tot_n += x.shape[0]
        tot_sq += float((x ** 2).sum())
    mean = tot_sum / max(tot_n, 1)
    D = artifact.decoder
    for x in per_shard:
        idx, cod = artifact.transform(x)
        recon = np.zeros_like(x)
        for j in range(idx.shape[1]):
            recon += cod[:, j:j + 1] * D[idx[:, j]]
        recon_sse += float(((x - recon) ** 2).sum())
    sst = tot_sq - tot_n * float((mean ** 2).sum())
    ev = 1.0 - recon_sse / sst if sst > 0 else float("nan")
    return ev, tot_n


def content_hash(decoder, t0, config):
    """Deterministic content hash: canonicalize atom order, then sha256 bytes."""
    D = np.asarray(decoder, dtype=np.float32)
    # Canonical atom order: sort by a rounded fingerprint (first few coords +
    # norm signature) so seed/threading permutations hash identically.
    key = np.round(D, 5)
    order = np.lexsort(key.T[::-1])
    Dc = np.ascontiguousarray(D[order])
    h = hashlib.sha256()
    h.update(Dc.tobytes())
    h.update(json.dumps({k: config[k] for k in sorted(config)}, sort_keys=True).encode())
    for k in sorted(t0):
        v = t0[k]
        h.update(k.encode())
        h.update(np.asarray(v, dtype=np.float32).tobytes() if not isinstance(v, (int, float, str)) else str(v).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32768)
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--score-tile", type=int, default=8192)
    ap.add_argument("--code-ridge", type=float, default=1e-6)
    ap.add_argument("--decoder-ridge", type=float, default=1e-6)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--seed-rows", type=int, default=200000,
                    help="rows from the first shard used to seed atom directions")
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import gamfit
    log(f"gamfit {gamfit.__version__}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_dec = ckpt_dir / "decoder.npy"
    ckpt_meta = ckpt_dir / "meta.json"

    train, held, P, t0, dtype = load_manifest(args.manifest)
    if not train:
        log("no train shards; abort")
        sys.exit(2)

    config = dict(k=args.k, active=args.active, minibatch=args.minibatch,
                  score_tile=args.score_tile, code_ridge=args.code_ridge,
                  decoder_ridge=args.decoder_ridge, tolerance=args.tolerance,
                  center=args.center)

    start_epoch = 0
    ev_history = []
    resumed = False
    if args.resume and ckpt_dec.exists() and ckpt_meta.exists():
        seed = np.load(ckpt_dec).astype(np.float32)
        meta = json.loads(ckpt_meta.read_text())
        start_epoch = meta["epoch"]
        ev_history = meta.get("ev_history", [])
        log(f"RESUME from epoch {start_epoch}, decoder {seed.shape}, last EV "
            f"{ev_history[-1] if ev_history else 'NA'}")
        resumed = True
    else:
        seed = apply_t0(load_shard(train[0], dtype, P), t0, args.center)
        if seed.shape[0] > args.seed_rows:
            seed = np.ascontiguousarray(seed[:args.seed_rows])
        log(f"seed sample {seed.shape} from {train[0].name}")

    if P is None:
        P = seed.shape[1]

    stream = gamfit.SparseDictStream(
        seed, args.k, active=args.active, minibatch=args.minibatch,
        max_epochs=args.max_epochs, score_tile=args.score_tile,
        code_ridge=args.code_ridge, decoder_ridge=args.decoder_ridge,
        tolerance=args.tolerance)
    log(f"stream constructed K={args.k} active={stream.active} P={P}")

    t_run0 = time.time()
    total_rows_seen = 0
    for epoch in range(start_epoch, args.max_epochs):
        t_ep = time.time()
        ep_rows = 0
        for si, sp in enumerate(train):
            x = apply_t0(load_shard(sp, dtype, P), t0, args.center)
            st = stream.partial_fit(x)
            ep_rows += st["rows"]
            total_rows_seen += st["rows"]
        stats = stream.end_epoch()
        ev_history.append(stats["explained_variance"])
        dt = time.time() - t_ep
        tps = ep_rows / dt if dt > 0 else 0
        log(f"epoch {epoch}: EV={stats['explained_variance']:.5f} "
            f"alive={args.k - stats['dead']} revived={stats['revived']} "
            f"rows={ep_rows} {dt:.1f}s {tps:,.0f} rows/s conv={stats['converged']}")
        # checkpoint at epoch boundary (resumable)
        np.save(ckpt_dec, stream.decoder)
        ckpt_meta.write_text(json.dumps(
            {"epoch": epoch + 1, "ev_history": ev_history, "config": config,
             "total_rows_seen": total_rows_seen}, indent=2))
        if stats["converged"]:
            log(f"converged at epoch {epoch}")
            break

    art = stream.finalize()
    run_s = time.time() - t_run0
    throughput = total_rows_seen / run_s if run_s > 0 else 0
    log(f"finalize: EV={art.explained_variance:.5f} decoder={art.decoder.shape} "
        f"converged={art.converged} epochs={art.epochs} {throughput:,.0f} rows/s")

    # held-out EV
    ho_ev, ho_n = (float("nan"), 0)
    if held:
        ho_ev, ho_n = heldout_ev(art, held, dtype, P, t0, args.center)
        log(f"held-out EV={ho_ev:.5f} over {ho_n} rows (L0={stream.active})")

    # export dictionary artifact with T0 baked in + content hash
    chash = content_hash(art.decoder, t0, config)
    np.save(out / "decoder.npy", art.decoder)
    result = {
        "workstream": "WS-C tier1",
        "gamfit_version": gamfit.__version__,
        "K": args.k,
        "active_L0": int(stream.active),
        "P": int(P),
        "train_shards": [str(s) for s in train],
        "heldout_shards": [str(s) for s in held],
        "train_ev_final": float(art.explained_variance),
        "ev_history": [float(e) for e in ev_history],
        "heldout_ev": float(ho_ev),
        "heldout_rows": int(ho_n),
        "epochs_run": int(art.epochs),
        "converged": bool(art.converged),
        "total_rows_seen": int(total_rows_seen),
        "wall_s": round(run_s, 1),
        "throughput_rows_per_s": round(throughput, 1),
        "t0_baked": {k: (v if isinstance(v, (int, float, str)) else "array") for k, v in t0.items()},
        "content_hash": chash,
        "config": config,
        "resumed": resumed,
    }
    (out / "tier1_result.json").write_text(json.dumps(result, indent=2))
    # bake a single-file artifact bundle (decoder + T0 + hash)
    npz = {"decoder": art.decoder}
    for k, v in t0.items():
        if not isinstance(v, (int, float, str)):
            npz[f"t0_{k}"] = np.asarray(v, dtype=np.float32)
    np.savez(out / "dictionary_artifact.npz", **npz,
             content_hash=np.array(chash), config=np.array(json.dumps(config)))
    log(f"WROTE {out}/tier1_result.json hash={chash[:16]}")
    print("TIER1_DONE " + json.dumps({"train_ev": result["train_ev_final"],
          "heldout_ev": result["heldout_ev"], "K": args.k, "L0": result["active_L0"],
          "throughput_rows_per_s": result["throughput_rows_per_s"],
          "hash": chash[:16]}), flush=True)


if __name__ == "__main__":
    main()
