#!/usr/bin/env python3
"""WS-C Tier-1 at scale over a residual_shard_io bf16 harvest (the real path).

Streams a WS-D harvest directory (Qwen3-32B residual activations, bf16 memmap)
through gamfit.SparseDictStream at large K with a small active budget, holds out
a deterministic ~fraction of rows (row-hash split; works for a single growing
shard or many), checkpoints the decoder every epoch (resumable), evaluates
held-out EV, and exports a content-hashed dictionary artifact with T0 baked in.

Reader contract: residual_shard_io.load_shards(dir).batches(n) -> float32
(<=n, d_model); T0 in reader.manifest["t0"] (mean/std/rms/rogue_dims/scale) or
the writer's reader.manifest["stats"] (mean/norm) fallback.

Math lives in the Rust core (SparseDictStream); this is thin orchestration.
CLI-flag driven (no env-var toggles).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_t0(manifest):
    """Return {mean, scale} float32 arrays from the harvest manifest.

    Prefers the WS-D 't0' block (mean + scale/std/rms); falls back to the
    ShardWriter 'stats' block (mean + norm). Missing pieces are left as None.
    """
    t0 = manifest.get("t0") or manifest.get("T0")
    stats = manifest.get("stats")
    mean = scale = None
    rogue = None
    if t0:
        if t0.get("mean") is not None:
            mean = np.asarray(t0["mean"], dtype=np.float32)
        for k in ("scale", "std", "rms", "norm"):
            if t0.get(k) is not None:
                scale = np.asarray(t0[k], dtype=np.float32); break
        rogue = t0.get("rogue_dims")
    elif stats:
        if stats.get("mean") is not None:
            mean = np.asarray(stats["mean"], dtype=np.float32)
        if stats.get("norm") is not None:
            scale = np.asarray(stats["norm"], dtype=np.float32)
    if scale is not None:
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return {"mean": mean, "scale": scale, "rogue_dims": rogue}


def apply_t0(x, t0, center):
    if not center:
        return np.ascontiguousarray(x, dtype=np.float32)
    if t0.get("mean") is not None:
        x = x - t0["mean"]
    if t0.get("scale") is not None:
        x = x / t0["scale"]
    return np.ascontiguousarray(x, dtype=np.float32)


def content_hash(decoder, t0, config):
    D = np.asarray(decoder, dtype=np.float32)
    order = np.lexsort(np.round(D, 5).T[::-1])
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(D[order]).tobytes())
    h.update(json.dumps({k: config[k] for k in sorted(config)}, sort_keys=True).encode())
    for k in ("mean", "scale"):
        if t0.get(k) is not None:
            h.update(k.encode()); h.update(np.asarray(t0[k], dtype=np.float32).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--residual-io", default="/models/sauers_build/gam_fable/examples",
                    help="dir containing residual_shard_io.py")
    ap.add_argument("--k", type=int, default=32768)
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--score-tile", type=int, default=8192)
    ap.add_argument("--code-ridge", type=float, default=1e-6)
    ap.add_argument("--decoder-ridge", type=float, default=1e-6)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--seed-rows", type=int, default=300000)
    ap.add_argument("--heldout-stride", type=int, default=20, help="1/N rows held out")
    ap.add_argument("--heldout-cap", type=int, default=200000)
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.residual_io)
    from residual_shard_io import load_shards, stratified_subsample
    import gamfit
    log(f"gamfit {gamfit.__version__}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "ckpt"; ckpt_dir.mkdir(exist_ok=True)
    ckpt_dec = ckpt_dir / "decoder.npy"; ckpt_meta = ckpt_dir / "meta.json"

    reader = load_shards(args.harvest_dir)
    P = reader.d_model
    t0 = get_t0(reader.manifest)
    log(f"harvest {args.harvest_dir}: total_tokens={reader.total_tokens} P={P} "
        f"shards={len(reader.shards)} t0_mean={'y' if t0['mean'] is not None else 'n'} "
        f"t0_scale={'y' if t0['scale'] is not None else 'n'} rogue={t0.get('rogue_dims')}")

    config = dict(k=args.k, active=args.active, minibatch=args.minibatch,
                  score_tile=args.score_tile, code_ridge=args.code_ridge,
                  decoder_ridge=args.decoder_ridge, tolerance=args.tolerance,
                  center=args.center, heldout_stride=args.heldout_stride)

    def train_held_batches(mb):
        """Yield (train_rows, held_rows_or_None) applying the row-hash split."""
        counter = 0
        held_taken = 0
        for b in reader.batches(mb):
            m = b.shape[0]
            idx = np.arange(counter, counter + m)
            hmask = (idx % args.heldout_stride == 0)
            if held_taken >= args.heldout_cap:
                hmask[:] = False
            counter += m
            tr = b[~hmask]
            hd = b[hmask]
            if hd.shape[0] and held_taken < args.heldout_cap:
                room = args.heldout_cap - held_taken
                hd = hd[:room]; held_taken += hd.shape[0]
            else:
                hd = None
            yield tr, hd

    start_epoch = 0
    ev_history = []
    resumed = False
    if args.resume and ckpt_dec.exists() and ckpt_meta.exists():
        seed = np.load(ckpt_dec).astype(np.float32)
        meta = json.loads(ckpt_meta.read_text())
        start_epoch = meta["epoch"]; ev_history = meta.get("ev_history", [])
        resumed = True
        log(f"RESUME epoch {start_epoch} decoder {seed.shape} last_EV "
            f"{ev_history[-1] if ev_history else 'NA'}")
    else:
        seed = apply_t0(stratified_subsample(reader, args.seed_rows), t0, args.center)
        log(f"seed sample {seed.shape} (stratified)")

    stream = gamfit.SparseDictStream(
        seed, args.k, active=args.active, minibatch=args.minibatch,
        max_epochs=args.max_epochs, score_tile=args.score_tile,
        code_ridge=args.code_ridge, decoder_ridge=args.decoder_ridge,
        tolerance=args.tolerance)
    log(f"stream K={args.k} active={stream.active} P={P}")

    # Collect the held-out set once (epoch 0); it is identical every epoch.
    held_chunks = []
    t_run0 = time.time(); total_rows = 0
    for epoch in range(start_epoch, args.max_epochs):
        t_ep = time.time(); ep_rows = 0
        collect = (epoch == start_epoch and not resumed)
        for tr, hd in train_held_batches(args.minibatch):
            if tr.shape[0]:
                st = stream.partial_fit(apply_t0(tr, t0, args.center))
                ep_rows += st["rows"]; total_rows += st["rows"]
            if collect and hd is not None and hd.shape[0]:
                held_chunks.append(apply_t0(hd, t0, args.center))
        stats = stream.end_epoch()
        ev_history.append(stats["explained_variance"])
        dt = time.time() - t_ep; tps = ep_rows / dt if dt > 0 else 0
        log(f"epoch {epoch}: EV={stats['explained_variance']:.5f} "
            f"alive={args.k - stats['dead']} revived={stats['revived']} rows={ep_rows} "
            f"{dt:.1f}s {tps:,.0f} rows/s conv={stats['converged']}")
        np.save(ckpt_dec, stream.decoder)
        ckpt_meta.write_text(json.dumps(
            {"epoch": epoch + 1, "ev_history": ev_history, "config": config,
             "total_rows_seen": total_rows}, indent=2))
        if stats["converged"]:
            log(f"converged at epoch {epoch}"); break

    art = stream.finalize()
    run_s = time.time() - t_run0
    throughput = total_rows / run_s if run_s > 0 else 0
    log(f"finalize EV={art.explained_variance:.5f} decoder={art.decoder.shape} "
        f"conv={art.converged} epochs={art.epochs} {throughput:,.0f} rows/s")

    ho_ev, ho_n = float("nan"), 0
    if held_chunks:
        H = np.concatenate(held_chunks, 0)
        D = art.decoder
        idx, cod = art.transform(H)
        recon = np.zeros_like(H)
        for j in range(idx.shape[1]):
            recon += cod[:, j:j + 1] * D[idx[:, j]]
        mu = H.mean(0)
        sst = float(((H - mu) ** 2).sum())
        sse = float(((H - recon) ** 2).sum())
        ho_ev = 1.0 - sse / sst if sst > 0 else float("nan")
        ho_n = H.shape[0]
        log(f"held-out EV={ho_ev:.5f} over {ho_n} rows (L0={stream.active})")

    chash = content_hash(art.decoder, t0, config)
    np.save(out / "decoder.npy", art.decoder)
    result = {
        "workstream": "WS-C tier1 (qwen3 harvest)", "gamfit_version": gamfit.__version__,
        "harvest_dir": args.harvest_dir, "model": reader.manifest.get("model", "Qwen3-32B"),
        "K": args.k, "active_L0": int(stream.active), "P": int(P),
        "total_tokens_available": int(reader.total_tokens),
        "train_ev_final": float(art.explained_variance),
        "ev_history": [float(e) for e in ev_history],
        "heldout_ev": float(ho_ev), "heldout_rows": int(ho_n),
        "epochs_run": int(art.epochs), "converged": bool(art.converged),
        "total_rows_seen": int(total_rows), "wall_s": round(run_s, 1),
        "throughput_rows_per_s": round(throughput, 1),
        "t0_center": args.center, "rogue_dims": t0.get("rogue_dims"),
        "content_hash": chash, "config": config, "resumed": resumed,
    }
    (out / "tier1_result.json").write_text(json.dumps(result, indent=2))
    npz = {"decoder": art.decoder}
    for k in ("mean", "scale"):
        if t0.get(k) is not None:
            npz[f"t0_{k}"] = np.asarray(t0[k], dtype=np.float32)
    np.savez(out / "dictionary_artifact.npz", **npz,
             content_hash=np.array(chash), config=np.array(json.dumps(config)))
    log(f"WROTE {out}/tier1_result.json hash={chash[:16]}")
    print("TIER1_DONE " + json.dumps({"train_ev": result["train_ev_final"],
          "heldout_ev": result["heldout_ev"], "K": args.k, "L0": result["active_L0"],
          "throughput_rows_per_s": result["throughput_rows_per_s"],
          "total_rows_seen": result["total_rows_seen"], "hash": chash[:16]}), flush=True)


if __name__ == "__main__":
    main()
