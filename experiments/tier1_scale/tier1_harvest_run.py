#!/usr/bin/env python3
"""WS-C Tier-1 at scale over a residual_shard_io bf16 harvest (the real path).

Streams a WS-D harvest directory (Qwen3-32B residual activations, bf16 memmap)
through gamfit.SparseDictStream at large K with a small active budget, holds out
a deterministic ~fraction of rows (row-hash split; works for a single growing
shard or many), checkpoints the decoder every epoch (resumable), evaluates
held-out EV, and exports a content-hashed dictionary artifact with T0 baked in.

Robust to a LIVE-GROWING harvest: shard row counts are derived from on-disk
file size with an explicit-shape memmap (bytes = rows * d_model * 2), so a stale
provisional manifest whose declared rows lag the growing last shard never
crashes the reshape. Only d_model and T0 are taken from manifest.json; T0 comes
from manifest["t0"] (mean/std/rms/rogue_dims/scale) or the writer's
manifest["stats"] (mean/norm) fallback.

Math lives in the Rust core (SparseDictStream); this is thin orchestration.
CLI-flag driven (no env-var toggles).
"""
import argparse, glob, hashlib, json, os, time
from pathlib import Path
import numpy as np

_DISK = np.dtype("<u2")  # bf16 bit-patterns, little-endian


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bf16_to_f32(bits):
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


class Harvest:
    """Self-contained robust reader over shard_*.bf16 files.

    Row counts are computed from file size (not the manifest), so a stale
    provisional manifest never breaks reads. Memmaps use an explicit shape so a
    file that grows after open is read only up to its size-at-open.
    """

    def __init__(self, out_dir):
        self.dir = out_dir
        with open(os.path.join(out_dir, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        self.d_model = int(self.manifest["d_model"])
        self.files = sorted(glob.glob(os.path.join(out_dir, "shard_*.bf16")))
        self.rows = [os.path.getsize(f) // (self.d_model * 2) for f in self.files]
        self.total = int(sum(self.rows))

    def _mm(self, i):
        return np.memmap(self.files[i], dtype=_DISK, mode="r",
                         shape=(self.rows[i], self.d_model))

    def batches(self, n):
        parts, have = [], 0
        for i in range(len(self.files)):
            if self.rows[i] == 0:
                continue
            mm = self._mm(i)
            pos, R = 0, self.rows[i]
            while pos < R:
                take = min(n - have, R - pos)
                parts.append(np.asarray(mm[pos:pos + take]))
                have += take
                pos += take
                if have == n:
                    yield bf16_to_f32(parts[0] if len(parts) == 1 else np.concatenate(parts, 0))
                    parts, have = [], 0
        if have:
            yield bf16_to_f32(parts[0] if len(parts) == 1 else np.concatenate(parts, 0))

    def seed_sample(self, n):
        """~n rows drawn evenly across shards (bf16->f32)."""
        if self.total == 0:
            return np.empty((0, self.d_model), dtype=np.float32)
        out, remaining = [], min(n, self.total)
        for i in range(len(self.files)):
            if remaining <= 0 or self.rows[i] == 0:
                break
            quota = min(self.rows[i], max(int(round(n * self.rows[i] / self.total)), 1), remaining)
            out.append(bf16_to_f32(np.asarray(self._mm(i)[:quota])))
            remaining -= quota
        return np.concatenate(out, 0) if out else np.empty((0, self.d_model), dtype=np.float32)


def get_t0(manifest):
    t0src = manifest.get("t0") or manifest.get("T0")
    stats = manifest.get("stats")
    mean = scale = rogue = None
    if t0src:
        if t0src.get("mean") is not None:
            mean = np.asarray(t0src["mean"], dtype=np.float32)
        for k in ("scale", "std", "rms", "norm"):
            if t0src.get(k) is not None:
                scale = np.asarray(t0src[k], dtype=np.float32); break
        rogue = t0src.get("rogue_dims")
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
    ap.add_argument("--k", type=int, default=32768)
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--score-tile", type=int, default=8192)
    ap.add_argument("--code-ridge", type=float, default=1e-6)
    ap.add_argument("--decoder-ridge", type=float, default=1e-6)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--seed-rows", type=int, default=100000)
    ap.add_argument("--heldout-stride", type=int, default=20)
    ap.add_argument("--heldout-cap", type=int, default=200000)
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import gamfit
    log(f"gamfit {gamfit.__version__}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "ckpt"; ckpt_dir.mkdir(exist_ok=True)
    ckpt_dec = ckpt_dir / "decoder.npy"; ckpt_meta = ckpt_dir / "meta.json"

    hv = Harvest(args.harvest_dir)
    P = hv.d_model
    t0 = get_t0(hv.manifest)
    log(f"harvest {args.harvest_dir}: files={len(hv.files)} rows={hv.total} P={P} "
        f"t0_mean={'y' if t0['mean'] is not None else 'n'} "
        f"t0_scale={'y' if t0['scale'] is not None else 'n'} "
        f"rogue={None if t0.get('rogue_dims') is None else len(t0['rogue_dims'])}")

    config = dict(k=args.k, active=args.active, minibatch=args.minibatch,
                  score_tile=args.score_tile, code_ridge=args.code_ridge,
                  decoder_ridge=args.decoder_ridge, tolerance=args.tolerance,
                  center=args.center, heldout_stride=args.heldout_stride)

    def train_held_batches(mb):
        counter = 0
        held_taken = 0
        for b in hv.batches(mb):
            m = b.shape[0]
            hmask = (np.arange(counter, counter + m) % args.heldout_stride == 0)
            if held_taken >= args.heldout_cap:
                hmask[:] = False
            counter += m
            tr = b[~hmask]
            hd = b[hmask]
            if hd.shape[0] and held_taken < args.heldout_cap:
                hd = hd[: args.heldout_cap - held_taken]; held_taken += hd.shape[0]
            else:
                hd = None
            yield tr, hd

    start_epoch, ev_history, resumed = 0, [], False
    if args.resume and ckpt_dec.exists() and ckpt_meta.exists():
        seed = np.load(ckpt_dec).astype(np.float32)
        meta = json.loads(ckpt_meta.read_text())
        start_epoch = meta["epoch"]; ev_history = meta.get("ev_history", [])
        resumed = True
        log(f"RESUME epoch {start_epoch} decoder {seed.shape} last_EV "
            f"{ev_history[-1] if ev_history else 'NA'}")
    else:
        seed = apply_t0(hv.seed_sample(args.seed_rows), t0, args.center)
        log(f"seed sample {seed.shape}")

    stream = gamfit.SparseDictStream(
        seed, args.k, active=args.active, minibatch=args.minibatch,
        max_epochs=args.max_epochs, score_tile=args.score_tile,
        code_ridge=args.code_ridge, decoder_ridge=args.decoder_ridge,
        tolerance=args.tolerance)
    log(f"stream K={args.k} active={stream.active} P={P}")

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
        "harvest_dir": args.harvest_dir, "model": hv.manifest.get("model", "Qwen3-32B"),
        "K": args.k, "active_L0": int(stream.active), "P": int(P),
        "total_tokens_available": int(hv.total),
        "train_ev_final": float(art.explained_variance),
        "ev_history": [float(e) for e in ev_history],
        "heldout_ev": float(ho_ev), "heldout_rows": int(ho_n),
        "epochs_run": int(art.epochs), "converged": bool(art.converged),
        "total_rows_seen": int(total_rows), "wall_s": round(run_s, 1),
        "throughput_rows_per_s": round(throughput, 1),
        "t0_center": args.center,
        "rogue_dims_count": None if t0.get("rogue_dims") is None else len(t0["rogue_dims"]),
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
