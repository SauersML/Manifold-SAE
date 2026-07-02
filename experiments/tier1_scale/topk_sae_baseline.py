"""External TopK SAE baseline (standard, PyTorch) at matched K and L0.

The comparison arm for WS-C acceptance: train a canonical TopK sparse autoencoder
(tied-bias pre-encoder centering, unit-norm decoder atoms, exact top-K activation)
on the SAME manifest shards with the SAME T0 centering as the streaming Tier-1
fit, then report held-out centered EV. This is an external reference implementation
(not part of gam); it exists only to benchmark the streaming dictionary.

CLI-flag driven. Runs on GPU if visible, else CPU.
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_manifest(mp):
    man = json.load(open(mp))
    base = Path(mp).parent
    train, held = [], []
    for e in man["shards"]:
        p = Path(e["path"]); p = p if p.is_absolute() else base / p
        (held if e.get("split") in ("heldout", "val", "test") else train).append(p)
    t0 = man.get("t0", {})
    return train, held, t0


def _bf16_to_f32(bits):
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_harvest(harvest_dir, residual_io, cap, heldout_stride, heldout_cap=200000):
    """Load a residual_shard_io bf16 harvest into (train_np, held_np, t0).

    Robust to a live-growing harvest: shard rows are derived from on-disk file
    size with an explicit-shape memmap (never trusts stale manifest row counts).
    Row-hash held-out split matches the T1 runner; train capped at `cap` rows."""
    import glob, os
    with open(os.path.join(harvest_dir, "manifest.json")) as fh:
        man = json.load(fh)
    P = int(man["d_model"])
    t0src = man.get("t0") or man.get("stats") or {}
    t0 = {}
    if t0src.get("mean") is not None:
        t0["mean"] = np.asarray(t0src["mean"], dtype=np.float32)
    for k in ("scale", "std", "rms", "norm"):
        if t0src.get(k) is not None:
            t0["scale"] = np.asarray(t0src[k], dtype=np.float32); break
    files = sorted(glob.glob(os.path.join(harvest_dir, "shard_*.bf16")))
    tr, hd = [], []
    counter = 0; ntr = 0; nhd = 0
    for f in files:
        rows = os.path.getsize(f) // (P * 2)
        if rows == 0:
            continue
        mm = np.memmap(f, dtype=np.dtype("<u2"), mode="r", shape=(rows, P))
        for pos in range(0, rows, 4096):
            b = _bf16_to_f32(np.asarray(mm[pos:pos + 4096]))
            m = b.shape[0]
            hmask = (np.arange(counter, counter + m) % heldout_stride == 0)
            counter += m
            if ntr < cap:
                t = b[~hmask][: cap - ntr]; tr.append(t); ntr += t.shape[0]
            if nhd < heldout_cap:
                h = b[hmask][: heldout_cap - nhd]; hd.append(h); nhd += h.shape[0]
            if ntr >= cap and nhd >= heldout_cap:
                break
        if ntr >= cap and nhd >= heldout_cap:
            break
    return np.concatenate(tr, 0), (np.concatenate(hd, 0) if hd else None), t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--harvest-dir")
    ap.add_argument("--residual-io", default="/models/sauers_build/gam_fable/examples")
    ap.add_argument("--cap", type=int, default=1000000, help="max train rows for harvest mode")
    ap.add_argument("--heldout-stride", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--active", type=int, default=32)   # L0
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} device={dev}")

    if args.harvest_dir:
        Xtr_np, Xho_np, t0 = load_harvest(args.harvest_dir, args.residual_io,
                                          args.cap, args.heldout_stride)
    else:
        train_p, held_p, t0 = load_manifest(args.manifest)
        Xtr_np = np.concatenate([np.load(p) for p in train_p], 0)
        Xho_np = np.concatenate([np.load(p) for p in held_p], 0) if held_p else None
    mean = torch.tensor(np.asarray(t0.get("mean"), dtype=np.float32)) if t0.get("mean") is not None else None
    scale = torch.tensor(np.asarray(t0.get("scale"), dtype=np.float32)) if t0.get("scale") is not None else None

    def prep(x):
        x = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
        if args.center and mean is not None:
            x = x - mean
        if args.center and scale is not None:
            x = x / torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        return x

    Xtr = prep(Xtr_np)
    Xho = prep(Xho_np) if Xho_np is not None else None
    N, P = Xtr.shape
    K = args.k
    log(f"train {tuple(Xtr.shape)} heldout {tuple(Xho.shape) if Xho is not None else None} K={K} L0={args.active}")

    g = torch.Generator().manual_seed(0)
    b_dec = Xtr.mean(0).clone()
    W_dec = torch.nn.functional.normalize(torch.randn(K, P, generator=g), dim=1)
    W_enc = W_dec.clone()                      # tied init
    b_enc = torch.zeros(K)
    # Data stays on CPU (a full harvest is far larger than free GPU memory on a
    # co-tenant B200); only the model, optimizer state, and the current minibatch
    # live on the device.
    Xtr = Xtr.contiguous()
    if Xho is not None: Xho = Xho.contiguous()
    W_dec, W_enc, b_enc, b_dec = (t.to(dev).detach().requires_grad_(True) for t in (W_dec, W_enc, b_enc, b_dec))
    opt = torch.optim.Adam([W_enc, W_dec, b_enc, b_dec], lr=args.lr)

    def encode_decode(x):
        z = (x - b_dec) @ W_enc.t() + b_enc
        topv, topi = z.topk(args.active, dim=1)
        topv = torch.relu(topv)
        acts = torch.zeros_like(z).scatter_(1, topi, topv)
        xhat = acts @ W_dec + b_dec
        return xhat

    t0t = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), generator=g, device="cpu")
        x = Xtr[idx].to(dev)
        xhat = encode_decode(x)
        loss = ((x - xhat) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():                  # keep decoder atoms unit-norm
            W_dec.grad -= (W_dec.grad * W_dec).sum(1, keepdim=True) * W_dec
        opt.step()
        with torch.no_grad():
            W_dec.data = torch.nn.functional.normalize(W_dec.data, dim=1)
        if step % 500 == 0 or step == args.steps - 1:
            log(f"step {step} loss={loss.item():.4f}")

    def ev(X):
        with torch.no_grad():
            sse = 0.0
            mu = X.mean(0)
            sst = float(((X - mu) ** 2).sum())
            for i in range(0, X.shape[0], args.batch):
                xb = X[i:i + args.batch].to(dev)
                sse += float(((xb - encode_decode(xb)) ** 2).sum())
            return 1.0 - sse / sst if sst > 0 else float("nan")

    tr_ev = ev(Xtr)
    ho_ev = ev(Xho) if Xho is not None else float("nan")
    wall = time.time() - t0t
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {"baseline": "external_topk_sae", "K": K, "active_L0": args.active, "P": int(P),
           "steps": args.steps, "train_ev": tr_ev, "heldout_ev": ho_ev,
           "wall_s": round(wall, 1), "device": dev}
    (out / "topk_baseline_result.json").write_text(json.dumps(res, indent=2))
    log(f"BASELINE_DONE train_ev={tr_ev:.5f} heldout_ev={ho_ev:.5f}")
    print("BASELINE_DONE " + json.dumps({"heldout_ev": ho_ev, "K": K, "L0": args.active}))


if __name__ == "__main__":
    main()
