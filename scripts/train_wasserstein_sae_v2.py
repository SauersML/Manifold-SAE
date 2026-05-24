"""Train WassersteinSAEv2 with temperature annealing on cogito-L40.

Fixes the v1 encoder collapse (π_max → 1.0). Tau is linearly annealed from
`tau_start` → `tau_end` across epochs.

Usage:
    uv run python scripts/train_wasserstein_sae_v2.py \
        --F 128 --M 64 --epochs 10 \
        --tau_start 4.0 --tau_end 1.0 \
        --out runs/WASSERSTEIN_SAE_V2_F128_M64
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from manifold_sae.wasserstein_sae_v2 import WassersteinSAEv2


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(path: str, val_frac: float = 0.05, seed: int = 0):
    X = np.load(path, mmap_mode="r")
    N, D = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(val_frac * N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return X, train_idx, val_idx, D


def iterate_batches(X, idx, batch_size: int, device, shuffle: bool = True):
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        batch = np.asarray(X[chunk], dtype=np.float32)
        yield torch.from_numpy(batch).to(device)


def compute_val_metrics(model, X, idx, batch_size, device):
    model.eval()
    sse, sst, n = 0.0, 0.0, 0
    ent_sum, pi_max_sum, n_rows = 0.0, 0.0, 0
    with torch.no_grad():
        mean_chunk = np.asarray(X[idx[:4096]], dtype=np.float32).mean(0)
        mean_t = torch.from_numpy(mean_chunk).to(device)
        for batch in iterate_batches(X, idx, batch_size, device, shuffle=False):
            out = model(batch)
            recon = out["recon"]
            pi = out["pi"]
            sse += (recon - batch).pow(2).sum().item()
            sst += (batch - mean_t).pow(2).sum().item()
            n += batch.shape[0]
            ent = -(pi * torch.log(pi.clamp_min(1e-30))).sum(-1)
            ent_sum += ent.sum().item()
            pi_max_sum += pi.max(-1).values.sum().item()
            n_rows += pi.shape[0]
    model.train()
    r2 = 1.0 - sse / max(sst, 1e-9)
    return {
        "val_R2": r2,
        "val_mse_per_row": sse / max(n, 1),
        "mean_pi_entropy": ent_sum / max(n_rows, 1),
        "mean_pi_max": pi_max_sum / max(n_rows, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/COLOR_COGITO_L40/X_L40.npy")
    p.add_argument("--out", default="runs/WASSERSTEIN_SAE_V2_F128_M64")
    p.add_argument("--F", type=int, default=128)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--sinkhorn_iters", type=int, default=15)
    p.add_argument("--neighbor_weight", type=float, default=1e-3)
    p.add_argument("--tau_start", type=float, default=4.0)
    p.add_argument("--tau_end", type=float, default=1.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[v2] device={device}", flush=True)

    X, train_idx, val_idx, D = load_data(args.data)
    print(f"[v2] data N={X.shape[0]} D={D} train={len(train_idx)} val={len(val_idx)}",
          flush=True)

    model = WassersteinSAEv2(
        F=args.F, M=args.M, D=D,
        eps=args.eps,
        n_sinkhorn_iter=args.sinkhorn_iters,
        neighbor_weight=args.neighbor_weight,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = []
    t0 = time.time()
    step = 0
    n_epochs = max(1, args.epochs)
    for epoch in range(n_epochs):
        # Linear tau schedule (epoch-level).
        frac = epoch / max(1, n_epochs - 1) if n_epochs > 1 else 1.0
        tau = args.tau_start + (args.tau_end - args.tau_start) * frac
        model.set_tau(tau)
        print(f"[v2] epoch={epoch} tau={tau:.3f}", flush=True)

        for batch in iterate_batches(X, train_idx, args.batch_size, device):
            out = model.loss(batch)
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                pi_max = out["pi"].max(-1).values.mean().item()
                ent = -(out["pi"] * torch.log(out["pi"].clamp_min(1e-30))).sum(-1).mean().item()
                print(f"[v2] ep{epoch} step{step} tau={tau:.2f} "
                      f"mse={out['mse'].item():.4f} pi_max={pi_max:.3f} "
                      f"H(pi)={ent:.3f} (log3={math.log(3):.3f}) dt={time.time()-t0:.1f}s",
                      flush=True)
            if args.quick and step >= 100:
                break

        val = compute_val_metrics(model, X, val_idx[:2048], args.batch_size, device)
        compact = model.atom_compactness().mean().item()
        rec = {"epoch": epoch, "tau": tau, "mean_compactness": compact,
               "step": step, **val}
        log.append(rec)
        print(f"[v2] epoch={epoch} val_R2={val['val_R2']:.4f} "
              f"H(pi)={val['mean_pi_entropy']:.3f} pi_max={val['mean_pi_max']:.3f} "
              f"compact={compact:.3f}", flush=True)
        torch.save(
            {"model": model.state_dict(),
             "config": {"F": args.F, "M": args.M, "D": D, "eps": args.eps,
                        "tau_start": args.tau_start, "tau_end": args.tau_end}},
            out_dir / "checkpoint.pt",
        )
        if args.quick:
            break

    atoms = model.atoms().detach().cpu().numpy()
    compact_arr = model.atom_compactness().detach().cpu().numpy()
    np.savez(out_dir / "atoms.npz", atoms=atoms, compactness=compact_arr)

    summary = {
        "args": vars(args),
        "log": log,
        "final": log[-1] if log else None,
        "verdict_multiatom": (log[-1]["mean_pi_entropy"] > math.log(3)) if log else False,
        "log_3_nats": math.log(3),
    }
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    final = log[-1] if log else {}
    print(f"[v2] DONE  R2={final.get('val_R2', float('nan')):.4f}  "
          f"H(pi)={final.get('mean_pi_entropy', float('nan')):.3f}  "
          f"pi_max={final.get('mean_pi_max', float('nan')):.3f}  "
          f"multi_atom={summary['verdict_multiatom']}", flush=True)


if __name__ == "__main__":
    main()
