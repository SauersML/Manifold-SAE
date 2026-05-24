"""Train WassersteinSAE on cogito-L40 activations.

Compares against the Manifold-SAE baseline (F=512, R²=0.913) at 4× fewer
atoms (F=128). Goal: equivalent or better R² because each atom is
intrinsically multi-modal (a hue distribution rather than a point).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from manifold_sae.wasserstein_sae import WassersteinSAE


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
        # mmap → in-memory: cast to float32 and move to device
        batch = np.asarray(X[chunk], dtype=np.float32)
        yield torch.from_numpy(batch).to(device)


def compute_r2(model, X, idx, batch_size: int, device) -> tuple[float, float]:
    model.eval()
    sse, sst, n = 0.0, 0.0, 0
    with torch.no_grad():
        # Compute mean on a chunk to avoid loading 26K × 7K floats.
        # Use first 4K of idx as proxy for global mean.
        mean_chunk = np.asarray(X[idx[:4096]], dtype=np.float32).mean(0)
        mean_t = torch.from_numpy(mean_chunk).to(device)
        for batch in iterate_batches(X, idx, batch_size, device, shuffle=False):
            out = model(batch)
            recon = out["recon"]
            sse += (recon - batch).pow(2).sum().item()
            sst += (batch - mean_t).pow(2).sum().item()
            n += batch.shape[0]
    model.train()
    return 1.0 - sse / max(sst, 1e-9), sse / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/COLOR_COGITO_L40/X_L40.npy")
    p.add_argument("--out", default="runs/WASSERSTEIN_SAE_F128_M64")
    p.add_argument("--F", type=int, default=128)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--sinkhorn_iters", type=int, default=15)
    p.add_argument("--neighbor_weight", type=float, default=1e-3)
    p.add_argument("--quick", action="store_true", help="One epoch, 200 batches")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[wsae] device={device}")

    X, train_idx, val_idx, D = load_data(args.data)
    print(f"[wsae] data: N={X.shape[0]}, D={D}, train={len(train_idx)}, val={len(val_idx)}")

    model = WassersteinSAE(
        F=args.F, M=args.M, D=D,
        eps=args.eps,
        n_sinkhorn_iter=args.sinkhorn_iters,
        neighbor_weight=args.neighbor_weight,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in iterate_batches(X, train_idx, args.batch_size, device):
            out = model.loss(batch)
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                pi_max = out["pi"].max(-1).values.mean().item()
                print(f"[wsae] ep{epoch} step{step} mse={out['mse'].item():.4f} "
                      f"neigh={out['neighbor'].item():.4f} pi_max={pi_max:.3f} "
                      f"dt={time.time()-t0:.1f}s")
            if args.quick and step >= 200:
                break
        r2, val_mse = compute_r2(model, X, val_idx[:2048], args.batch_size, device)
        compact = model.atom_compactness().mean().item()
        print(f"[wsae] epoch={epoch} val_R2={r2:.4f} val_mse={val_mse:.4f} "
              f"mean_compact={compact:.3f}")
        log.append({"epoch": epoch, "val_R2": r2, "val_mse": val_mse,
                    "mean_compactness": compact, "step": step})
        torch.save(
            {"model": model.state_dict(),
             "config": {"F": args.F, "M": args.M, "D": D, "eps": args.eps}},
            out_dir / "checkpoint.pt",
        )
        if args.quick:
            break

    # Final atoms + per-atom compactness
    atoms = model.atoms().detach().cpu().numpy()
    compact = model.atom_compactness().detach().cpu().numpy()
    np.savez(out_dir / "atoms.npz", atoms=atoms, compactness=compact)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"args": vars(args), "log": log,
                   "baseline_R2_F512": 0.913}, f, indent=2)
    print(f"[wsae] DONE — final R2={log[-1]['val_R2']:.4f} "
          f"compactness={log[-1]['mean_compactness']:.3f} "
          f"(baseline R2=0.913 at F=512; this is F={args.F})")


if __name__ == "__main__":
    main()
