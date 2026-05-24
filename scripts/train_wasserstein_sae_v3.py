"""Train WassersteinSAEv3 (NaN-hardened) on cogito-L40.

Fixes from v2:
  - ε floor coupled to τ (eps = max(eps_floor, eps_scale * τ))
  - τ_end raised to 1.5 (was 1.0)
  - Defensive nan_to_num on π and on barycenter
  - Encoder logit clamp at ±12 BEFORE the temperature divide
  - Gradient clipping at norm 1.0 (in trainer)

Usage:
    python scripts/train_wasserstein_sae_v3.py --F 128 --M 64 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from manifold_sae.wasserstein_sae_v3 import WassersteinSAEv3


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/COLOR_COGITO_L40/X_L40.npy")
    p.add_argument("--out", default="runs/WASSERSTEIN_SAE_V3_F128_M64")
    p.add_argument("--F", type=int, default=128)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps_floor", type=float, default=0.01)
    p.add_argument("--eps_scale", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=15)
    p.add_argument("--neighbor_weight", type=float, default=1e-3)
    p.add_argument("--tau_start", type=float, default=4.0)
    p.add_argument("--tau_end", type=float, default=1.5)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[v3] device={device}", flush=True)

    X = np.load(args.data, mmap_mode="r")
    N, D = X.shape
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = int(0.05 * N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[v3] N={N} D={D} train={len(train_idx)} val={len(val_idx)}", flush=True)

    model = WassersteinSAEv3(
        F=args.F, M=args.M, D=D,
        eps_floor=args.eps_floor, eps_scale=args.eps_scale,
        n_sinkhorn_iter=args.sinkhorn_iters,
        neighbor_weight=args.neighbor_weight,
        tau_start=args.tau_start, tau_end=args.tau_end,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def iterate(idx, bs, shuffle=True):
        if shuffle:
            idx = np.random.permutation(idx)
        for i in range(0, len(idx), bs):
            chunk = idx[i:i + bs]
            yield torch.from_numpy(np.asarray(X[chunk], dtype=np.float32)).to(device)

    def val_metrics():
        model.eval()
        sse = sst = ent_sum = pi_max_sum = 0.0
        n = n_rows = 0
        with torch.no_grad():
            mean_chunk = np.asarray(X[val_idx[:4096]], dtype=np.float32).mean(0)
            mean_t = torch.from_numpy(mean_chunk).to(device)
            for batch in iterate(val_idx[:2048], args.batch_size, shuffle=False):
                out = model(batch)
                recon = out["recon"]; pi = out["pi"]
                sse += (recon - batch).pow(2).sum().item()
                sst += (batch - mean_t).pow(2).sum().item()
                n += batch.shape[0]
                ent = -(pi * torch.log(pi.clamp_min(1e-30))).sum(-1)
                ent_sum += ent.sum().item()
                pi_max_sum += pi.max(-1).values.sum().item()
                n_rows += pi.shape[0]
        model.train()
        return {
            "val_R2": 1.0 - sse / max(sst, 1e-9),
            "mean_pi_entropy": ent_sum / max(n_rows, 1),
            "mean_pi_max": pi_max_sum / max(n_rows, 1),
        }

    log = []
    t0 = time.time()
    diverged = False
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        frac = epoch / max(1, n_epochs - 1) if n_epochs > 1 else 1.0
        tau = args.tau_start + (args.tau_end - args.tau_start) * frac
        model.set_tau(tau)
        print(f"[v3] epoch={epoch} tau={tau:.3f} eps={model.current_eps():.4f}",
              flush=True)
        step = 0
        for batch in iterate(train_idx, args.batch_size):
            out = model.loss(batch)
            if not torch.isfinite(out["total"]):
                print(f"[v3] !!! NaN/Inf loss at epoch={epoch} step={step}; aborting",
                      flush=True)
                diverged = True
                break
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                pi_max = out["pi"].max(-1).values.mean().item()
                ent = -(out["pi"] * torch.log(out["pi"].clamp_min(1e-30))).sum(-1).mean().item()
                print(f"[v3] ep{epoch} step{step} tau={tau:.2f} eps={model.current_eps():.3f} "
                      f"mse={out['mse'].item():.4f} pi_max={pi_max:.3f} "
                      f"H(pi)={ent:.3f} dt={time.time()-t0:.1f}s", flush=True)
        if diverged:
            break
        val = val_metrics()
        compact = model.atom_compactness().mean().item()
        rec = {"epoch": epoch, "tau": tau, "eps": model.current_eps(),
               "mean_compactness": compact, **val}
        log.append(rec)
        print(f"[v3] epoch={epoch} val_R2={val['val_R2']:.4f} "
              f"H(pi)={val['mean_pi_entropy']:.3f} pi_max={val['mean_pi_max']:.3f} "
              f"compact={compact:.3f}", flush=True)
        torch.save({"model": model.state_dict(),
                    "config": {"F": args.F, "M": args.M, "D": D}},
                   out_dir / "checkpoint.pt")

    summary = {
        "args": vars(args),
        "log": log,
        "final": log[-1] if log else None,
        "diverged": diverged,
        "verdict_multiatom": (log[-1]["mean_pi_entropy"] > math.log(3)) if log else False,
    }
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    final = log[-1] if log else {}
    print(f"[v3] DONE diverged={diverged} R2={final.get('val_R2', float('nan')):.4f} "
          f"H(pi)={final.get('mean_pi_entropy', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()
