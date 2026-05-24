"""Train a Manifold-SAE on Stable-Diffusion UNet residual activations.

Mirrors the cogito-L40 recipe (`manifold_sae.sae.ManifoldSAE` +
`manifold_sae.losses.total_loss`) at smaller F=128 / fewer epochs because the
SD-UNet residual is lower-dim (~640 for mid_block of SD-1.5) and we don't yet
know if the geometry is as rich.

Inputs:
    runs/COLOR_SD_UNET_MID/{X.npy, meta.json}   (from harvest_sd.py)

Outputs (under ``runs/COLOR_SD_UNET_MID/manifold_sae/`` by default):
    sae.pt          — torch state_dict of the trained ManifoldSAE
    train_log.json  — per-epoch (recon, sparsity, ortho, R²)
    z_locked.npy    — locked-mode atom positions (n_prompts, F) for downstream
                      hue-ring metric in auto_exp_77_diffusion_sae.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TrainConfig:
    in_dir: str = "runs/COLOR_SD_UNET_MID"
    out_subdir: str = "manifold_sae"
    n_features: int = 128
    n_basis: int = 16
    top_k: int = 16
    intrinsic_rank: int = 2
    sparsity_weight: float = 1e-2
    ortho_weight: float = 1e-2
    reml_weight: float = 1.0
    encoder_type: str = "linear"
    continuous_amp: bool = False
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    device: str = "mps"
    seed: int = 0


def _pick_device(prefer: str) -> str:
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sigma, mu.squeeze(0), sigma.squeeze(0)


def train(cfg: TrainConfig) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    from manifold_sae.losses import total_loss

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    in_dir = Path(cfg.in_dir)
    X = np.load(in_dir / "X.npy").astype(np.float32)
    meta = json.loads((in_dir / "meta.json").read_text())
    n, D = X.shape
    print(f"[train_sd_manifold_sae] X.shape={X.shape}  meta.D={meta.get('D')}",
          flush=True)

    Xn, mu, sigma = _normalize(X)
    var = float((Xn ** 2).mean())

    # Train / val split (80 / 20).
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]
    Xtr = torch.from_numpy(Xn[tr_idx]).float()
    Xva = torch.from_numpy(Xn[va_idx]).float()

    device = _pick_device(cfg.device)
    sae_cfg = ManifoldSAEConfig(
        input_dim=D,
        n_features=cfg.n_features,
        n_basis=cfg.n_basis,
        top_k=cfg.top_k,
        intrinsic_rank=cfg.intrinsic_rank,
        sparsity_weight=cfg.sparsity_weight,
        ortho_weight=cfg.ortho_weight,
        reml_weight=cfg.reml_weight,
        encoder_type=cfg.encoder_type,
        continuous_amp=cfg.continuous_amp,
    )
    model = ManifoldSAE(sae_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_sd_manifold_sae] F={cfg.n_features} D={D} params={n_params/1e6:.2f}M  device={device}",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(Xtr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    log: list[dict] = []
    t_train = time.time()
    for ep in range(cfg.epochs):
        model.train()
        ep_recon = 0.0; ep_n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            losses = total_loss(out, batch, sae_cfg)
            loss = losses["total"]; mse = losses["mse"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_recon += float(mse.item()) * batch.shape[0]
            ep_n += batch.shape[0]
        # Validation R².
        model.eval()
        with torch.no_grad():
            vb = Xva.to(device)
            out_v = model(vb)
            mse_v = float(torch.mean((out_v.reconstruction - vb) ** 2).item())
            r2_v = 1.0 - mse_v / var
            alive = int(((out_v.amplitudes.abs() > 1e-3).any(dim=0)).sum().item())
        row = dict(epoch=ep, train_mse=ep_recon / max(1, ep_n),
                   val_mse=mse_v, val_r2=r2_v, alive=alive)
        log.append(row)
        print(f"  [ep {ep:02d}] train_mse={row['train_mse']:.4e}  "
              f"val_mse={mse_v:.4e}  val_R2={r2_v:+.4f}  alive={alive}/{cfg.n_features}",
              flush=True)

    train_seconds = time.time() - t_train

    # Lock-and-cache snapshot on a held-out reference batch (training-mode REML
    # collapses to a single locked B at inference). z_locked is the per-prompt
    # latent atom-position matrix used by the hue-ring metric.
    model.eval()
    with torch.no_grad():
        snap = torch.from_numpy(Xn[:min(2048, n)]).float().to(device)
        try:
            model.update_snapshot(snap)
            model.inference_mode = True
        except Exception as e:
            print(f"[train_sd_manifold_sae] WARN snapshot failed: {e}", flush=True)

        all_x = torch.from_numpy(Xn).float().to(device)
        out_all = model(all_x)
        positions = out_all.positions.detach().cpu().numpy()      # (n, F)
        amplitudes = out_all.amplitudes.detach().cpu().numpy()    # (n, F)
        recon = out_all.reconstruction.detach().cpu().numpy()
        full_mse = float(((recon - Xn) ** 2).mean())
        full_r2 = 1.0 - full_mse / var

    out_dir = in_dir / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(state_dict=model.state_dict(),
             sae_cfg=sae_cfg.__dict__,
             mu=mu, sigma=sigma, D=D, F=cfg.n_features),
        out_dir / "sae.pt",
    )
    np.save(out_dir / "z_locked.npy", positions)
    np.save(out_dir / "amp_locked.npy", amplitudes)

    summary = dict(
        input_dim=D,
        n_features=cfg.n_features,
        epochs=cfg.epochs,
        train_seconds=train_seconds,
        full_val_r2=full_r2,
        last_val_r2=log[-1]["val_r2"] if log else float("nan"),
        last_alive=log[-1]["alive"] if log else 0,
        log=log,
        meta=meta,
    )
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[train_sd_manifold_sae] DONE  R²={full_r2:+.4f}  "
          f"train_seconds={train_seconds:.1f}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="runs/COLOR_SD_UNET_MID")
    ap.add_argument("--out_subdir", default="manifold_sae")
    ap.add_argument("--n_features", type=int, default=128)
    ap.add_argument("--n_basis", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
