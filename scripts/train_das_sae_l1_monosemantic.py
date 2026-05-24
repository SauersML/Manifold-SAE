"""DAS-SAE sweep WITHOUT TopK: find regime where a single feature carries hue.

Follow-up to scripts/train_das_sae.py + auto_exp_65: the prior agent found
single-feature swap R² ≈ 0 under TopK (atom i is rarely in b's active set).
This script sweeps over the L1-only regime (top_k=None) with small F and
high λ_intv, asking whether ANY config produces a feature with single-feature
swap R² > 0.5 — the falsifier for "hue is genuinely distributed in cogito-L40".

Grid:
  F           ∈ {64, 128, 256}
  λ_intv      ∈ {1, 5, 25, 100}
  λ_l1        ∈ {1e-3, 1e-2}

Per config: 5 epochs, then per-feature swap test. Reports best single-feature
R² + count of features with R² > 0.5. Saves grid.json + grid_heatmap.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from manifold_sae.das_sae import (
    DASSAE,
    DASSAEConfig,
    build_target_swap,
    fit_hue_direction,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_DIR = ROOT / "runs" / "das_sae_l1_sweep"
N_TEMPLATES = 28


def load_xkcd_rgb(n_colors: int) -> np.ndarray:
    rgb = []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            hexs = parts[1].lstrip("#")
            rgb.append((
                int(hexs[0:2], 16) / 255.0,
                int(hexs[2:4], 16) / 255.0,
                int(hexs[4:6], 16) / 255.0,
            ))
    return np.asarray(rgb[:n_colors], dtype=np.float32)


def per_color_centroids(X: np.ndarray, n_t: int) -> np.ndarray:
    n_rows, d = X.shape
    n_c = n_rows // n_t
    out = np.empty((n_c, d), dtype=np.float32)
    block = 64
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        chunk = np.asarray(X[cs * n_t : ce * n_t], dtype=np.float32)
        chunk = chunk.reshape(ce - cs, n_t, d).mean(axis=1)
        out[cs:ce] = chunk
    return out


def make_pairs(n: int, n_pairs: int, rng: np.random.Generator):
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    same = a == b
    while np.any(same):
        b[same] = rng.integers(0, n, size=same.sum())
        same = a == b
    return a, b


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    res = (pred - target).pow(2).sum().item()
    var = (target - target.mean(0, keepdim=True)).pow(2).sum().item()
    return 1.0 - res / max(var, 1e-12)


def train_one(
    X_t: torch.Tensor,
    hue_t: torch.Tensor,
    v_hue: torch.Tensor,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    F: int,
    lambda_intv: float,
    lambda_l1: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    D = X_t.shape[1]
    cfg = DASSAEConfig(
        input_dim=D,
        n_features=F,
        top_k=None,                   # L1-only regime
        init_gate_logit=-2.0,
        normalize_decoder=True,
    )
    sae = DASSAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    n_train = len(tr_idx)
    n_steps = max(8, n_train * 4 // batch_size)

    for epoch in range(epochs):
        sae.train()
        for _ in range(n_steps):
            a, b = make_pairs(n_train, batch_size, rng)
            a_g = tr_idx[a]
            b_g = tr_idx[b]
            x_a = X_t[a_g]
            x_b = X_t[b_g]
            ha = hue_t[a_g]
            hb = hue_t[b_g]
            tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
            losses = sae.compute_loss(
                x_a, x_b, tgt,
                lambda_intv=lambda_intv,
                lambda_gate=5e-4,
                lambda_l1=lambda_l1,
                lambda_gate_entropy=1e-2,
            )
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder_columns_()

    # Per-feature swap-R² eval on val pairs.
    sae.eval()
    with torch.no_grad():
        a, b = make_pairs(len(val_idx), min(128, len(val_idx) * 4), rng)
        x_a = X_t[val_idx[a]]
        x_b = X_t[val_idx[b]]
        ha = hue_t[val_idx[a]]
        hb = hue_t[val_idx[b]]
        tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
        scores = sae.per_feature_swap_score(x_a, x_b, tgt).cpu().numpy()
        # Recon R² on val.
        out_val = sae(X_t[val_idx])
        val_recon = r2(out_val.x_hat, X_t[val_idx])
        # Active features (mean fraction nonzero per row).
        z = out_val.z
        active_frac = float((z.abs() > 1e-6).float().mean().item())

    return {
        "F": F,
        "lambda_intv": lambda_intv,
        "lambda_l1": lambda_l1,
        "max_single_feature_r2": float(scores.max()),
        "n_features_r2_gt_0p5": int((scores > 0.5).sum()),
        "n_features_r2_gt_0p7": int((scores > 0.7).sum()),
        "val_recon_r2": val_recon,
        "active_frac": active_frac,
        "top10_scores": [float(s) for s in np.sort(scores)[-10:]],
        "scores": scores.tolist(),
        "state_dict_ref": None,                  # filled later if winner
        "_sae_state": sae.state_dict(),
        "_cfg": cfg.__dict__,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_winner", action="store_true", default=True)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[sweep] device={device}")

    # ---- Data ----------------------------------------------------------
    X_mmap = np.load(X_PATH, mmap_mode="r")
    centroids = per_color_centroids(X_mmap, N_TEMPLATES)
    n_c, D = centroids.shape
    print(f"[data] centroids={centroids.shape}")

    rgb = load_xkcd_rgb(n_c)
    hsv = np.stack([mcolors.rgb_to_hsv(rgb[i]) for i in range(n_c)], 0).astype(np.float32)
    hue_np = hsv[:, 0]

    mu = centroids.mean(0, keepdims=True)
    sd = centroids.std(0, keepdims=True).clip(min=1e-6)
    X_std = (centroids - mu) / sd
    X_t = torch.from_numpy(X_std).to(device)
    hue_t = torch.from_numpy(hue_np).to(device)

    perm = rng.permutation(n_c)
    n_val = max(32, n_c // 7)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    v_hue = fit_hue_direction(X_t[tr_idx], hue_t[tr_idx])

    # ---- Sweep ---------------------------------------------------------
    Fs = [64, 128, 256]
    intvs = [1.0, 5.0, 25.0, 100.0]
    l1s = [1e-3, 1e-2]
    results = []
    t0 = time.time()
    for F in Fs:
        for li in intvs:
            for l1 in l1s:
                t_cfg = time.time()
                print(f"[sweep] F={F} λ_intv={li} λ_l1={l1} …", flush=True)
                res = train_one(
                    X_t, hue_t, v_hue, tr_idx, val_idx,
                    F=F, lambda_intv=li, lambda_l1=l1,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=device, rng=rng,
                )
                dt = time.time() - t_cfg
                print(
                    f"        max_r2={res['max_single_feature_r2']:.3f} "
                    f"n>0.5={res['n_features_r2_gt_0p5']} "
                    f"recon={res['val_recon_r2']:.3f} "
                    f"active={res['active_frac']:.3f} ({dt:.1f}s)"
                )
                results.append(res)
    total = time.time() - t0
    print(f"[sweep] done {len(results)} cfgs in {total:.1f}s")

    # ---- Winner --------------------------------------------------------
    winner = max(results, key=lambda r: r["max_single_feature_r2"])
    print(
        f"[winner] F={winner['F']} λ_intv={winner['lambda_intv']} "
        f"λ_l1={winner['lambda_l1']} max_r2={winner['max_single_feature_r2']:.3f} "
        f"n>0.5={winner['n_features_r2_gt_0p5']}"
    )

    # ---- Save grid -----------------------------------------------------
    grid_serializable = []
    for r in results:
        d = {k: v for k, v in r.items() if not k.startswith("_") and k != "scores"}
        grid_serializable.append(d)
    grid_json = {
        "Fs": Fs, "intvs": intvs, "l1s": l1s,
        "epochs": args.epochs,
        "n_colors": int(n_c), "D": int(D),
        "runtime_seconds": total,
        "winner": {k: v for k, v in winner.items() if not k.startswith("_") and k != "scores"},
        "grid": grid_serializable,
    }
    (OUT_DIR / "grid.json").write_text(json.dumps(grid_json, indent=2))

    # Heatmap: one subplot per λ_l1; rows=F, cols=λ_intv; cell=max_single_r2.
    fig, axes = plt.subplots(1, len(l1s), figsize=(5 * len(l1s), 4), squeeze=False)
    for j, l1 in enumerate(l1s):
        mat = np.zeros((len(Fs), len(intvs)))
        for i_F, F in enumerate(Fs):
            for k_iv, iv in enumerate(intvs):
                for r in results:
                    if r["F"] == F and r["lambda_intv"] == iv and r["lambda_l1"] == l1:
                        mat[i_F, k_iv] = r["max_single_feature_r2"]
        ax = axes[0][j]
        im = ax.imshow(mat, cmap="viridis", vmin=-0.1, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(intvs)), [str(iv) for iv in intvs])
        ax.set_yticks(range(len(Fs)), [str(F) for F in Fs])
        ax.set_xlabel("λ_intv")
        ax.set_ylabel("F")
        ax.set_title(f"max single-feature swap R² (λ_l1={l1})")
        for i_F in range(len(Fs)):
            for k_iv in range(len(intvs)):
                ax.text(k_iv, i_F, f"{mat[i_F, k_iv]:.2f}",
                        ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im, ax=ax)
    fig.suptitle("DAS-SAE L1-only sweep on cogito-L40 (5 epochs)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grid_heatmap.png", dpi=120)
    plt.close(fig)

    # Save winner model.
    if args.save_winner:
        torch.save({
            "state_dict": winner["_sae_state"],
            "config": winner["_cfg"],
            "v_hue": v_hue.cpu(),
            "mu": mu, "sd": sd,
            "winner_meta": {k: v for k, v in winner.items() if not k.startswith("_")},
        }, OUT_DIR / "winner.pt")

    print(f"[done] -> {OUT_DIR}")
    return grid_json


if __name__ == "__main__":
    main()
