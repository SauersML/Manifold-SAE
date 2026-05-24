"""Train a DAS-SAE on cogito-L40 with interchange-intervention loss.

Causal abstraction: hsv_name. Hue is the supervised subspace; a learned
per-feature gate aligns a subset of features with hue. The interchange loss
asks: when we splice color-a's hue-features into color-b's latent, does the
decoder reproduce the cogito-L40 activation that would arise from the
"hue(a) on name(b)" hypothetical color?

Usage:
    uv run python scripts/train_das_sae.py [--epochs 10] [--F 256] [--device mps]

Reports
  - val recon R²
  - interchange-loss curve
  - count of features with per-feature swap R² > 0.7
  - the hue-mask histogram (sigmoid of learned gate logits)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.colors as mcolors

from manifold_sae.das_sae import (
    DASSAE,
    DASSAEConfig,
    build_target_swap,
    fit_hue_direction,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_DIR = ROOT / "runs" / "DAS_SAE_COGITO_L40"
N_TEMPLATES = 28


def load_xkcd_rgb(n_colors: int) -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((
                int(hexs[0:2], 16) / 255.0,
                int(hexs[2:4], 16) / 255.0,
                int(hexs[4:6], 16) / 255.0,
            ))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float32)


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


def make_pairs(hue: np.ndarray, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pair colors so each pair has a meaningful hue gap; mix near + far."""
    n = hue.shape[0]
    a_idx = rng.integers(0, n, size=n_pairs)
    b_idx = rng.integers(0, n, size=n_pairs)
    # Reject identical indices (probability ~ 1/n is tiny but be defensive).
    same = a_idx == b_idx
    while np.any(same):
        b_idx[same] = rng.integers(0, n, size=same.sum())
        same = a_idx == b_idx
    return a_idx, b_idx


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    res = (pred - target).pow(2).sum().item()
    var = (target - target.mean(0, keepdim=True)).pow(2).sum().item()
    return 1.0 - res / max(var, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--F", type=int, default=256)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda_intv", type=float, default=1.0)
    p.add_argument("--lambda_gate", type=float, default=5e-4)
    p.add_argument("--lambda_l1", type=float, default=1e-3)
    p.add_argument("--lambda_gate_entropy", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swap_test_threshold", type=float, default=0.7)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == "cpu" else "cpu")
    print(f"[das_sae] device={device}")

    # ---- Data ----------------------------------------------------------
    print(f"[data] mmap {X_PATH}")
    X_mmap = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X_mmap.shape}")
    centroids = per_color_centroids(X_mmap, N_TEMPLATES)  # (n_c, D)
    n_c, D = centroids.shape
    print(f"[data] centroids={centroids.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = np.stack([mcolors.rgb_to_hsv(rgb[i]) for i in range(n_c)], 0).astype(np.float32)
    hue_np = hsv[:, 0]
    print(f"[data] hue range=[{hue_np.min():.3f}, {hue_np.max():.3f}]")

    # Standardize centroids per-feature so the SAE has well-scaled input.
    mu = centroids.mean(0, keepdims=True)
    sd = centroids.std(0, keepdims=True).clip(min=1e-6)
    X_std = (centroids - mu) / sd
    X_t = torch.from_numpy(X_std).to(device)
    hue_t = torch.from_numpy(hue_np).to(device)

    # Train / val split on COLORS (not templates), 85 / 15.
    perm = rng.permutation(n_c)
    n_val = max(32, n_c // 7)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    print(f"[data] train colors={len(tr_idx)} val colors={len(val_idx)}")

    # Hue direction in STANDARDIZED ambient space, fit on TRAIN only.
    v_hue = fit_hue_direction(X_t[tr_idx], hue_t[tr_idx])
    print(f"[hue] ||v_hue||={v_hue.norm().item():.3f}")
    # Quick sanity: R² of (X @ v_hue) vs hue on val.
    hue_pred_val = (X_t[val_idx] - X_t[tr_idx].mean(0, keepdim=True)) @ v_hue
    val_hue_r2 = r2(hue_pred_val.unsqueeze(-1), (hue_t[val_idx] - hue_t[tr_idx].mean()).unsqueeze(-1))
    print(f"[hue] val R²(hue from v_hue) = {val_hue_r2:.3f}")

    # ---- Model ---------------------------------------------------------
    cfg = DASSAEConfig(input_dim=D, n_features=args.F, top_k=args.top_k)
    sae = DASSAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"[model] DAS-SAE F={args.F} D={D} top_k={args.top_k} params={n_params:_}")

    # ---- Train ---------------------------------------------------------
    history = []
    n_train = len(tr_idx)
    pairs_per_epoch = max(args.batch_size * 16, n_train * 4)
    n_steps = pairs_per_epoch // args.batch_size
    print(f"[train] {args.epochs} epochs × {n_steps} steps/epoch × batch {args.batch_size}")

    t0 = time.time()
    for epoch in range(args.epochs):
        sae.train()
        ep_losses = {k: 0.0 for k in ("loss", "recon", "intv", "l1", "gate_l1", "gate_entropy")}
        for step in range(n_steps):
            a_idx, b_idx = make_pairs(hue_np[tr_idx], args.batch_size, rng)
            a_g = tr_idx[a_idx]
            b_g = tr_idx[b_idx]
            x_a = X_t[a_g]
            x_b = X_t[b_g]
            hue_a = hue_t[a_g]
            hue_b = hue_t[b_g]
            target = build_target_swap(x_a, x_b, hue_a, hue_b, v_hue)
            losses = sae.compute_loss(
                x_a, x_b, target,
                lambda_intv=args.lambda_intv,
                lambda_gate=args.lambda_gate,
                lambda_l1=args.lambda_l1,
                lambda_gate_entropy=args.lambda_gate_entropy,
            )
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder_columns_()
            for k in ep_losses:
                ep_losses[k] += float(losses[k].item())
        for k in ep_losses:
            ep_losses[k] /= n_steps
        # Val metrics.
        sae.eval()
        with torch.no_grad():
            x_val = X_t[val_idx]
            out_val = sae(x_val)
            val_recon_r2 = r2(out_val.x_hat, x_val)
            # Val interchange loss.
            a_idx, b_idx = make_pairs(hue_np[val_idx], min(64, len(val_idx)), rng)
            x_a = X_t[val_idx[a_idx]]
            x_b = X_t[val_idx[b_idx]]
            ha = hue_t[val_idx[a_idx]]
            hb = hue_t[val_idx[b_idx]]
            tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
            zv_a = sae(x_a).z
            zv_b = sae(x_b).z
            z_swap = sae.swap(zv_b, zv_a, mask=sae.hue_mask())
            x_swap_hat = sae.decode(z_swap)
            val_intv_r2 = r2(x_swap_hat, tgt)
            n_hue_soft = int((sae.hue_mask() > 0.5).sum().item())
        entry = {"epoch": epoch, "val_recon_r2": val_recon_r2,
                 "val_intv_r2": val_intv_r2, "n_hue_soft": n_hue_soft, **ep_losses}
        history.append(entry)
        print(f"[ep {epoch:02d}] loss={ep_losses['loss']:.4f} recon={ep_losses['recon']:.4f} "
              f"intv={ep_losses['intv']:.4f} | val recon R²={val_recon_r2:.3f} "
              f"intv R²={val_intv_r2:.3f} | hue_soft_n={n_hue_soft}")

    # ---- Identification: per-feature swap test on val ------------------
    print("[ident] running per-feature swap test on val pairs...")
    sae.eval()
    with torch.no_grad():
        a_idx, b_idx = make_pairs(hue_np[val_idx], min(128, len(val_idx) * 4), rng)
        x_a = X_t[val_idx[a_idx]]
        x_b = X_t[val_idx[b_idx]]
        ha = hue_t[val_idx[a_idx]]
        hb = hue_t[val_idx[b_idx]]
        tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
        scores = sae.per_feature_swap_score(x_a, x_b, tgt).cpu().numpy()
    n_identified = int((scores > args.swap_test_threshold).sum())
    print(f"[ident] features with swap R² > {args.swap_test_threshold}: {n_identified} / {args.F}")
    print(f"[ident] top-5 scores: {np.sort(scores)[-5:].tolist()}")

    # ---- Persist -------------------------------------------------------
    hue_mask_np = sae.hue_mask().detach().cpu().numpy()
    runtime = time.time() - t0
    summary = {
        "config": vars(args),
        "n_colors": int(n_c),
        "D": int(D),
        "val_hue_r2": val_hue_r2,
        "history": history,
        "final_val_recon_r2": history[-1]["val_recon_r2"],
        "final_val_intv_r2": history[-1]["val_intv_r2"],
        "n_identified_hue_features": n_identified,
        "swap_test_threshold": args.swap_test_threshold,
        "swap_scores_top10": np.sort(scores)[-10:].tolist(),
        "hue_mask_mean": float(hue_mask_np.mean()),
        "hue_mask_max": float(hue_mask_np.max()),
        "n_hue_mask_gt_0.5": int((hue_mask_np > 0.5).sum()),
        "runtime_seconds": runtime,
    }
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    np.savez(OUT_DIR / "swap_scores.npz",
             scores=scores, hue_mask=hue_mask_np)
    torch.save({"state_dict": sae.state_dict(), "config": cfg.__dict__,
                "v_hue": v_hue.cpu(), "mu": mu, "sd": sd},
               OUT_DIR / "das_sae.pt")
    print(f"[done] runtime={runtime:.1f}s")
    print(f"[done] summary -> {out_json}")
    return summary


if __name__ == "__main__":
    main()
