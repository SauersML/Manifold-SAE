"""Train CylinderSAE on cogito-L40 cache.

Cylinder (S^1 x R) won auto_exp_67 topology selection on cogito-L40 by
ΔREML > 140 over Torus and > 1500 over Euclidean. This trains the
Cylinder-native Manifold-SAE: F=512 atoms, 15 epochs, MPS.

Usage:
    uv run python scripts/train_cylinder_sae.py

Saves: runs/CYLINDER_SAE_COGITO/{state.pt, metrics.json, log.json}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from manifold_sae.cylinder_sae import CylinderSAE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "CYLINDER_SAE_COGITO"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)

# -------------------- data --------------------
X_path = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
X_mm = np.load(X_path, mmap_mode="r")
N, D = X_mm.shape
print(f"[data] X shape={X_mm.shape} dtype={X_mm.dtype}", flush=True)

N_COLORS = 949
N_TPL = 28
rng = np.random.default_rng(0)
color_perm = rng.permutation(N_COLORS)
n_val_colors = int(0.2 * N_COLORS)
val_colors = set(color_perm[:n_val_colors].tolist())

row_color = np.arange(N) // N_TPL
train_idx = np.where(~np.isin(row_color, list(val_colors)))[0]
val_idx = np.where(np.isin(row_color, list(val_colors)))[0]

X_train_np = np.ascontiguousarray(X_mm[train_idx]).astype(np.float32)
X_val_np = np.ascontiguousarray(X_mm[val_idx]).astype(np.float32)
mu = X_train_np.mean(0)
X_train_np -= mu
X_val_np -= mu

X_train = torch.from_numpy(X_train_np)
X_val = torch.from_numpy(X_val_np).to(DEVICE)
print(f"[data] train={X_train.shape} val={X_val.shape}", flush=True)

# -------------------- model --------------------
F = 512
H = 3
K_ELL = 4
TOP_K = 32

model = CylinderSAE(
    input_dim=D,
    n_features=F,
    fourier_harm=H,
    lightness_basis_k=K_ELL,
    top_k=TOP_K,
    sparsity_weight=1e-3,
    ard_weight=1e-3,
    hidden_dim=512,  # shared encoder, no per-feature blowup post-0.1.123 migration
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[model] CylinderSAE F={F} H={H} K_ell={K_ELL} top_k={TOP_K} "
      f"params={n_params:,}", flush=True)

# -------------------- training --------------------
EPOCHS = 15
BATCH = 256
LR = 3e-4

opt = torch.optim.Adam(model.parameters(), lr=LR)

train_curve = []  # avg train loss per epoch
val_curve = []    # val recon (MSE) per epoch
val_r2_curve = [] # val R² per epoch
k_eff_curve = []
dead_curve = []

t0 = time.time()
N_train = X_train.shape[0]
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(N_train)
    ep_loss = 0.0
    ep_recon = 0.0
    ep_k = 0.0
    ep_dead = 0.0
    nb = 0
    for i in range(0, N_train, BATCH):
        idx = perm[i:i + BATCH]
        xb = X_train[idx].to(DEVICE)
        loss, info = model.loss(xb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += float(loss.detach())
        ep_recon += float(info["recon"].detach())
        ep_k += float(info["k_eff"].detach())
        ep_dead += float(info["dead_rate"].detach())
        nb += 1
    ep_loss /= max(1, nb)
    ep_recon /= max(1, nb)
    ep_k /= max(1, nb)
    ep_dead /= max(1, nb)

    # val
    model.eval()
    with torch.no_grad():
        parts = []
        BS = 512
        for i in range(0, X_val.shape[0], BS):
            xb = X_val[i:i + BS]
            parts.append(model(xb)["x_hat"])
        X_hat = torch.cat(parts, dim=0)
        ss_res = ((X_val - X_hat) ** 2).sum().item()
        ss_tot = ((X_val - X_val.mean(0, keepdim=True)) ** 2).sum().item()
        val_mse = float(((X_val - X_hat) ** 2).mean().item())
        val_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    train_curve.append(ep_loss)
    val_curve.append(val_mse)
    val_r2_curve.append(val_r2)
    k_eff_curve.append(ep_k)
    dead_curve.append(ep_dead)
    print(f"[ep {ep+1:02d}/{EPOCHS}] train_loss={ep_loss:.5f} recon={ep_recon:.5f} "
          f"k_eff={ep_k:.1f} dead={ep_dead:.3f} val_mse={val_mse:.5f} val_R²={val_r2:.4f} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)

# -------------------- save --------------------
torch.save(model.state_dict(), OUT / "state.pt")

metrics = {
    "config": {
        "input_dim": D, "F": F, "fourier_harm": H, "lightness_basis_k": K_ELL,
        "top_k": TOP_K, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "params": n_params,
    },
    "train_loss_curve": train_curve,
    "val_mse_curve": val_curve,
    "val_r2_curve": val_r2_curve,
    "k_eff_curve": k_eff_curve,
    "dead_rate_curve": dead_curve,
    "final_val_r2": val_r2_curve[-1],
    "final_k_eff": k_eff_curve[-1],
    "final_dead_rate": dead_curve[-1],
    "elapsed_sec": time.time() - t0,
}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[done] saved to {OUT}", flush=True)
print(f"[done] final val R² = {val_r2_curve[-1]:.4f}", flush=True)
print(f"[done] final k_eff  = {k_eff_curve[-1]:.1f}", flush=True)
print(f"[done] final dead   = {dead_curve[-1]:.4f}", flush=True)
