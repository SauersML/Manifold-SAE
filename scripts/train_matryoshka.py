"""Train pure-Python Matryoshka SAE on cogito-L40, F=512 outer, shells [64,128,256,512].

Reports per-shell val R², per-shell dead-rate. Also probes whether shell-64 atoms
look interpretably coarser than shell-512: we measure mean activation entropy
across xkcd colors per atom (low entropy → atom fires for narrow color set →
specific; high entropy → broad/coarse concept).
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_sae.matryoshka import (
    MatryoshkaSAE, MatryoshkaSAEConfig, matryoshka_loss, shell_r2_and_dead
)

OUT = ROOT / "runs" / "MATRYOSHKA_F512_SHELLS_64_128_256_512"
OUT.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[setup] device={DEVICE}", flush=True)

DATA = Path(os.environ.get("X_L40", str(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy")))
X = np.load(DATA, mmap_mode="r")
N, D = X.shape
print(f"[data] X shape={X.shape}", flush=True)

N_COLORS, N_TPL = 949, 28
rng = np.random.default_rng(0)
color_perm = rng.permutation(N_COLORS)
n_val_colors = int(0.2 * N_COLORS)
val_colors = set(color_perm[:n_val_colors].tolist())
row_color = np.arange(N) // N_TPL
train_idx = np.where(~np.isin(row_color, list(val_colors)))[0]
val_idx = np.where(np.isin(row_color, list(val_colors)))[0]

X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)
mu = X_train_np.mean(0)
X_train_np -= mu
X_val_np -= mu
val_var_t = float((X_val_np ** 2).mean())

X_train = torch.from_numpy(X_train_np).to(DEVICE)
X_val = torch.from_numpy(X_val_np).to(DEVICE)


def main():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    F_outer = int(os.environ.get("F", "512"))
    shells = (64, 128, 256, 512)
    epochs = int(os.environ.get("EPOCHS", "10"))
    bs = int(os.environ.get("BS", "256"))
    lr = float(os.environ.get("LR", "3e-4"))

    cfg = MatryoshkaSAEConfig(
        input_dim=D, n_features=F_outer, shells=shells, l1_weight=1e-3,
    )
    torch.manual_seed(0)
    model = MatryoshkaSAE(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params = {n_params/1e6:.1f} M  F={F_outer} shells={shells} ep={epochs}", flush=True)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(X_train.shape[0], device=DEVICE)
        running = 0.0; n_steps = 0
        for s in range(0, X_train.shape[0], bs):
            xb = X_train[order[s:s+bs]]
            opt.zero_grad()
            out = model(xb)
            loss, log = matryoshka_loss(out, xb, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()); n_steps += 1
        # Val metrics each epoch
        res = shell_r2_and_dead(model, X_val, val_var_t, bs=bs)
        line = " ".join(f"{k}={v:.3f}" for k, v in res.items() if k.startswith("r2_"))
        dead = " ".join(f"{k}={v:.2f}" for k, v in res.items() if k.startswith("dead_"))
        print(f"  ep {ep+1:2d}  loss={running/n_steps:.4f}  {line}  dead: {dead}  "
              f"alive={res['alive_total']}/{F_outer}  t={time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated()/1e6 if DEVICE.type=="cuda" else float("nan")

    # Final eval + interpretability probe
    final = shell_r2_and_dead(model, X_val, val_var_t, bs=bs)

    # Coarseness probe: for each atom, look at activation pattern over the 949
    # xkcd colors (averaging 28 templates per color). Compute per-atom entropy
    # H = -Σ p_c log p_c where p_c = mean_activation_on_color_c / sum_c. High H
    # = atom fires broadly across colors (coarse / superordinate); Low H = atom
    # fires for a narrow color set (specific).
    model.eval()
    with torch.no_grad():
        # Use full set for the probe
        X_all = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32) - mu).to(DEVICE)
        z_all = []
        for i in range(0, X_all.shape[0], 512):
            z_all.append(model.encode(X_all[i:i+512]).cpu())
        z_all = torch.cat(z_all, dim=0)  # (N, F)
        # average per color (group of 28 rows)
        z_per_color = z_all.view(N_COLORS, N_TPL, F_outer).mean(dim=1)  # (949, F)
        p = z_per_color / z_per_color.sum(dim=0, keepdim=True).clamp(min=1e-8)
        H = -(p * (p.clamp(min=1e-12)).log()).sum(dim=0)  # (F,)
    H_np = H.numpy()
    # Compare shells: smaller-shell atoms (first 64) should have HIGHER mean H
    # if Matryoshka pressure works.
    res_h = {}
    for s in shells:
        sub = H_np[:s]
        sub = sub[~np.isnan(sub)]
        res_h[f"H_mean_first{s}"] = float(np.mean(sub)) if len(sub) else float("nan")
        res_h[f"H_med_first{s}"] = float(np.median(sub)) if len(sub) else float("nan")
    # Atoms exclusive to outer shells (e.g. 256-512) → expected to be specific
    excl = H_np[256:512]
    excl = excl[~np.isnan(excl)]
    res_h["H_mean_excl_256_512"] = float(np.mean(excl)) if len(excl) else float("nan")
    res_h["H_mean_excl_128_256"] = float(np.mean(H_np[128:256][~np.isnan(H_np[128:256])])) if len(H_np[128:256]) else float("nan")
    res_h["H_mean_excl_64_128"]  = float(np.mean(H_np[64:128][~np.isnan(H_np[64:128])])) if len(H_np[64:128]) else float("nan")

    out_blob = {
        "F": F_outer, "shells": list(shells), "epochs": epochs, "bs": bs, "lr": lr,
        "wall_s": wall, "peak_mb": peak_mb,
        "val": final,
        "coarseness_entropy": res_h,
        "n_params_M": n_params / 1e6,
    }
    print(f"\n[done] {json.dumps(out_blob, indent=2)}", flush=True)
    (OUT / "result.json").write_text(json.dumps(out_blob, indent=2))
    torch.save(model.state_dict(), OUT / "model.pt")
    np.save(OUT / "H_per_atom.npy", H_np)


if __name__ == "__main__":
    main()
