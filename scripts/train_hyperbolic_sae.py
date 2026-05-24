"""Train HyperbolicSAE on cogito-L40 cache, compare to L1/Manifold baselines
at MATCHED total-parameter count.

Usage:
    uv run python scripts/train_hyperbolic_sae.py

Saves: runs/HYPERBOLIC_SAE_COGITO/{state.pt, metrics.json, atoms.npy}
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from manifold_sae.hyperbolic_sae import HyperbolicSAE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "HYPERBOLIC_SAE_COGITO"
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


# -------------------- models --------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# Hyperbolic config
F_HYP = 128
D_BALL = 32
CURV = 1.0
hyp = HyperbolicSAE(input_dim=D, n_features=F_HYP, ball_dim=D_BALL,
                    curvature=CURV, sparsity_weight=1e-3).to(DEVICE)
P_HYP = count_params(hyp)
print(f"[hyp] params={P_HYP:,}  (F={F_HYP}, d={D_BALL})", flush=True)


class L1SAE(nn.Module):
    """Vanilla L1 SAE, sized to match Hyperbolic parameter budget."""
    def __init__(self, D, F, l1=1e-3):
        super().__init__()
        self.enc = nn.Linear(D, F)
        self.dec = nn.Linear(F, D)
        self.l1 = l1
        nn.init.kaiming_uniform_(self.enc.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.dec.weight, a=5 ** 0.5)
        nn.init.zeros_(self.enc.bias)
        nn.init.zeros_(self.dec.bias)

    def loss(self, x):
        h = torch.relu(self.enc(x))
        x_hat = self.dec(h)
        recon = ((x_hat - x) ** 2).mean()
        sparsity = self.l1 * h.abs().mean()
        return recon + sparsity, {"recon": recon.detach(),
                                  "l1": sparsity.detach(),
                                  "active_frac": (h > 1e-6).float().mean().detach()}

    def encode(self, x):
        return torch.relu(self.enc(x))

    def forward(self, x):
        h = self.encode(x)
        return self.dec(h)


# Match params by tuning F_L1: L1 has 2*D*F + F + D params ≈ 2DF
F_L1 = max(1, P_HYP // (2 * D))
l1 = L1SAE(D, F_L1, l1=1e-3).to(DEVICE)
P_L1 = count_params(l1)
print(f"[l1 ] params={P_L1:,} (F={F_L1})", flush=True)


# Manifold baseline: use a simple "atoms in R^d + readout" without hyperbolic
# (Euclidean analog of HyperbolicSAE).
class EuclideanSAE(nn.Module):
    def __init__(self, D, F, d, l1=1e-3):
        super().__init__()
        self.D, self.F, self.d = D, F, d
        self.W_enc = nn.Linear(D, F * d)
        self.W_gate = nn.Linear(D, F)
        self.atoms = nn.Parameter(torch.randn(F, d) * 0.05)
        self.W_dec_out = nn.Linear(d, D)
        self.l1 = l1
        nn.init.kaiming_uniform_(self.W_enc.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_dec_out.weight, a=5 ** 0.5)

    def loss(self, x):
        B = x.shape[0]
        t = self.W_enc(x).view(B, self.F, self.d)
        gates = torch.relu(self.W_gate(x))
        z = (gates.unsqueeze(-1) * t).sum(dim=1)
        x_hat = self.W_dec_out(z)
        recon = ((x_hat - x) ** 2).mean()
        sparsity = self.l1 * gates.abs().mean()
        return recon + sparsity, {"recon": recon.detach(),
                                  "l1": sparsity.detach(),
                                  "active_frac": (gates > 1e-6).float().mean().detach()}


eu = EuclideanSAE(D, F_HYP, D_BALL, l1=1e-3).to(DEVICE)
P_EU = count_params(eu)
print(f"[eu ] params={P_EU:,} (F={F_HYP}, d={D_BALL})", flush=True)


# -------------------- training --------------------
EPOCHS = 10
BATCH = 256
LR = 3e-4


def train_model(name, model, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = X_train
    N_train = Xt.shape[0]
    t0 = time.time()
    last_loss = None
    for ep in range(epochs):
        perm = torch.randperm(N_train)
        ep_loss = 0.0
        nb = 0
        for i in range(0, N_train, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xt[idx].to(DEVICE)
            loss, _ = model.loss(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        last_loss = ep_loss / max(1, nb)
        print(f"[{name}] epoch {ep+1}/{epochs} loss={last_loss:.5f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
    return last_loss


@torch.no_grad()
def eval_r2(model, name) -> float:
    model.eval()
    # batched val pass to spare MPS memory
    parts = []
    BS = 512
    for i in range(0, X_val.shape[0], BS):
        xb = X_val[i:i + BS]
        out = model(xb) if hasattr(model, "forward") and not hasattr(model, "encode") else None
        if name == "hyp":
            out_dict = model(xb)
            xh = out_dict["x_hat"]
        elif name == "l1":
            xh = model(xb)
        else:  # eu
            B = xb.shape[0]
            t = model.W_enc(xb).view(B, model.F, model.d)
            g = torch.relu(model.W_gate(xb))
            z = (g.unsqueeze(-1) * t).sum(dim=1)
            xh = model.W_dec_out(z)
        parts.append(xh)
    X_hat = torch.cat(parts, dim=0)
    ss_res = ((X_val - X_hat) ** 2).sum().item()
    ss_tot = ((X_val - X_val.mean(0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2


metrics = {"params": {"hyp": P_HYP, "l1": P_L1, "eu": P_EU}}

for name, m in [("hyp", hyp), ("l1", l1), ("eu", eu)]:
    print(f"\n=== training {name} ===", flush=True)
    final = train_model(name, m)
    r2 = eval_r2(m, name)
    metrics[name] = {"final_train_loss": final, "val_r2": r2}
    print(f"[{name}] val R² = {r2:.4f}", flush=True)


# -------------------- save --------------------
with torch.no_grad():
    atoms = hyp.atom_positions().cpu().numpy()
    norms = hyp.feature_norms_in_ball().cpu().numpy()

np.save(OUT / "atoms.npy", atoms)
np.save(OUT / "atom_radii.npy", norms)
torch.save(hyp.state_dict(), OUT / "hyperbolic_state.pt")

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n[done] saved to", OUT, flush=True)
print(json.dumps(metrics, indent=2))
