"""CUDA-adapted F=65K Manifold-SAE TopK training. See train_sae_f65k.py for design."""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_sae.kernels.sparse_decode import sparse_curve_decode  # noqa: E402

OUT = ROOT / "runs" / "MANIFOLD_SAE_F65K"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)


def _peak_mb() -> float:
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


DATA = Path(os.environ.get("X_L40", str(Path.home() / "data" / "X_L40.npy")))
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


class ManifoldSAE_TopK(nn.Module):
    def __init__(self, d_in: int, n_feat: int, K_active: int = 64, M_F: int = 3):
        super().__init__()
        self.n_feat = n_feat
        self.K_active = K_active
        self.M_F = M_F
        P = 2 * M_F + 1
        self.P = P
        self.W_gate = nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        self.b_gate = nn.Parameter(torch.zeros(n_feat))
        self.W_theta = nn.Parameter(torch.randn(d_in, n_feat * 2) * (1.0 / np.sqrt(d_in)))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        # Init D_k directly on device to avoid 13GB CPU allocation at F=65K.
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D_k = nn.Parameter(
            torch.randn(n_feat, P, d_in, device=_device) * (0.1 / np.sqrt(P))
        )
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def fourier_basis(self, cs):
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c), c.clone(), s.clone()]
        ck, sk = c.clone(), s.clone()
        for _ in range(2, self.M_F + 1):
            ck, sk = ck * c - sk * s, sk * c + ck * s
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)

    def forward(self, x):
        xc = x - self.b_d
        scores = xc @ self.W_gate + self.b_gate
        topv, topi = scores.topk(self.K_active, dim=-1)
        gate = torch.zeros_like(scores)
        gate.scatter_(1, topi, F.relu(topv))

        amp = F.softplus(xc @ self.W_amp)
        tp = (xc @ self.W_theta).view(x.shape[0], self.n_feat, 2)
        cs = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(topi)
        cs_active = cs[b_idx, topi]
        phi_active = self.fourier_basis(cs_active)

        phi = torch.zeros(x.shape[0], self.n_feat, self.P, device=x.device, dtype=x.dtype)
        phi[b_idx.unsqueeze(-1).expand_as(phi_active),
            topi.unsqueeze(-1).expand_as(phi_active),
            torch.arange(self.P, device=x.device).view(1, 1, -1).expand_as(phi_active)] = phi_active

        weight = gate * amp
        recon = sparse_curve_decode(weight, phi, self.D_k, threshold=0.0) + self.b_d
        return recon, gate


def get_batches(X_np, bs):
    n = X_np.shape[0]
    order = np.random.permutation(n)
    for s in range(0, n, bs):
        yield torch.from_numpy(X_np[order[s:s+bs]]).to(DEVICE)


def train_one_epoch(F_atoms: int, K_active: int = 64, bs: int = 128, lr: float = 3e-4):
    print(f"\n[train] F={F_atoms} K_active={K_active} bs={bs}", flush=True)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = ManifoldSAE_TopK(D, F_atoms, K_active=K_active, M_F=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params = {n_params/1e6:.1f} M", flush=True)
    # Use SGD (no per-param state) to fit the 6.5 GB / 13 GB decoder + grads
    # in 16 GB V100. Adam's 2x momentum/variance overhead OOMs at F>=32K.
    # Plain SGD (no momentum state) — only optimizer choice that fits
    # 6.5 GB params + 6.5 GB grads + activations in 16 GB V100 at F=32K.
    # Adam needs 2x params for state → OOMs. SGD+momentum needs 1x → still OOMs.
    opt = torch.optim.SGD(model.parameters(), lr=lr * 30.0, momentum=0.0)

    t0 = time.time()
    model.train()
    for step, xb in enumerate(get_batches(X_train_np, bs)):
        opt.zero_grad()
        recon, gate = model(xb)
        loss = F.mse_loss(recon, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"  step {step:5d}  loss={loss.item():.4f}  peak={_peak_mb():.0f} MB  "
                  f"t={time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    peak = _peak_mb()

    model.eval()
    v_mse = 0.0; v_n = 0
    with torch.no_grad():
        for i in range(0, X_val_np.shape[0], 128):
            xb = torch.from_numpy(X_val_np[i:i+128]).to(DEVICE)
            recon, _ = model(xb)
            v_mse += F.mse_loss(recon, xb, reduction="sum").item()
            v_n += xb.numel()
    v_mse /= v_n
    val_r2 = 1.0 - v_mse / val_var_t
    return {
        "F": F_atoms, "K_active": K_active, "bs": bs,
        "wall_s": wall, "peak_mb": peak, "val_r2": val_r2,
        "n_params_M": n_params / 1e6,
    }


def main():
    target_F = int(os.environ.get("F", "65536"))
    fallback_F = int(os.environ.get("F_FALLBACK", "32768"))
    last_fallback_F = int(os.environ.get("F_FALLBACK2", "16384"))
    bs = int(os.environ.get("BS", "128"))
    fellbacks: list[int] = []
    last_err = ""
    for F_try in [target_F, fallback_F, last_fallback_F]:
        try:
            res = train_one_epoch(F_try, K_active=64, bs=bs)
            if fellbacks:
                res["fell_back_from"] = fellbacks
            break
        except (RuntimeError, MemoryError) as e:
            last_err = str(e)[:200]
            print(f"[train] F={F_try} failed: {last_err}", flush=True)
            fellbacks.append(F_try)
            if DEVICE.type == "cuda":
                import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            continue
    else:
        res = {"all_OOM": True, "tried": fellbacks, "last_err": last_err}

    print(f"\n[done] {json.dumps(res, indent=2)}", flush=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
