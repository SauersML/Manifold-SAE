"""Train Manifold-SAE at F=65536 on cogito-L40 cache (1 epoch, MPS).

Uses the sparse curve-decode kernel (manifold_sae.kernels.sparse_decode) via
``manifold_sae.scale.curve_decode_auto`` — at F=2^16 the dense path's
intermediate (F, P, D) decoder tensor alone is 65536 · 7 · 7168 · 4 B ≈ 13.1 GB,
which won't fit in MPS unified memory alongside activations and gradients.

To make F=65K feasible on a single laptop we use a HARD-TOP-K gate (K_active=64)
rather than the Gumbel-sigmoid gate used in the F=512 comparison script.
Reason: under sigmoid-Gumbel almost all atoms have gate ≈ 0.5 at eval time
which collapses the sparse kernel's memory savings (see verify_sparse_dispatch.py).

Reports wall time, peak alloc, val R². If F=65K OOMs (most likely on the
F·P·D = 13 GB decoder parameter itself), the script automatically drops to
F=32768 and reports what fit.

This is a single-epoch reference run; productionizing F=2^16 would need a
checkpoint-blocked decoder (load atom rows lazily) — out of scope here.
"""
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)


def _mps_peak_mb() -> float:
    if DEVICE.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except Exception:
            return float("nan")
    return float("nan")


X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
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
    """Hard-TopK Manifold-SAE with Fourier curve atoms + sparse decode.

    Encoder: linear gate-score → TopK selection of K_active atoms per row.
    Per-atom: theta_k (S^1) + amplitude a_k.
    Decoder: per-atom Fourier basis @ per-atom (P, D) coefficients.
    Reconstruction: via sparse_curve_decode (M = B·K_active).
    """
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
        # decoder: (F, P, D). At F=65K, D=7168, this is 13.1 GB at float32.
        # We keep it on-device; if that's the OOM site, fall back to a
        # smaller F (handled in main).
        self.D_k = nn.Parameter(torch.randn(n_feat, P, d_in) * (0.1 / np.sqrt(P)))
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def fourier_basis(self, cs: torch.Tensor) -> torch.Tensor:
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c), c.clone(), s.clone()]
        ck, sk = c.clone(), s.clone()
        for _ in range(2, self.M_F + 1):
            ck, sk = ck * c - sk * s, sk * c + ck * s
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xc = x - self.b_d
        scores = xc @ self.W_gate + self.b_gate                       # (B, F)
        topv, topi = scores.topk(self.K_active, dim=-1)
        gate = torch.zeros_like(scores)
        gate.scatter_(1, topi, F.relu(topv))                          # (B, F) K-sparse

        amp = F.softplus(xc @ self.W_amp)                             # (B, F)
        tp = (xc @ self.W_theta).view(x.shape[0], self.n_feat, 2)
        cs = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Build the Fourier basis ONLY for the K_active selected atoms.
        # cs is (B, F, 2); we gather the active subset to avoid an (B, F, P) tensor.
        b_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(topi)
        cs_active = cs[b_idx, topi]                                   # (B, K, 2)
        phi_active = self.fourier_basis(cs_active)                    # (B, K, P)

        # Re-scatter into (B, F, P) only on the active positions. Memory
        # still O(B·F·P) — for F=65K, B=256, P=7 this is ~470 MB which is
        # fine. The alternative (sparse-only atoms) requires plumbing the
        # active indices into the kernel — out of scope for v1.
        phi = torch.zeros(x.shape[0], self.n_feat, self.P, device=x.device, dtype=x.dtype)
        phi[b_idx.unsqueeze(-1).expand_as(phi_active),
            topi.unsqueeze(-1).expand_as(phi_active),
            torch.arange(self.P, device=x.device).view(1, 1, -1).expand_as(phi_active)] = phi_active

        weight = gate * amp                                            # (B, F) still K-sparse
        recon = sparse_curve_decode(weight, phi, self.D_k, threshold=0.0) + self.b_d
        return recon, gate


def get_batches(X_np, bs):
    n = X_np.shape[0]
    order = np.random.permutation(n)
    for s in range(0, n, bs):
        yield torch.from_numpy(X_np[order[s:s+bs]]).to(DEVICE)


def train_one_epoch(F_atoms: int, K_active: int = 64, bs: int = 128, lr: float = 3e-4):
    print(f"\n[train] F={F_atoms} K_active={K_active} bs={bs}", flush=True)
    torch.manual_seed(0)
    model = ManifoldSAE_TopK(D, F_atoms, K_active=K_active, M_F=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params = {n_params/1e6:.1f} M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    peak = 0.0
    model.train()
    for step, xb in enumerate(get_batches(X_train_np, bs)):
        opt.zero_grad()
        recon, gate = model(xb)
        loss = F.mse_loss(recon, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        peak = max(peak, _mps_peak_mb())
        if step % 20 == 0:
            print(f"  step {step:5d}  loss={loss.item():.4f}  peak={peak:.0f} MB  "
                  f"t={time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0

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
    bs = int(os.environ.get("BS", "128"))
    try:
        res = train_one_epoch(target_F, K_active=64, bs=bs)
    except (RuntimeError, MemoryError) as e:
        print(f"[train] F={target_F} failed: {str(e)[:200]}", flush=True)
        print(f"[train] retrying at F={fallback_F}", flush=True)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        res = train_one_epoch(fallback_F, K_active=64, bs=bs)
        res["fell_back_from"] = target_F

    print(f"\n[done] {json.dumps(res, indent=2)}", flush=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
