"""Floor experiment for the amortized Manifold-SAE: compact capture vs dilution.

The make-or-break test from the neural-geometry papers, run on data with known
ground truth. We plant K circles, each embedded in a random 2-plane of R^D, and
mix them additively and sparsely (each token activates a random subset). Then:

  * train the amortized Manifold-SAE (circle atoms), and
  * train a standard TopK SAE baseline,

and measure, per planted circle:
  1. SUPPORT SIZE — how many dictionary atoms fire on that circle's tokens.
     Compact capture ⇒ small (ideally ~1 for the manifold-SAE); dilution ⇒ large.
  2. COORDINATE ALIGNMENT (manifold-SAE only) — circular correlation
     (Jammalamadaka–Sarma, rotation/reflection-invariant) between the best
     atom's learned angle and the true angle. High ⇒ the atom's coordinate is
     the concept's coordinate. TopK has no per-atom coordinate, so this is the
     manifold-native win, reported alongside TopK's support for the contrast.

NOTE: the real cogito-L40 activation cache (runs/COLOR_COGITO_L40/X_L40.npy) is a
cluster harvest and is NOT present locally, so this runs on the controlled
synthetic floor. The real-data run is the same script pointed at that cache.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from manifold_sae.amortized_manifold_sae import (  # noqa: E402
    AmortizedManifoldSAE,
    AmortizedManifoldSAEConfig,
)

# gam's torch penalties are CPU/float64-only and currently crash on MPS (gam#362),
# so train on CPU here. On Linux+CUDA float64 works and CUDA can be used.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data: additive sparse mixture of planted circles in R^D.
# ---------------------------------------------------------------------------
def plant_circles(D: int, K: int, N: int, sparsity: float, noise: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    planes = torch.zeros(K, 2, D)
    for k in range(K):  # random orthonormal 2-plane per circle
        M = torch.randn(2, D, generator=g)
        Q, _ = torch.linalg.qr(M.T)
        planes[k] = Q.T[:2]
    angles = torch.rand(N, K, generator=g) * 2 * math.pi
    active = (torch.rand(N, K, generator=g) < sparsity)
    active[~active.any(dim=1), 0] = True  # ensure >=1 active per token
    emb = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (N,K,2)
    contrib = torch.einsum("nkc,kcd->nkd", emb, planes)                # (N,K,D)
    x = (contrib * active.unsqueeze(-1)).sum(dim=1)
    x = x + noise * torch.randn(N, D, generator=g)
    return x.float(), active, angles


def circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka–Sarma circular correlation, rotation/reflection-invariant."""
    a0 = a - math.atan2(np.sin(a).mean(), np.cos(a).mean())
    b0 = b - math.atan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a0) * np.sin(b0))
    den = math.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2))
    return float(abs(num / den)) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Baseline: standard TopK SAE.
# ---------------------------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, D: int, F: int, k: int):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(D, F) / math.sqrt(D))
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.W_dec = nn.Parameter(torch.randn(F, D) / math.sqrt(F))
        self.b_dec = nn.Parameter(torch.zeros(D))

    def encode(self, x):
        z = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        vals, idx = z.topk(self.k, dim=-1)
        out = torch.zeros_like(z).scatter_(-1, idx, vals)
        return out

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_dec + self.b_dec, z


def support_size(z_active: torch.Tensor, active_mask: np.ndarray, frac: float = 0.25):
    """For each planted circle, count atoms firing on > frac of its active tokens."""
    fired = (z_active.abs() > 1e-6).float().cpu().numpy()  # (N, F)
    sizes = []
    for k in range(active_mask.shape[1]):
        rows = active_mask[:, k]
        if rows.sum() < 10:
            sizes.append(0); continue
        rate = fired[rows].mean(axis=0)            # per-atom firing rate on circle k
        sizes.append(int((rate > frac).sum()))
    return sizes


def main():
    torch.manual_seed(0)
    D, K, N = 128, 4, 6000
    F = 16
    X, active, angles = plant_circles(D, K, N, sparsity=0.4, noise=0.03, seed=0)
    ntr = 5000
    Xtr, Xva = X[:ntr].to(DEVICE), X[ntr:].to(DEVICE)
    act_va, ang_va = active[ntr:].numpy(), angles[ntr:].numpy()
    var = float(Xva.var())
    OUT = ROOT / "runs" / "AMORTIZED_FLOOR"; OUT.mkdir(parents=True, exist_ok=True)

    def batches(Z, bs=512):
        for s in range(0, Z.shape[0], bs):
            yield Z[s:s + bs]

    # ---- Amortized Manifold-SAE ----
    # Pure planted circles ⇒ 1-harmonic atoms (cannot localize to an arc, so they
    # can't tile/dilute); fewer atoms; structure penalties at load-bearing strength.
    cfg = AmortizedManifoldSAEConfig(input_dim=D, n_atoms=F, fourier_harmonics=1,
                                     sparsity_weight=1e-2, incoherence_weight=1e-2,
                                     isometry_weight=1e-2,
                                     gate_threshold=0.05, dtype=torch.float32)
    msae = AmortizedManifoldSAE(cfg).to(DEVICE)
    opt = torch.optim.Adam(msae.parameters(), lr=4e-3)
    for ep in range(40):
        for xb in batches(Xtr):
            opt.zero_grad(); msae.loss(xb)["loss"].backward(); opt.step()
    msae.eval()
    with torch.no_grad():
        out = msae(Xva)
    m_r2 = 1.0 - float((out.x_hat - Xva).pow(2).mean()) / var
    m_support = support_size(out.gate, act_va)
    theta = out.theta.cpu().numpy()        # (Nva, F)
    gate = out.gate.abs().cpu().numpy()
    m_align = []
    for k in range(K):
        rows = act_va[:, k]
        if rows.sum() < 10:
            m_align.append(0.0); continue
        # best atom = the one most active on circle k; align its angle to truth.
        best = int(gate[rows].mean(axis=0).argmax())
        m_align.append(circ_corr(ang_va[rows, k], theta[rows, best]))

    # ---- TopK SAE baseline ----
    topk = TopKSAE(D, F, k=8).to(DEVICE)
    opt = torch.optim.Adam(topk.parameters(), lr=4e-3)
    for ep in range(40):
        for xb in batches(Xtr):
            opt.zero_grad()
            xh, z = topk(xb)
            (((xh - xb) ** 2).mean()).backward(); opt.step()
    topk.eval()
    with torch.no_grad():
        xh, z = topk(Xva)
    t_r2 = 1.0 - float((xh - Xva).pow(2).mean()) / var
    t_support = support_size(z, act_va)

    summary = {
        "device": str(DEVICE), "D": D, "K_planted": K, "F": F,
        "manifold_sae": {"val_r2": m_r2, "support_per_circle": m_support,
                         "mean_support": float(np.mean(m_support)),
                         "coord_alignment_per_circle": m_align,
                         "mean_alignment": float(np.mean(m_align))},
        "topk_sae": {"val_r2": t_r2, "support_per_circle": t_support,
                     "mean_support": float(np.mean(t_support))},
        "verdict_compact_capture": bool(np.mean(m_support) < np.mean(t_support)),
        "verdict_coordinate_recovered": bool(np.mean(m_align) > 0.6),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    torch.save({"config": cfg, "state_dict": msae.state_dict()}, OUT / "amortized_msae.pt")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[floor] Manifold-SAE: R2={m_r2:.3f}  mean_support={np.mean(m_support):.2f} "
          f"atoms/circle  mean_align={np.mean(m_align):.3f}", flush=True)
    print(f"[floor] TopK SAE   : R2={t_r2:.3f}  mean_support={np.mean(t_support):.2f} "
          f"atoms/circle (dilution baseline)", flush=True)
    print(f"[verdict] compact capture (fewer atoms/circle than TopK): "
          f"{summary['verdict_compact_capture']}", flush=True)
    print(f"[verdict] coordinate recovered (circ-corr>0.6): "
          f"{summary['verdict_coordinate_recovered']}", flush=True)


if __name__ == "__main__":
    main()
