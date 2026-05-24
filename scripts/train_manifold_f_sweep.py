"""Sweep ManifoldSAE over F ∈ {128, 256, 512, 1024, 2048, 4096} on V100.

K=32 fixed (matches existing sae_comparison), 10 epochs each. Reports
val R², dead-atom rate, n_active. Saves Pareto plot to
runs/manifold_f_sweep/{summary.json, pareto.png}.

Uses the ManifoldSAE class from scripts/train_sae_comparison.py via
``runpy``-style import of the model definition. Because that module
runs training at import time, we re-implement the class locally here
(verbatim copy from the comparison script — DO NOT modify shared file).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- ManifoldSAE: verbatim copy of class from train_sae_comparison.py ----
class ManifoldSAE(nn.Module):
    def __init__(self, d_in, n_feat, M_F=4):
        super().__init__()
        self.n_feat = n_feat; self.M_F = M_F
        self.W_gate = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_gate = nn.Parameter(torch.full((n_feat,), -2.0))
        self.W_theta = nn.Parameter(torch.randn(d_in, n_feat * 2) * (1.0/np.sqrt(d_in)))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        basis_dim = 2 * M_F + 1
        self.D_k = nn.Parameter(torch.randn(n_feat, basis_dim, d_in) * (0.1/np.sqrt(basis_dim)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.log_ard = nn.Parameter(torch.zeros(n_feat))

    def theta(self, x):
        xc = x - self.b_d
        tp = xc @ self.W_theta
        B = x.shape[0]
        tp = tp.view(B, self.n_feat, 2)
        tp = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tp

    def fourier_basis(self, cs):
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c)]
        ck, sk = c.clone(), s.clone()
        feats += [ck, sk]
        for m in range(2, self.M_F + 1):
            ck_new = ck * c - sk * s
            sk_new = sk * c + ck * s
            ck, sk = ck_new, sk_new
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)

    def forward(self, x, tau=1.0, hard=False):
        xc = x - self.b_d
        gate_logit = xc @ self.W_gate + self.b_gate
        if self.training:
            u = torch.rand_like(gate_logit).clamp(1e-6, 1-1e-6)
            g_noise = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + g_noise) / tau)
        else:
            gate = torch.sigmoid(gate_logit)
        if hard:
            gate = (gate > 0.5).float() + (gate - gate.detach())
        amp_raw = xc @ self.W_amp
        amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
        cs = self.theta(x)
        phi = self.fourier_basis(cs)
        from manifold_sae.scale import curve_decode_auto
        weight = gate * amp
        recon = curve_decode_auto(weight, atoms=phi, basis_coeffs=self.D_k) + self.b_d
        return recon, gate, amp

    def encode_for_eval(self, x):
        with torch.no_grad():
            xc = x - self.b_d
            gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
            amp_raw = xc @ self.W_amp
            amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
            return gate * amp


def run(F_atoms: int, X_train_np, X_val, val_var_t, mu, device, epochs=10, bs=512, lr=3e-4, sparsity_w=1e-2):
    D = X_train_np.shape[1]
    torch.manual_seed(0)
    model = ManifoldSAE(D, F_atoms, M_F=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [F={F_atoms}] params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps = epochs * (len(X_train_np) // bs)
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(X_train_np))
        ep_loss = 0.0; ep_n = 0
        for s in range(0, len(order), bs):
            idx = order[s:s+bs]
            xb = torch.from_numpy(X_train_np[idx]).to(device)
            opt.zero_grad()
            tau = max(0.3, 1.0 - 0.7 * (step / max(1, n_steps)))
            recon, gate, amp = model(xb, tau=tau)
            recon_loss = F.mse_loss(recon, xb)
            sp_loss = gate.mean()
            ard = torch.log(1e-2 + (model.D_k ** 2).sum(dim=(1,2))).mean()
            loss = recon_loss + sparsity_w * sp_loss + 1e-4 * ard
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * xb.shape[0]; ep_n += xb.shape[0]
            step += 1
        # val
        model.eval()
        with torch.no_grad():
            v_mse = 0.0; v_n = 0
            for i in range(0, X_val.shape[0], 1024):
                xb = X_val[i:i+1024]
                recon, _, _ = model(xb, tau=0.3, hard=False)
                v_mse += F.mse_loss(recon, xb, reduction='sum').item()
                v_n += xb.numel()
            v_mse /= v_n
            val_r2 = 1.0 - v_mse / val_var_t
        print(f"    ep {ep+1:02d}/{epochs}  val_R²={val_r2:.4f}  t={time.time()-t0:.1f}s", flush=True)

    # final activeness summary (subsample for speed)
    model.eval()
    with torch.no_grad():
        sample = X_train_np[np.random.permutation(len(X_train_np))[:8192]]
        sample_t = torch.from_numpy(sample).to(device)
        acts = []
        for i in range(0, sample.shape[0], 1024):
            acts.append(model.encode_for_eval(sample_t[i:i+1024]).cpu().numpy())
        acts = np.concatenate(acts, 0)
    active_per_atom = (acts > 1e-3).mean(0)
    dead = float((active_per_atom < 1e-5).mean())
    n_active = int((active_per_atom >= 1e-5).sum())
    mean_act = float(active_per_atom.mean())
    return {"F": F_atoms, "val_r2": float(val_r2), "dead_rate": dead,
            "n_active": n_active, "mean_activeness": mean_act,
            "n_params": n_params, "elapsed_s": time.time() - t0}


def main():
    ROOT = Path(__file__).resolve().parents[1]
    OUT = ROOT / "runs" / "manifold_f_sweep"
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    X_path = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
    X = np.load(X_path, mmap_mode="r")
    N, D = X.shape
    print(f"[data] shape={X.shape}", flush=True)

    N_COLORS = 949
    N_TPL = 28
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
    X_val = torch.from_numpy(X_val_np).to(device)
    val_var_t = X_val.var().item()
    print(f"[data] train={len(train_idx)} val={len(val_idx)}", flush=True)

    Fs = [128, 256, 512, 1024, 2048, 4096]
    results = []
    for Fa in Fs:
        print(f"\n=== F={Fa} ===", flush=True)
        try:
            r = run(Fa, X_train_np, X_val, val_var_t, mu, device, epochs=10, bs=512)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[F={Fa}] CUDA OOM: {e}; skipping", flush=True)
            torch.cuda.empty_cache()
            continue
        results.append(r)
        print(f"[F={Fa}] R²={r['val_r2']:.4f} dead={r['dead_rate']:.3f} "
              f"n_active={r['n_active']}/{Fa}", flush=True)
        torch.cuda.empty_cache()

    summary = {"sweep_results": results}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pareto plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Fs_done = [r["F"] for r in results]
        r2s = [r["val_r2"] for r in results]
        deads = [r["dead_rate"] for r in results]
        n_acts = [r["n_active"] for r in results]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(Fs_done, r2s, "o-", lw=2); axes[0].set_xscale("log")
        axes[0].set_xlabel("F (n_features)"); axes[0].set_ylabel("val R²")
        axes[0].set_title("Pareto: val R² vs F"); axes[0].grid(alpha=0.3)
        axes[1].plot(Fs_done, deads, "o-", lw=2, color="C3"); axes[1].set_xscale("log")
        axes[1].set_xlabel("F"); axes[1].set_ylabel("dead-atom rate")
        axes[1].set_title("Dead rate vs F"); axes[1].grid(alpha=0.3)
        axes[2].plot(Fs_done, n_acts, "o-", lw=2, color="C2"); axes[2].set_xscale("log")
        axes[2].set_xlabel("F"); axes[2].set_ylabel("n_active atoms")
        axes[2].plot(Fs_done, Fs_done, "k--", alpha=0.4, label="y=F")
        axes[2].set_title("n_active vs F"); axes[2].legend(); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "pareto.png", dpi=130)
        print(f"[done] wrote pareto.png", flush=True)
    except Exception as e:
        print(f"[plot] failed: {e}", flush=True)

    print(f"\n=== F-SWEEP SUMMARY ===", flush=True)
    print(f"{'F':>6} {'val_R²':>8} {'dead':>7} {'n_active':>9} {'params':>14} {'sec':>6}",
          flush=True)
    for r in results:
        print(f"{r['F']:>6} {r['val_r2']:>8.4f} {r['dead_rate']:>7.3f} "
              f"{r['n_active']:>9} {r['n_params']:>14,} {r['elapsed_s']:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
