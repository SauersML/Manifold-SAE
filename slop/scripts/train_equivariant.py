"""Train EquivariantSAE on cogito-L40 X_L40.npy.

64 SO(2) atoms + 448 trivial atoms = F=512. Compare to comparison.json baselines
(TopK 0.874, L1 0.882, Manifold 0.913).

Also computes the diagnostic: SO(2) atoms' learned hue angle vs ground-truth
xkcd HSV-H. Target: monotone wrap-around fit with R² > 0.85 from SO(2) atoms
alone (i.e. hue is genuinely a 1-manifold in the encoder).
"""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.equivariant import (
    EquivariantSAE, EquivariantSAEConfig, gauge_companion_loss
)

OUT = ROOT / "runs" / "equivariant_sae"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)

# ----- data -----
X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
N, D = X.shape
N_COLORS, N_TPL = 949, 28
assert N == N_COLORS * N_TPL

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
X_train_np -= mu; X_val_np -= mu
X_val = torch.from_numpy(X_val_np).to(DEVICE)
val_var_t = X_val.var().item()

# ----- xkcd HSV labels -----
def load_xkcd():
    out = []
    with open(ROOT / "experiments" / "xkcd_colors.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            name, hex_ = line.split("\t")[:2]
            hex_ = hex_.lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out.append((name, r/255.0, g/255.0, b/255.0))
    return out

xkcd = load_xkcd()[:N_COLORS]
rgb = np.array([[x[1], x[2], x[3]] for x in xkcd], dtype=np.float32)

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1), rgb.min(1); df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-8
    rm = mask & (mx == r); gm = mask & (mx == g); bm = mask & (mx == b)
    h[rm] = ((g[rm]-b[rm])/df[rm]) % 6
    h[gm] = ((b[gm]-r[gm])/df[gm]) + 2
    h[bm] = ((r[bm]-g[bm])/df[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-8, df/np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], 1)

hsv = rgb_to_hsv(rgb)                                    # (949, 3)
row_hsv_np = hsv[row_color]                              # (N, 3)
row_hsv_train = row_hsv_np[train_idx]
row_hsv_val = row_hsv_np[val_idx]

# ----- model -----
torch.manual_seed(0)
cfg = EquivariantSAEConfig(d_in=D, n_so2=64, n_trivial=448, d_aux_sup=3,
                           sparsity_weight=1e-3, eq_weight=1e-2, ard_weight=1e-4)
model = EquivariantSAE(cfg).to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=3e-4)
EPOCHS = int(__import__("os").environ.get("EPOCHS", "8"))
BS = 512
n_steps = EPOCHS * (len(X_train_np) // BS)
print(f"[run] EPOCHS={EPOCHS} BS={BS} n_steps={n_steps} F=512 (SO2:64 + triv:448)", flush=True)

t0 = time.time(); step = 0
history = []
for ep in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(X_train_np))
    ep_loss = ep_rec = ep_eq = ep_sp = ep_gc = 0.0
    nb = 0
    for s in range(0, len(perm), BS):
        b_idx = perm[s:s+BS]
        xb = torch.from_numpy(X_train_np[b_idx]).to(DEVICE)
        hsv_b = torch.from_numpy(row_hsv_train[b_idx]).to(DEVICE)
        tau = max(0.3, 1.0 - 0.7 * step / max(1, n_steps))
        out = model(xb, tau=tau, training=True)
        rec = F.mse_loss(out["recon"], xb)
        eq = model.equivariant_penalty(out["theta"], out["z_so2"])
        sp = model.sparsity_penalty(out["gate2"], out["gate0"])
        ard = model.ard_penalty()
        gc = gauge_companion_loss(out["theta"], hsv_b, d_aux_sup=cfg.d_aux_sup, weight=0.5)
        loss = rec + cfg.eq_weight * eq + cfg.sparsity_weight * sp + cfg.ard_weight * ard + gc
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item(); ep_rec += rec.item(); ep_eq += eq.item()
        ep_sp += sp.item(); ep_gc += gc.item(); nb += 1; step += 1
    # eval
    model.eval()
    with torch.no_grad():
        v_mse, v_n = 0.0, 0
        for i in range(0, X_val.shape[0], 1024):
            xb = X_val[i:i+1024]
            out = model(xb, tau=0.3, training=False)
            v_mse += F.mse_loss(out["recon"], xb, reduction="sum").item()
            v_n += xb.numel()
        v_mse /= v_n; val_r2 = 1.0 - v_mse / val_var_t
    history.append({"epoch": ep, "loss": ep_loss/nb, "rec": ep_rec/nb, "eq": ep_eq/nb,
                    "sp": ep_sp/nb, "gc": ep_gc/nb, "val_r2": val_r2,
                    "t": time.time()-t0})
    print(f"[ep {ep:02d}] loss={ep_loss/nb:.4f} rec={ep_rec/nb:.4f} eq={ep_eq/nb:.4e}"
          f" sp={ep_sp/nb:.4f} gc={ep_gc/nb:.4f} val_R²={val_r2:.4f} t={time.time()-t0:.1f}s",
          flush=True)

# ----- diagnostic: SO(2)-only val R² + hue alignment -----
model.eval()
with torch.no_grad():
    # SO(2) atoms alone: zero out trivial recon.
    v_mse_so2, v_n = 0.0, 0
    for i in range(0, X_val.shape[0], 1024):
        xb = X_val[i:i+1024]
        xc = xb - model.b_dec
        theta = model.group_head(xc)
        gate2, amp2 = model.amp_head_so2(xc, training=False)
        z2 = gate2 * amp2 * torch.exp(model.log_ard_so2)
        cs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        w_cs = z2.unsqueeze(-1) * cs
        recon_so2 = torch.einsum("bar,adr->bd", w_cs, model.W_so2) + model.b_dec
        v_mse_so2 += F.mse_loss(recon_so2, xb, reduction="sum").item()
        v_n += xb.numel()
    val_r2_so2_only = 1.0 - (v_mse_so2/v_n) / val_var_t

# Per-color θ aggregated across templates (centroid in (cos, sin) plane).
all_idx = np.concatenate([train_idx, val_idx])
all_hsv = row_hsv_np
with torch.no_grad():
    thetas = []
    for i in range(0, N, 1024):
        xb = torch.from_numpy(np.ascontiguousarray(
            X[i:i+1024]).astype(np.float32) - mu).to(DEVICE)
        out = model(xb, training=False)
        thetas.append(out["theta"].cpu().numpy())
thetas = np.concatenate(thetas, axis=0)                  # (N, 64)
# centroid (cos, sin) per color per atom
cs_all = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (N, 64, 2)
cs_color = np.zeros((N_COLORS, 64, 2))
cnt = np.zeros(N_COLORS)
for r in range(N):
    c = row_color[r]; cs_color[c] += cs_all[r]; cnt[c] += 1
cs_color /= np.maximum(cnt[:, None, None], 1)
theta_color = np.arctan2(cs_color[..., 1], cs_color[..., 0])    # (949, 64)

# Hue-alignment score per atom: |circular correlation| with HSV hue.
h_true = hsv[:, 0] * 2 * np.pi                                  # in [0, 2π)
def circ_corr(th, h):
    th = th - np.arctan2(np.sin(th).mean(), np.cos(th).mean())
    h_ = h - np.arctan2(np.sin(h).mean(), np.cos(h).mean())
    num = (np.sin(th) * np.sin(h_)).sum()
    den = np.sqrt((np.sin(th) ** 2).sum() * (np.sin(h_) ** 2).sum())
    return num / max(den, 1e-12)

circ_corrs = np.array([circ_corr(theta_color[:, a], h_true) for a in range(64)])

# Best-atom: smooth monotone wrap-around fit (linear regression of unwrapped θ vs H).
def monotone_r2(th, h):
    # Try both orientations; pick best linear fit of (cos h, sin h) <-> (cos th, sin th)
    # Equivalent to a 2D->2D rigid map: fit rotation + sign-flip, then R² in (cos, sin).
    X_ = np.stack([np.cos(h), np.sin(h)], 1)
    Y_ = np.stack([np.cos(th), np.sin(th)], 1)
    # Procrustes: best rotation R (det = ±1) minimizing ‖X R - Y‖²
    U, S, Vt = np.linalg.svd(X_.T @ Y_)
    R = U @ Vt
    Y_pred = X_ @ R
    ss_res = ((Y_ - Y_pred) ** 2).sum()
    ss_tot = ((Y_ - Y_.mean(0)) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)

per_atom_hue_r2 = np.array([monotone_r2(theta_color[:, a], h_true) for a in range(64)])
best_atom = int(per_atom_hue_r2.argmax())
best_r2 = float(per_atom_hue_r2[best_atom])

# Multi-atom ensemble: linear regression of (cos h, sin h) on the FULL 64 atoms' (cos θ, sin θ).
Y = np.stack([np.cos(h_true), np.sin(h_true)], 1)
Phi = np.concatenate([np.cos(theta_color), np.sin(theta_color)], axis=1)  # (949, 128)
Phi_ = np.concatenate([Phi, np.ones((949, 1))], axis=1)
# 5-fold CV by color
order = np.random.default_rng(0).permutation(N_COLORS)
folds = np.array_split(order, 5)
r2_cv = []
for k in range(5):
    val_c = folds[k]; trn_c = np.concatenate([folds[i] for i in range(5) if i != k])
    coef, *_ = np.linalg.lstsq(Phi_[trn_c] + 1e-6 * np.random.randn(*Phi_[trn_c].shape), Y[trn_c], rcond=None)
    Y_pred = Phi_[val_c] @ coef
    ss_res = ((Y[val_c] - Y_pred) ** 2).sum()
    ss_tot = ((Y[val_c] - Y[val_c].mean(0)) ** 2).sum()
    r2_cv.append(1.0 - ss_res / max(ss_tot, 1e-12))
ensemble_cv_r2 = float(np.mean(r2_cv))

print(f"\n[diag] SO(2)-only val R² = {val_r2_so2_only:.4f}", flush=True)
print(f"[diag] best per-atom hue R² (atom {best_atom}) = {best_r2:.4f}", flush=True)
print(f"[diag] 64-atom ensemble hue R² (5-fold CV) = {ensemble_cv_r2:.4f}", flush=True)

# ----- plot diagnostic -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.scatter(h_true, theta_color[:, best_atom],
           c=rgb, s=18, alpha=0.85, edgecolor='none')
ax.set_xlabel("xkcd HSV hue (rad)"); ax.set_ylabel(f"SO(2) atom {best_atom} θ (rad)")
ax.set_title(f"best-atom hue fit  R²={best_r2:.3f}")
ax.set_xlim(-0.2, 2 * np.pi + 0.2); ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
ax = axes[1]
ax.hist(per_atom_hue_r2, bins=24, edgecolor='k')
ax.axvline(0.85, color='r', linestyle='--', label="target R² > 0.85")
ax.set_xlabel("per-atom hue Procrustes R²"); ax.set_title("hue alignment across 64 SO(2) atoms")
ax.legend()
ax = axes[2]
ax.bar(["TopK", "L1", "Manifold", "Equivariant"],
       [0.874, 0.882, 0.913, val_r2_so2_only + 0],
       color=["#888", "#888", "#888", "#1f77b4"])
ax.set_ylabel("val R²"); ax.set_title("F=512 SAE comparison")
ax.set_ylim(0.85, max(0.95, val_r2_so2_only + 0.02))
fig.suptitle(f"EquivariantSAE on cogito-L40 — val R²={history[-1]['val_r2']:.4f}  (SO(2)-only={val_r2_so2_only:.4f})")
fig.tight_layout()
fig.savefig(OUT / "equivariant_diagnostic.png", dpi=140)
print(f"[plot] {OUT / 'equivariant_diagnostic.png'}", flush=True)

# ----- save artifacts -----
torch.save({"state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "history": history,
            "val_r2_so2_only": val_r2_so2_only,
            "best_atom_hue_r2": best_r2,
            "best_atom": best_atom,
            "ensemble_cv_r2": ensemble_cv_r2,
            "per_atom_hue_r2": per_atom_hue_r2.tolist(),
            "circ_corrs": circ_corrs.tolist(),
            }, OUT / "model_equivariant.pt")

summary = {
    "val_r2_full": float(history[-1]["val_r2"]),
    "val_r2_so2_only": float(val_r2_so2_only),
    "best_atom": int(best_atom),
    "best_atom_hue_r2": float(best_r2),
    "ensemble_64atom_hue_r2_cv": float(ensemble_cv_r2),
    "baselines": {"TopK": 0.874, "L1": 0.882, "Manifold": 0.913},
    "F": 512, "n_so2": 64, "n_trivial": 448, "epochs": EPOCHS,
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), flush=True)
