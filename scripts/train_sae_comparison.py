"""SAE comparison on cogito-L40: TopK vs L1 vs Manifold.

Direct PyTorch implementation. 20-min budget. Saves all artifacts to
runs/sae_comparison/.

Manifold SAE composition (per project_gamfit_composition_engine §6, mGPLVM
style):
  - K_atoms atoms, each with a Circle latent z_k in S^1 (parameterized via
    (cos θ, sin θ) where θ is a linear function of input).
  - per-atom amplitude a_k (continuous, softplus-gated).
  - per-atom gate g_k via IBP-Gumbel approximation (sigmoid(Gumbel-noised logit)
    annealed from soft → hard via temperature schedule).
  - decoder per-atom: ambient direction is a smooth function of θ_k,
    realized as a Fourier basis on the circle:
        u_k(θ) = D_k @ [cos θ, sin θ, cos 2θ, sin 2θ, ..., cos Mθ, sin Mθ, 1]
    so each atom's decoder output is a closed curve in R^D.
  - reconstruction: x_hat = b_dec + Σ_k g_k * a_k * u_k(θ_k).
  - ARD penalty on per-atom amplitude scale: sum_k log(λ + ||D_k||_F^2).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sae_comparison"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)

# -------------------- data --------------------
X_path = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
X = np.load(X_path, mmap_mode="r")
N, D = X.shape
print(f"[data] X shape={X.shape} dtype={X.dtype}", flush=True)

N_COLORS = 949
N_TPL = 28
assert N == N_COLORS * N_TPL, (N, N_COLORS, N_TPL)

# train/val split BY COLOR (so all 28 templates of a color stay together)
rng = np.random.default_rng(0)
color_perm = rng.permutation(N_COLORS)
n_val_colors = int(0.2 * N_COLORS)
val_colors = set(color_perm[:n_val_colors].tolist())
train_colors = set(color_perm[n_val_colors:].tolist())

row_color = np.arange(N) // N_TPL
train_idx = np.where(np.isin(row_color, list(train_colors)))[0]
val_idx = np.where(np.isin(row_color, list(val_colors)))[0]
print(f"[data] train rows={len(train_idx)} val rows={len(val_idx)}", flush=True)

# Materialize into RAM (760 MB at f32 → f16 = 380 MB to save mem on MPS).
# We'll keep train/val as f32 numpy and move per-batch to device.
X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)

# center using train stats only
mu = X_train_np.mean(0)
X_train_np -= mu
X_val_np -= mu
train_var = float((X_train_np ** 2).sum() / X_train_np.size)
val_var = float((X_val_np ** 2).sum() / X_val_np.size)
print(f"[data] train_var={train_var:.4f} val_var={val_var:.4f}", flush=True)

# Move val to device once (small enough)
X_val = torch.from_numpy(X_val_np).to(DEVICE)
val_var_t = X_val.var().item()

# -------------------- xkcd color labels + RGB (for interpretability) ----
def load_xkcd_colors():
    p = ROOT / "experiments" / "xkcd_colors.txt"
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hex_ = parts[0], parts[1]
            hex_ = hex_.lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out.append((name, r, g, b))
    return out

xkcd = load_xkcd_colors()[:N_COLORS]
color_names = [c[0] for c in xkcd]
color_rgb = np.array([(r/255.0, g/255.0, b/255.0) for _, r, g, b in xkcd], dtype=np.float32)

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(1); mn = rgb.min(1); df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-8
    rm = mask & (mx == r); gm = mask & (mx == g); bm = mask & (mx == b)
    h[rm] = ((g[rm]-b[rm])/df[rm]) % 6
    h[gm] = ((b[gm]-r[gm])/df[gm]) + 2
    h[bm] = ((r[bm]-g[bm])/df[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-8, df/np.maximum(mx, 1e-8), 0.0)
    v = mx
    return np.stack([h, s, v], 1)
color_hsv = rgb_to_hsv(color_rgb)

# -------------------- models --------------------
F_ATOMS = 512

class TopKSAE(nn.Module):
    def __init__(self, d_in, n_feat, top_k):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_feat))
        self.W_d = nn.Parameter(torch.randn(n_feat, d_in) * (1.0/np.sqrt(n_feat)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.top_k = top_k
    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        # TopK gating
        topv, topi = z.topk(self.top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topi, F.relu(topv))
        return z_sparse
    def forward(self, x):
        z = self.encode(x)
        recon = z @ self.W_d + self.b_d
        return recon, z

class L1SAE(nn.Module):
    def __init__(self, d_in, n_feat):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_feat))
        self.W_d = nn.Parameter(torch.randn(n_feat, d_in) * (1.0/np.sqrt(n_feat)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        return F.relu(z)
    def forward(self, x):
        z = self.encode(x)
        recon = z @ self.W_d + self.b_d
        return recon, z


class ManifoldSAE(nn.Module):
    """K_atoms atoms, each with a Circle latent + amplitude + IBP-Gumbel gate.

    Decoder per-atom: Fourier basis on S^1 (order M_F).
    """
    def __init__(self, d_in, n_feat, M_F=4):
        super().__init__()
        self.n_feat = n_feat; self.M_F = M_F
        # encoder produces, per atom: gate_logit, theta_pair (2), amp_raw
        # via a single linear map (production: linear encoder for speed)
        self.W_gate = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_gate = nn.Parameter(torch.full((n_feat,), -2.0))  # initially mostly off
        self.W_theta = nn.Parameter(torch.randn(d_in, n_feat * 2) * (1.0/np.sqrt(d_in)))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        # decoder: per-atom Fourier coefficients D_k ∈ R^(2M_F+1, D)
        basis_dim = 2 * M_F + 1
        self.D_k = nn.Parameter(torch.randn(n_feat, basis_dim, d_in) * (0.1/np.sqrt(basis_dim)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        # ARD: per-atom amplitude scale (log)
        self.log_ard = nn.Parameter(torch.zeros(n_feat))
    def theta(self, x):
        xc = x - self.b_d
        tp = xc @ self.W_theta  # (B, F*2)
        B = x.shape[0]
        tp = tp.view(B, self.n_feat, 2)
        # normalize to unit (cos,sin)
        tp = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tp  # (B, F, 2) representing (cos θ, sin θ)
    def fourier_basis(self, cs):
        # cs: (B, F, 2) -> (B, F, 2*M_F+1) = [1, cos θ, sin θ, cos 2θ, sin 2θ,...]
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c)]
        ck, sk = c.clone(), s.clone()
        feats += [ck, sk]
        for m in range(2, self.M_F + 1):
            # cos(mθ) = cos((m-1)θ)cos θ - sin((m-1)θ)sin θ
            ck_new = ck * c - sk * s
            sk_new = sk * c + ck * s
            ck, sk = ck_new, sk_new
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)  # (B, F, 2M+1)
    def forward(self, x, tau=1.0, hard=False):
        xc = x - self.b_d
        gate_logit = xc @ self.W_gate + self.b_gate  # (B, F)
        if self.training:
            # Concrete relaxation
            u = torch.rand_like(gate_logit).clamp(1e-6, 1-1e-6)
            g_noise = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + g_noise) / tau)
        else:
            gate = torch.sigmoid(gate_logit)
        if hard:
            gate = (gate > 0.5).float() + (gate - gate.detach())
        amp_raw = xc @ self.W_amp  # (B, F)
        amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
        cs = self.theta(x)  # (B, F, 2)
        phi = self.fourier_basis(cs)  # (B, F, 2M+1)
        # Memory-efficient: contract F first.
        # weight = gate*amp (B,F); weighted basis: w_phi = weight.unsqueeze(-1) * phi → (B,F,P)
        # Then recon[b,d] = sum_f sum_p w_phi[b,f,p] * D_k[f,p,d]
        # = w_phi.reshape(B, F*P) @ D_k.reshape(F*P, D)  → (B,D). NO (B,F,D) tensor.
        w = (gate * amp).unsqueeze(-1)  # (B,F,1)
        w_phi = (w * phi).reshape(x.shape[0], -1)  # (B, F*P)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])  # (F*P, D)
        recon = w_phi @ D_flat + self.b_d
        return recon, gate, amp
    def encode_for_eval(self, x):
        # returns (B, F) activation magnitude (gate * amp)
        with torch.no_grad():
            xc = x - self.b_d
            gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
            amp_raw = xc @ self.W_amp
            amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
            return gate * amp


# -------------------- training --------------------
def get_batches(X_np, bs, shuffle=True):
    n = X_np.shape[0]
    if shuffle:
        order = np.random.permutation(n)
    else:
        order = np.arange(n)
    for s in range(0, n, bs):
        idx = order[s:s+bs]
        yield torch.from_numpy(X_np[idx]).to(DEVICE)


def train_model(model, name, epochs=10, bs=512, lr=3e-4, sparsity_w=1e-3, top_k=None, manifold=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    history = []
    n_steps_total = epochs * (len(X_train_np) // bs)
    step = 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0; ep_recon = 0.0; ep_sp = 0.0
        for xb in get_batches(X_train_np, bs):
            opt.zero_grad()
            if manifold:
                # anneal tau from 1.0 → 0.3
                tau = max(0.3, 1.0 - 0.7 * (step / max(1, n_steps_total)))
                recon, gate, amp = model(xb, tau=tau)
                recon_loss = F.mse_loss(recon, xb)
                # IBP-style sparsity: penalize sum of gate probabilities
                sp_loss = gate.mean()
                # ARD penalty on amplitude norms
                ard = torch.log(1e-2 + (model.D_k ** 2).sum(dim=(1,2))).mean()
                loss = recon_loss + sparsity_w * sp_loss + 1e-4 * ard
            else:
                recon, z = model(xb)
                recon_loss = F.mse_loss(recon, xb)
                if top_k is None:
                    sp_loss = z.abs().mean()
                    loss = recon_loss + sparsity_w * sp_loss
                else:
                    sp_loss = (z > 0).float().mean()  # tracking, not penalty
                    loss = recon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * xb.shape[0]; ep_n += xb.shape[0]
            ep_recon += recon_loss.item() * xb.shape[0]
            ep_sp += sp_loss.item() * xb.shape[0]
            step += 1
        # val R²
        model.eval()
        with torch.no_grad():
            v_mse = 0.0; v_n = 0
            for i in range(0, X_val.shape[0], 1024):
                xb = X_val[i:i+1024]
                if manifold:
                    recon, _, _ = model(xb, tau=0.3, hard=False)
                else:
                    recon, _ = model(xb)
                v_mse += F.mse_loss(recon, xb, reduction='sum').item()
                v_n += xb.numel()
            v_mse /= v_n
            val_r2 = 1.0 - v_mse / val_var_t
        history.append({"epoch": ep, "train_loss": ep_loss/ep_n, "recon": ep_recon/ep_n,
                        "sp": ep_sp/ep_n, "val_r2": val_r2, "t": time.time()-t0})
        print(f"  [{name}] ep={ep:02d} train={ep_loss/ep_n:.4f} recon={ep_recon/ep_n:.4f}"
              f" sp={ep_sp/ep_n:.4f} val_R²={val_r2:.4f} t={time.time()-t0:.1f}s", flush=True)
    return history


def get_activations(model, X_np, bs=1024, manifold=False):
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], bs):
            xb = torch.from_numpy(X_np[i:i+bs]).to(DEVICE)
            if manifold:
                a = model.encode_for_eval(xb)
            else:
                _, a = model(xb) if not isinstance(model, TopKSAE) else (None, model.encode(xb))
                # for TopK above the destructuring works; for L1 it returns recon, z
                if isinstance(model, TopKSAE):
                    a = model.encode(xb)
                elif isinstance(model, L1SAE):
                    a = model.encode(xb)
            acts.append(a.cpu().numpy())
    return np.concatenate(acts, 0)


def evaluate_model(model, name, manifold=False):
    """Compute: val R², atom activeness, dead-atom rate, per-atom top colors."""
    # val R²
    model.eval()
    with torch.no_grad():
        v_mse = 0.0; v_n = 0
        for i in range(0, X_val.shape[0], 1024):
            xb = X_val[i:i+1024]
            if manifold:
                recon, _, _ = model(xb, tau=0.3, hard=False)
            else:
                recon, _ = model(xb)
            v_mse += F.mse_loss(recon, xb, reduction='sum').item()
            v_n += xb.numel()
        v_mse /= v_n
        val_r2 = 1.0 - v_mse / val_var_t

    # activations on full corpus
    acts_train = get_activations(model, X_train_np, manifold=manifold)
    acts_val = get_activations(model, X_val_np, manifold=manifold)
    acts_all = np.concatenate([acts_train, acts_val], 0)
    # rebuild row index alignment
    idx_all = np.concatenate([train_idx, val_idx], 0)

    # atom activeness: fraction of rows where atom fires (>threshold)
    thresh = 1e-3
    active_per_atom = (acts_all > thresh).mean(0)  # (F,)
    mean_activeness = float(active_per_atom.mean())
    dead = float((active_per_atom < 1e-5).mean())

    # per-atom top colors: aggregate by color (mean across templates)
    F_ = acts_all.shape[1]
    by_color = np.zeros((N_COLORS, F_), dtype=np.float32)
    counts = np.zeros(N_COLORS, dtype=np.int32)
    for r, gidx in enumerate(idx_all):
        ci = gidx // N_TPL
        by_color[ci] += acts_all[r]
        counts[ci] += 1
    by_color /= np.maximum(counts[:, None], 1)

    # rank atoms by total activeness
    atom_order = np.argsort(-active_per_atom)
    top20 = atom_order[:20]
    top_colors_per_atom = {}
    hsv_compactness = []  # std of hue+sat+val within top-10 colors per atom
    for k in top20:
        col_scores = by_color[:, k]
        top10_ci = np.argsort(-col_scores)[:10]
        names = [color_names[c] for c in top10_ci]
        hsv_sub = color_hsv[top10_ci]
        # circular std for hue
        hue_rad = hsv_sub[:, 0] * 2 * np.pi
        circ_var = 1 - np.sqrt(np.cos(hue_rad).mean()**2 + np.sin(hue_rad).mean()**2)
        sv_std = hsv_sub[:, 1:].std(0).mean()
        compact = 0.5 * circ_var + 0.5 * sv_std
        hsv_compactness.append(float(compact))
        top_colors_per_atom[int(k)] = {"colors": names, "score": float(col_scores[top10_ci].mean()),
                                       "hsv_compactness": float(compact)}
    coherence = 1.0 - float(np.mean(hsv_compactness))  # higher = more coherent

    return {
        "name": name,
        "val_r2": val_r2,
        "mean_activeness": mean_activeness,
        "dead_atom_rate": dead,
        "n_active_atoms": int((active_per_atom >= 1e-5).sum()),
        "top_atoms": top_colors_per_atom,
        "atom_activeness_array": active_per_atom.tolist(),
        "top20_coherence": coherence,
    }


# -------------------- run --------------------
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BS = int(os.environ.get("BS", "512"))
print(f"[run] EPOCHS={EPOCHS} BS={BS} F_ATOMS={F_ATOMS}", flush=True)

results = {}

# --- TopK ---
print("\n=== TopK SAE ===", flush=True)
torch.manual_seed(0)
m_topk = TopKSAE(D, F_ATOMS, top_k=32).to(DEVICE)
h_topk = train_model(m_topk, "TopK", epochs=EPOCHS, bs=BS, lr=3e-4, top_k=32)
r_topk = evaluate_model(m_topk, "TopK")
r_topk["history"] = h_topk
results["TopK"] = r_topk
torch.save(m_topk.state_dict(), OUT / "model_topk.pt")

# --- L1 ---
print("\n=== L1 SAE ===", flush=True)
torch.manual_seed(0)
m_l1 = L1SAE(D, F_ATOMS).to(DEVICE)
h_l1 = train_model(m_l1, "L1", epochs=EPOCHS, bs=BS, lr=3e-4, sparsity_w=1e-3)
r_l1 = evaluate_model(m_l1, "L1")
r_l1["history"] = h_l1
results["L1"] = r_l1
torch.save(m_l1.state_dict(), OUT / "model_l1.pt")

# --- Manifold ---
print("\n=== Manifold SAE ===", flush=True)
torch.manual_seed(0)
m_man = ManifoldSAE(D, F_ATOMS, M_F=3).to(DEVICE)
h_man = train_model(m_man, "Manifold", epochs=EPOCHS, bs=BS, lr=3e-4, sparsity_w=1e-2, manifold=True)
r_man = evaluate_model(m_man, "Manifold", manifold=True)
r_man["history"] = h_man
results["Manifold"] = r_man
torch.save(m_man.state_dict(), OUT / "model_manifold.pt")

# -------------------- summary table --------------------
print("\n=== COMPARISON TABLE ===", flush=True)
rows = []
for name in ["TopK", "L1", "Manifold"]:
    r = results[name]
    rows.append([name, r["val_r2"], r["mean_activeness"], r["dead_atom_rate"],
                 r["n_active_atoms"], r["top20_coherence"]])
hdr = ["model", "val_R²", "mean_activeness", "dead_rate", "n_active", "top20_coherence"]
print(f"{'model':<10} {'val_R²':>8} {'mean_act':>10} {'dead':>8} {'n_active':>9} {'coherence':>10}")
for row in rows:
    print(f"{row[0]:<10} {row[1]:>8.4f} {row[2]:>10.5f} {row[3]:>8.4f} {row[4]:>9d} {row[5]:>10.4f}")

# save JSON (strip atom_activeness_array to keep size down)
results_save = {}
for k, v in results.items():
    v2 = dict(v); v2.pop("atom_activeness_array", None)
    results_save[k] = v2
with open(OUT / "comparison.json", "w") as f:
    json.dump(results_save, f, indent=2)

# also save full activeness arrays
np.savez(OUT / "atom_activeness.npz",
         topk=np.array(results["TopK"]["atom_activeness_array"]),
         l1=np.array(results["L1"]["atom_activeness_array"]),
         manifold=np.array(results["Manifold"]["atom_activeness_array"]))

# -------------------- 4-panel figure --------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
for name, color in [("TopK", "C0"), ("L1", "C1"), ("Manifold", "C2")]:
    h = results[name]["history"]
    ax.plot([s["epoch"] for s in h], [s["val_r2"] for s in h], label=name, color=color, lw=2)
ax.set_xlabel("epoch"); ax.set_ylabel("validation R²")
ax.set_title("Validation R² (held-out 20% colors)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
xs = np.arange(3); names = ["TopK", "L1", "Manifold"]
r2s = [results[n]["val_r2"] for n in names]
deads = [results[n]["dead_atom_rate"] for n in names]
ax.bar(xs - 0.2, r2s, 0.4, label="val R²", color="C0")
ax.bar(xs + 0.2, deads, 0.4, label="dead-atom rate", color="C3")
ax.set_xticks(xs); ax.set_xticklabels(names)
ax.legend(); ax.set_title("R² vs Dead-atom rate"); ax.grid(alpha=0.3, axis='y')

ax = axes[1, 0]
for name, color in [("TopK", "C0"), ("L1", "C1"), ("Manifold", "C2")]:
    arr = np.array(results[name]["atom_activeness_array"])
    arr = np.sort(arr)[::-1]
    ax.plot(arr, label=name, color=color, lw=1.5)
ax.set_yscale("log"); ax.set_xlabel("atom rank"); ax.set_ylabel("activeness (firing rate)")
ax.set_title("Atom activeness distribution (sorted)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
coherences = [results[n]["top20_coherence"] for n in names]
ax.bar(xs, coherences, color=["C0", "C1", "C2"])
ax.set_xticks(xs); ax.set_xticklabels(names)
ax.set_ylabel("top-20 HSV coherence (higher=better)")
ax.set_title("Top-20-atom interpretability (HSV coherence)")
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_path = OUT / "comparison_4panel.png"
plt.savefig(fig_path, dpi=120)
print(f"\n[fig] saved {fig_path}", flush=True)

# -------------------- per-model atom-interpretability writeup --------------------
with open(OUT / "interpretability.md", "w") as f:
    f.write("# Per-model atom interpretability\n\n")
    f.write(f"Top-20 atoms by activeness, each showing the 10 most-activating xkcd colors.\n\n")
    for name in ["TopK", "L1", "Manifold"]:
        f.write(f"## {name} SAE\n\n")
        f.write(f"val R² = {results[name]['val_r2']:.4f}  "
                f"dead-atom rate = {results[name]['dead_atom_rate']:.3f}  "
                f"top-20 HSV coherence = {results[name]['top20_coherence']:.3f}\n\n")
        for atom_id, info in results[name]["top_atoms"].items():
            f.write(f"- atom {atom_id} (compact={info['hsv_compactness']:.3f}): "
                    f"{', '.join(info['colors'])}\n")
        f.write("\n")

print(f"[writeup] saved {OUT / 'interpretability.md'}", flush=True)
print(f"\n[done] total time={time.time():.0f}s", flush=True)
