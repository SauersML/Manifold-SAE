"""auto_exp_68_hyperbolic.py — Hyperbolic SAE on cogito-L40.

Trains scripts/train_hyperbolic_sae.py, then visualizes atoms on the Poincaré
disk (PCA-2D of tangent space). Goal: see hierarchical clustering — coarse color
families near origin, specific xkcd names near the boundary.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gamfit.torch import PoincareAtoms  # noqa: E402

from manifold_sae.hyperbolic_sae import HyperbolicSAE  # noqa: E402

# Standalone exp_0 / log_0 at the origin, built directly on the gamfit
# primitive (no shim module). Closed form for the Poincaré ball with geometric
# curvature c = -κ (κ > 0): with conformal factor λ and ‖·‖ the Euclidean norm,
#   exp_0(v) = tanh(√κ ‖v‖) · v / (√κ ‖v‖)
#   log_0(x) = atanh(√κ ‖x‖) · x / (√κ ‖x‖)
# These are the canonical maps the primitive implements internally; here we use
# them only for tangent-space PCA visualization. project_into_ball keeps inputs
# strictly interior so atanh never diverges.
_NORM_EPS = 1e-15


def _proj(x, kappa=1.0):
    atoms = PoincareAtoms(F=1, ball_dim=int(x.shape[-1]), curvature=-float(kappa))
    p = atoms.project_into_ball(x.contiguous())
    # Clamp strictly inside (the primitive saturates onto the boundary).
    max_norm = (1.0 - 1e-6) / (float(kappa) ** 0.5)
    n = p.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    return p * torch.clamp(max_norm / n, max=1.0)


def log_0(x, c=1.0):
    x = _proj(x, c)
    sc = float(c) ** 0.5
    n = x.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    return torch.atanh((sc * n).clamp(max=1.0 - 1e-7)) * x / (sc * n)


def exp_0(v, c=1.0):
    sc = float(c) ** 0.5
    n = v.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    out = torch.tanh(sc * n) * v / (sc * n)
    return _proj(out, c)

OUT_DIR = ROOT / "runs" / "HYPERBOLIC_SAE_COGITO"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 1) Run trainer (in-process, simpler than subprocess) ----------------------
print("[exp68] launching training", flush=True)
proc = subprocess.run(
    ["uv", "run", "python", str(ROOT / "scripts" / "train_hyperbolic_sae.py")],
    cwd=str(ROOT),
)
if proc.returncode != 0:
    print(f"[exp68] training failed rc={proc.returncode}", flush=True)
    sys.exit(proc.returncode)


# 2) Load metrics + atoms ---------------------------------------------------
with open(OUT_DIR / "metrics.json") as f:
    metrics = json.load(f)
atoms = np.load(OUT_DIR / "atoms.npy")   # (F, d) in ball
radii = np.load(OUT_DIR / "atom_radii.npy")  # (F,) geodesic dist from origin
F, d = atoms.shape
print(f"[exp68] atoms shape {atoms.shape}, mean radius {radii.mean():.3f}, "
      f"max {radii.max():.3f}", flush=True)


# 3) Score atoms against xkcd categorical structure -------------------------
# Load cogito + xkcd metadata using the same conventions as train_sae_comparison.
sd = torch.load(OUT_DIR / "hyperbolic_state.pt", map_location="cpu")
# gamfit-native keys: W_gate.weight (F, D), atoms_dict.atoms (F, d).
D_in = sd["W_gate.weight"].shape[1]
F_in = sd["atoms_dict.atoms"].shape[0]
d_in = sd["atoms_dict.atoms"].shape[1]
model = HyperbolicSAE(input_dim=D_in, n_features=F_in, ball_dim=d_in,
                      curvature=1.0, sparsity_weight=1e-3)
model.load_state_dict(sd)
model.eval()

X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
mu = X[:].mean(0).astype(np.float32)
X_c = X[:].astype(np.float32) - mu

N = X_c.shape[0]
N_TPL = 28
N_COLORS = 949
row_color = np.arange(N) // N_TPL

# encode all in chunks
gates_all = np.zeros((N, F_in), dtype=np.float32)
BS = 1024
with torch.no_grad():
    for i in range(0, N, BS):
        xb = torch.from_numpy(X_c[i:i+BS])
        _, g = model.encode(xb)
        gates_all[i:i+BS] = g.cpu().numpy()


# top-activating color per atom = color whose mean gate is max
mean_gate_per_color = np.zeros((N_COLORS, F_in), dtype=np.float32)
for c in range(N_COLORS):
    mean_gate_per_color[c] = gates_all[row_color == c].mean(0)
top_color = mean_gate_per_color.argmax(0)  # (F,) color idx per atom


# 4) xkcd category covariates ----------------------------------------------
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
names = [c[0] for c in xkcd]
rgb = np.array([(r/255.0, g/255.0, b/255.0) for _, r, g, b in xkcd])

# Per-color covariates:
#   modifier_count = #words in name - 1   (0 = monoword, 1 = "dark red", ...)
mod_count = np.array([len(n.split()) - 1 for n in names])
monoword = (mod_count == 0).astype(int)
# Hue octant 0..7
def rgb_to_hue(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(1); mn = rgb.min(1); df = mx - mn + 1e-8
    h = np.zeros_like(mx)
    rm = (mx == r); gm = (mx == g); bm = (mx == b)
    h[rm] = ((g[rm]-b[rm])/df[rm]) % 6
    h[gm] = ((b[gm]-r[gm])/df[gm]) + 2
    h[bm] = ((r[bm]-g[bm])/df[bm]) + 4
    return (h * 60.0) % 360.0
hue = rgb_to_hue(rgb)
hue_octant = (hue / 45.0).astype(int) % 8

atom_rgb = rgb[top_color]
atom_modcount = mod_count[top_color]
atom_monoword = monoword[top_color]
atom_hueoct = hue_octant[top_color]


# 5) PCA-2D of tangent representations of atoms -----------------------------
# Atoms live in B^d ⊂ R^d. Their log_0 in tangent space is the natural place to
# PCA before re-mapping. For visualization on the Poincaré DISK we keep them
# as-is in R^d, PCA to 2D, then map back through exp_0 to the disk.
A = torch.from_numpy(atoms)
T = log_0(A, c=1.0).numpy()  # (F, d)
T_c = T - T.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(T_c, full_matrices=False)
T2 = T_c @ Vt[:2].T  # (F, 2)
# Re-embed into disk (scale to keep inside)
scale = 0.95 / max(1e-6, np.linalg.norm(T2, axis=1).max())
disk2 = exp_0(torch.from_numpy(T2 * scale), c=1.0).numpy()


# 6) Visualization ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, color_arr, title in [
    (axes[0], atom_rgb, "by top-color RGB"),
    (axes[1], plt.cm.viridis(atom_modcount / max(1, atom_modcount.max())),
     "by modifier_count"),
    (axes[2], plt.cm.hsv(atom_hueoct / 8.0), "by hue octant"),
]:
    # Disk boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.8, alpha=0.4)
    sizes = 30 + 200 * (radii / max(1e-6, radii.max()))
    ax.scatter(disk2[:, 0], disk2[:, 1], c=color_arr, s=sizes, alpha=0.85,
               edgecolor="k", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(
    f"Hyperbolic SAE atoms on Poincaré disk (PCA of tangent space)\n"
    f"F={F}, d={d}, val R²(hyp)={metrics['hyp']['val_r2']:.3f}, "
    f"L1={metrics['l1']['val_r2']:.3f}, Eu={metrics['eu']['val_r2']:.3f}",
    y=1.02)
fig.tight_layout()
fig_path = OUT_DIR / "poincare_atoms.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"[exp68] saved {fig_path}", flush=True)


# 7) Hierarchical-clustering quantitative finding --------------------------
# Hypothesis: atom radius (dist from origin in ball) correlates with
# specificity = (modifier_count of its top color). High r ⇒ leaf concept.
from scipy.stats import spearmanr
rho_modcount, p_modcount = spearmanr(radii, atom_modcount)
# Monoword (generic single-word colors) should lie near origin (small radius).
mono_radii = radii[atom_monoword == 1]
modi_radii = radii[atom_monoword == 0]
mono_mean = float(mono_radii.mean()) if len(mono_radii) else float("nan")
modi_mean = float(modi_radii.mean()) if len(modi_radii) else float("nan")

cluster_stats = {
    "spearman_radius_vs_modcount": {"rho": float(rho_modcount),
                                    "p": float(p_modcount)},
    "mean_radius_monoword_atoms": mono_mean,
    "mean_radius_modifier_atoms": modi_mean,
    "n_monoword_atoms": int((atom_monoword == 1).sum()),
    "n_modifier_atoms": int((atom_monoword == 0).sum()),
    "atom_radius_min": float(radii.min()),
    "atom_radius_max": float(radii.max()),
    "atom_radius_mean": float(radii.mean()),
}

summary = {
    "metrics": metrics,
    "cluster": cluster_stats,
    "visualization": str(fig_path),
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"[exp68] done. fig={fig_path}", flush=True)
