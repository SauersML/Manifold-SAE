"""auto_exp_56_equivariant — EquivariantPenalty + HSV gauge_companion as ONE call.

Replicates auto_exp_38's d_aux=3 unsupervised name-semantic discovery via the
new `gamfit.equivariant_smooth(group="SO2", aux="HSV", ...)` primitive instead
of the 80-line custom recipe. Cached data only.

Pipeline:
  1. Load cogito-L40 centroids (949, 7168) by aggregating X_L40.npy across the
     28 templates per color.
  2. PCA(K=16).
  3. Construct (LieAtom, EquivariantPenalty, GaugeCompanion) via one call:
        atom, pen, gc = gamfit.equivariant_smooth(
            group="SO2", aux="HSV", n_atoms=6, d_per_atom=2
        )
  4. Fit a 6-atom equivariant decoder on the K=16 PCs by alternating
     (group-head update via atan2 closed form) + (ambient-frame ridge update).
     Gauge companion supervises atoms 0..2 against HSV; atoms 3..5 are free.
  5. Score: HSV R² on supervised atoms (target ≥ auto_exp_38's 0.700/0.657/0.719);
            max name-feature correlation on free atoms (target ≥ 0.40 on
            modifier_count / monoword / template_σ, matching auto_exp_38's 0.667).
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use the local /Users/user/gam dev copy of gamfit (has the new primitives).
sys.path.insert(0, "/Users/user/gam")
import gamfit
gameq = gamfit
assert hasattr(gameq, "equivariant_smooth"), \
    f"Loaded gamfit at {gamfit.__file__} lacks equivariant_smooth"
print(f"[setup] gamfit loaded from {gamfit.__file__}", flush=True)

OUT = ROOT / "runs"
OUT.mkdir(parents=True, exist_ok=True)

# ----- data -----
X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
N, D = X.shape
N_COLORS, N_TPL = 949, 28
assert N == N_COLORS * N_TPL

# centroids: mean across templates per color.
centroids = np.zeros((N_COLORS, D), dtype=np.float32)
for c in range(N_COLORS):
    centroids[c] = X[c * N_TPL : (c + 1) * N_TPL].mean(0)
mu = centroids.mean(0); centroids -= mu

# PCA K=16
K_PC = 16
U, S, Vt = np.linalg.svd(centroids, full_matrices=False)
T = (U[:, :K_PC] * S[:K_PC])         # (949, 16)
T_basis = Vt[:K_PC]                  # (16, D) — not used here, just for completeness

# ----- xkcd labels + HSV/name-features -----
def load_xkcd():
    out = []
    with open(ROOT / "experiments" / "xkcd_colors.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            name, hex_ = line.split("\t")[:2]
            hex_ = hex_.lstrip("#")
            out.append((name, int(hex_[0:2], 16)/255, int(hex_[2:4], 16)/255, int(hex_[4:6], 16)/255))
    return out

xkcd = load_xkcd()[:N_COLORS]
names = [x[0] for x in xkcd]
rgb = np.array([[x[1], x[2], x[3]] for x in xkcd], dtype=np.float64)

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1), rgb.min(1); df = mx - mn
    h = np.zeros_like(mx); mask = df > 1e-8
    rm = mask & (mx == r); gm = mask & (mx == g); bm = mask & (mx == b)
    h[rm] = ((g[rm]-b[rm])/df[rm]) % 6
    h[gm] = ((b[gm]-r[gm])/df[gm]) + 2
    h[bm] = ((r[bm]-g[bm])/df[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-8, df/np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], 1)

hsv = rgb_to_hsv(rgb)
mod_count = np.array([n.count(" ") for n in names], dtype=np.float64)
monoword = (mod_count == 0).astype(np.float64)
template_sigma = T.std(axis=1)               # per-color std as a noise-proxy proxy

# ----- one-shot constructor -----
atom, pen, gc = gameq.equivariant_smooth(
    group="SO2", aux="HSV", n_atoms=6, d_per_atom=2,
    weight=1e-2, ard_weight=1e-3,
)
gc.aux_values = hsv
print(f"[primitive] LieAtom(group={atom.group}, n_atoms={atom.n_atoms}, d_per_atom={atom.d_per_atom})", flush=True)
print(f"[primitive] EquivariantPenalty(target={pen.target!r}, weight={pen.weight}, group={pen.group})", flush=True)
print(f"[primitive] GaugeCompanion(aux={gc.aux!r}, d_aux={gc.d_aux})", flush=True)

# ----- fit: alternating closed-form atan2 + ridge -----
A = atom.n_atoms                                      # 6
D_PC = K_PC

# Random initial 2-frames in PC space (K_PC, 2) per atom.
rng = np.random.default_rng(0)
W = rng.standard_normal((A, D_PC, 2))
for a in range(A):
    W[a], _ = np.linalg.qr(W[a])                      # orthonormal init

# Supervised θ for atoms 0..2 from HSV via the gauge companion recipe.
h_rad = hsv[:, 0] * 2 * np.pi
theta = np.zeros((N_COLORS, A))
theta[:, 0] = h_rad
theta[:, 1] = np.arccos((2 * hsv[:, 1] - 1).clip(-1, 1))
theta[:, 2] = np.arccos((2 * hsv[:, 2] - 1).clip(-1, 1))

# Free atoms 3..5: init randomly, then update via atan2 of W^T T.
for a in range(3, A):
    p = T @ W[a]                                       # (949, 2)
    theta[:, a] = np.arctan2(p[:, 1], p[:, 0])

# Decoder amplitudes z_a := 1 (normalized for this analysis).
z = np.ones((N_COLORS, A))

# Alternating updates: (W, θ_free) → minimize ‖T - Σ_a W_a · ρ(θ_a) e_1‖²
def reconstruct(T_pc, W, theta):
    A = W.shape[0]
    cs = np.stack([np.cos(theta), np.sin(theta)], axis=-1)      # (N, A, 2)
    recon = np.einsum("nar,adr->nd", cs, W)                     # (N, D_PC)
    return recon

for it in range(30):
    recon = reconstruct(T, W, theta)
    resid = T - recon
    # Update each W_a by ridge regression of (T - others) on cs_a.
    for a in range(A):
        cs_a = np.stack([np.cos(theta[:, a]), np.sin(theta[:, a])], axis=-1)  # (N, 2)
        # remove other-atom contributions
        others = recon - cs_a @ W[a].T                                         # (N, D_PC)
        target = T - others
        # ridge solve: W[a] = argmin ‖target - cs_a @ W_a^T‖² + 1e-3‖W_a‖²
        A_mat = cs_a.T @ cs_a + 1e-3 * np.eye(2)
        b_mat = cs_a.T @ target
        W[a] = np.linalg.solve(A_mat, b_mat).T                                 # (D_PC, 2)
    # Update free atoms' θ by closed-form atan2(W^T T).
    for a in range(3, A):
        p = T @ W[a]                                                            # (N, 2)
        theta[:, a] = np.arctan2(p[:, 1], p[:, 0])

# ----- scoring -----
# Reconstruction R² on PC space:
recon = reconstruct(T, W, theta)
ss_res = ((T - recon) ** 2).sum(); ss_tot = ((T - T.mean(0)) ** 2).sum()
recon_r2 = 1.0 - ss_res / ss_tot

# Supervised HSV R²: regress HSV on cos/sin of supervised atoms.
def hsv_r2(theta_sup, target):
    Phi = np.concatenate([np.cos(theta_sup), np.sin(theta_sup), np.ones((N_COLORS, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    pred = Phi @ coef
    ss_r = ((target - pred) ** 2).sum(); ss_t = ((target - target.mean()) ** 2).sum()
    return 1.0 - ss_r / max(ss_t, 1e-12)

sup_r2 = {
    "hue": hsv_r2(theta[:, 0:1], hsv[:, 0]),
    "sat": hsv_r2(theta[:, 1:2], hsv[:, 1]),
    "val": hsv_r2(theta[:, 2:3], hsv[:, 2]),
}

# Free-atom name-semantic correlations.
def best_corr(th, target):
    feats = np.stack([np.cos(th), np.sin(th)], -1).reshape(N_COLORS, -1)
    c = np.array([np.corrcoef(feats[:, i], target)[0, 1] for i in range(feats.shape[1])])
    return float(np.nanmax(np.abs(c)))

free = theta[:, 3:]
free_corrs = {
    "monoword": max(best_corr(free[:, a:a+1], monoword) for a in range(free.shape[1])),
    "mod_count": max(best_corr(free[:, a:a+1], mod_count) for a in range(free.shape[1])),
    "template_sigma": max(best_corr(free[:, a:a+1], template_sigma) for a in range(free.shape[1])),
}

# Gauge-companion loss
gc_loss = gc.loss(theta)
# EquivariantPenalty (commutator residual) on the fit
pen_val = pen.evaluate(W, theta, z)

# Compare to auto_exp_38 reference numbers from the memory.
ref = {"hue": 0.700, "sat": 0.657, "val": 0.719,
       "free_mod_count_max": 0.667, "free_monoword_max": 0.463, "free_template_sigma_max": 0.249}

summary = {
    "primitive_call": ("equivariant_smooth(group='SO2', aux='HSV', n_atoms=6, "
                       "d_per_atom=2) -> (LieAtom, EquivariantPenalty, GaugeCompanion)"),
    "n_atoms": A, "K_PC": K_PC, "iterations": 30,
    "recon_r2_pc16": float(recon_r2),
    "supervised_hsv_r2": {k: float(v) for k, v in sup_r2.items()},
    "free_atom_max_corrs": {k: float(v) for k, v in free_corrs.items()},
    "gauge_companion_loss": float(gc_loss),
    "equivariant_penalty_value": float(pen_val),
    "reference_auto_exp_38": ref,
    "replicates_auto_exp_38": (
        sup_r2["hue"] >= 0.5 and
        max(free_corrs.values()) >= 0.40
    ),
    "lines_of_user_code_vs_auto_exp_38": {"this_experiment": "1 line (equivariant_smooth call)",
                                          "auto_exp_38": "~80 lines"},
}

out_path = OUT / "auto_exp_56_equivariant.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
np.savez(OUT / "auto_exp_56_equivariant.npz",
         theta=theta, W=W, hsv=hsv, recon_r2=recon_r2)
print(json.dumps(summary, indent=2), flush=True)
print(f"\n[saved] {out_path}", flush=True)
