"""auto_exp_45: IBP-Gumbel per-row active sets across color groups (cogito-L40).

GOAL: test whether IBP-MAP per-row active-set assignments DIFFER across color
groups (warm vs cool vs neutral). If they do, color-class structure in
cogito-L40 is sparse-coded per row across atoms — a SAE-style finding.

PATH: tried `from gamfit import IBPAssignmentPenalty, GumbelTemperatureSchedule`
first. If the production gamfit lacks them (today's 0.1.112 still does), fall
back to a Python emulator: per-row Bernoulli indicators Z (N x d_latent) fit by
EM-style coordinate descent with Gumbel-softmax noise annealed tau 1.0 -> 0.1
under a global IBP-style sparsity prior (logit-prior α per atom shared across
rows).

DESIGN:
  - d_latent=8 atoms, K=16 PCA features
  - Global atom dictionary W (K, d_latent) — top-8 residual PCs after HSV gauge
    fix (matches exp_38 free-axes construction, extended from 3 to 8 atoms)
  - Per-row indicators Z_ij in [0,1] anneal via Gumbel-softmax-Bernoulli
  - Reconstruction loss + IBP-like log-Bernoulli prior on Z + ARD on W cols
  - HSV-supervised gauge fix on the first 3 columns of W (axes 0,1,2)

The 8 atoms here are FIXED axes; per-row "active set" is the binarized Z row,
threshold 0.5. The hypothesis is about which atoms light up for which color
groups, not about how many atoms exist.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
RUN_DIR = ROOT / "runs"
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_NPZ = RUN_DIR / "auto_exp_45_ibp_groups.npz"
OUT_JSON = RUN_DIR / "auto_exp_45.json"
ABORT_JSON = RUN_DIR / "auto_exp_45_abort.json"
MEMORY_MD = Path(
    "/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/"
    "project_cogito_recovery_at_d_aux_3.md"
)

N_TEMPLATES = 28
K_PCS = 16
D_LATENT = 8
D_SUP = 3  # HSV-supervised gauge-fix axes (0,1,2)
N_OUTER = 12
TAU_START = 1.0
TAU_END = 0.1
IBP_ALPHA = 2.5  # IBP-style sparsity (lower -> sparser); aiming ~3 atoms/row
ARD_PRUNE_TAU = 1e-2
SIGMA_AUX = 0.5
AUX_WEIGHT = 8.0
N_ITER_SUP = 250
THRESH = 0.5
SEED = 45


# ---------------- pipeline helpers (from auto_exp_38) ----------------------
def load_xkcd_rgb(n_colors: int) -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((int(hexs[0:2], 16) / 255.0,
                        int(hexs[2:4], 16) / 255.0,
                        int(hexs[4:6], 16) / 255.0))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def per_color_stats_mmap(x_mmap, n_t, basis, k_pcs):
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    mu = basis["mu"]; sigma = basis["sigma"]; Vt = basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t; e = ce * n_t
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T
        Z = Z[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def color_groups(hsv: np.ndarray) -> np.ndarray:
    """0=warm, 1=cool, 2=neutral. hue in [0,1]."""
    hue_deg = hsv[:, 0] * 360.0
    sat = hsv[:, 1]
    groups = np.full(hsv.shape[0], 2, dtype=np.int64)  # neutral default
    saturated = sat >= 0.2
    warm = saturated & ((hue_deg < 60) | (hue_deg >= 300))
    cool = saturated & (hue_deg >= 180) & (hue_deg < 240)
    groups[warm] = 0
    groups[cool] = 1
    return groups


# ---------------- HSV gauge-fix on sup axes (from exp_38) ------------------
def fit_aux_supervised_hsv(T0, hsv, n_iter=N_ITER_SUP):
    rng = np.random.default_rng(SEED)
    n_c, K = T0.shape
    d_aux = hsv.shape[1]
    Tc = T0 - T0.mean(0, keepdims=True)
    aux_mu = hsv.mean(0, keepdims=True)
    aux_sd = hsv.std(0, keepdims=True).clip(min=1e-8)
    ac = (hsv - aux_mu) / aux_sd
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(d_aux)
    w_row = 1.0 / (SIGMA_AUX ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(np.var(ac))
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ ac / n_c
    for _ in range(n_iter):
        for j in range(d_aux):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    r2 = 1.0 - ((ac - Tc @ W) ** 2).sum(0) / (ac ** 2).sum(0).clip(min=1e-12)
    return W, r2


def build_dictionary(T0, W_sup, d_latent=D_LATENT):
    """Compose dictionary as [W_sup (gauge-fixed) | top residual PCs]."""
    Tc = T0 - T0.mean(0, keepdims=True)
    Q, _ = np.linalg.qr(W_sup)
    P_perp = np.eye(W_sup.shape[0]) - Q @ Q.T
    Tc_perp = Tc @ P_perp
    _, _, Vt_perp = np.linalg.svd(Tc_perp, full_matrices=False)
    d_free = d_latent - W_sup.shape[1]
    W_free = Vt_perp[:d_free].T  # (K, d_free)
    W_all = np.concatenate([W_sup, W_free], axis=1)  # (K, d_latent)
    # Normalize each atom direction (column) to unit-norm for stable codes
    W_norm = W_all / np.linalg.norm(W_all, axis=0, keepdims=True).clip(min=1e-12)
    return W_norm


# ---------------- IBP-Gumbel emulator ---------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gumbel_sample(shape, rng):
    u = rng.uniform(1e-8, 1 - 1e-8, size=shape)
    return -np.log(-np.log(u))


def fit_ibp_gumbel(T, W, rng, n_outer=N_OUTER,
                   tau_start=TAU_START, tau_end=TAU_END,
                   alpha=IBP_ALPHA):
    """Per-row Bernoulli indicator fit via Gumbel-softmax + closed-form amps.

    For each row i and atom j, indicator z_ij in [0,1] selects whether atom j
    contributes. Given z, per-row amplitudes a_ij = z_ij * coeff_ij where
    coeff = (W[:, active]^T W[:, active])^-1 W[:, active]^T T_i (least squares
    on the active subset, soft-relaxed via Z weighting).

    Soft relaxation: A = (W^T W * (z z^T) + lam I)^-1 (W^T T_i * z)
    Reconstruct: T_hat_i = W @ (z * A).
    """
    n_c, K = T.shape
    d = W.shape[1]
    # init logits modestly negative (~0.3 prob) -> learn sparsity
    logits = rng.normal(loc=-0.5, scale=0.1, size=(n_c, d))
    # Per-atom log-prior bias log(alpha/d) - log(1 - alpha/d), shared across rows
    p_atom = alpha / d
    prior_logit = np.log(p_atom / max(1 - p_atom, 1e-6))

    schedule = np.linspace(tau_start, tau_end, n_outer)
    lam = 1e-3  # tiny ridge on amplitude solve
    WtW = W.T @ W  # (d, d)
    WtT = T @ W    # (n_c, d) — least squares targets-per-atom

    A = np.zeros_like(logits)  # amplitudes
    final_loss = np.inf
    for outer, tau in enumerate(schedule):
        # ---- E-step: sample z via Gumbel-sigmoid relaxation
        g1 = gumbel_sample(logits.shape, rng)
        g0 = gumbel_sample(logits.shape, rng)
        z_logits = (logits + g1 - g0) / max(tau, 1e-4)
        Z = sigmoid(z_logits)  # (n_c, d)
        # ---- amplitude closed-form per row: solve (WtW * (z z^T) + lam I) a = z * WtT_i
        for i in range(n_c):
            zi = Z[i]
            M = WtW * np.outer(zi, zi) + lam * np.eye(d)
            rhs = zi * WtT[i]
            A[i] = np.linalg.solve(M, rhs)
        # Reconstruction
        recon = (Z * A) @ W.T  # (n_c, K)
        resid = T - recon
        recon_loss = float((resid ** 2).mean())
        # ---- M-step on logits: gradient on Bernoulli reparameterised loss.
        # Approximate: target z_i ~ 1 if removing it spikes recon loss.
        # Compute per-(i,j) marginal gain from setting z_ij=1 vs 0 holding A fixed.
        # contribution of atom j to recon_i: A_ij * W[:, j]
        # If we toggle z_ij off: recon_i' = recon_i - Z_ij * A_ij * W[:, j]
        contrib = (Z * A)[:, :, None] * W.T[None, :, :]  # (n_c, d, K)
        recon_wo = recon[:, None, :] - contrib            # (n_c, d, K) — recon if atom j off
        loss_wo = ((T[:, None, :] - recon_wo) ** 2).mean(axis=2)  # (n_c, d)
        loss_with = ((T[:, None, :] - recon[:, None, :]) ** 2).mean(axis=2)  # broadcast
        delta = loss_wo - loss_with  # >0 means atom helps reconstruction
        # Target prob via softmax-style: p_target = sigmoid(beta * delta / sigma_d + prior_logit)
        # Normalize delta by its scale so beta is dimensionless across runs.
        sigma_d = float(delta.std()) + 1e-8
        beta = 6.0  # sharpness — moderate so the IBP prior actually constrains
        target_logits = beta * (delta / sigma_d) + prior_logit
        # Soft update toward target
        lr = 0.5
        logits = (1 - lr) * logits + lr * target_logits
        final_loss = recon_loss
        if outer % 3 == 0:
            mean_active = float((Z > THRESH).sum(axis=1).mean())
            print(f"  [outer {outer}] tau={tau:.3f} recon={recon_loss:.4f} "
                  f"mean_active={mean_active:.2f}")

    # Final hard pass at tau_end with no gumbel noise -> deterministic Z
    Z_hard = sigmoid(logits)  # (n_c, d), no noise
    # Recompute amplitudes with hard Z
    for i in range(n_c):
        zi = Z_hard[i]
        M = WtW * np.outer(zi, zi) + lam * np.eye(d)
        rhs = zi * WtT[i]
        A[i] = np.linalg.solve(M, rhs)
    recon = (Z_hard * A) @ W.T
    final_loss = float(((T - recon) ** 2).mean())
    return Z_hard, A, logits, final_loss


# ---------------- group analysis -------------------------------------------
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = set(np.where(a)[0].tolist())
    b = set(np.where(b)[0].tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def analyze_groups(Z_bin: np.ndarray, groups: np.ndarray, group_names):
    out = {}
    for gi, gname in enumerate(group_names):
        mask = groups == gi
        n = int(mask.sum())
        if n == 0:
            out[gname] = {"n": 0, "mean_active": None,
                          "typical_set": [], "freq": []}
            continue
        Zg = Z_bin[mask]
        freq = Zg.mean(axis=0)  # per-atom frequency
        typical = (freq >= 0.5).astype(np.int64)
        out[gname] = {
            "n": n,
            "mean_active_per_row": float(Zg.sum(axis=1).mean()),
            "typical_set": typical.tolist(),
            "freq": freq.tolist(),
            "n_atoms_active_in_majority": int(typical.sum()),
        }
    return out


# ---------------- main -----------------------------------------------------
def try_gamfit_path():
    try:
        from gamfit import IBPAssignmentPenalty, GumbelTemperatureSchedule  # noqa: F401
        return "gamfit_production"
    except Exception:
        return "python_emulator"


def main():
    t0 = time.time()
    path = try_gamfit_path()
    print(f"[gamfit] path={path}")
    if path == "gamfit_production":
        print("[gamfit] production IBP+Gumbel found — using emulator anyway for "
              "consistency since no production wiring yet in this script")
        # If production exists we'd wire here. Keep emulator pipeline.

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    print("[pca] basis K=64 loaded")
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    groups = color_groups(hsv)
    group_names = ["warm", "cool", "neutral"]
    for gi, gn in enumerate(group_names):
        print(f"[group] {gn}: n={int((groups == gi).sum())}")

    # HSV gauge fix
    W_sup, r2_hsv = fit_aux_supervised_hsv(T0, hsv)
    print(f"[gauge] R^2 hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} val={r2_hsv[2]:.3f}")

    W = build_dictionary(T0, W_sup, d_latent=D_LATENT)
    print(f"[dict] W={W.shape} (cols 0..{D_SUP-1} gauge-fixed, rest residual PCs)")

    # IBP-Gumbel fit
    Tc = T0 - T0.mean(0, keepdims=True)
    rng = np.random.default_rng(SEED)
    print("[fit] IBP-Gumbel per-row indicators (emulator)")
    Z_soft, A, logits, recon_loss = fit_ibp_gumbel(Tc, W, rng)
    Z_bin = (Z_soft >= THRESH).astype(np.int64)
    print(f"[fit] final recon_loss={recon_loss:.4f}")
    print(f"[fit] global mean_active_per_row={Z_bin.sum(axis=1).mean():.2f}")

    # Per-group analysis
    analysis = analyze_groups(Z_bin, groups, group_names)
    # Jaccard cross-group
    typical = {gn: np.array(analysis[gn]["typical_set"]) for gn in group_names}
    jacc = {}
    for i, a in enumerate(group_names):
        for b in group_names[i + 1:]:
            jacc[f"{a}_vs_{b}"] = float(jaccard(typical[a], typical[b]))

    # Stdout table
    print("\n=== PER-GROUP ACTIVE SETS ===")
    print(f"{'group':<10}{'n':>5}{'maj_atoms':>12}{'mean/row':>12}"
          f"{'typical_set':>30}")
    for gn in group_names:
        a = analysis[gn]
        ts = "{" + ",".join(str(j) for j in np.where(np.array(a['typical_set']))[0]) + "}"
        print(f"{gn:<10}{a['n']:>5}{a['n_atoms_active_in_majority']:>12}"
              f"{a['mean_active_per_row']:>12.2f}{ts:>30}")
    print("\n=== JACCARD ===")
    for k, v in jacc.items():
        print(f"  {k}: {v:.3f}")

    # Verdict
    pairs = list(jacc.values())
    differentiated = all(v < 0.5 for v in pairs)
    verdict = ("DIFFERENTIATED: color-class structure IS sparse-coded per row"
               if differentiated
               else "NOT differentiated: typical active-sets overlap across "
                    "groups (Jaccard >= 0.5 somewhere)")
    print(f"\n[verdict] {verdict}")

    # Save npz
    np.savez(
        OUT_NPZ,
        Z_soft=Z_soft, Z_bin=Z_bin, A=A, logits=logits,
        group_labels=groups,
        W=W, T0=T0, hsv=hsv,
        typical_warm=typical["warm"],
        typical_cool=typical["cool"],
        typical_neutral=typical["neutral"],
    )
    print(f"[npz] saved {OUT_NPZ}")

    out = {
        "experiment": "auto_exp_45",
        "gamfit_path": path,
        "config": {
            "K_PCS": K_PCS, "D_LATENT": D_LATENT, "D_SUP": D_SUP,
            "N_OUTER": N_OUTER, "TAU_START": TAU_START, "TAU_END": TAU_END,
            "IBP_ALPHA": IBP_ALPHA, "THRESH": THRESH, "SEED": SEED,
            "n_colors": int(n_c),
        },
        "R2_hsv": {"hue": float(r2_hsv[0]), "sat": float(r2_hsv[1]),
                   "val": float(r2_hsv[2])},
        "recon_loss": float(recon_loss),
        "global_mean_active_per_row": float(Z_bin.sum(axis=1).mean()),
        "group_analysis": analysis,
        "jaccard_cross_group": jacc,
        "differentiated": bool(differentiated),
        "verdict": verdict,
        "runtime_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")

    # Append findings to memory
    append_memo(path, analysis, jacc, differentiated, recon_loss, r2_hsv)
    print(f"[memory] appended findings to {MEMORY_MD}")
    print(f"[runtime] {time.time() - t0:.1f}s")


def append_memo(path, analysis, jacc, differentiated, recon_loss, r2_hsv):
    header = "\n\n## auto_exp_45: IBP-Gumbel per-row active sets across color groups\n"
    body_lines = [
        f"- path={path}; recon_loss={recon_loss:.4f}; "
        f"R^2(hue)={r2_hsv[0]:.3f}",
        "- Per-group typical active-set (atoms active in >=50% of rows):",
    ]
    for gn in ["warm", "cool", "neutral"]:
        a = analysis[gn]
        ts = np.where(np.array(a["typical_set"]))[0].tolist()
        body_lines.append(
            f"  - {gn}: n={a['n']}, maj_atoms={a['n_atoms_active_in_majority']}, "
            f"mean/row={a['mean_active_per_row']:.2f}, typical={ts}"
        )
    body_lines.append("- Jaccard between typical sets:")
    for k, v in jacc.items():
        body_lines.append(f"  - {k}: {v:.3f}")
    body_lines.append(
        f"- VERDICT: {'DIFFERENTIATED' if differentiated else 'NOT differentiated'} "
        f"(all pairs Jaccard<0.5? {differentiated})"
    )
    if differentiated:
        body_lines.append(
            "- Implication: color-class structure in cogito-L40 IS sparse-coded "
            "per row across atoms — SAE-style finding consistent with the "
            "composition-engine §4(c) story."
        )
    else:
        body_lines.append(
            "- Implication: with the current emulator + d_latent=8, typical "
            "active sets overlap across color groups; the per-row IBP-MAP "
            "signal is not class-differentiated under these settings."
        )
    text = header + "\n".join(body_lines) + "\n"
    with open(MEMORY_MD, "a") as f:
        f.write(text)


if __name__ == "__main__":
    main()
