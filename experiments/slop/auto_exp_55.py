"""auto_exp_55: validate gamfit composition-engine recipe on SYNTHETIC manifolds.

GOAL
----
Tests transferability claim from auto_exp_54 on synthetic data with KNOWN
ground-truth topology. 4 topologies x 3 recipes = 12-cell scoring matrix.

Topologies (all N=949 rows, 16 noisy PC dims):
  1. Linear  (R^3)        : 3 Gaussian latents
  2. Circle  (S^1 x R^2)  : 1 angle + 2 Gaussians
  3. Sphere  (S^2 x R^1)  : (lat, lon) + 1 Gaussian
  4. Torus   (T^2 x R^1)  : (theta1, theta2) + 1 Gaussian

Recipes:
  (A) HSV-style SUPERVISED gauge-fix: regress 16D PCs onto ground-truth
      latents (linear/Procrustes for Linear; sin/cos expansion for circles).
      Report R^2 of recovered axes vs truth.
  (B) UNSUPERVISED Riemannian fit with the matching manifold (Linear=PCA,
      Circle=alternating vM, Sphere=alt project-to-S^2, Torus=alt 2-circle).
      Report Spearman or circular correlation with truth.
  (C) TopologyAutoSelector (GOLD test): emulated via TK-normalized recon
      loss with model-dimension penalty across all 4 candidate topologies.
      Does the engine PICK the right one?

gamfit path
-----------
gamfit 0.1.112 (production) exposes Sphere() basis + sphere_frechet_mean
but NO Circle/Torus/TopologyAutoSelector wrappers. So Recipe (B,C) use
documented Python emulators (vM-alt for S^1, Stiefel-projection alt for
S^2, two-circle alt for T^2). gamfit_path = "python_emulator".

Outputs
-------
runs/auto_exp_55_synthetic_benchmark.npz
runs/auto_exp_55_synthetic_benchmark.png
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_55_synthetic_benchmark.npz"
OUT_PNG = ROOT / "runs" / "auto_exp_55_synthetic_benchmark.png"

N_ROWS = 949
D_PC = 16
NOISE_SIGMA = 0.10
SEED = 55
TOPOS = ["linear", "circle", "sphere", "torus"]


# ---------------------- gamfit detection -----------------------------------
def detect_gamfit():
    try:
        import gamfit
        ver = getattr(gamfit, "__version__", "unknown")
        has_topo = any(
            "opologyauto" in n.lower() or "selector" in n.lower()
            for n in dir(gamfit)
        )
        has_circle = any("circle" in n.lower() for n in dir(gamfit))
        has_torus = any("torus" in n.lower() for n in dir(gamfit))
        has_sphere = any(n == "Sphere" for n in dir(gamfit))
        return ver, dict(topo=has_topo, circle=has_circle,
                         torus=has_torus, sphere=has_sphere)
    except Exception as e:
        return f"unavailable:{e!r}", {}


# ---------------------- synthetic data generators --------------------------
def make_dataset(topo: str, seed: int):
    """Return (X, latents_dict) with X shape (N, D_PC), latents = truth."""
    rng = np.random.default_rng(seed)
    N = N_ROWS
    if topo == "linear":
        Z = rng.normal(size=(N, 3))
        intrinsic = Z
        latents = {"Z": Z}
    elif topo == "circle":
        theta = rng.uniform(-np.pi, np.pi, size=N)
        nuis = rng.normal(size=(N, 2))
        Z = np.column_stack([np.cos(theta), np.sin(theta), nuis])  # (N, 4)
        intrinsic = Z
        latents = {"theta": theta, "nuis": nuis}
    elif topo == "sphere":
        # uniform on S^2 via Gaussian -> normalize
        g = rng.normal(size=(N, 3))
        S = g / np.linalg.norm(g, axis=1, keepdims=True)
        nuis = rng.normal(size=(N, 1))
        Z = np.concatenate([S, nuis], axis=1)  # (N, 4)
        intrinsic = Z
        latents = {"S": S, "nuis": nuis[:, 0]}
    elif topo == "torus":
        th1 = rng.uniform(-np.pi, np.pi, size=N)
        th2 = rng.uniform(-np.pi, np.pi, size=N)
        nuis = rng.normal(size=(N, 1))
        Z = np.column_stack([np.cos(th1), np.sin(th1),
                             np.cos(th2), np.sin(th2), nuis[:, 0]])
        intrinsic = Z
        latents = {"th1": th1, "th2": th2, "nuis": nuis[:, 0]}
    else:
        raise ValueError(topo)

    # Random orthogonal projection intrinsic -> R^{D_PC}, then add noise
    d_in = intrinsic.shape[1]
    G = rng.normal(size=(d_in, D_PC))
    # orthonormal rows in d_in space, mapping to D_PC
    Q, _ = np.linalg.qr(G.T)  # (D_PC, d_in)
    W = Q.T                    # (d_in, D_PC) with orthonormal rows
    X = intrinsic @ W
    # rescale to comparable variance across topos
    X = X / X.std()
    X = X + NOISE_SIGMA * rng.normal(size=X.shape)
    return X.astype(np.float64), latents


# ---------------------- Recipe A: supervised gauge-fit ---------------------
def recipe_A_supervised(X, latents, topo):
    """Linear-regress X -> truth-basis; return R^2."""
    if topo == "linear":
        Y = latents["Z"]                          # (N, 3)
    elif topo == "circle":
        Y = np.column_stack([np.cos(latents["theta"]), np.sin(latents["theta"])])
    elif topo == "sphere":
        Y = latents["S"]                          # (N, 3)
    elif topo == "torus":
        Y = np.column_stack([np.cos(latents["th1"]), np.sin(latents["th1"]),
                             np.cos(latents["th2"]), np.sin(latents["th2"])])
    Xc = X - X.mean(0)
    # ridge with very small reg to be safe
    A = Xc.T @ Xc + 1e-6 * np.eye(Xc.shape[1])
    B = np.linalg.solve(A, Xc.T @ (Y - Y.mean(0)))  # (D_PC, d_y)
    Y_hat = Xc @ B + Y.mean(0)
    ss_res = float(((Y - Y_hat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot
    return R2


# ---------------------- Recipe B helpers ----------------------------------
def fit_linear_pca(Xc, d):
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :d] * S[:d]), Vt[:d]  # (N, d), (d, D)


def _alt_circle(Xc, n_iter=120, ridge=1e-3, theta_init=None, rng=None):
    """Fit Xc ~ [cos t, sin t] @ W with alternating updates. Returns (theta, W, loss)."""
    n, D = Xc.shape
    if theta_init is None:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        plane = Xc @ Vt[:2].T
        theta = np.arctan2(plane[:, 1], plane[:, 0])
    else:
        theta = theta_init.copy()
    loss = np.inf
    W = None
    for _ in range(n_iter):
        Phi = np.column_stack([np.cos(theta), np.sin(theta)])
        A = Phi.T @ Phi + ridge * np.eye(2)
        W = np.linalg.solve(A, Phi.T @ Xc)        # (2, D)
        proj = Xc @ W.T
        theta_new = np.arctan2(proj[:, 1], proj[:, 0])
        new_loss = float(((Xc - Phi @ W) ** 2).sum())
        if abs(loss - new_loss) < 1e-8:
            theta = theta_new
            loss = new_loss
            break
        theta = theta_new
        loss = new_loss
    return theta, W, loss


def fit_circle(Xc, n_restarts=8, rng=None):
    """Multi-restart alt fit on circle. Returns best (theta, recon_loss)."""
    rng = rng or np.random.default_rng(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    inits = []
    for i in range(min(4, Vt.shape[0])):
        for j in range(i + 1, min(5, Vt.shape[0])):
            plane = Xc @ Vt[[i, j]].T
            inits.append(np.arctan2(plane[:, 1], plane[:, 0]))
    for _ in range(n_restarts):
        inits.append(rng.uniform(-np.pi, np.pi, size=Xc.shape[0]))
    best = None
    for th0 in inits:
        th, W, loss = _alt_circle(Xc, theta_init=th0)
        if best is None or loss < best[2]:
            best = (th, W, loss)
    return best


def fit_sphere(Xc, n_iter=100, n_restarts=6, ridge=1e-3, rng=None):
    """Fit Xc ~ S @ W where S in S^2 (each row unit-norm in R^3).

    Alternate: given W, S_new = (Xc W^T) projected to unit sphere; given S,
    W = ridge LS of Xc on S. Multi-restart from random unit S's + top-3 PC.
    """
    rng = rng or np.random.default_rng(0)
    n, D = Xc.shape
    inits = []
    U, S0, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :3] * S0[:3]
    inits.append(P / np.linalg.norm(P, axis=1, keepdims=True))
    for _ in range(n_restarts):
        g = rng.normal(size=(n, 3))
        inits.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    best = None
    for S_init in inits:
        S = S_init.copy()
        loss = np.inf
        W = None
        for _ in range(n_iter):
            A = S.T @ S + ridge * np.eye(3)
            W = np.linalg.solve(A, S.T @ Xc)         # (3, D)
            S_raw = Xc @ W.T                          # (n, 3)
            nrm = np.linalg.norm(S_raw, axis=1, keepdims=True)
            nrm = np.where(nrm < 1e-10, 1.0, nrm)
            S_new = S_raw / nrm
            new_loss = float(((Xc - S_new @ W) ** 2).sum())
            if abs(loss - new_loss) < 1e-7:
                S = S_new
                loss = new_loss
                break
            S = S_new
            loss = new_loss
        if best is None or loss < best[2]:
            best = (S, W, loss)
    return best


def fit_torus(Xc, n_iter=150, n_restarts=6, ridge=1e-3, rng=None):
    """Fit Xc ~ [cos t1, sin t1, cos t2, sin t2] @ W, alt updates.

    Each circle factor updated by atan2 of its own 2-block residual projection.
    """
    rng = rng or np.random.default_rng(0)
    n, D = Xc.shape
    inits = []
    U, S0, Vt = np.linalg.svd(Xc, full_matrices=False)
    p12 = Xc @ Vt[[0, 1]].T
    p34 = Xc @ Vt[[2, 3]].T if Vt.shape[0] >= 4 else Xc @ Vt[[0, 1]].T
    inits.append((np.arctan2(p12[:, 1], p12[:, 0]),
                  np.arctan2(p34[:, 1], p34[:, 0])))
    for _ in range(n_restarts):
        inits.append((rng.uniform(-np.pi, np.pi, n),
                      rng.uniform(-np.pi, np.pi, n)))
    best = None
    for t1_0, t2_0 in inits:
        t1, t2 = t1_0.copy(), t2_0.copy()
        loss = np.inf
        W = None
        for _ in range(n_iter):
            Phi = np.column_stack([np.cos(t1), np.sin(t1),
                                   np.cos(t2), np.sin(t2)])
            A = Phi.T @ Phi + ridge * np.eye(4)
            W = np.linalg.solve(A, Phi.T @ Xc)        # (4, D)
            # update t1 fixing t2 contribution
            resid1 = Xc - Phi[:, 2:] @ W[2:]
            P1 = resid1 @ W[:2].T                     # (n, 2) — note: not exact normal eqs
            # Better: project residual onto the *unit* tangent basis for circle 1
            # using ridge:
            # Solve [cos t1, sin t1] s = resid1 @ W[:2].T / ||W[:2]||^2 -- approximate
            t1_new = np.arctan2(P1[:, 1], P1[:, 0])
            resid2 = Xc - np.column_stack([np.cos(t1_new), np.sin(t1_new)]) @ W[:2]
            P2 = resid2 @ W[2:].T
            t2_new = np.arctan2(P2[:, 1], P2[:, 0])
            Phi_new = np.column_stack([np.cos(t1_new), np.sin(t1_new),
                                       np.cos(t2_new), np.sin(t2_new)])
            new_loss = float(((Xc - Phi_new @ W) ** 2).sum())
            if abs(loss - new_loss) < 1e-7:
                t1, t2 = t1_new, t2_new
                loss = new_loss
                break
            t1, t2 = t1_new, t2_new
            loss = new_loss
        if best is None or loss < best[2]:
            best = ((t1, t2), W, loss)
    return best


# ---------------------- Recipe B scoring ----------------------------------
def circ_corr(a, b):
    """Jammalamadaka circular correlation, in [-1, 1]."""
    a_bar = np.angle(np.mean(np.exp(1j * a)))
    b_bar = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - a_bar) * np.sin(b - b_bar))
    den = np.sqrt(np.sum(np.sin(a - a_bar) ** 2)
                  * np.sum(np.sin(b - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def recipe_B_unsupervised(X, latents, topo, rng):
    """Fit matching manifold, score against truth. Returns (score_name, score)."""
    Xc = X - X.mean(0)
    if topo == "linear":
        T, V = fit_linear_pca(Xc, 3)
        # Procrustes-style R^2 on Z
        Z = latents["Z"]
        # ridge LS T -> Z
        A = T.T @ T + 1e-6 * np.eye(3)
        B = np.linalg.solve(A, T.T @ (Z - Z.mean(0)))
        Zhat = T @ B + Z.mean(0)
        r2 = 1.0 - float(((Z - Zhat) ** 2).sum()) / float(((Z - Z.mean(0)) ** 2).sum())
        return ("R^2(Z_hat vs Z)", r2)
    if topo == "circle":
        theta, W, _ = fit_circle(Xc, rng=rng)
        cc = circ_corr(theta, latents["theta"])
        return ("circ_corr(theta_hat, theta)", abs(cc))
    if topo == "sphere":
        S_hat, W, _ = fit_sphere(Xc, rng=rng)
        # Procrustes alignment of S_hat to S
        S = latents["S"]
        # best 3x3 rotation R s.t. S_hat @ R ~ S
        M = S_hat.T @ S
        U_, _, Vt_ = np.linalg.svd(M)
        R = U_ @ Vt_
        # allow reflection
        S_aligned = S_hat @ R
        # spherical R^2 = 1 - mean(1 - cos) / 1
        cos_align = np.sum(S * S_aligned, axis=1)
        r2_sph = float(cos_align.mean())   # in [-1, 1]; 1 = perfect
        return ("mean_cos(S_aligned, S)", r2_sph)
    if topo == "torus":
        (t1, t2), W, _ = fit_torus(Xc, rng=rng)
        # try both pairings + signs (4 perms)
        truth1, truth2 = latents["th1"], latents["th2"]
        best = -1.0
        for p in [(t1, t2), (t2, t1)]:
            for s1 in (+1, -1):
                for s2 in (+1, -1):
                    c1 = abs(circ_corr(s1 * p[0], truth1))
                    c2 = abs(circ_corr(s2 * p[1], truth2))
                    best = max(best, 0.5 * (c1 + c2))
        return ("mean(|circ_corr|) over (th1,th2)", best)


# ---------------------- Recipe C: emulated TopologyAutoSelector -----------
def recon_loss_for_topology(X, topo, rng):
    """Return (loss, n_params_eff) for the BEST fit of `topo` on X."""
    Xc = X - X.mean(0)
    n, D = Xc.shape
    if topo == "linear":
        T, V = fit_linear_pca(Xc, 3)
        recon = T @ V
        loss = float(((Xc - recon) ** 2).sum())
        # params: 3*D (basis) + 3*n (latents) (free continuous)
        kparam = 3 * D + 3
        return loss, kparam
    if topo == "circle":
        theta, W, loss = fit_circle(Xc, rng=rng)
        # 2*D (basis) + 1 latent per row (angle), but model dim ~ 2D+2
        kparam = 2 * D + 2
        return loss, kparam
    if topo == "sphere":
        S_hat, W, loss = fit_sphere(Xc, rng=rng)
        kparam = 3 * D + 3      # 3D basis + 2 intrinsic dims (+1 for constraint)
        return loss, kparam
    if topo == "torus":
        (t1, t2), W, loss = fit_torus(Xc, rng=rng)
        kparam = 4 * D + 4
        return loss, kparam


def topology_auto_select(X, rng):
    """Emulated TopologyAutoSelector: pick topo with lowest TK-normalized BIC.

    BIC = n*D*log(loss/(n*D)) + k * log(n*D)
    (Tierney-Kadane normalization: divide loss by (n*D) before log so different
    fit dimensions are comparable.)
    """
    n, D = X.shape
    scores = {}
    losses = {}
    for topo in TOPOS:
        loss, k = recon_loss_for_topology(X, topo, rng)
        bic = n * D * np.log(max(loss, 1e-12) / (n * D)) + k * np.log(n * D)
        scores[topo] = bic
        losses[topo] = loss
    pick = min(scores, key=scores.get)
    return pick, scores, losses


# ---------------------- main ----------------------------------------------
def main():
    t0 = time.time()
    ver, caps = detect_gamfit()
    has_topo_native = bool(caps.get("topo", False))
    gamfit_path = ("gamfit_native_TopologyAutoSelector"
                   if has_topo_native else "python_emulator")
    print(f"[gamfit] version={ver}  caps={caps}")
    print(f"[gamfit] path = {gamfit_path}")

    rng = np.random.default_rng(SEED)
    cells = {}  # (topo, recipe) -> score
    selector_picks = {}
    selector_bics = {}
    recipe_score_names = {}

    for topo in TOPOS:
        X, latents = make_dataset(topo, seed=SEED + hash(topo) % 1000)
        print(f"\n[topo={topo}] X={X.shape}")
        # A
        r2_A = recipe_A_supervised(X, latents, topo)
        cells[(topo, "A_supervised")] = r2_A
        print(f"  A supervised R^2 = {r2_A:.4f}")
        # B
        name, score_B = recipe_B_unsupervised(X, latents, topo, rng)
        cells[(topo, "B_unsupervised")] = score_B
        recipe_score_names[(topo, "B_unsupervised")] = name
        print(f"  B unsupervised {name} = {score_B:.4f}")
        # C
        pick, bics, losses = topology_auto_select(X, rng)
        selector_picks[topo] = pick
        selector_bics[topo] = bics
        cells[(topo, "C_selector_correct")] = float(pick == topo)
        print(f"  C TopologyAutoSelector pick = '{pick}'  (true: '{topo}')  "
              f"-> {'CORRECT' if pick == topo else 'WRONG'}")
        print(f"      BICs: " + ", ".join(f"{k}={v:.0f}" for k, v in bics.items()))

    n_correct = sum(1 for t in TOPOS if selector_picks[t] == t)
    print(f"\n[selector] {n_correct}/{len(TOPOS)} correct picks")

    # ----------------- plot -----------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax = axs[0]
    # heatmap-style table
    grid = np.zeros((len(TOPOS), 3))
    for i, t in enumerate(TOPOS):
        grid[i, 0] = cells[(t, "A_supervised")]
        grid[i, 1] = cells[(t, "B_unsupervised")]
        grid[i, 2] = cells[(t, "C_selector_correct")]
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A: supervised\nR^2",
                        "B: unsupervised\nscore",
                        "C: selector\ncorrect?"])
    ax.set_yticks(range(len(TOPOS)))
    ax.set_yticklabels(TOPOS)
    for i in range(len(TOPOS)):
        for j in range(3):
            v = grid[i, j]
            txt = f"{v:.3f}" if j < 2 else ("OK" if v > 0.5 else "FAIL")
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontsize=11, fontweight="bold")
    ax.set_title(f"auto_exp_55 scoring matrix (4 topos x 3 recipes)\n"
                 f"gamfit path = {gamfit_path}")
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axs[1]
    # BIC matrix: rows = true topo, cols = candidate
    bic_grid = np.zeros((len(TOPOS), len(TOPOS)))
    for i, t in enumerate(TOPOS):
        for j, c in enumerate(TOPOS):
            bic_grid[i, j] = selector_bics[t][c]
    # normalize per row (delta from min)
    bic_norm = bic_grid - bic_grid.min(axis=1, keepdims=True)
    im2 = ax.imshow(bic_norm, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(TOPOS))); ax.set_xticklabels(TOPOS)
    ax.set_yticks(range(len(TOPOS))); ax.set_yticklabels(TOPOS)
    ax.set_xlabel("candidate topology")
    ax.set_ylabel("true topology")
    ax.set_title("TK-normalized BIC (per-row Delta from best; 0=picked)")
    for i in range(len(TOPOS)):
        for j in range(len(TOPOS)):
            star = " *" if selector_picks[TOPOS[i]] == TOPOS[j] else ""
            ax.text(j, i, f"{bic_norm[i, j]:.0f}{star}", ha="center",
                    va="center", color="white", fontsize=9)
    fig.colorbar(im2, ax=ax, fraction=0.045)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    np.savez(
        OUT_NPZ,
        topologies=np.array(TOPOS),
        score_matrix=grid,
        bic_grid=bic_grid,
        selector_picks=np.array([selector_picks[t] for t in TOPOS]),
        n_selector_correct=n_correct,
        gamfit_version=ver,
        gamfit_path=gamfit_path,
        recipe_B_score_names=np.array(
            [recipe_score_names[(t, "B_unsupervised")] for t in TOPOS]),
        N_rows=N_ROWS, D_pc=D_PC, noise_sigma=NOISE_SIGMA, seed=SEED,
    )
    print(f"[npz] saved {OUT_NPZ}")
    print(f"[runtime] {time.time() - t0:.1f}s")

    # print final matrix
    print("\n========== FINAL 4 x 3 SCORING MATRIX ==========")
    print(f"{'topo':<10s} | {'A sup R^2':>10s} | {'B unsup':>10s} | {'C correct?':>10s}")
    print("-" * 52)
    for t in TOPOS:
        a = cells[(t, 'A_supervised')]
        b = cells[(t, 'B_unsupervised')]
        c = "OK" if cells[(t, 'C_selector_correct')] > 0.5 else "FAIL"
        print(f"{t:<10s} | {a:>10.4f} | {b:>10.4f} | {c:>10s}")
    print(f"\nSelector accuracy: {n_correct}/{len(TOPOS)}")


if __name__ == "__main__":
    main()
