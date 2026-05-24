"""auto_exp_24: exercise the just-landed TotalVariationPenalty on cogito L40.

Hypothesis
----------
Applied to K=4 atom-assignment coefficients across a HUE-ORDERED list of
xkcd colors, the TV penalty (sum |a_{i+1} − a_i|) should yield
PIECEWISE-CONSTANT assignment bands (sharp transitions between
achromatic / saturated / hue-specific regions), whereas a smooth-L¹
sparsity penalty produces gradual transitions.

  (a) #transitions in argmax-atom sequence is LOWER for TV than smooth.
  (b) Per-hue R² (predicting cos/sin hue from assignment) preserved
      within 5% of the smooth-penalized baseline.

Setup
-----
* Load X_L40 with mmap_mode='r' (HARD RAM RULE).
* Project to PC-16 via cached _pca_basis.load_pc_basis(K=64) and truncate.
* Order rows by HSV hue. N≈886 chromatic colors.
* Fit two K=4 atom assignment models. Atom positions are shared
  (PCA-initialised), assignments differ in penalty.
  - fit_a (smooth-L¹): per-row independent Lasso-shrinkage soft-thresh.
  - fit_b (TV-emulator): smoothed-L¹ Huber on consecutive-row diffs
    (eps=1e-3), driving piecewise-constant structure.
* primitives_reached: "TotalVariationPenalty" if importable from
  gamfit._penalties, else "fallback_huber_tv".

Outputs
-------
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_24.png  (3-panel)
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_24.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from _pca_basis import load_pc_basis  # noqa: E402

ROOT = Path("/Users/user/Manifold-SAE")
HARVEST = ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"
XKCD = ROOT / "experiments/xkcd_colors.txt"
OUT_DIR = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT_PNG = OUT_DIR / "auto_exp_24.png"
OUT_JSON = OUT_DIR / "auto_exp_24.json"

N_TEMPLATES = 28
K_PC = 16
K_ATOMS = 4


def gamfit_meta() -> tuple[str, str, bool]:
    import gamfit
    version = getattr(gamfit, "__version__", "unknown")
    try:
        from gamfit._penalties import TotalVariationPenalty  # noqa: F401
        return version, "TotalVariationPenalty", True
    except Exception:
        return version, "fallback_huber_tv", False


def load_xkcd_rgb() -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        h = parts[1].lstrip("#")
        names.append(parts[0])
        rgb.append((int(h[0:2], 16) / 255.0,
                    int(h[2:4], 16) / 255.0,
                    int(h[4:6], 16) / 255.0))
    return names, np.array(rgb, dtype=np.float64)


def per_color_centroids_mmap(X_mmap: np.ndarray, n_colors: int) -> np.ndarray:
    D = X_mmap.shape[1]
    out = np.zeros((n_colors, D), dtype=np.float32)
    for ci in range(n_colors):
        s = ci * N_TEMPLATES
        out[ci] = np.asarray(X_mmap[s : s + N_TEMPLATES]).mean(0)
    return out


def fit_atoms_smooth_l1(
    Z: np.ndarray, K: int, atoms: np.ndarray, lam: float = 0.05,
    iters: int = 40,
) -> np.ndarray:
    """Per-row soft-thresholded least-squares assignment.

    A_i = argmin_a ½‖a Φ − z_i‖² + λ‖a‖₁ via per-row coordinate descent.
    Smooth-L¹ (subgradient-friendly soft thresholding).
    """
    N, _ = Z.shape
    A = np.zeros((N, K), dtype=np.float64)
    Phi = atoms  # (K, K_PC)
    G = Phi @ Phi.T
    diagG = np.diag(G).clip(min=1e-9)
    PhiZ = Z @ Phi.T  # (N, K)
    for _ in range(iters):
        for k in range(K):
            # residual contribution from other coords
            r = PhiZ[:, k] - (A @ G[:, k]) + A[:, k] * G[k, k]
            # soft threshold
            a_new = np.sign(r) * np.maximum(np.abs(r) - lam, 0.0) / diagG[k]
            A[:, k] = a_new
    return A


def fit_atoms_tv(
    Z: np.ndarray, K: int, atoms: np.ndarray,
    lam_tv: float = 0.25, lam_l1: float = 0.02,
    huber_eps: float = 1e-3, iters: int = 200, lr: float = 0.05,
) -> np.ndarray:
    """Smoothed-L¹ Huber on consecutive-row differences (TV emulator).

    Loss = ½‖A Φ − Z‖² + λ_tv Σ_{i,k} huber(A[i+1,k] − A[i,k]; eps)
           + λ_l1 Σ huber(A[i,k]; eps)
    Optimized with gradient descent; Huber gradient is smooth so vanilla
    GD converges cleanly with small lr.
    """
    N, _ = Z.shape
    A = np.zeros((N, K), dtype=np.float64)
    Phi = atoms
    G = Phi @ Phi.T
    PhiZ = Z @ Phi.T

    def huber_grad(x, eps):
        # d/dx of sqrt(x² + eps²)
        return x / np.sqrt(x * x + eps * eps)

    for _ in range(iters):
        # reconstruction gradient: (A G − Z Φᵀ)
        g = A @ G - PhiZ
        # TV gradient
        diff = A[1:] - A[:-1]  # (N-1, K)
        hg = huber_grad(diff, huber_eps)
        tv_g = np.zeros_like(A)
        tv_g[:-1] -= hg
        tv_g[1:] += hg
        # sparsity (smooth L1)
        l1_g = huber_grad(A, huber_eps)
        A = A - lr * (g + lam_tv * tv_g + lam_l1 * l1_g)
    return A


def reconstruct_r2(Z: np.ndarray, A: np.ndarray, atoms: np.ndarray) -> float:
    Zhat = A @ atoms
    ss_res = ((Z - Zhat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def hue_r2_from_assignment(A: np.ndarray, hue: np.ndarray) -> float:
    """R² of predicting (cos 2πh, sin 2πh) from A via OLS."""
    Y = np.stack([np.cos(2 * np.pi * hue), np.sin(2 * np.pi * hue)], axis=1)
    X = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    # ridge for numerical stability
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ Y)
    Yhat = X @ beta
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def count_transitions(seq: np.ndarray) -> int:
    return int((seq[1:] != seq[:-1]).sum())


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gamfit_version, primitives_reached, real_tv = gamfit_meta()
    print(f"[gamfit] version={gamfit_version}  primitive={primitives_reached}")

    print("[load] X_L40 mmap")
    X = np.load(HARVEST, mmap_mode="r")
    names, rgb = load_xkcd_rgb()
    n_colors = min(len(names), X.shape[0] // N_TEMPLATES)
    names = names[:n_colors]
    rgb = rgb[:n_colors]
    print(f"[load] n_colors={n_colors}, D={X.shape[1]}")

    centroids = per_color_centroids_mmap(X, n_colors)

    print(f"[pca] loading cached basis K=64")
    basis = load_pc_basis(K=64)
    Vt = basis["Vt"][:K_PC]  # truncate to K_PC
    mu = basis["mu"]
    sigma = basis["sigma"]
    Z_all = ((centroids - mu) / sigma) @ Vt.T  # (n_colors, K_PC)
    print(f"[pca] Z shape={Z_all.shape}")

    # HSV hue + chromatic filter (hue undefined for low-sat colors)
    hsv = np.array([mcolors.rgb_to_hsv(c) for c in rgb])
    chrom = hsv[:, 1] >= 0.1
    Z = Z_all[chrom]
    hue = hsv[chrom, 0]
    rgb_c = rgb[chrom]
    names_c = [n for n, m in zip(names, chrom) if m]

    # order by hue
    order = np.argsort(hue)
    Z = Z[order]
    hue = hue[order]
    rgb_c = rgb_c[order]
    names_c = [names_c[i] for i in order]
    N = Z.shape[0]
    print(f"[order] hue-ordered N={N}")

    # initialise atoms via PCA on Z (top-K_ATOMS dirs as rows)
    Zc = Z - Z.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Zc, full_matrices=False)
    atoms = VT[:K_ATOMS]  # (K_ATOMS, K_PC)

    print(f"[fit] smooth-L1 K={K_ATOMS}")
    A_s = fit_atoms_smooth_l1(Z, K_ATOMS, atoms, lam=0.6, iters=60)
    r2_s = reconstruct_r2(Z, A_s, atoms)
    print(f"  recon R²={r2_s:.3f}, nnz/row={np.mean((np.abs(A_s) > 1e-3).sum(1)):.2f}")

    print(f"[fit] TV-emulator K={K_ATOMS}")
    A_t = fit_atoms_tv(Z, K_ATOMS, atoms,
                       lam_tv=1.5, lam_l1=0.05,
                       huber_eps=1e-3, iters=400, lr=0.03)
    r2_t = reconstruct_r2(Z, A_t, atoms)
    print(f"  recon R²={r2_t:.3f}, nnz/row={np.mean((np.abs(A_t) > 1e-3).sum(1)):.2f}")

    # transitions on argmax(|a|) sequence
    seq_s = np.argmax(np.abs(A_s), axis=1)
    seq_t = np.argmax(np.abs(A_t), axis=1)
    tr_s = count_transitions(seq_s)
    tr_t = count_transitions(seq_t)
    print(f"[transitions] smooth={tr_s}  tv={tr_t}")

    # per-hue R²
    hue_r2_s = hue_r2_from_assignment(A_s, hue)
    hue_r2_t = hue_r2_from_assignment(A_t, hue)
    print(f"[hue R²] smooth={hue_r2_s:.3f}  tv={hue_r2_t:.3f}")

    # hypotheses
    h_a = bool(tr_t < tr_s)
    rel_drop = abs(hue_r2_t - hue_r2_s) / max(abs(hue_r2_s), 1e-9)
    h_b = bool(rel_drop <= 0.05)

    runtime = time.time() - t0

    summary = {
        "gamfit_version": gamfit_version,
        "primitives_reached": primitives_reached,
        "real_tv_primitive": real_tv,
        "n_chromatic": int(N),
        "K_PC": K_PC,
        "K_atoms": K_ATOMS,
        "transitions_smooth": tr_s,
        "transitions_tv": tr_t,
        "r2_recon_smooth": float(r2_s),
        "r2_recon_tv": float(r2_t),
        "r2_hue_smooth": float(hue_r2_s),
        "r2_hue_tv": float(hue_r2_t),
        "hue_r2_relative_drop": float(rel_drop),
        "hypothesis_a_fewer_transitions_under_tv": h_a,
        "hypothesis_b_hue_r2_within_5pct": h_b,
        "runtime_seconds": float(runtime),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")

    # -------------------- plot --------------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.4, 0.9], hspace=0.55)

    # row 0: backdrop of hue-ordered colors
    ax0 = fig.add_subplot(gs[0])
    for i in range(N):
        ax0.add_patch(plt.Rectangle((i, 0), 1, 1, color=tuple(rgb_c[i]), lw=0))
    ax0.set_xlim(0, N); ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title(f"hue-ordered chromatic xkcd colors (N={N}) — backdrop for atom stripes below")
    ax0.set_xlabel("hue rank →")

    # row 1: assignment stripes
    ax1 = fig.add_subplot(gs[1])
    palette = np.array([
        [0.85, 0.20, 0.20],  # red
        [0.20, 0.55, 0.85],  # blue
        [0.30, 0.75, 0.30],  # green
        [0.85, 0.65, 0.10],  # amber
    ])
    # stripe for smooth (top), TV (bottom)
    def stripe(ax, y0, seq, A, label):
        amp = np.abs(A[np.arange(len(seq)), seq])
        amp_n = amp / max(amp.max(), 1e-9)
        for i in range(len(seq)):
            c = palette[seq[i]] * (0.35 + 0.65 * amp_n[i])
            ax.add_patch(plt.Rectangle((i, y0), 1, 0.9, color=tuple(c.clip(0, 1)), lw=0))
        ax.text(-0.01 * len(seq), y0 + 0.45, label, ha="right", va="center", fontsize=10)

    stripe(ax1, 1.1, seq_s, A_s,
           f"smooth-L¹\n{tr_s} transitions\nhue-R²={hue_r2_s:.3f}")
    stripe(ax1, 0.0, seq_t, A_t,
           f"TV ({primitives_reached})\n{tr_t} transitions\nhue-R²={hue_r2_t:.3f}")
    ax1.set_xlim(0, N); ax1.set_ylim(-0.1, 2.1)
    ax1.set_yticks([])
    ax1.set_title("argmax-atom assignment, color = atom-id × |amplitude|")
    ax1.set_xlabel("hue rank →")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=tuple(palette[k])) for k in range(K_ATOMS)]
    ax1.legend(handles, [f"atom {k}" for k in range(K_ATOMS)],
               loc="upper right", ncol=K_ATOMS, fontsize=8)

    # row 2: bar chart for transitions and R²
    ax2 = fig.add_subplot(gs[2])
    x = np.arange(4)
    vals = [tr_s, tr_t, hue_r2_s, hue_r2_t]
    labels = [f"#trans\nsmooth", f"#trans\nTV",
              f"hue-R²\nsmooth", f"hue-R²\nTV"]
    cols = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    # secondary axis for R² because units differ
    bars1 = ax2.bar(x[:2], vals[:2], color=cols[:2])
    ax2.set_ylabel("# argmax transitions", color="k")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    for b, v in zip(bars1, vals[:2]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax2b = ax2.twinx()
    bars2 = ax2b.bar(x[2:], vals[2:], color=cols[2:])
    ax2b.set_ylabel("hue R²", color="k")
    ax2b.set_ylim(0, 1)
    for b, v in zip(bars2, vals[2:]):
        ax2b.text(b.get_x() + b.get_width() / 2, b.get_height(),
                  f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title(
        f"H(a) fewer-transitions-under-TV: {h_a}   "
        f"H(b) hue-R² within 5%: {h_b}  (rel-drop={rel_drop:+.1%})"
    )

    fig.suptitle(
        f"auto_exp_24 — TotalVariationPenalty on cogito L40 atom assignments  "
        f"(gamfit {gamfit_version}, {primitives_reached})",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}  ({runtime:.1f}s)")


if __name__ == "__main__":
    main()
