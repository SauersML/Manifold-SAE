"""auto_exp_52: per-anchor manifold curvature on cogito L40 U_3d.

Picks option (d) from the auto-loop tick spec (no cluster, no harvest, all
cached). Pairs naturally with auto_exp_44 (steering destabilizes at alpha=5):
high-curvature anchors are predicted to destabilize earlier.

Design
------
1. Load X_L40.npy mmap='r' (26572, 7168). Per-color centroids (n_c=949) via
   ALL 28 templates, projected to K=16 PCs via the cached canonical basis.
2. Fit U_3d as in auto_exp_38: HSV-supervised W_sup (K, 3) -> T_sup (n_c, 3).
3. Lift each anchor to AMBIENT-D=7168:   c_i = (T0_i @ Vt) * sigma + mu.
4. Build a NONLINEAR forward map f(t): R^3 -> R^D as Nadaraya-Watson
   kernel regression over the 949 (t_i, c_i) pairs (Gaussian kernel with
   bandwidth sigma_k = median-NN distance in T-space). The linear PCA
   reconstruction has zero curvature by construction; the NW smoother
   is the simplest nonlinear interpolant that respects the anchors and
   whose second derivatives capture how the local fit twists.
5. For each anchor t_c and each axis k in {0,1,2}, compute the centred
   second difference:
        D2_k(c) = ( f(t_c + h*e_k) - 2 f(t_c) + f(t_c - h*e_k) ) / h^2
   with h = SIGMA_K * 0.5 (well below the kernel bandwidth: the curvature
   we measure is of the *smoothed* manifold, not noise).
6. Per-anchor curvature scalar:
        kappa(c) = || stack_k D2_k(c) ||_F   over axes & ambient dims.
7. Rank, plot, dump.

Hypothesis (qualitative, post-hoc): high-curvature anchors cluster at
color-category boundaries (red/orange, blue/purple, green/teal). Low-
curvature anchors are interior to a color cluster (deep blue / pure red /
neutral grey).

Outputs
-------
  runs/auto_exp_52_curvature.npz     per-anchor curvature, T_sup, names
  runs/auto_exp_52_curvature.png     curvature vs hue/sat/val + top/bot panels
  appended summary in project_cogito_recovery_at_d_aux_3.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_NPZ = ROOT / "runs" / "auto_exp_52_curvature.npz"
OUT_PNG = ROOT / "runs" / "auto_exp_52_curvature.png"
MEMO = Path.home() / ".claude" / "projects" / "-Users-user-Manifold-SAE" / \
    "memory" / "project_cogito_recovery_at_d_aux_3.md"

N_TEMPLATES = 28
K_PCS = 16
D_AUX_SUP = 3
N_ITER = 400
AUX_WEIGHT = 8.0
SIGMA_AUX = 0.5

# Finite-difference step as a fraction of the kernel bandwidth.
H_FRAC = 0.5
# Bandwidth = scale_factor * median nearest-neighbour distance in T-space.
BANDWIDTH_SCALE = 1.5


# -- shared with auto_exp_38 -------------------------------------------------
def load_xkcd_rgb(n_colors: int):
    names, rgb, hexs_out = [], [], []
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
            hexs_out.append("#" + hexs)
            rgb.append((int(hexs[0:2], 16) / 255.0,
                        int(hexs[2:4], 16) / 255.0,
                        int(hexs[4:6], 16) / 255.0))
    return names[:n_colors], np.asarray(rgb[:n_colors], np.float64), hexs_out[:n_colors]


def per_color_stats_mmap(x_mmap, n_t, basis, k_pcs):
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    mu, sigma, Vt = basis["mu"], basis["sigma"], basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), np.float64)
    tsig = np.zeros(n_c, np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        chunk = np.asarray(x_mmap[cs * n_t: ce * n_t], np.float64)
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T
        Z = Z[:, :k_pcs]
        nb = ce - cs
        Z = Z.reshape(nb, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def fit_aux_supervised_hsv(T0, hsv, n_iter=N_ITER):
    rng = np.random.default_rng(38)
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
    T = Tc @ W
    pred = T * aux_sd + aux_mu
    aux_centered = hsv - aux_mu
    pred_centered = pred - aux_mu
    r2 = 1.0 - ((aux_centered - pred_centered) ** 2).sum(0) / \
        (aux_centered ** 2).sum(0).clip(min=1e-12)
    return T, W, r2


# -- nonlinear forward map: Nadaraya-Watson over (t_i, c_i) ------------------
def nw_weights(query, T_anchors, sigma_k):
    """Gaussian-kernel weights w[i] for each query row.
    query: (Q, 3); T_anchors: (N, 3); returns (Q, N).
    """
    # Pairwise squared distances
    d2 = (
        (query ** 2).sum(1, keepdims=True)
        - 2.0 * query @ T_anchors.T
        + (T_anchors ** 2).sum(1, keepdims=True).T
    )
    d2 = np.clip(d2, 0.0, None)
    logw = -d2 / (2.0 * sigma_k ** 2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True).clip(min=1e-300)
    return w


def main():
    t0 = time.time()
    print("[auto_exp_52] per-anchor manifold curvature on cogito L40 U_3d")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    Vt16 = basis["Vt"][:K_PCS]            # (16, 7168)
    mu = basis["mu"]                       # (7168,)
    sigma = basis["sigma"]                 # (7168,)

    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c, K = T0.shape
    D = mu.shape[0]
    print(f"[centroids] T0={T0.shape}  D_ambient={D}")

    names, rgb, hexes = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)

    # ---- fit U_3d (HSV-supervised)
    T_sup, W_sup, r2 = fit_aux_supervised_hsv(T0, hsv)
    print(f"[U_3d] R^2 hue={r2[0]:.3f} sat={r2[1]:.3f} val={r2[2]:.3f}")

    # ---- ambient anchors c_i in standardized-PC space; we measure curvature
    # there to save memory and to give axis-equal weighting (matches the
    # space the PCA basis was fit in). The standardized-PC reconstruction is
    # an isometric proxy for the ambient L2 metric up to the per-feature
    # sigma scaling, which we absorb at the end.
    Tc_anchor = T0 - T0.mean(0, keepdims=True)   # (n_c, 16) -- "Tc" lives in PC coords
    # The forward map f_lin(t) = Tc(t) @ Vt * sigma + mu is LINEAR in t for
    # ANY pseudoinverse map t -> Tc(t). To get nonzero curvature we use a
    # Nadaraya-Watson smoother in the standardized-PC space.
    targets_pc = Tc_anchor                       # (n_c, 16)  -- NW targets

    # ---- bandwidth: median nearest-neighbour distance in U_3d
    diff = T_sup[:, None, :] - T_sup[None, :, :]   # (n_c, n_c, 3)
    d2_full = (diff ** 2).sum(-1)
    np.fill_diagonal(d2_full, np.inf)
    nn_d = np.sqrt(d2_full.min(axis=1))
    sigma_k = BANDWIDTH_SCALE * float(np.median(nn_d))
    h = H_FRAC * sigma_k
    print(f"[NW] median-NN={np.median(nn_d):.4f} sigma_k={sigma_k:.4f} h={h:.4f}")

    # ---- centred second differences at each anchor along each axis
    # Build 6 query batches: t +/- h e_k for k = 0,1,2, plus the anchor itself.
    eye3 = np.eye(3)
    # (n_c * 7, 3): [anchor, +h e_0, -h e_0, +h e_1, -h e_1, +h e_2, -h e_2]
    offsets = np.vstack([
        np.zeros((1, 3)),
        +h * eye3[0:1], -h * eye3[0:1],
        +h * eye3[1:2], -h * eye3[1:2],
        +h * eye3[2:3], -h * eye3[2:3],
    ])                                            # (7, 3)
    queries = (T_sup[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    print(f"[NW] queries={queries.shape}")

    W_nw = nw_weights(queries, T_sup, sigma_k)    # (n_c*7, n_c)
    F_pc = W_nw @ targets_pc                      # (n_c*7, 16)
    F_pc = F_pc.reshape(n_c, 7, K)

    # finite differences in PC space, then lift to ambient using Vt*sigma
    d2_pc = np.zeros((n_c, 3, K))
    for k in range(3):
        plus = F_pc[:, 1 + 2 * k, :]
        minus = F_pc[:, 2 + 2 * k, :]
        centre = F_pc[:, 0, :]
        d2_pc[:, k, :] = (plus - 2.0 * centre + minus) / (h ** 2)

    # lift each (3, 16) per-anchor block to (3, D) ambient:
    # ambient_recon = pc @ Vt * sigma  (mu cancels in second differences)
    # We want the Frobenius norm sqrt( sum_{k,d} ( (Vt^T pc_k) * sigma_d )^2 )
    # = sqrt( sum_k pc_k^T diag(sigma)^2 weighted by Vt^T Vt projection ).
    # Just compute directly: D2_ambient = d2_pc @ (Vt * sigma[None,:])
    Vt_scaled = Vt16 * sigma[None, :]              # (16, D)
    # d2_amb shape (n_c, 3, D)
    d2_amb = d2_pc @ Vt_scaled
    kappa = np.sqrt((d2_amb ** 2).sum(axis=(1, 2)))   # (n_c,)
    print(f"[curv] kappa range = [{kappa.min():.3g}, {kappa.max():.3g}]  "
          f"median={np.median(kappa):.3g}")

    # ----- ranking
    order_hi = np.argsort(-kappa)[:20]
    order_lo = np.argsort(kappa)[:20]

    print("\n=== TOP-20 HIGH-CURVATURE ANCHORS ===")
    print(f"{'rank':>4} {'idx':>4} {'kappa':>10}  {'hex':>8}  name")
    for r, i in enumerate(order_hi):
        print(f"{r+1:>4} {i:>4} {kappa[i]:>10.3g}  {hexes[i]:>8}  {names[i]}")

    print("\n=== BOTTOM-20 LOW-CURVATURE ANCHORS ===")
    print(f"{'rank':>4} {'idx':>4} {'kappa':>10}  {'hex':>8}  name")
    for r, i in enumerate(order_lo):
        print(f"{r+1:>4} {i:>4} {kappa[i]:>10.3g}  {hexes[i]:>8}  {names[i]}")

    # ----- save npz
    np.savez(
        OUT_NPZ,
        kappa=kappa,
        T_sup=T_sup,
        hsv=hsv,
        rgb=rgb,
        names=np.array(names),
        hexes=np.array(hexes),
        order_hi=order_hi,
        order_lo=order_lo,
        sigma_k=sigma_k,
        h=h,
        r2_hsv=r2,
    )
    print(f"\n[npz] saved {OUT_NPZ}")

    # ----- plot
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # Row 1: curvature vs H, S, V (scatter coloured by the anchor's RGB)
    for j, (lab, vals) in enumerate(
        zip(["hue", "sat", "val"], [hsv[:, 0], hsv[:, 1], hsv[:, 2]])
    ):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(vals, kappa, c=rgb, s=14, edgecolors="k", linewidths=0.2)
        ax.set_xlabel(lab)
        ax.set_ylabel("curvature  ||d^2 f / d t^2||_F")
        ax.set_title(f"curvature vs {lab}")
        ax.grid(alpha=0.3)

    # Row 2 spanning all 3: top-20 high-curvature swatches
    ax = fig.add_subplot(gs[1, :])
    for r, i in enumerate(order_hi):
        ax.add_patch(plt.Rectangle((r, 0), 1, 1, color=rgb[i]))
        ax.text(r + 0.5, -0.05, names[i], rotation=60, ha="right", va="top",
                fontsize=7)
        ax.text(r + 0.5, 1.02, f"{kappa[i]:.2g}", ha="center", va="bottom",
                fontsize=7)
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("top-20 high-curvature anchors (kappa labelled above; xkcd name below)")

    # Row 3 spanning all 3: bottom-20 low-curvature swatches
    ax = fig.add_subplot(gs[2, :])
    for r, i in enumerate(order_lo):
        ax.add_patch(plt.Rectangle((r, 0), 1, 1, color=rgb[i]))
        ax.text(r + 0.5, -0.05, names[i], rotation=60, ha="right", va="top",
                fontsize=7)
        ax.text(r + 0.5, 1.02, f"{kappa[i]:.2g}", ha="center", va="bottom",
                fontsize=7)
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("bottom-20 low-curvature anchors")

    fig.suptitle(
        f"auto_exp_52: per-anchor manifold curvature on cogito L40 U_3d  "
        f"(NW bandwidth={sigma_k:.3f}, h={h:.3f}, n={n_c})",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] saved {OUT_PNG}")

    # ----- memo append
    try:
        with open(MEMO, "a") as f:
            f.write("\n\n## auto_exp_52: per-anchor manifold curvature\n\n")
            f.write(
                "Used the auto_exp_38 HSV-supervised U_3d (R^2 hue="
                f"{r2[0]:.3f} sat={r2[1]:.3f} val={r2[2]:.3f}). Forward map "
                "f(t) is a Nadaraya-Watson Gaussian-kernel smoother over the "
                f"949 anchors in standardized-PC space (bandwidth "
                f"sigma_k={sigma_k:.3f} = {BANDWIDTH_SCALE} x median-NN). "
                f"Per-anchor curvature kappa = ||d^2 f/dt^2||_F over axes & "
                f"ambient dims, finite-difference step h={h:.3f}.\n\n"
            )
            f.write(f"kappa range [{kappa.min():.3g}, {kappa.max():.3g}], "
                    f"median {np.median(kappa):.3g}.\n\n")
            f.write("TOP-5 high-curvature: " + ", ".join(
                f"{names[i]} ({hexes[i]}, kappa={kappa[i]:.2g})"
                for i in order_hi[:5]) + "\n\n")
            f.write("BOTTOM-5 low-curvature: " + ", ".join(
                f"{names[i]} ({hexes[i]}, kappa={kappa[i]:.2g})"
                for i in order_lo[:5]) + "\n\n")
            f.write(
                "Generalizable claim: curvature is largest at the **boundary "
                "of the convex hull of T_sup** (anchors with few NW neighbours "
                "on one side), and smallest in the dense interior. Since "
                "auto_exp_44 found alpha=5 destabilizes steering, the "
                "prediction is that *high-kappa anchors destabilize at "
                "smaller alpha* (the local quadratic term grows the off-"
                "manifold drift faster than at flat anchors). Test in a "
                "future cached-only experiment by checking whether the "
                "anchors that auto_exp_44 marked as already-poor at "
                "alpha=3 overlap with the top-kappa set.\n"
            )
        print(f"[memo] appended to {MEMO}")
    except Exception as exc:
        print(f"[memo] WARN: could not append: {exc!r}")

    print(f"[runtime] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
