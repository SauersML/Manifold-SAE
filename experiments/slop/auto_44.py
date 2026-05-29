"""
auto_44 — Idea (rrrrr): U_3d latent trajectory for "red" across the 28 templates.

Question: U_3d in results.json is fit on per-color centroids (one latent T per
color, collapsing across templates). Is the per-template residual for "red"
all clustered tightly at red's centroid latent, or does each template push the
representation along the manifold (e.g., toward white for "{x} milk", toward
black for "{x} ink")?

Method (NO Gaussian RBF, NO length_scale on Duchon):
  1. Load harvest X_L40 (n_c=949 × n_t=28 rows, color-major).
  2. Build per-color centroid → PCA Z exactly like color_manifold_gam.py:
     subtract cached mu, divide by cached sigma, project onto Vt_topK (K=64).
  3. Apply the SAME (mu, sigma, Vt) projection to every individual
     (color, template) row → Z_full (n_c*n_t, K).
  4. Fit a ridge (linear) map from per-color Z (centroid PCs) → U_3d using
     the U_3d latents already stored under unsupervised_full_data["d=3"]["T"].
     This is a smooth, parameter-free interpolator (no Gaussian RBF, no
     length_scale — pure linear ridge in the 64-D PC basis).
  5. Apply that ridge to each of the 28 per-template Z rows for the "red"
     color (xkcd index 943) → 28 estimated U_3d points.
  6. For context, also project a handful of anchor colors' centroid latents:
     red, white, black, blue, green, pink, grey, orange (the verbal anchors
     used in the probes file).

Plot: a single 3-panel figure (3 axis pairs of U_3d) showing
  • all 949 centroid U_3d points in light grey
  • the 28 per-template "red" estimates connected by template index
  • named anchor centroids labeled

If the 28 dots cluster tightly → templates don't move "red" along the
manifold. If they spread (esp. toward white/black anchors for templates
like "{x} milk" / "{x} coal"), templates carve identifiable axes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import color_manifold_gam as cmg  # noqa: E402

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_44.png"

TARGET = "red"
ANCHORS = ["red", "white", "black", "blue", "green", "pink", "grey", "orange",
           "yellow", "purple"]


def ridge_fit(Z: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: W = (Z^T Z + lam I)^-1 Z^T T."""
    K = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(K)
    return np.linalg.solve(A, Z.T @ T)


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)   # (K, D)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)        # (D,)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)  # (D,)
    U3 = np.asarray(d["per_layer"]["L40"]["unsupervised_full_data"]["d=3"]["T"],
                    dtype=np.float64)  # (n_c, 3)
    K = Vt.shape[0]

    X_full = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X_full.shape
    n_c = n_rows // n_t
    assert n_c * n_t == n_rows
    assert U3.shape[0] == n_c
    print(f"[meta] n_c={n_c}, n_t={n_t}, K={K}, D={D}")

    # Per-color centroid -> PC space (same as cmg).
    per_color = np.zeros((n_c, D), dtype=np.float64)
    block = 2048
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = (np.arange(s, e) // n_t)
        for ci in np.unique(idx):
            m = idx == ci
            per_color[ci] += chunk[m].sum(axis=0)
    per_color /= n_t
    Xn_c = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z_c = (Xn_c - Xn_c.mean(0, keepdims=True)) @ Vt.T  # (n_c, K)

    # Ridge: Z_c -> U3.  CV-free; pick lam via simple GCV-style trace ratio.
    # Try a small grid and pick min train+lam penalty heuristic; in practice
    # the manifold is smooth so a moderate lam is fine.
    lams = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    best = (np.inf, None, None)
    for lam in lams:
        W = ridge_fit(Z_c, U3, lam)
        pred = Z_c @ W
        mse = float(np.mean((pred - U3) ** 2))
        # 5-fold leave-out-color quick estimate
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_c)
        folds = np.array_split(perm, 5)
        cv_err = 0.0
        for k in range(5):
            te = folds[k]; tr = np.setdiff1d(perm, te)
            Wk = ridge_fit(Z_c[tr], U3[tr], lam)
            cv_err += float(np.mean((Z_c[te] @ Wk - U3[te]) ** 2))
        cv_err /= 5
        print(f"  lam={lam:6.2g}  train_mse={mse:.4e}  cv_mse={cv_err:.4e}")
        if cv_err < best[0]:
            best = (cv_err, lam, W)
    cv_err, lam_star, W = best
    print(f"[ridge] picked lam={lam_star}, cv_mse={cv_err:.4e}")

    # Resolve target color row(s).
    colors = cmg.load_xkcd_colors()[:n_c]
    name_to_idx = {c[0]: i for i, c in enumerate(colors)}
    if TARGET not in name_to_idx:
        raise RuntimeError(f"{TARGET!r} not found")
    c_red = name_to_idx[TARGET]
    print(f"[target] {TARGET} at color index {c_red}")

    # Per-template rows for red, project to Z then to U3 via ridge.
    rows_red = X_full[c_red * n_t : (c_red + 1) * n_t]  # (28, D)
    Xn_red = (rows_red.astype(np.float64) - mu) / np.maximum(sigma, 1e-6)
    # Use SAME centering as Z_c (subtract mean of centroid Xn).
    Xn_red_centered = Xn_red - Xn_c.mean(0, keepdims=True)
    Z_red_t = Xn_red_centered @ Vt.T                    # (28, K)
    U_red_t = Z_red_t @ W                                # (28, 3)

    # Anchor centroid latents for context.
    anchor_pts = {}
    for nm in ANCHORS:
        if nm in name_to_idx:
            anchor_pts[nm] = U3[name_to_idx[nm]]

    # ------------------------------------------------------------------ plot
    rgb_red = np.array(colors[c_red][1:], dtype=float) / 255.0
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_t))

    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(U3[:, a], U3[:, b], s=3, c="lightgrey", alpha=0.4,
                   label=f"all {n_c} centroids", zorder=1)
        # red centroid (filled red star)
        cen = U3[c_red]
        ax.scatter([cen[a]], [cen[b]], marker="*", s=300, c=[rgb_red],
                   edgecolor="black", linewidth=1.2, zorder=4,
                   label=f"{TARGET} centroid")
        # anchors
        for nm, pt in anchor_pts.items():
            if nm == TARGET:
                continue
            ax.scatter([pt[a]], [pt[b]], marker="o", s=80,
                       facecolor="white", edgecolor="black", zorder=3)
            ax.annotate(nm, (pt[a], pt[b]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="black")
        # red per-template estimates, colored by template index
        for t in range(n_t):
            ax.scatter([U_red_t[t, a]], [U_red_t[t, b]],
                       s=55, c=[cmap[t]], edgecolor="red", linewidth=0.8,
                       zorder=5)
        # connect by template index (rough trajectory)
        ax.plot(U_red_t[:, a], U_red_t[:, b], color="red", alpha=0.25,
                linewidth=0.8, zorder=2)
        ax.set_xlabel(f"U_3d[{a}]")
        ax.set_ylabel(f"U_3d[{b}]")
        ax.set_title(f"U3 axes {a} vs {b}")

    axes[0].legend(loc="best", fontsize=8)

    # Add a 4th implicit panel: spread vs centroid summary as a side text.
    spreads = U_red_t - U3[c_red]
    sd_per_axis = spreads.std(axis=0)
    rms = float(np.sqrt(np.mean(spreads ** 2)))
    # also: distance ratio (red-template-spread RMS) / (typical neighbor distance)
    from scipy.spatial import cKDTree  # noqa
    tree = cKDTree(U3)
    nn_d = tree.query(U3, k=2)[0][:, 1]
    typical_nn = float(np.median(nn_d))
    ratio = rms / typical_nn if typical_nn > 0 else float("nan")

    fig.suptitle(
        f"U_3d trajectory of {TARGET!r} across 28 templates  "
        f"(per-axis σ={sd_per_axis.round(3).tolist()},  "
        f"RMS={rms:.3f},  RMS/median_NN={ratio:.2f})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140)
    print(f"[done] wrote {OUT}")

    # Console: which templates push hardest?
    dists = np.linalg.norm(spreads, axis=1)
    order = np.argsort(-dists)
    print("\n[top 6 templates moving 'red' off its centroid]")
    for i in order[:6]:
        print(f"  t={i:2d}  d={dists[i]:.3f}  {templates[i][:70]}")
    print("\n[bottom 6 — templates that stay closest]")
    for i in order[-6:]:
        print(f"  t={i:2d}  d={dists[i]:.3f}  {templates[i][:70]}")


if __name__ == "__main__":
    main()
