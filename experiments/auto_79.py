"""auto_79 — U_3d's discovered latent T colored by RGB.

Fresh angle (option d from the brief). auto_77 = per-spec PC sweep,
auto_78 = bulk-vs-tail scatter. Neither LOOKS at U_3d's latent.

U_3d is the unsupervised 3D embedding fit by the GAM zoo. Its T is
(949, 3). If cogito's L40 color manifold is "really" 3D and RGB-aligned,
the cloud should fill an RGB-cube shape. If it's hue-dominated, it
should be a torus or cone. If it's just noise around a low-d core, it
will be a pancake.

We plot:
  - 3 pairwise 2D scatters (t1×t2, t1×t3, t2×t3) colored by each color's
    true xkcd RGB
  - a Procrustes fit  T → A·RGB + b  with held-out R² (5-fold by color)
  - bar chart of |Spearman ρ| between each of {R,G,B,H,S,V,luminance}
    and each of {t1,t2,t3}

Outputs:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_79.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_79.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS = Path(
    "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
)
OUT_PNG = RESULTS.parent / "auto_79.png"
OUT_JSON = RESULTS.parent / "auto_79.json"


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold = np.empty(n, dtype=int)
    fold[perm] = np.arange(n) % k
    return [(np.where(fold != i)[0], np.where(fold == i)[0]) for i in range(k)]


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ridge_fit(Phi, Y, alpha=1e-3):
    A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ Y)


def main() -> int:
    r = json.load(open(RESULTS))
    pl = r["per_layer"]["L40"]
    T = np.array(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    n = T.shape[0]
    axes = r["color_axes_per_color_index"]
    R = np.array(axes["R"][:n], dtype=np.float64)
    G = np.array(axes["G"][:n], dtype=np.float64)
    B = np.array(axes["B"][:n], dtype=np.float64)
    H = np.array(axes["hue"][:n], dtype=np.float64)
    S = np.array(axes["sat"][:n], dtype=np.float64)
    V = np.array(axes["value"][:n], dtype=np.float64)
    L = np.array(axes["luminance"][:n], dtype=np.float64)
    print(f"[auto_79] T shape {T.shape}  n_colors used = {n}")

    rgb_facecolor = np.clip(np.stack([R, G, B], axis=1), 0, 1)

    # Procrustes / linear map  RGB -> T  with held-out R² (predict T from RGB)
    # i.e. how much of T's geometry is a linear function of RGB?
    Phi_rgb = np.concatenate([np.stack([R, G, B], axis=1),
                              np.ones((n, 1))], axis=1)
    folds = kfold_indices(n, 5)
    r2_folds = []
    pred_full = np.zeros_like(T)
    for tr, te in folds:
        W = ridge_fit(Phi_rgb[tr], T[tr], alpha=1e-3)
        pred = Phi_rgb[te] @ W
        pred_full[te] = pred
        r2_folds.append(r2_macro(T[te], pred))
    r2_rgb_to_t = (float(np.mean(r2_folds)), float(np.std(r2_folds)))
    print(f"[auto_79] RGB->T held-out R² = {r2_rgb_to_t[0]:+.3f} ± {r2_rgb_to_t[1]:.3f}")

    # Also the reverse: how much of RGB is a linear function of T?
    Phi_t = np.concatenate([T, np.ones((n, 1))], axis=1)
    r2_folds_rev = []
    RGB = np.stack([R, G, B], axis=1)
    for tr, te in folds:
        W = ridge_fit(Phi_t[tr], RGB[tr], alpha=1e-3)
        r2_folds_rev.append(r2_macro(RGB[te], Phi_t[te] @ W))
    r2_t_to_rgb = (float(np.mean(r2_folds_rev)), float(np.std(r2_folds_rev)))
    print(f"[auto_79] T->RGB held-out R² = {r2_t_to_rgb[0]:+.3f} ± {r2_t_to_rgb[1]:.3f}")

    # Spearman corrs (use stored values where possible)
    axis_names = ["R", "G", "B", "hue", "sat", "value", "luminance"]
    spearman_table = pl["unsupervised_full_data"]["d=3"]["axis_to_latent_spearman"]
    rho_mat = np.array([spearman_table[a]["per_latent_rho"] for a in axis_names])
    # rho_mat shape (7 axes, 3 latents)

    # ---- Plot
    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85],
                          hspace=0.35, wspace=0.25)

    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [("t1", "t2"), ("t1", "t3"), ("t2", "t3")]
    for k, ((i, j), (lx, ly)) in enumerate(zip(pairs, pair_labels)):
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(T[:, i], T[:, j], c=rgb_facecolor, s=16,
                   edgecolors="black", linewidths=0.15, alpha=0.92)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(f"U_3d latent  {lx} × {ly}  (color = true xkcd RGB)",
                     fontsize=10.5)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")

    # Bottom-left: bar chart |ρ| of each axis on each latent
    ax_bar = fig.add_subplot(gs[1, 0:2])
    width = 0.27
    xs = np.arange(len(axis_names))
    palette = ["#d62728", "#2ca02c", "#1f77b4"]   # t1=red t2=green t3=blue
    for li in range(3):
        ax_bar.bar(xs + (li - 1) * width, np.abs(rho_mat[:, li]),
                   width, color=palette[li], edgecolor="black",
                   linewidth=0.4, label=f"t{li+1}")
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(axis_names)
    ax_bar.set_ylabel("|Spearman ρ|  (latent  vs  color axis)")
    ax_bar.set_title(
        "How aligned is each U_3d latent with each named color axis?\n"
        "If t1,t2,t3 = R,G,B you'd see red→t1, green→t2, blue→t3 dominate.",
        fontsize=10.5,
    )
    ax_bar.legend(loc="upper right", frameon=True, fontsize=9)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bar.axhline(0, color="black", linewidth=0.5)

    # Bottom-right: held-out R² panel as a text block
    ax_txt = fig.add_subplot(gs[1, 2]); ax_txt.axis("off")
    best_axis_for = {f"t{i+1}":
                     axis_names[int(np.argmax(np.abs(rho_mat[:, i])))]
                     for i in range(3)}
    best_rho_for = {f"t{i+1}":
                    float(np.max(np.abs(rho_mat[:, i]))) for i in range(3)}
    msg = (
        "Held-out 5-fold linear map\n"
        f"  RGB → T : R² = {r2_rgb_to_t[0]:+.3f} ± {r2_rgb_to_t[1]:.3f}\n"
        f"  T → RGB : R² = {r2_t_to_rgb[0]:+.3f} ± {r2_t_to_rgb[1]:.3f}\n\n"
        "Best axis per latent (|ρ|):\n"
        + "\n".join(f"  {tk}: {best_axis_for[tk]:>9s}  "
                    f"|ρ|={best_rho_for[tk]:.3f}"
                    for tk in ["t1", "t2", "t3"])
        + "\n\nIf the cloud looked RGB-cube-shaped\n"
        "both R²s would be near 1.\n"
        "Low R² ⇒ U_3d found a NON-RGB geometry\n"
        "(e.g. hue ring + name-semantic axes)."
    )
    ax_txt.text(0.0, 1.0, msg, fontsize=10.5, va="top", family="monospace")

    plt.suptitle(
        "auto_79 — U_3d latent T painted with each color's true RGB\n"
        f"cogito L40 · n={n} · unsupervised d=3 fit, log_λ="
        f"{pl['unsupervised_full_data']['d=3']['log_lambda']:.2f}",
        fontsize=12, y=0.995,
    )
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT_PNG}")

    json.dump({
        "n": int(n),
        "rgb_to_T_r2_mean": r2_rgb_to_t[0],
        "rgb_to_T_r2_std": r2_rgb_to_t[1],
        "T_to_rgb_r2_mean": r2_t_to_rgb[0],
        "T_to_rgb_r2_std": r2_t_to_rgb[1],
        "axis_names": axis_names,
        "abs_spearman_per_latent": np.abs(rho_mat).tolist(),
        "best_axis_per_latent": best_axis_for,
        "best_rho_per_latent": best_rho_for,
    }, open(OUT_JSON, "w"), indent=2)
    print(f"[done] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
