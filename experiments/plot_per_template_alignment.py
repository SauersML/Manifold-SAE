"""Per-template supervised alignment.

For each of the 28 templates, take ONLY that template's prompts (one per
color, 949 prompts), PCA to 64, and fit our best supervised models
(linear HSV, linear RGB, quartic-polynomial HSV, 3D Duchon Lab).
Report held-out R² per template via 5-fold CV by color.

Reveals which templates produce the cleanest color signal in cogito.
Templates that score high: cogito reads {x} as a color word. Templates
that score low: cogito reads {x} as an object or the syntax breaks down.

Outputs:
  per_template_alignment.png   — bar chart of R² per template per model
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


N_T = 28


def load_xkcd_colors():
    from plot_color_geometry import load_xkcd_colors as _f
    return _f()


def load_harvest(p: Path) -> np.ndarray:
    from plot_color_geometry import load_harvest as _f
    return _f(p)


def kfold_color_indices(n_colors: int, n_folds: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_colors)
    fold_of = np.empty(n_colors, dtype=int)
    fold_of[perm] = np.arange(n_colors) % n_folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
            for k in range(n_folds)]


def r2_macro(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ridge_fit(Phi, Y, alpha):
    PtP = Phi.T @ Phi
    A = PtP + alpha * np.eye(PtP.shape[0])
    return np.linalg.solve(A, Phi.T @ Y)


def poly_features_degree2(X):
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def poly_features_degree4(X):
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append((X[:, i] * X[:, j] * X[:, k])[:, None])
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                for l_ in range(k, d):
                    cols.append((X[:, i] * X[:, j] * X[:, k] * X[:, l_])[:, None])
    return np.concatenate(cols, axis=1)


SPECS = [
    ("Linear RGB",        "rgb_lin"),
    ("Linear HSV",        "hsv_lin"),
    ("Quartic poly HSV",  "hsv_poly4"),
]


def fit_predict(spec_id: str, rgb_tr, hsv4_tr, Z_tr, rgb_te, hsv4_te):
    if spec_id == "rgb_lin":
        Phi_tr = np.concatenate([rgb_tr, np.ones((rgb_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([rgb_te, np.ones((rgb_te.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, Z_tr, alpha=1.0)
        return Phi_te @ W
    if spec_id == "hsv_lin":
        Phi_tr = np.concatenate([hsv4_tr, np.ones((hsv4_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([hsv4_te, np.ones((hsv4_te.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, Z_tr, alpha=1.0)
        return Phi_te @ W
    if spec_id == "hsv_poly4":
        Phi_tr = poly_features_degree4(hsv4_tr)
        Phi_te = poly_features_degree4(hsv4_te)
        W = ridge_fit(Phi_tr, Z_tr, alpha=10.0)
        return Phi_te @ W
    raise ValueError(spec_id)


def main() -> int:
    cache_path = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy",
    ))
    out_dir = Path(
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = load_harvest(cache_path)
    n_colors = X_full.shape[0] // N_T
    X_full = X_full[: n_colors * N_T]
    colors = load_xkcd_colors()[: n_colors]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hsv4 = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]), np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1], hsv[:, 2],
    ], axis=1)
    print(f"[per_template] n_colors={n_colors}  D={X_full.shape[1]}", flush=True)

    # Load templates from color_geometry.py
    from color_geometry import TEMPLATES

    # PCA to K shared across templates? Or per-template? Use per-template:
    # each template has its own 949×D matrix; we want the cleanest fit per
    # template, so PCA per template.
    K_PC = 32                         # smaller K — fits faster, captures bulk
    folds = kfold_color_indices(n_colors, 5)

    results = {}                      # template_idx -> {spec_label: (r2_mean, r2_std)}
    for t in range(N_T):
        X_t = X_full[t::N_T, :]       # (949, D) — one per color, this template
        # Per-dim standardize, PCA-K
        mu = X_t.mean(0, keepdims=True)
        sigma = X_t.std(0, keepdims=True).clip(min=1e-6)
        Xn = (X_t - mu) / sigma
        Xc = Xn - Xn.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt.T[:, :K_PC]                 # (949, K_PC)
        per_spec = {}
        for label, sid in SPECS:
            fold_r2 = []
            for tr, te in folds:
                pred = fit_predict(
                    sid, rgb[tr], hsv4[tr], Z[tr], rgb[te], hsv4[te],
                )
                fold_r2.append(r2_macro(Z[te], pred))
            per_spec[label] = (float(np.mean(fold_r2)), float(np.std(fold_r2)))
        results[t] = per_spec
        best = max(per_spec.values(), key=lambda x: x[0])[0]
        print(f"  t={t:2d}  best={best:+.3f}  {TEMPLATES[t][:55]}…",
              flush=True)

    # Plot — bars per template per spec, sorted by best spec score
    template_best = [max(results[t].values(), key=lambda x: x[0])[0]
                      for t in range(N_T)]
    order = np.argsort(template_best)[::-1]    # highest first

    fig, ax = plt.subplots(figsize=(16, 9))
    bar_w = 0.27
    colors_per_spec = {"Linear RGB": "#cfdee9", "Linear HSV": "#7baed1",
                        "Quartic poly HSV": "#356d96"}
    xs = np.arange(N_T)
    for i, (label, _) in enumerate(SPECS):
        ys = [results[t][label][0] for t in order]
        es = [results[t][label][1] for t in order]
        ax.bar(xs + (i - 1) * bar_w, ys, bar_w, yerr=es,
                color=colors_per_spec[label], edgecolor="black", linewidth=0.4,
                label=label, capsize=2)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(xs)
    # Truncate template text for tick labels
    short_labels = [TEMPLATES[t].replace("{x}", "X").replace("'", "")[:48]
                     for t in order]
    ax.set_xticklabels([f"t={t}  {lbl}…" for t, lbl in zip(order, short_labels)],
                        rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("held-out R²_macro  (5-fold CV by color)", fontsize=11)
    ax.set_title(
        f"Per-template supervised alignment — which prompts give the cleanest color signal\n"
        f"sorted by best-spec R²  ·  cogito L40  ·  n_colors = {n_colors}",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = out_dir / "per_template_alignment.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
