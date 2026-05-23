"""auto_40: Hue-quantile R^2 (idea fffff).

Split the 949 colors into 4 hue quartiles and compute test-fold macro
R^2 inside each quartile, for several supervised feature sets. Answers
the question: *which hue regions does cogito's L40 representation
encode well vs poorly?*

Pipeline
--------
- Load harvest X (n_colors*28, 7168), make per-color centroids.
- Standardize centroids -> SVD -> top-64 PCs (matches results.json).
- Cross-validated ridge (5-fold over colors), each color appears once
  in a test fold; we collect predicted Z for every color.
- Per-color R^2 across PCs is reported by aggregating SS_res / SS_tot
  *within* each hue quartile using the test-fold predictions.
- Feature sets compared:
    * L_lin_rgb         (3 features)
    * L_joint_rgb_with_hue
    * joint_rgb_poly    (10 features, matches auto_39's winner)
    * knn_rgb_k20       (nonparametric baseline)

Output
------
runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_40.{json,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_40.png"
OUT_JSON = OUT_DIR / "auto_40.json"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5
N_QUARTILES = 4
RIDGE_ALPHA = 1.0  # auto_39 optimum is near 1.0


def lin_rgb(rgb, hue):
    return rgb.copy()


def joint_rgb_with_hue(rgb, hue):
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    h = hue
    return np.stack([
        R, G, B,
        R * G, R * B, G * B, R * G * B,
        np.cos(2 * np.pi * h), np.sin(2 * np.pi * h),
        np.cos(4 * np.pi * h), np.sin(4 * np.pi * h),
    ], axis=1)


def joint_rgb_poly(rgb, hue):
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.stack(
        [R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, R * G * B],
        axis=1,
    )


FEATURE_FNS = {
    "L_lin_rgb": lin_rgb,
    "L_joint_rgb_with_hue": joint_rgb_with_hue,
    "L_joint_rgb_poly": joint_rgb_poly,
}


def cv_ridge_predict(Phi, Z, n_folds, alpha, seed=0):
    """Return predicted Z (n_colors, n_pcs) via out-of-fold ridge."""
    n = Phi.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    Zhat = np.zeros_like(Z)
    for tr, te in kf.split(np.arange(n)):
        mu = Phi[tr].mean(0, keepdims=True)
        sd = Phi[tr].std(0, keepdims=True).clip(min=1e-9)
        Ptr = (Phi[tr] - mu) / sd
        Pte = (Phi[te] - mu) / sd
        mod = Ridge(alpha=alpha, fit_intercept=True)
        mod.fit(Ptr, Z[tr])
        Zhat[te] = mod.predict(Pte)
    return Zhat


def cv_knn_predict(rgb, Z, n_folds, k=20, seed=0):
    n = rgb.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    Zhat = np.zeros_like(Z)
    for tr, te in kf.split(np.arange(n)):
        nn = NearestNeighbors(n_neighbors=k).fit(rgb[tr])
        _, idx = nn.kneighbors(rgb[te])
        Zhat[te] = Z[tr][idx].mean(axis=1)
    return Zhat


def quartile_r2(Z, Zhat, hue, n_q=4):
    """Macro R^2 (mean over PCs) computed *inside* each hue quartile,
    using each quartile's own per-PC mean as the null model."""
    edges = np.quantile(hue, np.linspace(0, 1, n_q + 1))
    edges[-1] += 1e-9
    out = []
    for qi in range(n_q):
        m = (hue >= edges[qi]) & (hue < edges[qi + 1])
        n_in = int(m.sum())
        Zq, Zhq = Z[m], Zhat[m]
        ss_res = ((Zq - Zhq) ** 2).sum(0)
        ss_tot = ((Zq - Zq.mean(0, keepdims=True)) ** 2).sum(0)
        r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
        out.append({
            "lo": float(edges[qi]),
            "hi": float(edges[qi + 1]),
            "n": n_in,
            "r2_macro": float(r2.mean()),
            "r2_per_pc": r2.tolist(),
        })
    return out, edges.tolist()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] results.json", flush=True)
    res = json.loads(RESULTS.read_text())
    ax = res["color_axes_per_color_index"]
    R = np.asarray(ax["R"], dtype=np.float64)
    G = np.asarray(ax["G"], dtype=np.float64)
    B = np.asarray(ax["B"], dtype=np.float64)
    hue = np.asarray(ax["hue"], dtype=np.float64)
    rgb = np.stack([R, G, B], axis=1)
    n_colors = rgb.shape[0]
    print(f"[load] n_colors={n_colors}", flush=True)

    print("[load] harvest", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    assert N == n_colors * N_TEMPLATES, f"{N} != {n_colors}*{N_TEMPLATES}"

    # Per-color centroids
    centroids = X.reshape(n_colors, N_TEMPLATES, D).mean(axis=1)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma
    Cc = Cn - Cn.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    Z = Cn @ Vt[:N_PCS].T
    print(f"[pca] Z.shape={Z.shape}", flush=True)

    # Global R^2 (sanity) and per-quartile R^2 for each feature set
    results_by_spec = {}
    for name, fn in FEATURE_FNS.items():
        Phi = fn(rgb, hue)
        Zhat = cv_ridge_predict(Phi, Z, N_FOLDS, RIDGE_ALPHA, seed=0)
        ss_res = ((Z - Zhat) ** 2).sum(0)
        ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum(0)
        r2_global = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
        qres, edges = quartile_r2(Z, Zhat, hue, n_q=N_QUARTILES)
        results_by_spec[name] = {
            "r2_macro_global": float(r2_global.mean()),
            "quartiles": qres,
            "hue_edges": edges,
            "n_features": int(Phi.shape[1]),
        }
        print(f"[{name}] global={r2_global.mean():+.4f}  " +
              "  ".join(f"q{i}={q['r2_macro']:+.3f}(n={q['n']})"
                       for i, q in enumerate(qres)), flush=True)

    # kNN nonparametric baseline (k=20)
    Zhat_knn = cv_knn_predict(rgb, Z, N_FOLDS, k=20, seed=0)
    ss_res = ((Z - Zhat_knn) ** 2).sum(0)
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum(0)
    r2_global_knn = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    qres_knn, edges_knn = quartile_r2(Z, Zhat_knn, hue, n_q=N_QUARTILES)
    results_by_spec["N_knn_rgb_k20"] = {
        "r2_macro_global": float(r2_global_knn.mean()),
        "quartiles": qres_knn,
        "hue_edges": edges_knn,
        "n_features": 3,
    }
    print(f"[knn_k20] global={r2_global_knn.mean():+.4f}  " +
          "  ".join(f"q{i}={q['r2_macro']:+.3f}" for i, q in enumerate(qres_knn)),
          flush=True)

    summary = {
        "config": {
            "n_colors": n_colors, "n_templates": N_TEMPLATES,
            "n_pcs": N_PCS, "n_folds": N_FOLDS,
            "ridge_alpha": RIDGE_ALPHA, "n_quartiles": N_QUARTILES,
        },
        "specs": results_by_spec,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # ---------- Plot ----------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    spec_names = list(results_by_spec.keys())
    colors = ["#3050a0", "#c04050", "#208060", "#a06030"]

    # (a) Grouped bar: R^2 by quartile, per spec
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(N_QUARTILES)
    w = 0.18
    for si, name in enumerate(spec_names):
        ys = [q["r2_macro"] for q in results_by_spec[name]["quartiles"]]
        ax.bar(x + (si - (len(spec_names) - 1) / 2) * w, ys, w,
               color=colors[si % len(colors)], label=name, edgecolor="k", lw=0.4)
    edges = results_by_spec[spec_names[0]]["hue_edges"]
    xticklabels = [f"Q{i+1}\n[{edges[i]:.2f},{edges[i+1]:.2f})"
                   for i in range(N_QUARTILES)]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("macro R^2 (mean over 64 PCs)")
    ax.set_title("Hue-quartile R^2 by feature set (test-fold predictions, 5-fold CV)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")

    # (b) Swatches of hue quartile centers (HSV w/ s=1, v=1)
    import colorsys
    ax = fig.add_subplot(gs[0, 2])
    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(N_QUARTILES)]
    for i, hc in enumerate(bin_centers):
        rgb_swatch = colorsys.hsv_to_rgb(hc, 1.0, 1.0)
        ax.add_patch(plt.Rectangle((0, N_QUARTILES - 1 - i), 1, 1,
                                    facecolor=rgb_swatch, edgecolor="k"))
        ax.text(1.05, N_QUARTILES - 1 - i + 0.5,
                f"Q{i+1}: hue~{hc:.2f}  n={results_by_spec[spec_names[0]]['quartiles'][i]['n']}",
                va="center", fontsize=9)
    ax.set_xlim(0, 3.5); ax.set_ylim(0, N_QUARTILES)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Hue quartile centers (s=v=1)")

    # (c) Best spec: per-PC R^2 across quartiles (heatmap)
    best_name = max(spec_names, key=lambda n: results_by_spec[n]["r2_macro_global"])
    ax = fig.add_subplot(gs[1, 0])
    M = np.array([q["r2_per_pc"] for q in results_by_spec[best_name]["quartiles"]])
    vmin, vmax = -0.3, float(M.max())
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax)
    ax.set_yticks(range(N_QUARTILES))
    ax.set_yticklabels([f"Q{i+1}" for i in range(N_QUARTILES)])
    ax.set_xlabel("PC index")
    ax.set_title(f"Per-PC R^2 by quartile\n[best spec: {best_name}]")
    plt.colorbar(im, ax=ax, label="R^2")

    # (d) Hue histogram with quartile boundaries
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(hue, bins=40, color="gray", edgecolor="k", lw=0.4)
    for e in edges[1:-1]:
        ax.axvline(e, color="red", ls="--", lw=1)
    ax.set_xlabel("hue")
    ax.set_ylabel("count (colors)")
    ax.set_title("Hue distribution + quartile edges")

    # (e) Spec ranking
    ax = fig.add_subplot(gs[1, 2])
    spec_globals = [(n, results_by_spec[n]["r2_macro_global"]) for n in spec_names]
    spec_globals.sort(key=lambda t: t[1])
    names_s = [t[0] for t in spec_globals]
    vals_s = [t[1] for t in spec_globals]
    ax.barh(names_s, vals_s, color="#406080", edgecolor="k", lw=0.4)
    for i, v in enumerate(vals_s):
        ax.text(v + 0.002, i, f"{v:+.3f}", va="center", fontsize=9)
    ax.set_xlabel("global macro R^2")
    ax.set_title("Spec ranking (global, 5-fold CV)")
    ax.set_xlim(0, max(vals_s) * 1.18)

    fig.suptitle("auto_40  -  Hue-quantile R^2 (4 quartiles) on cogito L40 PCs",
                 fontsize=13, y=0.995)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
