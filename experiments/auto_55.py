"""
auto_55 - (uuuuuu) Radius of curvature at each canonical color on fitted U_3d.

Question
--------
For the unsupervised d=3 cogito manifold T (n_colors x 3), what is the
*local radius of curvature* of the embedded surface at each canonical
color point? A small radius means the manifold sharply bends through
that color; a large radius means the surface is locally flat there.
This is a 1/Lipschitz-style local geometry measure.

Method (allow-listed primitives: PCA, ridge, k-NN)
--------------------------------------------------
1. Load T (n_colors x 3) from results.json (unsup d=3 GAM manifold).
2. For each color point p_i:
     a. Find K=30 nearest neighbours in T via k-NN.
     b. Local PCA on neighbours -> top-2 PCs define the local tangent
        plane (u, v); the 3rd PC is the local normal n.
     c. Express each neighbour as (u_j, v_j, w_j) in this local frame.
     d. Fit (ridge, very small alpha) a quadratic Monge form
            w = a*u^2 + b*u*v + c*v^2 + d*u + e*v + f
        to the neighbours.  The shape operator is
            H = [[2a, b], [b, 2c]]
        with principal curvatures = eigenvalues of H (in the tangent
        frame, since the metric is identity to leading order).
     e. Mean curvature kappa_mean = (k1 + k2)/2;
        Gaussian curvature K_gauss = k1 * k2;
        radius of curvature R = 1 / max(|kappa_mean|, eps).
3. Pick 12 canonical xkcd anchors (red, orange, ..., brown) and report
   their radius of curvature, plus a global histogram over all colors.

Outputs
-------
PNG : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_55.png  (4 panels)
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_55.json

Constraints: no Gaussian RBF; no Duchon length_scale set;
uses PCA + ridge + k-NN only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
XKCD     = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG  = RUN_DIR / "auto_55.png"
OUT_JSON = RUN_DIR / "auto_55.json"

ANCHORS = [
    "red", "orange", "yellow", "lime green", "green",
    "teal", "cyan", "blue", "purple", "magenta",
    "pink", "brown",
]


def load_xkcd() -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        name, hex_ = parts[0], parts[1].lstrip("#")
        if len(hex_) != 6:
            continue
        r = int(hex_[0:2], 16) / 255.0
        g = int(hex_[2:4], 16) / 255.0
        b = int(hex_[4:6], 16) / 255.0
        names.append(name)
        rgb.append([r, g, b])
    return names, np.asarray(rgb, dtype=np.float64)


def local_curvature(p: np.ndarray, neigh: np.ndarray,
                    alpha: float = 1e-6) -> dict:
    """Fit a Monge quadratic via local-PCA + ridge.

    Returns dict with principal curvatures k1,k2, mean H, Gaussian K,
    and radius R = 1 / max(|H|, eps).
    """
    # local PCA: tangent plane (PC1, PC2), normal (PC3)
    N = neigh - p
    pca = PCA(n_components=3).fit(N)
    comps = pca.components_  # (3, 3): rows are PCs in T-space
    evr = pca.explained_variance_ratio_
    # local frame: u=PC1, v=PC2, n=PC3
    coords = N @ comps.T  # (k, 3)
    u, v, w = coords[:, 0], coords[:, 1], coords[:, 2]
    # quadratic Monge: w = a u^2 + b uv + c v^2 + d u + e v + f
    Phi = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    rr = Ridge(alpha=alpha, fit_intercept=False).fit(Phi, w)
    a, b, c, _d, _e, _f = rr.coef_
    H = np.array([[2.0 * a, b], [b, 2.0 * c]], dtype=np.float64)
    eigs = np.linalg.eigvalsh(H)  # principal curvatures in tangent frame
    k1, k2 = float(eigs[0]), float(eigs[1])
    kappa_mean = 0.5 * (k1 + k2)
    K_gauss = k1 * k2
    eps = 1e-8
    R_mean = 1.0 / max(abs(kappa_mean), eps)
    R_max  = 1.0 / max(max(abs(k1), abs(k2)), eps)  # tightest principal R
    # residual goodness-of-fit
    w_hat = Phi @ rr.coef_
    ss_res = float(np.sum((w - w_hat) ** 2))
    ss_tot = float(np.sum((w - w.mean()) ** 2) + 1e-12)
    fit_r2 = 1.0 - ss_res / ss_tot
    return {
        "k1": k1, "k2": k2,
        "kappa_mean": float(kappa_mean),
        "K_gauss": float(K_gauss),
        "R_mean": float(R_mean),
        "R_tight": float(R_max),
        "fit_r2": fit_r2,
        "evr_local": evr.tolist(),
        "normal_evr_frac": float(evr[2]),
    }


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    T = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.column_stack([Rax, Gax, Bax])
    n_c = T.shape[0]
    print(f"[shapes] T={T.shape}, rgb={rgb.shape}")

    names, _ = load_xkcd()
    names = names[:n_c]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    anchors = [a for a in ANCHORS if a in name_to_idx]
    print(f"[anchors] {len(anchors)} / {len(ANCHORS)}: {anchors}")

    # k-NN over T
    K = 30
    knn = NearestNeighbors(n_neighbors=K + 1).fit(T)
    _, all_idx = knn.kneighbors(T, return_distance=True)

    # Compute curvature for every point
    R_mean_all  = np.zeros(n_c)
    R_tight_all = np.zeros(n_c)
    kappa_all   = np.zeros(n_c)
    Kg_all      = np.zeros(n_c)
    fit_r2_all  = np.zeros(n_c)
    nrm_evr_all = np.zeros(n_c)

    for i in range(n_c):
        neigh = T[all_idx[i, 1:]]
        info = local_curvature(T[i], neigh)
        R_mean_all[i]  = info["R_mean"]
        R_tight_all[i] = info["R_tight"]
        kappa_all[i]   = info["kappa_mean"]
        Kg_all[i]      = info["K_gauss"]
        fit_r2_all[i]  = info["fit_r2"]
        nrm_evr_all[i] = info["normal_evr_frac"]

    # Scale of the manifold (for context)
    T_scale = float(np.median(np.linalg.norm(T - T.mean(axis=0), axis=1)))
    R_mean_norm  = R_mean_all  / T_scale
    R_tight_norm = R_tight_all / T_scale

    # Per-anchor table
    anchor_rows = []
    for a in anchors:
        i = name_to_idx[a]
        anchor_rows.append({
            "anchor": a,
            "idx": int(i),
            "rgb": rgb[i].tolist(),
            "R_mean": float(R_mean_all[i]),
            "R_tight": float(R_tight_all[i]),
            "R_mean_over_scale": float(R_mean_norm[i]),
            "R_tight_over_scale": float(R_tight_norm[i]),
            "kappa_mean": float(kappa_all[i]),
            "K_gauss": float(Kg_all[i]),
            "monge_fit_r2": float(fit_r2_all[i]),
            "normal_evr_frac": float(nrm_evr_all[i]),
        })

    # Rank ALL colors by tightness (smallest R = most bent)
    order_tight = np.argsort(R_tight_all)
    top_bent = [{"name": names[int(j)], "idx": int(j),
                 "rgb": rgb[int(j)].tolist(),
                 "R_tight": float(R_tight_all[j]),
                 "R_tight_over_scale": float(R_tight_norm[j])}
                for j in order_tight[:15]]
    top_flat = [{"name": names[int(j)], "idx": int(j),
                 "rgb": rgb[int(j)].tolist(),
                 "R_tight": float(R_tight_all[j]),
                 "R_tight_over_scale": float(R_tight_norm[j])}
                for j in order_tight[-15:][::-1]]

    # ----------------- plotting ---------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.30,
                          left=0.07, right=0.98, top=0.92, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A: anchor radii (mean + tight) bar chart with swatches
    n_a = len(anchor_rows)
    xpos = np.arange(n_a)
    Rm_vals = np.array([r["R_mean_over_scale"]  for r in anchor_rows])
    Rt_vals = np.array([r["R_tight_over_scale"] for r in anchor_rows])
    a_labels = [r["anchor"] for r in anchor_rows]
    a_rgb = np.array([r["rgb"] for r in anchor_rows])
    w = 0.4
    axA.bar(xpos - w/2, Rm_vals, width=w, color="#1f77b4",
            label="R_mean / scale  (1/|H|)")
    axA.bar(xpos + w/2, Rt_vals, width=w, color="#d62728",
            label="R_tight / scale (1/max|k_i|)")
    axA.set_yscale("log")
    axA.set_xticks(xpos)
    axA.set_xticklabels(a_labels, rotation=30, ha="right", fontsize=9)
    axA.set_ylabel("radius of curvature / median |T - mean|")
    axA.set_title("Local radius of curvature at canonical hue anchors\n"
                  "(small = sharply bent; large = locally flat)")
    axA.legend(fontsize=9, loc="upper right")
    axA.grid(axis="y", alpha=0.3, which="both")
    # swatch row at bottom
    ymin, ymax = axA.get_ylim()
    # log scale: place swatches just under the visible bottom
    sw_h = (np.log10(ymax) - np.log10(ymin)) * 0.06
    sw_y = 10 ** (np.log10(ymin) - sw_h)
    sw_top = ymin
    axA.set_ylim(sw_y * 0.95, ymax)
    for i, c in enumerate(a_rgb):
        axA.add_patch(Rectangle((i - 0.45, sw_y), 0.9, sw_top - sw_y,
                                facecolor=c, edgecolor="black", lw=0.4,
                                clip_on=False))

    # B: distribution of log10(R_tight / scale) over all colors with anchor lines
    log_R = np.log10(np.clip(R_tight_norm, 1e-6, None))
    axB.hist(log_R, bins=50, color="#888888", edgecolor="black", alpha=0.75)
    for r in anchor_rows:
        x = np.log10(max(r["R_tight_over_scale"], 1e-6))
        axB.axvline(x, color=r["rgb"], lw=2.0, alpha=0.95)
        axB.text(x, axB.get_ylim()[1] * 0.95, r["anchor"],
                 rotation=90, va="top", ha="right",
                 fontsize=7, color="black",
                 bbox={"facecolor": r["rgb"], "edgecolor": "black",
                       "pad": 1.0, "alpha": 0.8})
    axB.set_xlabel("log10( R_tight / scale )")
    axB.set_ylabel("count of colors")
    axB.set_title("Distribution of local tightest radius across all colors\n"
                  "(anchors marked by colored vertical lines)")
    axB.grid(axis="y", alpha=0.3)

    # C: scatter R_tight/scale vs Monge-fit quality (sanity)
    sc = axC.scatter(fit_r2_all, R_tight_norm, c=rgb, s=14,
                     edgecolors="black", linewidths=0.2, alpha=0.85)
    axC.set_yscale("log")
    axC.set_xlabel("Monge quadratic fit R^2 (per point)")
    axC.set_ylabel("R_tight / scale")
    axC.set_title("Curvature estimate vs local-quadratic fit quality\n"
                  "(low fit R^2 -> curvature estimate is noisy)")
    axC.grid(alpha=0.3, which="both")
    for r in anchor_rows:
        i = r["idx"]
        axC.annotate(r["anchor"], (fit_r2_all[i], R_tight_norm[i]),
                     fontsize=7, xytext=(4, 2),
                     textcoords="offset points")

    # D: swatch grid: 15 most-bent (left) vs 15 most-flat (right)
    axD.axis("off")
    axD.set_xlim(0, 10)
    axD.set_ylim(0, 16)
    axD.set_title("Most bent (left)            vs           Most flat (right)\n"
                  "ranked by R_tight on U_3d", fontsize=10)
    for k, r in enumerate(top_bent):
        y = 15 - k
        axD.add_patch(Rectangle((0.2, y - 0.45), 0.8, 0.9,
                                facecolor=r["rgb"], edgecolor="black", lw=0.4))
        axD.text(1.15, y, f"{r['name'][:22]:<22s}  R/s={r['R_tight_over_scale']:.3f}",
                 fontsize=8, va="center", family="monospace")
    for k, r in enumerate(top_flat):
        y = 15 - k
        axD.add_patch(Rectangle((5.2, y - 0.45), 0.8, 0.9,
                                facecolor=r["rgb"], edgecolor="black", lw=0.4))
        axD.text(6.15, y, f"{r['name'][:22]:<22s}  R/s={r['R_tight_over_scale']:.2f}",
                 fontsize=8, va="center", family="monospace")

    fig.suptitle(
        "auto_55 (uuuuuu): local radius of curvature on fitted U_3d "
        "manifold  [cogito L40, unsup d=3 GAM, k-NN + local PCA + ridge Monge]",
        fontsize=12,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    summary = {
        "idea": "uuuuuu",
        "n_colors": int(n_c),
        "knn_k": int(K),
        "T_scale_median_norm": T_scale,
        "global": {
            "R_tight_over_scale_median": float(np.median(R_tight_norm)),
            "R_tight_over_scale_p10": float(np.quantile(R_tight_norm, 0.10)),
            "R_tight_over_scale_p90": float(np.quantile(R_tight_norm, 0.90)),
            "R_mean_over_scale_median":  float(np.median(R_mean_norm)),
            "Gaussian_K_median":         float(np.median(Kg_all)),
            "Gaussian_K_pos_frac":       float(np.mean(Kg_all > 0)),
            "Monge_fit_r2_median":       float(np.median(fit_r2_all)),
            "normal_evr_frac_median":    float(np.median(nrm_evr_all)),
        },
        "per_anchor": anchor_rows,
        "top_bent": top_bent,
        "top_flat": top_flat,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
