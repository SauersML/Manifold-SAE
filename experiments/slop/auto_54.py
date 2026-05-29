"""
auto_54 — (ssssss) Hue-derivative test on the cogito L40 manifold.

Question
--------
At each canonical-hue anchor on the unsupervised d=3 cogito manifold,
which direction is its *local tangent*? Specifically:
  if I take "red" and follow the principal local tangent of the manifold
  (in d=3 coordinate space), does that tangent decode to RGB motion
  toward "orange" / "pink" — i.e. the perceptual neighbours of red on
  the hue wheel?
If yes, the cogito manifold has correctly aligned its local geometry to
the perceptual color wheel — locally, motion along the surface == motion
along hue.

Method (allowed primitives only: PCA, ridge, k-NN)
--------------------------------------------------
1. Load T (949 x 3) from results.json (unsupervised d=3 GAM manifold)
   and the per-color RGB axes.
2. Pick 12 canonical color anchors by xkcd name (red, orange, yellow,
   green, ..., wrapping back to red). For each:
      a. take its k=25 nearest neighbours in T (k-NN on allow-list);
      b. centre that neighbourhood and run PCA;
      c. the top local PC of the neighbourhood IS the local tangent of
         the manifold there (PC1) — and PC2 is the secondary tangent.
3. Fit a *ridge* regressor T -> RGB (global, fold-free, very small
   regularisation 1e-6) — ridge is on the allow-list. The Jacobian
   J = beta (3x3 since T is 3d) maps a tangent direction in T-space
   to its predicted RGB direction.
4. Hue-derivative score: for each anchor, the unit RGB direction
   J @ tangent (sign-aligned to point away from anchor towards its
   higher-hue side) is compared to the empirical RGB delta from this
   anchor to (i) its two perceptual hue neighbours (cw/ccw on the
   colour wheel), and (ii) a random control set. We report the cosine
   similarity in *RGB space* between the predicted tangent direction
   and the actual RGB-delta to the named neighbours.

Outputs
-------
PNG: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_54.png  (4 panels)
JSON: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_54.json

Constraints satisfied: no Gaussian RBF; no Duchon length_scale set;
uses PCA + ridge + k-NN only.
"""
from __future__ import annotations

import json
import colorsys
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
OUT_PNG  = RUN_DIR / "auto_54.png"
OUT_JSON = RUN_DIR / "auto_54.json"

# Anchors that appear in xkcd_colors.txt, arranged ~clockwise on hue wheel.
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


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    T = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.column_stack([Rax, Gax, Bax])
    n_c = rgb.shape[0]
    assert T.shape == (n_c, 3)

    names, _ = load_xkcd()
    names = names[:n_c]
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    # Verify anchors all present
    anchors = [a for a in ANCHORS if a in name_to_idx]
    print(f"[anchors] using {len(anchors)} / {len(ANCHORS)}: {anchors}")

    # Global ridge T -> RGB (Jacobian = coefficients, since T is 3D)
    ridge = Ridge(alpha=1e-6, fit_intercept=True).fit(T, rgb)
    J = ridge.coef_  # shape (3 rgb, 3 T)
    rgb_hat = ridge.predict(T)
    global_r2 = 1.0 - np.sum((rgb - rgb_hat) ** 2) / np.sum(
        (rgb - rgb.mean(axis=0)) ** 2
    )
    print(f"[ridge] global T->RGB  R^2 = {global_r2:.4f}")
    print(f"[ridge] |J|_F = {np.linalg.norm(J):.4f}")

    # k-NN structure
    K = 25
    knn = NearestNeighbors(n_neighbors=K + 1).fit(T)

    results_per_anchor = []
    rng = np.random.default_rng(0)
    n_random_baseline = 200

    for anchor in anchors:
        ai = name_to_idx[anchor]
        # neighbour indices (exclude self)
        dist, idx = knn.kneighbors(T[ai:ai + 1], return_distance=True)
        neigh = idx[0, 1:]

        # local PCA on neighbourhood
        N = T[neigh] - T[ai]
        pca = PCA(n_components=3).fit(N)
        tangent1 = pca.components_[0]
        tangent2 = pca.components_[1]
        evr = pca.explained_variance_ratio_.tolist()

        # predicted RGB direction of tangent via Jacobian
        rgb_tan1 = J @ tangent1
        rgb_tan2 = J @ tangent2
        u_rgb_tan1 = unit(rgb_tan1)
        u_rgb_tan2 = unit(rgb_tan2)

        # neighbour by name on hue wheel: cw and ccw in ANCHORS list
        pos = anchors.index(anchor)
        ccw = anchors[(pos - 1) % len(anchors)]
        cw  = anchors[(pos + 1) % len(anchors)]
        d_ccw = unit(rgb[name_to_idx[ccw]] - rgb[ai])
        d_cw  = unit(rgb[name_to_idx[cw]]  - rgb[ai])

        # Sign tangent1 to maximise cosine to (cw) direction
        cos_cw_pos = cos_sim(u_rgb_tan1, d_cw)
        cos_cw_neg = cos_sim(-u_rgb_tan1, d_cw)
        if cos_cw_neg > cos_cw_pos:
            u_rgb_tan1 = -u_rgb_tan1
        # Best absolute cosine to either neighbour for tangent1 OR tangent2
        cs_t1_cw  = cos_sim(u_rgb_tan1, d_cw)
        cs_t1_ccw = cos_sim(u_rgb_tan1, d_ccw)
        cs_t2_cw  = cos_sim(u_rgb_tan2, d_cw)
        cs_t2_ccw = cos_sim(u_rgb_tan2, d_ccw)

        # Best of {t1, -t1, t2, -t2} for each neighbour (most generous)
        best_cw = max(cs_t1_cw, -cs_t1_cw, cs_t2_cw, -cs_t2_cw)
        best_ccw = max(cs_t1_ccw, -cs_t1_ccw, cs_t2_ccw, -cs_t2_ccw)

        # baseline: cosine between u_rgb_tan1 and 200 random unit dirs in RGB
        Z = rng.normal(size=(n_random_baseline, 3))
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        baseline_cos = np.abs(Z @ u_rgb_tan1)
        baseline_mean = float(baseline_cos.mean())
        baseline_p95  = float(np.quantile(baseline_cos, 0.95))

        results_per_anchor.append({
            "anchor": anchor,
            "ccw_neighbour": ccw,
            "cw_neighbour":  cw,
            "evr_local": evr,
            "rgb_tan1": rgb_tan1.tolist(),
            "rgb_tan2": rgb_tan2.tolist(),
            "u_rgb_tan1": u_rgb_tan1.tolist(),
            "u_rgb_tan2": u_rgb_tan2.tolist(),
            "d_cw_rgb": d_cw.tolist(),
            "d_ccw_rgb": d_ccw.tolist(),
            "cos_tan1_cw":  cs_t1_cw,
            "cos_tan1_ccw": cs_t1_ccw,
            "cos_tan2_cw":  cs_t2_cw,
            "cos_tan2_ccw": cs_t2_ccw,
            "best_abs_cos_cw":  best_cw,
            "best_abs_cos_ccw": best_ccw,
            "rand_baseline_abs_cos_mean": baseline_mean,
            "rand_baseline_abs_cos_p95":  baseline_p95,
        })

    # ------------------- plotting -------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0],
                          hspace=0.42, wspace=0.28,
                          left=0.06, right=0.98, top=0.92, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    n_a = len(results_per_anchor)
    xpos = np.arange(n_a)
    cw_cos  = np.array([r["cos_tan1_cw"]  for r in results_per_anchor])
    ccw_cos = np.array([r["cos_tan1_ccw"] for r in results_per_anchor])
    best_cw  = np.array([r["best_abs_cos_cw"]  for r in results_per_anchor])
    best_ccw = np.array([r["best_abs_cos_ccw"] for r in results_per_anchor])
    base_mean = np.array([r["rand_baseline_abs_cos_mean"]
                          for r in results_per_anchor])
    base_p95  = np.array([r["rand_baseline_abs_cos_p95"]
                          for r in results_per_anchor])
    anchor_labels = [r["anchor"] for r in results_per_anchor]
    anchor_rgb = np.array([rgb[name_to_idx[a]] for a in anchor_labels])

    # A: tan1 signed cosine to cw / ccw, with anchor swatch under x-axis
    w = 0.4
    axA.bar(xpos - w/2, cw_cos,  width=w, color="#1f77b4",
            label="cos(tan1_RGB, delta_cw)")
    axA.bar(xpos + w/2, ccw_cos, width=w, color="#ff7f0e",
            label="cos(tan1_RGB, delta_ccw)")
    axA.axhline(0, color="black", lw=0.6)
    axA.set_xticks(xpos)
    axA.set_xticklabels(anchor_labels, rotation=30, ha="right", fontsize=9)
    axA.set_ylabel("cosine in RGB space")
    axA.set_title("Local tangent 1 (PC1 of neighbourhood) mapped to RGB:\n"
                  "does it align with delta-RGB to hue-neighbours?")
    axA.legend(fontsize=9, loc="lower right")
    axA.grid(axis="y", alpha=0.3)
    # swatch row
    ymin, ymax = axA.get_ylim()
    pad = (ymax - ymin) * 0.06
    axA.set_ylim(ymin - pad, ymax)
    for i, rgb_i in enumerate(anchor_rgb):
        axA.add_patch(Rectangle((i - 0.45, ymin - pad), 0.9, pad * 0.85,
                                facecolor=rgb_i, edgecolor="black", lw=0.4,
                                clip_on=False))

    # B: best-of-tangents absolute cosine vs random baseline
    axB.bar(xpos - w/2, best_cw,  width=w, color="#2ca02c",
            label="best |cos| to cw")
    axB.bar(xpos + w/2, best_ccw, width=w, color="#9467bd",
            label="best |cos| to ccw")
    axB.plot(xpos, base_mean, "k--", marker="o", ms=4,
             label="random |cos| mean")
    axB.plot(xpos, base_p95, "k:",  marker="s", ms=4,
             label="random |cos| p95")
    axB.set_xticks(xpos)
    axB.set_xticklabels(anchor_labels, rotation=30, ha="right", fontsize=9)
    axB.set_ylabel("|cosine|")
    axB.set_title("Best-of-tangents alignment vs random-direction baseline")
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(axis="y", alpha=0.3)
    axB.set_ylim(0, max(1.0, base_p95.max() * 1.1, best_cw.max() * 1.1))

    # C: quiver in 2D RGB-projected slice (R, G) showing predicted vs actual
    axC.set_title("Predicted local hue-derivative (tan1->RGB) vs actual\n"
                  "delta-RGB to hue-neighbours  [projected on R-G plane]")
    for i, r in enumerate(results_per_anchor):
        a_rgb = anchor_rgb[i]
        # predicted tangent (red) at anchor in RG plane
        axC.scatter([a_rgb[0]], [a_rgb[1]], s=110, c=[a_rgb],
                    edgecolors="black", linewidths=0.7)
        axC.annotate(r["anchor"], (a_rgb[0], a_rgb[1]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
        u = np.array(r["u_rgb_tan1"]) * 0.12
        axC.arrow(a_rgb[0], a_rgb[1], u[0], u[1],
                  head_width=0.012, head_length=0.012,
                  fc="red", ec="red", alpha=0.85, length_includes_head=True)
        # actual cw direction
        d = np.array(r["d_cw_rgb"]) * 0.12
        axC.arrow(a_rgb[0], a_rgb[1], d[0], d[1],
                  head_width=0.012, head_length=0.012,
                  fc="black", ec="black", alpha=0.65,
                  length_includes_head=True)
    axC.set_xlabel("R")
    axC.set_ylabel("G")
    axC.set_xlim(-0.05, 1.05)
    axC.set_ylim(-0.05, 1.05)
    axC.set_aspect("equal")
    axC.grid(alpha=0.3)
    # legend proxies
    axC.plot([], [], color="red", label="predicted tan1 (RGB)")
    axC.plot([], [], color="black", label="actual delta to cw neighbour")
    axC.legend(fontsize=8, loc="lower right")

    # D: summary table-as-text panel
    axD.axis("off")
    mean_signed_cw  = float(cw_cos.mean())
    mean_signed_ccw = float(ccw_cos.mean())
    mean_best_cw  = float(best_cw.mean())
    mean_best_ccw = float(best_ccw.mean())
    mean_base_p95 = float(base_p95.mean())
    sgn_pos_cw  = int(np.sum(cw_cos > 0))
    sgn_pos_ccw = int(np.sum(ccw_cos > 0))
    n_above_p95_cw  = int(np.sum(best_cw  > base_p95))
    n_above_p95_ccw = int(np.sum(best_ccw > base_p95))

    txt = (
        "Summary (auto_54, idea ssssss)\n"
        "================================\n\n"
        f"ridge T->RGB  R^2  = {global_r2:.4f}\n"
        f"anchors used        : {n_a} / {len(ANCHORS)}\n"
        f"neighbourhood k     : {K}\n\n"
        "Tan1 (signed, sign-aligned to cw)\n"
        f"  mean cos to cw  = {mean_signed_cw:+.3f}   "
        f"({sgn_pos_cw}/{n_a} anchors > 0)\n"
        f"  mean cos to ccw = {mean_signed_ccw:+.3f}   "
        f"({sgn_pos_ccw}/{n_a} anchors > 0)\n\n"
        "Best-of-tangents |cos|\n"
        f"  mean |cos| to cw  = {mean_best_cw:.3f}\n"
        f"  mean |cos| to ccw = {mean_best_ccw:.3f}\n"
        f"  random |cos| p95 mean = {mean_base_p95:.3f}\n"
        f"  anchors above random-p95 (cw)  : {n_above_p95_cw}/{n_a}\n"
        f"  anchors above random-p95 (ccw) : {n_above_p95_ccw}/{n_a}\n\n"
        "Interpretation\n"
        "--------------\n"
        "If the manifold's local tangent at 'red' decodes (via ridge) to\n"
        "an RGB direction that points toward orange/pink, the manifold\n"
        "has internalised the perceptual hue-wheel locally.\n"
        "Values >> random baseline indicate the manifold's d=3 surface\n"
        "is locally co-oriented with the colour wheel; values near the\n"
        "random baseline indicate cogito's geometry is locally unaligned\n"
        "with named-colour neighbour structure."
    )
    axD.text(0.02, 0.98, txt, family="monospace", fontsize=10,
             verticalalignment="top", transform=axD.transAxes)

    fig.suptitle(
        "auto_54 (ssssss): manifold local tangent at canonical hue anchors "
        "decoded into RGB direction  [cogito L40, unsup d=3 GAM manifold]",
        fontsize=12,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    summary = {
        "idea": "ssssss",
        "n_colors": int(n_c),
        "anchors": anchor_labels,
        "knn_k": K,
        "ridge_global_T_to_RGB_R2": float(global_r2),
        "jacobian_T_to_RGB": J.tolist(),
        "mean_signed_cos_tan1_cw":  mean_signed_cw,
        "mean_signed_cos_tan1_ccw": mean_signed_ccw,
        "mean_best_abs_cos_cw":  mean_best_cw,
        "mean_best_abs_cos_ccw": mean_best_ccw,
        "mean_random_baseline_p95": mean_base_p95,
        "per_anchor": results_per_anchor,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
