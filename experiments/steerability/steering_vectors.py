"""Derive cogito-L40 steering vectors and validate them offline.

Three families of steering directions are built from the harvested
residual-stream cache (`runs/COLOR_COGITO_L40/X_L40.npy`, shape
(N_prompts=26572, D=7168) = 949 xkcd colors x 28 templates):

  * HSV axes (axis_hue, axis_sat, axis_val): ridge-regress each row of
    X onto the (hue, sat, val) of its named color. Each axis_* is the
    direction in residual space whose 1D projection best predicts the
    corresponding HSV scalar (i.e. it's the OLS / ridge coefficient
    vector for that scalar regressed on X).  We Z-normalise these to
    unit norm because their natural scale is unit-of-1/feature.

  * U_3d axes: top-3 standardised-per-color-centroid PCs from
    `_pca_basis.load_pc_basis(K=64)` (just the first three rows of Vt).

  * Concept axes: difference-of-means -- for each concept in
    {red, blue, green, achromatic}, take the rows whose color name
    matches the concept (by Euclidean distance from the color's HSV to
    the prototype HSV), average them, then subtract the global mean.

Offline validation: for each axis, project X onto the unit axis, plot
a histogram coloured by ground-truth hue, and compute an AUC for
"top-decile of axis projection" vs "hue within 30deg of the axis's
target hue".

Outputs (next to this file):
  * `hsv_axes_L40.npz`         (axis_hue, axis_sat, axis_val)
  * `u3d_axes_L40.npz`         (axis_pc1, axis_pc2, axis_pc3, evr)
  * `concept_axes_L40.npz`     (axis_red, axis_blue, axis_green, axis_achromatic)
  * `steering_vectors_offline_report.png`
  * `steering_vectors_offline_report.json`
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np

# Repo paths
HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40")
HARVEST = RUN_DIR / "X_L40.npy"
N_TEMPLATES = 28

sys.path.insert(0, str(EXP_DIR))
from plot_color_geometry import load_xkcd_colors  # noqa: E402
from _pca_basis import load_pc_basis  # noqa: E402


def _rgb_to_hsv_arr(rgb01: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb01)
    for i, (r, g, b) in enumerate(rgb01):
        out[i] = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    return out


def build_per_row_labels(n_rows: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (HSV per row, RGB per row, color_name per row)."""
    xs = load_xkcd_colors()
    n_colors = n_rows // N_TEMPLATES
    xs = xs[:n_colors]
    names = [n for (n, _, _, _) in xs]
    rgb01 = np.array([[r / 255.0, g / 255.0, b / 255.0] for (_, r, g, b) in xs])
    hsv = _rgb_to_hsv_arr(rgb01)
    # broadcast to rows: row i has color (i // N_TEMPLATES)
    color_idx = np.arange(n_rows) // N_TEMPLATES
    return hsv[color_idx], rgb01[color_idx], [names[i] for i in color_idx]


def ridge_axis(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Return w such that X @ w ~ y.  Centered ridge; w unit-normed."""
    mu = X.mean(axis=0)
    Xc = X - mu
    yc = y - y.mean()
    # Use the dual form because D >> N is false here (N=26572, D=7168);
    # primal D x D is fine but heavy.  We use SVD on Xc for stability.
    # Solve (X'X + lam I) w = X' y  via SVD.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # w = V diag(s/(s^2+lam)) U' y
    coef = (s / (s ** 2 + lam)) * (U.T @ yc)
    w = Vt.T @ coef
    n = np.linalg.norm(w)
    return w / (n + 1e-12)


def hue_distance_deg(h1: np.ndarray, h2: float) -> np.ndarray:
    """Min circular distance in degrees between hue (0..1) arrays and target (0..1)."""
    d = np.abs(h1 - h2)
    return 360.0 * np.minimum(d, 1.0 - d)


def auc_top_decile_vs_hue(proj: np.ndarray, hue: np.ndarray, target_hue: float,
                          half_window_deg: float = 30.0) -> float:
    """AUC for: positive = hue within `half_window_deg` of target_hue,
    score = |proj| ranked decile-style (we use raw proj signed).

    Following the spec: "top-decile of axis projection" vs "row's hue
    within 30 deg of axis target".  We turn this into a binary
    classification AUC of the proj (or -proj if it would give AUC<0.5).
    """
    positive = hue_distance_deg(hue, target_hue) <= half_window_deg
    if positive.sum() == 0 or positive.sum() == len(positive):
        return float("nan")
    # AUC via Mann-Whitney
    order = np.argsort(proj)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(proj) + 1)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


CONCEPTS = {
    # name -> prototype HSV (hue, sat, val)
    "red":         (0.00, 0.85, 0.80),
    "blue":        (0.62, 0.85, 0.70),
    "green":       (0.33, 0.85, 0.55),
    "achromatic":  (0.00, 0.00, 0.50),  # gray-ish, low saturation
}


def concept_mask(hsv: np.ndarray, proto: tuple[float, float, float]) -> np.ndarray:
    h, s, v = proto
    if s < 0.1:
        # achromatic: rows with saturation below 0.15
        return hsv[:, 1] < 0.15
    return (hue_distance_deg(hsv[:, 0], h) <= 25.0) & (hsv[:, 1] > 0.4)


def main() -> int:
    print(f"[load] mmap'ing {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    N, D = X.shape
    print(f"[load] X shape={X.shape} dtype={X.dtype}", flush=True)

    hsv, rgb, names = build_per_row_labels(N)
    print(f"[labels] hsv={hsv.shape} rgb={rgb.shape}", flush=True)

    # Subsample for ridge fit: 26572 x 7168 is ~760 MB; SVD on the full
    # matrix is ~10-20s, manageable. But mmap reads are slow, so cast
    # to a contiguous float32 with a column subsample if needed.
    # We'll fit on a row subsample of 6000 for speed and use the full
    # X only for projection in the validation step.
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(N, size=min(6000, N), replace=False)
    X_fit = np.ascontiguousarray(X[fit_idx], dtype=np.float32)

    # HSV axes
    print("[hsv] fitting ridge axes", flush=True)
    axes = {}
    for j, name in enumerate(("hue", "sat", "val")):
        y = hsv[fit_idx, j].astype(np.float32)
        if name == "hue":
            # hue is periodic. Regress sin/cos separately then take
            # the dominant of the two as the axis (the one with larger
            # variance explained).  This avoids the wrap-around problem.
            y_sin = np.sin(2 * np.pi * y).astype(np.float32)
            y_cos = np.cos(2 * np.pi * y).astype(np.float32)
            w_sin = ridge_axis(X_fit, y_sin)
            w_cos = ridge_axis(X_fit, y_cos)
            # Combine: take w_cos by default (red <-> cyan axis is the
            # most distinctive hue direction in xkcd's distribution).
            # Save sin separately for diagnostics.
            axes["axis_hue"] = w_cos
            axes["axis_hue_sin"] = w_sin
        else:
            axes[f"axis_{name}"] = ridge_axis(X_fit, y)

    np.savez(
        HERE / "hsv_axes_L40.npz",
        axis_hue=axes["axis_hue"], axis_hue_sin=axes["axis_hue_sin"],
        axis_sat=axes["axis_sat"], axis_val=axes["axis_val"],
    )
    print(f"[hsv] wrote {HERE / 'hsv_axes_L40.npz'}", flush=True)

    # U_3d axes from cached PCA basis
    print("[pca] loading cached K=64 basis", flush=True)
    pc = load_pc_basis(K=64)
    Vt = pc["Vt"]  # (K, D)
    evr = pc["evr"][:3]
    # normalise to unit norm (PCs already are, but make explicit)
    pcs = Vt[:3] / np.linalg.norm(Vt[:3], axis=1, keepdims=True)
    np.savez(
        HERE / "u3d_axes_L40.npz",
        axis_pc1=pcs[0], axis_pc2=pcs[1], axis_pc3=pcs[2], evr=evr,
    )
    print(f"[pca] wrote {HERE / 'u3d_axes_L40.npz'}  evr={evr}", flush=True)

    # Concept axes (difference of means)
    print("[concept] fitting axes", flush=True)
    # Use a streaming mean per concept to avoid loading whole X into RAM.
    global_mu = np.zeros(D, dtype=np.float64)
    counts_global = 0
    concept_sums = {k: np.zeros(D, dtype=np.float64) for k in CONCEPTS}
    concept_counts = {k: 0 for k in CONCEPTS}
    chunk = 1024
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        block = np.asarray(X[start:end], dtype=np.float32)
        global_mu += block.sum(axis=0)
        counts_global += block.shape[0]
        for cname, proto in CONCEPTS.items():
            m = concept_mask(hsv[start:end], proto)
            if m.any():
                concept_sums[cname] += block[m].sum(axis=0)
                concept_counts[cname] += int(m.sum())
    global_mu /= counts_global
    concept_axes = {}
    for cname in CONCEPTS:
        if concept_counts[cname] == 0:
            concept_axes[cname] = np.zeros(D, dtype=np.float32)
            continue
        mu_c = concept_sums[cname] / concept_counts[cname]
        diff = (mu_c - global_mu).astype(np.float32)
        diff /= np.linalg.norm(diff) + 1e-12
        concept_axes[cname] = diff
    np.savez(
        HERE / "concept_axes_L40.npz",
        axis_red=concept_axes["red"], axis_blue=concept_axes["blue"],
        axis_green=concept_axes["green"], axis_achromatic=concept_axes["achromatic"],
        counts=np.array([concept_counts[c] for c in CONCEPTS]),
    )
    print(f"[concept] wrote {HERE / 'concept_axes_L40.npz'} counts={concept_counts}",
          flush=True)

    # Validation: project X onto every axis (streaming), compute AUC + hist.
    print("[validate] projecting X onto all axes (streaming)", flush=True)
    axis_list: list[tuple[str, np.ndarray, float]] = [
        ("hue",   axes["axis_hue"],     0.0),   # cos axis ~ red/cyan
        ("sat",   axes["axis_sat"],     0.0),   # AUC vs hue: meaningless,
        ("val",   axes["axis_val"],     0.0),   # meaningless too -- we just
                                                # validate hue + concepts.
        ("pc1",   pcs[0],               0.0),
        ("pc2",   pcs[1],               0.33),  # green direction
        ("pc3",   pcs[2],               0.62),  # blue direction
        ("red",   concept_axes["red"],         0.00),
        ("blue",  concept_axes["blue"],        0.62),
        ("green", concept_axes["green"],       0.33),
        ("achromatic", concept_axes["achromatic"], 0.00),  # AUC vs hue: meaningless
    ]
    projections = {name: np.zeros(N, dtype=np.float32) for name, _, _ in axis_list}
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        block = np.asarray(X[start:end], dtype=np.float32)
        for name, w, _ in axis_list:
            projections[name][start:end] = block @ w.astype(np.float32)

    report = {"axes": {}, "n_rows": int(N), "d": int(D)}
    for name, w, target in axis_list:
        auc = auc_top_decile_vs_hue(projections[name], hsv[:, 0], target)
        report["axes"][name] = {
            "auc_hue_within_30deg": auc,
            "target_hue": target,
            "proj_mean": float(projections[name].mean()),
            "proj_std": float(projections[name].std()),
            "norm": float(np.linalg.norm(w)),
        }
        print(f"  axis={name:11s} AUC={auc:.3f} target_hue={target:.2f}", flush=True)
    (HERE / "steering_vectors_offline_report.json").write_text(
        json.dumps(report, indent=2)
    )

    # Histograms (8 panels): hue, sat, val, pc1, pc2, pc3, red, blue
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        fig, axarr = plt.subplots(2, 4, figsize=(16, 7))
        panel_names = ["hue", "sat", "val", "pc1", "pc2", "pc3", "red", "blue"]
        # Colour each row by its true hue (full HSV at high chroma).
        colors_per_row = hsv_to_rgb(np.stack([
            hsv[:, 0], np.clip(hsv[:, 1] + 0.3, 0, 1),
            np.clip(hsv[:, 2] + 0.2, 0, 1),
        ], axis=1))
        for k, pname in enumerate(panel_names):
            ax = axarr[k // 4, k % 4]
            proj = projections[pname]
            # Bin by proj, average true RGB per bin to colour bars.
            nb = 40
            edges = np.linspace(proj.min(), proj.max(), nb + 1)
            idx = np.clip(np.digitize(proj, edges) - 1, 0, nb - 1)
            counts = np.zeros(nb)
            mean_rgb = np.zeros((nb, 3))
            for b in range(nb):
                sel = idx == b
                counts[b] = sel.sum()
                if counts[b] > 0:
                    mean_rgb[b] = colors_per_row[sel].mean(axis=0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for b in range(nb):
                ax.bar(centers[b], counts[b], width=(edges[1] - edges[0]) * 0.9,
                       color=tuple(mean_rgb[b]))
            ax.set_title(f"axis={pname}  AUC={report['axes'][pname]['auc_hue_within_30deg']:.3f}")
            ax.set_xlabel("projection")
            ax.set_ylabel("count")
        fig.tight_layout()
        out_png = HERE / "steering_vectors_offline_report.png"
        fig.savefig(out_png, dpi=110)
        print(f"[validate] wrote {out_png}", flush=True)
    except Exception as e:
        print(f"[validate] plotting skipped: {e!r}", flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
