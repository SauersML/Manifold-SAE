"""auto_64: Pairwise-distance preservation between cogito-T reps and RGB
(idea ppppppp).

For every color c (n_c = 949) we have a cogito L40 representation. We
collapse template variation by taking the mean over the 28 templates,
giving Zc (n_c, K=64) in the published top-K PCA basis. We then ask:
  - Do pairwise distances between colors in cogito-T space track
    pairwise distances between their RGB triplets?

We compute:
  D_T   : (n_c, n_c) Euclidean distance in cogito-T mean PC space
  D_T_z : same on z-scored per-PC means (so each PC contributes equally)
  D_RGB : Euclidean distance between RGB triplets in [0,1]^3
  D_HSV : Euclidean distance in (hue, sat, value) with hue on the
          unit circle (hue -> (cos 2pi h, sin 2pi h)) so it wraps
  D_LUM : |luminance_i - luminance_j|

Then Pearson r between the upper-triangle of D_T (and D_T_z) vs each of
{D_RGB, D_HSV, D_LUM}. Reports the whole-matrix Pearson asked for, plus
Spearman as a robustness check.

We also look at per-(color, template) reps (no template-averaging):
build Z (N=n_c*n_t, K), z-score per PC, and compare its full pairwise
distances to the RGB pairwise distances of the matching colors --- this
sees how much template noise smears the geometry.

Pure numpy + matplotlib. No Gaussian RBF, no Duchon, no kernels.
Uses PCA basis from results.json.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_64.{json,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG  = RUN_DIR / "auto_64.png"
OUT_JSON = RUN_DIR / "auto_64.json"


def upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def pdist_euclid(X: np.ndarray) -> np.ndarray:
    # squared form, clamp to non-negative for numerical safety
    sq = (X * X).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    # rank with ties broken by avg
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {RESULTS}")
    res = json.loads(RESULTS.read_text())
    pl = res["per_layer"]["L40"]
    Vt    = np.asarray(pl["Vt_topK"], dtype=np.float32)            # (K, D)
    mu    = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)
    K, D = Vt.shape

    R = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    G = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    B = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    H = np.asarray(res["color_axes_per_color_index"]["hue"], dtype=np.float64)
    S = np.asarray(res["color_axes_per_color_index"]["sat"], dtype=np.float64)
    V = np.asarray(res["color_axes_per_color_index"]["value"], dtype=np.float64)
    L = np.asarray(res["color_axes_per_color_index"]["luminance"], dtype=np.float64)
    n_t = len(res["templates"])
    n_c = R.shape[0]
    N = n_c * n_t
    print(f"[layout] n_colors={n_c} n_templates={n_t} K={K} D={D}")

    # Project X -> Z (n_c, n_t, K).
    print(f"[load] X {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, (X.shape, N)
    Z = np.zeros((N, K), dtype=np.float32)
    chunk = 4096
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Xc = (Xc - mu) / sigma
        Z[i:j] = Xc @ Vt.T
    Z3 = Z.reshape(n_c, n_t, K).astype(np.float64)
    Zc = Z3.mean(axis=1)                          # (n_c, K) template-mean
    print(f"[project] Z {Z3.shape}  Zc {Zc.shape}")

    # z-score Zc per PC for the "equal-weight" distance.
    Zc_z = (Zc - Zc.mean(0, keepdims=True)) / (Zc.std(0, keepdims=True) + 1e-12)

    # Build the reference geometries.
    rgb = np.stack([R, G, B], axis=1)              # (n_c, 3) in [0,1]
    hsv_wrap = np.stack([
        np.cos(2.0 * np.pi * H), np.sin(2.0 * np.pi * H),
        S, V,
    ], axis=1)                                     # hue wraps

    print("[pdist] D_T  (template-mean PC, raw)")
    D_T   = pdist_euclid(Zc)
    print("[pdist] D_Tz (template-mean PC, z-scored)")
    D_Tz  = pdist_euclid(Zc_z)
    print("[pdist] D_RGB")
    D_RGB = pdist_euclid(rgb)
    print("[pdist] D_HSV (hue on unit circle)")
    D_HSV = pdist_euclid(hsv_wrap)
    print("[pdist] D_LUM")
    D_LUM = np.abs(L[:, None] - L[None, :])

    # Pearson + Spearman on the upper triangle of each pair.
    pairs = {
        "T_vs_RGB":  (D_T,  D_RGB),
        "T_vs_HSV":  (D_T,  D_HSV),
        "T_vs_LUM":  (D_T,  D_LUM),
        "Tz_vs_RGB": (D_Tz, D_RGB),
        "Tz_vs_HSV": (D_Tz, D_HSV),
        "Tz_vs_LUM": (D_Tz, D_LUM),
    }
    corr = {}
    upper_cache = {}
    for name, (A, B_) in pairs.items():
        a = upper_tri(A); b = upper_tri(B_)
        corr[name] = {
            "pearson": pearson(a, b),
            "spearman": spearman(a, b),
            "n_pairs": int(a.size),
        }
        upper_cache[name] = (a, b)
        print(f"[corr] {name:11s} pearson={corr[name]['pearson']:.4f}  "
              f"spearman={corr[name]['spearman']:.4f}")

    # Without template-averaging: per-(c,t) reps z-scored, compare to RGB
    # of matching colors. This is N x N; we subsample to keep it tractable.
    rng = np.random.default_rng(0)
    n_sub = min(2000, N)
    idx = rng.choice(N, size=n_sub, replace=False)
    Zfull_z = (Z.astype(np.float64) - Z.mean(0, keepdims=True)) / (
        Z.std(0, keepdims=True) + 1e-12)
    Zsub = Zfull_z[idx]
    rgb_full = np.repeat(rgb, n_t, axis=0)         # (N, 3)
    rgb_sub = rgb_full[idx]
    D_T_full = pdist_euclid(Zsub)
    D_RGB_full = pdist_euclid(rgb_sub)
    a = upper_tri(D_T_full); b = upper_tri(D_RGB_full)
    corr_full = {
        "pearson":  pearson(a, b),
        "spearman": spearman(a, b),
        "n_pairs":  int(a.size),
        "n_sub":    n_sub,
    }
    print(f"[corr] FULL Zz vs RGB  (n_sub={n_sub})  "
          f"pearson={corr_full['pearson']:.4f}  spearman={corr_full['spearman']:.4f}")

    summary = {
        "n_colors": int(n_c),
        "n_templates": int(n_t),
        "K": int(K),
        "template_mean_pairs": corr,
        "full_per_ct_subsample_vs_rgb": corr_full,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    rgb_face = np.clip(rgb, 0, 1)

    def scatter_pair(ax, a, b, xlab, ylab, title, max_pts=20000):
        if a.size > max_pts:
            sel = rng.choice(a.size, size=max_pts, replace=False)
            a = a[sel]; b = b[sel]
        ax.scatter(a, b, s=2, alpha=0.12, color="#1f77b4", rasterized=True)
        # linear fit line
        slope, intercept = np.polyfit(a, b, 1)
        xs = np.linspace(a.min(), a.max(), 50)
        ax.plot(xs, slope * xs + intercept, "r-", lw=1.2,
                label=f"slope={slope:.3g}")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    # (a) D_T vs D_RGB
    a, b = upper_cache["T_vs_RGB"]
    ax = fig.add_subplot(gs[0, 0])
    scatter_pair(ax, a, b,
                 "cogito-T pairwise distance (template-mean PCs)",
                 "RGB pairwise distance",
                 f"(a) T vs RGB  r={corr['T_vs_RGB']['pearson']:.3f}  "
                 f"rho={corr['T_vs_RGB']['spearman']:.3f}")

    # (b) D_Tz vs D_RGB
    a, b = upper_cache["Tz_vs_RGB"]
    ax = fig.add_subplot(gs[0, 1])
    scatter_pair(ax, a, b,
                 "cogito-T pairwise distance (z-scored PCs)",
                 "RGB pairwise distance",
                 f"(b) Tz vs RGB  r={corr['Tz_vs_RGB']['pearson']:.3f}  "
                 f"rho={corr['Tz_vs_RGB']['spearman']:.3f}")

    # (c) bar chart of pearson + spearman across reference geometries
    ax = fig.add_subplot(gs[0, 2])
    labels = ["T-RGB", "T-HSV", "T-LUM", "Tz-RGB", "Tz-HSV", "Tz-LUM"]
    keys   = ["T_vs_RGB", "T_vs_HSV", "T_vs_LUM",
              "Tz_vs_RGB", "Tz_vs_HSV", "Tz_vs_LUM"]
    pear = [corr[k]["pearson"]  for k in keys]
    spea = [corr[k]["spearman"] for k in keys]
    xs = np.arange(len(labels)); w = 0.4
    ax.bar(xs - w/2, pear, w, color="#1b9e77", label="Pearson")
    ax.bar(xs + w/2, spea, w, color="#d95f02", label="Spearman")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right",
                                          fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("correlation")
    ax.set_title("(c) Distance-preservation across reference geometries")
    ax.set_ylim(min(0, min(pear + spea) - 0.05), 1.0)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (d) hexbin density of T vs RGB (better than scatter for 449k pairs)
    a, b = upper_cache["T_vs_RGB"]
    ax = fig.add_subplot(gs[1, 0])
    hb = ax.hexbin(a, b, gridsize=80, mincnt=1, cmap="magma", bins="log")
    ax.set_xlabel("cogito-T pairwise distance (template-mean PCs)")
    ax.set_ylabel("RGB pairwise distance")
    ax.set_title("(d) Pair density (log) T vs RGB")
    fig.colorbar(hb, ax=ax, shrink=0.9, label="log10(count)")

    # (e) hexbin Tz vs HSV (wrapped)
    a, b = upper_cache["Tz_vs_HSV"]
    ax = fig.add_subplot(gs[1, 1])
    hb = ax.hexbin(a, b, gridsize=80, mincnt=1, cmap="viridis", bins="log")
    ax.set_xlabel("cogito-T pairwise distance (z-scored PCs)")
    ax.set_ylabel("HSV pairwise distance (hue wrapped)")
    ax.set_title(f"(e) Pair density Tz vs HSV  r={corr['Tz_vs_HSV']['pearson']:.3f}")
    fig.colorbar(hb, ax=ax, shrink=0.9, label="log10(count)")

    # (f) full per-(c,t) reps vs RGB on subsample
    ax = fig.add_subplot(gs[1, 2])
    a = upper_tri(D_T_full); b = upper_tri(D_RGB_full)
    sel = rng.choice(a.size, size=min(40000, a.size), replace=False)
    ax.scatter(a[sel], b[sel], s=2, alpha=0.08, color="#666", rasterized=True)
    slope, intercept = np.polyfit(a, b, 1)
    xs = np.linspace(a.min(), a.max(), 50)
    ax.plot(xs, slope * xs + intercept, "r-", lw=1.2,
            label=f"slope={slope:.3g}")
    ax.set_xlabel("per-(c,t) cogito-T pairwise distance (z-scored)")
    ax.set_ylabel("RGB pairwise distance (matched colors)")
    ax.set_title(f"(f) Full reps (no template avg)  "
                 f"r={corr_full['pearson']:.3f}  "
                 f"rho={corr_full['spearman']:.3f}  n_sub={n_sub}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"auto_64 - pairwise distance preservation: cogito-T vs RGB | "
        f"L40 | n_c={n_c} pairs={n_c*(n_c-1)//2}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
