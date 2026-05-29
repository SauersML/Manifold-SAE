"""auto_20: t-SNE of per-color centroids in cogito L40 residual space (idea qq).

Question
--------
PCA / Isomap / Procrustes (auto_01, auto_06, auto_17) all live in
*linear* projections of the centroid manifold. What does a non-linear
embedding say? Does t-SNE recover a recognizable color wheel from the
949-color × 7168-D centroid matrix, with hue smoothly arranged around
a ring and saturation/luminance encoded radially or as the second axis?

We compute per-color centroids (mean across the 28 templates), z-score
per feature (matching the GAM run's preprocessing), reduce to 50 PCs
to denoise, then run t-SNE (perplexity 30). We render the 2-D embedding
three ways: (a) colored by true RGB, (b) by hue, (c) by luminance.

Diagnostic: how monotone is hue along the embedding's angular
coordinate? We fit centre = embedding mean, compute polar angle for
each point, and report Spearman rho(hue, angle) (mod 2π — searching
over rotational offsets) plus mean-circular-distance.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_20.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_20.json"
OUT_PNG = OUT_DIR / "auto_20.png"
N_TEMPLATES = 28
N_PCS_DENOISE = 50
PERPLEXITY = 30
SEED = 0


def load_xkcd_rgb(n: int) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[0]
        hex_code = parts[1].lstrip("#")
        r = int(hex_code[0:2], 16) / 255.0
        g = int(hex_code[2:4], 16) / 255.0
        b = int(hex_code[4:6], 16) / 255.0
        names.append(name)
        rgb.append((r, g, b))
        if len(names) >= n:
            break
    return names, np.array(rgb, dtype=np.float64)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb (...,3) in [0,1] -> hsv (...,3) in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    df = mx - mn
    h = np.zeros_like(mx)
    safe = df > 1e-12
    rmax = safe & (mx == r)
    gmax = safe & (mx == g)
    bmax = safe & (mx == b)
    h[rmax] = ((g[rmax] - b[rmax]) / df[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / df[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / df[bmax]) + 4
    h = (h / 6.0) % 1.0
    s = np.where(mx > 0, df / np.where(mx > 0, mx, 1), 0.0)
    return np.stack([h, s, mx], axis=-1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float32)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    print(f"[load] X={X.shape}  n_colors={n_colors}  templates={N_TEMPLATES}", flush=True)

    names, rgb = load_xkcd_rgb(n_colors)
    assert rgb.shape[0] == n_colors, (rgb.shape, n_colors)
    hsv = rgb_to_hsv(rgb)
    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    # Per-color centroids
    centroids = X.reshape(n_colors, N_TEMPLATES, D).mean(axis=1).astype(np.float64)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma

    # PCA denoise to 50 dims (matches typical t-SNE pre-step)
    pca = PCA(n_components=N_PCS_DENOISE, random_state=SEED)
    Z = pca.fit_transform(Cn)
    print(f"[pca] {Z.shape} cum_evr@{N_PCS_DENOISE} = {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # t-SNE
    print(f"[tsne] perplexity={PERPLEXITY} ...", flush=True)
    ts = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca",
              learning_rate="auto", random_state=SEED, max_iter=2000)
    E = ts.fit_transform(Z)
    print(f"[tsne] kl_divergence={ts.kl_divergence_:.4f}", flush=True)

    # Polar coords of embedding around its centroid
    Ec = E - E.mean(0, keepdims=True)
    angle = np.arctan2(Ec[:, 1], Ec[:, 0]) % (2 * np.pi)
    radius = np.linalg.norm(Ec, axis=1)

    # Hue monotonicity: search for best rotational offset that maximizes
    # |spearman(circular_diff_hue, angle)|. Use unwrapped angle/hue along
    # sort order and a circular shift test.
    hue_rad = 2 * np.pi * hue
    # Rank correlation of cosine-similarity in circular sense: convert
    # both to (cos,sin) and compare via sum |spearman(c1,c2)|+|spearman(s1,s2)|
    rho_cos, _ = spearmanr(np.cos(hue_rad), np.cos(angle))
    rho_sin, _ = spearmanr(np.sin(hue_rad), np.sin(angle))
    circ_spearman = float(0.5 * (abs(rho_cos) + abs(rho_sin)))

    # Best rigid rotation/flip of angle to match hue (closed-form on unit circle)
    best = {"rho": -2.0, "shift": 0.0, "flip": 1}
    for flip in (1, -1):
        a = flip * angle
        for shift in np.linspace(0, 2 * np.pi, 361, endpoint=False):
            d = np.abs(((a + shift) - hue_rad + np.pi) % (2 * np.pi) - np.pi)
            mean_circ = float(d.mean())
            # convert mean to a "score" (smaller better) -> rho = 1 - 2*mean/pi
            rho = 1.0 - 2.0 * mean_circ / np.pi
            if rho > best["rho"]:
                best = {"rho": rho, "shift": float(shift), "flip": int(flip),
                        "mean_circ_dist_rad": mean_circ}

    # Radius vs saturation and luminance
    rho_rad_sat, _ = spearmanr(radius, sat)
    rho_rad_val, _ = spearmanr(radius, val)
    rho_rad_lum, _ = spearmanr(radius, lum)

    out = {
        "n_colors": int(n_colors),
        "n_templates": int(N_TEMPLATES),
        "pca_denoise_dim": N_PCS_DENOISE,
        "pca_cum_evr": float(pca.explained_variance_ratio_.sum()),
        "tsne": {"perplexity": PERPLEXITY, "seed": SEED,
                 "kl_divergence": float(ts.kl_divergence_)},
        "hue_vs_angle_circ_spearman": circ_spearman,
        "best_hue_angle_alignment": best,
        "radius_vs_sat_spearman": float(rho_rad_sat),
        "radius_vs_val_spearman": float(rho_rad_val),
        "radius_vs_lum_spearman": float(rho_rad_lum),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[circ] hue<->angle circ_spearman = {circ_spearman:.3f}", flush=True)
    print(f"[best] rho={best['rho']:.3f}  mean_circ_dist={best['mean_circ_dist_rad']:.3f} rad", flush=True)
    print(f"[radial] r<->sat {rho_rad_sat:+.3f}  r<->val {rho_rad_val:+.3f}  r<->lum {rho_rad_lum:+.3f}", flush=True)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    common_kw = dict(s=18, edgecolor="none")

    ax = axes[0]
    ax.scatter(E[:, 0], E[:, 1], c=rgb, **common_kw)
    ax.set_title("t-SNE of L40 centroids — colored by true RGB")
    ax.set_xlabel("tsne-1"); ax.set_ylabel("tsne-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    hue_rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=1))
    ax.scatter(E[:, 0], E[:, 1], c=hue_rgb, **common_kw)
    ax.set_title(f"colored by hue\ncirc-spearman={circ_spearman:.2f}  best-rot rho={best['rho']:.2f}")
    ax.set_xlabel("tsne-1"); ax.set_ylabel("tsne-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    sc = ax.scatter(E[:, 0], E[:, 1], c=lum, cmap="viridis", **common_kw)
    ax.set_title(f"colored by luminance\nr<->lum={rho_rad_lum:+.2f}  r<->sat={rho_rad_sat:+.2f}")
    ax.set_xlabel("tsne-1"); ax.set_ylabel("tsne-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.04, label="luminance")

    fig.suptitle(
        f"auto_20 — t-SNE of {n_colors} color centroids (cogito L40, perplexity={PERPLEXITY}, PCA-denoise={N_PCS_DENOISE}) — idea qq",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
