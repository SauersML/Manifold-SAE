"""auto_15: Procrustes alignment of U_3d latent T to the canonical RGB cube.

Question (ii): the unsupervised d=3 GAM fit yields T in R^3 per color. How well
does T align to the canonical sRGB cube via similarity transform (rotation +
isotropic scale + translation)? We compute the orthogonal Procrustes solution
and report the residual error, both globally and per-color (so we can flag the
worst-aligned colors and check whether failure correlates with luminance, sat,
or hue).

Also reports a control: HSV-periodic 4D target (cos h, sin h, sat, val) reduced
to 3D via PCA, to test whether RGB or HSV is the better fit for T.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RES = json.load(open(RUN_DIR / "results.json"))

T = np.array(RES["per_layer"]["L40"]["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
ax = RES["color_axes_per_color_index"]
R_ = np.array(ax["R"]); G_ = np.array(ax["G"]); B_ = np.array(ax["B"])
H_ = np.array(ax["hue"]); S_ = np.array(ax["sat"]); V_ = np.array(ax["value"])
L_ = np.array(ax["luminance"])
RGB = np.stack([R_, G_, B_], axis=1)              # (C, 3) in [0,1]
N = T.shape[0]
print(f"[data] T={T.shape}  RGB={RGB.shape}  N={N}")


def procrustes_similarity(A: np.ndarray, B: np.ndarray):
    """Find s, R, t s.t. s*A@R + t ~= B (orthogonal R). Returns aligned A and stats."""
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    R, scale_sum = orthogonal_procrustes(A0, B0)  # rotates A0 to B0; scale_sum=trace(...)
    # Optimal isotropic scale (Schönemann / Sibson)
    normA2 = (A0 ** 2).sum()
    s = scale_sum / normA2
    A_aligned = s * A0 @ R + muB
    resid = A_aligned - B
    per_color_err = np.linalg.norm(resid, axis=1)
    ss_resid = (resid ** 2).sum()
    ss_total = ((B - muB) ** 2).sum()
    r2 = 1.0 - ss_resid / ss_total
    return A_aligned, dict(s=float(s), r2=float(r2),
                           per_color_err=per_color_err,
                           rmse=float(np.sqrt(ss_resid / N)))


# --- RGB alignment ---
T_to_rgb, st_rgb = procrustes_similarity(T, RGB)
print(f"[RGB ] r2={st_rgb['r2']:.4f}  rmse={st_rgb['rmse']:.4f}  scale={st_rgb['s']:.4f}")

# --- HSV-periodic alignment: 4D -> PCA-3 target ---
HSV4 = np.stack([
    np.cos(2 * np.pi * H_), np.sin(2 * np.pi * H_), S_, V_
], axis=1)
HSV3 = PCA(n_components=3).fit_transform(HSV4)
T_to_hsv, st_hsv = procrustes_similarity(T, HSV3)
print(f"[HSV3] r2={st_hsv['r2']:.4f}  rmse={st_hsv['rmse']:.4f}  scale={st_hsv['s']:.4f}")

# Hardest colors under RGB alignment
err = st_rgb["per_color_err"]
order_bad = np.argsort(-err)[:20]
order_good = np.argsort(err)[:10]

# Correlate residual with axes
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

corr_err_axes = {
    "R": spearman(err, R_), "G": spearman(err, G_), "B": spearman(err, B_),
    "hue": spearman(err, H_), "sat": spearman(err, S_),
    "value": spearman(err, V_), "luminance": spearman(err, L_),
}
print("[err~axis spearman]", {k: round(v, 3) for k, v in corr_err_axes.items()})

# Hue-binned residual
hue_bins = np.linspace(0, 1, 13)
hb_centers = 0.5 * (hue_bins[:-1] + hue_bins[1:])
binned_err = []
binned_n = []
for i in range(len(hue_bins) - 1):
    m = (H_ >= hue_bins[i]) & (H_ < hue_bins[i + 1])
    binned_err.append(err[m].mean() if m.sum() else np.nan)
    binned_n.append(int(m.sum()))

# ---------------- Plot ----------------
fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "auto_15 · Procrustes alignment of unsupervised U_3d (L40) to canonical RGB cube",
    fontsize=14, fontweight="bold",
)

# Panel A: 3D scatter of aligned T (colored by true RGB), with target RGB cube vertices.
axA = fig.add_subplot(2, 3, 1, projection="3d")
axA.scatter(T_to_rgb[:, 0], T_to_rgb[:, 1], T_to_rgb[:, 2],
            c=np.clip(RGB, 0, 1), s=12, alpha=0.85, edgecolors="none")
# Cube vertices for reference
verts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
axA.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c="k", s=40, marker="x")
axA.set_xlabel("R"); axA.set_ylabel("G"); axA.set_zlabel("B")
axA.set_title(f"A. Procrustes-aligned T in RGB space\nR²={st_rgb['r2']:.3f}  RMSE={st_rgb['rmse']:.3f}")

# Panel B: RGB residual error histogram + HSV-PCA3 comparison.
axB = fig.add_subplot(2, 3, 2)
axB.hist(err, bins=40, alpha=0.7, label=f"vs RGB  (R²={st_rgb['r2']:.3f})", color="steelblue")
axB.hist(st_hsv["per_color_err"], bins=40, alpha=0.5,
         label=f"vs HSV-PCA3  (R²={st_hsv['r2']:.3f})", color="orange")
axB.set_xlabel("per-color Procrustes residual ‖·‖")
axB.set_ylabel("count")
axB.set_title("B. Residual distribution: RGB vs HSV target")
axB.legend(fontsize=9)
axB.grid(alpha=0.3)

# Panel C: residual vs each axis (scatter, RGB sub-rows). Plot err vs luminance,
# colored by true RGB (gives intuition).
axC = fig.add_subplot(2, 3, 3)
sc = axC.scatter(L_, err, c=np.clip(RGB, 0, 1), s=18, edgecolors="k", linewidths=0.3)
axC.set_xlabel("luminance (0.299R + 0.587G + 0.114B)")
axC.set_ylabel("Procrustes residual")
corr_txt = "  ".join(f"{k}={v:+.2f}" for k, v in corr_err_axes.items())
axC.set_title("C. Residual vs luminance (point=true RGB)")
axC.text(0.02, 0.98, "Spearman(err, axis):\n" + "\n".join(
    f"  {k:10s}{v:+.3f}" for k, v in corr_err_axes.items()
), transform=axC.transAxes, va="top", ha="left", fontsize=8, family="monospace")
axC.grid(alpha=0.3)

# Panel D: hue-binned mean residual (line plot).
axD = fig.add_subplot(2, 3, 4)
axD.bar(hb_centers, binned_err, width=(1 / 12) * 0.9,
        color=[plt.get_cmap("hsv")(h) for h in hb_centers],
        edgecolor="black", linewidth=0.4)
axD.set_xlabel("hue bin centre")
axD.set_ylabel("mean Procrustes residual")
axD.set_title("D. Residual binned by hue (which hues align worst?)")
for x, n in zip(hb_centers, binned_n):
    axD.text(x, axD.get_ylim()[1] * 0.02, f"n={n}", ha="center", fontsize=7, rotation=90)
axD.grid(alpha=0.3, axis="y")

# Panel E: worst-20 swatches (true RGB) annotated with residual.
axE = fig.add_subplot(2, 3, 5)
axE.axis("off")
axE.set_title("E. 20 worst-aligned colors (sRGB swatches)")
for i, ci in enumerate(order_bad):
    row, col = i // 5, i % 5
    rect = plt.Rectangle((col, -row), 0.9, 0.9,
                         facecolor=np.clip(RGB[ci], 0, 1), edgecolor="black")
    axE.add_patch(rect)
    axE.text(col + 0.45, -row - 0.15, f"#{ci}\nerr={err[ci]:.2f}",
             ha="center", va="top", fontsize=7,
             color=("white" if L_[ci] < 0.5 else "black"))
axE.set_xlim(-0.2, 5); axE.set_ylim(-4.4, 1.2); axE.set_aspect("equal")

# Panel F: best-10 swatches.
axF = fig.add_subplot(2, 3, 6)
axF.axis("off")
axF.set_title("F. 10 best-aligned colors")
for i, ci in enumerate(order_good):
    row, col = i // 5, i % 5
    rect = plt.Rectangle((col, -row), 0.9, 0.9,
                         facecolor=np.clip(RGB[ci], 0, 1), edgecolor="black")
    axF.add_patch(rect)
    axF.text(col + 0.45, -row - 0.15, f"#{ci}\nerr={err[ci]:.2f}",
             ha="center", va="top", fontsize=7,
             color=("white" if L_[ci] < 0.5 else "black"))
axF.set_xlim(-0.2, 5); axF.set_ylim(-2.4, 1.2); axF.set_aspect("equal")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = RUN_DIR / "auto_15.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"[save] {out}")

# JSON sidecar
side = {
    "n_colors": int(N),
    "rgb_alignment": {
        "r2": st_rgb["r2"], "rmse": st_rgb["rmse"], "scale": st_rgb["s"],
        "median_err": float(np.median(err)), "max_err": float(err.max()),
    },
    "hsv3_alignment": {
        "r2": st_hsv["r2"], "rmse": st_hsv["rmse"], "scale": st_hsv["s"],
    },
    "spearman_err_vs_axis": corr_err_axes,
    "hue_bin_centers": hb_centers.tolist(),
    "hue_bin_mean_err": [None if np.isnan(x) else float(x) for x in binned_err],
    "hue_bin_n": binned_n,
    "worst20_color_idx": order_bad.tolist(),
    "worst20_err": [float(err[i]) for i in order_bad],
    "best10_color_idx": order_good.tolist(),
    "best10_err": [float(err[i]) for i in order_good],
}
with open(RUN_DIR / "auto_15.json", "w") as f:
    json.dump(side, f, indent=2)
print(f"[save] {RUN_DIR / 'auto_15.json'}")
