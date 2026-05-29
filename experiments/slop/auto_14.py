"""auto_14: template clustering in residual space.

Question (hh): if we remove per-color mean from the L40 residual stream and
average the template-conditional offset across colors, do the 28 templates
form interpretable clusters in the resulting "template-effect" space?

We compute T[t] = mean_c (X[c, t] - mean_t' X[c, t']) over 949 colors,
project to PCA-3 / PCA-2, and visualise + hierarchical cluster the 28
templates. We also report a leave-one-color-out style stability check via
split-half correlation of T across two random halves of the color set.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RES = json.load(open(RUN_DIR / "results.json"))
TEMPLATES = RES["templates"]
N_T = len(TEMPLATES)

X = np.load("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy", mmap_mode="r")
N_C = X.shape[0] // N_T
print(f"[data] X={X.shape}  n_colors={N_C}  n_templates={N_T}")

# Reshape into (C, T, D); use float32. 949*28*7168*4 = ~760 MB — fine.
X_ct = np.array(X[: N_C * N_T], dtype=np.float32).reshape(N_C, N_T, -1)
print(f"[reshape] X_ct={X_ct.shape}")

# Remove per-color mean across templates -> pure template-effect residual.
per_color_mean = X_ct.mean(axis=1, keepdims=True)  # (C, 1, D)
R = X_ct - per_color_mean  # (C, T, D)
T_eff = R.mean(axis=0)  # (T, D)  template-conditional mean residual
print(f"[T_eff] shape={T_eff.shape}  ||T_eff||_F={np.linalg.norm(T_eff):.2f}")

# Split-half stability across colors.
rng = np.random.default_rng(0)
perm = rng.permutation(N_C)
half = N_C // 2
T_a = R[perm[:half]].mean(0)
T_b = R[perm[half : 2 * half]].mean(0)
# Per-template cosine between two halves
cos = (T_a * T_b).sum(1) / (np.linalg.norm(T_a, axis=1) * np.linalg.norm(T_b, axis=1) + 1e-12)
print(f"[stability] median split-half cos={np.median(cos):.3f}  min={cos.min():.3f}")

# PCA-3 over the 28 template vectors.
pca = PCA(n_components=min(10, N_T - 1)).fit(T_eff)
T3 = pca.transform(T_eff)[:, :3]
evr = pca.explained_variance_ratio_
print(f"[pca] EVR top10={np.round(evr,3)}")

# Hierarchical clustering on cosine distance over T_eff.
T_norm = T_eff / (np.linalg.norm(T_eff, axis=1, keepdims=True) + 1e-12)
D_cos = 1 - T_norm @ T_norm.T
np.fill_diagonal(D_cos, 0.0)
Z = linkage(squareform(D_cos, checks=False), method="average")
K_CLUSTERS = 5
labels = fcluster(Z, t=K_CLUSTERS, criterion="maxclust")
print(f"[cluster] {K_CLUSTERS} clusters; sizes={np.bincount(labels)[1:]}")

# Short labels for templates (first 4 informative words).
def short(s: str, n: int = 28) -> str:
    s = s.replace("{x}", "X")
    return s if len(s) <= n else s[: n - 1] + "…"

short_labels = [f"{i:02d}: {short(t, 26)}" for i, t in enumerate(TEMPLATES)]

# ---------------- Plot ----------------
fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "auto_14 · Template clustering in L40 residual space (per-color mean removed)",
    fontsize=14, fontweight="bold",
)

cmap = plt.get_cmap("tab10")
cluster_colors = [cmap((l - 1) % 10) for l in labels]

# Panel A: PCA-2 scatter of template effects, colored by cluster.
axA = fig.add_subplot(2, 2, 1)
for i in range(N_T):
    axA.scatter(T3[i, 0], T3[i, 1], s=120, color=cluster_colors[i],
                edgecolors="black", linewidths=0.6, zorder=3)
    axA.annotate(str(i), (T3[i, 0], T3[i, 1]), fontsize=8,
                 ha="center", va="center", zorder=4)
axA.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
axA.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
axA.set_title("A. Templates in PC1-PC2 of template-effect space")
axA.grid(alpha=0.3)

# Panel B: PC1-PC3.
axB = fig.add_subplot(2, 2, 2)
for i in range(N_T):
    axB.scatter(T3[i, 0], T3[i, 2], s=120, color=cluster_colors[i],
                edgecolors="black", linewidths=0.6, zorder=3)
    axB.annotate(str(i), (T3[i, 0], T3[i, 2]), fontsize=8,
                 ha="center", va="center", zorder=4)
axB.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
axB.set_ylabel(f"PC3 ({evr[2]*100:.1f}%)")
axB.set_title("B. Templates in PC1-PC3")
axB.grid(alpha=0.3)

# Panel C: cosine-distance heatmap, reordered by cluster.
order = np.argsort(labels)
D_ord = D_cos[np.ix_(order, order)]
axC = fig.add_subplot(2, 2, 3)
im = axC.imshow(D_ord, cmap="viridis", vmin=0, vmax=D_cos.max())
axC.set_xticks(range(N_T)); axC.set_yticks(range(N_T))
axC.set_xticklabels([str(o) for o in order], fontsize=7, rotation=90)
axC.set_yticklabels([str(o) for o in order], fontsize=7)
axC.set_title("C. Cosine-distance matrix (reordered by cluster)")
plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="1 − cos")
# Draw cluster block boundaries.
lab_sorted = labels[order]
boundaries = np.where(np.diff(lab_sorted) != 0)[0] + 0.5
for b in boundaries:
    axC.axhline(b, color="white", lw=1.2)
    axC.axvline(b, color="white", lw=1.2)

# Panel D: legend table of templates with cluster id & split-half cos stability.
axD = fig.add_subplot(2, 2, 4)
axD.axis("off")
# Build text lines grouped by cluster.
lines = [f"Split-half cosine stability across {half} colors: "
         f"median={np.median(cos):.3f}, min={cos.min():.3f}, max={cos.max():.3f}",
         f"PCA EVR (top-3) = {evr[:3].sum()*100:.1f}%   "
         f"||T_eff||_F = {np.linalg.norm(T_eff):.1f}",
         ""]
for k in range(1, K_CLUSTERS + 1):
    members = [i for i in range(N_T) if labels[i] == k]
    color_hex = "#{:02x}{:02x}{:02x}".format(*(int(255*c) for c in cmap((k-1) % 10)[:3]))
    lines.append(f"Cluster {k}  (n={len(members)})  [{color_hex}]")
    for i in members:
        lines.append(f"   {i:02d}  cos_stab={cos[i]:.2f}  {short(TEMPLATES[i], 56)}")
    lines.append("")
axD.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=7.2,
         va="top", ha="left", transform=axD.transAxes)
axD.set_title("D. Cluster membership & split-half stability per template")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = RUN_DIR / "auto_14.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"[save] {out}")

# JSON sidecar.
side = {
    "n_colors_used": int(N_C),
    "n_templates": int(N_T),
    "pca_evr_top10": evr.tolist(),
    "split_half_cos_per_template": cos.tolist(),
    "split_half_cos_median": float(np.median(cos)),
    "cluster_labels": labels.tolist(),
    "k_clusters": K_CLUSTERS,
    "cluster_members": {
        str(k): [i for i in range(N_T) if labels[i] == k]
        for k in range(1, K_CLUSTERS + 1)
    },
    "T_eff_norms": np.linalg.norm(T_eff, axis=1).tolist(),
}
with open(RUN_DIR / "auto_14.json", "w") as f:
    json.dump(side, f, indent=2)
print(f"[save] {RUN_DIR / 'auto_14.json'}")
