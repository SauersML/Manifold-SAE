"""
auto_84: spec-by-spec redundancy across the GAM zoo.

For each of the 108 specs, take its 64-dim per-PC R^2 profile, then compute
the pairwise Pearson correlation across all spec pairs. Cluster (hierarchical)
to reveal whether the unsupervised / supervised / kNN / manifold families all
agree on WHICH PCs are predictable, or whether they decompose into discordant
"information clusters".

This is the per-PC analog of the "do specs agree per-color" question
(per-color residuals were not saved in results.json). It still answers:
how redundant is the zoo? do unsupervised methods rediscover the same
PC importance pattern that supervised color spaces find?
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from pathlib import Path

ROOT = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
d = json.load(open(ROOT / "results.json"))
specs = d["per_layer"]["L40"]["specs"]
evr = np.array(d["per_layer"]["L40"]["explained_variance_ratio_topK"])

names = [n for n in specs.keys()
         if n != "L_const_mean" and "r2_per_pc_mean" in specs[n]
         and specs[n]["r2_per_pc_mean"] is not None
         and len(specs[n]["r2_per_pc_mean"]) > 0]
print(f"kept {len(names)} / {len(specs)} specs with per-PC R^2")
M = np.array([specs[n]["r2_per_pc_mean"] for n in names])  # (S, 64)
S, K = M.shape
print(f"S={S} specs, K={K} PCs")

# Pearson corr across specs (rows)
Mc = M - M.mean(axis=1, keepdims=True)
norms = np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12
Mn = Mc / norms
C = Mn @ Mn.T  # (S,S)

# hierarchical cluster on 1-C
Z = linkage(1 - C[np.triu_indices(S, k=1)], method="average")
order = leaves_list(Z)
C_ord = C[order][:, order]
names_ord = [names[i] for i in order]

# family color tags
def fam(n):
    if n.startswith("U_pca") or n.startswith("U_nmf") or n.startswith("U_kmeans") or n.startswith("U_centroid") or n.startswith("U_loop") or n.startswith("U_3d") or n in ("U_1d","U_2d","U_4d","U_5d","U_6d","U_8d"):
        return "unsup"
    if n.startswith("N_knn"):
        return "knn"
    if n.startswith("M_"):
        return "manifold"
    return "linear/poly"
fam_colors = {"unsup":"#377eb8","knn":"#984ea3","manifold":"#4daf4a","linear/poly":"#e41a1c"}
row_cols = [fam_colors[fam(n)] for n in names_ord]

# best-PC for each spec (which PC dominates its R^2)
best_pc = M.argmax(axis=1)
mean_r2 = M.mean(axis=1)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.04], height_ratios=[0.04, 1],
                       hspace=0.02, wspace=0.02)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.imshow(np.array(row_cols).reshape(1, -1).repeat(3, axis=0).reshape(1,-1,3) if False else np.zeros((1,S)),
              aspect="auto", cmap="gray", alpha=0)
ax_top.set_xlim(-0.5, S-0.5)
for i, c in enumerate(row_cols):
    ax_top.axvspan(i-0.5, i+0.5, color=c)
ax_top.set_xticks([]); ax_top.set_yticks([])

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(S)); ax.set_xticklabels(names_ord, rotation=90, fontsize=5)
ax.set_yticks(range(S)); ax.set_yticklabels(names_ord, fontsize=5)
for tl, c in zip(ax.get_xticklabels(), row_cols): tl.set_color(c)
for tl, c in zip(ax.get_yticklabels(), row_cols): tl.set_color(c)
ax.set_title(f"Spec-by-spec Pearson correlation of 64-PC R^2 profiles "
             f"(S={S}, hierarchical-clustered)\n"
             "Red = same PCs predictable; Blue = anti-correlated PC importance",
             fontsize=11)

cax = fig.add_subplot(gs[1, 1])
plt.colorbar(im, cax=cax)

# legend
handles = [plt.Rectangle((0,0),1,1, color=v) for v in fam_colors.values()]
ax.legend(handles, list(fam_colors.keys()), loc="lower left",
          bbox_to_anchor=(0, -0.18), ncol=4, fontsize=9, frameon=False)

out = ROOT / "auto_84.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)

# Print summary
mean_off = (C.sum() - S) / (S*(S-1))
print(f"mean off-diagonal corr: {mean_off:.3f}")
# Are unsup specs as agreeing as linear specs?
def block(tag):
    idx = [i for i,n in enumerate(names) if fam(n)==tag]
    sub = C[np.ix_(idx, idx)]
    return (sub.sum() - len(idx)) / (len(idx)*(len(idx)-1))
for t in fam_colors:
    print(f"within-{t} mean corr: {block(t):.3f}")
# cross-family
def cross(a,b):
    ia = [i for i,n in enumerate(names) if fam(n)==a]
    ib = [i for i,n in enumerate(names) if fam(n)==b]
    return C[np.ix_(ia,ib)].mean()
print(f"unsup x linear/poly: {cross('unsup','linear/poly'):.3f}")
print(f"unsup x knn:         {cross('unsup','knn'):.3f}")
print(f"manifold x linear:   {cross('manifold','linear/poly'):.3f}")

# Which spec is the "median informant" (highest mean corr with all others)
agreement = (C.sum(axis=1) - 1) / (S - 1)
ord2 = np.argsort(-agreement)
print("\nTop-10 most-agreeing specs (centroid of the zoo):")
for i in ord2[:10]:
    print(f"  {names[i]:35s}  mean_corr={agreement[i]:.3f}  mean_R^2={mean_r2[i]:.3f}")
print("\nBottom-10 most-disagreeing specs (outliers):")
for i in ord2[-10:]:
    print(f"  {names[i]:35s}  mean_corr={agreement[i]:.3f}  mean_R^2={mean_r2[i]:.3f}")
