"""auto_10: spec redundancy via correlation matrix of per-PC R^2 across GAM specs.

Idea (v): which specs are redundant? Two specs that "explain the same PCs the same way"
have highly correlated r2_per_pc_mean vectors. We compute Pearson correlation between
all spec pairs over the 64 PCs, hierarchically order them, and show macro-R^2 as a
side bar.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT = os.path.join(RUN, "auto_10.png")

with open(os.path.join(RUN, "results.json")) as f:
    d = json.load(f)

specs = d["per_layer"]["L40"]["specs"]
names_all = [n for n in specs if "r2_per_pc_mean" in specs[n]]
mat_all = np.array([specs[n]["r2_per_pc_mean"] for n in names_all])
# drop specs with zero variance over PCs (corr undefined)
nz = mat_all.std(axis=1) > 1e-10
names = [n for n, ok in zip(names_all, nz) if ok]
mat = mat_all[nz]
macro = np.array([specs[n]["r2_macro_mean"] for n in names])
print(
    f"Using {len(names)} specs "
    f"(skipped {len(specs)-len(names_all)} errored, "
    f"{int((~nz).sum())} constant)"
)

# Pearson correlation across PCs
C = np.corrcoef(mat)

# Hierarchical order (1 - corr as distance)
dist = 1.0 - C
np.fill_diagonal(dist, 0.0)
# condensed
S = len(names)
condensed = dist[np.triu_indices(S, k=1)]
Z = linkage(condensed, method="average")
order = leaves_list(Z)
C_ord = C[np.ix_(order, order)]
names_ord = [names[i] for i in order]
macro_ord = macro[order]

fig, (ax, axb) = plt.subplots(
    1, 2, figsize=(20, 18),
    gridspec_kw={"width_ratios": [10, 1.2], "wspace": 0.03},
)

im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(S)); ax.set_yticks(range(S))
fs = max(4, min(8, int(700 / S)))
ax.set_xticklabels(names_ord, rotation=90, fontsize=fs)
ax.set_yticklabels(names_ord, fontsize=fs)
ax.set_title("Spec redundancy: Pearson corr of per-PC R^2 (L40, hier-ordered)")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("corr over 64 PCs")

# Annotate top redundant non-trivial pairs
pairs = []
for i in range(S):
    for j in range(i+1, S):
        pairs.append((C[i,j], names[i], names[j]))
pairs.sort(reverse=True)
print("Top 5 most redundant spec pairs (Pearson corr of per-PC R^2):")
for c, a, b in pairs[:5]:
    print(f"  {c:.4f}  {a}  <->  {b}")
print("Top 5 most divergent:")
for c, a, b in pairs[-5:]:
    print(f"  {c:.4f}  {a}  <->  {b}")

# side bar: macro R^2
axb.barh(range(S), macro_ord, color="black")
axb.set_yticks(range(S)); axb.set_yticklabels([])
axb.invert_yaxis()
axb.set_xlabel("macro R^2")
axb.set_title("R^2")
axb.set_ylim(ax.get_ylim())

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"\nSaved {OUT}")
print(f"Mean off-diag corr: {C[np.triu_indices(S,k=1)].mean():.3f}")
print(f"Median off-diag corr: {np.median(C[np.triu_indices(S,k=1)]):.3f}")
