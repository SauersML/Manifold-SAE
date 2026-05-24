"""
auto_89: First-crossing PC index per spec — recovery-rate heatmap.

For each of the 108 specs in COLOR_MANIFOLD_GAM_COGITO_L40, we measure how
many PCs are needed to reach a given fraction of that spec's max-per-PC R².
This is a "spec parsimony" diagnostic: which specs concentrate their explanatory
power into a few PCs vs. smear it across many?

Plot: heatmap rows=specs (sorted by macro R²), cols=thresholds
{10%, 25%, 50%, 75%, 90%, 95%} of per-spec max R²_k; cell = smallest PC index k
(1-indexed) whose r2_per_pc_mean[k] first exceeds that threshold.

Right panel: scatter of macro R² vs. "PCs-to-50%" for the top-50 specs, colored
by spec family prefix (L_lin / L_poly / L_add / L_joint / L_te / U_*).
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
with open(f"{RUN}/results.json") as f:
    R = json.load(f)

specs_all = R["per_layer"]["L40"]["specs"]
# Keep only specs with per-PC R² vectors
names = [n for n, s in specs_all.items() if "r2_per_pc_mean" in s]
specs = {n: specs_all[n] for n in names}
NPC = max(len(specs[n]["r2_per_pc_mean"]) for n in names)
print(f"{len(names)} specs with r2_per_pc_mean (skipped {len(specs_all)-len(names)})")

# matrix of per-PC R²: rows = specs, cols = PCs (pad short rows with NaN)
M = np.full((len(names), NPC), np.nan)
for i, n in enumerate(names):
    v = specs[n]["r2_per_pc_mean"]
    M[i, :len(v)] = v
macro = np.array([specs[n]["r2_macro_mean"] for n in names])

# per-spec max R²_k (use max over PCs to define "saturation")
maxR = np.nanmax(M, axis=1, keepdims=True)
maxR_safe = np.where(maxR > 1e-9, maxR, 1e-9)

thresholds = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
H = np.full((len(names), len(thresholds)), np.nan)
for i in range(len(names)):
    if maxR[i, 0] <= 1e-9:
        continue
    # cumulative max so threshold-crossing is monotone (first PC index whose
    # running-max R²_k ≥ thresh * spec_max)
    row = np.where(np.isnan(M[i]), -np.inf, M[i])
    running = np.maximum.accumulate(row)
    for j, t in enumerate(thresholds):
        target = t * maxR[i, 0]
        ks = np.where(running >= target)[0]
        if ks.size:
            H[i, j] = ks[0] + 1  # 1-indexed

# Sort specs by macro R² descending
order = np.argsort(-macro)
H_s = H[order]
names_s = [names[i] for i in order]
macro_s = macro[order]

# ---- plot ----
fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.18, 1.4], wspace=0.05)

# Heatmap
ax = fig.add_subplot(gs[0, 0])
cmap = plt.cm.viridis_r
im = ax.imshow(H_s, aspect="auto", cmap=cmap, vmin=1, vmax=NPC,
               interpolation="nearest")
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds], fontsize=10)
ax.set_xlabel("Threshold (fraction of spec's max per-PC R²)", fontsize=11)
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=5.5)
ax.set_title("First-crossing PC index k* (smaller = power concentrated in low PCs)\n"
             "Rows sorted by macro R² (best on top)", fontsize=11)
for i in range(H_s.shape[0]):
    for j in range(H_s.shape[1]):
        v = H_s[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    fontsize=4.5, color="white" if v > NPC/2 else "black")

cax = fig.add_subplot(gs[0, 1])
plt.colorbar(im, cax=cax, label="PC index k* (1-64)")

# Right: macro R² vs. PCs-to-50% for top-50
ax2 = fig.add_subplot(gs[0, 2])
TOP = 50
prefix_re = re.compile(r"^([A-Z]_[a-z]+)")
family_color = {
    "L_lin": "#1f77b4", "L_poly": "#ff7f0e", "L_add": "#2ca02c",
    "L_joint": "#d62728", "L_te": "#9467bd",
    "U_pca": "#8c564b", "U_kmeans": "#e377c2", "U_loop": "#7f7f7f",
    "U_3d": "#bcbd22",
}
def family_of(n):
    m = prefix_re.match(n)
    if not m:
        return "other"
    p = m.group(1)
    # collapse U_* groups
    for k in family_color:
        if n.startswith(k):
            return k
    return p

p50_idx = thresholds.index(0.50)
x = H_s[:TOP, p50_idx]
y = macro_s[:TOP]
fams = [family_of(n) for n in names_s[:TOP]]
for fam in set(fams):
    mask = np.array([f == fam for f in fams])
    ax2.scatter(x[mask], y[mask], c=family_color.get(fam, "#444"),
                s=55, alpha=0.85, edgecolor="k", linewidth=0.4, label=fam)
ax2.set_xlabel("PCs to reach 50% of spec-max R²_k", fontsize=10)
ax2.set_ylabel("macro R²", fontsize=10)
ax2.set_title(f"Parsimony vs. quality (top {TOP} specs)", fontsize=11)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=7, loc="lower right", ncol=2)

# Annotate the most parsimonious & most-powerful specs
best = np.argmax(y)
ax2.annotate(names_s[best], (x[best], y[best]),
             xytext=(6, -10), textcoords="offset points", fontsize=7)
most_parsimonious = np.nanargmin(x + (1 - y))  # combined rank
ax2.annotate(names_s[most_parsimonious],
             (x[most_parsimonious], y[most_parsimonious]),
             xytext=(6, 6), textcoords="offset points", fontsize=7, color="darkred")

plt.suptitle("auto_89 — Spec parsimony: PC-index recovery thresholds (L40, 949 colors)",
             fontsize=13, y=0.995)
out = f"{RUN}/auto_89.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

# also dump a tiny summary
summary = {
    "n_specs": len(names),
    "top10_by_macro": [
        {"spec": names_s[i], "macro_r2": float(macro_s[i]),
         "pc_to_50pct": None if np.isnan(H_s[i, p50_idx]) else int(H_s[i, p50_idx]),
         "pc_to_90pct": None if np.isnan(H_s[i, thresholds.index(0.9)])
                         else int(H_s[i, thresholds.index(0.9)])}
        for i in range(10)
    ],
}
with open(f"{RUN}/auto_89.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
