"""auto_94: cross-correlation of top-1 PC R^2 vs macro R^2 per spec.

Question: do specs that score well on PC1 also score well overall, or are
some specs "tail-compensators" — weak on PC1 but pulling macro up via PC2..PC64?

Inputs : runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
Output : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_94.png
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT = os.path.join(RUN, "auto_94.png")

d = json.load(open(os.path.join(RUN, "results.json")))
specs = d["per_layer"]["L40"]["specs"]
evr = np.array(d["per_layer"]["L40"]["explained_variance_ratio_topK"])

names, pc1, macro, pc_top4_var_weighted = [], [], [], []
n_err = 0
for name, s in specs.items():
    if "r2_per_pc_mean" not in s:
        n_err += 1
        continue
    r2pc = np.array(s["r2_per_pc_mean"])
    names.append(name)
    pc1.append(r2pc[0])
    macro.append(s["r2_macro_mean"])
    # variance-weighted R^2 over PC2..PC4 (the "near-tail" leaders)
    w = evr[1:4] / evr[1:4].sum()
    pc_top4_var_weighted.append(float((r2pc[1:4] * w).sum()))

names = np.array(names)
pc1 = np.array(pc1)
macro = np.array(macro)
nt = np.array(pc_top4_var_weighted)

pear_r, pear_p = pearsonr(pc1, macro)
spear_r, spear_p = spearmanr(pc1, macro)

# residual from the linear fit macro ~ a + b*pc1; positive resid = tail-compensator
b, a = np.polyfit(pc1, macro, 1)
resid = macro - (a + b * pc1)
order_resid = np.argsort(resid)
top_compensators = order_resid[-6:][::-1]  # high positive residual
top_pc1_driven   = order_resid[:6]           # high negative residual (over-relies on PC1)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

ax = axes[0]
ax.scatter(pc1, macro, s=22, c="#3a6ea5", alpha=0.75, edgecolor="white", linewidth=0.4)
xs = np.linspace(pc1.min(), pc1.max(), 100)
ax.plot(xs, a + b * xs, "--", color="0.4", lw=1, label=f"fit slope={b:.2f}")
for i in top_compensators:
    ax.annotate(names[i], (pc1[i], macro[i]), fontsize=7,
                xytext=(4, 3), textcoords="offset points", color="#b8002e")
for i in top_pc1_driven:
    ax.annotate(names[i], (pc1[i], macro[i]), fontsize=7,
                xytext=(4, -8), textcoords="offset points", color="#1f6b1f")
ax.set_xlabel("R² on PC1 (per-spec)")
ax.set_ylabel("R² macro (mean across 64 PCs)")
ax.set_title(f"per-spec PC1 vs macro  (Pearson r={pear_r:.3f}, p={pear_p:.1e}; "
             f"Spearman ρ={spear_r:.3f})\nred = tail-compensators, "
             "green = PC1-over-reliant")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)

# right panel: top compensators and PC1-driven, sorted by residual
ax = axes[1]
n_show = 10
idx_show = np.concatenate([order_resid[-n_show:][::-1], order_resid[:n_show]])
labels = [names[i] for i in idx_show]
vals = resid[idx_show]
colors = ["#b8002e"] * n_show + ["#1f6b1f"] * n_show
ypos = np.arange(len(idx_show))
ax.barh(ypos, vals, color=colors, alpha=0.85)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color="0.3", lw=0.8)
ax.set_xlabel("residual: macro − linear-fit(PC1)\n"
              "right = tail-compensator, left = PC1-over-reliant")
ax.set_title(f"top {n_show} tail-compensators and PC1-driven specs")
ax.grid(axis="x", alpha=0.3)

fig.suptitle("auto_94: do PC1-strong specs win overall? — cogito L40, 108 specs",
             fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved {OUT}")
print(f"specs ok: {len(names)},  errored-out specs skipped: {n_err}")
print(f"Pearson r(pc1, macro) = {pear_r:.4f}  p = {pear_p:.3e}")
print(f"Spearman ρ            = {spear_r:.4f}  p = {spear_p:.3e}")
print(f"linear fit: macro = {a:.4f} + {b:.4f}*pc1")
print("\nTop tail-compensators (high macro despite weak PC1):")
for i in top_compensators:
    print(f"  {names[i]:35s}  pc1={pc1[i]:.3f}  macro={macro[i]:.3f}  resid={resid[i]:+.3f}")
print("\nTop PC1-over-reliant (high PC1 but weak macro):")
for i in top_pc1_driven:
    print(f"  {names[i]:35s}  pc1={pc1[i]:.3f}  macro={macro[i]:.3f}  resid={resid[i]:+.3f}")
