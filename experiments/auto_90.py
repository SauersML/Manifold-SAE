"""
auto_90: Fold-stability vs mean-CV-R^2 Pareto per spec.

Novelty vs auto_80-89: none of the prior 80-89 plots inspected
across-fold variability. Each spec stores per_fold_r2_macro over 5 folds;
fold-range (max - min) is a direct proxy for spec instability /
data-hunger / effective-DoF-vs-N. Plotting (mean_R2, fold_range) reveals:
  - Pareto front of high-mean, low-instability specs (the "trust" set)
  - high-mean + high-range specs that look great on the bulk number but
    move >0.02 R^2 across folds (sensitive to which colors are held out)
  - low-mean + low-range floor (consistent under-fitters)
Coloring by spec family (L_/M_/N_/U_) shows which families dominate
each regime.
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
d = json.load(open(os.path.join(RUN, "results.json")))
specs = d["per_layer"]["L40"]["specs"]

rows = []
for name, s in specs.items():
    f = s.get("per_fold_r2_macro")
    if not f or len(f) < 2:
        continue
    f = np.asarray(f, dtype=float)
    rows.append((name, float(f.mean()), float(f.std(ddof=0)),
                 float(f.max() - f.min())))

names  = [r[0] for r in rows]
means  = np.array([r[1] for r in rows])
stds   = np.array([r[2] for r in rows])
ranges = np.array([r[3] for r in rows])

def family(n):
    pref = n.split("_", 1)[0]
    return pref if pref in ("L", "M", "N", "U") else "?"

fam = np.array([family(n) for n in names])
fam_colors = {"L": "#1f77b4", "M": "#2ca02c", "N": "#d62728",
              "U": "#9467bd", "?": "#7f7f7f"}
fam_labels = {"L": "Linear/parametric (L_)", "M": "Manifold (M_)",
              "N": "Nearest-neighbor (N_)", "U": "Unsupervised (U_)"}

# Pareto front: high mean & low range
order = np.argsort(-means)
pareto = []
best_range = np.inf
for i in order:
    if ranges[i] < best_range:
        pareto.append(i)
        best_range = ranges[i]
pareto = np.array(pareto)

fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 2]})
ax, ax2 = axes

main_mask = means > -0.5  # drop catastrophic-failure specs from main view
# Re-pareto using only main_mask specs:
order_m = np.where(main_mask)[0][np.argsort(-means[main_mask])]
pareto2 = []
best_r = np.inf
for i in order_m:
    if ranges[i] < best_r:
        pareto2.append(i)
        best_r = ranges[i]
pareto2 = np.array(pareto2)
po = pareto2[np.argsort(means[pareto2])]

for fkey, col in fam_colors.items():
    mask = (fam == fkey) & main_mask
    if not mask.any():
        continue
    ax.scatter(means[mask], ranges[mask], s=48, c=col, alpha=0.78,
               edgecolors="white", linewidths=0.5, label=fam_labels.get(fkey, fkey))

ax.plot(means[po], ranges[po], "k--", lw=1.3, alpha=0.6,
        label="Pareto front (high R^2, low fold-range)")

to_label = set()
to_label.update(np.argsort(-means)[:6].tolist())
to_label.update(np.argsort(ranges[main_mask])[:3].tolist())
# most-unstable among main_mask
unstable_idx = np.where(main_mask)[0][np.argsort(-ranges[main_mask])[:6]]
to_label.update(unstable_idx.tolist())
to_label.update(pareto2.tolist())
for i in to_label:
    if not main_mask[i]:
        continue
    ax.annotate(names[i], (means[i], ranges[i]),
                fontsize=7, alpha=0.85,
                xytext=(4, 3), textcoords="offset points")

ax.set_xlabel("mean CV R^2 (5-fold)")
ax.set_ylabel("fold-range  max(R^2) - min(R^2)  across 5 folds")
ax.set_title("Main view (specs with mean R^2 > -0.5)\n"
             "low+left = consistent under-fit | high+low = trustworthy | high+high = unstable")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

# Companion: per-fold trace for the most-interesting specs (top-5 mean, top-3 unstable, 2 stable mid)
sel_names = []
for i in np.argsort(-means)[:5]:
    sel_names.append(names[i])
for i in unstable_idx[:3]:
    if names[i] not in sel_names: sel_names.append(names[i])
# 2 mid-R^2 stable picks
mid_mask = (means > 0.15) & (means < 0.5) & main_mask
mid_idx = np.where(mid_mask)[0][np.argsort(ranges[mid_mask])[:2]]
for i in mid_idx:
    if names[i] not in sel_names: sel_names.append(names[i])

cmap = plt.cm.viridis(np.linspace(0, 0.95, len(sel_names)))
for ci, nm in enumerate(sel_names):
    f = np.asarray(specs[nm]["per_fold_r2_macro"])
    ax2.plot(range(1, len(f)+1), f, "-o", color=cmap[ci], lw=1.2, ms=4,
             label=nm, alpha=0.9)
ax2.set_xlabel("fold index")
ax2.set_ylabel("CV R^2 on that fold")
ax2.set_title("Per-fold R^2 trace\n(top-mean / most-unstable / mid-stable)")
ax2.set_xticks(range(1, 6))
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7, loc="lower right", framealpha=0.9)

fig.suptitle("auto_90: Fold-stability vs mean CV R^2 per spec  (cogito L40, 949 colors)",
             fontsize=12, y=1.00)

out = os.path.join(RUN, "auto_90.png")
plt.tight_layout()
plt.savefig(out, dpi=140)
print(f"saved {out}")

# Stdout summary
ratio = ranges / np.maximum(means, 1e-6)
print(f"\nn_specs analyzed: {len(names)}")
print(f"median mean R^2: {np.median(means):.4f}   median range: {np.median(ranges):.4f}")
print(f"\nTop-5 mean R^2 (with fold-range):")
for i in np.argsort(-means)[:5]:
    print(f"  {names[i]:36s} mean={means[i]:.4f}  range={ranges[i]:.4f}  std={stds[i]:.4f}")
print(f"\nMost-unstable 5 (largest fold-range):")
for i in np.argsort(-ranges)[:5]:
    print(f"  {names[i]:36s} mean={means[i]:.4f}  range={ranges[i]:.4f}  ratio={ratio[i]:.3f}")
print(f"\nMost-stable 5 (smallest fold-range, mean>0.05):")
mask = means > 0.05
idxs = np.where(mask)[0]
for i in idxs[np.argsort(ranges[idxs])][:5]:
    print(f"  {names[i]:36s} mean={means[i]:.4f}  range={ranges[i]:.4f}")
print(f"\nPareto front ({len(pareto)} specs):")
for i in po[::-1]:
    print(f"  {names[i]:36s} mean={means[i]:.4f}  range={ranges[i]:.4f}")
