"""auto_11: spec leaderboard — top-20 GAM specs by macro R^2 with CV error bars.

Idea (o): a flat table view of which feature parametrisations win, sorted by
cross-validated macro R^2 across the top-K PCs. We also annotate (i) the family
(linear / additive / poly / joint), (ii) the color basis (rgb / hsv / lab / etc.),
(iii) per-fold std as an error bar, and (iv) the spread (max-min) across folds
as a secondary marker — touching idea (x) for free.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT = os.path.join(RUN, "auto_11.png")

with open(os.path.join(RUN, "results.json")) as f:
    d = json.load(f)

specs = d["per_layer"]["L40"]["specs"]
rows = []
for name, info in specs.items():
    if "r2_macro_mean" not in info:
        continue
    folds = np.array(info.get("per_fold_r2_macro", []), dtype=float)
    rows.append({
        "name": name,
        "mean": info["r2_macro_mean"],
        "std":  info.get("r2_macro_std", float(folds.std()) if folds.size else 0.0),
        "spread": float(folds.max() - folds.min()) if folds.size else 0.0,
        "folds": folds,
    })

rows.sort(key=lambda r: r["mean"], reverse=True)
TOP = 20
top = rows[:TOP]

# family / basis tagging from name prefix (e.g. L_poly_rgb, NL_joint_lab, ...)
def family(n: str) -> str:
    for tag in ("lin", "add", "poly", "joint", "te", "ti", "tp"):
        if f"_{tag}_" in n or n.endswith(f"_{tag}"):
            return tag
    return "?"
def basis(n: str) -> str:
    for b in ("rgb", "hsv", "lab", "luminance", "hue", "sat", "value", "yuv", "xyz"):
        if n.endswith(b) or f"_{b}_" in n or f"_{b}" in n:
            return b
    return "?"

fam_colors = {
    "lin":   "#1f77b4",
    "add":   "#2ca02c",
    "poly":  "#d62728",
    "joint": "#9467bd",
    "te":    "#ff7f0e",
    "ti":    "#8c564b",
    "tp":    "#17becf",
    "?":     "#666666",
}

names = [r["name"] for r in top]
means = np.array([r["mean"]   for r in top])
stds  = np.array([r["std"]    for r in top])
sprd  = np.array([r["spread"] for r in top])
fams  = [family(n) for n in names]
bs    = [basis(n)  for n in names]
colors = [fam_colors[f] for f in fams]

fig, ax = plt.subplots(figsize=(11, 9))
y = np.arange(TOP)[::-1]  # rank 1 at top
ax.barh(y, means, color=colors, edgecolor="black", linewidth=0.5,
        xerr=stds, ecolor="black", error_kw={"capsize": 3, "lw": 0.8})
# secondary spread marker (max-min across folds) as faint whiskers
for yi, m, sp in zip(y, means, sprd):
    ax.plot([m - sp/2, m + sp/2], [yi, yi], color="white",
            lw=2.5, alpha=0.6, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([f"{i+1:2d}. {n}" for i, n in enumerate(names)],
                   fontsize=9, family="monospace")
ax.set_xlabel("macro R^2 (mean over PCs, 5-fold CV)")
ax.set_title(f"auto_11 — top-{TOP} GAM specs on L40 (n_total={len(rows)})")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, max(means.max() * 1.15, means.max() + stds.max() * 2))

# family legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.5)
           for c in fam_colors.values()]
ax.legend(handles, list(fam_colors.keys()), title="family",
          loc="lower right", fontsize=8, ncol=2)

# annotate values
for yi, m, s in zip(y, means, stds):
    ax.text(m + stds.max() * 0.15, yi, f"{m:.3f} +/- {s:.3f}",
            va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved {OUT}")

print("\n=== auto_11 spec leaderboard (top-20) ===")
print(f"{'rank':>4}  {'name':40s}  {'R2':>7}  {'std':>6}  {'spread':>7}  fam     basis")
for i, r in enumerate(top, 1):
    print(f"{i:>4}  {r['name']:40s}  {r['mean']:7.4f}  {r['std']:6.4f}  "
          f"{r['spread']:7.4f}  {family(r['name']):6s}  {basis(r['name'])}")

# bottom-of-distribution summary
print("\nDistribution stats over all specs:")
all_means = np.array([r["mean"] for r in rows])
print(f"  n={len(rows)}  median={np.median(all_means):.4f}  "
      f"max={all_means.max():.4f}  min={all_means.min():.4f}")
print(f"  top-20 mean={means.mean():.4f}  top-20 floor={means.min():.4f}")
