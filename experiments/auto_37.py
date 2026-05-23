"""auto_37: Bootstrap 5-fold R^2 CIs for top-10 specs.

Idea (vvvv): the per_fold_r2_macro arrays in results.json give us 5 fold R^2
values per spec. We rank specs by mean and, for the top 10, compute bootstrap
(B=10_000) percentile CIs for the mean R^2 from the 5 folds. Plot the ranked
specs with point estimate + 95% CI whiskers so we can see which "wins" are
actually distinguishable from each other given only 5 folds of evidence.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_37.png"

d = json.load(open(RUN / "results.json"))
specs = d["per_layer"]["L40"]["specs"]

rows = []
n_err = 0
for name, s in specs.items():
    if not isinstance(s, dict) or "per_fold_r2_macro" not in s:
        n_err += 1
        continue
    folds = np.asarray(s["per_fold_r2_macro"], dtype=float)
    if folds.size < 2 or not np.all(np.isfinite(folds)):
        continue
    rows.append((name, folds, float(np.mean(folds))))
print(f"skipped {n_err} errored specs")

rows.sort(key=lambda r: r[2], reverse=True)
top = rows[:10]

rng = np.random.default_rng(0)
B = 10_000
names, means, lo, hi, stds = [], [], [], [], []
for name, folds, m in top:
    n = folds.size
    idx = rng.integers(0, n, size=(B, n))
    boot = folds[idx].mean(axis=1)
    names.append(name)
    means.append(m)
    lo.append(np.percentile(boot, 2.5))
    hi.append(np.percentile(boot, 97.5))
    stds.append(float(folds.std(ddof=1)))

means = np.array(means); lo = np.array(lo); hi = np.array(hi); stds = np.array(stds)
y = np.arange(len(names))[::-1]  # best at top

# Overlap-with-#1 test: does each CI overlap top-1's CI?
top_lo, top_hi = lo[0], hi[0]
overlap = (lo <= top_hi) & (hi >= top_lo)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ca02c" if o else "#d62728" for o in overlap]
colors[0] = "#1f77b4"
ax.errorbar(means, y, xerr=[means - lo, hi - means],
            fmt='o', color='k', ecolor='gray', capsize=4, lw=1.5, ms=7, zorder=3)
for yi, c, m in zip(y, colors, means):
    ax.scatter([m], [yi], s=80, c=c, zorder=4, edgecolor='k', linewidth=0.6)

# also show individual fold points as small dots
for (name, folds, m), yi in zip(top, y):
    ax.scatter(folds, [yi] * folds.size, s=14, c='gray', alpha=0.45, zorder=2)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Macro R² (mean across 5 folds, 95% bootstrap CI; gray dots = folds)")
ax.set_title("auto_37 — Bootstrap 5-fold R² CIs, top-10 specs (L40)\n"
             "blue=#1, green=CI overlaps #1, red=separated from #1 (B=10,000)")
ax.grid(axis='x', alpha=0.3)
ax.axvline(means[0], color='#1f77b4', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"saved -> {OUT}")
print(f"n_specs ranked: {len(rows)}; top mean R²: {means[0]:.4f}")
for n, m, l, h, sd, ov in zip(names, means, lo, hi, stds, overlap):
    print(f"  {n:35s}  R²={m:.4f}  CI=[{l:.4f},{h:.4f}]  fold_sd={sd:.4f}  overlap#1={bool(ov)}")
