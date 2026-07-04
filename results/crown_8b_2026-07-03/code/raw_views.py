#!/usr/bin/env python3
"""RAW weekday data views: rogue dims, circle plane, heatmap, interactive 3D."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/private/tmp/claude-501/-Users-user/402ec9d9-07ac-42f4-87a0-73d65d949d5b/scratchpad"
z = np.load(f"{SP}/harvest_weekday.npz", allow_pickle=True)
X, T = z["X_last"], z["tmpl_mean"]
D = X - T
n = D.shape[0]
labels = np.tile(np.arange(7), n // 7)   # template-major prompt order (verified in build_prompts)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COLS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold"})

# ---- rogue dims: per-dim scale of the between-day centroids ----
cent = np.stack([D[labels == g].mean(0) for g in range(7)])
rng = cent.max(0) - cent.min(0)
order = np.argsort(rng)[::-1]
print("dims with largest day-to-day range:", order[:6], "ranges:", np.round(rng[order[:6]], 1))
med = np.median(rng)
rogue = order[rng[order] > 50 * med]
print(f"rogue dims (range >50x median {med:.2f}):", rogue)
keep = np.setdiff1d(np.arange(D.shape[1]), rogue)
Dk = D[:, keep]
# drop outlier TEMPLATES (whole-prompt-group demeaned norm >> median)
tmpl = np.arange(n) // 7
tnorm = np.array([np.linalg.norm(Dk[tmpl == t]) for t in range(n // 7)])
bad_t = np.where(tnorm > 2.5 * np.median(tnorm))[0]
print("outlier templates dropped:", bad_t, "norms:", np.round(tnorm, 0))
mask = ~np.isin(tmpl, bad_t)
Dk, labels = Dk[mask], labels[mask]
n = Dk.shape[0]

# ---- FIG 7: the circle plane, raw points projected (post rogue-drop) ----
ck = np.stack([Dk[labels == g].mean(0) for g in range(7)])
cc = ck - ck.mean(0)
_, sv, Wt = np.linalg.svd(cc, full_matrices=False)
P2 = (Dk - ck.mean(0)) @ Wt[:2].T
C2 = (cc) @ Wt[:2].T
fig, ax = plt.subplots(figsize=(8.4, 7.6))
ax.set_aspect("equal")
for g, day in enumerate(days):
    pts = P2[labels == g]
    ax.scatter(pts[:, 0], pts[:, 1], s=55, color=COLS[g], alpha=0.75, label=day)
    ax.annotate(day, C2[g], xytext=(C2[g][0] * 1.18, C2[g][1] * 1.18), ha="center",
                fontsize=12, fontweight="bold", color=COLS[g])
r = np.linalg.norm(C2, axis=1).mean()
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(r * np.cos(th), r * np.sin(th), color="#9a9890", lw=1.6, zorder=0)
ax.set_title("Raw activations: the days of the week form a circle\n"
             "each dot = one prompt's layer-18 vector, 9 prompt wordings per day", loc="left")
ax.set_xlabel("direction in activation space where days differ most (PC1 of the 7 day-averages)")
ax.set_ylabel("second-most day-separating direction (PC2 of the day-averages)")
fig.tight_layout(); fig.savefig(f"{SP}/fig7_raw_circle_plane.png", dpi=200); plt.close(fig)
print(f"between-day top-2 axes carry {(sv[:2]**2).sum()/(sv**2).sum()*100:.1f}% of day structure (rogue-dropped)")

# ---- FIG 8: raw heatmap, most day-informative dims ----
F = []
for j in range(Dk.shape[1]):
    col = Dk[:, j]
    b = np.var([col[labels == g].mean() for g in range(7)])
    w = np.mean([np.var(col[labels == g]) for g in range(7)]) + 1e-9
    F.append(b / w)
topd = np.argsort(F)[::-1][:40]
row_order = np.argsort(labels, kind='stable')
M = Dk[row_order][:, topd]
M = M / np.abs(M).max(0)
fig, ax = plt.subplots(figsize=(11, 6.4))
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
for g in range(1, 7):
    ax.axhline(g * 10 - 0.5, color=INK, lw=0.8)
ax.set_yticks([g * 10 + 4.5 for g in range(7)], days)
ax.set_xlabel("the 40 most day-informative residual dimensions (of 4096), each scaled to ±1")
ax.set_title("The raw numbers: 70 prompts × 40 dimensions, rows grouped by weekday\n"
             "horizontal banding = the same day activates the same dims regardless of prompt wording", loc="left")
fig.colorbar(im, ax=ax, shrink=0.8, label="demeaned activation (scaled)")
fig.tight_layout(); fig.savefig(f"{SP}/fig8_raw_heatmap.png", dpi=200); plt.close(fig)

# ---- interactive 3D (rogue-dropped, between-day axes) ----
import plotly.graph_objects as go
ax3 = Wt[:3]
P3 = (Dk - ck.mean(0)) @ ax3.T
C3 = cc @ ax3.T
r3 = np.linalg.norm(C3[:, :2], axis=1).mean()
ring = np.stack([r3 * np.cos(th), r3 * np.sin(th), np.zeros_like(th)], 1)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                           line=dict(color="#9a9890", width=6), name="fitted circle"))
for g, day in enumerate(days):
    pts = P3[labels == g]
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                               marker=dict(size=6, color=COLS[g]), name=day))
    fig.add_trace(go.Scatter3d(x=[C3[g, 0]], y=[C3[g, 1]], z=[C3[g, 2]], mode="markers+text",
                               marker=dict(size=11, color=COLS[g], symbol="diamond"),
                               text=[day[:3]], textposition="top center", showlegend=False,
                               textfont=dict(size=14, color="#0b0b0b")))
fig.update_layout(title=dict(text="RAW Qwen3-8B L18 weekday activations in 3D — drag to rotate<br>"
        "<sup>70 real prompts · per-template demeaned · 3 rogue-scale dims removed · axes = the week's own top-3 directions</sup>"),
    scene=dict(xaxis_title="week axis 1", yaxis_title="week axis 2", zaxis_title="week axis 3",
               aspectmode="data"), template="plotly_white")
fig.write_html(f"{SP}/weekday_pca3d.html", include_plotlyjs=True, full_html=True)
print("wrote fig7, fig8, weekday_pca3d.html")

# raw csv of the projected points
import csv
with open(f"{SP}/weekday_raw_projected.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["day", "template_idx", "plane_x", "plane_y", "axis3"])
    for i in range(n):
        w.writerow([days[labels[i]], int(np.arange(len(labels))[i]) // 7, f"{P3[i,0]:.4f}", f"{P3[i,1]:.4f}", f"{P3[i,2]:.4f}"])
print("wrote weekday_raw_projected.csv")
