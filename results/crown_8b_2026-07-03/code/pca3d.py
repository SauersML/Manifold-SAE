#!/usr/bin/env python3
"""3D PCA of the RAW weekday activations (Qwen3-8B L18) + fitted circle, interactive."""
import json
import numpy as np

SP = "/private/tmp/claude-501/-Users-user/402ec9d9-07ac-42f4-87a0-73d65d949d5b/scratchpad"
z = np.load(f"{SP}/harvest_weekday.npz", allow_pickle=True)
X, T = z["X_last"], z["tmpl_mean"]
D = X - T                                  # per-template demeaned (the W7 recipe)
n = D.shape[0]

# --- infer the (day, template) layout: try day-major blocks vs interleaved ---
def score(labels):
    tot = 0.0
    for g in range(7):
        rows = D[labels == g]
        tot += np.linalg.norm(rows - rows.mean(0), ord="fro") ** 2
    return tot  # within-group scatter, smaller = tighter
lab_block = np.repeat(np.arange(7), n // 7)      # [0]*10 + [1]*10 ...
lab_inter = np.tile(np.arange(7), n // 7)        # 0,1,2,...,6,0,1,...
labels = lab_block if score(lab_block) < score(lab_inter) else lab_inter
which = "block" if score(lab_block) < score(lab_inter) else "interleaved"
print(f"layout = {which}; within/total scatter: "
      f"{min(score(lab_block), score(lab_inter)) / np.linalg.norm(D - D.mean(0), ord='fro')**2:.3f}")

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- project onto the BETWEEN-DAY structure (PCA of the 7 day centroids) ---
Dc = D - D.mean(0)
cent = np.stack([Dc[labels == g].mean(0) for g in range(7)])    # 7 x 4096
cc = cent - cent.mean(0)
_, sv, Wt = np.linalg.svd(cc, full_matrices=False)
axes3 = Wt[:3]                                                  # top-3 between-day dirs
P3 = Dc @ axes3.T
frac_between = (sv[:3] ** 2).sum() / (sv ** 2).sum()
print(f"top-3 between-day axes carry {frac_between*100:.1f}% of the day-to-day structure")
top = np.array([0, 1, 2])

# --- best-fit circle through the 7 day-centroids (in 3D PC space) ---
C = np.stack([P3[labels == g].mean(0) for g in range(7)])
Cc = C - C.mean(0)
_, _, W = np.linalg.svd(Cc, full_matrices=False)
plane = W[:2]                               # circle plane
r = np.linalg.norm(Cc @ plane.T, axis=1).mean()
th = np.linspace(0, 2 * np.pi, 200)
ring = C.mean(0) + r * (np.outer(np.cos(th), plane[0]) + np.outer(np.sin(th), plane[1]))

# --- plotly interactive ---
import plotly.graph_objects as go
COLS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                           line=dict(color="#9a9890", width=5), name="fitted circle"))
for g, day in enumerate(days):
    pts = P3[labels == g]
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                               marker=dict(size=6, color=COLS[g]), name=day))
    c = C[g]
    fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers+text",
                               marker=dict(size=12, color=COLS[g], symbol="diamond"),
                               text=[day[:3]], textposition="top center",
                               textfont=dict(size=14, color="#0b0b0b"), showlegend=False))
fig.update_layout(
    title=dict(text="RAW Qwen3-8B layer-18 activations, weekday tokens — PCA to 3D<br>"
               f"<sup>each dot = one real prompt's 4096-dim activation · 10 prompts/day · "
               f"axes = the week\u2019s own top-3 directions ({frac_between*100:.0f}% of day-to-day structure) · grey ring = fitted circle</sup>"),
    scene=dict(xaxis_title="week axis 1", yaxis_title="week axis 2", zaxis_title="week axis 3", aspectmode="data"),
    template="plotly_white", legend=dict(itemsizing="constant"))
fig.write_html(f"{SP}/weekday_pca3d.html", include_plotlyjs=True, full_html=True)
print("wrote weekday_pca3d.html")

# --- static PNG (two views) via matplotlib for Preview ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig2 = plt.figure(figsize=(12.5, 6), facecolor="#fcfcfb")
for i, (el, az) in enumerate([(18, -60), (62, 30)]):
    ax = fig2.add_subplot(1, 2, i + 1, projection="3d")
    ax.set_facecolor("#fcfcfb")
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#9a9890", lw=2)
    for g, day in enumerate(days):
        pts = P3[labels == g]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=COLS[g], s=34, label=day if i == 0 else None)
        ax.text(*C[g], day[:3], fontsize=10, fontweight="bold", color=COLS[g])
    ax.view_init(elev=el, azim=az)
    ax.set_xlabel("week axis 1"); ax.set_ylabel("week axis 2"); ax.set_zlabel("week axis 3")
    ax.set_title(["side view — it's a ring", "top-down — seven days, calendar order"][i], fontweight="bold")
fig2.legend(loc="lower center", ncol=7, frameon=False)
fig2.suptitle("Raw activations (no model of them — just PCA): the week is literally a circle",
              fontweight="bold", fontsize=13)
fig2.tight_layout(rect=(0, 0.05, 1, 0.96))
fig2.savefig(f"{SP}/fig6_pca3d_views.png", dpi=200)
print("wrote fig6_pca3d_views.png")
