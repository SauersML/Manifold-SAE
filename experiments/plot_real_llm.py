"""Plot the real-LLM result: Qwen2.5-1.5B day-circle + the read/steer/gate gap."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# from the real run (Qwen2.5-1.5B, layer 14, real_llm_result.json)
proj = np.array([[-2.799, 13.013], [-8.812, 7.498], [-10.598, -1.308], [-8.375, -8.471],
                 [0.448, -10.482], [12.760, -5.794], [17.375, 5.543]])
ring_order = [2, 3, 4, 5, 6, 0, 1]   # Wed Thu Fri Sat Sun Mon Tue = correct weekly cycle
best_read, gw_read = 0.74, 0.16
best_steer, gw_steer = 0.79, 0.0

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Real model (Qwen2.5-1.5B, layer 14, days of week) — manifold-SAE finds a causal dial,\n"
             "but the firing atom isn't it", fontsize=12, fontweight="bold")

# A: the day circle in the model
a = ax[0]
loop = ring_order + [ring_order[0]]
a.plot(proj[loop, 0], proj[loop, 1], "-", c="#aac", lw=1.5, zorder=1)
a.scatter(proj[:, 0], proj[:, 1], s=120, c=range(7), cmap="hsv", zorder=2, edgecolor="k")
for i, d in enumerate(DAYS):
    a.annotate(d, proj[i], fontsize=11, fontweight="bold", ha="center", va="center",
               xytext=(0, 0), textcoords="offset points")
a.set_title("A. The 7 day-centroids lie on a circle in the model\n"
            "(top-2 PCA var = 0.60; connected in true weekly order)", fontsize=10)
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_aspect("equal")

# B: read vs steer, best-read atom vs gate-winner atom
b = ax[1]
x = np.arange(2); w = 0.35
b.bar(x - w / 2, [best_read, best_steer], w, label="best-aligned atom (#30)", color="#2a7")
b.bar(x + w / 2, [gw_read, gw_steer], w, label="GATE-WINNER atom (#41)", color="#c33")
b.axhline(0.7, ls="--", c="gray", lw=1)
b.set_xticks(x); b.set_xticklabels(["coordinate READ\n(corr w/ true day)", "STEER\n(move dial → model's\npredicted day shifts)"])
b.set_ylim(0, 1); b.legend(fontsize=9); b.set_ylabel("score")
for i, (v1, v2) in enumerate([(best_read, gw_read), (best_steer, gw_steer)]):
    b.text(i - w / 2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=9)
    b.text(i + w / 2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=9)
b.set_title("B. One atom reads (0.74) AND steers (0.79) the real model;\n"
            "the atom that fires for days does neither (0.16, ~0)", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/Users/user/Manifold-SAE/runs/REAL_LLM_PLOT.png", dpi=130)
print("saved runs/REAL_LLM_PLOT.png")
