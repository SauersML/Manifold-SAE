#!/usr/bin/env python3
"""
UMAP color-manifold video on the SHARED timeline (synced_timeline.CP_TIME).
Keeps umap_movie2.py design: dark theme, true-RGB dots with glow + fading trails,
stage-colored title + live step counter, timeline playhead, minimal-frame-distance
sequential Procrustes alignment, no connecting lines.

The ONLY change from umap_movie2 is timing: instead of fixed HOLD/F frame counts,
every video frame's time t (seconds) is converted to a fractional checkpoint index
via the shared CP_TIME mapping, then the two bracketing checkpoint embeddings are
smoothstep-blended. So checkpoint i is fully shown exactly at CP_TIME[i] -- the same
instant the audio's descriptor for checkpoint i is fully active.
"""
import os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import orthogonal_procrustes
import umap
import synced_timeline as T

plt.rcParams.update({"font.family": "DejaVu Sans"})
BG = "#0c0e14"; FG = "#e8e8ee"
SC = {"pretrain": "#5b8def", "SFT": "#f2a34a", "DPO": "#b06cf0", "RL 3.0": "#2fd089", "RL 3.1": "#19c6b0"}

files = T.FILES
stg, stp = T.STAGES, T.STEPS
embs = []
for f in files:
    V = np.load(f)["V"].astype(np.float64)
    Vc = V - V.mean(0)
    U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    Vdef = Vc - np.outer(Vc @ Vt[0], Vt[0])
    embs.append(umap.UMAP(n_neighbors=7, min_dist=0.25, metric="cosine", random_state=0).fit_transform(Vdef))
RGB = np.load(files[0])["rgb"] / 255

def nrm(E):
    E = E - E.mean(0); return E / (np.linalg.norm(E) + 1e-9)

# minimal frame-to-frame motion: sequential Procrustes to previous frame
norms = [nrm(E) for E in embs]; A = [norms[0]]
for En in norms[1:]:
    R, _ = orthogonal_procrustes(En, A[-1]); A.append(En @ R)
A = np.array(A) * 8.0; lim = np.abs(A).max() * 1.12

# ---- SHARED TIMELINE -> per-frame fractional checkpoint index ----
CP = T.CP_TIME
NF = int(round(T.DUR * T.FPS))                      # total video frames
t_frames = np.arange(NF) / T.FPS                    # seconds
t_frames[-1] = min(t_frames[-1], T.DUR)
# invert CP_TIME: time -> fractional checkpoint index (the canonical mapping)
ci_frac = np.interp(t_frames, CP, np.arange(T.NC))
i0 = np.clip(np.floor(ci_frac).astype(int), 0, T.NC - 2)
fr = ci_frac - i0
w = fr * fr * (3 - 2 * fr)                           # smoothstep blend

frames = ((1 - w)[:, None, None] * A[i0] + w[:, None, None] * A[i0 + 1])
# active checkpoint for title/step/playhead = nearest of the two
near = np.where(w < 0.5, i0, i0 + 1)
fstage = [stg[k] for k in near]
fstep = [stp[k] for k in near]
fci = ci_frac

fig = plt.figure(figsize=(7.2, 7.8), facecolor=BG)
ax = fig.add_axes([0.04, 0.10, 0.92, 0.80]); ax.set_facecolor(BG)
tax = fig.add_axes([0.08, 0.045, 0.84, 0.022]); tax.set_xlim(0, T.NC - 1); tax.set_ylim(0, 1); tax.axis("off")
seg = {}
for i, s in enumerate(stg): seg.setdefault(s, [i, i]); seg[s][1] = i
for s, (a, b) in seg.items(): tax.add_patch(plt.Rectangle((a, 0), b - a + 0.9, 1, color=SC[s], alpha=.5, lw=0))
play = tax.scatter([0], [0.5], s=70, color="white", zorder=5, edgecolors=SC["pretrain"], lw=2)
title = fig.text(0.5, 0.955, "", ha="center", va="center", fontsize=21, fontweight="bold", color=FG)
sub = fig.text(0.5, 0.917, "", ha="center", va="center", fontsize=11, color="#9aa0b0")
fig.text(0.5, 0.012, "OLMo-3-32B  ·  color manifold (UMAP, sequential-aligned)  ·  pretrain → SFT → DPO → RL",
         ha="center", fontsize=8.5, color="#666")
TR = 9

def draw(t):
    ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(BG)
    for h in range(TR, 0, -1):
        if t - h >= 0:
            ax.scatter(frames[t - h][:, 0], frames[t - h][:, 1], c=RGB, s=260 * (1 - h / TR * 0.6),
                       alpha=0.10 * (1 - h / TR), edgecolors="none", zorder=1)
    P = frames[t]
    ax.scatter(P[:, 0], P[:, 1], c=RGB, s=900, alpha=0.16, edgecolors="none", zorder=2)
    ax.scatter(P[:, 0], P[:, 1], c=RGB, s=360, edgecolors="white", linewidths=1.1, zorder=3)
    s = fstage[t]; title.set_text(s); title.set_color(SC[s]); sub.set_text(f"step {fstep[t]:,}")
    play.set_offsets([[fci[t], 0.5]]); play.set_edgecolor(SC[s])
    return ()

ani = animation.FuncAnimation(fig, draw, frames=NF, blit=False)
ani.save("/tmp/synced_video.mp4", writer=animation.FFMpegWriter(fps=T.FPS, bitrate=4200),
         dpi=120, savefig_kwargs={"facecolor": BG})
print("saved /tmp/synced_video.mp4  frames=%d  dur=%.3fs  fps=%d" % (NF, NF / T.FPS, T.FPS))

# emit the title-change times (seconds) for sync verification
tc = []
for k in range(1, NF):
    if fstage[k] != fstage[k - 1]:
        tc.append((fstage[k - 1], fstage[k], round(k / T.FPS, 3)))
np.savez("/tmp/synced_video_meta.npz",
         near=near, fci=fci, t_frames=t_frames,
         title_change_times=np.array([x[2] for x in tc]),
         title_change_labels=np.array([f"{a}->{b}" for a, b, _ in tc]))
print("stage-title changes (s):", tc)
