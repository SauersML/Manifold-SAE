"""auto_16: Jaccard overlap of "hardest PCs" across top-performing specs.

Angle (y), reframed onto PCs (which is what results.json exposes per-spec):
auto_10 looked at pairwise *Pearson correlation* of r2_per_pc across specs
(continuous, global). Here we ask the orthogonal, tail-focused question:
when each spec stumbles, does it stumble on the *same PCs*?

For each spec we define hard_K(spec) = the K PCs with the lowest mean R^2.
Then for the top-N specs (ranked by macro R^2) we compute pairwise Jaccard
overlap of those sets. High overlap => failure modes are shared (an
intrinsic-limit story); low overlap => each parameterization has its own
weak spots (a model-capacity story).

We sweep K and plot:
  (left)  Jaccard heatmap among top-N specs at a fixed K.
  (right) mean off-diagonal Jaccard as K varies — how concentrated is the
          shared-failure signal?

Also dumps a JSON sidecar with the hardest-PC sets per spec for downstream
inspection.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RUN = "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT_PNG = os.path.join(RUN, "auto_16.png")
OUT_JSON = os.path.join(RUN, "auto_16.json")

with open(os.path.join(RUN, "results.json")) as f:
    d = json.load(f)

specs = d["per_layer"]["L40"]["specs"]
ev = np.array(d["per_layer"]["L40"]["explained_variance_ratio_topK"])

# Keep specs with a real per-PC vector and finite macro R^2.
names_all = [n for n in specs if "r2_per_pc_mean" in specs[n]]
mat = np.array([specs[n]["r2_per_pc_mean"] for n in names_all], dtype=float)
macro = np.array([specs[n]["r2_macro_mean"] for n in names_all], dtype=float)

# Drop pathological / constant rows (Jaccard on argsort is undefined-ish).
nz = (mat.std(axis=1) > 1e-10) & np.isfinite(macro)
names = [n for n, ok in zip(names_all, nz) if ok]
mat = mat[nz]
macro = macro[nz]

n_pcs = mat.shape[1]
print(f"Using {len(names)} specs over {n_pcs} PCs.")

# ---- Pick a sensible "top-N" subset of strong specs to compare. ----
TOP_N = 12
order = np.argsort(-macro)[:TOP_N]
top_names = [names[i] for i in order]
top_mat = mat[order]
top_macro = macro[order]
print("Top-N specs (macro R^2):")
for n, m in zip(top_names, top_macro):
    print(f"  {n:30s}  {m:.4f}")

# ---- Hardest-PC sets per spec, swept over K. ----
K_grid = [4, 6, 8, 10, 12, 16, 20]
# For the heatmap, use a middle K.
K_HEATMAP = 10
hard_sets_at = {K: [set(np.argsort(row)[:K].tolist()) for row in top_mat] for K in K_grid}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


# pairwise Jaccard matrix at K_HEATMAP
N = len(top_names)
J = np.zeros((N, N))
hs = hard_sets_at[K_HEATMAP]
for i in range(N):
    for j in range(N):
        J[i, j] = jaccard(hs[i], hs[j])

# mean off-diagonal Jaccard per K, plus a null baseline (random K-of-n_pcs).
mean_off = []
for K in K_grid:
    hsK = hard_sets_at[K]
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            vals.append(jaccard(hsK[i], hsK[j]))
    mean_off.append(float(np.mean(vals)))

# Random baseline: E[|A ∩ B|] = K^2 / n_pcs; E[|A ∪ B|] = 2K - K^2/n_pcs.
# So E[J] ≈ K / (2 n_pcs - K) for independent uniform K-subsets.
null = [K / (2 * n_pcs - K) for K in K_grid]

# ---- Plot. ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2),
                         gridspec_kw={"width_ratios": [1.1, 1.0]})

ax = axes[0]
im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(top_names, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels([f"{n}  (R²={m:.3f})" for n, m in zip(top_names, top_macro)],
                   fontsize=8)
ax.set_title(f"Jaccard overlap of {K_HEATMAP} hardest PCs\n"
             f"(top-{N} specs by macro R²)")
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                color="white" if J[i, j] < 0.5 else "black", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")

ax = axes[1]
ax.plot(K_grid, mean_off, marker="o", lw=2, label="observed (top-N specs)")
ax.plot(K_grid, null, marker="s", ls="--", color="gray",
        label=f"null: random K-subsets of {n_pcs}")
ax.set_xlabel("K (# of hardest PCs per spec)")
ax.set_ylabel("mean pairwise Jaccard (off-diagonal)")
ax.set_title("Concentration of shared failure modes vs K")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()

# Annotate which PCs are "universally hard" at K_HEATMAP (in >= half of top specs).
counts = np.zeros(n_pcs, dtype=int)
for s in hs:
    for p in s:
        counts[p] += 1
universal = np.where(counts >= (N // 2 + 1))[0]
txt = (f"PCs hard in ≥{N // 2 + 1}/{N} top specs (K={K_HEATMAP}):\n"
       + ", ".join(f"PC{p} (×{counts[p]}, ev={ev[p]:.2g})" for p in universal[:12]))
fig.text(0.5, -0.02, txt, ha="center", fontsize=8, wrap=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"wrote {OUT_PNG}")

# Save sidecar.
out = {
    "top_specs": top_names,
    "top_macro_r2": top_macro.tolist(),
    "K_heatmap": K_HEATMAP,
    "jaccard_matrix": J.tolist(),
    "K_grid": K_grid,
    "mean_offdiag_jaccard": mean_off,
    "null_jaccard": null,
    "n_pcs": int(n_pcs),
    "hardest_pcs_per_spec_at_K_heatmap": {
        n: sorted(list(s)) for n, s in zip(top_names, hs)
    },
    "universal_hard_pcs": {int(p): int(counts[p]) for p in universal.tolist()},
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT_JSON}")
