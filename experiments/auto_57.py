"""
auto_57: t-SNE of the 28 templates in residual-PC space (idea aaaaaaa)

Follow-up to auto_47 which used a PCA of template means. Here we focus
purely on a *visual* embedding: each of the 28 templates becomes a
single point, defined by its mean activation across all 949 colors in
the top-K PCA subspace of the layer-40 residual stream. We then run
t-SNE (perplexity=8, n=28) and UMAP-via-PCA-fallback (we only have
scikit-learn here, no umap-learn, so we use a PCA companion panel).

What we plot (4 panels):
  1. t-SNE 2D scatter of the 28 templates, each labelled with a short
     stub of the template string. Marker colour = first PC of the
     template-mean subspace (i.e. the dominant axis from auto_47),
     marker size = character length of the template.
  2. PCA 2D scatter of the same 28 points for comparison (so we can
     judge whether t-SNE found additional structure beyond linear).
  3. k-NN graph (k=3) overlaid on the t-SNE coords: directed edges
     from each template to its 3 nearest neighbours in the *original*
     PC-space (not in t-SNE space), so we can see whether t-SNE
     preserves neighbourhoods.
  4. Bar chart: mean distance to k-NN neighbours per template -- a
     "template isolation" score (high = idiosyncratic template, low =
     blends into the pack).

Allow-listed primitives only: PCA, k-NN, t-SNE. No Gaussian RBF, no
Duchon (length_scale or otherwise). We do not refit any GAM here.

Outputs
-------
PNG  : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_57.png
JSON : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_57.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
X_PATH = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_57.png"
OUT_JSON = RUN / "auto_57.json"

K_PCS = 64
K_NN = 3
PERPLEXITY = 8
RNG = 0


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    n_colors = len(d["color_axes_per_color_index"]["R"])
    print(f"templates={n_t} colors={n_colors}")

    # Use the saved PCA basis to avoid re-fitting on 26572 x 7168.
    layer = d["per_layer"]["L40"]
    Vt = np.asarray(layer["Vt_topK"], dtype=np.float32)        # (K, H)
    mu = np.asarray(layer["mu"], dtype=np.float32)             # (H,)
    sigma = np.asarray(layer["sigma"], dtype=np.float32)       # scalar or (H,)
    print("Vt:", Vt.shape, "mu:", mu.shape, "sigma:", np.shape(sigma))

    X = np.load(X_PATH, mmap_mode="r")                          # (N, H)
    assert X.shape[0] == n_t * n_colors, (X.shape, n_t, n_colors)

    # Project to PCA scores: Z = ((X - mu)/sigma) @ Vt.T
    # Done in chunks to keep memory modest.
    Z = np.empty((X.shape[0], K_PCS), dtype=np.float32)
    chunk = 2048
    for i in range(0, X.shape[0], chunk):
        block = np.asarray(X[i:i + chunk], dtype=np.float32)
        block = (block - mu) / sigma
        Z[i:i + chunk] = block @ Vt.T
    print("Z:", Z.shape)

    # Row layout: harvest order is (template, color) outer x inner.
    # Confirm via reshape; centroid per template:
    Z3 = Z.reshape(n_t, n_colors, K_PCS)
    T = Z3.mean(axis=1)                                         # (n_t, K)
    print("T:", T.shape)

    # PCA of the 28 template centroids (interpret-axis colour for marker)
    pca28 = PCA(n_components=min(n_t - 1, K_PCS), svd_solver="full")
    T_pca = pca28.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc2 = T_pca[:, 1]
    evr28 = pca28.explained_variance_ratio_
    print("PCA-of-28 EVR top4:", np.round(evr28[:4], 3))

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=RNG,
        max_iter=2000,
    )
    T_tsne = tsne.fit_transform(T)

    # k-NN in original PC-space (excluding self)
    nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(T)
    dists, idxs = nn.kneighbors(T)
    dists, idxs = dists[:, 1:], idxs[:, 1:]
    iso_score = dists.mean(axis=1)

    # ---- plot ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    char_len = np.array([len(s) for s in templates], dtype=float)
    sizes = 60 + 220 * (char_len - char_len.min()) / max(1.0, float(char_len.max() - char_len.min()))

    def stub(s: str, n: int = 28) -> str:
        s = s.replace("{x}", "X")
        return s if len(s) <= n else s[: n - 1] + "…"

    # (1) t-SNE
    ax = axes[0, 0]
    sc = ax.scatter(T_tsne[:, 0], T_tsne[:, 1], c=pc1, s=sizes,
                    cmap="coolwarm", edgecolor="black", linewidth=0.6, zorder=3)
    for i, s in enumerate(templates):
        ax.annotate(stub(s), (T_tsne[i, 0], T_tsne[i, 1]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                    zorder=4)
    ax.set_title(f"t-SNE of 28 template centroids in top-{K_PCS} PC space\n"
                 f"(perplexity={PERPLEXITY}; colour=PC1, size=char-length)")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="PC1 of template-mean subspace")
    ax.grid(alpha=0.25)

    # (2) PCA companion
    ax = axes[0, 1]
    ax.scatter(pc1, pc2, c=pc1, s=sizes, cmap="coolwarm",
               edgecolor="black", linewidth=0.6, zorder=3)
    for i, s in enumerate(templates):
        ax.annotate(stub(s), (pc1[i], pc2[i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                    zorder=4)
    ax.set_title(f"PCA companion (EVR1={evr28[0]:.2f}, EVR2={evr28[1]:.2f})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)

    # (3) t-SNE + kNN(original-space) edges
    ax = axes[1, 0]
    ax.scatter(T_tsne[:, 0], T_tsne[:, 1], c=pc1, s=sizes,
               cmap="coolwarm", edgecolor="black", linewidth=0.6, zorder=3)
    for i in range(n_t):
        for j in idxs[i]:
            ax.plot([T_tsne[i, 0], T_tsne[j, 0]],
                    [T_tsne[i, 1], T_tsne[j, 1]],
                    color="grey", alpha=0.45, linewidth=0.8, zorder=1)
    for i in range(n_t):
        ax.annotate(str(i), (T_tsne[i, 0], T_tsne[i, 1]),
                    fontsize=8, fontweight="bold", ha="center", va="center",
                    zorder=5)
    ax.set_title(f"t-SNE coords + k={K_NN}-NN edges from original PC-space\n"
                 "(edges crossing widely = t-SNE distorted neighbourhood)")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
    ax.grid(alpha=0.25)

    # (4) isolation score
    ax = axes[1, 1]
    order = np.argsort(iso_score)
    ax.barh(np.arange(n_t),
            iso_score[order],
            color=plt.cm.viridis(np.linspace(0.15, 0.95, n_t)))
    ax.set_yticks(np.arange(n_t))
    ax.set_yticklabels([f"{order[i]:>2}  {stub(templates[order[i]], 36)}"
                        for i in range(n_t)], fontsize=7)
    ax.set_xlabel(f"mean distance to {K_NN}-NN in {K_PCS}-PC space")
    ax.set_title("Template isolation ranking (low = blends in, high = idiosyncratic)")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("auto_57: visual t-SNE of 28 template centroids "
                 "(cogito L40, top-64 PCs)", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print("wrote", OUT_PNG)

    # ---- json --------------------------------------------------------
    payload = {
        "n_templates": n_t,
        "n_colors": n_colors,
        "k_pcs": K_PCS,
        "tsne_perplexity": PERPLEXITY,
        "knn_k": K_NN,
        "pca28_evr_top4": [float(x) for x in evr28[:4]],
        "tsne_coords": T_tsne.tolist(),
        "pca_coords_pc12": np.column_stack([pc1, pc2]).tolist(),
        "knn_neighbors": idxs.tolist(),
        "knn_distances": dists.tolist(),
        "isolation_score": iso_score.tolist(),
        "isolation_ranking_low_to_high": [int(i) for i in order],
        "templates": templates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
