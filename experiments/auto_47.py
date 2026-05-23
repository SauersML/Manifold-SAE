"""
auto_47: What does PC1 of the template-mean subspace encode?  (idea zzzzz)

Follow-up to auto_46, which decomposed per-PC variance into color / template /
residual. Here we *zoom into the template-only subspace*: for the top-K PCA
features Z (N=26572 rows = 949 colors x 28 templates), we form the
(n_t, K) matrix T of per-template marginal means, run a second PCA on T,
and ask:

    What does PC1 (and PC2) of the template-mean subspace encode?

We characterize each template axis with surface features computed directly
from the template strings (length in chars/words, position of the "{x}"
slot, presence of broad semantic categories like fabric/sky/food/etc.) and
report:

  - Pearson correlations of each surface feature with the per-template
    PC1 / PC2 scores.
  - A 2-panel figure:
      (left)  PC1 vs PC2 scatter, every point labeled with its (truncated)
              template; templates sorted by PC1.
      (right) Bar of PC1 scores sorted, with per-template label, and a
              second axis showing template character length.

Tools: PCA only (numpy.linalg.svd). No Gaussian RBF, no Duchon, no kNN
(this is a pure descriptive PC1-interpretation plot, the constraints in
the prompt list are about *which fitters are allowed* — here we only fit a
PCA).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG = RUN_DIR / "auto_47.png"


# Hand-picked semantic buckets covering each of the 28 prompts. Each template
# is tagged with a primary category for visual grouping; this is *not* used
# in PCA -- only as a color in the scatter.
CATEGORY_KEYWORDS = [
    ("fabric",   ["silk", "velvet", "scarf"]),
    ("sky",      ["sky", "fog", "dawn", "night", "sunset"]),
    ("nature",   ["meadow", "wildflowers", "ocean", "leaf", "stallion", "plain", "harbor"]),
    ("art",      ["painter", "pigments", "canvas", "brush", "palette",
                  "renaissance", "stained-glass", "fresco"]),
    ("object",   ["car", "candle", "chapel", "pen", "neon", "diner",
                  "antique", "stone", "bronze"]),
    ("food",     ["chef", "reduction", "duck", "macaron", "filling"]),
    ("body",     ["hair", "skin", "eyes", "freckles", "throat", "hummingbird", "cat"]),
    ("abstract", ["grief", "writing", "glasses", "walls", "bedroom"]),
]


def categorize(template: str) -> str:
    s = template.lower()
    for cat, keys in CATEGORY_KEYWORDS:
        for kw in keys:
            if kw in s:
                return cat
    return "other"


def short_label(template: str, maxlen: int = 38) -> str:
    s = template.replace("{x}", "<x>")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def template_features(templates: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (F, names) of per-template surface features."""
    names: list[str] = [
        "char_len",
        "word_count",
        "x_slot_word_idx_norm",  # position of <x> in [0,1]
        "n_commas",
        "has_color_word",
        "has_sensory_word",  # see/look/watch/eyes
        "has_animate",       # her/his/he/she/I
        "is_metaphor",       # grief/writing/world/refused
    ]
    F = np.zeros((len(templates), len(names)), dtype=np.float64)
    color_words = {"painted", "glow", "luminous", "iridescent", "neon",
                   "pigment", "pigments", "stained-glass", "chrome"}
    sensory = {"saw", "see", "look", "looked", "eyes", "watched",
               "flashed", "shone", "glowed", "lit", "caught"}
    animate = {"her", "his", "he", "she", "i"}
    metaphor = {"grief", "writing", "world", "refused", "kind"}

    for i, t in enumerate(templates):
        F[i, 0] = len(t)
        words = re.findall(r"\S+", t)
        F[i, 1] = len(words)
        # slot position
        slot_idx = next((k for k, w in enumerate(words) if "{x}" in w), len(words))
        F[i, 2] = slot_idx / max(len(words) - 1, 1)
        F[i, 3] = t.count(",")
        low = set(re.findall(r"[A-Za-z\-]+", t.lower()))
        F[i, 4] = float(len(low & color_words) > 0)
        F[i, 5] = float(len(low & sensory)    > 0)
        F[i, 6] = float(len(low & animate)    > 0)
        F[i, 7] = float(len(low & metaphor)   > 0)
    return F, names


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    with RESULTS.open() as f:
        res = json.load(f)

    templates: list[str] = res["templates"]
    pl = res["per_layer"]["L40"]
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float32)                      # (K, D)
    mu = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)            # (1, D)
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)      # (1, D)
    evr = np.asarray(pl["explained_variance_ratio_topK"], dtype=np.float64)
    K, D = Vt.shape
    n_t = len(templates)
    n_c = len(res["color_axes_per_color_index"]["R"])
    N = n_c * n_t
    print(f"[layout] K={K} D={D} n_colors={n_c} n_templates={n_t} N={N}", flush=True)

    # Stream-project X -> Z (top-K PCs) using the GAM pipeline normalization.
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape == (N, D) or X.shape[0] >= N
    chunk = 2048
    Z = np.zeros((N, K), dtype=np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Z[i:j] = ((Xc - mu) / sigma) @ Vt.T
    print(f"[project] Z {Z.shape}", flush=True)

    # Row r -> color = r // n_t, template = r % n_t.
    templ_idx = np.tile(np.arange(n_t), n_c)

    # Per-template marginal means in PCA space: T (n_t, K).
    T = np.zeros((n_t, K), dtype=np.float64)
    np.add.at(T, templ_idx, Z)
    T /= n_c

    # Center, then PCA via SVD: T_c = U S V^T, scores = U*S.
    T_c = T - T.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(T_c, full_matrices=False)
    scores = U * S                          # (n_t, n_t)
    evr_t = (S ** 2) / (S ** 2).sum()
    print(f"[template-PCA] evr top4 = {evr_t[:4]}", flush=True)

    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(n_t)

    # Sign convention: longer prompts -> positive PC1 (interpretability).
    char_len = np.array([len(t) for t in templates], dtype=np.float64)
    if np.corrcoef(pc1, char_len)[0, 1] < 0:
        pc1 = -pc1
    # similar for PC2 against word_count
    word_count = np.array([len(re.findall(r"\S+", t)) for t in templates], dtype=np.float64)
    if np.corrcoef(pc2, word_count)[0, 1] < 0:
        pc2 = -pc2

    F, fnames = template_features(templates)
    print("[feature correlations] (Pearson r with PC1 / PC2)")
    corrs = []
    for k, name in enumerate(fnames):
        rho1 = float(np.corrcoef(F[:, k], pc1)[0, 1]) if F[:, k].std() > 0 else float("nan")
        rho2 = float(np.corrcoef(F[:, k], pc2)[0, 1]) if F[:, k].std() > 0 else float("nan")
        print(f"  {name:24s}  r1={rho1:+.3f}  r2={rho2:+.3f}", flush=True)
        corrs.append((name, rho1, rho2))

    cats = [categorize(t) for t in templates]
    cat_set = sorted(set(cats))
    cmap = plt.get_cmap("tab10")
    cat_colors = {c: cmap(i % 10) for i, c in enumerate(cat_set)}

    # ------------ Plot ------------
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    # (left) PC1 vs PC2 scatter with truncated labels
    ax1 = fig.add_subplot(gs[0, 0])
    for c in cat_set:
        m = np.array([cc == c for cc in cats])
        ax1.scatter(pc1[m], pc2[m], s=90, c=[cat_colors[c]],
                    edgecolor="black", linewidth=0.6, label=c, alpha=0.9)
    for i in range(n_t):
        ax1.annotate(short_label(templates[i], 30),
                     (pc1[i], pc2[i]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    ax1.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel(f"template-PC1  (evr={evr_t[0]*100:.1f}%)")
    ax1.set_ylabel(f"template-PC2  (evr={evr_t[1]*100:.1f}%)")
    ax1.set_title("Template-mean subspace: PC1 vs PC2  (label = template prefix)")
    ax1.legend(fontsize=8, loc="best", ncol=2)
    ax1.grid(True, alpha=0.25)

    # (right) bar plot: templates sorted by PC1, with char-length overlay
    ax2 = fig.add_subplot(gs[0, 1])
    order = np.argsort(pc1)
    ys = np.arange(n_t)
    bars = ax2.barh(ys, pc1[order],
                    color=[cat_colors[cats[i]] for i in order],
                    edgecolor="black", linewidth=0.4)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([short_label(templates[i], 50) for i in order], fontsize=7)
    ax2.set_xlabel("template-PC1 score")
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", linewidth=0.6)
    ax2.set_title("Templates ranked by PC1  (color = semantic category)")
    ax2.grid(True, alpha=0.25, axis="x")

    ax2b = ax2.twiny()
    ax2b.plot(char_len[order], ys, "o-", color="black",
              linewidth=0.8, markersize=3, alpha=0.6, label="char-length")
    ax2b.set_xlabel("template char-length", color="black")
    ax2b.legend(loc="lower right", fontsize=8)

    # Annotate the top of the figure with feature correlations
    corr_lines = ["Pearson r with PC1 / PC2 :"] + \
        [f"  {n:<22s} r1={r1:+.2f}  r2={r2:+.2f}" for (n, r1, r2) in corrs]
    fig.text(0.005, 0.5, "\n".join(corr_lines),
             fontsize=8, family="monospace", va="center")

    fig.suptitle(
        "auto_47 — Template-mean subspace PCA (PC1 interpretation)  "
        f"[n_t={n_t}, K={K}, second-PCA evr1={evr_t[0]*100:.1f}%, "
        f"evr2={evr_t[1]*100:.1f}%]",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[save] {OUT_PNG}", flush=True)

    summary = {
        "evr_template_pca_top4": [float(x) for x in evr_t[:4]],
        "feature_corr_with_PC1": {n: r1 for (n, r1, _) in corrs},
        "feature_corr_with_PC2": {n: r2 for (n, _, r2) in corrs},
        "pc1_sorted_template_indices": [int(i) for i in np.argsort(pc1)],
        "n_t": int(n_t), "K": int(K), "n_colors": int(n_c),
    }
    (RUN_DIR / "auto_47.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {RUN_DIR/'auto_47.json'}", flush=True)


if __name__ == "__main__":
    main()
