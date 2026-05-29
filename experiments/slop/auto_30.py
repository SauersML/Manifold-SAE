"""
auto_30: first-nearest-neighbor purity within color-name word groups (idea yyy).

Question: in the cogito L40 per-color centroid manifold, do colors whose names
share a base word (e.g. "blue", "green", "red", "pink", ...) cluster together?
i.e. is each color's 1-NN (in the residual stream) likely to share at least one
modifier word with it?

For each color c (xkcd name n_c), let words(c) be the set of whitespace-
separated tokens in n_c (e.g. "dark pastel green" -> {dark, pastel, green}).
Two colors "share a word" if their word sets intersect. For each color, we
find its 1-NN in centroid space (excluding self) and record whether the NN
shares any word. We do this in TWO spaces:

  (A) the L40 per-color centroid projected onto top-K PCA components,
  (B) the same colors' raw RGB ∈ [0,1]^3,

so we can compare LM geometry to pure perceptual baseline.

Purity is then aggregated per "color family" (the most common base words:
red, orange, yellow, green, blue, purple, pink, brown, grey, black, white).
For each family F we compute: among colors with F in their word set, what
fraction have a 1-NN that ALSO contains F? Chance baseline = (|F|-1)/(N-1).

No Gaussian RBF / radial bases — only PCA + Euclidean NN on raw features.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
COGITO_DIR = ROOT / "runs" / "COLOR_COGITO_L40"
OUT_PNG = RUN_DIR / "auto_30.png"
OUT_JSON = RUN_DIR / "auto_30.json"

N_TEMPLATES = 28
N_PCS = 32
FAMILIES = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "grey", "black", "white",
]
# also accept "gray" -> "grey", "violet"/"magenta" stay as their own surface forms
ALIASES = {"gray": "grey"}


def load_xkcd() -> tuple[list[str], np.ndarray]:
    cache = ROOT / "experiments" / "xkcd_colors.txt"
    names, rgb = [], []
    with open(cache) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            hexstr = parts[1].lstrip("#")
            names.append(name)
            rgb.append((int(hexstr[0:2], 16) / 255.0,
                        int(hexstr[2:4], 16) / 255.0,
                        int(hexstr[4:6], 16) / 255.0))
    return names, np.asarray(rgb, dtype=np.float64)


def name_words(name: str) -> set[str]:
    toks = re.split(r"[\s/'-]+", name.lower())
    out = set()
    for t in toks:
        if not t:
            continue
        out.add(ALIASES.get(t, t))
    return out


def first_nn(F: np.ndarray) -> np.ndarray:
    """Return index of 1-NN (excluding self) for each row of F."""
    # Pairwise squared distance via (a-b)^2 = a^2 + b^2 - 2ab
    sq = (F * F).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (F @ F.T)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1)


def main() -> None:
    print(f"[load] {COGITO_DIR / 'X_L40.npy'}", flush=True)
    X = np.load(COGITO_DIR / "X_L40.npy")  # (n_c * n_t, D)
    n_rows, d = X.shape
    names_all, rgb_all = load_xkcd()
    n_c = n_rows // N_TEMPLATES
    names = names_all[:n_c]
    rgb = rgb_all[:n_c]
    Xc = X[:n_c * N_TEMPLATES].astype(np.float32).reshape(n_c, N_TEMPLATES, d)

    centroid = Xc.mean(axis=1).astype(np.float64)  # (n_c, D)
    mu = centroid.mean(0, keepdims=True)
    sigma = centroid.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroid - mu) / sigma
    Cn -= Cn.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Cn, full_matrices=False)
    V = Vt[:N_PCS].T
    Z = Cn @ V                                       # (n_c, K)
    print(f"[pca] K={N_PCS}, evr={ (S[:N_PCS]**2).sum() / (S**2).sum():.3f}", flush=True)

    words = [name_words(n) for n in names]

    # 1-NN in each space.
    nn_z = first_nn(Z)
    nn_rgb = first_nn(rgb)
    # Also a third baseline: a random permutation (expectation ~ general-pop chance)
    rng = np.random.default_rng(0)
    nn_rand = rng.permutation(n_c)
    # ensure no self
    self_hits = np.where(nn_rand == np.arange(n_c))[0]
    if self_hits.size:
        nn_rand[self_hits] = (nn_rand[self_hits] + 1) % n_c

    def shares_any(i: int, j: int) -> int:
        return int(bool(words[i] & words[j]))

    share_z = np.array([shares_any(i, nn_z[i]) for i in range(n_c)])
    share_rgb = np.array([shares_any(i, nn_rgb[i]) for i in range(n_c)])
    share_rand = np.array([shares_any(i, nn_rand[i]) for i in range(n_c)])

    overall = {
        "n_colors": int(n_c),
        "share_any_word_L40_PCA": float(share_z.mean()),
        "share_any_word_RGB": float(share_rgb.mean()),
        "share_any_word_random": float(share_rand.mean()),
    }
    print("[overall]", overall, flush=True)

    # Per-family purity: among colors WHOSE NAME CONTAINS family F, what frac
    # of 1-NN also contains F?
    family_rows = []
    for F in FAMILIES:
        idx = [i for i in range(n_c) if F in words[i]]
        if not idx:
            continue
        n_F = len(idx)
        chance = (n_F - 1) / (n_c - 1)
        pur_z = float(np.mean([F in words[nn_z[i]] for i in idx]))
        pur_rgb = float(np.mean([F in words[nn_rgb[i]] for i in idx]))
        family_rows.append((F, n_F, chance, pur_z, pur_rgb))
        print(f"  [family {F:7s}] n={n_F:4d}  chance={chance:.3f}  "
              f"L40={pur_z:.3f}  RGB={pur_rgb:.3f}", flush=True)

    # Per-family lift over chance (L40)
    fam_names = [r[0] for r in family_rows]
    fam_n = np.array([r[1] for r in family_rows])
    fam_chance = np.array([r[2] for r in family_rows])
    fam_z = np.array([r[3] for r in family_rows])
    fam_rgb = np.array([r[4] for r in family_rows])

    # ----- plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.5, 2.2, 2.2]})

    # (a) overall share-any-word bar
    ax = axes[0]
    cats = ["L40 PCA", "raw RGB", "random"]
    vals = [overall["share_any_word_L40_PCA"],
            overall["share_any_word_RGB"],
            overall["share_any_word_random"]]
    colors = ["steelblue", "darkorange", "grey"]
    ax.bar(cats, vals, color=colors)
    for c, v in zip(cats, vals):
        ax.text(c, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("frac of colors whose 1-NN shares ≥1 name word")
    ax.set_title(f"overall 1-NN word-overlap purity\n(cogito L40 centroids, n_c={n_c}, K={N_PCS})")
    ax.grid(axis="y", alpha=0.3)

    # (b) per-family purity grouped bars
    ax = axes[1]
    x = np.arange(len(fam_names))
    w = 0.28
    ax.bar(x - w, fam_chance, w, color="lightgrey", label="chance (|F|-1)/(N-1)")
    ax.bar(x,     fam_z,      w, color="steelblue", label="L40 PCA 1-NN")
    ax.bar(x + w, fam_rgb,    w, color="darkorange", label="raw RGB 1-NN")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{F}\n(n={n})" for F, n in zip(fam_names, fam_n)], fontsize=8)
    ax.set_ylabel("P( family F in NN's name | family F in name )")
    ax.set_title("per-family 1-NN purity vs chance")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # (c) lift = L40 / chance, sorted
    ax = axes[2]
    lift_z = fam_z / np.maximum(fam_chance, 1e-9)
    lift_rgb = fam_rgb / np.maximum(fam_chance, 1e-9)
    order = np.argsort(-lift_z)
    x2 = np.arange(len(order))
    ax.bar(x2 - 0.2, lift_z[order], 0.4, color="steelblue", label="L40 lift")
    ax.bar(x2 + 0.2, lift_rgb[order], 0.4, color="darkorange", label="RGB lift")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", label="chance")
    ax.set_xticks(x2)
    ax.set_xticklabels([fam_names[i] for i in order], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("purity / chance (lift)")
    ax.set_title("families ranked by L40 lift\n(>1 = LM clusters this family tighter than chance)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}", flush=True)

    summary = {
        **overall,
        "n_pcs": int(N_PCS),
        "families": [
            {"family": F, "n": int(n), "chance": float(c),
             "purity_L40_PCA": float(z), "purity_RGB": float(rg),
             "lift_L40": float(z / max(c, 1e-9)),
             "lift_RGB": float(rg / max(c, 1e-9))}
            for (F, n, c, z, rg) in family_rows
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
