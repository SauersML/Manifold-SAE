"""auto_27: spec complementarity matrix.

Idea (fresh, not covered by auto_01..26 or auto_exp_04..08):

For two specs s1, s2, an *upper bound* on the macro EVR-weighted R^2 you'd
get from picking the better of the two on every PC is

    pc_max[k]      = max(R2[s1, k], R2[s2, k])
    union_macro[s1,s2] = sum_k clip(pc_max[k], 0, 1) * EVR[k] / sum_k EVR[k]

If union_macro is much larger than max(macro[s1], macro[s2]) the two specs
are *complementary* (they explain different PCs). If union_macro ~ better of
the two, one dominates. This is a per-PC oracle, not a real ensemble, but
it is a clean upper bound on per-PC routing.

We compute this on the top-N specs (by macro R^2), plus a normalized "gain"
matrix:

    gain[s1,s2] = union_macro[s1,s2] - max(macro[s1], macro[s2])

Output two heatmaps + a ranked list of most-complementary pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_27.png"
OUT_JSON = RUN / "auto_27.json"

TOP_N = 24


def main() -> None:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    evr = np.array(L["explained_variance_ratio_topK"])  # (K,)
    evr_sum = evr.sum()

    specs = L["specs"]
    rows = [(n, v) for n, v in specs.items() if "r2_per_pc_mean" in v]
    names_all = [n for n, _ in rows]
    r2_pc_all = np.array([v["r2_per_pc_mean"] for _, v in rows])  # (S, K)
    macro_all = np.array([v["r2_macro_mean"] for _, v in rows])

    # Drop the saturated U_pca_{64,96,128}d (R^2 == 1.0; trivial, K<=d_pc).
    keep = ~((np.isclose(macro_all, 1.0)) & np.array(
        [n.startswith("U_pca_") for n in names_all]
    ))
    names = [n for n, k in zip(names_all, keep) if k]
    r2_pc = r2_pc_all[keep]
    macro = macro_all[keep]

    # Take top-N by macro R^2 (after dropping the trivial ones).
    order = np.argsort(-macro)[:TOP_N]
    names_t = [names[i] for i in order]
    r2_pc_t = r2_pc[order]                       # (N, K)
    macro_t = macro[order]
    N = len(names_t)

    # Pairwise per-PC max R^2 (clipped), EVR-weighted -> union_macro[i,j].
    r2c = np.clip(r2_pc_t, 0.0, 1.0)             # (N, K)
    # pairwise max via broadcasting: (N,1,K) vs (1,N,K)
    pc_max = np.maximum(r2c[:, None, :], r2c[None, :, :])  # (N,N,K)
    union = (pc_max * evr[None, None, :]).sum(axis=2) / evr_sum  # (N,N)

    best_solo = np.maximum(macro_t[:, None], macro_t[None, :])
    gain = union - best_solo
    np.fill_diagonal(gain, 0.0)

    # Top complementary pairs (i<j) by gain.
    iu, ju = np.triu_indices(N, k=1)
    pair_gains = gain[iu, ju]
    rank = np.argsort(-pair_gains)
    top_pairs = [
        {
            "s1": names_t[iu[r]],
            "s2": names_t[ju[r]],
            "macro_s1": float(macro_t[iu[r]]),
            "macro_s2": float(macro_t[ju[r]]),
            "union_macro": float(union[iu[r], ju[r]]),
            "gain": float(pair_gains[r]),
        }
        for r in rank[:25]
    ]

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.4), constrained_layout=True)

    ax = axes[0]
    im0 = ax.imshow(union, cmap="viridis", vmin=union.min(), vmax=union.max())
    ax.set_xticks(range(N)); ax.set_xticklabels(names_t, rotation=90, fontsize=7)
    ax.set_yticks(range(N))
    ax.set_yticklabels(
        [f"{n}  ({m:.3f})" for n, m in zip(names_t, macro_t)], fontsize=7
    )
    ax.set_title(
        "auto_27: per-PC oracle union R^2 (EVR-weighted)\n"
        "union[i,j] = sum_k max(R2_i[k], R2_j[k]) * EVR[k] / sum EVR  "
        f"(top {N} specs at L40)",
        fontsize=10,
    )
    fig.colorbar(im0, ax=ax, fraction=0.045, label="union macro R^2")

    ax = axes[1]
    vlim = max(abs(gain.min()), abs(gain.max()))
    im1 = ax.imshow(gain, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(N)); ax.set_xticklabels(names_t, rotation=90, fontsize=7)
    ax.set_yticks(range(N))
    ax.set_yticklabels(names_t, fontsize=7)
    ax.set_title(
        "complementarity gain = union - max(solo_i, solo_j)\n"
        "red = the pair is more than either alone (orthogonal PC coverage)",
        fontsize=10,
    )
    fig.colorbar(im1, ax=ax, fraction=0.045, label="gain in macro R^2")

    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "top_n": N,
        "names": names_t,
        "macro_solo": macro_t.tolist(),
        "union_macro_min": float(union.min()),
        "union_macro_max": float(union.max()),
        "gain_max": float(gain.max()),
        "top_complementary_pairs": top_pairs,
        "note": (
            "Dropped saturated U_pca_{64,96,128}d (R^2==1.0; trivial, "
            "d_pc>=K=64). Union is a per-PC oracle upper bound, not an "
            "actual ensemble; gain>0 means the two specs cover different PCs."
        ),
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
