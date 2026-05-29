"""auto_exp_31 — Group-Lasso (BlockSparsityPenalty) on cogito L40 with
auxiliary-conditional targets (composition_engine.md §4(c) test).

Motivation
----------
Per memory project_ard_gauge_fix_doesnt_help_cogito: ARD alone and ARD+Ortho
both fail to prune cogito's d_aux=4 latent block because the data variance is
spread evenly across the auxiliary subspace. Hypothesis: grouping the dims
ex ante (Group A = perceptual, Group B = name-semantic) and applying the
smoothed group-L1 penalty (BlockSparsityPenalty) lets us identify WHOLE
inactive groups even when per-axis variance is balanced.

The pairing of "auxiliary-conditional prior" (the regression target picks
which signal must be encoded) + "group-lasso" (group-level identifiability)
is the §4(c)-style identifiability promise of the gamfit proposal. If it
works, this vindicates the proposal architecture at the cost of ARD/Ortho.

Setup
-----
- Centroids: cached PCA basis K=64 -> keep first K_PC=16 -> aux latent
  T in R^{N x 4} with groups = [[0, 1], [2, 3]].
- Three aux regression targets:
    (i)  HSV(R, G, B) regression on the named-color RGB        (purely perceptual)
    (ii) name-token proxy: (modifier_count, monoword, name_len) (purely semantic)
    (iii) concat of both                                       (mixed)
- Soft emulator of BlockSparsityPenalty: smoothed L1 on group L2 norms
    Pen(T) = w_block * sum_g sqrt(||T[:, g]||_F^2 + eps^2)
  Joint loss:
    L(T, beta) = (1/2) ||Y_aux - T @ beta||_F^2 + Pen(T)
  Optimised by 200 alternating gradient steps; w_block sweep
  {0.01, 0.1, 1.0, 10.0}; CV-select best by 5-fold-by-color.

Hypotheses (preregistered, strict TRUE/FALSE)
---------------------------------------------
(a) Y_aux = HSV  -> ||T[:, B]||_F < 0.10 * ||T[:, A]||_F
(b) Y_aux = name -> ||T[:, A]||_F < 0.10 * ||T[:, B]||_F
(c) Y_aux = both -> both groups survive: min ratio >= 0.20

Gamfit version handling
-----------------------
Installed gamfit is 0.1.112 (no _penalties module). BlockSparsityPenalty
landed post-0.1.120; this script falls back to a Python soft emulator and
records path_taken="fallback". When v0.1.121+ ships, swap the inner penalty
for `gamfit._penalties.BlockSparsityPenalty(groups=[[0,1],[2,3]], weight=w)`.

Outputs
-------
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_31.png    (4-panel)
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_31.json
"""

from __future__ import annotations

import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402
from color_manifold_gam import load_xkcd_colors  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_31.png"
OUT_JSON = OUT_DIR / "auto_exp_31.json"

K_PC          = 16
D_AUX         = 4
GROUPS        = [[0, 1], [2, 3]]      # A = perceptual, B = name-semantic
N_ITERS       = 200
W_SWEEP       = [0.01, 0.1, 1.0, 10.0]
LR            = 0.05
EPS_SMOOTH    = 1e-3                  # smoothing for group L2-norm L1
RIDGE_BETA    = 1e-4                  # tiny ridge on beta solve
N_CV_FOLDS    = 5
SEED          = 0
ACTIVE_RATIO_DEAD     = 0.10          # group "dead" if norm < this * other group
ACTIVE_RATIO_BOTH_MIN = 0.20          # both groups "alive" if min ratio >= this


# ---------------------------------------------------------------------------
# gamfit probe (will fall back; we record the version we ran against)
# ---------------------------------------------------------------------------
import gamfit  # noqa: E402

GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")


def _probe():
    try:
        from gamfit._penalties import BlockSparsityPenalty  # noqa: F401
        return {
            "reached": True,
            "detail": "imported from gamfit._penalties",
        }
    except Exception as e:
        return {
            "reached": False,
            "detail": (
                f"{type(e).__name__}: {e}; fallback: smoothed L1 on group L2 "
                "norms via numpy."
            ),
        }


PRIMITIVE_STATUS = _probe()
PATH_TAKEN = "native" if PRIMITIVE_STATUS["reached"] else "fallback"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_inputs():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    print(f"[load] X shape = ({n_total}, {H})", flush=True)
    n_raw = n_total // N_TEMPLATES

    centroids = np.zeros((n_raw, H), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_TEMPLATES + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(axis=0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    names = [n for n, *_ in kept]

    basis = load_pc_basis(K=64)
    Z16 = project(centroids, basis)[:, :K_PC]
    # standardize PC scores so the penalty magnitudes are comparable across
    # aux targets.
    Z16 = (Z16 - Z16.mean(0)) / Z16.std(0).clip(min=1e-6)

    # HSV from RGB
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    # Use the original RGB-as-perceptual coords (3 dims) — equivalent
    # subspace to HSV up to a smooth bijection and avoids the hue wrap.
    Y_hsv = np.stack([rgb[:, 0], rgb[:, 1], rgb[:, 2]], axis=1)

    # Name-token semantic targets (proxy for modifier_count / monoword /
    # template-sigma from project_cogito_color_manifold_decomposition.md).
    name_arrs = []
    for nm in names:
        n_words = len(nm.split())
        modifier_count = max(0, n_words - 1)
        monoword = 1.0 if n_words == 1 else 0.0
        name_len = float(len(nm))
        name_arrs.append([modifier_count, monoword, name_len])
    Y_name = np.array(name_arrs, dtype=np.float64)
    # standardize each Y matrix so penalty weights compare across targets
    Y_hsv = (Y_hsv - Y_hsv.mean(0)) / Y_hsv.std(0).clip(min=1e-6)
    Y_name = (Y_name - Y_name.mean(0)) / Y_name.std(0).clip(min=1e-6)

    return Z16, Y_hsv, Y_name, names


# ---------------------------------------------------------------------------
# Soft emulator of BlockSparsityPenalty (smoothed group L1)
# ---------------------------------------------------------------------------
def _group_norms(T, groups, eps=EPS_SMOOTH):
    """Smoothed L2 norm of each group block: sqrt(||T[:, g]||_F^2 + eps^2)."""
    out = np.empty(len(groups))
    for gi, g in enumerate(groups):
        out[gi] = np.sqrt((T[:, g] ** 2).sum() + eps ** 2)
    return out


def _grad_block_sparsity(T, groups, w, eps=EPS_SMOOTH):
    """d/dT of  w * sum_g sqrt(||T[:, g]||_F^2 + eps^2).
       For columns in group g, grad = w * T[:, g] / sqrt(||T[:, g]||^2 + eps^2)."""
    g_out = np.zeros_like(T)
    for g in groups:
        norm_g = np.sqrt((T[:, g] ** 2).sum() + eps ** 2)
        g_out[:, g] = w * T[:, g] / norm_g
    return g_out


def fit_block_sparse_aux(Z, Y, *, w_block, d_aux=D_AUX, groups=GROUPS,
                         n_iters=N_ITERS, lr=LR, seed=SEED,
                         train_mask=None):
    """Joint fit of T (N x d_aux) and beta (d_aux x p_y) under

        L = (1/2) || (Y - T @ beta) ||_F^2  on train rows
          + w * sum_g sqrt(||T[:, g]||_F^2 + eps^2)

    T is the latent aux block; we project the *standardized* PC scores Z
    (N x K_PC) into a d_aux-dim space via a learned linear map R (K_PC x d_aux).
    Then T = Z @ R is what we sparsify by groups.

    We optimise R (not T directly) so the resulting "axis" is a real linear
    function of features — i.e. group sparsity acts on the EMBEDDING DIMS
    rather than per-sample latents. This mirrors what a smoothed
    BlockSparsityPenalty does inside a GAM smooth's basis coefficients.
    """
    rng = np.random.default_rng(seed)
    N = Z.shape[0]
    p_y = Y.shape[1]
    if train_mask is None:
        train_mask = np.ones(N, dtype=bool)
    tr = train_mask

    # init R: small random + slight bias so groups start non-degenerate
    R = rng.normal(scale=0.1, size=(Z.shape[1], d_aux))
    beta = rng.normal(scale=0.1, size=(d_aux, p_y))

    history = {"loss": [], "group_norms": []}

    Z_tr = Z[tr]
    Y_tr = Y[tr]

    for it in range(n_iters):
        # forward
        T = Z @ R
        T_tr = T[tr]

        # closed-form beta given T (with tiny ridge)
        TtT = T_tr.T @ T_tr + RIDGE_BETA * np.eye(d_aux)
        TtY = T_tr.T @ Y_tr
        beta = np.linalg.solve(TtT, TtY)

        # gradient on R from data term  (only train rows)
        # d/dR (1/2)||Y_tr - Z_tr R beta||^2_F
        # = -Z_tr.T @ (Y_tr - Z_tr R beta) @ beta.T
        resid = Y_tr - Z_tr @ R @ beta
        grad_data = -Z_tr.T @ resid @ beta.T

        # gradient from block sparsity: penalty is on T = Z R; we approximate
        # via T-space grad pulled back through R.  grad_T -> grad_R = Z.T @ grad_T
        T_full = Z @ R
        grad_T = _grad_block_sparsity(T_full, groups, w_block)
        grad_block = Z.T @ grad_T

        # adaptive step on R
        grad_R = grad_data + grad_block
        # heuristic Lipschitz: ||Z||^2 * ||beta||^2 + w (group L1 has unit
        # subgradient bound on per-column direction)
        L_data = float((Z_tr ** 2).sum(axis=0).max() * (beta ** 2).sum())
        L_pen  = float(w_block)
        step = lr / (L_data + L_pen + 1.0)
        R = R - step * grad_R

        # track diagnostics
        T_full = Z @ R
        gn = _group_norms(T_full, groups, eps=EPS_SMOOTH)
        data_loss = 0.5 * float(((Y_tr - Z_tr @ R @ beta) ** 2).sum())
        pen_loss = float(w_block * gn.sum())
        history["loss"].append(data_loss + pen_loss)
        history["group_norms"].append(gn.tolist())

        if it % 50 == 0 or it == n_iters - 1:
            print(f"  [w={w_block:>6.3f}] iter {it:3d}  "
                  f"loss={data_loss + pen_loss:.4f}  group_norms={gn.tolist()}",
                  flush=True)

    return {
        "R": R,
        "beta": beta,
        "T": Z @ R,
        "group_norms": _group_norms(Z @ R, groups, eps=EPS_SMOOTH),
        "history": history,
    }


def cv_score(Z, Y, w_block, *, n_folds=N_CV_FOLDS, seed=SEED):
    N = Z.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    fold_sizes = np.full(n_folds, N // n_folds)
    fold_sizes[: N % n_folds] += 1
    cuts = np.cumsum(fold_sizes)
    starts = np.concatenate([[0], cuts[:-1]])

    scores = []
    for f in range(n_folds):
        test_idx = perm[starts[f]:cuts[f]]
        mask = np.ones(N, dtype=bool)
        mask[test_idx] = False
        out = fit_block_sparse_aux(Z, Y, w_block=w_block, train_mask=mask,
                                   n_iters=N_ITERS, seed=seed + f)
        T_te = Z[test_idx] @ out["R"]
        Y_te_hat = T_te @ out["beta"]
        ss_res = float(((Y[test_idx] - Y_te_hat) ** 2).sum())
        ss_tot = float(((Y[test_idx] - Y[test_idx].mean(0)) ** 2).sum())
        scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(scores)), scores


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------
def verdict_dead_group(norms, dead_idx):
    """Group `dead_idx` should be ~zero relative to the other."""
    other_idx = 1 - dead_idx
    if norms[other_idx] < 1e-9:
        return False
    ratio = norms[dead_idx] / norms[other_idx]
    return bool(ratio < ACTIVE_RATIO_DEAD), float(ratio)


def verdict_both_alive(norms):
    mn, mx = float(min(norms)), float(max(norms))
    if mx < 1e-9:
        return False, 0.0
    return bool(mn / mx >= ACTIVE_RATIO_BOTH_MIN), float(mn / mx)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[gamfit] version={GAMFIT_VERSION}  path_taken={PATH_TAKEN}",
          flush=True)
    print(f"[probe]  BlockSparsityPenalty: {PRIMITIVE_STATUS}", flush=True)

    Z, Y_hsv, Y_name, names = build_inputs()
    Y_both = np.concatenate([Y_hsv, Y_name], axis=1)
    print(f"[data] Z={Z.shape}  Y_hsv={Y_hsv.shape}  Y_name={Y_name.shape}  "
          f"Y_both={Y_both.shape}", flush=True)

    targets = {
        "hsv":  Y_hsv,
        "name": Y_name,
        "both": Y_both,
    }

    results = {}
    for tname, Y in targets.items():
        print(f"\n=== aux target: {tname} ===", flush=True)
        per_w = []
        for w in W_SWEEP:
            cv_mean, cv_folds = cv_score(Z, Y, w_block=w)
            # full fit at this w to get final group norms + history
            full = fit_block_sparse_aux(Z, Y, w_block=w)
            per_w.append({
                "w": w,
                "cv_mean": cv_mean,
                "cv_folds": cv_folds,
                "group_norms": full["group_norms"].tolist(),
                "history_loss": full["history"]["loss"],
                "history_gn": full["history"]["group_norms"],
            })
            print(f"  -> w={w}: CV R^2={cv_mean:.4f}  "
                  f"final_group_norms={full['group_norms'].tolist()}",
                  flush=True)
        best = max(per_w, key=lambda r: r["cv_mean"])
        results[tname] = {"per_w": per_w, "best": best}

    # verdicts
    norms_hsv = results["hsv"]["best"]["group_norms"]
    norms_name = results["name"]["best"]["group_norms"]
    norms_both = results["both"]["best"]["group_norms"]

    # (a) HSV target -> Group B (idx 1) dead relative to Group A (idx 0)
    verdict_a, ratio_a = verdict_dead_group(norms_hsv, dead_idx=1)
    # (b) name target -> Group A (idx 0) dead relative to Group B (idx 1)
    verdict_b, ratio_b = verdict_dead_group(norms_name, dead_idx=0)
    # (c) both target -> both alive
    verdict_c, ratio_c = verdict_both_alive(norms_both)

    print("\n=== VERDICTS ===", flush=True)
    print(f" (a) HSV  prunes B: {verdict_a}  (ratio_B/A={ratio_a:.3f})", flush=True)
    print(f" (b) Name prunes A: {verdict_b}  (ratio_A/B={ratio_b:.3f})", flush=True)
    print(f" (c) Both alive   : {verdict_c}  (min/max={ratio_c:.3f})", flush=True)

    # ---- plot 4 panels ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel (0,0): per-group L2 norms vs w_block for each aux target
    ax = axes[0, 0]
    colors_per_target = {"hsv": "tab:blue", "name": "tab:orange", "both": "tab:green"}
    for tname in ["hsv", "name", "both"]:
        ws = [r["w"] for r in results[tname]["per_w"]]
        gA = [r["group_norms"][0] for r in results[tname]["per_w"]]
        gB = [r["group_norms"][1] for r in results[tname]["per_w"]]
        c = colors_per_target[tname]
        ax.plot(ws, gA, "-o", color=c, label=f"{tname}: Group A")
        ax.plot(ws, gB, "--s", color=c, label=f"{tname}: Group B")
    ax.set_xscale("log")
    ax.set_xlabel("w_block")
    ax.set_ylabel("group L2 norm")
    ax.set_title("Per-group ‖T[:, g]‖_F vs w_block")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7, ncol=2)

    # Panel (0,1): CV R^2 traces per aux target
    ax = axes[0, 1]
    for tname in ["hsv", "name", "both"]:
        ws = [r["w"] for r in results[tname]["per_w"]]
        cv = [r["cv_mean"] for r in results[tname]["per_w"]]
        ax.plot(ws, cv, "-o", color=colors_per_target[tname], label=tname)
        best_w = results[tname]["best"]["w"]
        best_cv = results[tname]["best"]["cv_mean"]
        ax.scatter([best_w], [best_cv], s=120,
                   facecolor="none", edgecolor=colors_per_target[tname], lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("w_block")
    ax.set_ylabel("5-fold CV R^2")
    ax.set_title("CV score (circle = chosen w_block)")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    # Panel (1,0): group-norm history at best w for each target
    ax = axes[1, 0]
    for tname in ["hsv", "name", "both"]:
        gn_hist = np.array(results[tname]["best"]["history_gn"])  # (iters, 2)
        c = colors_per_target[tname]
        ax.plot(gn_hist[:, 0], "-", color=c, label=f"{tname}: A")
        ax.plot(gn_hist[:, 1], "--", color=c, label=f"{tname}: B")
    ax.set_xlabel("iter")
    ax.set_ylabel("group L2 norm")
    ax.set_title("Group norms during fit @ best w_block")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7, ncol=2)

    # Panel (1,1): verdict bar chart
    ax = axes[1, 1]
    bar_labels = [
        "(a) HSV prunes B",
        "(b) Name prunes A",
        "(c) Both alive",
    ]
    bar_ratios = [ratio_a, ratio_b, ratio_c]
    bar_colors = [("tab:green" if v else "tab:red")
                  for v in (verdict_a, verdict_b, verdict_c)]
    ax.bar(range(3), bar_ratios, color=bar_colors)
    ax.axhline(ACTIVE_RATIO_DEAD, color="k", ls="--", alpha=0.4,
               label=f"dead thr={ACTIVE_RATIO_DEAD}")
    ax.axhline(ACTIVE_RATIO_BOTH_MIN, color="k", ls=":", alpha=0.4,
               label=f"alive thr={ACTIVE_RATIO_BOTH_MIN}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(bar_labels, fontsize=8, rotation=10)
    ax.set_ylabel("norm ratio (smaller / larger)")
    ax.set_title(
        f"Verdicts  ·  gamfit={GAMFIT_VERSION}  path={PATH_TAKEN}\n"
        f"a={verdict_a}  b={verdict_b}  c={verdict_c}"
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.4, axis="y")

    fig.suptitle(
        "auto_exp_31  ·  BlockSparsityPenalty (smoothed group-L1) on cogito L40\n"
        "Groups = [[0,1] perceptual, [2,3] name-semantic].  "
        "Aux targets: HSV / name-tokens / both.",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_PNG}", flush=True)

    runtime = time.time() - t0

    summary = {
        "gamfit_version": GAMFIT_VERSION,
        "path_taken": PATH_TAKEN,
        "primitive_status": PRIMITIVE_STATUS,
        "k_pc": K_PC,
        "d_aux": D_AUX,
        "groups": GROUPS,
        "w_sweep": W_SWEEP,
        "n_iters": N_ITERS,
        "n_cv_folds": N_CV_FOLDS,
        "n_colors": int(Z.shape[0]),
        "best_w_per_aux": {
            "hsv":  results["hsv"]["best"]["w"],
            "name": results["name"]["best"]["w"],
            "both": results["both"]["best"]["w"],
        },
        "cv_r2_per_aux_at_best": {
            "hsv":  results["hsv"]["best"]["cv_mean"],
            "name": results["name"]["best"]["cv_mean"],
            "both": results["both"]["best"]["cv_mean"],
        },
        "group_A_norm_per_aux": {
            "hsv":  norms_hsv[0],
            "name": norms_name[0],
            "both": norms_both[0],
        },
        "group_B_norm_per_aux": {
            "hsv":  norms_hsv[1],
            "name": norms_name[1],
            "both": norms_both[1],
        },
        "hypothesis_verdicts": {
            "a_hsv_prunes_B":  {"verdict": verdict_a, "ratio_B_over_A": ratio_a,
                                "threshold": ACTIVE_RATIO_DEAD},
            "b_name_prunes_A": {"verdict": verdict_b, "ratio_A_over_B": ratio_b,
                                "threshold": ACTIVE_RATIO_DEAD},
            "c_both_alive":    {"verdict": verdict_c, "min_over_max": ratio_c,
                                "threshold": ACTIVE_RATIO_BOTH_MIN},
        },
        "per_w_full": {
            tname: [
                {
                    "w": r["w"],
                    "cv_mean": r["cv_mean"],
                    "cv_folds": r["cv_folds"],
                    "group_norms": r["group_norms"],
                }
                for r in results[tname]["per_w"]
            ]
            for tname in ["hsv", "name", "both"]
        },
        "runtime_seconds": runtime,
        "prediction_slot_v121": {
            "note": (
                "When gamfit >= 0.1.121 ships BlockSparsityPenalty, re-run "
                "with `from gamfit._penalties import BlockSparsityPenalty; "
                "pen = BlockSparsityPenalty(groups=[[0,1], [2,3]], weight=w)` "
                "in place of the smoothed-L1 fallback. Expected: same verdicts."
            ),
            "fallback_verdicts": {
                "a": verdict_a, "b": verdict_b, "c": verdict_c,
            },
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[save] {OUT_JSON}", flush=True)
    print(f"[done] runtime = {runtime:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
