"""auto_exp_20 - IBP-MAP K-pruning robustness across K_init.

GOAL
----
Stress-test the auto_exp_18 prediction (K=2 effective atoms: perceptual +
name-semantic) by sweeping the initial atom count K_init in
{2, 4, 8, 16, 32, 50} and asking whether an IBP-MAP-style sparse-prior
assignment consistently prunes back to K_eff ~= 2.

If yes, the prediction for what `sae_manifold_fit(..., assignment_prior=
"ibp_map")` should produce on this cogito L40 dataset is robust to the
chosen over-parameterization, and falsifying it just requires the
v0.1.117 wheels to land and disagree at any one K_init.

DESIGN
------
1. Load cogito L40 centroids the same way auto_exp_18 / auto_exp_19 do:
   mmap=r on X_L40.npy, average over TOP_TEMPLATES, filter to 886, project
   via cached load_pc_basis(K=64) and keep the top-16 PCs.
2. For each K_init: emulate IBP-MAP by
     - init K_init random unit vectors in R^16
     - alternate (a) per-row soft assignment via softmax over
       similarity * inv_temp - sparsity_penalty, (b) atom re-estimate
       as a mass-weighted mean of rows, re-normalized to unit norm
     - count atoms with row-mass fraction >= 5% as "effective"
3. Plot K_eff vs K_init -> expect a flat line ~2.
4. For each K_init, identify the top-2 effective atoms (by mass), score
   each row's saturation, and visualize the mean RGB swatch + the top
   atom's perceptual/achromatic profile to confirm atom-0 = perceptual,
   atom-1 = name-semantic.
5. JSON has a `v0_1_117_actual` slot, null until the wheels land.

If gamfit.sae_manifold_fit is importable (HAS_NEW_API), additionally run
it once at K_init=8 as a cross-check.

RAM
---
- mmap=r on X_L40.npy
- cached load_pc_basis(K=64)
- No K_PC > 16 retained, no full residual stack
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

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_20.png"
OUT_JSON = OUT_DIR / "auto_exp_20.json"

K_PC          = 16
K_INIT_SWEEP  = [2, 4, 8, 16, 32, 50]
N_ITERS       = 80
SOFTMAX_TEMP  = 0.20            # inv-temp = 1/temp; smaller temp -> sharper
# IBP-MAP-style absolute Beta-Bernoulli concentration. Higher = more
# evidence the prior demands before keeping an atom around.
IBP_ALPHA     = 8.0
MIN_TEMP      = 0.04
# Kill atoms whose absolute mass falls below this fraction of N. With
# N=886 and KILL_FRAC=0.04, an atom needs >= 35 rows to survive --
# matches the IBP-MAP "minimum customer count to keep a dish" intuition.
KILL_FRAC     = 0.04
MASS_FRAC_THR = 0.05            # >=5% of total row mass to count as effective
SEED          = 0
CROSS_CHECK_KINIT = 8

# Auto_exp_18 prediction we are stress-testing
PRED_K = 2


# ---------------------------------------------------------------------------
# Optional new-API hookup
# ---------------------------------------------------------------------------
HAS_NEW_API = False
NEW_API_REASON = ""
try:
    from gamfit import sae_manifold_fit  # noqa: F401
    HAS_NEW_API = True
    NEW_API_REASON = "sae_manifold_fit present"
except (ImportError, AttributeError) as e:
    NEW_API_REASON = f"{type(e).__name__}: {e}"

import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
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
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, names, rgb, hsv


# ---------------------------------------------------------------------------
# IBP-MAP emulator
# ---------------------------------------------------------------------------
def ibp_map_emulate(Z, K_init, seed=0,
                    n_iters=N_ITERS,
                    temp=SOFTMAX_TEMP,
                    alpha=IBP_ALPHA,
                    kill_frac=KILL_FRAC,
                    mass_thr=MASS_FRAC_THR):
    """
    Emulate IBP-MAP sparse-assignment decomposition with hard pruning.

    The IBP prior (Indian Buffet Process) places a Beta(alpha/K, 1) prior
    on each atom's activation rate; the MAP under this prior penalizes
    keeping rarely-used atoms. We emulate it as:

      score[i,k] = (Z_unit[i] . A[k]) / temp + log(pi_k)
      pi_k       = (mass_k + 1e-3) / (N + alpha)       (Beta-Bernoulli MAP)
      assign[i,k]= softmax_k(score[i,k])

    Per iteration, atoms whose absolute soft-count `mass_k` drops below
    `kill_thr` are PRUNED (mass forced to 0, removed from the assignment
    pool). This is the hard "MAP collapse" the production IBP-MAP code
    is expected to perform.

    Returns dict with mass_share (over the final ALIVE atoms slot-aligned
    to K_init, zero where killed), K_eff, assign matrix, and atoms.
    """
    rng = np.random.default_rng(seed)
    N, D = Z.shape

    Zc = Z - Z.mean(axis=0, keepdims=True)
    row_norm = np.linalg.norm(Zc, axis=1, keepdims=True).clip(min=1e-12)
    Zu = Zc / row_norm

    A = rng.normal(size=(K_init, D))
    A /= np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)

    alive = np.ones(K_init, dtype=bool)
    mass = np.full(K_init, N / K_init, dtype=np.float64)
    assign = np.zeros((N, K_init), dtype=np.float64)
    kill_thr = kill_frac * N

    for it in range(n_iters):
        if alive.sum() == 0:
            break
        # Beta-Bernoulli MAP prior on per-atom activation rate.
        pi = (mass + 1e-3) / (N + alpha)
        log_pi = np.log(pi.clip(min=1e-12))

        sim = Zu @ A.T                                # (N, K)
        score = sim / temp + log_pi[None, :]
        # Mask dead atoms with -inf so they cannot win the softmax.
        score[:, ~alive] = -1e18

        score = score - score.max(axis=1, keepdims=True)
        ex = np.exp(score)
        ex[:, ~alive] = 0.0
        denom = ex.sum(axis=1, keepdims=True).clip(min=1e-30)
        assign = ex / denom

        mass = assign.sum(axis=0)                     # absolute soft-counts

        # Hard MAP kill: prune atoms with negligible mass.
        new_alive = alive & (mass >= kill_thr)
        # If killing would zero everything, keep the top-1.
        if not new_alive.any():
            new_alive = np.zeros(K_init, dtype=bool)
            new_alive[int(np.argmax(mass))] = True
        alive = new_alive

        # Re-estimate atoms among alive set.
        new_A = A.copy()
        for k in np.where(alive)[0]:
            v = assign[:, k:k+1] * Zu
            m_vec = v.sum(axis=0)
            nm = np.linalg.norm(m_vec)
            if nm > 1e-12:
                new_A[k] = m_vec / nm
        A = new_A

        # Anneal temp during the second half.
        if it >= n_iters // 2:
            temp = max(temp * 0.95, MIN_TEMP)

    # Final assignment.
    pi = (mass + 1e-3) / (N + alpha)
    log_pi = np.log(pi.clip(min=1e-12))
    sim = Zu @ A.T
    score = sim / temp + log_pi[None, :]
    score[:, ~alive] = -1e18
    score = score - score.max(axis=1, keepdims=True)
    ex = np.exp(score)
    ex[:, ~alive] = 0.0
    denom = ex.sum(axis=1, keepdims=True).clip(min=1e-30)
    assign = ex / denom
    mass = assign.sum(axis=0)
    mass_share = mass / mass.sum() if mass.sum() > 0 else mass
    K_eff = int((mass_share >= mass_thr).sum())

    return {
        "K_init": int(K_init),
        "K_eff": K_eff,
        "n_alive": int(alive.sum()),
        "mass_share": mass_share,
        "assign": assign,
        "atoms": A,
    }


def variance_rank_effective_K(Z, K_init, seed=0, var_frac_thr=0.05):
    """
    Secondary, more well-behaved "effective K" measure that complements
    the IBP-MAP emulator: fit K_init atoms with plain soft K-means (no
    sparse prior), rank atoms by the variance of (assign[:, k] * proj),
    and count atoms whose explained-variance fraction >= var_frac_thr.

    This mimics IBP-MAP's "rank by activation magnitude, then prune the
    long tail" step that production code performs after MAP fitting.
    Returns: (K_eff_var, var_share_sorted_desc, assign, atoms).
    """
    rng = np.random.default_rng(seed)
    N, D = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)
    A = rng.normal(size=(K_init, D))
    A /= np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)

    temp = 0.15
    for it in range(50):
        sim = (Zc / np.linalg.norm(Zc, axis=1, keepdims=True).clip(min=1e-12)
               ) @ A.T
        score = sim / temp
        score = score - score.max(axis=1, keepdims=True)
        ex = np.exp(score)
        assign = ex / ex.sum(axis=1, keepdims=True).clip(min=1e-30)
        for k in range(K_init):
            w = assign[:, k:k+1]
            wsum = w.sum()
            if wsum > 1e-9:
                v = (w * Zc).sum(axis=0)
                nm = np.linalg.norm(v)
                if nm > 1e-12:
                    A[k] = v / nm
        if it >= 25:
            temp = max(temp * 0.95, 0.05)

    # Per-atom explained variance: project each assigned slice onto its atom.
    var_per_atom = np.zeros(K_init)
    for k in range(K_init):
        w = assign[:, k]
        if w.sum() < 1e-6:
            continue
        proj = (Zc @ A[k])           # scalar projection per row
        var_per_atom[k] = float(np.sum(w * proj * proj))
    total_var = var_per_atom.sum()
    var_share = (var_per_atom / total_var) if total_var > 0 else var_per_atom
    K_eff_var = int((var_share >= var_frac_thr).sum())
    return K_eff_var, np.sort(var_share)[::-1], assign, A


def per_atom_rgb_swatch(rgb, assign_col):
    """Mean RGB weighted by per-row assignment mass to one atom."""
    w = assign_col.reshape(-1, 1)
    s = w.sum()
    if s <= 1e-9:
        return np.array([0.5, 0.5, 0.5])
    return (rgb * w).sum(axis=0) / s


def per_atom_saturation(hsv, assign_col):
    """Mass-weighted mean HSV saturation: ~0 = name-semantic / achromatic,
       ~1 = perceptual / vivid color."""
    w = assign_col
    s = w.sum()
    if s <= 1e-9:
        return float("nan")
    return float((hsv[:, 1] * w).sum() / s)


def classify_top2(result, rgb, hsv):
    """Return the (perc_idx, name_idx, perc_swatch, name_swatch,
       perc_sat, name_sat) for the two top-mass effective atoms.
       If only one atom is effective, the second is reported as None."""
    mass = result["mass_share"]
    K_init = result["K_init"]
    K_eff = result["K_eff"]
    order = np.argsort(-mass)
    top = [int(k) for k in order if mass[k] >= MASS_FRAC_THR][:2]
    while len(top) < 2:
        top.append(None)

    swatches = []
    sats = []
    for k in top:
        if k is None:
            swatches.append(None); sats.append(None)
        else:
            sw = per_atom_rgb_swatch(rgb, result["assign"][:, k])
            sa = per_atom_saturation(hsv, result["assign"][:, k])
            swatches.append(sw); sats.append(sa)

    # Choose perc vs name by saturation: higher sat = perceptual.
    perc_slot, name_slot = 0, 1
    if (sats[0] is not None and sats[1] is not None and
            sats[1] > sats[0]):
        perc_slot, name_slot = 1, 0

    return {
        "atom_perc_idx":     top[perc_slot],
        "atom_name_idx":     top[name_slot],
        "atom_perc_swatch":  swatches[perc_slot],
        "atom_name_swatch":  swatches[name_slot],
        "atom_perc_sat":     sats[perc_slot],
        "atom_name_sat":     sats[name_slot],
        "atom_perc_mass":    (float(result["mass_share"][top[perc_slot]])
                              if top[perc_slot] is not None else None),
        "atom_name_mass":    (float(result["mass_share"][top[name_slot]])
                              if top[name_slot] is not None else None),
        "K_eff":             K_eff,
        "K_init":            K_init,
    }


# ---------------------------------------------------------------------------
# Optional new-API cross-check
# ---------------------------------------------------------------------------
def try_sae_manifold(Z, K_init):
    """Call sae_manifold_fit with assignment_prior='ibp_map' if available."""
    try:
        from gamfit import sae_manifold_fit  # noqa: F811
        result = sae_manifold_fit(
            Z,
            n_atoms=K_init,
            atom_basis=["periodic"] + ["duchon"] * (K_init - 1),
            atom_dim=[3] * K_init,
            sparsity_strength="auto",
            smoothness="auto",
            assignment_prior="ibp_map",
            alpha="auto",
        )
        z_act = np.asarray(getattr(result, "z",
                                   getattr(result, "Z", None)))
        atoms_active = (z_act > 0.5).any(axis=0)
        return {"ok": True,
                "K_init": int(K_init),
                "n_atoms_active": int(atoms_active.sum()),
                "atoms_active_mask": atoms_active.tolist(),
                "error": None}
    except Exception as e:
        return {"ok": False,
                "K_init": int(K_init),
                "n_atoms_active": None,
                "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    print(f"[api ] HAS_NEW_API = {HAS_NEW_API}  ({NEW_API_REASON})",
          flush=True)

    centroids, names, rgb, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC]
    evr = float(basis["evr"][:K_PC].sum())
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = {evr:.3f}",
          flush=True)

    # Sweep K_init
    sweep_results = []
    sweep_class = []
    var_rank_K = []
    var_rank_top = []
    for K_init in K_INIT_SWEEP:
        print(f"[ibp ] K_init = {K_init} ...", flush=True)
        res = ibp_map_emulate(Z, K_init, seed=SEED)
        cls = classify_top2(res, rgb, hsv)
        print(f"         IBP K_eff = {res['K_eff']}  "
              f"top-2 mass = "
              f"{sorted(res['mass_share'], reverse=True)[:2]}  "
              f"perc_sat = {cls['atom_perc_sat']}  "
              f"name_sat = {cls['atom_name_sat']}",
              flush=True)
        sweep_results.append(res)
        sweep_class.append(cls)

        K_var, var_share, _, _ = variance_rank_effective_K(
            Z, K_init, seed=SEED)
        var_rank_K.append(K_var)
        var_rank_top.append([float(x) for x in var_share[:5]])
        print(f"         VAR K_eff = {K_var}  "
              f"top-5 var-share = {var_rank_top[-1]}",
              flush=True)

    # Optional cross-check
    cross_check = None
    if HAS_NEW_API:
        print(f"[api ] cross-check sae_manifold_fit at K_init="
              f"{CROSS_CHECK_KINIT}", flush=True)
        cross_check = try_sae_manifold(Z, CROSS_CHECK_KINIT)
        print(f"[api ] cross-check = {cross_check}", flush=True)

    # ---------------- Plot ----------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, len(K_INIT_SWEEP),
                          height_ratios=[1.3, 0.6, 0.9],
                          hspace=0.55, wspace=0.35)

    # Top row spans all columns: K_eff vs K_init
    ax0 = fig.add_subplot(gs[0, :])
    K_eff_arr = [r["K_eff"] for r in sweep_results]
    ax0.plot(K_INIT_SWEEP, K_eff_arr, "o-",
             ms=10, lw=2, color="#1f77b4",
             label="K_eff (emulated IBP-MAP, mass>=5%)")
    ax0.plot(K_INIT_SWEEP, var_rank_K, "s--",
             ms=9, lw=1.6, color="#2ca02c",
             label="K_eff (variance-rank, share>=5%)")
    ax0.axhline(PRED_K, color="#d62728", ls="--", lw=1.2,
                label=f"auto_exp_18 prediction: K={PRED_K}")
    ax0.plot(K_INIT_SWEEP, K_INIT_SWEEP, "k:", alpha=0.4,
             label="y = x (no pruning)")
    if cross_check and cross_check.get("ok"):
        ax0.scatter([CROSS_CHECK_KINIT],
                    [cross_check["n_atoms_active"]],
                    marker="*", s=260, color="gold",
                    edgecolor="black", lw=1, zorder=5,
                    label=f"v0.1.117+ sae_manifold_fit @ K_init="
                          f"{CROSS_CHECK_KINIT}")
    ax0.set_xlabel("K_init")
    ax0.set_ylabel("K_eff (atoms with >= 5% row-mass share)")
    ax0.set_xticks(K_INIT_SWEEP)
    ax0.set_yticks(np.arange(0, max(K_INIT_SWEEP) + 1,
                             max(1, max(K_INIT_SWEEP) // 10)))
    ax0.set_ylim(0, max(K_INIT_SWEEP) + 2)
    ax0.set_title("IBP-MAP pruning robustness: K_eff vs K_init "
                  "(flat line at K=2 == prediction holds)",
                  fontsize=11)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="upper left", fontsize=9)

    # Middle row: per-K_init RGB swatch pair (atom_perc | atom_name)
    for i, (K_init, cls) in enumerate(zip(K_INIT_SWEEP, sweep_class)):
        ax = fig.add_subplot(gs[1, i])
        sw = np.ones((1, 2, 3))
        sw[0, 0] = (cls["atom_perc_swatch"]
                    if cls["atom_perc_swatch"] is not None
                    else [0.85, 0.85, 0.85])
        sw[0, 1] = (cls["atom_name_swatch"]
                    if cls["atom_name_swatch"] is not None
                    else [0.85, 0.85, 0.85])
        ax.imshow(sw, aspect="equal", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["perc", "name"], fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"K_init={K_init}\nK_eff={cls['K_eff']}",
                     fontsize=9)

    # Bottom row: per-K_init saturation bar (perc vs name)
    for i, (K_init, cls) in enumerate(zip(K_INIT_SWEEP, sweep_class)):
        ax = fig.add_subplot(gs[2, i])
        sats = [cls["atom_perc_sat"] if cls["atom_perc_sat"] is not None
                else 0.0,
                cls["atom_name_sat"] if cls["atom_name_sat"] is not None
                else 0.0]
        ax.bar(["perc", "name"], sats,
               color=["#2ca02c", "#9467bd"],
               edgecolor="black", lw=0.6)
        for j, v in enumerate(sats):
            ax.text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("mass-w. mean sat" if i == 0 else "")
        ax.set_title(f"K_init={K_init}", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    title = (f"auto_exp_20 . IBP-MAP K-pruning robustness on cogito L40 "
             f"(gamfit=={GAMFIT_VERSION}, HAS_NEW_API={HAS_NEW_API})\n"
             f"N={N}, K_PC={K_PC}  |  prediction (auto_exp_18): K_eff=2 "
             f"across all K_init in {K_INIT_SWEEP}")
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[plot] -> {OUT_PNG}", flush=True)

    # ---------------- JSON ----------------
    def _swatch(s):
        return None if s is None else [float(s[0]), float(s[1]), float(s[2])]

    per_kinit = []
    for K_init, res, cls, Kv, vt in zip(
            K_INIT_SWEEP, sweep_results, sweep_class,
            var_rank_K, var_rank_top):
        per_kinit.append({
            "K_init":            int(K_init),
            "K_eff":             int(res["K_eff"]),
            "K_eff_var_rank":    int(Kv),
            "var_share_top5":    vt,
            "mass_share_sorted_desc": [
                float(x) for x in sorted(res["mass_share"], reverse=True)],
            "atom_perc_idx":     cls["atom_perc_idx"],
            "atom_name_idx":     cls["atom_name_idx"],
            "atom_perc_swatch":  _swatch(cls["atom_perc_swatch"]),
            "atom_name_swatch":  _swatch(cls["atom_name_swatch"]),
            "atom_perc_sat":     cls["atom_perc_sat"],
            "atom_name_sat":     cls["atom_name_sat"],
            "atom_perc_mass":    cls["atom_perc_mass"],
            "atom_name_mass":    cls["atom_name_mass"],
        })

    K_eff_constant = (len(set(K_eff_arr)) == 1)
    K_var_constant = (len(set(var_rank_K)) == 1)
    K_var_modal = int(max(set(var_rank_K), key=var_rank_K.count))
    all_perc_sat_higher = all(
        (cls["atom_perc_sat"] is not None and
         cls["atom_name_sat"] is not None and
         cls["atom_perc_sat"] > cls["atom_name_sat"])
        for cls in sweep_class)

    summary = {
        "experiment": "auto_exp_20",
        "question": ("Does IBP-MAP-style sparse-assignment pruning land "
                     "on K_eff~=2 regardless of K_init? (stress-test of "
                     "auto_exp_18's K=2 prediction)"),
        "gamfit_version": GAMFIT_VERSION,
        "has_new_api": HAS_NEW_API,
        "new_api_reason": NEW_API_REASON,
        "config": {
            "K_PC":           K_PC,
            "K_init_sweep":   K_INIT_SWEEP,
            "n_iters":        N_ITERS,
            "softmax_temp":   SOFTMAX_TEMP,
            "ibp_alpha":      IBP_ALPHA,
            "kill_frac":      KILL_FRAC,
            "min_temp":       MIN_TEMP,
            "mass_frac_thr":  MASS_FRAC_THR,
            "n_colors":       int(N),
            "evr_top_K_PC":   evr,
            "cross_check_K_init": CROSS_CHECK_KINIT,
        },
        "auto_exp_18_prediction": {
            "K_eff_predicted":            PRED_K,
            "atom_0_role":                "perceptual (hue+sv)",
            "atom_1_role":                "name-semantic (achromatic / envelope)",
        },
        "per_kinit_results": per_kinit,
        "K_eff_constant_across_sweep": K_eff_constant,
        "K_eff_modal_value": int(max(set(K_eff_arr), key=K_eff_arr.count)),
        "K_eff_var_rank_constant_across_sweep": K_var_constant,
        "K_eff_var_rank_modal_value": K_var_modal,
        "all_perc_sat_higher_than_name": all_perc_sat_higher,
        "sae_manifold_cross_check": cross_check,
        "v0_1_117_actual": None,
        "verdict": {
            "robust_K2_prediction": bool(
                K_eff_constant and K_eff_arr[0] == PRED_K
                and all_perc_sat_higher),
            "comment": (
                "Emulated IBP-MAP pruning lands on a single K_eff value of "
                f"{max(set(K_eff_arr), key=K_eff_arr.count)} across "
                f"K_init in {K_INIT_SWEEP}. "
                "If v0.1.117 sae_manifold_fit(assignment_prior='ibp_map') "
                "reports n_atoms_active != that K_eff at any K_init, "
                "either the sparse prior in production uses a stricter "
                "complexity penalty, or the cogito L40 manifold "
                "supports more atoms than our centroid-only emulator "
                "can see."
                if K_eff_constant else
                "Emulator did NOT collapse to a constant K_eff across the "
                "sweep; auto_exp_18's K=2 claim is therefore not yet "
                "robust under this stress test - investigate per-K_init "
                "rows in `per_kinit_results`."
            ),
            "falsification_when_wheels_land": (
                "Fill in v0_1_117_actual with the n_atoms_active reported "
                "by sae_manifold_fit at each K_init; any disagreement with "
                "the per_kinit_results.K_eff column falsifies the "
                "stress-tested prediction."
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
