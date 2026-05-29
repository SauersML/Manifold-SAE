"""auto_70.py — Discover the cogito color circle topologically.

Uses the de Silva–Morozov–Vejdemo-Johansson circular-coordinates algorithm
(persistent cohomology + harmonic smoothing, implemented in dreimac's
CircularCoords) to recover a circle-valued coordinate θ_disc ∈ S¹ from
cogito L40 activations alone — no RGB / hue / saturation labels enter the
discovery.

We then check whether θ_disc tracks the xkcd ground-truth hue (perceptual
color wheel) up to the residual S¹ gauge (rotation + handedness sign), and
whether a 1D periodic Duchon on θ_disc recovers as much CV R² as a 1D
periodic Duchon on the ground-truth hue (auto_66: 0.251).

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import colorsys
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
N_FOLDS = 5
K_PC = 64
N_LANDMARKS = 200
HUE_CENTERS = 40

# Reference numbers from prior runs (kept as constants for the bar plot).
R2_AUTO66_HUE = 0.251       # 1D periodic Duchon on prescribed hue
R2_AUTO67_JOINT = 0.321     # γ(hue)+g(sat,val) joint


def _ensure_dreimac():
    try:
        import dreimac  # noqa: F401
        return
    except ImportError:
        print("[install] dreimac not found; uv pip install dreimac")
        subprocess.check_call(["uv", "pip", "install", "dreimac"])
        import dreimac  # noqa: F401


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def hue_basis(hue01, n_centers=HUE_CENTERS):
    """1D periodic Duchon m=2 (auto_66 / auto_67 convention)."""
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    )
    return Phi, P, centers


def reml_fit(Phi, Y, P):
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return np.asarray(out["coefficients"]), float(out["lambda"])


def circ_dist(a, b):
    """Signed minimal angular distance, output in (-pi, pi]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def best_align(theta_disc, hue_true_rad):
    """Find optimal (sign, rotation) so that sign*theta + alpha ≈ hue_true.

    We use the cos/sin circular Pearson correlation as the alignment score:
        score(s, a) = corr_pearson([cos(s*θ - a), sin(s*θ - a)],
                                    [cos(h),         sin(h)])
    summed across the (cos, sin) pair (equivalent to Re of complex inner
    product up to a constant). For a true 1-1 alignment this peaks near +1.
    """
    alphas = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    best = (-np.inf, 0.0, +1)
    for s in (+1, -1):
        th = s * theta_disc
        # Evaluate corr for each alpha
        for a in alphas:
            shifted = (th - a + np.pi) % (2 * np.pi) - np.pi
            # circular Pearson on circle: use complex correlation magnitude
            z1 = np.exp(1j * shifted)
            z2 = np.exp(1j * hue_true_rad)
            num = np.abs((z1 * np.conj(z2)).mean())
            score = float(num)  # in [0,1]; larger = better aligned
            if score > best[0]:
                best = (score, float(a), int(s))
    return best  # (score, alpha_star, sign)


def circ_corr_jammalamadaka(alpha, beta):
    """Jammalamadaka–Sarma circular-circular correlation coefficient.

    Returns a value in [-1, +1]. Invariant under rotations of either
    variable (so it equals the post-alignment correlation as long as we
    first multiply θ_disc by the chosen sign).
    """
    a_bar = np.angle(np.exp(1j * alpha).mean())
    b_bar = np.angle(np.exp(1j * beta).mean())
    sa = np.sin(alpha - a_bar)
    sb = np.sin(beta - b_bar)
    num = (sa * sb).sum()
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_dreimac()
    from dreimac import CircularCoords

    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[load] X mmap shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = len(kept)
    print(f"[load] N={N} filtered colors")

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    print(f"[load] Z shape={Z.shape}  EVR_top{K_PC}={float(basis['evr'].sum()):.3f}")

    # ----- Circular coordinates from persistent cohomology -----
    print(f"[dreimac] CircularCoords on Z (N={N}, d={K_PC}), "
          f"n_landmarks={N_LANDMARKS}")
    cc = CircularCoords(Z, n_landmarks=N_LANDMARKS, prime=41, maxdim=1,
                        verbose=False)
    try:
        theta_disc = np.asarray(cc.get_coordinates(perc=0.5, cocycle_idx=0))
        used_standard_range = True
    except Exception as e:
        # Short cohomology class — drop the standard_range guard
        print(f"[dreimac] standard_range failed ({e}); retrying with "
              f"standard_range=False")
        theta_disc = np.asarray(cc.get_coordinates(
            perc=0.5, cocycle_idx=0, standard_range=False))
        used_standard_range = False
    print(f"[dreimac] theta_disc shape={theta_disc.shape}  "
          f"range=[{theta_disc.min():.3f}, {theta_disc.max():.3f}]")

    # H¹ persistence ratio of the picked class (top1 lifetime / top2 lifetime)
    dgm_h1 = np.asarray(cc.dgms_[1])
    lifetimes = dgm_h1[:, 1] - dgm_h1[:, 0]
    lifetimes_sorted = np.sort(lifetimes)[::-1]
    if len(lifetimes_sorted) >= 2 and lifetimes_sorted[1] > 0:
        pers_ratio = float(lifetimes_sorted[0] / lifetimes_sorted[1])
    else:
        pers_ratio = float("inf")
    print(f"[H¹] top lifetimes = {lifetimes_sorted[:5].round(4).tolist()}  "
          f"ratio = {pers_ratio:.3f}")

    # ----- Align θ_disc to hue (rotation + sign) -----
    hue_true_rad = hue * 2 * np.pi
    score, alpha_star, sign = best_align(theta_disc, hue_true_rad)
    theta_aligned = (sign * theta_disc - alpha_star) % (2 * np.pi)

    # Standard circular-circular correlation (rotation-invariant; we still
    # need to fix the sign because Jammalamadaka is sign-sensitive).
    cc_corr = circ_corr_jammalamadaka(sign * theta_disc, hue_true_rad)
    residual = circ_dist(theta_aligned, hue_true_rad)
    med_abs_resid = float(np.median(np.abs(residual)))
    print(f"[align] alpha*={alpha_star:+.3f} rad  sign={sign}  "
          f"|·| score={score:.3f}  Jammalamadaka cc={cc_corr:+.3f}  "
          f"median |residual|={np.degrees(med_abs_resid):.1f}°")

    # Chance baseline: shuffle θ
    rng = np.random.default_rng(0)
    chance_scores = []
    chance_meds = []
    for _ in range(20):
        th_shuf = rng.permutation(theta_disc)
        s_chance, a_chance, sg_chance = best_align(th_shuf, hue_true_rad)
        th_al = (sg_chance * th_shuf - a_chance) % (2 * np.pi)
        chance_scores.append(s_chance)
        chance_meds.append(float(np.median(np.abs(circ_dist(th_al, hue_true_rad)))))
    chance_score_mean = float(np.mean(chance_scores))
    chance_med_mean = float(np.mean(chance_meds))
    print(f"[chance] mean align-score={chance_score_mean:.3f}  "
          f"mean median |residual|={np.degrees(chance_med_mean):.1f}°  "
          f"(over 20 shuffles, best-aligned)")

    # ----- 1D periodic Duchon on θ_disc, 5-fold CV macro R² -----
    theta_for_basis = theta_aligned / (2 * np.pi)  # in [0,1) for Duchon
    Phi_full, P_h, _ = hue_basis(theta_for_basis)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS
    preds_disc = np.zeros_like(Z)
    lambdas_disc = []
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        B, lam = reml_fit(Phi_full[tr], Z[tr], P_h)
        preds_disc[te] = Phi_full[te] @ B
        lambdas_disc.append(lam)
    r2_disc = float(r2_macro(Z, preds_disc))
    print(f"[CV] θ_disc-Duchon macro R² = {r2_disc:+.4f}   "
          f"(reference: hue-Duchon {R2_AUTO66_HUE:.3f}, "
          f"hue+sv {R2_AUTO67_JOINT:.3f})")

    # Also fit hue-only with the same machinery to confirm reproducibility.
    Phi_hue_full, P_h2, _ = hue_basis(hue)
    preds_hue = np.zeros_like(Z)
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        B, _ = reml_fit(Phi_hue_full[tr], Z[tr], P_h2)
        preds_hue[te] = Phi_hue_full[te] @ B
    r2_hue_check = float(r2_macro(Z, preds_hue))
    print(f"[CV] (sanity) prescribed-hue Duchon macro R² = {r2_hue_check:+.4f}")

    # ----- Plot 4 panels -----
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # (a) θ_aligned vs hue_true scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(np.degrees(hue_true_rad), np.degrees(theta_aligned),
               c=rgb, s=22, edgecolor="black", linewidth=0.3)
    ax.plot([0, 360], [0, 360], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("xkcd hue (deg)")
    ax.set_ylabel("θ_disc aligned (deg)")
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_title(f"(a) discovered θ vs perceptual hue\n"
                 f"Jammalamadaka cc = {cc_corr:+.3f}  "
                 f"(α*={alpha_star:+.2f}, sign={sign:+d})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) angular residual histogram
    ax = fig.add_subplot(gs[0, 1])
    resid_deg = np.degrees(residual)
    ax.hist(resid_deg, bins=60, color="#1f77b4", edgecolor="black", alpha=0.85,
            label=f"discovered θ (median |·|={np.degrees(med_abs_resid):.1f}°)")
    ax.axvline(np.degrees(med_abs_resid), color="red", ls="--", lw=1,
               label="median |residual|")
    ax.axvline(-np.degrees(med_abs_resid), color="red", ls="--", lw=1)
    ax.axvline(np.degrees(chance_med_mean), color="grey", ls=":", lw=1,
               label=f"chance median ≈ {np.degrees(chance_med_mean):.0f}°")
    ax.axvline(-np.degrees(chance_med_mean), color="grey", ls=":", lw=1)
    ax.set_xlabel("angular residual (deg)")
    ax.set_ylabel("count")
    ax.set_title("(b) θ_disc − hue residual after best alignment")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (c) PC1-PC2 scatter colored by θ_disc (BEFORE alignment, raw)
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=theta_disc, cmap="hsv",
                    s=22, edgecolor="black", linewidth=0.3, vmin=0,
                    vmax=2 * np.pi)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("θ_disc (rad)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(c) Z_top64 PC1-PC2 colored by discovered θ\n"
                 "(should look like a hue wheel if the circle is real)")
    ax.grid(alpha=0.3)

    # (d) bar comparing CV R²
    ax = fig.add_subplot(gs[1, 1])
    names = ["prescribed hue\n(auto_66)", "discovered θ\n(this run)",
             "hue+sat,val joint\n(auto_67 ceiling)"]
    vals = [R2_AUTO66_HUE, r2_disc, R2_AUTO67_JOINT]
    colors_b = ["#d62728", "#1f77b4", "#2ca02c"]
    ax.bar(names, vals, color=colors_b, edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:+.3f}", ha="center", fontsize=10)
    ax.axhline(0.246, color="grey", ls="--", lw=1, label="kNN-Lab ceiling 0.246")
    ax.set_ylabel("5-fold CV macro R² on Z_top64")
    ax.set_title("(d) Duchon CV R²: discovered vs prescribed circle")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"auto_70 · topological color circle via persistent cohomology "
        f"(dreimac CircularCoords) · cogito L40 · N={N} colors",
        fontsize=13,
    )
    out_png = OUT_DIR / "auto_70.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    payload = {
        "n_colors": int(N),
        "K_PC": K_PC,
        "n_landmarks": int(N_LANDMARKS),
        "dreimac_perc": 0.5,
        "dreimac_prime": 41,
        "dreimac_used_standard_range": bool(used_standard_range),
        "h1_top_lifetimes": [float(x) for x in lifetimes_sorted[:5]],
        "h1_persistence_ratio_top1_over_top2": float(pers_ratio),
        "best_alignment": {
            "alpha_star_rad": float(alpha_star),
            "alpha_star_deg": float(np.degrees(alpha_star)),
            "sign": int(sign),
            "complex_align_score": float(score),
        },
        "circular_correlation_jammalamadaka": float(cc_corr),
        "median_abs_residual_rad": float(med_abs_resid),
        "median_abs_residual_deg": float(np.degrees(med_abs_resid)),
        "chance": {
            "mean_align_score": float(chance_score_mean),
            "mean_median_abs_residual_deg": float(np.degrees(chance_med_mean)),
            "n_shuffles": 20,
        },
        "cv_macro_r2_theta_disc_duchon": float(r2_disc),
        "cv_macro_r2_hue_duchon_sanity": float(r2_hue_check),
        "ref_cv_macro_r2_auto66_hue": float(R2_AUTO66_HUE),
        "ref_cv_macro_r2_auto67_joint": float(R2_AUTO67_JOINT),
        "fold_lambdas_theta_disc": [float(x) for x in lambdas_disc],
        "notes": (
            "dreimac.CircularCoords (de Silva–Morozov–Vejdemo-Johansson). "
            "Z = standardized PCA top-64 of cogito L40 per-color centroids "
            "(TOP_TEMPLATES averaged). θ_disc obtained from the most-persistent "
            "H¹ class via harmonic smoothing of the integer cocycle. Alignment "
            "to xkcd hue is gauge-fixed up to one rotation + one handedness "
            "sign (the residual S¹ symmetry). CV R² of θ_disc-Duchon vs "
            "prescribed-hue Duchon is the leakage-free test that the discovered "
            "circle carries the same information as perceptual hue. No "
            "Gaussian RBF, no Duchon length_scale, no B-splines."
        ),
    }
    (OUT_DIR / "auto_70.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_70.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
