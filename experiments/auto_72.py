"""auto_72.py — Marchenko–Pastur signal-rank denoising of cogito's color
manifold, then re-run persistent cohomology + circular coordinates.

Builds on auto_70 (null circular result on Z_top64) and auto_exp_15 (top-1
H¹ persistence 19.22 vs noise median 2.88). Hypothesis: in Z_top64 the
dominant hue cycle is swamped by noise dimensions; projecting onto only
the MP-significant principal subspace should sharpen the circle.

Pipeline
--------
1. Load X_L40 mmap → TOP_TEMPLATES centroids → color_filter_list → 886.
2. Center centroid matrix (N × D, D=7168). Compute economy SVD, get
   singular values s_i and eigenvalues λ_i = s_i^2 / N of the sample
   covariance estimator (N-normalized).
3. MP signal-rank: empirical eigenvalues above λ+ = σ²(1+√(N/D))² are
   signal. Estimate σ² by minimizing the L2 mismatch between the
   empirical eigenvalue density restricted to the candidate bulk and the
   theoretical MP density. We grid-search over σ² and over an assumed
   K* (the number of eigenvalues to mask out as signal), pick the
   (σ², K*) that minimizes a KS-like statistic between the empirical
   bulk CDF and the MP CDF (truncated to [λ-, λ+]).
4. Project centroids onto top-K* principal directions → Z_clean.
5. ripser H¹ on Z_clean — top-3 persistences vs auto_exp_15.
6. dreimac CircularCoords on Z_clean → θ_disc; align to xkcd hue.
7. 1D periodic Duchon on hue (m=2, n_centers=40, no length_scale) on
   Z_clean as target; 5-fold CV macro R².

HARD: no Gaussian RBF, no Duchon length_scale, no B-splines.
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

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
N_FOLDS = 5
N_LANDMARKS = 200
HUE_CENTERS = 40

# Reference numbers from prior runs
R2_AUTO66_HUE = 0.251
R2_AUTO67_JOINT = 0.321
AUTO70_CC = 0.13
AUTOEXP15_TOP3 = [19.22, 18.1, 16.7]  # approximate, for comparison annotations


# --------- Marchenko–Pastur helpers ---------

def mp_density(lam: np.ndarray, sigma2: float, q: float) -> np.ndarray:
    """Marchenko–Pastur density for ratio q = N/D, variance sigma^2.

    q must be in (0, 1] for our N <= D regime; if q > 1 we get a delta
    at 0 in addition to the bulk. We use the bulk-only formula here:
        f(λ) = (1/(2π σ² q)) * sqrt((λ+ - λ)(λ - λ-)) / λ   for λ in [λ-, λ+]
    """
    lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    out = np.zeros_like(lam, dtype=np.float64)
    mask = (lam > lam_minus) & (lam < lam_plus) & (lam > 0)
    if not mask.any():
        return out
    inside = np.sqrt(np.clip((lam_plus - lam[mask]) * (lam[mask] - lam_minus),
                              0.0, None))
    out[mask] = inside / (2.0 * np.pi * sigma2 * q * lam[mask])
    return out


def fit_mp_sigma2(eigs_sorted_desc: np.ndarray, q: float):
    """Find (sigma2, K*) by fitting MP density to the bulk.

    Strategy: for each candidate K* in a reasonable range, take the
    "bulk" eigenvalues eigs[K*:] (the smaller ones, hypothesized noise).
    Estimate sigma2 from the bulk mean:
        E[λ_bulk] under MP = sigma2   (for q ≤ 1)
    Then evaluate goodness-of-fit as a KS distance between the bulk
    empirical CDF and the MP CDF on [λ-, λ+]. Pick the K* with the
    smallest KS, subject to the cutoff being self-consistent (eigs[K*-1]
    > λ+ and eigs[K*] ≤ λ+ + small tol).
    """
    n_bulk_min = 50
    K_max_search = min(len(eigs_sorted_desc) - n_bulk_min, len(eigs_sorted_desc) - 1)
    best = None  # (score, K*, sigma2, lam_plus)
    for K in range(0, K_max_search + 1):
        bulk = eigs_sorted_desc[K:]
        # sigma2 estimate from MP mean (E[λ] = sigma2 for q in (0,1])
        # More robust: use mean over the proposed bulk.
        sigma2 = float(bulk.mean())
        if sigma2 <= 0:
            continue
        lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
        lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
        # Self-consistency: eigenvalue at K should be ≤ lam_plus,
        # and eigenvalue at K-1 (if exists) > lam_plus.
        ok_low = bulk[0] <= lam_plus * 1.02  # small tol
        ok_high = (K == 0) or (eigs_sorted_desc[K - 1] > lam_plus * 0.98)
        if not (ok_low and ok_high):
            continue
        # KS distance between bulk empirical CDF and MP CDF on bulk support
        # Compute MP CDF numerically on a grid.
        grid = np.linspace(lam_minus, lam_plus, 2000)
        dens = mp_density(grid, sigma2, q)
        cdf_mp = np.cumsum(dens) * (grid[1] - grid[0])
        cdf_mp = cdf_mp / max(cdf_mp[-1], 1e-12)
        # Empirical CDF on bulk, restricted to [lam_minus, lam_plus]
        b_in = bulk[(bulk >= lam_minus) & (bulk <= lam_plus)]
        if len(b_in) < 10:
            continue
        b_sorted = np.sort(b_in)
        ecdf = np.arange(1, len(b_sorted) + 1) / len(b_sorted)
        # interpolate cdf_mp at b_sorted
        mp_at = np.interp(b_sorted, grid, cdf_mp)
        ks = float(np.max(np.abs(ecdf - mp_at)))
        cand = (ks, K, sigma2, lam_plus)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        # Fallback: assume all noise
        sigma2 = float(eigs_sorted_desc.mean())
        lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
        return 0, sigma2, lam_plus, float("nan")
    ks, K, sigma2, lam_plus = best
    return K, sigma2, lam_plus, ks


# --------- Duchon basis (1D periodic, m=2) ---------

def hue_basis(hue01, n_centers=HUE_CENTERS):
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


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def circ_dist(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def best_align(theta_disc, hue_true_rad):
    alphas = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    best = (-np.inf, 0.0, +1)
    for s in (+1, -1):
        th = s * theta_disc
        for a in alphas:
            shifted = (th - a + np.pi) % (2 * np.pi) - np.pi
            z1 = np.exp(1j * shifted)
            z2 = np.exp(1j * hue_true_rad)
            score = float(np.abs((z1 * np.conj(z2)).mean()))
            if score > best[0]:
                best = (score, float(a), int(s))
    return best


def circ_corr_jammalamadaka(alpha, beta):
    a_bar = np.angle(np.exp(1j * alpha).mean())
    b_bar = np.angle(np.exp(1j * beta).mean())
    sa = np.sin(alpha - a_bar)
    sb = np.sin(beta - b_bar)
    num = (sa * sb).sum()
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def _ensure_deps():
    for mod, pkg in (("ripser", "ripser"), ("dreimac", "dreimac")):
        try:
            __import__(mod)
        except ImportError:
            print(f"[install] {pkg}")
            subprocess.check_call(["uv", "pip", "install", pkg])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_deps()
    from ripser import ripser
    from dreimac import CircularCoords

    # ---- Load centroids ----
    X = np.load(HARVEST, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    D_dim = X.shape[1]
    print(f"[load] X={X.shape}, n_raw={n_raw}, D={D_dim}")
    centroids = np.zeros((n_raw, D_dim), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue = hsv[:, 0]
    N = centroids.shape[0]
    D = D_dim
    print(f"[load] N={N}, D={D}")

    # ---- Center + economy SVD ----
    mu = centroids.mean(axis=0, keepdims=True)
    Xc = centroids - mu
    # economy SVD: U (N×N), S (N,), Vt (N×D)  since N < D
    print(f"[svd] computing economy SVD on ({N}, {D}) ...")
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"[svd] s range [{s[-1]:.4f}, {s[0]:.4f}], len={len(s)}")
    # Sample-covariance eigenvalues (N-normalized): λ_i = s_i^2 / N
    eigs = (s ** 2) / N
    eigs_sorted = np.sort(eigs)[::-1]  # already sorted desc from SVD, but be safe

    # ---- MP fit ----
    q = N / D  # since N < D, q < 1; bulk MP applies to N nonzero eigs
    print(f"[mp] q = N/D = {q:.4f}")
    K_star, sigma2_hat, lam_plus, ks = fit_mp_sigma2(eigs_sorted, q)
    lam_minus = sigma2_hat * (1 - np.sqrt(q)) ** 2
    signal_fraction = float(eigs_sorted[:K_star].sum() / eigs_sorted.sum()) \
        if K_star > 0 else 0.0
    print(f"[mp] K*={K_star}, σ²={sigma2_hat:.5f}, λ+={lam_plus:.5f}, "
          f"λ-={lam_minus:.5f}, KS={ks:.4f}, signal_frac={signal_fraction:.3f}")

    # ---- Project to top-K* signal subspace ----
    K_use = max(K_star, 2)
    Z_clean = U[:, :K_use] * s[:K_use]  # (N, K*) -- principal scores
    print(f"[proj] Z_clean shape={Z_clean.shape}")

    # ---- Persistent H¹ on Z_clean ----
    print(f"[ripser] H¹ on Z_clean (N={N}, K={K_use}) ...")
    out = ripser(Z_clean, maxdim=1)
    h1 = out["dgms"][1]
    if h1.size:
        h1f = h1[np.isfinite(h1[:, 1])]
        pers = h1f[:, 1] - h1f[:, 0]
        order = np.argsort(pers)[::-1]
        top_pers = pers[order]
        top_h1_pairs = h1f[order]
    else:
        top_pers = np.array([])
        top_h1_pairs = np.zeros((0, 2))
    top3 = top_pers[:3].tolist() if len(top_pers) >= 3 else top_pers.tolist()
    if len(top_pers) >= 2 and top_pers[1] > 0:
        pers_ratio = float(top_pers[0] / top_pers[1])
    else:
        pers_ratio = float("inf")
    med_noise = float(np.median(top_pers)) if len(top_pers) > 0 else 0.0
    ratio_top1_noise = (float(top_pers[0]) / med_noise) if med_noise > 0 and len(top_pers) > 0 else float("inf")
    print(f"[H¹] top-3 persistences: {[f'{x:.3f}' for x in top3]}  "
          f"ratio top1/top2={pers_ratio:.3f}  top1/median={ratio_top1_noise:.2f}")

    # ---- dreimac CircularCoords ----
    print(f"[dreimac] CircularCoords ...")
    cc = CircularCoords(Z_clean, n_landmarks=N_LANDMARKS, prime=41,
                        maxdim=1, verbose=False)
    try:
        theta_disc = np.asarray(cc.get_coordinates(perc=0.5, cocycle_idx=0))
        used_standard_range = True
    except Exception as e:
        print(f"[dreimac] standard_range failed ({e}); retry")
        theta_disc = np.asarray(cc.get_coordinates(
            perc=0.5, cocycle_idx=0, standard_range=False))
        used_standard_range = False

    hue_true_rad = hue * 2 * np.pi
    score, alpha_star, sign = best_align(theta_disc, hue_true_rad)
    theta_aligned = (sign * theta_disc - alpha_star) % (2 * np.pi)
    cc_corr = circ_corr_jammalamadaka(sign * theta_disc, hue_true_rad)
    resid = circ_dist(theta_aligned, hue_true_rad)
    med_abs_resid = float(np.median(np.abs(resid)))
    print(f"[align] alpha*={alpha_star:+.3f} sign={sign} score={score:.3f} "
          f"cc={cc_corr:+.3f} med|resid|={np.degrees(med_abs_resid):.1f}°")

    # ---- Hue-prescribed Duchon CV on Z_clean ----
    Phi_hue, P_h, _ = hue_basis(hue)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS
    preds = np.zeros_like(Z_clean)
    lambdas = []
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        B, lam = reml_fit(Phi_hue[tr], Z_clean[tr], P_h)
        preds[te] = Phi_hue[te] @ B
        lambdas.append(lam)
    r2_hue_clean = float(r2_macro(Z_clean, preds))
    print(f"[CV] hue-Duchon on Z_clean macro R² = {r2_hue_clean:+.4f}   "
          f"(auto_66 on Z_top64: {R2_AUTO66_HUE:.3f})")

    # ---- Plot 4 panels ----
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # (1) singular spectrum with MP edge
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3, color="#1f77b4",
                label="singular values")
    s_plus = np.sqrt(lam_plus * N)
    ax.axhline(s_plus, color="red", ls="--", lw=1,
               label=f"MP edge s+ = √(λ+·N) = {s_plus:.3f}")
    ax.axvline(K_star, color="green", ls=":", lw=1,
               label=f"K* = {K_star}")
    ax.set_xlabel("index i")
    ax.set_ylabel("singular value s_i (log)")
    ax.set_title(f"(1) Singular spectrum of centered centroids\n"
                 f"K* = {K_star}, σ̂² = {sigma2_hat:.4f}, signal frac = {signal_fraction:.3f}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # (2) Eigenvalue histogram vs MP density
    ax = fig.add_subplot(gs[0, 1])
    # Show only the bulk region for clarity
    bulk_eigs = eigs_sorted[K_star:]
    nbins = 60
    counts, edges, _ = ax.hist(bulk_eigs, bins=nbins, density=True,
                                alpha=0.6, color="#1f77b4",
                                edgecolor="black", label=f"empirical bulk (n={len(bulk_eigs)})")
    grid = np.linspace(max(lam_minus, edges[0]), lam_plus, 500)
    dens = mp_density(grid, sigma2_hat, q)
    ax.plot(grid, dens, "r-", lw=2,
            label=f"MP density (σ²={sigma2_hat:.4f}, q={q:.3f})")
    ax.axvline(lam_plus, color="red", ls="--", lw=1, label=f"λ+={lam_plus:.4f}")
    ax.axvline(lam_minus, color="red", ls=":", lw=1, label=f"λ-={lam_minus:.4f}")
    # Also mark a few signal eigenvalues above the bulk
    for ie in eigs_sorted[:min(K_star, 8)]:
        ax.axvline(ie, color="green", alpha=0.4, lw=0.6)
    ax.set_xlabel("eigenvalue λ = s²/N")
    ax.set_ylabel("density")
    ax.set_title(f"(2) Bulk eigenvalues vs MP fit\n"
                 f"KS = {ks:.4f}, green lines = signal eigs above λ+")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # (3) H¹ persistence diagram on Z_clean
    ax = fig.add_subplot(gs[1, 0])
    h0 = out["dgms"][0]
    if h0.size:
        h0f = h0[np.isfinite(h0[:, 1])]
        if h0f.size:
            ax.scatter(h0f[:, 0], h0f[:, 1], s=10, c="steelblue", alpha=0.4,
                       label=f"H0 ({len(h0f)})", edgecolor="none")
    if top_h1_pairs.size:
        ax.scatter(top_h1_pairs[:, 0], top_h1_pairs[:, 1], s=18,
                   c="firebrick", alpha=0.45,
                   label=f"H1 ({len(top_h1_pairs)})", edgecolor="none")
        for rank in range(min(3, len(top_h1_pairs))):
            b, d = top_h1_pairs[rank]
            ax.scatter([b], [d], s=180, facecolor="none", edgecolor="black",
                       lw=1.6, zorder=10)
            ax.annotate(f"#{rank+1} p={top_pers[rank]:.2f}", (b, d),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)
    all_finite = np.concatenate([
        h0[np.isfinite(h0[:, 1])][:, 1] if h0.size else np.array([]),
        top_h1_pairs[:, 1] if top_h1_pairs.size else np.array([]),
    ])
    lim_hi = float(all_finite.max()) * 1.05 if all_finite.size else 1.0
    ax.plot([0, lim_hi], [0, lim_hi], "k--", lw=0.6)
    ax.set_xlim(0, lim_hi); ax.set_ylim(0, lim_hi)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(f"(3) H¹ persistence on Z_clean (K*={K_star})\n"
                 f"top-3: {[f'{x:.2f}' for x in top3]}   "
                 f"(auto_exp_15 K=64: {AUTOEXP15_TOP3})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (4) discovered θ vs hue
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(np.degrees(hue_true_rad), np.degrees(theta_aligned),
               c=rgb, s=22, edgecolor="black", linewidth=0.3)
    ax.plot([0, 360], [0, 360], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("xkcd hue (deg)")
    ax.set_ylabel("θ_disc aligned (deg)")
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_title(f"(4) discovered θ vs hue on Z_clean\n"
                 f"Jammalamadaka cc = {cc_corr:+.3f}   "
                 f"(auto_70 on Z_top64: {AUTO70_CC:+.3f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_72 · MP-denoised cogito color manifold · "
        f"K*={K_star}/{len(s)} signal dims (σ̂²={sigma2_hat:.4f}, "
        f"signal frac={signal_fraction:.3f}, N={N})",
        fontsize=12,
    )
    out_png = OUT_DIR / "auto_72.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")

    payload = {
        "n_colors": int(N),
        "D": int(D),
        "q_ratio_N_over_D": float(q),
        "K_star": int(K_star),
        "sigma2_hat": float(sigma2_hat),
        "lambda_plus": float(lam_plus),
        "lambda_minus": float(lam_minus),
        "mp_ks_statistic": float(ks),
        "signal_fraction": float(signal_fraction),
        "h1_top3_persistences_clean": [float(x) for x in top3],
        "h1_persistence_ratio_top1_over_top2_clean": float(pers_ratio),
        "h1_ratio_top1_over_median_clean": float(ratio_top1_noise),
        "h1_n_bars_clean": int(len(top_pers)),
        "ref_h1_top3_auto_exp_15_K64": AUTOEXP15_TOP3,
        "dreimac_used_standard_range": bool(used_standard_range),
        "circ_corr_jammalamadaka_clean": float(cc_corr),
        "ref_circ_corr_auto70_Z_top64": AUTO70_CC,
        "best_alignment": {
            "alpha_star_rad": float(alpha_star),
            "sign": int(sign),
            "complex_align_score": float(score),
        },
        "median_abs_residual_deg": float(np.degrees(med_abs_resid)),
        "cv_macro_r2_hue_duchon_on_Z_clean": float(r2_hue_clean),
        "ref_cv_macro_r2_auto66_hue_on_Z_top64": float(R2_AUTO66_HUE),
        "ref_cv_macro_r2_auto67_joint": float(R2_AUTO67_JOINT),
        "fold_lambdas": [float(x) for x in lambdas],
        "notes": (
            "Marchenko–Pastur signal-rank thresholding of the N×D centered "
            "centroid matrix (N=886 cogito-L40 colors, D=7168). σ² fitted by "
            "minimizing KS distance between empirical bulk-eigenvalue CDF "
            "and theoretical MP CDF, with K* chosen self-consistently "
            "(λ_{K*-1} > λ+ ≥ λ_{K*}). Z_clean = U[:,:K*] * s[:K*] = top-K* "
            "principal scores. Then ripser H¹ + dreimac CircularCoords + "
            "1D periodic Duchon (m=2, n_centers=40, no length_scale, no "
            "B-splines) on hue. No Gaussian RBF anywhere."
        ),
    }
    (OUT_DIR / "auto_72.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_72.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
