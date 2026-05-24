"""auto_exp_67: Formal topology model-comparison on cogito-L40 HSV subspace.

We rank 5 candidate latent topologies for the cogito-L40 color manifold:

    Euclidean, Circle, Sphere, Torus, Cylinder

Each topology is fit as a single smooth that maps the candidate's INTRINSIC
coordinate(s) (e.g. (theta) for S^1, (lat,lon) for S^2, (theta1,theta2) for
T^2, (theta,z) for S^1xR) to the K=64-dimensional cogito-L40 PCA response.

Coordinates are SEEDED from the HSV labels (hue->theta, value/lightness->z,
RGB-unit-vector->sphere) so that the comparison is "given the HSV-supervised
geometry, which topology of the supervised coordinates best explains L40?"
This is the right test to run: prior memory entries
  - project_cogito_color_manifold_decomposition.md
  - project_cogito_recovery_at_d_aux_3.md
established that HSV lives in L40; the open question is its topology.

Evidence/score per fit (lower = better, except holdout R²):
  - REML score              (gamfit gaussian_reml_fit)
  - BIC                     (n log(SSE/n) + k log(n), k = edf)
  - Tierney-Kadane normalizer = REML + 0.5 * k * log(2*pi)
  - holdout R²              (3-fold CV across templates; train on 21, test 7)

gamfit version pinned: uses public API gamfit.gaussian_reml_fit (low-level
closed-form Gaussian REML). gamfit.compare_models does NOT exist in 0.1.112
(checked at runtime); we fall back to manual per-fit comparison and assemble
the table ourselves.

Outputs:
  runs/auto_exp_67_topology/comparison.json
  runs/auto_exp_67_topology/comparison.png

Memory output:
  project_topology_selector_cogito.md  (+ MEMORY.md index entry)
"""
from __future__ import annotations

import colorsys
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore


ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_DIR = ROOT / "runs" / "auto_exp_67_topology"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "comparison.json"
OUT_PNG = OUT_DIR / "comparison.png"

N_TEMPLATES = 28
K_PCS = 64

# ---------- xkcd loader (mirrors auto_exp_38) -----------------------------
def load_xkcd_rgb(n_colors: int):
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((int(hexs[0:2], 16) / 255.0,
                        int(hexs[2:4], 16) / 255.0,
                        int(hexs[4:6], 16) / 255.0))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = colorsys.rgb_to_hsv(*c)
    return out


# ---------- per-color, per-template PCA projection ------------------------
def per_color_per_template_pcs(x_mmap, basis, k_pcs, n_templates=N_TEMPLATES):
    """Returns Z of shape (n_colors, n_templates, k_pcs).

    Standardizes per feature using the basis (mu, sigma), then projects onto
    the top-k_pcs PCA directions. Block-streamed to respect the X_L40 mmap.
    """
    n_rows, _ = x_mmap.shape
    n_c = n_rows // n_templates
    mu = basis["mu"]
    sigma = basis["sigma"]
    Vt = basis["Vt"][:k_pcs]
    Z = np.zeros((n_c, n_templates, k_pcs), dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_templates
        e = ce * n_templates
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Zc = chunk @ Vt.T   # (block*n_t, k_pcs)
        Zc = Zc.reshape(ce - cs, n_templates, k_pcs)
        Z[cs:ce] = Zc
    return Z


# ---------- topology coordinate builders ----------------------------------
def coords_euclidean(hsv):
    """1D: V (value) is the dominant scalar axis; this is the trivial null."""
    return hsv[:, 2:3].copy()  # (n, 1)


def coords_circle(hsv):
    """S^1 angle in [0, 2π) from hue."""
    return (2 * np.pi * hsv[:, 0:1]).copy()  # (n, 1)


def coords_sphere(rgb):
    """S^2: normalize RGB to unit vector, return (lat, lon) in DEGREES.

    gamfit.sphere_basis convention: degrees by default; (lat, lon).
    """
    v = rgb - rgb.mean(0, keepdims=True)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return np.stack([lat, lon], axis=1)  # (n, 2) degrees


def coords_torus(hsv):
    """T^2 = S^1 × S^1: (2π·hue, 2π·value) — hue ring × value ring."""
    return np.stack(
        [2 * np.pi * hsv[:, 0], 2 * np.pi * hsv[:, 2]], axis=1
    )  # (n, 2) radians


def coords_cylinder(hsv):
    """S^1 × R: (2π·hue, value)."""
    return np.stack([2 * np.pi * hsv[:, 0], hsv[:, 2]], axis=1)


# ---------- design + penalty builders -------------------------------------
def fourier_basis(theta_rad, n_harmonics):
    """Periodic Fourier basis with 1 + 2*H columns (1, cos, sin, ...)."""
    n = theta_rad.shape[0]
    cols = [np.ones(n)]
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(h * theta_rad))
        cols.append(np.sin(h * theta_rad))
    return np.stack(cols, axis=1)


def fourier_penalty_sqrt(n_harmonics, penalty_order=2):
    """Sobolev-style penalty: weight harmonic h by h^(2*penalty_order).

    Returns SQUARE-ROOT penalty S (k, k) such that S @ S.T = penalty.
    Intercept is unpenalized.
    """
    k = 1 + 2 * n_harmonics
    diag = np.zeros(k)
    for h in range(1, n_harmonics + 1):
        w = h ** penalty_order  # sqrt weight, so total penalty ~ h^(2m)
        diag[1 + 2 * (h - 1)] = w
        diag[1 + 2 * (h - 1) + 1] = w
    return np.diag(diag)


def bspline_design_penalty(t, n_knots=10, degree=3, penalty_order=2):
    """1D B-spline design + sqrt of 2nd-difference penalty.

    Uses gamfit.bspline_basis. Returns (X, S) with S.T @ S == penalty.
    """
    knots = np.linspace(float(t.min()) - 1e-6, float(t.max()) + 1e-6, n_knots)
    X = gamfit.bspline_basis(np.asarray(t), knots=knots, degree=degree, periodic=False)
    X = np.asarray(X, dtype=np.float64)
    k = X.shape[1]
    # 2nd-order difference matrix; build P = D.T D then sqrt
    D = np.eye(k)
    for _ in range(penalty_order):
        D = np.diff(D, axis=0)
    P = D.T @ D
    eig_vals, eig_vecs = np.linalg.eigh((P + P.T) / 2)
    eig_vals = np.clip(eig_vals, 0.0, None)
    S = (eig_vecs * np.sqrt(eig_vals)) @ eig_vecs.T
    return X, S


def tensor_fourier_design(theta1, theta2, h1=4, h2=4):
    """Tensor product of two Fourier bases. Outer-product columns."""
    A = fourier_basis(theta1, h1)
    B = fourier_basis(theta2, h2)
    n = A.shape[0]
    X = np.einsum("ni,nj->nij", A, B).reshape(n, -1)
    return X


def tensor_fourier_penalty_sqrt(h1, h2, penalty_order=2):
    """Penalize each marginal independently; sum of square-roots stacked."""
    k1 = 1 + 2 * h1
    k2 = 1 + 2 * h2
    k = k1 * k2
    # weight per column = sqrt(h_a^(2m) + h_b^(2m)) (a/b are harmonic indices)
    diag = np.zeros(k)
    for i in range(k1):
        h_a = 0 if i == 0 else (i + 1) // 2
        for j in range(k2):
            h_b = 0 if j == 0 else (j + 1) // 2
            w2 = h_a ** (2 * penalty_order) + h_b ** (2 * penalty_order)
            diag[i * k2 + j] = np.sqrt(w2)
    return np.diag(diag)


def cylinder_design_penalty(theta, z, h_theta=4, n_knots_z=8):
    """S^1 × R: tensor Fourier(theta) × BSpline(z)."""
    F = fourier_basis(theta, h_theta)
    knots = np.linspace(float(z.min()) - 1e-6, float(z.max()) + 1e-6, n_knots_z)
    Bz = gamfit.bspline_basis(np.asarray(z), knots=knots, degree=3, periodic=False)
    Bz = np.asarray(Bz, dtype=np.float64)
    n = F.shape[0]
    kF = F.shape[1]
    kB = Bz.shape[1]
    X = np.einsum("ni,nj->nij", F, Bz).reshape(n, kF * kB)
    # Penalty: Fourier-weight on F axis, 2nd-diff on B axis (additive structure)
    fw = np.zeros(kF)
    for h in range(1, h_theta + 1):
        fw[1 + 2 * (h - 1)] = h ** 2
        fw[1 + 2 * (h - 1) + 1] = h ** 2
    Db = np.eye(kB)
    for _ in range(2):
        Db = np.diff(Db, axis=0)
    # Combined sqrt penalty as block-diag-ish: use sum of two roots stacked
    k = kF * kB
    # weight per column = sqrt(fw_i^2 + db penalty trace_j)  -- approximate
    # Construct two penalty matrices and stack vertically.
    S1 = np.zeros((k, k))
    for i in range(kF):
        for j in range(kB):
            S1[i * kB + j, i * kB + j] = fw[i]
    # Roughness in z: difference-on-each-Fourier-slice; build sparse
    S2 = np.zeros((kF * (Db.shape[0]), k))
    for i in range(kF):
        row_off = i * Db.shape[0]
        col_off = i * kB
        S2[row_off:row_off + Db.shape[0], col_off:col_off + kB] = Db
    S_tall = np.vstack([S1, S2])
    P = S_tall.T @ S_tall
    eig_vals, eig_vecs = np.linalg.eigh((P + P.T) / 2)
    eig_vals = np.clip(eig_vals, 0.0, None)
    S = (eig_vecs * np.sqrt(eig_vals)) @ eig_vecs.T
    return X, S


def sphere_design_penalty(latlon_deg, n_centers=30, penalty_order=2):
    """Spherical-spline design + penalty via gamfit.sphere_basis."""
    X, P = gamfit.sphere_basis(
        np.asarray(latlon_deg), n_centers=n_centers,
        penalty_order=penalty_order, kernel="sobolev", radians=False,
    )
    X = np.asarray(X, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    # gamfit.sphere_basis returns penalty (k, k). gaussian_reml_fit wants
    # the SQRT (m, k); compute by Cholesky with tiny jitter.
    k = P.shape[0]
    P = (P + P.T) / 2
    eig_vals, eig_vecs = np.linalg.eigh(P)
    eig_vals = np.clip(eig_vals, 0.0, None)
    S = (eig_vecs * np.sqrt(eig_vals)) @ eig_vecs.T
    return X, S


# ---------- fitting + scoring --------------------------------------------
def fit_one_topology(X_design, S_penalty, Y, name):
    """Fit Y (n, K_pcs) against design X using REML.

    For multi-output Y we average per-column REML scores (gamfit's
    closed-form Gaussian REML is single-response; loop over K_pcs columns).
    Returns dict with reml, edf, sse, sigma2, fitted, error.
    """
    n, _ = X_design.shape
    K = Y.shape[1]
    reml_sum = 0.0
    edf_sum = 0.0
    sse = 0.0
    fitted = np.zeros_like(Y)
    failures = 0
    for k in range(K):
        y_col = Y[:, k:k + 1]
        try:
            out = gamfit.gaussian_reml_fit(
                np.ascontiguousarray(X_design),
                np.ascontiguousarray(y_col),
                np.ascontiguousarray(S_penalty),
            )
        except Exception as exc:
            failures += 1
            if failures <= 2:
                warnings.warn(f"[{name}] col {k} fit failed: {exc!r}")
            continue
        reml_sum += float(np.asarray(out["reml_score"]))
        edf_sum += float(np.asarray(out["edf"]))
        fc = np.asarray(out["fitted"]).reshape(-1)
        fitted[:, k] = fc
        sse += float(((y_col.ravel() - fc) ** 2).sum())
    if failures == K:
        return {"name": name, "error": "all_columns_failed"}
    n_eff = n * K
    bic = n_eff * np.log(sse / n_eff) + edf_sum * np.log(n_eff)
    tk = reml_sum + 0.5 * edf_sum * np.log(2 * np.pi)
    return {
        "name": name,
        "reml": reml_sum,
        "edf": edf_sum,
        "sse": sse,
        "bic": bic,
        "tk": tk,
        "n_params": float(X_design.shape[1] * K),
        "n_obs": float(n_eff),
        "fit_failures": failures,
        "fitted": fitted,
    }


def holdout_r2_by_template(
    Z_per_template, coord_fn, design_fn, name, n_folds=3, seed=67,
):
    """3-fold CV across templates: 21 train templates, 7 held out.

    Z_per_template: (n_colors, n_templates, k_pcs)
    coord_fn: () -> coords (n_colors, d)  [coords don't depend on template]
    design_fn: coords -> (X, S)

    For each fold:
      - average Z over train templates -> Y_train (n_colors, K)
      - average Z over test templates -> Y_test
      - fit per-column on Y_train, predict on coords (same coords), compute
        R² on Y_test (per-column average then mean).
    """
    n_c, n_t, K = Z_per_template.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_t)
    folds = np.array_split(perm, n_folds)
    r2s = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n_t), test_idx)
        Y_tr = Z_per_template[:, train_idx, :].mean(axis=1)
        Y_te = Z_per_template[:, test_idx, :].mean(axis=1)
        # Note: coord_fn / design_fn are deterministic given the labels;
        # the same X is used for both halves (the response moves, not coords).
        try:
            X, S = design_fn()
        except Exception as exc:
            warnings.warn(f"[{name}] fold {fi} design failed: {exc!r}")
            r2s.append(float("nan"))
            continue
        # Total variance of held-out (sum of column variances * n)
        y_te_center = Y_te - Y_te.mean(axis=0, keepdims=True)
        ss_tot = float((y_te_center ** 2).sum()) + 1e-12
        # Per-column ridge-via-REML fit on Y_tr, predict via fitted on coords
        fitted_te = np.zeros_like(Y_te)
        for k in range(K):
            try:
                out = gamfit.gaussian_reml_fit(
                    np.ascontiguousarray(X),
                    np.ascontiguousarray(Y_tr[:, k:k + 1]),
                    np.ascontiguousarray(S),
                )
                coef = np.asarray(out["coefficients"]).reshape(-1)
                fitted_te[:, k] = X @ coef
            except Exception:
                fitted_te[:, k] = Y_tr[:, k].mean()
        ss_res = float(((Y_te - fitted_te) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        r2s.append(float(r2))
    return float(np.nanmean(r2s)), float(np.nanstd(r2s))


# ---------- topology registry --------------------------------------------
def build_topologies(hsv, rgb):
    """Returns list of (name, design_factory, predicted_winner_hint)."""
    topo = []

    def euclid_design():
        t = coords_euclidean(hsv).ravel()
        return bspline_design_penalty(t, n_knots=10, degree=3, penalty_order=2)
    topo.append(("Euclidean", euclid_design))

    def circle_design():
        th = coords_circle(hsv).ravel()
        return fourier_basis(th, n_harmonics=6), fourier_penalty_sqrt(6)
    topo.append(("Circle", circle_design))

    def sphere_design():
        ll = coords_sphere(rgb)
        return sphere_design_penalty(ll, n_centers=30, penalty_order=2)
    topo.append(("Sphere", sphere_design))

    def torus_design():
        c = coords_torus(hsv)
        X = tensor_fourier_design(c[:, 0], c[:, 1], h1=4, h2=4)
        S = tensor_fourier_penalty_sqrt(4, 4)
        return X, S
    topo.append(("Torus", torus_design))

    def cylinder_design():
        c = coords_cylinder(hsv)
        return cylinder_design_penalty(c[:, 0], c[:, 1], h_theta=4, n_knots_z=8)
    topo.append(("Cylinder", cylinder_design))

    return topo


# ---------- main ---------------------------------------------------------
def main():
    t_start = time.time()
    print("[auto_exp_67] Topology selector for cogito-L40 HSV-supervised subspace")
    print(f"[gamfit] version = {gamfit.__version__}  "
          f"compare_models present? {hasattr(gamfit, 'compare_models')}")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X = {X.shape}  (D = {X.shape[1]})")
    n_c = X.shape[0] // N_TEMPLATES
    print(f"[data] n_colors = {n_c}  n_templates = {N_TEMPLATES}")

    basis = load_pc_basis(K=64)
    print(f"[pca] cached at {basis['cached_path']}  evr[:5] = "
          f"{basis['evr'][:5].round(3).tolist()}")

    print("[stream] projecting per-color per-template PCs ...")
    Z = per_color_per_template_pcs(X, basis, K_PCS, N_TEMPLATES)
    print(f"[stream] Z = {Z.shape}")
    # Color-focused TOP_TEMPLATES average for the global fit
    TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
    Y_global = Z[:, TOP_TEMPLATES, :].mean(axis=1)
    print(f"[stream] Y_global = {Y_global.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    print(f"[labels] hsv ranges: hue=[{hsv[:,0].min():.3f},{hsv[:,0].max():.3f}] "
          f"sat=[{hsv[:,1].min():.3f},{hsv[:,1].max():.3f}] "
          f"val=[{hsv[:,2].min():.3f},{hsv[:,2].max():.3f}]")

    topologies = build_topologies(hsv, rgb)

    rows = []
    for name, design_fn in topologies:
        t_fit = time.time()
        try:
            X_des, S_pen = design_fn()
        except Exception as exc:
            print(f"[{name}] DESIGN FAILED: {exc!r}")
            rows.append({
                "topology": name, "status": "design_failed",
                "error": repr(exc),
            })
            continue
        print(f"[{name}] design = {X_des.shape}  penalty = {S_pen.shape}")
        res = fit_one_topology(X_des, S_pen, Y_global, name)
        if "error" in res:
            print(f"[{name}] FIT FAILED: {res['error']}")
            rows.append({
                "topology": name, "status": "fit_failed",
                "error": res["error"],
            })
            continue

        r2_mean, r2_std = holdout_r2_by_template(
            Z, lambda hsv=hsv: None, design_fn,
            name=name, n_folds=3, seed=67,
        )
        elapsed = time.time() - t_fit
        rows.append({
            "topology": name,
            "status": "ok",
            "reml": res["reml"],
            "bic": res["bic"],
            "tk": res["tk"],
            "edf": res["edf"],
            "sse": res["sse"],
            "n_params": res["n_params"],
            "n_obs": res["n_obs"],
            "holdout_r2_mean": r2_mean,
            "holdout_r2_std": r2_std,
            "fit_failures": res["fit_failures"],
            "elapsed_sec": elapsed,
        })
        print(f"[{name}] REML={res['reml']:.2f}  BIC={res['bic']:.2f}  "
              f"TK={res['tk']:.2f}  EDF={res['edf']:.2f}  "
              f"R²={r2_mean:.3f}±{r2_std:.3f}  ({elapsed:.1f}s)")

    # ---------- ranking + interpretation -------------------------------
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("[verdict] no topology fit successfully")
        verdict = "all_failed"
        winner = None
        margin_reml = float("nan")
        margin_bic = float("nan")
    else:
        by_reml = sorted(ok_rows, key=lambda r: r["reml"])
        by_bic = sorted(ok_rows, key=lambda r: r["bic"])
        by_r2 = sorted(ok_rows, key=lambda r: -r["holdout_r2_mean"])
        winner = by_reml[0]["topology"]
        margin_reml = by_reml[1]["reml"] - by_reml[0]["reml"] if len(by_reml) > 1 else float("inf")
        margin_bic = by_bic[1]["bic"] - by_bic[0]["bic"] if len(by_bic) > 1 else float("inf")
        verdict_map = {
            "Circle": "Hue-ring confirmed: cogito-L40 color subspace is S^1",
            "Torus": "Hue x value 2-torus: perceptual + lightness ring structure",
            "Sphere": "RGB-like sphere geometry dominates",
            "Cylinder": "S^1 x R: hue ring with lightness axis",
            "Euclidean": "Manifold hypothesis falsified: flat axis dominates",
        }
        verdict = verdict_map.get(winner, f"unexpected winner: {winner}")
        print(f"\n[ranking REML] " + " < ".join(
            f"{r['topology']}({r['reml']:.1f})" for r in by_reml))
        print(f"[ranking BIC]  " + " < ".join(
            f"{r['topology']}({r['bic']:.1f})" for r in by_bic))
        print(f"[ranking R²]   " + " > ".join(
            f"{r['topology']}({r['holdout_r2_mean']:.3f})" for r in by_r2))
        print(f"\n[verdict] winner={winner}  "
              f"ΔREML={margin_reml:.2f}  ΔBIC={margin_bic:.2f}")
        print(f"[interpret] {verdict}")

    # ---------- save table ---------------------------------------------
    summary = {
        "experiment": "auto_exp_67_topology_selector",
        "gamfit_version": gamfit.__version__,
        "compare_models_used": False,
        "n_colors": int(n_c),
        "n_templates": int(N_TEMPLATES),
        "k_pcs": int(K_PCS),
        "rows": rows,
        "winner": winner,
        "margin_reml": margin_reml,
        "margin_bic": margin_bic,
        "verdict": verdict,
        "runtime_sec": time.time() - t_start,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[json] saved {OUT_JSON}")

    # ---------- 2-panel plot -------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if ok_rows:
        names_ok = [r["topology"] for r in ok_rows]
        remls = [r["reml"] for r in ok_rows]
        bics = [r["bic"] for r in ok_rows]
        r2_m = [r["holdout_r2_mean"] for r in ok_rows]
        r2_s = [r["holdout_r2_std"] for r in ok_rows]

        ax = axs[0]
        # Plot REML (lower = better evidence; flip sign to "evidence")
        evidence = [-r for r in remls]
        bars = ax.bar(names_ok, evidence, color="#3a7", edgecolor="k")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("evidence (= -REML; higher = better)")
        ax.set_title("Marginal likelihood per topology")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
        # Annotate BIC
        for b, bic in zip(bars, bics):
            ax.annotate(f"BIC={bic:.0f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)

        ax = axs[1]
        ax.bar(names_ok, r2_m, yerr=r2_s, color="#37a", edgecolor="k", capsize=4)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("holdout R² (3-fold across templates)")
        ax.set_title("Predictive performance per topology")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"auto_exp_67: topology selector for cogito-L40 HSV subspace\n"
        f"winner = {winner}   ΔREML = {margin_reml:.1f}   ΔBIC = {margin_bic:.1f}",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
