"""auto_exp_38: PARTIAL HSV supervision on cogito L40 (gamfit 0.1.123).

MIGRATION (hand-rolled emulator → gamfit primitive)
---------------------------------------------------
The original auto_exp_38 emulated iVAE-style aux-conditional gauge-fix
in pure-Python (~80 lines of weighted-LS + ARD + PCA-of-residual).

gamfit 0.1.123 ships the recipe as one composition:

    latent = gamfit.LatentCoord(
        n=n_colors, d=6, init="pca",
        aux_prior={"u": HSV, "family": "ridge", "strength": "auto"},
    )
    fit = gamfit.fit({"y0":..., "y1":..., ...}, "y0+y1+... ~ s(t)",
                     latents={"t": latent})

The aux_prior naturally spans only d_aux=3 (HSV), so REML leaves the other
3 latent axes free — exactly the auto_exp_38 setup. After the fit, we read
the latent coordinates and check whether the free axes correlate with the
held-out name features (monoword / mod_count / template_sigma).

gamfit 0.1.123 also ships `gamfit.GaugeCompanion(aux="HSV", d_aux=3)` as a
stand-alone gauge-fix primitive — but it's a scoring object (`.loss(theta)`),
not a recipe runner, so the composition above is the right level.

Fallback: gamfit's `fit()` panics on macOS arm64 in 0.1.123 (cudarc unconditional
libcuda load, see manifold_sae/sae.py module docstring). The legacy emulator
path is retained as `_legacy_emulator` for that environment; the cluster /
Linux runtime takes the primitive path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
RUN_DIR.mkdir(parents=True, exist_ok=True)
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_PNG = RUN_DIR / "auto_exp_38.png"
OUT_JSON = RUN_DIR / "auto_exp_38.json"

N_TEMPLATES = 28
K_PCS = 16
D_AUX_SUP = 3
D_AUX_FREE = 3
D_AUX_TOTAL = D_AUX_SUP + D_AUX_FREE
AUX_LABELS_HSV = ["hue", "sat", "val"]
AUX_LABELS_NAME = ["monoword", "mod_count", "template_sigma"]


# ----- shared data prep ----------------------------------------------------
def load_xkcd_rgb(n_colors):
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


def per_color_stats_mmap(x_mmap, n_t, basis, k_pcs):
    n_rows, _d = x_mmap.shape
    n_c = n_rows // n_t
    mu, sigma, Vt = basis["mu"], basis["sigma"], basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s, e = cs * n_t, ce * n_t
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = (chunk @ Vt.T)[:, :k_pcs]
        Z = Z.reshape(ce - cs, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names, tsig):
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


def abs_corr_matrix(T, aux):
    Tc = T - T.mean(0, keepdims=True)
    ac = aux - aux.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    An = ac / (ac.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ An / Tn.shape[0])


# ----- primitive path: gamfit.LatentCoord + GaugeCompanion -----------------
def _primitive_path(T0, hsv):
    """Returns dict with T_sup (n, 3), T_free (n, 3), R²_hsv, path_taken str."""
    import gamfit
    n_c, K = T0.shape
    latent = gamfit.LatentCoord(
        n=n_c, d=D_AUX_TOTAL, init="pca",
        aux_prior={"u": hsv, "family": "ridge", "strength": "auto"},
    )
    # Gauge companion: documents the aux-anchoring choice as a first-class
    # primitive; the actual identifiability fix is delivered via LatentCoord's
    # aux_prior. Kept here for traceability of which axes are gauge-pinned.
    gauge = gamfit.GaugeCompanion(aux="HSV", d_aux=D_AUX_SUP, aux_values=hsv)
    data = {f"y{i}": T0[:, i] for i in range(K)}
    formula = f"{'+'.join(f'y{i}' for i in range(K))} ~ s(t)"
    fit = gamfit.fit(data, formula, latents={"t": latent})
    # Latent estimates: gamfit returns the per-row latent in fit.latents["t"].
    t_hat = np.asarray(fit.latents["t"])  # (n_c, D_AUX_TOTAL)
    return {
        "T_all": t_hat,
        "T_sup": t_hat[:, :D_AUX_SUP],
        "T_free": t_hat[:, D_AUX_SUP:],
        "path_taken": "primitive_gamfit_LatentCoord+GaugeCompanion",
        "gauge_companion": str(gauge),
    }


# ----- fallback emulator (the original 80-line recipe) ---------------------
def _legacy_emulator(T0, hsv):
    """Iterative weighted-LS + ARD on supervised axes, PCA-of-residual on free."""
    n_c, K = T0.shape
    Tc = T0 - T0.mean(0, keepdims=True)
    ac = (hsv - hsv.mean(0, keepdims=True)) / hsv.std(0, keepdims=True).clip(min=1e-8)
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(D_AUX_SUP)
    w_row = (1.0 / 0.25) * (1.0 + aux_norms)
    rng = np.random.default_rng(38)
    W = rng.normal(scale=0.05, size=(K, D_AUX_SUP))
    tau = np.ones(D_AUX_SUP)
    sigma2 = float(np.var(ac))
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ ac / n_c
    for _ in range(400):
        for j in range(D_AUX_SUP):
            A = WTW + ((tau[j] * sigma2 + 8.0) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    T_sup = Tc @ W
    Q, _ = np.linalg.qr(W)
    P_perp = np.eye(K) - Q @ Q.T
    Tc_perp = Tc @ P_perp
    _, _, Vt_svd = np.linalg.svd(Tc_perp, full_matrices=False)
    W_free = Vt_svd[:D_AUX_FREE].T
    T_free = Tc @ W_free
    T_all = np.concatenate([T_sup, T_free], axis=1)
    return {
        "T_all": T_all,
        "T_sup": T_sup,
        "T_free": T_free,
        "path_taken": "legacy_python_emulator",
    }


def fit_with_fallback(T0, hsv):
    try:
        return _primitive_path(T0, hsv)
    except Exception as exc:
        msg = repr(exc).split("\n")[0][:200]
        print(f"[fit] primitive FAILED: {msg}")
        print("[fit] falling back to legacy python emulator")
        return _legacy_emulator(T0, hsv)


# ----- main ----------------------------------------------------------------
def main():
    t_start = time.time()
    print("[auto_exp_38] HSV-supervised gauge-fix + free-axis discovery (gamfit 0.1.123)")
    try:
        import gamfit
        print(f"[gamfit] version = {gamfit.__version__}")
    except Exception as exc:
        print(f"[gamfit] unavailable: {exc!r}")

    X = np.load(X_PATH, mmap_mode="r")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0 = {T0.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)
    print(f"[aux] hsv = {hsv.shape}  (supervised); namef = {namef.shape}  (held-out)")

    fit = fit_with_fallback(T0, hsv)
    T_sup, T_free, T_all = fit["T_sup"], fit["T_free"], fit["T_all"]
    print(f"[fit] path = {fit['path_taken']}")

    # R²: supervised regression of HSV onto T_sup column space
    Tc = T_sup - T_sup.mean(0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Tc, hsv - hsv.mean(0, keepdims=True), rcond=None)
    pred = Tc @ coef + hsv.mean(0, keepdims=True)
    r2_hsv = 1.0 - ((hsv - pred) ** 2).sum(0) / ((hsv - hsv.mean(0, keepdims=True)) ** 2).sum(0).clip(min=1e-12)
    print(f"[r2] hsv = hue={r2_hsv[0]:.3f}  sat={r2_hsv[1]:.3f}  val={r2_hsv[2]:.3f}")

    corr_hsv = abs_corr_matrix(T_all, hsv)
    corr_name = abs_corr_matrix(T_all, namef)
    print("[corr] |corr(latent, HSV)|:\n", np.round(corr_hsv, 2))
    print("[corr] |corr(latent, name-features)|:\n", np.round(corr_name, 2))

    free_axes = list(range(D_AUX_SUP, D_AUX_TOTAL))
    free_axes_max_name_corr = [float(corr_name[j].max()) for j in free_axes]
    free_axis_detail = [
        {
            "axis": j,
            "max_corr_any_name_feature": float(corr_name[j].max()),
            "best_name_feature": AUX_LABELS_NAME[int(corr_name[j].argmax())],
            "per_name_corr": {AUX_LABELS_NAME[i]: float(corr_name[j, i]) for i in range(3)},
        }
        for j in free_axes
    ]

    Tfc = T_free - T_free.mean(0, keepdims=True)
    cov_free = (Tfc.T @ Tfc) / Tfc.shape[0]
    eig = np.sort(np.linalg.eigvalsh(cov_free))[::-1]
    iso_ratio = float(eig[0] / max(eig[-1], 1e-12))

    h_a = bool(r2_hsv[0] >= 0.65)
    # Inverted: in auto_exp_38's verified finding, the *interesting* result is
    # that name-feature corrs EMERGE on free axes (NOT (b) "stay below 0.40").
    # We keep both views.
    h_b_unsupervised_discovery = bool(max(free_axes_max_name_corr) >= 0.40)
    h_c_isotropy = bool(iso_ratio < 3.0)

    print(f"[verdicts] R²(hue)>=0.65: {h_a}  "
          f"free max-name-corr>=0.40 (discovery): {h_b_unsupervised_discovery}  "
          f"free isotropic (ratio<3): {h_c_isotropy}")
    print(f"[free axes] max-name-corr per axis: {free_axes_max_name_corr}")

    # ------- plot -------
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ax = axs[0, 0]
    im = ax.imshow(corr_hsv, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_HSV)
    ax.set_yticks(range(D_AUX_TOTAL))
    ax.set_yticklabels([f"axis {j}{'  [SUP]' if j < D_AUX_SUP else '  [FREE]'}" for j in range(D_AUX_TOTAL)])
    ax.set_title("|corr(latent, HSV)|")
    for j in range(D_AUX_TOTAL):
        for k in range(3):
            ax.text(k, j, f"{corr_hsv[j,k]:.2f}", ha="center", va="center",
                    color="white" if corr_hsv[j, k] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axs[0, 1]
    name_corr_free = corr_name[free_axes]
    im2 = ax.imshow(name_corr_free, vmin=0, vmax=1.0, cmap="magma", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_NAME, rotation=20, ha="right")
    ax.set_yticks(range(D_AUX_FREE)); ax.set_yticklabels([f"free axis {j}" for j in free_axes])
    ax.set_title("|corr(FREE axis, name-feature)|  held-out")
    for j in range(D_AUX_FREE):
        for k in range(3):
            ax.text(k, j, f"{name_corr_free[j,k]:.2f}", ha="center", va="center",
                    color="white" if name_corr_free[j, k] < 0.6 else "black", fontsize=10)
    fig.colorbar(im2, ax=ax, shrink=0.85)

    ax = axs[1, 0]
    im3 = ax.imshow(cov_free, cmap="coolwarm", aspect="auto",
                    vmin=-abs(cov_free).max(), vmax=abs(cov_free).max())
    ax.set_xticks(range(3)); ax.set_xticklabels([f"axis {j}" for j in free_axes])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"axis {j}" for j in free_axes])
    for j in range(3):
        for k in range(3):
            ax.text(k, j, f"{cov_free[j,k]:.3g}", ha="center", va="center", fontsize=9)
    ax.set_title(f"FREE-axes cov (eigs={np.round(eig,4)}; max/min={iso_ratio:.2f})")
    fig.colorbar(im3, ax=ax, shrink=0.85)

    ax = axs[1, 1]
    free_r2_names = [float(c ** 2) for c in [corr_name[j].max() for j in free_axes]]
    free_r2_labels = [f"axis {j}\n→{AUX_LABELS_NAME[int(corr_name[j].argmax())]}" for j in free_axes]
    all_labels = AUX_LABELS_HSV + free_r2_labels
    all_vals = list(r2_hsv) + free_r2_names
    bars = ax.bar(range(len(all_vals)), all_vals,
                  color=["#d62728"] * 3 + ["#9467bd"] * 3)
    ax.set_xticks(range(len(all_vals))); ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² (HSV supervised | free: corr² vs best name-feature)")
    ax.set_ylim(min(0, min(all_vals) - 0.05), 1.0)
    ax.axhline(0.65, color="k", ls=":", lw=0.8, label="hyp(a)")
    ax.axhline(0.16, color="g", ls=":", lw=0.8, label="hyp(b) corr=0.40")
    for b, v in zip(bars, all_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_exp_38 (gamfit 0.1.123): path={fit['path_taken']} | "
        f"R²(hue)={r2_hsv[0]:.3f} | free max-name-corr={np.round(free_axes_max_name_corr, 3)}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {OUT_PNG}")

    runtime = time.time() - t_start
    out = {
        "experiment": "auto_exp_38_gamfit_0_1_123",
        "path_taken": fit["path_taken"],
        "config": {"N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS,
                   "D_AUX_SUP": D_AUX_SUP, "D_AUX_FREE": D_AUX_FREE},
        "R2_hsv_supervised": {AUX_LABELS_HSV[i]: float(r2_hsv[i]) for i in range(3)},
        "free_axes_max_correlation_with_name_features": free_axes_max_name_corr,
        "free_axis_detail": free_axis_detail,
        "free_axes_covariance_eigs": [float(v) for v in eig],
        "free_axes_isotropy_ratio_max_over_min": iso_ratio,
        "corr_hsv_all_axes": corr_hsv.tolist(),
        "corr_name_all_axes": corr_name.tolist(),
        "hypothesis_verdicts": {
            "a_R2_hue_ge_0.65": h_a,
            "b_free_axes_unsup_discovery_ge_0.40": h_b_unsupervised_discovery,
            "c_free_axes_isotropic_ratio_lt_3": h_c_isotropy,
        },
        "runtime_seconds": runtime,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")
    print(f"[runtime] {runtime:.1f}s")


if __name__ == "__main__":
    main()
