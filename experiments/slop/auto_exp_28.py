"""auto_exp_28: NegBin vs Poisson GLM on modifier_count of cogito L40 colors.

Fills the non-Gaussian-family coverage gap from a1bc08ee82d17d012's
AUTO_EXP_PLAN.md by fitting NegBin and Poisson GLMs to a count response
(per-color modifier count, e.g. "pale dusty orange" -> 3) using
PCA(L40, K=16) per-color centroids as predictors.

Hypotheses:
  (a) REML(negbin) > REML(poisson)              -- overdispersion captured
  (b) R^2(negbin on modifier_count) >= 0.40    -- modifier count is real axis
  (c) theta_hat in (0.1, 100)                   -- not collapsed to Poisson/mixture

gamfit 0.1.112 does NOT export glm_reml_fit_latent — we use the scipy
fallback path described in the task spec (analytic NegBin log-lik via
scipy.optimize.minimize on a ridge-penalized linear predictor, with an
outer Brent search on theta for the negbin-with-estimated-theta fit).
The output JSON includes prediction_slots so a future v0.1.121 re-run
on the same design can drop in real REML scores.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from _pca_basis import load_pc_basis, project  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD_TXT = EXP_DIR / "xkcd_colors.txt"
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "auto_exp_28.png"
OUT_JSON = OUT_DIR / "auto_exp_28.json"

K_LATENT = 16
N_TEMPLATES = 28
RIDGE = 1e-2          # mild L2 on coefficients (acts like crude REML prior)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_color_names() -> list[str]:
    names: list[str] = []
    for line in XKCD_TXT.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("License") or s.startswith("Copyright"):
            continue
        # tab-separated:  "cloudy blue\t#acc2d9\t"
        parts = s.split("\t")
        names.append(parts[0].strip())
    return names


def modifier_count(name: str) -> int:
    """Number of modifier tokens preceding the color noun.

    Convention: the LAST whitespace-separated token is the head color noun
    ("blue", "green", "orange", ...). Everything before it is a modifier.
    "blue"            -> 0 modifiers
    "dark blue"       -> 1
    "dark pastel green" -> 2
    "pale baby blue"  -> 2
    "macaroni and cheese" -> 2  (yes, "and" counts as a modifier token)
    """
    toks = name.split()
    return max(0, len(toks) - 1)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (Z, y, names) where Z is (N_colors, K_LATENT) PCA scores
    and y is the integer modifier count per color."""
    print(f"[data] mmap-loading {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, D = X.shape
    n_colors = n_total // N_TEMPLATES
    print(f"[data] X={X.shape} -> {n_colors} colors x {N_TEMPLATES} templates", flush=True)

    print(f"[data] load_pc_basis(K=64) cached", flush=True)
    basis = load_pc_basis(K=64)

    # Per-color centroids over the TOP_TEMPLATES subset (matches the basis).
    from _pca_basis import _per_color_centroids
    centroids = _per_color_centroids(np.asarray(X[: n_colors * N_TEMPLATES]))
    print(f"[data] centroids={centroids.shape}", flush=True)

    Z_full = project(centroids, basis)          # (n_colors, 64)
    Z = Z_full[:, :K_LATENT]                    # truncate to K_LATENT

    names = load_color_names()[:n_colors]
    y = np.array([modifier_count(n) for n in names], dtype=np.int64)
    print(f"[data] modifier_count: min={y.min()} max={y.max()} mean={y.mean():.3f}"
          f" var={y.var():.3f} (var/mean={y.var()/max(y.mean(), 1e-6):.3f})", flush=True)
    return Z, y, names


# ---------------------------------------------------------------------------
# GLMs (fallback path — gamfit 0.1.112 has no glm_reml_fit_latent)
# ---------------------------------------------------------------------------
def _design(Z: np.ndarray) -> np.ndarray:
    """Add intercept column."""
    return np.hstack([np.ones((Z.shape[0], 1)), Z])


def poisson_neg_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                       ridge: float = RIDGE) -> float:
    eta = X @ beta
    eta = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta)
    ll = (y * eta - mu - gammaln(y + 1.0)).sum()
    return -ll + 0.5 * ridge * (beta[1:] ** 2).sum()


def poisson_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                 ridge: float = RIDGE) -> np.ndarray:
    eta = X @ beta
    eta = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta)
    g = -X.T @ (y - mu)
    pen = ridge * beta
    pen[0] = 0.0
    return g + pen


def negbin_neg_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                      theta: float, ridge: float = RIDGE) -> float:
    eta = X @ beta
    eta = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta)
    # NB log-lik in (theta, mu) parameterization
    ll = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
          + theta * (np.log(theta) - np.log(theta + mu))
          + y * (np.log(mu + 1e-300) - np.log(theta + mu))).sum()
    return -ll + 0.5 * ridge * (beta[1:] ** 2).sum()


def negbin_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                theta: float, ridge: float = RIDGE) -> np.ndarray:
    eta = X @ beta
    eta = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta)
    # d log L / d eta = (y - mu) * theta / (theta + mu)
    w = (y - mu) * theta / (theta + mu)
    g = -X.T @ w
    pen = ridge * beta
    pen[0] = 0.0
    return g + pen


def fit_poisson(X: np.ndarray, y: np.ndarray) -> dict:
    p = X.shape[1]
    beta0 = np.zeros(p)
    beta0[0] = np.log(max(y.mean(), 1e-3))
    res = minimize(poisson_neg_loglik, beta0, jac=poisson_grad,
                   args=(X, y, RIDGE), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    beta = res.x
    mu = np.exp(np.clip(X @ beta, -30, 30))
    # Saturated log-lik (per-row): y log y - y - lgamma(y+1) ; with 0*log0=0
    y_safe = np.where(y > 0, y, 1.0)
    ll_sat = (np.where(y > 0, y * np.log(y_safe), 0.0) - y - gammaln(y + 1.0)).sum()
    ll_fit = -(poisson_neg_loglik(beta, X, y, ridge=0.0))
    ll_null = -(poisson_neg_loglik(np.array([beta0[0]] + [0.0] * (p - 1)),
                                   X, y, ridge=0.0))
    dev = 2.0 * (ll_sat - ll_fit)
    dev_null = 2.0 * (ll_sat - ll_null)
    r2_dev = 1.0 - dev / dev_null if dev_null > 0 else float("nan")
    # crude REML-ish score: marginal-likelihood proxy = log-lik - 0.5 log|H + R|
    # We use the negative neg-loglik with the ridge as our reported "reml_proxy"
    reml_proxy = ll_fit - 0.5 * RIDGE * (beta[1:] ** 2).sum()
    return {
        "beta": beta, "mu": mu, "ll": ll_fit, "ll_sat": ll_sat,
        "deviance": dev, "deviance_null": dev_null, "r2_deviance": float(r2_dev),
        "reml_proxy": float(reml_proxy), "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


def fit_negbin_fixed(X: np.ndarray, y: np.ndarray, theta: float) -> dict:
    p = X.shape[1]
    beta0 = np.zeros(p)
    beta0[0] = np.log(max(y.mean(), 1e-3))
    res = minimize(negbin_neg_loglik, beta0, jac=negbin_grad,
                   args=(X, y, theta, RIDGE), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    beta = res.x
    mu = np.exp(np.clip(X @ beta, -30, 30))
    ll_fit = -(negbin_neg_loglik(beta, X, y, theta, ridge=0.0))
    # saturated NB log-lik: mu_i = y_i
    mu_sat = np.where(y > 0, y.astype(float), 1e-300)
    ll_sat = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
              + theta * (np.log(theta) - np.log(theta + mu_sat))
              + y * (np.log(mu_sat) - np.log(theta + mu_sat))).sum()
    null_beta = np.array([beta0[0]] + [0.0] * (p - 1))
    ll_null = -(negbin_neg_loglik(null_beta, X, y, theta, ridge=0.0))
    dev = 2.0 * (ll_sat - ll_fit)
    dev_null = 2.0 * (ll_sat - ll_null)
    r2_dev = 1.0 - dev / dev_null if dev_null > 0 else float("nan")
    reml_proxy = ll_fit - 0.5 * RIDGE * (beta[1:] ** 2).sum()
    return {
        "beta": beta, "mu": mu, "ll": ll_fit, "ll_sat": ll_sat, "theta": theta,
        "deviance": dev, "deviance_null": dev_null, "r2_deviance": float(r2_dev),
        "reml_proxy": float(reml_proxy), "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


def fit_negbin_estimate_theta(X: np.ndarray, y: np.ndarray) -> dict:
    """Outer Brent search on log10(theta); inner L-BFGS-B on beta. Records trace."""
    trace: list[tuple[float, float]] = []

    def neg_profile_ll(log10_theta: float) -> float:
        theta = float(10.0 ** log10_theta)
        f = fit_negbin_fixed(X, y, theta=theta)
        trace.append((theta, float(f["ll"])))
        return -float(f["ll"])

    res = minimize_scalar(neg_profile_ll, bounds=(-2.0, 3.0), method="bounded",
                          options={"xatol": 1e-3, "maxiter": 40})
    theta_hat = float(10.0 ** res.x)
    final = fit_negbin_fixed(X, y, theta=theta_hat)
    final["theta_hat"] = theta_hat
    final["theta_trace"] = trace
    return final


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(y, fits: dict, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: actual vs predicted scatter, all three families
    ax = axes[0]
    colors = {"poisson": "#1f77b4", "negbin_t1": "#ff7f0e",
              "negbin_t5": "#2ca02c", "negbin_est": "#d62728"}
    jitter = RNG.normal(0, 0.06, size=y.shape)
    for label, f in fits.items():
        ax.scatter(y + jitter, f["mu"], s=8, alpha=0.4,
                   color=colors.get(label, "k"),
                   label=f"{label} (R²_dev={f['r2_deviance']:.3f})")
    lim = max(float(y.max()) + 0.5, max(float(f["mu"].max()) for f in fits.values()) + 0.5)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("actual modifier_count (jittered)")
    ax.set_ylabel("predicted mu")
    ax.set_title("Actual vs predicted modifier_count\n(L40 PCA-16 design)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 2: REML-proxy / log-lik bars
    ax = axes[1]
    labels = list(fits.keys())
    ll_vals = [fits[k]["ll"] for k in labels]
    reml_vals = [fits[k]["reml_proxy"] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, ll_vals, width=0.4, label="log-lik (fit)", color="steelblue")
    ax.bar(x + 0.2, reml_vals, width=0.4, label="REML proxy", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("log-likelihood / REML proxy")
    ax.set_title("Model fit comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: theta-stability trace
    ax = axes[2]
    est = fits.get("negbin_est")
    if est is not None and est.get("theta_trace"):
        tr = sorted(est["theta_trace"])
        thetas = [t for t, _ in tr]
        lls = [l for _, l in tr]
        ax.semilogx(thetas, lls, "o-", color="#d62728", markersize=4)
        ax.axvline(est["theta_hat"], color="k", ls="--", lw=1,
                   label=f"θ̂ = {est['theta_hat']:.3f}")
        ax.set_xlabel("θ (log scale)")
        ax.set_ylabel("profile log-lik")
        ax.set_title("Brent profile search for θ")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no theta trace", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("auto_exp_28 — NegBin vs Poisson GLM on cogito L40 modifier_count",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def detect_gamfit_path() -> tuple[str, str]:
    """Returns (gamfit_version, path) where path is 'real' or 'fallback'."""
    import gamfit
    ver = getattr(gamfit, "__version__", "unknown")
    has_real = hasattr(gamfit, "glm_reml_fit_latent")
    if has_real:
        # also need negbin_theta kwarg support — best-effort probe
        try:
            import inspect
            sig = inspect.signature(gamfit.glm_reml_fit_latent)
            if "negbin_theta" in sig.parameters or "family" in sig.parameters:
                return ver, "real"
        except (ValueError, TypeError):
            pass
        return ver, "real_but_kwarg_unclear"
    return ver, "fallback"


def main() -> None:
    t0 = time.time()
    gamfit_ver, family_path = detect_gamfit_path()
    print(f"[gamfit] version={gamfit_ver} path={family_path}", flush=True)

    Z, y, names = load_data()
    print(f"[design] Z={Z.shape} y={y.shape}  (n={len(y)})", flush=True)

    Xd = _design(Z)

    print("[fit ] poisson ...", flush=True)
    f_pois = fit_poisson(Xd, y)
    print(f"        ll={f_pois['ll']:.2f}  R²_dev={f_pois['r2_deviance']:.4f}"
          f"  conv={f_pois['converged']}", flush=True)

    print("[fit ] negbin theta=1.0 ...", flush=True)
    f_nb1 = fit_negbin_fixed(Xd, y, theta=1.0)
    print(f"        ll={f_nb1['ll']:.2f}  R²_dev={f_nb1['r2_deviance']:.4f}"
          f"  conv={f_nb1['converged']}", flush=True)

    print("[fit ] negbin theta=5.0 ...", flush=True)
    f_nb5 = fit_negbin_fixed(Xd, y, theta=5.0)
    print(f"        ll={f_nb5['ll']:.2f}  R²_dev={f_nb5['r2_deviance']:.4f}"
          f"  conv={f_nb5['converged']}", flush=True)

    print("[fit ] negbin estimate theta (Brent profile) ...", flush=True)
    f_nbE = fit_negbin_estimate_theta(Xd, y)
    print(f"        θ̂={f_nbE['theta_hat']:.4f}  ll={f_nbE['ll']:.2f}"
          f"  R²_dev={f_nbE['r2_deviance']:.4f}", flush=True)

    fits = {
        "poisson": f_pois,
        "negbin_t1": f_nb1,
        "negbin_t5": f_nb5,
        "negbin_est": f_nbE,
    }
    make_plot(y, fits, OUT_PNG)

    # ---------- hypothesis verdicts ----------
    # Best NB (by log-lik) for the head-to-head vs Poisson
    best_nb_key = max(["negbin_t1", "negbin_t5", "negbin_est"],
                      key=lambda k: fits[k]["reml_proxy"])
    best_nb = fits[best_nb_key]
    h_a = bool(best_nb["reml_proxy"] > f_pois["reml_proxy"])
    h_b = bool(best_nb["r2_deviance"] >= 0.40)
    theta_hat = float(f_nbE["theta_hat"])
    h_c = bool(0.1 < theta_hat < 100.0)

    out = {
        "exp": "auto_exp_28",
        "description": "NegBin vs Poisson GLM on per-color modifier_count "
                       "with PCA(L40, K=16) design.",
        "gamfit_version": gamfit_ver,
        "family_path": family_path,            # 'real' or 'fallback'
        "n_colors": int(len(y)),
        "K_latent": K_LATENT,
        "ridge": RIDGE,
        "y_stats": {
            "min": int(y.min()), "max": int(y.max()),
            "mean": float(y.mean()), "var": float(y.var()),
            "var_over_mean": float(y.var() / max(y.mean(), 1e-6)),
        },
        "hypothesis_verdicts": {
            "a_negbin_reml_gt_poisson": h_a,
            "b_r2_negbin_ge_0p40": h_b,
            "c_theta_hat_in_0p1_to_100": h_c,
        },
        "best_negbin_key": best_nb_key,
        "R2_poisson": float(f_pois["r2_deviance"]),
        "R2_negbin_t1": float(f_nb1["r2_deviance"]),
        "R2_negbin_t5": float(f_nb5["r2_deviance"]),
        "R2_negbin_est": float(f_nbE["r2_deviance"]),
        "R2_negbin_best": float(best_nb["r2_deviance"]),
        "theta_hat": theta_hat,
        "theta_fixed_compared": [1.0, 5.0],
        "REML_poisson": float(f_pois["reml_proxy"]),
        "REML_negbin_t1": float(f_nb1["reml_proxy"]),
        "REML_negbin_t5": float(f_nb5["reml_proxy"]),
        "REML_negbin_est": float(f_nbE["reml_proxy"]),
        "loglik_poisson": float(f_pois["ll"]),
        "loglik_negbin_t1": float(f_nb1["ll"]),
        "loglik_negbin_t5": float(f_nb5["ll"]),
        "loglik_negbin_est": float(f_nbE["ll"]),
        "theta_trace": [[float(t), float(ll)] for t, ll in f_nbE["theta_trace"]],
        "runtime_seconds": float(time.time() - t0),
        "prediction_slot_for_v0_1_121": {
            "note": "Drop in gamfit.glm_reml_fit_latent(family='negbin', "
                    "negbin_theta=theta_hat, ...) results once gamfit ships "
                    "the latent-REML NB family. Compare against fallback REML "
                    "proxies above.",
            "expected_keys": [
                "REML_negbin_real", "REML_poisson_real",
                "theta_hat_real", "R2_negbin_real",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] wrote {OUT_JSON}", flush=True)
    print(f"[done] {out['runtime_seconds']:.1f}s  verdicts: "
          f"a={h_a}  b={h_b}  c={h_c}", flush=True)


if __name__ == "__main__":
    main()
