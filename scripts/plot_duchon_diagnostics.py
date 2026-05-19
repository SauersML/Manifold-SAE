"""Hard look at the 1D open Duchon basis + REML fit. Six panels.

Tests for any bug beyond inherent boundary bias:
1. The K=12 Duchon basis functions over [0, 1].
2. The K=30 basis functions (denser; should not "blow up" anywhere).
3. Noiseless fit of a single-coordinate cosine (truth = cos(2πt)) — fit vs truth.
4. Same, residuals vs t.
5. Compare REML λ vs OLS interpolation (manual ridge with tiny λ).
6. Same fit on a half-period truth (truth = cos(π·t), which is monotone on [0,1]
   so an open basis has no "seam" excuse to be bad).
"""

from __future__ import annotations

import numpy as np
import gamfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_reml_single(t, y, K, init_lambda=1.0):
    centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
    penalty = np.eye(K, dtype=np.float64)
    offsets = np.array([0, len(t)], dtype=np.uintp)
    by = np.ones(len(t), dtype=np.float64)
    out = gamfit.gaussian_reml_fit_positions_batched(
        np.ascontiguousarray(t, dtype=np.float64),
        np.ascontiguousarray(y[:, None] if y.ndim == 1 else y, dtype=np.float64),
        offsets, "duchon", centers, penalty,
        basis_order=2, periodic=False, period=None,
        by=by, init_lambda=init_lambda,
    )
    return np.asarray(out["coefficients"])[0], float(np.asarray(out["lambda"])[0]), centers


def fit_ols_ridge(t, y, K, lam=1e-10):
    """Manual interpolation: same Duchon basis, hand-rolled near-zero ridge."""
    centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
    phi = np.asarray(gamfit.duchon_basis_1d(np.ascontiguousarray(t, dtype=np.float64),
                                             centers, m=2, periodic=False))
    A = phi.T @ phi + lam * np.eye(K)
    b = phi.T @ (y[:, None] if y.ndim == 1 else y)
    return np.linalg.solve(A, b), centers


def main() -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    t_grid = np.linspace(0.0, 1.0, 401, dtype=np.float64)

    # Panel 1: K=12 basis functions
    ax = fig.add_subplot(gs[0, 0])
    K = 12
    centers = np.linspace(0, 1, K, dtype=np.float64)
    phi = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))
    for k in range(K):
        ax.plot(t_grid, phi[:, k], lw=1)
    ax.set_title(f"Duchon m=2 basis, K={K} (uniform centers shown as red dots)")
    ax.scatter(centers, np.zeros_like(centers), color="red", s=30, zorder=5)
    ax.set_xlabel("t"); ax.set_ylabel("phi_k(t)"); ax.grid(alpha=0.3)

    # Panel 2: K=30 basis
    ax = fig.add_subplot(gs[0, 1])
    K = 30
    centers = np.linspace(0, 1, K, dtype=np.float64)
    phi = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))
    for k in range(K):
        ax.plot(t_grid, phi[:, k], lw=0.7)
    ax.set_title(f"Duchon m=2 basis, K={K}  (should look denser, no blowup)")
    ax.scatter(centers, np.zeros_like(centers), color="red", s=15, zorder=5)
    ax.set_xlabel("t"); ax.set_ylabel("phi_k(t)"); ax.grid(alpha=0.3)

    # Panels 3 & 4: noiseless cosine fit (full period, [0,1])
    n = 2000
    t_data = np.sort(np.random.default_rng(0).uniform(0, 1, n)).astype(np.float64)
    y_data = np.cos(2 * np.pi * t_data)
    K = 12
    coef_reml, lam, centers = fit_reml_single(t_data, y_data, K)
    coef_ols, _ = fit_ols_ridge(t_data, y_data, K, lam=1e-10)
    phi_grid = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))
    fit_reml = (phi_grid @ coef_reml).ravel()
    fit_ols = (phi_grid @ coef_ols).ravel()
    truth = np.cos(2 * np.pi * t_grid)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_grid, truth, "k-", lw=2, label="truth: cos(2πt)")
    ax.plot(t_grid, fit_reml, "C1-", lw=1.5, label=f"REML fit (λ={lam:.2e})")
    ax.plot(t_grid, fit_ols, "C2--", lw=1.5, label="OLS (manual, λ=1e-10)")
    ax.set_title("Single-coord noiseless fit, n=2000, K=12, full period [0,1]")
    ax.set_xlabel("t"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_grid, fit_reml - truth, "C1-", lw=1.5, label=f"REML residual (mean |r|={np.abs(fit_reml-truth).mean():.4f})")
    ax.plot(t_grid, fit_ols - truth, "C2--", lw=1.5, label=f"OLS residual  (mean |r|={np.abs(fit_ols-truth).mean():.4f})")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Residual vs t (open Duchon, no seam between t=0 and t=1)")
    ax.set_xlabel("t"); ax.set_ylabel("fit − truth"); ax.legend(); ax.grid(alpha=0.3)

    # Panels 5 & 6: half-period truth (monotone, no closure issue at all)
    y_half = np.cos(np.pi * t_data)  # decreases monotonically from 1 to -1 over [0,1]
    coef_reml, lam_h, _ = fit_reml_single(t_data, y_half, K)
    coef_ols, _ = fit_ols_ridge(t_data, y_half, K, lam=1e-10)
    fit_reml = (phi_grid @ coef_reml).ravel()
    fit_ols = (phi_grid @ coef_ols).ravel()
    truth = np.cos(np.pi * t_grid)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_grid, truth, "k-", lw=2, label="truth: cos(πt)  (monotone, open by construction)")
    ax.plot(t_grid, fit_reml, "C1-", lw=1.5, label=f"REML fit (λ={lam_h:.2e})")
    ax.plot(t_grid, fit_ols, "C2--", lw=1.5, label="OLS (manual)")
    ax.set_title("Half-period truth (no periodicity excuse for edge error)")
    ax.set_xlabel("t"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_grid, fit_reml - truth, "C1-", lw=1.5, label=f"REML residual (mean |r|={np.abs(fit_reml-truth).mean():.4f})")
    ax.plot(t_grid, fit_ols - truth, "C2--", lw=1.5, label=f"OLS residual  (mean |r|={np.abs(fit_ols-truth).mean():.4f})")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Residual vs t — if there's a Duchon bug, it shows up here")
    ax.set_xlabel("t"); ax.set_ylabel("fit − truth"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("1D open Duchon basis + REML, controlled noiseless tests", fontsize=14, y=0.995)
    out_path = "runs/duchon_diagnostics.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
