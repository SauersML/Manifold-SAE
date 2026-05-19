"""Visualize the Duchon basis conditioning bug and the fix.

Top row:
  L: raw basis, all 12 columns plotted on the same y-axis. Polynomial columns
     (constant=1, linear=t) dwarf the 10 RBF columns. With identity penalty,
     the RBFs are invisible to the solver.
  R: same basis after column normalization (each col scaled to unit L2 norm).
     Now every column has comparable magnitude.

Bottom row: noiseless cos(2πt) fit with each parametrization.
  L: raw basis + identity penalty (current gamfit_glue behavior).
  R: column-normalized basis + identity penalty.
"""

from __future__ import annotations

import numpy as np
import gamfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    K = 12
    centers = np.linspace(0, 1, K, dtype=np.float64)
    t_grid = np.linspace(0, 1, 401, dtype=np.float64)
    phi_raw = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))

    # Normalize each column to unit L2 norm.
    col_norms = np.linalg.norm(phi_raw, axis=0)
    phi_norm = phi_raw / col_norms[None, :]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Top-left: raw basis.
    ax = axes[0, 0]
    for k in range(K):
        ax.plot(t_grid, phi_raw[:, k], lw=1, label=f"k={k}  ||col||={col_norms[k]:.3f}")
    ax.set_title(f"RAW Duchon basis (gamfit default). Col norms span {col_norms.min():.3f} to {col_norms.max():.2f}\n"
                 f"Two cols (constant=1, linear=t) dominate; the 10 RBF cols are invisible.")
    ax.set_xlabel("t"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))

    # Top-right: normalized basis.
    ax = axes[0, 1]
    for k in range(K):
        ax.plot(t_grid, phi_norm[:, k], lw=1, label=f"k={k}")
    ax.set_title("COLUMN-NORMALIZED Duchon basis (each col rescaled to ||col||=1)\n"
                 "Now you can actually see all 12 functions; RBFs are bumps at each center.")
    ax.set_xlabel("t"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))

    # Now fit noiseless cos(2πt) with each parametrization, identity penalty, fixed λ.
    n = 2000
    t_data = np.sort(np.random.default_rng(0).uniform(0, 1, n)).astype(np.float64)
    y_data = np.cos(2 * np.pi * t_data)[:, None]
    truth = np.cos(2 * np.pi * t_grid)
    phi_train_raw = np.asarray(gamfit.duchon_basis_1d(t_data, centers, m=2, periodic=False))
    phi_train_norm = phi_train_raw / col_norms[None, :]

    # Same identity penalty, same nominal lambda for fair comparison.
    lam = 1e-4
    def fit(phi, lam):
        A = phi.T @ phi + lam * np.eye(K)
        b = phi.T @ y_data
        return np.linalg.solve(A, b)
    B_raw = fit(phi_train_raw, lam)
    B_norm = fit(phi_train_norm, lam)
    fit_raw = (phi_raw @ B_raw).ravel()
    fit_norm = (phi_norm @ B_norm).ravel()

    # Per-bin error.
    def per_bin(err, t, n_bins=20):
        bins = np.linspace(0, 1, n_bins + 1)
        out = np.full(n_bins, np.nan)
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mb = (t >= lo) & (t < hi)
            if mb.sum() > 0:
                out[i] = np.abs(err[mb]).mean()
        return 0.5 * (bins[:-1] + bins[1:]), out

    ax = axes[1, 0]
    err_raw = fit_raw - truth
    bcent, berr = per_bin(err_raw, t_grid)
    ax.plot(t_grid, truth, "k-", lw=1.5, label="truth: cos(2πt)")
    ax.plot(t_grid, fit_raw, "C1-", lw=1.5, label=f"fit (mean |err|={np.abs(err_raw).mean():.4f})")
    ax.fill_between(t_grid, truth - 0.005, truth + 0.005, color="black", alpha=0.1)
    ax.set_title(f"RAW basis + identity penalty λ={lam}\n"
                 f"cond(Φ'Φ + λI) = {np.linalg.cond(phi_train_raw.T @ phi_train_raw + lam*np.eye(K)):.1e}")
    ax.set_xlabel("t"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    err_norm = fit_norm - truth
    ax.plot(t_grid, truth, "k-", lw=1.5, label="truth: cos(2πt)")
    ax.plot(t_grid, fit_norm, "C2-", lw=1.5, label=f"fit (mean |err|={np.abs(err_norm).mean():.4f})")
    ax.fill_between(t_grid, truth - 0.005, truth + 0.005, color="black", alpha=0.1)
    ax.set_title(f"COLUMN-NORMALIZED basis + identity penalty λ={lam}\n"
                 f"cond(Φ'Φ + λI) = {np.linalg.cond(phi_train_norm.T @ phi_train_norm + lam*np.eye(K)):.1e}")
    ax.set_xlabel("t"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        "Duchon basis conditioning. RAW basis is what gamfit_glue uses today; "
        "column-normalized is the one-line fix.", fontsize=13, y=0.995
    )
    fig.tight_layout()
    out_path = "runs/duchon_conditioning.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"wrote {out_path}")
    print(f"raw         basis: mean |err| = {np.abs(err_raw).mean():.4f}  edge |err| = {(np.abs(err_raw)[:20].mean()+np.abs(err_raw)[-20:].mean())/2:.4f}")
    print(f"normalized  basis: mean |err| = {np.abs(err_norm).mean():.4f}  edge |err| = {(np.abs(err_norm)[:20].mean()+np.abs(err_norm)[-20:].mean())/2:.4f}")


if __name__ == "__main__":
    main()
