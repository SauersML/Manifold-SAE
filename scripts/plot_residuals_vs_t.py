"""Diagnostic: residual magnitude vs t for the GT-input gamfit recovery.

Two panels per feature:
  top:    ||fit(t) - truth(t)||  vs  t   (red overlay = histogram of training t)
  bottom: per-component residual (fit_d - truth_d) vs t for the first 3 dims

If residuals at endpoints are massively larger than interior, that diagnoses
either a basis edge effect or a data-density edge effect (or both).
"""

from __future__ import annotations

import numpy as np
import gamfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_sae.data_synthetic import SyntheticDataset


def main() -> None:
    ds = SyntheticDataset(d_ambient=64, n_features=5, n_samples=8192, seed=0)
    gt = ds.ground_truth
    active = gt["active"].numpy().astype(bool)
    ts = gt["ts"].numpy()
    amps = gt["amps"].numpy() * active

    N, F = ts.shape
    y = ds.x.numpy().astype(np.float64)
    K = 12
    centers = np.linspace(0, 1, K, dtype=np.float64)
    penalty = np.eye(K, dtype=np.float64)
    t_packed = ts.T.reshape(-1).astype(np.float64)
    by_packed = amps.T.reshape(-1).astype(np.float64)
    y_packed = np.tile(y, (F, 1))
    offsets = (np.arange(F + 1, dtype=np.uintp) * np.uintp(N))

    out = gamfit.gaussian_reml_fit_positions_batched(
        t_packed, y_packed, offsets, "duchon", centers, penalty,
        basis_order=2, periodic=False, period=None, by=by_packed, init_lambda=1.0,
    )
    coef = np.asarray(out["coefficients"])  # (F, K, D)
    edf = np.asarray(out["edf"])

    t_grid = np.linspace(0, 1, 401)
    phi = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))
    curves_learned = np.einsum("tk,fkd->ftd", phi, coef)         # (F, T, D)
    curves_truth = np.stack([feat.evaluate(t_grid) for feat in ds.features], axis=0)
    residuals = curves_learned - curves_truth                    # (F, T, D)
    err_norm = np.linalg.norm(residuals, axis=-1)                # (F, T)

    # Per-feature t-distribution among active samples.
    fig, axes = plt.subplots(F, 2, figsize=(13, 2.6 * F), squeeze=False)
    for k, feat in enumerate(ds.features):
        ax_err, ax_resid = axes[k]
        n_active = int(active[:, k].sum())
        active_ts = ts[active[:, k], k]

        ax_err.plot(t_grid, err_norm[k], color="C0", lw=2, label="||fit - truth||")
        ax_err.set_ylabel("residual norm")
        ax_err.set_xlim(0, 1)
        ax_err.grid(alpha=0.3)
        # Twin axis: histogram of active training ts.
        ax_hist = ax_err.twinx()
        ax_hist.hist(active_ts, bins=40, alpha=0.25, color="red", label=f"training ts (N={n_active})")
        ax_hist.set_ylabel("count", color="red")
        ax_err.set_title(
            f"{feat.name}  edf={edf[k]:.1f}  "
            f"mid-mean err={err_norm[k, 100:300].mean():.4f}  "
            f"endpoint err={(err_norm[k, :5].mean()+err_norm[k, -5:].mean())/2:.4f}"
        )
        lines_l, labels_l = ax_err.get_legend_handles_labels()
        lines_r, labels_r = ax_hist.get_legend_handles_labels()
        ax_err.legend(lines_l + lines_r, labels_l + labels_r, loc="upper center", fontsize=8)

        n_show = min(3, residuals.shape[-1])
        for d in range(n_show):
            ax_resid.plot(t_grid, residuals[k, :, d], lw=1.0, label=f"dim {d}")
        ax_resid.axhline(0, color="black", lw=0.5)
        ax_resid.set_xlim(0, 1)
        ax_resid.set_xlabel("t")
        ax_resid.set_ylabel("residual")
        ax_resid.grid(alpha=0.3)
        ax_resid.legend(loc="upper center", fontsize=8, ncol=n_show)

    fig.suptitle(
        "Gamfit (GT-input) residuals vs t. Blue = ||fit - truth|| over t-grid; "
        "red = density of training t-values for that feature.",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = "runs/residuals_vs_t.png"
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")
    # Quantify edge vs interior bias per feature.
    for k, feat in enumerate(ds.features):
        edge = (err_norm[k, :10].mean() + err_norm[k, -10:].mean()) / 2
        interior = err_norm[k, 50:-50].mean()
        ratio = edge / max(interior, 1e-12)
        print(f"  {feat.name:10s}  edge_err={edge:.4f}  interior_err={interior:.4f}  edge/interior={ratio:.2f}x")


if __name__ == "__main__":
    main()
