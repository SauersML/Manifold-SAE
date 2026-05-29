"""auto_exp_72_sindy_lorenz.py — Lorenz coefficient recovery via SINDy-SAE.

Verifies the SINDy-SAE architecture: atoms are SPARSE GOVERNING-EQUATION ROWS
of Θ ∈ R^{state_dim × P}, not static directions. We integrate the Lorenz
attractor, train SINDy-SAE on (z, dz/dt), and check whether the recovered
sparse Θ matches the known Lorenz coefficients (σ=10, ρ=28, β=8/3) within
||Θ_recovered - Θ_lorenz||_F / ||Θ_lorenz||_F < 0.1.

Outputs (under `runs/SINDY_LORENZ/`):
  theta_hat.npy, theta_true.npy, Z.npy, dZ.npy, report.json
  lorenz_recovery.png  — true vs recovered trajectory + Θ sparsity pattern
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_sindy_synthetic_lorenz import (
    LIBRARY,
    TrainConfig,
    integrate_lorenz,
    train,
    true_theta,
)
from manifold_sae.sindy_sae import library_term_names


def _simulate_from_theta(Theta: np.ndarray, state0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """RK4 integrate using Θ in the LIBRARY ordering: dz = Θ φ(z)."""
    names = library_term_names(3, LIBRARY)

    def phi(z: np.ndarray) -> np.ndarray:
        cols = []
        for n in names:
            if n == "1":
                cols.append(1.0)
            elif n.startswith("z") and "*" not in n and "^" not in n:
                cols.append(z[int(n[1:])])
            elif "^2" in n:
                cols.append(z[int(n[1:-2])] ** 2)
            elif "*" in n:
                i, j = n.split("*")
                cols.append(z[int(i[1:])] * z[int(j[1:])])
            else:
                raise RuntimeError(n)
        return np.array(cols)

    def rhs(z):
        return Theta @ phi(z)

    traj = np.empty((n_steps, 3))
    z = state0.copy()
    for i in range(n_steps):
        k1 = rhs(z)
        k2 = rhs(z + 0.5 * dt * k1)
        k3 = rhs(z + 0.5 * dt * k2)
        k4 = rhs(z + dt * k3)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = z
    return traj


def main() -> None:
    cfg = TrainConfig(out_dir=Path("runs/SINDY_LORENZ"))
    report = train(cfg)
    rel_err = report["rel_frobenius_error"]
    print(f"||Θ̂ - Θ*||_F / ||Θ*||_F = {rel_err:.4f}")

    Theta_hat = np.array(report["Theta_hat"], dtype=np.float64)
    Theta_true = np.array(report["Theta_true"], dtype=np.float64)
    names = report["library_columns"]

    # ---- simulate both for trajectory comparison ------------------------
    state0 = np.array([-8.0, 7.0, 27.0])
    Z_true, _ = integrate_lorenz(state0=tuple(state0), dt=cfg.dt, n_steps=cfg.n_steps, burn_in=0)
    Z_hat = _simulate_from_theta(Theta_hat, state0, dt=cfg.dt, n_steps=cfg.n_steps)

    # ---- plot ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot(Z_true[:, 0], Z_true[:, 1], Z_true[:, 2], lw=0.4, c="k")
    ax1.set_title("true Lorenz")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot(Z_hat[:, 0], Z_hat[:, 1], Z_hat[:, 2], lw=0.4, c="C3")
    ax2.set_title(f"recovered (rel err {rel_err:.3f})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    ax3 = fig.add_subplot(1, 3, 3)
    vmax = max(np.abs(Theta_true).max(), np.abs(Theta_hat).max())
    pattern = np.where(
        np.abs(Theta_hat) > 1e-3,
        np.sign(Theta_hat),
        0.0,
    )
    # overlay: true non-zero entries outlined
    im = ax3.imshow(Theta_hat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(Theta_true.shape[0]):
        for j in range(Theta_true.shape[1]):
            if abs(Theta_true[i, j]) > 1e-8:
                ax3.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="lime", lw=1.6))
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["dx/dt", "dy/dt", "dz/dt"])
    ax3.set_title("Θ̂ (green box = true non-zero)")
    plt.colorbar(im, ax=ax3, fraction=0.04)

    fig.tight_layout()
    out_png = cfg.out_dir / "lorenz_recovery.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"plot → {out_png}")

    (cfg.out_dir / "auto_exp_72_summary.json").write_text(
        json.dumps(
            {
                "rel_frobenius_error": rel_err,
                "passed": rel_err < 0.1,
                "plot": str(out_png),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
