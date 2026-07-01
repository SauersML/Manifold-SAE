"""Train SINDy-SAE on Lorenz attractor data; verify coefficient recovery.

Lorenz system (σ=10, ρ=28, β=8/3):
    dx/dt = σ (y - x)
    dy/dt = x (ρ - z) - y
    dz/dt = x y - β z

Goal: ||Θ_recovered - Θ_lorenz||_F / ||Θ_lorenz||_F < 0.1.

Run:
    python -m scripts.train_sindy_synthetic_lorenz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from manifold_sae.sindy_sae import (
    SINDySAE,
    library_size,
    library_term_names,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_rhs(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array(
        [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z],
        dtype=np.float64,
    )


def integrate_lorenz(
    state0: tuple[float, float, float] = (-8.0, 7.0, 27.0),
    dt: float = 0.001,
    n_steps: int = 20000,
    burn_in: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 integrate the Lorenz system. Returns (Z, dZ) with shapes (T, 3)."""
    state = np.array(state0, dtype=np.float64)
    traj = np.empty((n_steps, 3), dtype=np.float64)
    for i in range(n_steps):
        k1 = lorenz_rhs(state)
        k2 = lorenz_rhs(state + 0.5 * dt * k1)
        k3 = lorenz_rhs(state + 0.5 * dt * k2)
        k4 = lorenz_rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = state
    traj = traj[burn_in:]
    dtraj = np.stack([lorenz_rhs(s) for s in traj], axis=0)
    return traj.astype(np.float32), dtraj.astype(np.float32)


# ---------------------------------------------------------------------------
# Build the GROUND-TRUTH Θ in the chosen library
# ---------------------------------------------------------------------------


LIBRARY = ("constant", "identity", "square", "product")


def true_theta(state_dim: int = 3) -> np.ndarray:
    """Ground-truth Θ ∈ R^{3 × P} in the LIBRARY ordering.

    Library columns (state_dim=3):
        constant: [1]                                 -> 1 col
        identity: [z0, z1, z2]                        -> 3 cols
        square:   [z0^2, z1^2, z2^2]                  -> 3 cols
        product:  [z0*z1, z0*z2, z1*z2]               -> 3 cols
      P = 10
    """
    P = library_size(state_dim, LIBRARY)
    Theta = np.zeros((state_dim, P), dtype=np.float32)
    names = library_term_names(state_dim, LIBRARY)
    name_to_idx = {n: i for i, n in enumerate(names)}
    # dx/dt = σ(y - x) = -σ z0 + σ z1
    Theta[0, name_to_idx["z0"]] = -SIGMA
    Theta[0, name_to_idx["z1"]] = SIGMA
    # dy/dt = x(ρ - z) - y = ρ z0 - z1 - z0*z2
    Theta[1, name_to_idx["z0"]] = RHO
    Theta[1, name_to_idx["z1"]] = -1.0
    Theta[1, name_to_idx["z0*z2"]] = -1.0
    # dz/dt = x y - β z = -β z2 + z0*z1
    Theta[2, name_to_idx["z2"]] = -BETA
    Theta[2, name_to_idx["z0*z1"]] = 1.0
    return Theta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    state_dim: int = 3
    n_steps: int = 20000
    dt: float = 0.001
    lr: float = 5e-3
    n_iters: int = 6000
    sparsity: float = 0.05
    threshold_eps: float = 0.05
    n_stlsq_rounds: int = 4
    seed: int = 0
    out_dir: Path = Path("runs/SINDY_LORENZ")


def train(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Z, dZ = integrate_lorenz(dt=cfg.dt, n_steps=cfg.n_steps)
    # Normalize state to help optimization (we will UNDO this before comparing Θ).
    z_mean = Z.mean(axis=0, keepdims=True)
    z_std = Z.std(axis=0, keepdims=True) + 1e-8
    # SINDy is easier WITHOUT normalization because the library terms are
    # nonlinear; we therefore work in original units and just scale the loss.
    # Match torch default dtype (tests set f64 globally via conftest).
    _dtype = torch.get_default_dtype()
    Zt = torch.from_numpy(Z).to(_dtype)
    dZt = torch.from_numpy(dZ).to(_dtype)

    sindy = SINDySAE(
        state_dim=cfg.state_dim,
        library_terms=LIBRARY,
        sparsity=cfg.sparsity,
        init_scale=0.0,  # start at 0; least-squares step will fill in
    )

    # ---- least-squares warm start: solve Θ ≈ dZ φ(Z)^+ ------------------
    with torch.no_grad():
        phi = sindy.phi(Zt)  # (T, P)
        sol = torch.linalg.lstsq(phi, dZt).solution  # (P, 3)
        sindy.Theta.data.copy_(sol.T)

    opt = torch.optim.Adam(sindy.parameters(), lr=cfg.lr)
    history: list[dict] = []
    for it in range(cfg.n_iters):
        opt.zero_grad()
        out = sindy.loss(Zt, dZt)
        out["total"].backward()
        opt.step()
        if it % 500 == 0 or it == cfg.n_iters - 1:
            history.append({k: float(v.detach()) for k, v in out.items()} | {"it": it})

    # ---- STLSQ rounds: threshold then refit unmasked entries ------------
    for _ in range(cfg.n_stlsq_rounds):
        sindy.threshold(cfg.threshold_eps)
        # Refit Adam under current mask
        opt = torch.optim.Adam(sindy.parameters(), lr=cfg.lr * 0.3)
        for _ in range(1500):
            opt.zero_grad()
            out = sindy.loss(Zt, dZt, sparsity=0.0)
            out["total"].backward()
            # Keep masked entries at zero
            with torch.no_grad():
                sindy.Theta.grad.mul_(sindy.mask)
            opt.step()
            with torch.no_grad():
                sindy.Theta.data.mul_(sindy.mask)

    Theta_hat = sindy.effective_Theta().detach().cpu().numpy()
    Theta_true = true_theta(cfg.state_dim)
    rel_err = float(
        np.linalg.norm(Theta_hat - Theta_true) / max(np.linalg.norm(Theta_true), 1e-12)
    )

    # Save
    np.save(cfg.out_dir / "theta_hat.npy", Theta_hat)
    np.save(cfg.out_dir / "theta_true.npy", Theta_true)
    np.save(cfg.out_dir / "Z.npy", Z)
    np.save(cfg.out_dir / "dZ.npy", dZ)

    names = library_term_names(cfg.state_dim, LIBRARY)
    report = {
        "rel_frobenius_error": rel_err,
        "library_terms": list(LIBRARY),
        "library_columns": names,
        "Theta_hat": Theta_hat.tolist(),
        "Theta_true": Theta_true.tolist(),
        "history": history,
        "out_dir": str(cfg.out_dir),
    }
    import json

    (cfg.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = train()
    print(f"||Θ̂ - Θ*||_F / ||Θ*||_F = {rep['rel_frobenius_error']:.4f}")
    print("OK" if rep["rel_frobenius_error"] < 0.1 else "MISS (>0.1)")
