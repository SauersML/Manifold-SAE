"""SINDy-SAE adapter for single-snapshot data — SMOKE TEST ONLY.

WARNING — READ THIS BEFORE USING:
=================================
This adapter pretends consecutive snapshots in BATCH ORDER form a trajectory
and computes a fake "time derivative"

    dz_fake[i] ≈ (z[i+1] - z[i]) / dt

This is a CLEARLY-BROKEN trick whose ONLY purpose is to validate that the
SINDy-SAE architectural pipeline (library evaluation, Θ regression, L1
sparsification) runs end-to-end on activation-shaped data. The recovered Θ
has NO SCIENTIFIC MEANING because:

  * Batch order is arbitrary (shuffled by DataLoader).
  * Adjacent activations come from unrelated prompts.
  * There is no physical "dt".

To do REAL SINDy on transformer activations you need a MULTI-TOKEN harvest
where successive token positions share a trajectory (residual stream evolving
across positions in the same prompt). The existing cogito-L40 harvest is
single-token (one activation per prompt, 949 colors × 28 templates) and
therefore unsuitable for actual governing-equation discovery.
"""

from __future__ import annotations

import torch

from .sindy_sae import SINDySAE


def fake_derivative_from_batch_order(
    z: torch.Tensor, dt: float = 1.0, mode: str = "forward"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z_in, dz_fake) by pretending batch order is a trajectory.

    Parameters
    ----------
    z : (B, state_dim)
    dt : float
        Fake step size; arbitrary, here for unit consistency.
    mode : "forward" | "central"

    Returns
    -------
    z_in : (B', state_dim)        states paired with dz_fake
    dz_fake : (B', state_dim)     fake "time derivatives"
    """
    if z.ndim != 2:
        raise ValueError(f"expected (B, D), got {z.shape}")
    if mode == "forward":
        z_in = z[:-1]
        dz_fake = (z[1:] - z[:-1]) / dt
    elif mode == "central":
        z_in = z[1:-1]
        dz_fake = (z[2:] - z[:-2]) / (2.0 * dt)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return z_in, dz_fake


def smoke_fit(
    sindy: SINDySAE,
    z: torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-2,
    dt: float = 1.0,
    sparsity: float | None = None,
) -> dict:
    """One-shot training using the fake-derivative trick. SMOKE TEST ONLY.

    Returns a dict with final loss components. Result Θ is MEANINGLESS.
    """
    z_in, dz_fake = fake_derivative_from_batch_order(z, dt=dt, mode="forward")
    opt = torch.optim.Adam(sindy.parameters(), lr=lr)
    last: dict = {}
    for step in range(n_steps):
        opt.zero_grad()
        out = sindy.loss(z_in, dz_fake, sparsity=sparsity)
        out["total"].backward()
        opt.step()
        last = {k: float(v.detach().cpu()) for k, v in out.items()}
    last["WARNING"] = (
        "fake-derivative smoke test — Θ has no scientific meaning; "
        "real SINDy needs a multi-token trajectory harvest."
    )
    return last
