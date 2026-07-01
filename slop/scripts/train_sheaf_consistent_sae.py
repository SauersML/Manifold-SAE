"""Train 3 paired SAEs (F=256) with a sheaf-consistency auxiliary loss.

We deliberately use a *tiny self-contained* TopK SAE per layer (rather than
the in-flight ManifoldSAE/Crosscoder) to keep the sheaf primitive isolated
and to avoid touching other agents' WIP. The shared ‖δz‖² loss is supplied
by ``manifold_sae.sheaf.SheafConsistencyHead``.

Outputs (under ``runs/SHEAF_PAIRED_3LAYER/train/``):
  - saes.pt          : torch checkpoint
  - history.npz      : losses + harmonic-atom count over training
  - harmonic_curve.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae.sheaf import (
    CellularSheaf,
    SheafConsistencyHead,
    harmonic_atoms,
    sheaf_consistency_loss,
)


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "runs/SHEAF_PAIRED_3LAYER"
OUT = DATA / "train"
F = 256
K = 32
EPOCHS = 30
BATCH = 512
LR = 1e-3
SHEAF_WEIGHT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TinyTopKSAE(nn.Module):
    """Minimal TopK SAE — independent per layer."""

    def __init__(self, D: int, F: int, k: int) -> None:
        super().__init__()
        self.D, self.F, self.k = int(D), int(F), int(k)
        self.W_enc = nn.Parameter(torch.randn(D, F) / D**0.5)
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.W_dec = nn.Parameter(torch.randn(F, D) / F**0.5)
        self.b_dec = nn.Parameter(torch.zeros(D))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = torch.topk(z_pre, k=self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, idx, torch.relu(vals))
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = z @ self.W_dec + self.b_dec
        return z, recon


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    layer_files = sorted(DATA.glob("X_layer*.npy"))
    if not layer_files:
        raise FileNotFoundError(
            f"No layer arrays in {DATA}; run scripts/synthesize_paired_layers.py first"
        )
    Xs = [torch.from_numpy(np.load(p)).float() for p in layer_files]
    dims = [int(X.shape[1]) for X in Xs]
    N = Xs[0].shape[0]
    print(f"[train_sheaf] N={N}, dims={dims}, F={F}, k={K}")

    saes = nn.ModuleList([TinyTopKSAE(D, F, K) for D in dims]).to(DEVICE)
    head = SheafConsistencyHead(n_layers=len(dims), F=F).to(DEVICE)

    params = list(saes.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    Xs_dev = [X.to(DEVICE) for X in Xs]

    history = {"recon": [], "sheaf": [], "harmonic_count": []}
    g = torch.Generator(device="cpu").manual_seed(0)
    steps_per_epoch = max(1, N // BATCH)

    for epoch in range(EPOCHS):
        perm = torch.randperm(N, generator=g)
        ep_recon = 0.0
        ep_sheaf = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH : (s + 1) * BATCH].to(DEVICE)
            x_layers = [X[idx] for X in Xs_dev]
            codes = []
            recon_loss = x_layers[0].new_zeros(())
            for sae, x in zip(saes, x_layers, strict=True):
                z, recon = sae(x)
                codes.append(z)
                recon_loss = recon_loss + ((recon - x) ** 2).mean()
            sheaf_e = head.energy(codes)
            loss = recon_loss + SHEAF_WEIGHT * sheaf_e
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_recon += float(recon_loss.detach().cpu())
            ep_sheaf += float(sheaf_e.detach().cpu())

        # ----- harmonic-atom count via the sheaf object -----
        with torch.no_grad():
            sheaf = CellularSheaf(
                layer_saes=list(saes),
                restriction_maps=head.restriction_dict(),
            )
            # Tol = 5% of mean eigenvalue: count "near-harmonic" modes
            # (eigval ≪ bulk). Sheaf energy decreasing should grow this set.
            L = sheaf.laplacian()
            eigvals = np.linalg.eigvalsh(L)
            mean_ev = float(eigvals.mean())
            tol = max(1e-9, 0.05 * mean_ev)
            modes, atoms = harmonic_atoms(sheaf, tol=tol)
            # kernel dim (# of harmonic modes) is the headline metric:
            # # of independent globally-consistent feature directions.
            n_harm = int(modes.shape[0])

        history["recon"].append(ep_recon / steps_per_epoch)
        history["sheaf"].append(ep_sheaf / steps_per_epoch)
        history["harmonic_count"].append(n_harm)
        print(
            f"[epoch {epoch:02d}] recon={history['recon'][-1]:.4f} "
            f"sheaf={history['sheaf'][-1]:.4f} harmonic={n_harm}"
        )

    torch.save({"saes": saes.state_dict(), "head": head.state_dict(), "dims": dims}, OUT / "saes.pt")
    np.savez(OUT / "history.npz", **{k: np.array(v) for k, v in history.items()})

    # ----- plot -----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history["sheaf"], label="‖δz‖²")
        ax1.plot(history["recon"], label="recon MSE")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_yscale("log")
        ax1.set_title("Sheaf energy + recon")
        ax2.plot(history["harmonic_count"], marker="o", color="C2")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("# harmonic atoms")
        ax2.set_title("Harmonic atom count (ker L_sheaf)")
        fig.tight_layout()
        fig.savefig(OUT / "harmonic_curve.png", dpi=130)
        print(f"[train_sheaf] wrote {OUT/'harmonic_curve.png'}")
    except Exception as e:
        print(f"[train_sheaf] plot skipped: {e}")


if __name__ == "__main__":
    main()
