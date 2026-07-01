"""Smoke test of SINDy-SAE on cogito-L40 activations.

READ THIS FIRST
===============
The existing cogito-L40 harvest is SINGLE-TOKEN-POSITION (one activation per
prompt: 949 xkcd colors × 28 templates → ~26572 rows, width 7168). There is
NO TIME / TOKEN TRAJECTORY in this data.

SINDy requires a trajectory z(t) and its time derivative dz/dt. This script
uses the `sindy_sae_static.fake_derivative_from_batch_order` adapter, which
pretends consecutive rows in batch order form a trajectory and computes a
finite-difference "derivative". This is a CLEARLY-BROKEN smoke test:

  * Batch order is shuffled / arbitrary.
  * Adjacent rows are unrelated prompts.
  * Recovered Θ has NO SCIENTIFIC MEANING.

PURPOSE: confirm the pipeline (library evaluation, sparse Θ regression,
L1 penalty, STLSQ thresholding) runs end-to-end at cogito scale and
shape. NOTHING ELSE.

To do real SINDy on cogito activations a MULTI-TOKEN re-harvest is
required, capturing the residual stream across token positions inside the
same prompt. That harvest is blocked by the current no-cluster-access rule
and is filed as future work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from manifold_sae.sindy_sae import SINDySAE
from manifold_sae.sindy_sae_static import smoke_fit


def _load_cogito(path: Path, n_rows: int, d_pca: int) -> torch.Tensor:
    """Load cogito-L40 harvest, project to small PCA basis for tractability.

    Falls back to random Gaussian if the file is missing (so the smoke test
    still runs on a laptop). PCA is essential: state_dim=7168 with a product
    library has P ≈ 25M which is absurd.
    """
    if not path.exists():
        print(f"[smoke] no harvest at {path}; using synthetic Gaussian stand-in")
        rng = np.random.default_rng(0)
        return torch.from_numpy(rng.standard_normal((n_rows, d_pca)).astype(np.float32))
    X = np.load(path, mmap_mode="r")
    if X.shape[0] > n_rows:
        idx = np.random.default_rng(0).choice(X.shape[0], size=n_rows, replace=False)
        X = np.asarray(X[idx])
    X = X.astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    # cheap top-K PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xp = U[:, :d_pca] * S[:d_pca]
    return torch.from_numpy(Xp.astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", type=Path, default=Path("data/cogito_L40.npy"))
    p.add_argument("--n_rows", type=int, default=4000)
    p.add_argument("--d_pca", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=300)
    args = p.parse_args()

    print(__doc__)
    print("=" * 72)

    z = _load_cogito(args.harvest, args.n_rows, args.d_pca)
    sindy = SINDySAE(
        state_dim=args.d_pca,
        library_terms=("identity", "square", "product"),
        sparsity=0.01,
    )
    result = smoke_fit(sindy, z, n_steps=args.n_steps, lr=1e-2)
    print("smoke fit done:", result)
    Theta = sindy.effective_Theta().detach().cpu().numpy()
    sparsity_frac = float((np.abs(Theta) > 1e-3).mean())
    print(f"Θ shape: {Theta.shape}   |Θ|>1e-3 fraction: {sparsity_frac:.3f}")
    print(
        "REMINDER: Θ above is MEANINGLESS. Real SINDy on cogito needs a "
        "multi-token trajectory harvest."
    )


if __name__ == "__main__":
    main()
