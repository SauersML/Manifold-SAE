"""Why do recovered atoms collapse to straight lines? Controlled ablation.

Single planted manifold, single SAE atom (+1 slack), isolate each suspect:
  - monotonicity prior fighting curvature
  - position collapse (encoder positions don't span [0,1])
  - too few steps
  - atom topology (circle needs cyclic manifold)

For each setting we train a custom loop (so we can toggle loss weights), then
report position spread and overlay the recovered curve on ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from manifold_sae.data_synthetic import SyntheticDataset, CURVE_TYPES, chamfer_distance
from manifold_sae.sae import (
    ManifoldSAE, ManifoldSAEConfig, DecoderConfig, SparsityConfig, RemlConfig,
)
from manifold_sae.losses import _position_coverage_loss

NAMES = {n: i for i, (n, *_rest) in enumerate(CURVE_TYPES)}


def train_one(curve, n_steps, mono_w, cov_w, manifold, seed=0, d=24, n_atoms=2):
    torch.manual_seed(seed); np.random.seed(seed)
    ci = NAMES[curve]
    ds = SyntheticDataset(d_ambient=d, n_features=1, n_samples=4000, sparsity=1.0,
                          noise=0.01, seed=seed, orthogonal_subspaces=True, curve_indices=[ci],
                          t_grid_size=128)
    rank = 1 if manifold == "circle" else 2
    cfg = ManifoldSAEConfig(input_dim=d, n_atoms=n_atoms, intrinsic_rank=rank,
                            atom_manifold=manifold, n_basis_per_atom=10,
                            sparsity=SparsityConfig(kind="softmax_topk", target_k=1),
                            decoder=DecoderConfig(ortho_weight=1e-2), reml=RemlConfig())
    sae = ManifoldSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    X = ds.x.to(cfg.dtype)
    bs = 256
    for step in range(n_steps):
        idx = torch.randint(0, X.shape[0], (bs,))
        xb = X[idx]
        opt.zero_grad(set_to_none=True)
        out = sae(xb)
        mse = ((out.x_hat - xb) ** 2).mean()
        cov = _position_coverage_loss(out.positions, out.amplitudes)
        mono = sae.decoder_monotonicity_penalty()
        loss = mse + cov_w * cov + mono_w * mono + 1e-3 * out.z.abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
    # diagnostics
    with torch.no_grad():
        out = sae(X)
        pos = out.positions[..., 0]                       # (N, atoms)
        amp = out.amplitudes                              # (N, atoms)
        # spread of positions on the most-active atom
        active_atom = int(amp.mean(0).argmax())
        p = pos[:, active_atom].cpu().numpy()
        spread = float(p.max() - p.min())
        std = float(p.std())
        cd = sae.extract_feature_curves(grid_size=128)
        learned = torch.stack([cd[k] for k in range(n_atoms)], 0).cpu().numpy()
    gt = ds.ground_truth["curve_points"][0]               # (T, D)
    # best chamfer over atoms
    ch = min(chamfer_distance(gt, learned[k]) for k in range(n_atoms))
    return dict(curve=curve, n_steps=n_steps, mono_w=mono_w, cov_w=cov_w,
                manifold=manifold, pos_spread=spread, pos_std=std,
                final_mse=float(mse), chamfer=ch), (gt, learned, active_atom)


SETTINGS = [
    # baseline (what the experiment used)
    dict(curve="parabola", n_steps=200,  mono_w=1e-2, cov_w=1e-2, manifold="product"),
    # more steps
    dict(curve="parabola", n_steps=2000, mono_w=1e-2, cov_w=1e-2, manifold="product"),
    # monotonicity off
    dict(curve="parabola", n_steps=2000, mono_w=0.0,  cov_w=1e-2, manifold="product"),
    # mono off + strong coverage
    dict(curve="parabola", n_steps=2000, mono_w=0.0,  cov_w=1e-1, manifold="product"),
    # circle with product (wrong topology)
    dict(curve="circle",   n_steps=2000, mono_w=0.0,  cov_w=1e-1, manifold="product"),
    # circle with cyclic topology
    dict(curve="circle",   n_steps=2000, mono_w=0.0,  cov_w=1e-1, manifold="circle"),
]


def main():
    out_dir = Path("runs/diag"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    plots = []
    for s in SETTINGS:
        rec, viz = train_one(**s)
        rows.append(rec); plots.append((rec, viz))
        print(f"{rec['curve']:<9} steps={rec['n_steps']:<5} mono={rec['mono_w']:<5} "
              f"cov={rec['cov_w']:<5} mani={rec['manifold']:<8} | "
              f"pos_spread={rec['pos_spread']:.3f} pos_std={rec['pos_std']:.3f} "
              f"mse={rec['final_mse']:.4f} chamfer={rec['chamfer']:.3f}")

    # plot grid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(plots); cols = 3; rows_p = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_p, cols, figsize=(4 * cols, 4 * rows_p), squeeze=False)
    for i, (rec, (gt, learned, ak)) in enumerate(plots):
        ax = axes[i // cols, i % cols]
        lp = learned[ak]
        def proj2(a):
            c = a - a.mean(0, keepdims=True)
            c = c / max(np.linalg.norm(c), 1e-12)
            return c
        g = proj2(gt); l = proj2(lp)
        joint = np.concatenate([g, l], 0)
        _, _, vt = np.linalg.svd(joint, full_matrices=False)
        pcs = vt[:2]
        g2, l2 = g @ pcs.T, l @ pcs.T
        M = l2.T @ g2; U, _, Vt = np.linalg.svd(M); l2 = l2 @ (U @ Vt)
        ax.plot(g2[:, 0], g2[:, 1], "o-", c="C0", ms=2, label="GT")
        ax.plot(l2[:, 0], l2[:, 1], "x-", c="C1", ms=3, label="learned")
        ax.set_title(f"{rec['curve']} s={rec['n_steps']} mono={rec['mono_w']}\n"
                     f"mani={rec['manifold']} cov={rec['cov_w']}\n"
                     f"spread={rec['pos_spread']:.2f} ch={rec['chamfer']:.2f}", fontsize=8)
        ax.legend(fontsize=7); ax.set_aspect("equal", adjustable="datalim")
    for j in range(n, rows_p * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    p = out_dir / "diag_straightline.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\n[plot] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
