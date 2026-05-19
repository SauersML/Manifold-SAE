"""Visualize gamfit's recovery when fed GROUND-TRUTH positions and amplitudes.

Skips the SAE encoder entirely. If gamfit had a bug, these plots would still
show distorted curves. They don't.
"""

from __future__ import annotations

import numpy as np
import gamfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_sae.data_synthetic import SyntheticDataset, chamfer_distance


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
    lam = np.asarray(out["lambda"])

    t_grid = np.linspace(0, 1, 256)
    phi = np.asarray(gamfit.duchon_basis_1d(t_grid, centers, m=2, periodic=False))
    curves_learned = np.einsum("tk,fkd->ftd", phi, coef)  # (F, T, D)
    curves_truth = np.stack([feat.evaluate(t_grid) for feat in ds.features], axis=0)

    # Layout: 2 rows x 3 cols, helix gets a 3D axes since its intrinsic curve is 3D.
    fig = plt.figure(figsize=(15, 9))
    chamfers = []
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]  # (row, col) per feature
    gs = fig.add_gridspec(2, 3)

    for k, feat in enumerate(ds.features):
        r, c = layout[k]
        proj = feat.projection.T  # (d_ambient, d_intrinsic)
        d_intrinsic = proj.shape[1]
        gt_intrinsic = curves_truth[k] @ proj  # (T, d_intrinsic)
        lp_intrinsic = curves_learned[k] @ proj
        cd = chamfer_distance(curves_truth[k], curves_learned[k])
        chamfers.append(cd)

        is_helix_3d = (feat.name == "helix" and d_intrinsic >= 3)
        if is_helix_3d:
            ax = fig.add_subplot(gs[r, c], projection="3d")
            ax.plot(gt_intrinsic[:, 0], gt_intrinsic[:, 1], gt_intrinsic[:, 2],
                    "o-", color="C0", markersize=2, label="ground truth")
            ax.plot(lp_intrinsic[:, 0], lp_intrinsic[:, 1], lp_intrinsic[:, 2],
                    "x-", color="C1", markersize=3, label="gamfit (gt inputs)")
            # Mark t=0 and t=1 to show whether the curve closes.
            ax.scatter([gt_intrinsic[0, 0], gt_intrinsic[-1, 0]],
                       [gt_intrinsic[0, 1], gt_intrinsic[-1, 1]],
                       [gt_intrinsic[0, 2], gt_intrinsic[-1, 2]],
                       color="C2", s=40, marker="s", label="t=0, t=1 (truth)")
            ax.scatter([lp_intrinsic[0, 0], lp_intrinsic[-1, 0]],
                       [lp_intrinsic[0, 1], lp_intrinsic[-1, 1]],
                       [lp_intrinsic[0, 2], lp_intrinsic[-1, 2]],
                       color="C3", s=40, marker="^", label="t=0, t=1 (fit)")
        else:
            ax = fig.add_subplot(gs[r, c])
            view = proj[:, :2] if d_intrinsic >= 2 else proj
            gt2 = curves_truth[k] @ view
            lp2 = curves_learned[k] @ view
            ax.plot(gt2[:, 0], gt2[:, 1], "o-", color="C0", markersize=2, label="ground truth")
            ax.plot(lp2[:, 0], lp2[:, 1], "x-", color="C1", markersize=3, label="gamfit (gt inputs)")
            # Annotate t=0 and t=1 endpoints — the seam on the circle lives here.
            ax.scatter([gt2[0, 0], gt2[-1, 0]], [gt2[0, 1], gt2[-1, 1]],
                       color="C2", s=60, marker="s", zorder=5, label="t=0, t=1 (truth)")
            ax.scatter([lp2[0, 0], lp2[-1, 0]], [lp2[0, 1], lp2[-1, 1]],
                       color="C3", s=60, marker="^", zorder=5, label="t=0, t=1 (fit)")
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{feat.name}  chamfer={cd:.3f}  edf={edf[k]:.1f}  λ={lam[k]:.1e}")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Gamfit recovery on planted curves, given GROUND-TRUTH (t, by). "
        f"Mean chamfer = {np.mean(chamfers):.3f} (threshold 0.3).\n"
        f"The circle's t=0/t=1 endpoint gap is the open-Duchon-basis seam — "
        f"v1 spec accepts this; periodic Duchon lacks REML support upstream.",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = "runs/gamfit_ground_truth_fit.png"
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")
    print(f"mean chamfer: {np.mean(chamfers):.4f}")


if __name__ == "__main__":
    main()
