"""auto_exp_60: Mechanism-sparsity-only (NO HSV supervision) on cogito-L40.

The falsifiable test of the Lachapelle 2401.04890 mechanism-sparsity-
identifiability theorem applied to the cogito color manifold:

  If the mechanism-sparsity penalty on the decoder Jacobian column-2-norms
  is sufficient to gauge-fix the latent up to a coordinate permutation
  (under sparse mechanisms in the true generative process), then dropping
  the iVAE supervision entirely should still surface HSV-aligned axes
  somewhere in the 6-axis fit.

We compare two regimes:
  - n_supervised = 0, n_free = 6 — pure mechanism-sparsity
  - n_supervised = 0, n_free = 6, weight_mech = 0 — pure PCA reference

For each, we evaluate per-axis max correlation against HSV and against
name-features, and report whether ANY axis hits |corr| >= 0.45 with HSV
hue (the falsifiability threshold).

Even a NEGATIVE result is publishable: it would mean cogito-L40 violates
the mechanism-sparsity sufficient condition, evidence that the
auxiliary-conditional iVAE-style identifiability primitive is strictly
necessary on this dataset (consistent with auto_exp_38's finding that 5
prior unsupervised attempts all failed without a gauge-fix companion).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/user/Manifold-SAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402
from gamfit.torch import MechanismSparsityPenalty  # noqa: E402

from _pca_basis import load_pc_basis  # type: ignore  # noqa: E402
from manifold_sae.identifiable import abs_corr  # noqa: E402

RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
RUN_DIR.mkdir(parents=True, exist_ok=True)
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_PNG = RUN_DIR / "auto_exp_60.png"
OUT_JSON = RUN_DIR / "auto_exp_60.json"

N_TEMPLATES = 28
K_PCS = 16
N_AXES = 6


def load_xkcd_rgb(n_colors: int):
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append(
                (
                    int(hexs[0:2], 16) / 255.0,
                    int(hexs[2:4], 16) / 255.0,
                    int(hexs[4:6], 16) / 255.0,
                )
            )
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def per_color_stats(X, n_t, basis, k_pcs):
    n_rows, _ = X.shape
    n_c = n_rows // n_t
    mu = basis["mu"]; sigma = basis["sigma"]; Vt = basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t; e = ce * n_t
        chunk = np.asarray(X[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = (chunk @ Vt.T)[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names, tsig):
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


class _MechFit:
    """Lightweight result holder mirroring IdentifiableFit (W, T)."""

    def __init__(self, W, T, final_loss):
        self.W = W
        self.T = T
        self.final_loss = final_loss


def fit_one(T0, weight_mech, n_iter, seed):
    """Pure mechanism-sparsity-only fit (no aux supervision).

    gamfit.identifiable_factor_fit requires n_supervised >= 1, so the
    unsupervised (n_supervised == 0) regime this experiment studies is no
    longer expressible through the library primitive. We therefore drive the
    public gamfit.torch.MechanismSparsityPenalty building block directly: an
    SVD-warm-started torch-autograd fit of X ≈ T @ W.T with a group-lasso
    column penalty on the decoder W (the Lachapelle 2401.04890 functional) and
    a light Gaussian prior on T.
    """
    torch.manual_seed(int(seed))
    Xnp = np.ascontiguousarray(T0, dtype=np.float64)
    n, D = Xnp.shape
    Xc_np = Xnp - Xnp.mean(0, keepdims=True)

    U, S, Vt = np.linalg.svd(Xc_np, full_matrices=False)
    k = min(N_AXES, S.shape[0])
    W0 = np.zeros((D, N_AXES), dtype=np.float64)
    W0[:, :k] = Vt[:k].T * S[:k][None, :] / max(np.sqrt(n - 1), 1.0)
    T0init = Xc_np @ np.linalg.pinv(W0).T

    Xc = torch.as_tensor(Xc_np, dtype=torch.float64)
    T = torch.nn.Parameter(torch.as_tensor(T0init, dtype=torch.float64))
    W = torch.nn.Parameter(torch.as_tensor(W0, dtype=torch.float64))

    mech = MechanismSparsityPenalty(
        feature_groups=[[d] for d in range(D)],
        weight=float(weight_mech),
        n_eff=float(N_AXES),
        smoothing_eps=1.0e-6,
    )
    opt = torch.optim.Adam([T, W], lr=5.0e-2)

    final_loss = {}
    for _ in range(int(n_iter)):
        opt.zero_grad(set_to_none=True)
        recon = 0.5 * ((Xc - T @ W.t()) ** 2).sum()
        free_prior = 0.5 * 1.0e-2 * (T ** 2).sum()
        mech_loss = mech.forward(W.t().contiguous())
        total = recon + free_prior + mech_loss
        total.backward()
        opt.step()
        final_loss = {
            "total": float(total.detach()),
            "recon": float(recon.detach()),
            "mech": float(mech_loss.detach()),
        }

    return _MechFit(
        W=W.detach().cpu().numpy().astype(np.float64),
        T=T.detach().cpu().numpy().astype(np.float64),
        final_loss=final_loss,
    )


def main():
    t0 = time.time()
    print("[auto_exp_60] Mechanism-sparsity-ONLY on cogito-L40 (no HSV supervision)")

    X = np.load(X_PATH, mmap_mode="r")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)
    print(f"[data] T0={T0.shape} hsv={hsv.shape} namef={namef.shape}")

    configs = [
        ("pca_only", 1.0e-12, 60),
        ("mech_weak", 1.0e-3, 200),
        ("mech_mid", 1.0e-2, 200),
        ("mech_strong", 5.0e-2, 200),
    ]

    results = {}
    for name, w_mech, n_iter in configs:
        print(f"[fit] {name}: weight_mech={w_mech} n_iter={n_iter}")
        fit = fit_one(T0, w_mech, n_iter, seed=60)
        corr_hsv = abs_corr(fit.T, hsv)
        corr_name = abs_corr(fit.T, namef)
        # Best axis-target alignment over all 6 axes
        best_hsv = corr_hsv.max(axis=0)  # (3,) — max over axes per HSV component
        best_name = corr_name.max(axis=0)
        results[name] = {
            "weight_mech": w_mech,
            "n_iter": n_iter,
            "final_loss": fit.final_loss,
            "corr_hsv": corr_hsv.tolist(),
            "corr_name": corr_name.tolist(),
            "best_axis_per_hsv_component": best_hsv.tolist(),
            "best_axis_per_name_component": best_name.tolist(),
        }
        print(
            f"  best per-HSV-component axis-corr: "
            f"hue={best_hsv[0]:.2f}  sat={best_hsv[1]:.2f}  val={best_hsv[2]:.2f}"
        )
        print(
            f"  best per-name-component axis-corr: "
            f"mono={best_name[0]:.2f}  modc={best_name[1]:.2f}  tsig={best_name[2]:.2f}"
        )

    # ---- Falsifiability check
    falsification_threshold = 0.45
    mech_best_hue = max(
        results[k]["best_axis_per_hsv_component"][0]
        for k in ("mech_weak", "mech_mid", "mech_strong")
    )
    pca_best_hue = results["pca_only"]["best_axis_per_hsv_component"][0]
    theorem_supported = bool(
        mech_best_hue >= falsification_threshold
        and mech_best_hue >= pca_best_hue + 0.05
    )
    print(
        f"[theorem-test] best mech-sparsity hue-corr={mech_best_hue:.2f}  "
        f"PCA baseline={pca_best_hue:.2f}  "
        f"mech-sparsity-identifiability supported={theorem_supported}"
    )

    # ---- Plot grid: per-config heatmaps
    fig, axs = plt.subplots(len(configs), 2, figsize=(11, 3.0 * len(configs)),
                             constrained_layout=True)
    for ri, (cname, _, _) in enumerate(configs):
        r = results[cname]
        corr_hsv_arr = np.array(r["corr_hsv"])
        corr_name_arr = np.array(r["corr_name"])
        ax = axs[ri, 0]
        im = ax.imshow(corr_hsv_arr, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{cname}: |corr(axis, HSV)|  (w_mech={r['weight_mech']:.1e})")
        ax.set_xticks(range(3)); ax.set_xticklabels(["hue", "sat", "val"])
        ax.set_yticks(range(N_AXES)); ax.set_yticklabels([f"a{i}" for i in range(N_AXES)])
        for j in range(N_AXES):
            for k in range(3):
                ax.text(k, j, f"{corr_hsv_arr[j, k]:.2f}", ha="center", va="center",
                        color="white" if corr_hsv_arr[j, k] < 0.6 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)

        ax = axs[ri, 1]
        im = ax.imshow(corr_name_arr, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(f"{cname}: |corr(axis, name-feat)|")
        ax.set_xticks(range(3)); ax.set_xticklabels(["mono", "modc", "tsig"])
        ax.set_yticks(range(N_AXES)); ax.set_yticklabels([f"a{i}" for i in range(N_AXES)])
        for j in range(N_AXES):
            for k in range(3):
                ax.text(k, j, f"{corr_name_arr[j, k]:.2f}", ha="center", va="center",
                        color="white" if corr_name_arr[j, k] < 0.6 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"auto_exp_60 — mech-sparsity-only identifiability test "
        f"(theorem supported = {theorem_supported})",
        fontsize=13,
    )
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[plot] wrote {OUT_PNG}")

    out = {
        "experiment": "auto_exp_60_mechsparsity_unsupervised",
        "configs": results,
        "falsification_threshold": falsification_threshold,
        "best_mech_hue_corr": mech_best_hue,
        "pca_baseline_hue_corr": pca_best_hue,
        "mech_sparsity_identifiability_supported": theorem_supported,
        "elapsed_s": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] wrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
