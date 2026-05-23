"""auto_exp_16: gauge-free local geometry via score-Jacobian on cogito L40.

GOAL
----
Localize the color manifold in ambient PC-space without choosing a chart.
We fit a denoising score model on the 886 per-color centroids in Z_top16,
then at each centroid eigendecompose J = ds/dz to split tangent
(low-magnitude eigvals) from normal (high-magnitude) directions.

PRIOR WORK
----------
- auto_exp_07: local-PCA d_intrinsic ~ 5-10 (kNN).
- auto_54:     local PC1 tangent of T-space, mean signed cos = +0.52 with
               clockwise hue at 12/12 anchors.
- auto_55:     Monge-patch curvature ranks yellow/purple/pink/teal bendiest.
- auto_67:     gauge-free supervised CV ceiling R^2 = 0.321.

METHOD
------
1. Centroids = mean over TOP_TEMPLATES of mmap'd X_L40.npy, filtered.
2. Z = project to top-K_PC=16 canonical PCs.
3. Train a small MLP s_theta(z) ~= grad log p(z) via denoising score
   matching with annealed sigma ~ {0.05,0.1,0.2} * scale(Z).
4. Compute Jacobian of s_theta at each centroid via autograd.
   The negative-semidefinite eigvals give tangent/normal split:
   - tangent  = directions of small |lambda(J_sym)|  (manifold stays flat
                under the gradient flow).
   - normal   = directions of large |lambda(J_sym)|  (score field pulls
                strongly back to the manifold).
5. Validate top-1 tangent at 12 canonical hue anchors against the
   cw-neighbour direction in Z-space (mirrors auto_54).

NO Gaussian RBF, NO Duchon length_scale, NO B-splines, mmap harvest.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_exp_16.png"
OUT_JSON = OUT_DIR / "auto_exp_16.json"

K_PC = 16
HIDDEN = 128
N_LAYERS = 2
N_ITERS = 4000
BATCH = 256
SIGMA_LEVELS = [0.20, 0.10, 0.05]   # annealed (typical-norm fractions)
LR = 1e-3
SEED = 0

ANCHORS = [
    "red", "orange", "yellow", "lime green", "green",
    "teal", "cyan", "blue", "purple", "magenta",
    "pink", "brown",
]


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < eps else v / n


def _build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    n_colors_raw = n_total // N_TEMPLATES
    centroids = np.zeros((n_colors_raw, H), dtype=np.float64)
    for ci in range(n_colors_raw):
        base = ci * N_TEMPLATES
        rows = [base + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(axis=0)
    print(f"[cent] centroids={centroids.shape}", flush=True)

    colors_all = load_xkcd_colors()[:n_colors_raw]
    kept, kept_idx = filter_colors(colors_all)
    kept_idx = np.array(kept_idx, dtype=np.int64)
    centroids = centroids[kept_idx]
    names = [c[0] for c in kept]
    rgb = np.array([[c[1], c[2], c[3]] for c in kept], dtype=np.float64) / 255.0
    print(f"[filt] kept {centroids.shape[0]} colors", flush=True)
    return centroids, names, rgb


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    centroids, names, rgb = _build_centroids()

    basis = load_pc_basis(K=max(K_PC, 16))
    Z_full = project(centroids, basis)            # (N, >=K_PC)
    Z = Z_full[:, :K_PC].astype(np.float64)
    N, D = Z.shape
    print(f"[pca ] Z={Z.shape}, EVR_sum={basis['evr'][:K_PC].sum():.3f}",
          flush=True)

    # Centre+scale Z so sigmas are interpretable in fractions of typical norm.
    z_mean = Z.mean(axis=0, keepdims=True)
    z_std  = Z.std(axis=0, keepdims=True).clip(min=1e-6)
    Zn = (Z - z_mean) / z_std                                 # whitened-ish
    typ_norm = float(np.median(np.linalg.norm(Zn, axis=1)))
    print(f"[norm] median ||z_n||={typ_norm:.3f}", flush=True)

    # --- Torch denoising score matching ---
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    Zt = torch.from_numpy(Zn).float().to(device)

    class ScoreMLP(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
                nn.Linear(h, d),
            )

        def forward(self, z):
            # Parameterise s(z) = MLP(z) - z so that with no training the
            # score points back to the origin (a good prior for centred data).
            return self.net(z) - z

    model = ScoreMLP(D, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    sigmas = [s * typ_norm for s in SIGMA_LEVELS]
    print(f"[train] sigma levels (whitened units): "
          + ", ".join(f"{s:.3f}" for s in sigmas), flush=True)

    loss_curve = []
    for it in range(N_ITERS):
        idx = torch.randint(0, N, (BATCH,), device=device)
        z0 = Zt[idx]
        # pick sigma per-sample uniformly across levels
        sig_idx = torch.randint(0, len(sigmas), (BATCH, 1), device=device)
        sig = torch.tensor(sigmas, device=device).float()[sig_idx[:, 0]].unsqueeze(1)
        eps = torch.randn_like(z0)
        z_noisy = z0 + sig * eps
        # target score for q(z_noisy | z0) Gaussian: -(z_noisy - z0)/sigma^2 = -eps/sigma
        target = -eps / sig
        s_pred = model(z_noisy)
        # weight by sigma^2 (denoising score matching standard weighting)
        loss = ((sig ** 2) * (s_pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0 or it == N_ITERS - 1:
            loss_curve.append((it, float(loss.item())))
            print(f"[train] it={it:5d}  loss={float(loss.item()):.4f}", flush=True)

    final_loss = loss_curve[-1][1]

    # --- Per-centroid Jacobians ---
    print(f"[jac ] computing Jacobians at {N} centroids ...", flush=True)
    model.eval()

    def score_np(z_np: np.ndarray) -> np.ndarray:
        zt = torch.from_numpy(z_np.astype(np.float32)).to(device)
        with torch.no_grad():
            return model(zt).cpu().numpy().astype(np.float64)

    s_at_cent = score_np(Zn)                                   # (N, D)
    s_mags = np.linalg.norm(s_at_cent, axis=1)
    # baseline: score magnitude at random ambient points (same whitened scale)
    rand_pts = rng.normal(size=(N, D)) * typ_norm
    s_at_rand = score_np(rand_pts)
    s_rand_mags = np.linalg.norm(s_at_rand, axis=1)
    print(f"[jac ] median ||s(centroid)||={np.median(s_mags):.3f}, "
          f"random={np.median(s_rand_mags):.3f}", flush=True)

    # Jacobian via autograd, one sample at a time.
    Js = np.zeros((N, D, D), dtype=np.float64)
    for i in range(N):
        zi = torch.from_numpy(Zn[i]).float().to(device).requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda z: model(z.unsqueeze(0)).squeeze(0), zi, vectorize=True,
        )
        Js[i] = J.detach().cpu().numpy().astype(np.float64)
        if i % 200 == 0:
            print(f"[jac ]   {i}/{N}", flush=True)

    # Symmetrise; the score's symmetric part is the Hessian of log p.
    Jsym = 0.5 * (Js + Js.transpose(0, 2, 1))

    # Eigendecomp per centroid. Use -Jsym so that *positive* eigvals mean
    # "score pulls back" (i.e. log p concave / normal direction). Tangent
    # directions are near-zero eigvals.
    eigvals = np.zeros((N, D), dtype=np.float64)
    eigvecs = np.zeros((N, D, D), dtype=np.float64)
    for i in range(N):
        w, v = np.linalg.eigh(-Jsym[i])
        # eigh returns ascending; flip to descending
        order = np.argsort(w)[::-1]
        eigvals[i] = w[order]
        eigvecs[i] = v[:, order]
    print(f"[jac ] eigvals: median max={np.median(eigvals[:,0]):.3f}, "
          f"median min={np.median(eigvals[:,-1]):.3f}", flush=True)

    # Tangent dim per centroid: count eigvals below 10% of that centroid's max.
    # (Tangent eigvals are *small in magnitude*; we use |eigval| < 0.1 max(|eigval|).)
    abs_e = np.abs(eigvals)
    thr_each = 0.10 * abs_e.max(axis=1, keepdims=True)
    d_tan = (abs_e < thr_each).sum(axis=1)
    med_dtan = int(np.median(d_tan))
    print(f"[geom] d_tan: median={med_dtan}, mean={d_tan.mean():.2f}, "
          f"p25={np.percentile(d_tan,25):.0f}, p75={np.percentile(d_tan,75):.0f}",
          flush=True)

    # --- Anchor validation: tangent at hue anchors vs cw-neighbour in Z ---
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    anchors = [a for a in ANCHORS if a in name_to_idx]
    print(f"[anch] {len(anchors)}/{len(ANCHORS)} anchors present: {anchors}",
          flush=True)

    anchor_results = []
    cos_signed_list = []
    for pos, a in enumerate(anchors):
        ai = name_to_idx[a]
        cw  = anchors[(pos + 1) % len(anchors)]
        ccw = anchors[(pos - 1) % len(anchors)]
        ci = name_to_idx[cw]
        ki = name_to_idx[ccw]
        # tangent_top1 = eigvec with SMALLEST |eigval|.
        order_small = np.argsort(np.abs(eigvals[ai]))
        t1 = eigvecs[ai][:, order_small[0]]
        t2 = eigvecs[ai][:, order_small[1]]
        d_cw  = _unit(Zn[ci] - Zn[ai])
        d_ccw = _unit(Zn[ki] - Zn[ai])
        cos_t1_cw = float(np.dot(t1, d_cw))
        # sign flip so positive == aligned with cw
        if cos_t1_cw < 0:
            t1 = -t1
            cos_t1_cw = -cos_t1_cw
        cos_t1_ccw = float(np.dot(t1, d_ccw))
        cos_t2_cw = float(np.dot(t2, d_cw))
        best_abs = max(abs(cos_t1_cw), abs(cos_t2_cw))
        cos_signed_list.append(cos_t1_cw)
        anchor_results.append({
            "anchor": a, "cw": cw, "ccw": ccw,
            "cos_t1_cw_signed": cos_t1_cw,
            "cos_t1_ccw": cos_t1_ccw,
            "cos_t2_cw_abs": abs(cos_t2_cw),
            "best_abs_cos_cw_t12": best_abs,
            "d_tan": int(d_tan[ai]),
            "score_mag": float(s_mags[ai]),
        })
    mean_cos_signed = float(np.mean(cos_signed_list))
    n_positive = int(np.sum(np.array(cos_signed_list) > 0))
    print(f"[anch] mean signed cos(t1, d_cw) = {mean_cos_signed:.3f}  "
          f"({n_positive}/{len(anchors)} positive after sign flip)", flush=True)

    summary = {
        "config": {
            "K_PC": K_PC,
            "denoiser": "MLP score-matching (2x128 GELU, s(z)=MLP(z)-z)",
            "n_iters": N_ITERS,
            "batch": BATCH,
            "lr": LR,
            "sigma_levels_fracs": SIGMA_LEVELS,
            "sigma_levels_whitened": sigmas,
            "n_colors": int(N),
        },
        "train": {
            "final_loss": final_loss,
            "loss_curve": loss_curve,
        },
        "score_field": {
            "median_score_mag_at_centroids": float(np.median(s_mags)),
            "median_score_mag_at_random": float(np.median(s_rand_mags)),
            "ratio_random_over_centroid": float(
                np.median(s_rand_mags) / max(np.median(s_mags), 1e-12)),
        },
        "geometry": {
            "median_d_tan": med_dtan,
            "mean_d_tan": float(d_tan.mean()),
            "p25_d_tan": float(np.percentile(d_tan, 25)),
            "p75_d_tan": float(np.percentile(d_tan, 75)),
            "comparison_auto_exp_07": "auto_exp_07 local-PCA d_intrinsic ~ 5-10",
        },
        "anchors": {
            "list": anchors,
            "per_anchor": anchor_results,
            "mean_signed_cos_t1_cw": mean_cos_signed,
            "n_positive": n_positive,
            "n_total": len(anchors),
            "comparison_auto_54": "auto_54 mean signed cos = +0.52",
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # --- Plot ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) eigenvalue spectrum overlay
    ax = axes[0, 0]
    sort_abs = np.sort(np.abs(eigvals), axis=1)[:, ::-1]   # (N, D) desc
    # Colour each curve by its xkcd RGB
    for i in range(N):
        ax.plot(np.arange(1, D + 1), sort_abs[i],
                color=rgb[i], lw=0.4, alpha=0.35)
    med_curve = np.median(sort_abs, axis=0)
    ax.plot(np.arange(1, D + 1), med_curve, "k-", lw=2.0, label="median")
    ax.axhline(0.10 * med_curve[0], color="red", ls="--", lw=1,
               label="10% threshold (median)")
    ax.set_xlabel("rank")
    ax.set_ylabel("|eigval| of -J_sym (descending)")
    ax.set_title(f"Score-Jacobian spectra at {N} centroids (K_PC={K_PC})")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (2) histogram d_tan
    ax = axes[0, 1]
    bins = np.arange(0, D + 2) - 0.5
    ax.hist(d_tan, bins=bins, color="steelblue", edgecolor="k", alpha=0.85)
    ax.axvline(med_dtan, color="firebrick", lw=2,
               label=f"median d_tan = {med_dtan}")
    ax.axvspan(5, 10, color="orange", alpha=0.18,
               label="auto_exp_07 local-PCA range [5,10]")
    ax.set_xlabel("tangent dimension (|eigval| < 10% of max)")
    ax.set_ylabel("# colours")
    ax.set_title("Per-colour tangent dim vs auto_exp_07 local-PCA")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (3) Anchors: tangent vs cw-neighbour in PC1-PC2 plane
    ax = axes[1, 0]
    # background centroids
    ax.scatter(Zn[:, 0], Zn[:, 1], c=rgb, s=8, alpha=0.35,
               edgecolor="none")
    for r in anchor_results:
        a = r["anchor"]
        ai = name_to_idx[a]
        ci = name_to_idx[r["cw"]]
        # recompute the (sign-flipped) tangent for plot
        order_small = np.argsort(np.abs(eigvals[ai]))
        t1 = eigvecs[ai][:, order_small[0]]
        if np.dot(t1, _unit(Zn[ci] - Zn[ai])) < 0:
            t1 = -t1
        # length: half median nn distance for visibility
        scale = 0.5 * np.median(np.linalg.norm(
            Zn - Zn[ai], axis=1)[np.argsort(np.linalg.norm(
                Zn - Zn[ai], axis=1))[1:6]])
        # tangent arrow (PC1-PC2 projection)
        ax.annotate("", xy=(Zn[ai, 0] + scale * t1[0],
                            Zn[ai, 1] + scale * t1[1]),
                    xytext=(Zn[ai, 0], Zn[ai, 1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
        # cw target arrow (dashed grey)
        ax.annotate("", xy=(Zn[ci, 0], Zn[ci, 1]),
                    xytext=(Zn[ai, 0], Zn[ai, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    lw=0.8, alpha=0.6, linestyle="dashed"))
        ax.text(Zn[ai, 0], Zn[ai, 1], f" {a}", fontsize=7,
                color="black", weight="bold")
    ax.set_xlabel("PC1 (whitened)")
    ax.set_ylabel("PC2 (whitened)")
    ax.set_title(f"Top-1 tangent at hue anchors vs cw neighbour\n"
                 f"mean signed cos = {mean_cos_signed:+.3f}  "
                 f"({n_positive}/{len(anchors)} positive)  "
                 f"[auto_54: +0.52]")
    ax.grid(alpha=0.25)

    # (4) score-magnitude histograms
    ax = axes[1, 1]
    bins = np.linspace(0, max(s_mags.max(), s_rand_mags.max()), 50)
    ax.hist(s_mags, bins=bins, color="steelblue", alpha=0.75,
            label=f"||s(centroid)||  med={np.median(s_mags):.2f}",
            edgecolor="k", lw=0.3)
    ax.hist(s_rand_mags, bins=bins, color="firebrick", alpha=0.55,
            label=f"||s(random)||  med={np.median(s_rand_mags):.2f}",
            edgecolor="k", lw=0.3)
    ax.set_xlabel("||score(z)||")
    ax.set_ylabel("count")
    ax.set_title("Score magnitude: centroids near manifold ~ low; random ~ high")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(
        f"auto_exp_16: score-Jacobian local geometry of cogito L40 color "
        f"manifold (N={N}, K_PC={K_PC})\n"
        f"median d_tan={med_dtan} [auto_exp_07: 5-10]   "
        f"mean signed cos(t1,cw)={mean_cos_signed:+.3f} [auto_54: +0.52]   "
        f"||s||_med rand/cent={np.median(s_rand_mags)/max(np.median(s_mags),1e-12):.1f}x",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    print(f"[time] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
