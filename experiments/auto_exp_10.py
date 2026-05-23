"""auto_exp_10: per-color Lipschitz of the fitted U_3d.

Motivation
----------
auto_exp_06/07/09 established that U_3d is a stable, low-distortion 3D
latent for the cogito-L40 color manifold. But "stable on average" hides
local structure: are there *anchors* where the fitted decoder
Z(T) = Phi(T) @ B has a near-singular fold — i.e. where a small move in
latent T explodes into a large move in residual space? Those colors
would be exactly the ones near sharp edges of color semantics (e.g. the
gray<->saturated boundary, the achromatic axis, the boundary between
hue families). They are also the colors where any downstream linear
probe will be brittle.

We measure, for each of the n_colors anchors:

  J_i = (d/dT) [Phi(T) @ B] |_{T = T_i}   ∈ R^{N_PCS × d}     (d=3)

via central finite differences on the gamfit Duchon basis (m chosen
internally by duchon_basis_radial; no Gaussian RBF, no length_scale).
Per anchor we report:

  - L_op   : top singular value of J_i  (operator-norm Lipschitz)
  - L_fro  : ||J_i||_F                 (total local stretch)
  - cond   : s_max / s_min              (local anisotropy)
  - top_dir: leading right-singular vector in T-space (the "fold axis")

Also reported globally:
  - rank ordering: top-20 / bottom-20 anchors by L_op (with xkcd names
    + hex when xkcd cache parses cleanly)
  - hue-binned L_op profile (12 hue bins) — do sharp edges concentrate
    where we'd expect (red<->magenta, green<->yellow)?
  - L_op vs distance-to-achromatic-axis scatter

Cheap: one U_3d fit (cached from auto_exp_06/07 settings) + 2 * 3
basis evaluations per anchor at ~954 points. No server calls. Minutes
on CPU.

Output:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_10_lipschitz.{json,png}
"""
from __future__ import annotations

import colorsys
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_10_lipschitz.json"
OUT_PNG = OUT_DIR / "auto_exp_10_lipschitz.png"
XKCD = Path(__file__).parent / "xkcd_colors.txt"

N_TEMPLATES = 28
N_PCS = 64
D_LATENT = 3
N_ITERS = 12
EPS = 1e-3   # central-difference step in latent-T units
HUE_BINS = 12


def _load_xkcd_names_hex(n_expected: int) -> tuple[list[str], list[str]] | None:
    """Returns (names, hexes) aligned with the harvest row order *iff* the
    cached xkcd file parses to exactly n_expected entries. Otherwise None."""
    if not XKCD.exists():
        return None
    names: list[str] = []
    hexes: list[str] = []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("License") or s.startswith("Copyright"):
            continue
        m = re.match(r"^(.+?)\s+#?([0-9a-fA-F]{6})$", s)
        if not m:
            continue
        names.append(m.group(1).strip())
        hexes.append("#" + m.group(2).lower())
    if len(names) != n_expected:
        print(f"[xkcd] parsed {len(names)} != {n_expected}; skipping labels",
              flush=True)
        return None
    return names, hexes


def _hue_sat_val_from_hex(hx: str) -> tuple[float, float, float]:
    r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def jacobian_at(T_i: np.ndarray, B: np.ndarray, centers: np.ndarray,
                eps: float = EPS) -> np.ndarray:
    """Central-difference Jacobian d/dT [Phi(T) @ B] at a single point T_i.

    T_i: (d,)  B: (K, N_PCS)  centers: (K, d)
    Returns J of shape (N_PCS, d).
    """
    d = T_i.shape[0]
    J = np.zeros((B.shape[1], d), dtype=np.float64)
    for k in range(d):
        e = np.zeros(d); e[k] = eps
        Tp = (T_i + e).reshape(1, -1)
        Tm = (T_i - e).reshape(1, -1)
        Phi_p, _ = cmg.duchon_basis_radial(Tp, centers)
        Phi_m, _ = cmg.duchon_basis_radial(Tm, centers)
        # (Phi_p - Phi_m) @ B / (2 eps)  -> (1, N_PCS)
        J[:, k] = ((Phi_p - Phi_m) @ B).ravel() / (2.0 * eps)
    return J


def jacobian_at_batched(T: np.ndarray, B: np.ndarray, centers: np.ndarray,
                          eps: float = EPS) -> np.ndarray:
    """Vectorized over anchors. T: (n, d) -> J: (n, N_PCS, d). Calls
    duchon_basis_radial 2*d times total instead of 2*d*n times."""
    n, d = T.shape
    K = centers.shape[0]
    Pcs = B.shape[1]
    J = np.zeros((n, Pcs, d), dtype=np.float64)
    for k in range(d):
        e = np.zeros(d); e[k] = eps
        Tp = T + e
        Tm = T - e
        Phi_p, _ = cmg.duchon_basis_radial(Tp, centers)   # (n, K)
        Phi_m, _ = cmg.duchon_basis_radial(Tm, centers)
        # (Phi_p - Phi_m) @ B  -> (n, N_PCS)
        col = (Phi_p - Phi_m) @ B / (2.0 * eps)
        J[:, :, k] = col
    return J


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, Dfull = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # ---- Standard 954xK PCA target (matches auto_exp_04/05/06/07/09) ----
    centroids = np.zeros((n_colors, Dfull), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z = centroids_n @ V_topK.T          # (n_colors, N_PCS)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # ---- Fit U_3d once on all centroids ----
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=1,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))
    print("[fit] U_3d on all centroids ...", flush=True)
    t0 = time.time()
    fit = cmg.fit_unsupervised_manifold(Z, D_LATENT, cfg, n_iters=N_ITERS,
                                         verbose=False)
    print(f"  done ({time.time()-t0:.1f}s)  log_lambda={fit['log_lambda']:+.3f}",
          flush=True)
    T = np.asarray(fit["T"])         # (n_colors, 3)
    B = np.asarray(fit["B"])         # (K_basis, N_PCS)
    centers = np.asarray(fit["centers"])
    print(f"[fit] T={T.shape} B={B.shape} centers={centers.shape}", flush=True)

    # Sanity: training reconstruction
    Phi_T, _ = cmg.duchon_basis_radial(T, centers)
    Z_hat = Phi_T @ B
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    train_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    print(f"[sanity] U_3d train R^2 = {train_r2:+.4f}", flush=True)

    # ---- Per-anchor Jacobians ----
    print(f"[lip] computing finite-diff Jacobians (eps={EPS})...", flush=True)
    t0 = time.time()
    J_all = jacobian_at_batched(T, B, centers, eps=EPS)   # (n, N_PCS, d)
    print(f"  J_all={J_all.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # Per-anchor SVD
    L_op = np.zeros(n_colors)
    L_fro = np.zeros(n_colors)
    cond = np.zeros(n_colors)
    top_dir = np.zeros((n_colors, D_LATENT))
    sv_full = np.zeros((n_colors, D_LATENT))
    for i in range(n_colors):
        Ji = J_all[i]                                # (N_PCS, d)
        U, sv, Vt_loc = np.linalg.svd(Ji, full_matrices=False)
        sv_full[i] = sv
        L_op[i] = float(sv[0])
        L_fro[i] = float(np.linalg.norm(Ji))
        cond[i] = float(sv[0] / max(sv[-1], 1e-12))
        # right-singular vector in T-space for the top stretch direction
        top_dir[i] = Vt_loc[0]

    print(f"[lip] L_op  mean={L_op.mean():.3e}  median={np.median(L_op):.3e}  "
          f"min={L_op.min():.3e}  max={L_op.max():.3e}", flush=True)
    print(f"[lip] cond  mean={cond.mean():.3e}  median={np.median(cond):.3e}  "
          f"max={cond.max():.3e}", flush=True)

    # ---- Hue / sat / value labels (xkcd) ----
    labels = _load_xkcd_names_hex(n_colors)
    hues = sats = vals = None
    if labels is not None:
        names, hexes = labels
        hues = np.array([_hue_sat_val_from_hex(hx)[0] for hx in hexes])
        sats = np.array([_hue_sat_val_from_hex(hx)[1] for hx in hexes])
        vals = np.array([_hue_sat_val_from_hex(hx)[2] for hx in hexes])

    # ---- Distance to achromatic axis: ||(R,G,B) - mean(R,G,B)*[1,1,1]|| ----
    # achromatic = R==G==B. We don't have raw RGB but we have hexes if labels.
    achrom_dist = None
    if labels is not None:
        rgb01 = np.array([[int(hx[1:3], 16) / 255.0,
                            int(hx[3:5], 16) / 255.0,
                            int(hx[5:7], 16) / 255.0] for hx in hexes])
        proj = rgb01.mean(1, keepdims=True) * np.ones((1, 3))
        achrom_dist = np.linalg.norm(rgb01 - proj, axis=1)

    # ---- Top / bottom rankings ----
    order = np.argsort(-L_op)
    top20 = order[:20].tolist()
    bot20 = order[-20:].tolist()

    def _row(i: int) -> dict:
        out = {
            "anchor_idx": int(i),
            "L_op": float(L_op[i]),
            "L_fro": float(L_fro[i]),
            "cond": float(cond[i]),
            "singular_values": sv_full[i].tolist(),
            "top_dir_in_T": top_dir[i].tolist(),
        }
        if labels is not None:
            out["name"] = labels[0][i]
            out["hex"] = labels[1][i]
            out["hue"] = float(hues[i])
            out["sat"] = float(sats[i])
            out["val"] = float(vals[i])
            out["achromatic_dist"] = float(achrom_dist[i])
        return out

    top_rows = [_row(i) for i in top20]
    bot_rows = [_row(i) for i in bot20]

    # ---- Hue-binned L_op profile ----
    hue_profile = None
    if hues is not None:
        edges = np.linspace(0.0, 1.0, HUE_BINS + 1)
        bin_idx = np.clip(np.digitize(hues, edges) - 1, 0, HUE_BINS - 1)
        hp_mean = np.zeros(HUE_BINS)
        hp_median = np.zeros(HUE_BINS)
        hp_count = np.zeros(HUE_BINS, dtype=np.int64)
        for b in range(HUE_BINS):
            mask = (bin_idx == b)
            hp_count[b] = int(mask.sum())
            if hp_count[b]:
                hp_mean[b] = float(L_op[mask].mean())
                hp_median[b] = float(np.median(L_op[mask]))
        hue_profile = {
            "edges": edges.tolist(),
            "mean": hp_mean.tolist(),
            "median": hp_median.tolist(),
            "count": hp_count.tolist(),
        }

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
            "d_latent": D_LATENT, "n_iters": N_ITERS,
            "eps_finite_diff": EPS, "lattice_per_side": 5,
            "hue_bins": HUE_BINS,
        },
        "u3d_train_r2": train_r2,
        "u3d_log_lambda": float(fit["log_lambda"]),
        "lipschitz_stats": {
            "L_op_mean": float(L_op.mean()),
            "L_op_median": float(np.median(L_op)),
            "L_op_p90": float(np.quantile(L_op, 0.90)),
            "L_op_p99": float(np.quantile(L_op, 0.99)),
            "L_op_min": float(L_op.min()),
            "L_op_max": float(L_op.max()),
            "cond_mean": float(cond.mean()),
            "cond_median": float(np.median(cond)),
            "cond_max": float(cond.max()),
        },
        "top20_by_L_op": top_rows,
        "bottom20_by_L_op": bot_rows,
        "hue_profile": hue_profile,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) histogram of L_op
    ax = axes[0, 0]
    ax.hist(L_op, bins=60, color="#4060a0", alpha=0.85, edgecolor="k",
            linewidth=0.3)
    ax.axvline(np.median(L_op), color="k", linestyle="--", linewidth=1,
                label=f"median={np.median(L_op):.2e}")
    ax.axvline(np.quantile(L_op, 0.99), color="#a04060", linestyle=":",
                linewidth=1, label=f"p99={np.quantile(L_op, 0.99):.2e}")
    ax.set_xlabel("L_op (operator-norm Lipschitz of dZ/dT)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-color Lipschitz over {n_colors} anchors\n"
                  f"U_3d train R^2 = {train_r2:+.4f}, eps={EPS}")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    # (b) condition number vs L_op, colored by hue if available
    ax = axes[0, 1]
    if hues is not None:
        sc = ax.scatter(L_op, cond, c=hues, cmap="hsv", s=10, alpha=0.7,
                         edgecolor="none")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("xkcd hue (0=red, 1/3=green, 2/3=blue)", fontsize=8)
    else:
        ax.scatter(L_op, cond, s=10, alpha=0.6, color="#4060a0",
                   edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L_op")
    ax.set_ylabel("condition number s_max / s_min")
    ax.set_title("Lipschitz vs local anisotropy")
    ax.grid(which="both", linestyle=":", alpha=0.4)

    # (c) hue-binned L_op profile
    ax = axes[1, 0]
    if hue_profile is not None:
        edges = np.array(hue_profile["edges"])
        centers_hb = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers_hb, hue_profile["median"], width=1.0 / HUE_BINS * 0.9,
                color=[plt.cm.hsv(h) for h in centers_hb], edgecolor="k",
                linewidth=0.4, alpha=0.95)
        ax.set_xlabel("hue (0..1)")
        ax.set_ylabel("median L_op in bin")
        ax.set_title(f"Hue-binned median Lipschitz ({HUE_BINS} bins)\n"
                      f"counts: {hue_profile['count']}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "xkcd labels unavailable", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # (d) top-15 anchors by L_op as horizontal bar with hex swatches
    ax = axes[1, 1]
    show = top_rows[:15]
    y = np.arange(len(show))
    vals_op = [r["L_op"] for r in show]
    colors_hex = [r.get("hex", "#888888") for r in show]
    names_show = [r.get("name", f"#{r['anchor_idx']}") for r in show]
    ax.barh(y, vals_op, color=colors_hex, edgecolor="k", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_show, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("L_op")
    ax.set_title("Top-15 sharpest anchors (bar color = xkcd hex)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
