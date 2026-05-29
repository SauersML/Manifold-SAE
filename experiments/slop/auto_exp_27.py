"""auto_exp_27 — three-penalty stack `[Orthogonality, ARD, TV-graph]` on a
Circle-manifold LatentCoord (cogito L40 colors).

The composition-engine claim is that analytic penalties "just compose"
through the REML outer loop. auto_exp_23 verified the pair (Ortho × ARD).
auto_exp_24/26 verified TV in isolation. **No experiment** yet stacks
three analytic penalties simultaneously through `penalties=`. This is the
gap this experiment closes.

Setup
-----
- LatentCoord on a 2D Euclidean aux block U (d=2 aux), driven by a
  separately-fit Circle-manifold hue coordinate θ. Total latent block is
  T = [cos θ, sin θ, U[:,0], U[:,1]] (4D).
- Decoder: linear PC-16 map B s.t. Z ≈ T B.
- Penalties stacked (all three, in one outer loop):
    Orthogonality on U   — gauge-fix the rotation of the aux frame.
    ARD on U             — one log-precision per aux dim.
    TV on U over the hue-kNN graph — piecewise-smooth aux atom maps.
- Four ablation configs (control crescendo):
    C0: no penalties              (everything free)
    C1: ARD only                  (rotation-invariant, ARD can't prune)
    C2: Ortho + ARD               (gauge-fixed, ARD prunes)
    C3: Ortho + ARD + TV-graph    (gauge + prune + piecewise-smooth)

Hypotheses (preregistered, STRICT)
----------------------------------
  H1 (gauge fix releases ARD): aux_dims_kept(C2) < aux_dims_kept(C1).
  H2 (TV doesn't undo ARD):    aux_dims_kept(C3) <= aux_dims_kept(C2).
  H3 (TV reduces transitions): tr(C3) < tr(C2)  where tr = sum of
                               argmax-aux-axis flips along hue-kNN edges.
  H4 (recon stays competitive): r2_recon(C3) >= 0.85 * r2_recon(C0).

Composition-engine status
-------------------------
Installed gamfit 0.1.112 lacks `LatentCoord`, `OrthogonalityPenalty`,
`ARDPenalty`, `TotalVariationPenalty`, and the `penalties=` kwarg. All
four configs are emulated with the documented graceful-fallback pattern
from auto_exp_23 (QR + EM-ARD + Huber-TV). `prediction_slot` flags this
for a v0.1.121 re-run that should reach the real Rust primitives in a
single `gamfit.fit(...)` call.

Outputs
-------
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_27.{png,json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _pca_basis import load_pc_basis  # noqa: E402

ROOT = Path("/Users/user/Manifold-SAE")
HARVEST = ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"
XKCD = ROOT / "experiments/xkcd_colors.txt"
OUT_DIR = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40"
OUT_PNG = OUT_DIR / "auto_exp_27.png"
OUT_JSON = OUT_DIR / "auto_exp_27.json"

N_TEMPLATES = 28
K_PC = 16
D_AUX = 4
K_NN = 5
N_ITERS = 80
ARD_PRUNE_THR = 0.10
SEED = 0


# ---- gamfit probe -----------------------------------------------------

def gamfit_meta() -> tuple[str, dict[str, bool]]:
    import gamfit
    version = getattr(gamfit, "__version__", "unknown")
    flags = {}
    for n in ["LatentCoord", "OrthogonalityPenalty", "ARDPenalty",
              "TotalVariationPenalty", "select_topology", "topology"]:
        flags[n] = hasattr(gamfit, n)
    import inspect
    try:
        flags["fit_penalties_kwarg"] = "penalties" in inspect.signature(
            gamfit.fit).parameters
    except Exception:
        flags["fit_penalties_kwarg"] = False
    return version, flags


# ---- data -------------------------------------------------------------

def load_xkcd_rgb() -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        h = parts[1].lstrip("#")
        names.append(parts[0])
        rgb.append((int(h[0:2], 16) / 255.0,
                    int(h[2:4], 16) / 255.0,
                    int(h[4:6], 16) / 255.0))
    return names, np.array(rgb, dtype=np.float64)


def per_color_centroids_mmap(X_mmap: np.ndarray, n_colors: int) -> np.ndarray:
    D = X_mmap.shape[1]
    out = np.zeros((n_colors, D), dtype=np.float32)
    for ci in range(n_colors):
        s = ci * N_TEMPLATES
        out[ci] = np.asarray(X_mmap[s: s + N_TEMPLATES]).mean(0)
    return out


# ---- circular kNN graph (reused from auto_exp_26) -------------------

def circular_hue_knn_edges(hue: np.ndarray, k: int) -> np.ndarray:
    cs = np.stack([np.cos(2 * np.pi * hue), np.sin(2 * np.pi * hue)], axis=1)
    d2 = ((cs[:, None, :] - cs[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argpartition(d2, k, axis=1)[:, :k]
    edges = set()
    for i in range(d2.shape[0]):
        for j in nbr[i]:
            a, b = int(min(i, j)), int(max(i, j))
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


# ---- composition-engine fallback (one solver, four configs) --------

def huber_grad(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(x * x + eps * eps)


def fit_three_penalty_stack(
    Z: np.ndarray,
    theta: np.ndarray,             # (N,) circle coordinate, fixed
    edges: np.ndarray,
    d_aux: int,
    *,
    use_ortho: bool,
    use_ard: bool,
    use_tv: bool,
    lam_ortho: float = 1.0,
    lam_tv: float = 1.0,
    ard_eps: float = 1e-3,
    huber_eps: float = 1e-3,
    n_iters: int = N_ITERS,
    seed: int = SEED,
) -> dict:
    """Joint outer loop:
        T = [cos θ, sin θ, U]              (N, 2 + d_aux)
        B (decoder coefs):  Z ≈ T B        (least-squares solve)
        U (per-row latent): updated by gradient step under stacked penalties
        tau_j (ARD per-axis precision):    EM update
    Fallback for `gamfit.fit(latents=[LatentCoord(...)],
                              penalties=[Ortho, ARD, TV(graph_edges)])`.
    """
    rng = np.random.default_rng(seed)
    N, K = Z.shape

    fix = np.stack([np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta)],
                   axis=1)                                        # (N, 2)
    U = 0.1 * rng.standard_normal((N, d_aux))
    tau = np.ones(d_aux)                                          # ARD prec

    u_idx, v_idx = edges[:, 0], edges[:, 1]

    history = {"recon_loss": [], "tau": []}
    for it in range(n_iters):
        T = np.concatenate([fix, U], axis=1)                      # (N, 2+d)
        # closed-form B given T (ridge for stability)
        G = T.T @ T + 1e-6 * np.eye(T.shape[1])
        B = np.linalg.solve(G, T.T @ Z)                           # (2+d, K)
        # gradient w.r.t. U (aux block only): - (Z - T B) B[2:].T
        resid = Z - T @ B                                          # (N, K)
        gU = -resid @ B[2:].T                                      # (N, d)
        # ARD axis-wise shrinkage
        if use_ard:
            # EM: tau_j ~ N / (sum U[:,j]^2 + ard_eps)
            tau = N / (np.sum(U * U, axis=0) + ard_eps)
            tau = np.minimum(tau, 1e6)
            gU = gU + (U * tau[None, :])
        # Orthogonality: penalize off-diagonal of (U^T U)/N
        if use_ortho:
            M = (U.T @ U) / max(N, 1)
            off = M - np.diag(np.diag(M))                         # (d, d)
            # d/dU of ½ ||off||_F² = U @ (off + off.T) / N
            gU = gU + lam_ortho * (U @ (off + off.T)) / max(N, 1)
        # TV on graph edges
        if use_tv:
            diff = U[v_idx] - U[u_idx]                            # (E, d)
            hg = huber_grad(diff, huber_eps)
            tv_g = np.zeros_like(U)
            np.add.at(tv_g, u_idx, -hg)
            np.add.at(tv_g, v_idx,  hg)
            gU = gU + lam_tv * tv_g
        # adaptive step
        lr = 0.02 / (1.0 + 0.005 * it)
        U = U - lr * gU
        history["recon_loss"].append(float((resid ** 2).mean()))
        history["tau"].append(tau.copy().tolist())

    # final fit & metrics
    T = np.concatenate([fix, U], axis=1)
    G = T.T @ T + 1e-6 * np.eye(T.shape[1])
    B = np.linalg.solve(G, T.T @ Z)
    Zhat = T @ B
    ss_res = ((Z - Zhat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # ARD-kept: axes whose var/max_var >= threshold
    var_axis = U.var(axis=0)
    aux_kept = int((var_axis / max(var_axis.max(), 1e-12) >= ARD_PRUNE_THR
                    ).sum())

    # transitions: argmax(|U_j|) over hue-kNN edges
    seq = np.argmax(np.abs(U), axis=1)
    transitions_graph = int((seq[u_idx] != seq[v_idx]).sum())

    return {
        "U": U, "B": B, "r2_recon": float(r2),
        "var_axis": var_axis.tolist(),
        "aux_dims_kept": aux_kept,
        "transitions_graph": transitions_graph,
        "history": history,
        "argmax_seq": seq,
    }


def fit_circle_theta(Z: np.ndarray, hue: np.ndarray) -> np.ndarray:
    """Use the HSV hue (already periodic) as initial θ. In real
    composition-engine code this would be the Riemannian Circle
    LatentCoord — the fallback just pins it to HSV hue."""
    return hue.copy()


# ---- main -------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    version, flags = gamfit_meta()
    primitives_reached = [k for k, v in flags.items() if v]
    primitives_fallback = [k for k, v in flags.items() if not v]
    print(f"[gamfit] version={version}")
    print(f"[gamfit] reached={primitives_reached}")
    print(f"[gamfit] fallback for={primitives_fallback}")

    print("[load] X_L40 mmap")
    X = np.load(HARVEST, mmap_mode="r")
    names, rgb = load_xkcd_rgb()
    n_colors = min(len(names), X.shape[0] // N_TEMPLATES)
    names = names[:n_colors]
    rgb = rgb[:n_colors]
    centroids = per_color_centroids_mmap(X, n_colors)
    del X

    basis = load_pc_basis(K=64)
    Vt = basis["Vt"][:K_PC]
    Z_all = ((centroids - basis["mu"]) / basis["sigma"]) @ Vt.T

    hsv = np.array([mcolors.rgb_to_hsv(c) for c in rgb])
    chrom = hsv[:, 1] >= 0.1
    Z = Z_all[chrom]
    hue = hsv[chrom, 0]
    rgb_c = rgb[chrom]
    order = np.argsort(hue)
    Z = Z[order]; hue = hue[order]; rgb_c = rgb_c[order]
    N = Z.shape[0]
    print(f"[setup] N={N}, K_PC={K_PC}, d_aux={D_AUX}")

    edges = circular_hue_knn_edges(hue, K_NN)
    n_1d = N - 1
    lam_tv_norm = 1.0 * n_1d / max(len(edges), 1)
    print(f"[graph] |E|={len(edges)} lam_tv_norm={lam_tv_norm:.4f}")

    theta = fit_circle_theta(Z, hue)

    configs = [
        ("C0_free",         dict(use_ortho=False, use_ard=False, use_tv=False)),
        ("C1_ARD",          dict(use_ortho=False, use_ard=True,  use_tv=False)),
        ("C2_Ortho_ARD",    dict(use_ortho=True,  use_ard=True,  use_tv=False)),
        ("C3_Ortho_ARD_TV", dict(use_ortho=True,  use_ard=True,  use_tv=True)),
    ]
    results = {}
    for name, kw in configs:
        print(f"[fit] {name} ...")
        res = fit_three_penalty_stack(
            Z, theta, edges, D_AUX,
            lam_ortho=2.0, lam_tv=lam_tv_norm,
            **kw,
        )
        results[name] = res
        print(f"  r2={res['r2_recon']:.3f}  kept={res['aux_dims_kept']}/{D_AUX}  "
              f"transitions_graph={res['transitions_graph']}")

    # ---- hypotheses ----
    kept_c1 = results["C1_ARD"]["aux_dims_kept"]
    kept_c2 = results["C2_Ortho_ARD"]["aux_dims_kept"]
    kept_c3 = results["C3_Ortho_ARD_TV"]["aux_dims_kept"]
    tr_c2   = results["C2_Ortho_ARD"]["transitions_graph"]
    tr_c3   = results["C3_Ortho_ARD_TV"]["transitions_graph"]
    r2_c0   = results["C0_free"]["r2_recon"]
    r2_c3   = results["C3_Ortho_ARD_TV"]["r2_recon"]

    H1 = bool(kept_c2 < kept_c1)
    H2 = bool(kept_c3 <= kept_c2)
    H3 = bool(tr_c3 < tr_c2)
    H4 = bool(r2_c3 >= 0.85 * max(r2_c0, 0.0))
    verdicts = {
        "H1_gauge_fix_releases_ARD":    H1,
        "H2_TV_does_not_undo_ARD":      H2,
        "H3_TV_reduces_graph_transitions": H3,
        "H4_recon_stays_competitive":   H4,
    }

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # (a) per-config |U| variance per axis (ARD shrinkage signature)
    ax = axes[0, 0]
    width = 0.18
    x = np.arange(D_AUX)
    for j, (name, _) in enumerate(configs):
        va = np.array(results[name]["var_axis"])
        ax.bar(x + (j - 1.5) * width, va / max(va.max(), 1e-12),
               width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"u_{j}" for j in range(D_AUX)])
    ax.set_ylabel("normalized var(U_j)")
    ax.set_title("aux-axis variances (ARD pruning signature)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) recon R² + kept dims summary
    ax = axes[0, 1]
    rs = [results[n]["r2_recon"] for n, _ in configs]
    ks = [results[n]["aux_dims_kept"] for n, _ in configs]
    trs = [results[n]["transitions_graph"] for n, _ in configs]
    ax2 = ax.twinx()
    pos = np.arange(len(configs))
    ax.bar(pos - 0.2, rs, width=0.4, label="recon R²", color="#48a")
    ax2.bar(pos + 0.2, ks, width=0.4, label="aux dims kept",
            color="#c44", alpha=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels([n for n, _ in configs], rotation=20, fontsize=8)
    ax.set_ylabel("recon R²", color="#48a")
    ax2.set_ylabel("aux dims kept (of 4)", color="#c44")
    ax.set_title(
        f"recon vs ARD pruning  |  transitions={trs}")
    ax.grid(True, alpha=0.3)

    # (c) C2 vs C3 argmax-aux sequence over hue-ordered rows
    ax = axes[1, 0]
    ax.plot(results["C2_Ortho_ARD"]["argmax_seq"],
            label=f"C2 (Ortho+ARD)  tr={tr_c2}",
            color="#888", alpha=0.7)
    ax.plot(results["C3_Ortho_ARD_TV"]["argmax_seq"],
            label=f"C3 (+TV graph)   tr={tr_c3}",
            color="#c44", alpha=0.8)
    ax.set_xlabel("hue-ordered color index")
    ax.set_ylabel("argmax aux axis")
    ax.set_title("argmax-axis sequence: C2 vs C3")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) recon-loss history per config
    ax = axes[1, 1]
    for name, _ in configs:
        ax.plot(results[name]["history"]["recon_loss"], label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("recon MSE")
    ax.set_title("convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"auto_exp_27: 3-penalty stack [Ortho, ARD, TV-graph] on cogito L40 "
        f"(N={N}, d_aux={D_AUX})  |  H1={H1} H2={H2} H3={H3} H4={H4}",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[plot] wrote {OUT_PNG}")

    runtime = time.time() - t0
    summary = {
        "experiment": "auto_exp_27",
        "hypothesis": (
            "Three analytic penalties [Ortho, ARD, TV-graph] stacked through "
            "the composition engine each contribute orthogonally: gauge-fix "
            "releases ARD pruning, TV does not undo ARD, TV reduces graph "
            "transitions, and recon stays >=85% of unpenalized."
        ),
        "gamfit_version_actually_used": version,
        "primitives_reached":   primitives_reached,
        "primitives_fallback":  primitives_fallback,
        "n_chromatic": int(N),
        "K_PC": K_PC,
        "d_aux": D_AUX,
        "k_NN": K_NN,
        "n_graph_edges": int(len(edges)),
        "lam_ortho": 2.0,
        "lam_tv_normalized": float(lam_tv_norm),
        "per_config": {
            name: {
                "r2_recon":          float(results[name]["r2_recon"]),
                "aux_dims_kept":     int(results[name]["aux_dims_kept"]),
                "var_axis":          results[name]["var_axis"],
                "transitions_graph": int(results[name]["transitions_graph"]),
            } for name, _ in configs
        },
        "hypothesis_verdicts": verdicts,
        "runtime_seconds": float(runtime),
        "prediction_slot": {
            "rerun_under_v0_1_121_using": (
                "gamfit.fit(latents=[LatentCoord(d=2+d_aux, "
                "manifold=Circle(dim=0))], "
                "penalties=[OrthogonalityPenalty(weight=2.0, n_eff=N), "
                "ARDPenalty(n_dims=d_aux, strength='auto'), "
                "TotalVariationPenalty(weight=lam_tv_norm, n_eff=N, "
                "difference_op=edges)])"
            ),
            "expected_H1_under_real_primitive": True,
            "expected_H2_under_real_primitive": True,
            "expected_H3_under_real_primitive": True,
            "expected_H4_under_real_primitive": True,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[json] wrote {OUT_JSON}")
    print(f"[verdict] H1={H1} H2={H2} H3={H3} H4={H4} runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()
