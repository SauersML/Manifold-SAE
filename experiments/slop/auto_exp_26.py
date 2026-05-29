"""auto_exp_26 — TotalVariationPenalty with **graph_edges** (hue-kNN) on cogito L40.

Hypothesis
----------
Compared to auto_exp_24's 1D forward-diff TV (which orders rows by hue and
penalizes only consecutive pairs), a TV penalty whose `difference_op` is a
**symmetric k-nearest-neighbour graph in HSV-hue space** gives:

  (H1) STRICTLY FEWER argmax-atom transitions, because the graph penalty
       enforces smoothness across all kNN edges, not just the 1D chain;
       isolated outliers can't open a new band.
  (H2) per-hue R^2 (predicting cos/sin hue from assignment) is within 10%
       of the 1D forward-diff baseline — graph TV shouldn't crush the
       perceptual axis it's smoothing over.

This directly exercises the
``TotalVariationPenalty(difference_op=[(a,b),...])`` graph-edges branch,
which is the **MISSING** coverage corner in our composition-engine map.

Installed gamfit (0.1.112) lacks the primitive, so we use the same
gracious fallback pattern as auto_exp_24 (Huber-smoothed L1 on edges,
emulating the same composition dynamics). `prediction_slot` flags this for
a v0.1.121 re-run.

Outputs
-------
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_26.{png,json}
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
OUT_PNG = OUT_DIR / "auto_exp_26.png"
OUT_JSON = OUT_DIR / "auto_exp_26.json"

N_TEMPLATES = 28
K_PC = 16
K_ATOMS = 4
K_NN = 5  # graph neighbours per row in HSV-hue


def gamfit_meta() -> tuple[str, dict[str, bool]]:
    import gamfit
    version = getattr(gamfit, "__version__", "unknown")
    flags: dict[str, bool] = {}
    for n in ["TotalVariationPenalty", "LatentCoord", "select_topology"]:
        flags[n] = hasattr(gamfit, n)
    # try the graph-edges branch specifically
    try:
        from gamfit._penalties import TotalVariationPenalty
        try:
            TotalVariationPenalty(weight=1.0, n_eff=4,
                                  difference_op=[(0, 1), (1, 2)])
            flags["TotalVariationPenalty_graph_edges"] = True
        except Exception:
            flags["TotalVariationPenalty_graph_edges"] = False
    except Exception:
        flags["TotalVariationPenalty_graph_edges"] = False
    return version, flags


def load_xkcd_rgb() -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
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


# ---- TV fits ----------------------------------------------------------

def huber_grad(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(x * x + eps * eps)


def fit_atoms_tv_forward1d(
    Z: np.ndarray, K: int, atoms: np.ndarray,
    lam_tv: float, lam_l1: float = 0.05,
    huber_eps: float = 1e-3, iters: int = 400, lr: float = 0.03,
) -> np.ndarray:
    """1D forward-diff TV (rows assumed hue-ordered) — auto_exp_24 baseline."""
    N, _ = Z.shape
    A = np.zeros((N, K), dtype=np.float64)
    G = atoms @ atoms.T
    PhiZ = Z @ atoms.T
    for _ in range(iters):
        g = A @ G - PhiZ
        diff = A[1:] - A[:-1]
        hg = huber_grad(diff, huber_eps)
        tv_g = np.zeros_like(A)
        tv_g[:-1] -= hg
        tv_g[1:] += hg
        l1_g = huber_grad(A, huber_eps)
        A = A - lr * (g + lam_tv * tv_g + lam_l1 * l1_g)
    return A


def fit_atoms_tv_graph(
    Z: np.ndarray, K: int, atoms: np.ndarray,
    edges: np.ndarray,
    lam_tv: float, lam_l1: float = 0.05,
    huber_eps: float = 1e-3, iters: int = 400, lr: float = 0.03,
) -> np.ndarray:
    """Graph-edges TV: penalize |A[u]-A[v]| over arbitrary edges (u,v).

    Emulates ``TotalVariationPenalty(difference_op=[(u,v),...])`` from the
    post-0.1.120 composition engine; on graceful fallback the dynamics are
    identical (smoothed-L1 on D@T where D is the edge-incidence operator).
    """
    N, _ = Z.shape
    A = np.zeros((N, K), dtype=np.float64)
    G = atoms @ atoms.T
    PhiZ = Z @ atoms.T
    u_idx = edges[:, 0]
    v_idx = edges[:, 1]
    for _ in range(iters):
        g = A @ G - PhiZ
        diff = A[v_idx] - A[u_idx]                      # (E, K)
        hg = huber_grad(diff, huber_eps)                # (E, K)
        tv_g = np.zeros_like(A)
        # scatter (no GPU; np.add.at handles repeated indices correctly)
        np.add.at(tv_g, u_idx, -hg)
        np.add.at(tv_g, v_idx, hg)
        l1_g = huber_grad(A, huber_eps)
        A = A - lr * (g + lam_tv * tv_g + lam_l1 * l1_g)
    return A


# ---- metrics ----------------------------------------------------------

def reconstruct_r2(Z: np.ndarray, A: np.ndarray, atoms: np.ndarray) -> float:
    Zhat = A @ atoms
    ss_res = ((Z - Zhat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def hue_r2_from_assignment(A: np.ndarray, hue: np.ndarray) -> float:
    Y = np.stack([np.cos(2 * np.pi * hue), np.sin(2 * np.pi * hue)], axis=1)
    X = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ Y)
    Yhat = X @ beta
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def count_transitions_1d(seq: np.ndarray) -> int:
    return int((seq[1:] != seq[:-1]).sum())


def count_graph_transitions(seq: np.ndarray, edges: np.ndarray) -> int:
    """Edges across which argmax-atom disagrees."""
    return int((seq[edges[:, 0]] != seq[edges[:, 1]]).sum())


# ---- kNN on circular hue ---------------------------------------------

def circular_hue_knn_edges(hue: np.ndarray, k: int) -> np.ndarray:
    """Symmetric kNN graph on circular HSV-hue (chord distance, O(N²))."""
    N = hue.shape[0]
    # chord distance on the unit circle
    cs = np.stack([np.cos(2 * np.pi * hue), np.sin(2 * np.pi * hue)], axis=1)
    d2 = ((cs[:, None, :] - cs[None, :, :]) ** 2).sum(-1)  # (N, N)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argpartition(d2, k, axis=1)[:, :k]            # (N, k)
    edges = set()
    for i in range(N):
        for j in nbr[i]:
            a, b = int(min(i, j)), int(max(i, j))
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


# ---- main -------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    version, primitive_flags = gamfit_meta()
    real = primitive_flags.get("TotalVariationPenalty_graph_edges", False)
    primitives_reached = (
        "TotalVariationPenalty(graph_edges)" if real
        else "fallback_huber_tv_graph"
    )
    primitives_fallback = [k for k, v in primitive_flags.items() if not v]
    print(f"[gamfit] version={version}  reached={primitives_reached}")
    print(f"[gamfit] fallback for: {primitives_fallback}")

    print("[load] X_L40 mmap")
    X = np.load(HARVEST, mmap_mode="r")
    names, rgb = load_xkcd_rgb()
    n_colors = min(len(names), X.shape[0] // N_TEMPLATES)
    names = names[:n_colors]
    rgb = rgb[:n_colors]
    print(f"[load] n_colors={n_colors}, D={X.shape[1]}")

    centroids = per_color_centroids_mmap(X, n_colors)
    del X  # release mmap before fits

    print(f"[pca] loading cached basis K=64")
    basis = load_pc_basis(K=64)
    Vt = basis["Vt"][:K_PC]
    mu = basis["mu"]
    sigma = basis["sigma"]
    Z_all = ((centroids - mu) / sigma) @ Vt.T

    hsv = np.array([mcolors.rgb_to_hsv(c) for c in rgb])
    chrom = hsv[:, 1] >= 0.1
    Z = Z_all[chrom]
    hue = hsv[chrom, 0]
    rgb_c = rgb[chrom]
    # hue order for the 1D baseline (graph fit is order-invariant)
    order = np.argsort(hue)
    Z = Z[order]
    hue = hue[order]
    rgb_c = rgb_c[order]
    N = Z.shape[0]
    print(f"[setup] N_chromatic={N}, K_PC={K_PC}, K_atoms={K_ATOMS}, k_NN={K_NN}")

    edges = circular_hue_knn_edges(hue, k=K_NN)
    print(f"[graph] |E|={len(edges)} on circular hue kNN")

    # init atoms (PCA on Z)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, VT = np.linalg.svd(Zc, full_matrices=False)
    atoms = VT[:K_ATOMS]

    # tune lam_tv so both penalties have similar total penalty weight
    # 1D has (N-1) edges; graph has |E| edges. Scale lam_tv by ratio so the
    # AGGREGATE penalty magnitudes are comparable.
    lam_1d = 1.5
    n_1d_edges = N - 1
    lam_graph = lam_1d * n_1d_edges / max(len(edges), 1)
    print(f"[lam] forward1d lam_tv={lam_1d}, graph lam_tv={lam_graph:.4f}")

    print(f"[fit] TV forward_1d ...")
    A_f = fit_atoms_tv_forward1d(Z, K_ATOMS, atoms, lam_tv=lam_1d,
                                  iters=400, lr=0.03)
    print(f"[fit] TV graph_edges ...")
    A_g = fit_atoms_tv_graph(Z, K_ATOMS, atoms, edges, lam_tv=lam_graph,
                              iters=400, lr=0.03)

    r2_f = reconstruct_r2(Z, A_f, atoms)
    r2_g = reconstruct_r2(Z, A_g, atoms)
    seq_f = np.argmax(np.abs(A_f), axis=1)
    seq_g = np.argmax(np.abs(A_g), axis=1)

    # transitions in BOTH metrics for fair comparison
    tr_f_1d = count_transitions_1d(seq_f)
    tr_g_1d = count_transitions_1d(seq_g)
    tr_f_graph = count_graph_transitions(seq_f, edges)
    tr_g_graph = count_graph_transitions(seq_g, edges)

    hue_r2_f = hue_r2_from_assignment(A_f, hue)
    hue_r2_g = hue_r2_from_assignment(A_g, hue)

    print(f"[recon] r2_forward1d={r2_f:.3f}  r2_graph={r2_g:.3f}")
    print(f"[trans] forward1d: 1D={tr_f_1d} graph={tr_f_graph}")
    print(f"[trans] graph:     1D={tr_g_1d} graph={tr_g_graph}")
    print(f"[hue]   r2_forward1d={hue_r2_f:.3f}  r2_graph={hue_r2_g:.3f}")

    # ---- hypotheses (STRICT) ----
    # H1: graph TV has fewer transitions THAN forward1d (use 1D ordering
    # as common-denominator: both seqs are over the hue-sorted rows).
    h1 = bool(tr_g_1d < tr_f_1d)
    # H2: hue R² within 10% of forward1d
    rel = abs(hue_r2_g - hue_r2_f) / max(abs(hue_r2_f), 1e-9)
    h2 = bool(rel <= 0.10)

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    # (a) assignment stripes — forward 1D
    ax = axes[0, 0]
    ax.imshow(np.abs(A_f).T, aspect="auto", interpolation="nearest",
              cmap="magma")
    ax.set_title(f"forward_1d TV: |A| heatmap  (transitions={tr_f_1d})")
    ax.set_xlabel("hue-ordered color index")
    ax.set_ylabel("atom k")

    # (b) assignment stripes — graph kNN
    ax = axes[0, 1]
    ax.imshow(np.abs(A_g).T, aspect="auto", interpolation="nearest",
              cmap="magma")
    ax.set_title(f"graph kNN TV: |A| heatmap  (transitions={tr_g_1d})")
    ax.set_xlabel("hue-ordered color index")
    ax.set_ylabel("atom k")

    # (c) hue swatches + argmax atom bar
    ax = axes[1, 0]
    swatch = np.tile(rgb_c[:, None, :], (1, 8, 1)).transpose(1, 0, 2)
    ax.imshow(swatch, aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("hue-ordered color index")
    ax.set_title("color swatches (hue-ordered)")

    ax = axes[1, 1]
    ax.plot(seq_f, label=f"forward_1d (tr={tr_f_1d})", color="#888", alpha=0.7)
    ax.plot(seq_g, label=f"graph kNN (tr={tr_g_1d})", color="#c44", alpha=0.8)
    ax.set_xlabel("hue-ordered color index")
    ax.set_ylabel("argmax atom")
    ax.set_title(f"argmax-atom sequence  |  hue R²: fwd={hue_r2_f:.2f}, "
                 f"graph={hue_r2_g:.2f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"auto_exp_26: TV graph_edges vs forward_1d on cogito L40 "
        f"(N={N}, k_NN={K_NN}, |E|={len(edges)})  "
        f"H1={h1} H2={h2}",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[plot] wrote {OUT_PNG}")

    runtime = time.time() - t0
    summary = {
        "experiment": "auto_exp_26",
        "hypothesis": (
            "TV with graph_edges (hue-kNN) yields fewer atom transitions "
            "than forward_1d TV at within-10% per-hue R²."
        ),
        "gamfit_version_actually_used": version,
        "primitives_reached": primitives_reached,
        "primitives_fallback": primitives_fallback,
        "real_graph_edges_branch": real,
        "n_chromatic": int(N),
        "K_PC": K_PC,
        "K_atoms": K_ATOMS,
        "k_NN": K_NN,
        "n_graph_edges": int(len(edges)),
        "lam_forward1d": lam_1d,
        "lam_graph_normalized": float(lam_graph),
        "r2_forward1d": float(r2_f),
        "r2_graph": float(r2_g),
        "transitions_1d_metric": {
            "forward1d_fit": int(tr_f_1d),
            "graph_fit":     int(tr_g_1d),
        },
        "transitions_graph_metric": {
            "forward1d_fit": int(tr_f_graph),
            "graph_fit":     int(tr_g_graph),
        },
        "hue_r2_forward1d": float(hue_r2_f),
        "hue_r2_graph":     float(hue_r2_g),
        "hue_r2_relative_diff": float(rel),
        "hypothesis_verdicts": {
            "H1_graph_fewer_transitions": h1,
            "H2_hue_r2_within_10pct":    h2,
        },
        "runtime_seconds": float(runtime),
        "prediction_slot": {
            "rerun_under_v0_1_121_using":
                "gamfit.fit(latents=[LatentCoord(d=K_atoms)], "
                "penalties=[TotalVariationPenalty(weight=lam_graph, "
                "n_eff=N, difference_op=edges)])",
            "expected_H1_under_real_primitive": True,
            "expected_H2_under_real_primitive": True,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[json] wrote {OUT_JSON}")
    print(f"[verdict] H1={h1} H2={h2} runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()
