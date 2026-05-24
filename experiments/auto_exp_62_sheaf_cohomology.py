"""auto_exp_62 — Sheaf cohomology of a paired 3-layer SAE on cogito-L40.

Pipeline:
  1. synthesize paired layers (PCA stand-in or reuse Crosscoder stack)
  2. train 3 paired TopK SAEs with sheaf-consistency aux loss
  3. extract harmonic atoms from ker L_sheaf
  4. build a Mapper-style nearest-neighbour graph over harmonic-atom
     decoder vectors (per-layer-averaged) and compute persistent H¹ via a
     30-LOC inline Vietoris-Rips-on-1-skeleton routine

Falsifiable prediction
----------------------
The cogito hue manifold is circular. If sheaf cohomology recovers true
cross-layer features (rather than per-layer artefacts), at least one
long-persistence H¹ class — the hue ring — should appear in the harmonic
subset. If H¹ is empty: the "circular hue across layers" story is wrong on
this stand-in.

Outputs under ``runs/auto_exp_62_sheaf/``:
  history.npz, harmonic_curve.png (from training)
  persistence_h1.png
  results.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/auto_exp_62_sheaf"
PAIRED = REPO / "runs/SHEAF_PAIRED_3LAYER"
TRAIN = PAIRED / "train"


def _run(script: str) -> None:
    cmd = [sys.executable, str(REPO / script)]
    print(f"[auto_exp_62] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO))


# ---------------------------------------------------------------------------
# Inline 1D persistent H¹: classify cycles in a Vietoris-Rips 1-skeleton via
# union-find + edge-by-edge addition (Edelsbrunner-Letscher-Zomorodian).
# H¹ generators = edges that create a cycle; persistence = (birth, death=∞)
# truncated by the filtration cap.
# ---------------------------------------------------------------------------


def persistent_h1(points: np.ndarray, max_radius: float, n_steps: int = 60):
    """Return list of (birth, death) for H¹ classes up to ``max_radius``.

    Tiny VR-style: we add edges in order of pairwise distance, track H⁰
    component count via union-find; an edge that joins two already-connected
    nodes is a 1-cycle generator (birth = its length). Death = radius at
    which the cycle is filled by a triangle (we approximate triangle birth
    = max edge length of its 3 edges). For sparse Mapper-style graphs this
    is exact on the 1-skeleton.
    """
    N = points.shape[0]
    if N < 3:
        return []
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    iu = np.triu_indices(N, k=1)
    edges = sorted(zip(iu[0].tolist(), iu[1].tolist(), D[iu].tolist()), key=lambda e: e[2])

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    cycle_births: list[tuple[int, int, float]] = []  # (u, v, birth)
    accepted_edges: list[tuple[int, int, float]] = []
    for u, v, d in edges:
        if d > max_radius:
            break
        if not union(u, v):
            cycle_births.append((u, v, d))
        accepted_edges.append((u, v, d))

    # Cycle death: smallest filtration radius r at which a triangle covers the
    # cycle's defining edge — i.e. minimum, over nodes w adjacent to both u
    # and v at radius ≤ r, of max(d(u,w), d(v,w), d(u,v)).
    pers: list[tuple[float, float]] = []
    for (u, v, birth) in cycle_births:
        # candidate triangles: any w
        tri_radii = []
        for w in range(N):
            if w == u or w == v:
                continue
            r = max(D[u, w], D[v, w], birth)
            if r <= max_radius:
                tri_radii.append(r)
        death = min(tri_radii) if tri_radii else max_radius
        pers.append((float(birth), float(death)))
    pers.sort(key=lambda bd: -(bd[1] - bd[0]))
    return pers


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1+2: synthesize + train ---
    if not any(PAIRED.glob("X_layer*.npy")):
        _run("scripts/synthesize_paired_layers.py")
    if not (TRAIN / "saes.pt").exists():
        _run("scripts/train_sheaf_consistent_sae.py")

    # --- 3: harmonic atoms ---
    from manifold_sae.sheaf import CellularSheaf, SheafConsistencyHead, harmonic_atoms
    from scripts.train_sheaf_consistent_sae import TinyTopKSAE

    ckpt = torch.load(TRAIN / "saes.pt", map_location="cpu", weights_only=False)
    dims = ckpt["dims"]
    saes = torch.nn.ModuleList([TinyTopKSAE(D, F=256, k=32) for D in dims])
    saes.load_state_dict(ckpt["saes"])
    head = SheafConsistencyHead(n_layers=len(dims), F=256)
    head.load_state_dict(ckpt["head"])

    sheaf = CellularSheaf(layer_saes=list(saes), restriction_maps=head.restriction_dict())
    L = sheaf.laplacian()
    eigvals = np.linalg.eigvalsh(L)
    # same tol rule as the training script
    tol = max(1e-9, 0.05 * float(eigvals.mean()))
    modes, atoms = harmonic_atoms(sheaf, tol=tol)
    n_modes = int(modes.shape[0])
    print(f"[auto_exp_62] kernel dim = {n_modes} harmonic modes "
          f"({len(atoms)} atoms have mass>1/F); tol={tol:.2e}; "
          f"spectrum [{eigvals.min():.2e}, {eigvals.max():.2e}]")

    # --- 4: Mapper-style graph + H¹ ---
    # Use the per-atom decoder direction at the LAST layer (the only one with
    # ambient dimension matching real L40) as the point cloud over which we
    # measure topology. For harmonic atoms whose meaning is shared across
    # layers, this captures the true feature geometry.
    if len(atoms) >= 3:
        W = saes[-1].W_dec.detach().cpu().numpy()[atoms]   # (n_harm, D_last)
        # normalize to unit vectors for cosine-style proximity
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=-1)
        # filtration cap = 25th percentile distance: a SPARSE 1-skeleton
        # where cycles are genuinely topological (not just metric noise).
        nz = D[D > 0]
        max_r = float(np.percentile(nz, 50)) if nz.size else 1.0
        # subsample atoms if too many — H¹ on 256 points is O(N³) for our
        # naive triangle-fill check; cap at 80 by uniform random subset.
        if W.shape[0] > 80:
            rng2 = np.random.default_rng(0)
            sub = rng2.choice(W.shape[0], 80, replace=False)
            W_sub = W[sub]
        else:
            W_sub = W
        pers_all = persistent_h1(W_sub, max_radius=max_r)
        # rank by persistence; keep top 20 only.
        pers_all.sort(key=lambda bd: -(bd[1] - bd[0]))
        pers = pers_all[:20]
    else:
        pers = []

    longest = pers[0] if pers else None
    print(f"[auto_exp_62] H¹ classes: {len(pers)}; longest={longest}")

    # --- plot persistence diagram ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        if pers:
            arr = np.array(pers)
            ax.scatter(arr[:, 0], arr[:, 1], s=30, c="C3", alpha=0.8)
            lim = float(arr.max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", lw=0.7)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title(f"H¹ persistence — {len(atoms)} harmonic atoms")
        fig.tight_layout()
        fig.savefig(OUT / "persistence_h1.png", dpi=130)
    except Exception as e:
        print(f"[auto_exp_62] plot skipped: {e}")

    # --- copy history figure for convenience ---
    src = TRAIN / "harmonic_curve.png"
    if src.exists():
        import shutil
        shutil.copy(src, OUT / "harmonic_curve.png")

    results = {
        "n_harmonic_modes": n_modes,
        "n_harmonic_atoms": int(len(atoms)),
        "laplacian_eigval_min": float(eigvals.min()),
        "laplacian_eigval_max": float(eigvals.max()),
        "h1_classes": pers,
        "longest_h1": longest,
        "interpretation": (
            "If longest H¹ persistence ≫ noise floor (median bd-gap), the hue "
            "ring survives at the cross-layer harmonic level — consistent with "
            "circular hue being a global circuit feature. Empty H¹ falsifies "
            "the prediction on this stand-in."
        ),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[auto_exp_62] wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
