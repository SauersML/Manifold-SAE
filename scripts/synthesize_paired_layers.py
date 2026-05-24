"""Build a 3-layer paired-SAE training stack.

Honest stand-in
---------------
The real motivation for sheaf cohomology across layers is to glue
*independent* transformer layer SAEs via cross-layer transcoders. The
Crosscoder agent's ``synthesize_multilayer_cogito.py`` ships a 3-view stack
(L20/L30/L40 stand-ins) by random projection of L40. We prefer to reuse
that output if it exists; otherwise we build a 3-view PCA-projected stack
of L40 inline.

PCA views are NOT independent layers — they share information by
construction. But they suffice to (a) wire up the sheaf primitive
end-to-end, (b) verify ‖δs‖² decreases under training, and (c) check the
hue ring shows up in H¹ even on this stand-in. The real test waits for
re-harvested L20/L30 from cogito-Qwen-7B once cluster access returns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runs/COLOR_COGITO_L40/X_L40.npy"
PROJECTED = REPO / "runs/COLOR_COGITO_MULTILAYER"
OUT = REPO / "runs/SHEAF_PAIRED_3LAYER"


def build_pca_views(X: np.ndarray, dims: tuple[int, int, int], rng) -> list[np.ndarray]:
    # SVD on X to get a basis; take three nested PCA projections.
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use randomized SVD for speed at D=7168.
    from numpy.linalg import svd
    k = max(dims)
    # Project to k via Gaussian sketch first to keep this < a few seconds.
    G = rng.standard_normal((X.shape[1], k)).astype(np.float32) / np.sqrt(X.shape[1])
    Y = Xc @ G                                                   # (N, k)
    U, S, Vt = svd(Y, full_matrices=False)                       # cheap
    Z = U * S                                                    # (N, k) PCA scores
    return [Z[:, :d].astype(np.float32).copy() for d in dims]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xC0DE)

    # ----- Try Crosscoder agent's stack first -----
    cross_files = sorted(PROJECTED.glob("X_l*.npy"))
    if len(cross_files) >= 3:
        Xs = [np.load(p) for p in cross_files[:3]]
        provenance = f"reused Crosscoder stack at {PROJECTED}"
    else:
        if not SRC.exists():
            raise FileNotFoundError(f"Need {SRC} to build PCA stand-in")
        X = np.load(SRC, mmap_mode="r")[: 8000]                  # cap for speed
        X = np.ascontiguousarray(X)
        Xs = build_pca_views(X, dims=(256, 384, 512), rng=rng)
        provenance = (
            f"PCA stand-in from {SRC} (first 8000 rows); "
            f"three nested PCA scores at dims [256, 384, 512]."
        )

    for i, X in enumerate(Xs):
        np.save(OUT / f"X_layer{i}.npy", X)
    (OUT / "README.md").write_text(
        "# Sheaf paired-layer stack\n\n"
        f"Provenance: {provenance}\n\n"
        "STAND-IN: views share information; real test waits for L20/L30 re-harvest.\n"
        f"Shapes: {[X.shape for X in Xs]}\n"
    )
    print(f"[synthesize_paired_layers] wrote 3 layers to {OUT}")
    for i, X in enumerate(Xs):
        print(f"  layer {i}: shape={X.shape}  dtype={X.dtype}")


if __name__ == "__main__":
    main()
