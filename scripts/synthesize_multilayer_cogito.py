"""Synthesize a 3-layer cogito stack from cached L40 via random Gaussian projection.

CAVEAT (honest)
---------------
The real motivation for a crosscoder is to find features that appear at
multiple TRANSFORMER layers (e.g. L20 → L30 → L40). Cluster access is
currently revoked, so we cannot re-harvest L20/L30 from cogito-Qwen-7B.

This script builds a STAND-IN: three views of L40 projected via fixed
Gaussian matrices to widths {2048, 4096, 7168}. Because the projections are
fixed, a feature in L40 will of course show up in all three views — but
that's the POINT for testing the crosscoder primitive end-to-end. When
cluster access returns, swap the three np.load() calls in
``train_crosscoder.py`` for the real per-layer caches.

Output
------
runs/COLOR_COGITO_MULTILAYER/
    X_l1.npy   (N, 2048) — "L20 stand-in"
    X_l2.npy   (N, 4096) — "L30 stand-in"
    X_l3.npy   (N, 7168) — "L40 — actual"
    README.md  — honesty note
    projections.npz — the random matrices, for reproducibility
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runs/COLOR_COGITO_L40/X_L40.npy"
OUT = REPO / "runs/COLOR_COGITO_MULTILAYER"
SEED = 0xC0DE


def gaussian_projection(D_in: int, D_out: int, rng: np.random.Generator) -> np.ndarray:
    """Standard JL-style random projection, variance 1/D_in for norm preservation."""
    return rng.standard_normal((D_in, D_out)).astype(np.float32) / np.sqrt(D_in)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(SRC, mmap_mode="r")  # (N, 7168)
    N, D = X.shape
    print(f"[synthesize] loaded {SRC}: shape={X.shape} dtype={X.dtype}")

    rng = np.random.default_rng(SEED)
    target_dims = [2048, 4096, 7168]
    projs: dict[str, np.ndarray] = {}

    # Layer 3 is the real L40 — no projection.
    for i, D_out in enumerate(target_dims, start=1):
        name = f"X_l{i}.npy"
        if D_out == D:
            out = np.ascontiguousarray(X[:])  # copy out of mmap
            print(f"[synthesize] l{i} = identity ({D}-dim, real L40)")
        else:
            P = gaussian_projection(D, D_out, rng)
            projs[f"P_l{i}"] = P
            # Project in chunks to keep RAM bounded.
            out = np.empty((N, D_out), dtype=np.float32)
            chunk = 4096
            for s in range(0, N, chunk):
                out[s:s + chunk] = X[s:s + chunk] @ P
            print(f"[synthesize] l{i} = X @ P  ({D}→{D_out})")
        np.save(OUT / name, out)
        print(f"[synthesize] wrote {OUT / name} shape={out.shape}")

    if projs:
        np.savez(OUT / "projections.npz", **projs)
        print(f"[synthesize] wrote {OUT / 'projections.npz'}")

    readme = OUT / "README.md"
    readme.write_text(
        "# COLOR_COGITO_MULTILAYER — synthetic 3-layer stand-in\n\n"
        "These three arrays are NOT independent layer harvests. They are\n"
        "fixed Gaussian projections of `runs/COLOR_COGITO_L40/X_L40.npy` to\n"
        "widths {2048, 4096, 7168}. Used as a stand-in to develop the\n"
        "crosscoder primitive end-to-end while cluster access is revoked.\n\n"
        f"Source: `{SRC.relative_to(REPO)}`\n"
        f"Seed:   `{SEED:#x}`\n"
        "Shapes: X_l1 = (N, 2048), X_l2 = (N, 4096), X_l3 = (N, 7168)\n\n"
        "When cluster access returns, replace these with real L20/L30/L40\n"
        "harvests and rerun `scripts/train_crosscoder.py`.\n"
    )
    print(f"[synthesize] wrote {readme}")


if __name__ == "__main__":
    main()
