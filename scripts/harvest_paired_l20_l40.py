"""Dump paired (L_in, L_out) residual blocks for skip-transcoder training.

CLUSTER STATUS
--------------
The cogito server's /v1/encode endpoint exposes ONLY layer 40 (see
``runs/COLOR_COGITO_L20_L40_mini/abort.json`` — the L20 hook is not
registered and a server restart is out of scope under the no-cluster rule).
Cached on disk we therefore have only ``X_L40 ∈ R^(26572, 7168)``.

HONEST FALLBACK
---------------
Per the task spec, when L20 is not cached we synthesize a stand-in
"shallower" 4096-dim representation by PCA-projecting L40 down to its
top-K_in components:

    X_in  = (X_L40 - mu) @ V_K_in^T   ∈ R^(N, 4096)    # "L20 stand-in"
    Y_out =  X_L40                    ∈ R^(N, 7168)    # true L40

This is NOT a true L20 residual: it is a low-rank linear projection of L40.
The transcoder can therefore at best learn the *non-linear* lift from a
shallow PCA envelope back up to the full L40 manifold. The linear part
trivializes — the rank-r skip will absorb most of it, which actually makes
the comparison HARDER for the dictionary (it has to find genuinely nonlinear
structure, which is the regime where transcoders are claimed to beat SAEs).

This script writes ``runs/COGITO_PAIRED_L20_L40_STANDIN/paired.pt`` with
``{"X_in", "Y_out", "labels", "values", "hsv", "meta"}``. ``meta`` records
the fallback provenance for downstream readers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


COGITO_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COGITO_PAIRED_L20_L40_STANDIN")


def _hsv_from_xkcd_names(values: list[str]) -> torch.Tensor:
    """Cheap HSV decoder from xkcd color names → (N, 3) float tensor.

    Uses matplotlib's xkcd palette; falls back to grey for unknown names so
    the script never crashes on a stray label.
    """
    import matplotlib.colors as mcolors

    out = np.zeros((len(values), 3), dtype=np.float32)
    palette = mcolors.XKCD_COLORS
    for i, name in enumerate(values):
        key = f"xkcd:{name}".lower()
        rgb = mcolors.to_rgb(palette.get(key, "#808080"))
        out[i] = mcolors.rgb_to_hsv(rgb)
    return torch.from_numpy(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dim", type=int, default=4096,
                   help="stand-in 'L20' width (PCA truncation of L40)")
    p.add_argument("--n_rows", type=int, default=None,
                   help="cap rows for development; default = all")
    p.add_argument("--out", type=Path, default=OUT_DIR / "paired.pt")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    X_path = COGITO_DIR / "X_L40.npy"
    snap_path = COGITO_DIR / "X_L40.snapshot.npz"
    print(f"[harvest_paired] mmap-loading {X_path}")
    X_full = np.load(X_path, mmap_mode="r")                  # (N, 7168)
    n = X_full.shape[0] if args.n_rows is None else min(args.n_rows, X_full.shape[0])
    X = np.asarray(X_full[:n], dtype=np.float32)
    print(f"[harvest_paired] loaded X with shape {X.shape}")

    # Reuse the cached PCA basis whose width covers `in_dim`.
    # pca_basis_K128 has Vt ∈ R^(128, 7168); enough for in_dim ≤ 128.
    # For in_dim=4096 we need to compute a fresh truncated SVD on the
    # mean-centered X. Memory budget: float32 7168·4096·4B ≈ 117MB for Vt,
    # well within limits.
    if args.in_dim <= 128:
        pca_blob = np.load(COGITO_DIR / "pca_basis_K128.npz")
        mu = pca_blob["mu"].astype(np.float32)
        Vt = pca_blob["Vt"].astype(np.float32)[:args.in_dim]
        print(f"[harvest_paired] using cached PCA-128 basis truncated to {args.in_dim}")
    else:
        print(f"[harvest_paired] computing fresh top-{args.in_dim} SVD on N={n}")
        mu = X.mean(axis=0)
        Xc = X - mu
        # Economy SVD via torch on CPU (deterministic, no MPS shenanigans).
        U, S, Vh = torch.linalg.svd(torch.from_numpy(Xc), full_matrices=False)
        Vt = Vh[: args.in_dim].numpy().astype(np.float32)
        del U, S, Vh

    X_in = (X - mu) @ Vt.T                                   # (n, in_dim)
    Y_out = X                                                # (n, 7168)

    # Recover labels/values from the snapshot file (smaller, cheap to load).
    snap = np.load(snap_path, allow_pickle=True)
    snap_keys = list(snap.keys())
    labels = list(snap["labels"][:n]) if "labels" in snap_keys else [""] * n
    values = list(snap["values"][:n]) if "values" in snap_keys else [""] * n
    hsv = _hsv_from_xkcd_names(values)

    meta = {
        "provenance": "PCA-stand-in fallback (L20 hook not available on cogito server)",
        "abort_ref": str(COGITO_DIR.parent / "COLOR_COGITO_L20_L40_mini" / "abort.json"),
        "in_dim": int(args.in_dim),
        "out_dim": int(Y_out.shape[1]),
        "n_rows": int(n),
        "source_layer": 40,
        "method": "PCA-truncate(L40) → X_in;  X_L40 → Y_out",
    }
    blob = {
        "X_in": torch.from_numpy(X_in.astype(np.float32)),
        "Y_out": torch.from_numpy(Y_out.astype(np.float32)),
        "labels": labels,
        "values": values,
        "hsv": hsv,
        "meta": meta,
    }
    torch.save(blob, args.out)
    print(f"[harvest_paired] wrote {args.out}  ({n} rows, in_dim={args.in_dim}, out_dim={Y_out.shape[1]})")
    print(f"[harvest_paired] meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
