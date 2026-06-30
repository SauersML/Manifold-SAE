"""Load the existing trained crosscoder and run JUST the HSV correlation step.

Avoids re-training (saves GPU lock contention) and produces the top-5
atoms by |HSV correlation| against the probes built in
``scripts/build_color_probes_l40.py``.

Saves: runs/crosscoder/hsv_correlation_topatoms.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from manifold_sae.crosscoder import Crosscoder


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "runs" / "COLOR_COGITO_MULTILAYER"
OUT = REPO / "runs" / "crosscoder"
PROBES = REPO / "runs" / "COLOR_COGITO_L40" / "color_manifold_probes_layer40.npz"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def main():
    device = _device()
    print(f"[hsv] device={device}")

    ckpt = torch.load(OUT / "crosscoder.pt", map_location=device, weights_only=False)
    model = Crosscoder(
        layer_dims=ckpt["config"]["layer_dims"],
        n_atoms=ckpt["config"]["n_atoms"],
        sparsity_weight=ckpt["config"]["sparsity_weight"],
        activation=ckpt["config"]["activation"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    F = ckpt["config"]["n_atoms"]
    print(f"[hsv] loaded crosscoder F={F} layers={ckpt['config']['layer_dims']}")

    probes = np.load(PROBES, allow_pickle=True)
    probe_dirs = probes["directions"].astype(np.float32)  # (15, 7168)
    probe_labels = [str(s) for s in probes["labels"]]
    print(f"[hsv] {len(probe_labels)} probes: {probe_labels}")

    # The crosscoder was trained on the synthetic MULTILAYER cache (X_l1, X_l2,
    # X_l3 with dims 2048, 4096, 7168) — NOT directly on real cogito-L40. So
    # the encoder takes concatenated [X_l1 | X_l2 | X_l3]. We replay it on a
    # random subsample.
    Xs_mm = [np.load(DATA / f"X_l{i}.npy", mmap_mode="r") for i in (1, 2, 3)]
    N = Xs_mm[0].shape[0]
    n_eval = min(8192, N)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)[:n_eval]
    # Normalize same as training.
    Xs_eval = []
    for Xn in Xs_mm:
        Xa = np.asarray(Xn[idx], dtype=np.float32)
        mu_full = np.asarray(Xn).astype(np.float32).mean(0, keepdims=True)
        sd_full = np.asarray(Xn).astype(np.float32).std(0, keepdims=True).clip(min=1e-6)
        Xs_eval.append(torch.from_numpy((Xa - mu_full) / sd_full))

    with torch.no_grad():
        x_cat = torch.cat([X.to(device) for X in Xs_eval], dim=-1)
        z = model.encode(x_cat).cpu().numpy()  # (n_eval, F)
    print(f"[hsv] z shape={z.shape} mean firing fraction={(z > 0).mean():.4f}")

    # L40 = X_l3 (width 7168).
    X3_eval = np.asarray(Xs_mm[2][idx], dtype=np.float32)
    probe_signals = X3_eval @ probe_dirs.T  # (n_eval, 15)

    # Correlate ALL F atoms × probes — then pick top 5 by max |corr|.
    corr = np.zeros((F, len(probe_labels)))
    for f in range(F):
        zf = z[:, f]
        if zf.std() < 1e-9:
            continue
        for j in range(len(probe_labels)):
            corr[f, j] = _corr(zf, probe_signals[:, j])
    abs_corr = np.abs(corr)
    best_per_atom = abs_corr.max(axis=1)
    best_label_per_atom = abs_corr.argmax(axis=1)

    top5 = np.argsort(-best_per_atom)[:5]
    out = {"top5_atoms": []}
    print("\n=== TOP-5 atoms by |HSV correlation| ===")
    print(f"{'atom':>6} {'label':>12} {'corr':>8}   all-labels")
    for a in top5:
        lbl = probe_labels[best_label_per_atom[a]]
        c = corr[a, best_label_per_atom[a]]
        print(f"{int(a):>6} {lbl:>12} {c:>+.4f}   "
              f"[" + ", ".join(f"{l}={corr[a,j]:+.2f}" for j,l in enumerate(probe_labels)) + "]")
        out["top5_atoms"].append({
            "atom": int(a),
            "best_label": lbl,
            "best_corr": float(c),
            "all_corrs": {l: float(corr[a, j]) for j, l in enumerate(probe_labels)},
        })

    # Restrict to HSV-only labels for a tighter answer.
    hsv_labels = ["hue_x", "hue_y", "sat", "val"]
    hsv_cols = [probe_labels.index(l) for l in hsv_labels if l in probe_labels]
    abs_corr_hsv = abs_corr[:, hsv_cols]
    best_hsv = abs_corr_hsv.max(axis=1)
    best_hsv_idx_in_hsvcols = abs_corr_hsv.argmax(axis=1)
    top5_hsv = np.argsort(-best_hsv)[:5]
    out["top5_atoms_hsv_only"] = []
    print("\n=== TOP-5 atoms by |HSV-only correlation| (hue_x/hue_y/sat/val) ===")
    for a in top5_hsv:
        local = best_hsv_idx_in_hsvcols[a]
        lbl = hsv_labels[local]
        c = corr[a, hsv_cols[local]]
        print(f"  atom {int(a):>4}  best={lbl} corr={c:+.4f}  "
              f"(hx={corr[a,hsv_cols[0]]:+.2f} hy={corr[a,hsv_cols[1]]:+.2f} "
              f"s={corr[a,hsv_cols[2]]:+.2f} v={corr[a,hsv_cols[3]]:+.2f})")
        out["top5_atoms_hsv_only"].append({
            "atom": int(a),
            "best_label": lbl,
            "best_corr": float(c),
        })

    with open(OUT / "hsv_correlation_topatoms.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[hsv] saved {OUT/'hsv_correlation_topatoms.json'}")


if __name__ == "__main__":
    main()
