"""ONE canonical held-out EV — own the definition, don't trust reported numbers.

Lead's integrity fix: recompute held-out EV for EVERY frontier point through a
single function with the TRAIN-mean TSS baseline, rather than trusting each lane's
reported number (T1's standalone frontier used the HELD-OUT column mean as its TSS
denominator — a different, non-comparable baseline; see its `_heldout_ev`). On the
Tier-0-standardized rows the train mean IS the origin (the per-dim train mean was
already subtracted), so the honest baseline is the origin.

Canonical definition (stated in REPORT_35B.md):
    EV = 1 − SSE / TSS,   SSE = ||X − recon||²,   TSS = ||X||²  (about the ORIGIN
    = train mean on Tier-0 rows), reconstruction = tied Top-K encode against the
    unit-norm decoder (project onto atoms, keep top-`active` by |projection|, sum) —
    the same standard TopK-SAE reconstruction T1 uses.

Every point (T1 TopK at each K, composed, baseline) goes through THIS function, so
fig1's absolute numbers are uniformly honest and "matches TopK at 0.X" is real. If a
lane's reported EV differs from this recompute, this recompute is authoritative for
the figure; the delta is reported so we can flag which number was off and by how much.

Run on MSI where the decoders + held-out matrix live:
  python canonical_ev.py --heldout L17_heldout.f32.npy --tier0 tier0.json \
      --decoders t1_run/ckpt --active 32 --out l17_t1_frontier_canonical.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np


def load_tier0(path: str):
    t = json.load(open(path))
    mean = np.asarray(t["per_dim_mean"], dtype=np.float64)
    scale = float(t.get("global_rms_scale") or t.get("global_rms") or 1.0)
    return mean, scale


def standardize_stream(X, mean, scale, chunk: int = 16384):
    """Yield Tier-0-standardized row chunks (x-mean)/scale without materializing all."""
    n = X.shape[0]
    for i in range(0, n, chunk):
        yield ((np.asarray(X[i:i + chunk], dtype=np.float64) - mean) / scale)


def tss_origin_and_colmean(X, mean, scale, chunk: int = 16384):
    """TSS about the origin (train mean) and about the held-out column mean, one pass."""
    n, d = X.shape
    ssq = 0.0
    colsum = np.zeros(d, dtype=np.float64)
    for xb in standardize_stream(X, mean, scale, chunk):
        ssq += float((xb * xb).sum())
        colsum += xb.sum(axis=0)
    colmean = colsum / n
    tss_origin = ssq
    tss_colmean = ssq - n * float((colmean * colmean).sum())
    return tss_origin, tss_colmean


def _unit_norm(D: np.ndarray, d: int) -> np.ndarray:
    """Return decoder as (K, d) with unit-norm atoms (rows). Orient by matching the
    ambient dimension `d` (= X.shape[1]) — robust when K > d (the real regime) where
    a shape-only heuristic would guess wrong."""
    if D.shape[1] == d:
        atoms = D  # already (K, d)
    elif D.shape[0] == d:
        atoms = D.T  # was (d, K)
    else:
        raise ValueError(f"decoder {D.shape} has no axis matching ambient dim {d}")
    nrm = np.linalg.norm(atoms, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return atoms / nrm


def sse_topk(X, mean, scale, D: np.ndarray, active: int,
             row_chunk: int = 4096) -> float:
    """SSE of the tied Top-K reconstruction against unit-norm decoder `D`, streamed
    over rows so a K=32k dictionary never blows memory (scores are per-chunk)."""
    atoms = _unit_norm(D, X.shape[1])  # (K, d)
    K = atoms.shape[0]
    a = min(active, K)
    sse = 0.0
    for xb in standardize_stream(X, mean, scale, row_chunk):
        scores = xb @ atoms.T                                   # (m, K)
        idx = np.argpartition(-np.abs(scores), a - 1, axis=1)[:, :a]
        rows = np.arange(xb.shape[0])[:, None]
        cod = scores[rows, idx]                                 # (m, a)
        recon = np.zeros_like(xb)
        for j in range(a):
            recon += cod[:, [j]] * atoms[idx[:, j]]
        diff = xb - recon
        sse += float((diff * diff).sum())
    return sse


def recompute_frontier(heldout: str, tier0: str, decoders_dir: str,
                       active: int, out: str) -> dict:
    mean, scale = load_tier0(tier0)
    X = np.load(heldout, mmap_mode="r")
    tss_o, tss_c = tss_origin_and_colmean(X, mean, scale)
    ratio = tss_c / tss_o
    rows = []
    for f in sorted(glob.glob(os.path.join(decoders_dir, "decoder_K*.npy"))):
        m = re.search(r"decoder_K(\d+)\.npy", os.path.basename(f))
        K = int(m.group(1)) if m else None
        D = np.load(f).astype(np.float64)
        sse = sse_topk(X, mean, scale, D, active)
        ev_origin = 1.0 - sse / tss_o          # CANONICAL (train-mean)
        ev_colmean = 1.0 - sse / tss_c          # what T1 reported (held-out colmean)
        rows.append({"K": K, "l0": active, "heldout_ev": ev_origin,
                     "heldout_ev_colmean_baseline": ev_colmean,
                     "sse": sse, "delta_vs_colmean": ev_origin - ev_colmean,
                     "decoder_shape": list(D.shape)})
        print(f"K={K}: EV_canonical(train-mean)={ev_origin:.5f}  "
              f"EV_colmean(T1-reported)={ev_colmean:.5f}  Δ={ev_origin-ev_colmean:+.5f}")
    out_obj = {
        "ev_baseline": "train_mean",
        "ev_definition": "1 - SSE/TSS, TSS about origin (train mean) on Tier-0 rows",
        "reconstruction": f"tied_topk_active{active}",
        "tss_origin": tss_o, "tss_heldout_colmean": tss_c,
        "ratio_colmean_over_origin": ratio,
        "authoritative": True, "source": "canonical_ev.py recompute",
        "frontier": rows,
    }
    Path(out).write_text(json.dumps(out_obj, indent=1))
    print(f"wrote {out}  ({len(rows)} K points; ratio colmean/origin={ratio:.4f})")
    return out_obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--tier0", required=True)
    ap.add_argument("--decoders", required=True, help="dir with decoder_K*.npy")
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--out", default="l17_t1_frontier_canonical.json")
    a = ap.parse_args()
    recompute_frontier(a.heldout, a.tier0, a.decoders, a.active, a.out)


if __name__ == "__main__":
    main()
