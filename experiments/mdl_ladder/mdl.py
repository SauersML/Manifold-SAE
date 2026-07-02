"""MDL scorer for the featurizer ladder: directions -> blocks -> charts.

Goodfire's "Block-Sparse Featurizers" (BSF) argues a block code describes an
activation in fewer bits than a direction (TopK) code, with an MDL optimum at
block width b ~ 2-4. This module extends the same accounting one rung: a *chart*
(a curved atom with an intrinsic coordinate map Phi) pays for reconstruction with
its intrinsic dimension d_i rather than the block's extrinsic dimension b. For a
cyclic feature (a circle: b_ext = 2 extrinsic dims, d_i = 1 intrinsic angle) the
chart codes ONE number per firing where the block codes TWO -- at the price of
storing extra harmonic decoder vectors (Phi) in the dictionary.

Two-part (Rissanen) description length, in bits, over a corpus of N tokens on
which a feature fires f times:

    L_total = L_code + L_dict
    L_code  = f * ( sum_j 0.5*log2(1 + v_j/delta2)   [rate to code the m active
                                                       coefficients to distortion]
                    + log2 C(G, k) )                  [selection: which k of G atoms]
    L_dict  = P * L_param                             [P decoder scalars, L_param bits each]

    bits/token = L_total / N

`v_j` are the per-coordinate signal variances of the m coefficients the featurizer
emits (block: its b PCA eigenvalues; chart: the single angle coordinate's variance).
`delta2` is the per-token distortion floor (task-derived MSE, see README). The rate
term is the exact scalar Gaussian rate-distortion function 0.5*log2(1+SNR), which
reduces to the high-rate MDL form 0.5*log2(v/delta2) when v >> delta2. A featurizer
whose achieved residual exceeds delta2 cannot reach the floor with its m coords --
it is flagged `distortion_infeasible` (a direction cannot trace a circle at any rate).

This maps term-for-term onto gamfit's REML negative-log-evidence, which IS a
description length: divide the REML criterion by ln 2 for bits. See
gam/crates/gam-sae/src/manifold/construction.rs:6526
    v = loss.total() + extra_penalty_energy + 0.5*log_det - occam
  * loss.data_fit (0.5*||whiten(z - zhat)||^2)  ->  L_code data term (residual to floor)
  * assignment_sparsity                          ->  L_code selection bits (softmax/gate)
  * 0.5*log_det (0.5*log|X^T X + S|) - occam      ->  L_dict effective-parameter bits
The chart's L_dict is larger (extra Phi harmonics inflate log|H|); its L_code is
smaller (d_i < b coded coords). The crossover in f is where the trade flips.

DERIVATION.md gives the algebra; this file is the executable scorer. It runs on
(i) frontier_out artifacts, (ii) probe_out K=1 chart fits vs linear PC baselines,
and (iii) a JSON interface for the G-bsf / N-nursery lanes (see README.md).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LN2 = math.log(2.0)
HERE = Path(__file__).resolve().parent
EXPT = HERE.parent


# ---------------------------------------------------------------------------
# rate-distortion primitives
# ---------------------------------------------------------------------------

def scalar_rate_bits(signal_var: float, delta2: float) -> float:
    """Bits to code one Gaussian scalar of variance `signal_var` to MSE `delta2`.

    Exact rate-distortion R(D) = 0.5*log2(sigma^2/D) for sigma^2 > D, else 0.
    We use the numerically kind form 0.5*log2(1 + sigma^2/delta2) so it stays
    finite and >= 0 at low SNR; it agrees with 0.5*log2(sigma^2/delta2) to O(1)
    bit once sigma^2 >> delta2 (the regime the MDL high-rate form assumes)."""
    if delta2 <= 0:
        return float("inf")
    return 0.5 * math.log2(1.0 + max(0.0, signal_var) / delta2)


def selection_bits(g_dict: int, k_active: int) -> float:
    """log2 C(G, k): bits to name which k of G dictionary atoms fired.

    This is the pointer/selection cost shared by every featurizer in the ladder;
    it cancels when comparing featurizers at equal (G, k)."""
    if g_dict <= 0 or k_active <= 0:
        return 0.0
    k = min(k_active, g_dict)
    return math.log2(math.comb(g_dict, k))


def reverse_water_filling(eigs: Sequence[float], delta2: float) -> tuple[float, list[float]]:
    """Rate (bits/sample) of the optimal linear code of a Gaussian source with
    covariance eigenvalues `eigs`, coded to total MSE `delta2`. Returns
    (rate_bits, per_coordinate_bits). This is the block/direction featurizer's
    lower bound: it is what a *linear* code can achieve, so a chart that beats it
    is beating the best possible linear featurizer at that distortion."""
    eigs = sorted((float(e) for e in eigs), reverse=True)
    lo, hi = 0.0, max(eigs) if eigs else delta2
    for _ in range(200):  # bisect the water level theta
        theta = 0.5 * (lo + hi)
        dist = sum(min(e, theta) for e in eigs)
        if dist > delta2:
            hi = theta
        else:
            lo = theta
    theta = 0.5 * (lo + hi)
    per = [max(0.0, 0.5 * math.log2(e / theta)) for e in eigs]
    return sum(per), per


# ---------------------------------------------------------------------------
# featurizer description-length model
# ---------------------------------------------------------------------------

@dataclass
class Featurizer:
    """One rung of the ladder. `coded_var` are the per-coordinate signal variances
    of the m coefficients emitted per firing (m = len(coded_var)): a direction has
    m=1, a b-block m=b, a chart m=d_i. `n_params` is the DICTIONARY scalar count
    (decoder): direction b_ext=1 -> p; b-block -> b*p; circle chart -> n_basis*p.
    `total_var` V and `ev` fix the achieved residual (1-ev)*V for feasibility."""
    name: str
    kind: str                       # "direction" | "block" | "chart"
    coded_var: list[float]          # per active coefficient signal variance
    n_params: int                   # dictionary decoder scalars
    ev: float                       # explained variance fraction achieved
    total_var: float                # V, total per-token variance to reconstruct
    n_tokens: int                   # N corpus tokens
    n_firings: int                  # f firings of this feature
    g_dict: int = 1                 # dictionary size (selection bits)
    k_active: int = 1               # atoms active per firing

    @property
    def m(self) -> int:
        return len(self.coded_var)

    @property
    def residual(self) -> float:
        return (1.0 - self.ev) * self.total_var


def score(feat: Featurizer, delta2: float, l_param_bits: float | None = None) -> dict[str, Any]:
    """Bits/token description length for `feat` at per-token distortion floor `delta2`.

    l_param_bits: bits to store one dictionary scalar. Default = distortion-matched
    (a decoder weight is quantized to the same per-scalar precision as a code
    coefficient), i.e. the mean per-coefficient code rate. Pass a fixed value
    (e.g. 16 for fp16) to override."""
    code_coeff = sum(scalar_rate_bits(v, delta2) for v in feat.coded_var)
    sel = selection_bits(feat.g_dict, feat.k_active)
    code_per_firing = code_coeff + sel
    if l_param_bits is None:
        # distortion-matched dictionary precision: same bits/scalar as the code
        l_param_bits = (code_coeff / feat.m) if feat.m else scalar_rate_bits(feat.total_var, delta2)
    dict_bits = feat.n_params * l_param_bits
    code_total = code_per_firing * feat.n_firings
    total = code_total + dict_bits
    return {
        "name": feat.name,
        "kind": feat.kind,
        "coded_dim_m": feat.m,
        "code_bits_per_firing": round(code_per_firing, 4),
        "code_coeff_bits_per_firing": round(code_coeff, 4),
        "selection_bits_per_firing": round(sel, 4),
        "n_params": feat.n_params,
        "l_param_bits": round(l_param_bits, 4),
        "dict_bits": round(dict_bits, 2),
        "code_bits_total": round(code_total, 2),
        "total_bits": round(total, 2),
        "bits_per_token": round(total / feat.n_tokens, 4),
        "residual_achieved": round(feat.residual, 5),
        "distortion_floor": round(delta2, 5),
        "distortion_infeasible": bool(feat.residual > delta2 * 1.02),
    }


def crossover_firings(block: Featurizer, chart: Featurizer, delta2: float,
                      l_param_bits: float | None = None) -> dict[str, Any]:
    """f*: the firing count at which the chart's total DL drops below the block's.

    L_chart(f) - L_block(f) = f*(code_c - code_b) + (P_c - P_b)*L_param.
    Chart wins when f*(code_b - code_c) > (P_c - P_b)*L_param = Phi*L_param, i.e.
        f*  =  Phi * L_param / ( (b - d_i) * r )
    with Phi = P_c - P_b the extra dictionary scalars, r the per-freed-coordinate
    code rate. Under distortion-matched precision (L_param = r) this collapses to
    the SNR-independent  f* = Phi / (m_block - m_chart)."""
    code_b = sum(scalar_rate_bits(v, delta2) for v in block.coded_var)
    code_c = sum(scalar_rate_bits(v, delta2) for v in chart.coded_var)
    dcode = code_b - code_c                       # (b - d_i) * r, bits/firing freed
    phi = chart.n_params - block.n_params          # extra dictionary scalars
    r_per_coord = code_b / block.m if block.m else float("nan")
    if l_param_bits is None:
        l_param_bits = r_per_coord                 # distortion-matched
    fstar = (phi * l_param_bits / dcode) if dcode > 0 else float("inf")
    return {
        "block": block.name,
        "chart": chart.name,
        "delta_code_bits_per_firing": round(dcode, 4),
        "phi_extra_params": phi,
        "r_per_freed_coord_bits": round(r_per_coord, 4),
        "l_param_bits": round(l_param_bits, 4),
        "f_star": round(fstar, 2),
        "f_star_matched_precision": round(phi / dcode * (code_b / block.m), 2)
        if dcode > 0 else float("inf"),
        "f_star_matched_simple": round(phi / (block.m - chart.m), 2)
        if block.m != chart.m else float("inf"),
        "chart_wins_at_actual_f": bool(chart.n_firings >= fstar),
        "actual_firings": chart.n_firings,
    }


# ---------------------------------------------------------------------------
# JSON interface (for G-bsf / N-nursery lanes) -- see README.md
# ---------------------------------------------------------------------------

def featurizer_from_json(d: dict[str, Any]) -> Featurizer:
    """Build a Featurizer from a lane-supplied dict. Required keys:
        name, kind, n_params, total_var, n_tokens, n_firings
    Plus ONE of:
        coded_var: [v_1, ...]              (per-coefficient signal variances), or
        ev + coded_dim (+ optional coeff_var_total): we split ev*total_var across
                                                      coded_dim coords equally.
    Optional: g_dict, k_active, ev (if coded_var given, ev may be derived)."""
    total_var = float(d["total_var"])
    if "coded_var" in d:
        coded_var = [float(x) for x in d["coded_var"]]
        ev = float(d.get("ev", sum(coded_var) / total_var if total_var else 0.0))
    else:
        m = int(d["coded_dim"])
        ev = float(d["ev"])
        s = ev * total_var
        coded_var = [s / m] * m
    return Featurizer(
        name=str(d["name"]), kind=str(d.get("kind", "block")),
        coded_var=coded_var, n_params=int(d["n_params"]), ev=ev,
        total_var=total_var, n_tokens=int(d["n_tokens"]),
        n_firings=int(d["n_firings"]), g_dict=int(d.get("g_dict", 1)),
        k_active=int(d.get("k_active", 1)),
    )


def score_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a full ladder from JSON. `payload`:
        { "delta2": <float or null>,  # null -> use chart-residual convention
          "l_param_bits": <float or null>,
          "featurizers": [ {<featurizer>}, ... ],
          "block_name": "<name>", "chart_name": "<name>" }  # for crossover
    Returns { "rows": [...scored...], "crossover": {...} }."""
    feats = [featurizer_from_json(x) for x in payload["featurizers"]]
    by_name = {f.name: f for f in feats}
    delta2 = payload.get("delta2")
    if delta2 is None:
        # task-derived floor = residual of the best chart present, else best feat
        charts = [f for f in feats if f.kind == "chart"]
        ref = max(charts or feats, key=lambda f: f.ev)
        delta2 = ref.residual
    lp = payload.get("l_param_bits")
    rows = [score(f, delta2, lp) for f in feats]
    out: dict[str, Any] = {"delta2": round(delta2, 5), "rows": rows}
    bn, cn = payload.get("block_name"), payload.get("chart_name")
    if bn in by_name and cn in by_name:
        out["crossover"] = crossover_firings(by_name[bn], by_name[cn], delta2, lp)
    return out


# ---------------------------------------------------------------------------
# adapters for existing Manifold-SAE artifacts
# ---------------------------------------------------------------------------

def _demean_per_template(X: np.ndarray, tidx: np.ndarray) -> np.ndarray:
    Xd = X.copy()
    for t in np.unique(tidx):
        m = tidx == t
        Xd[m] -= Xd[m].mean(0, keepdims=True)
    return Xd


def _pca_eigs(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (S ** 2) / (X.shape[0] - 1)


def build_probe_ladder(npz_path: str | Path, layer: str, probe_json: str | Path,
                       set_name: str, reduce_dim: int = 16, n_basis: int = 4) -> dict:
    """Rescore a real curved-feature probe (weekday/month) in bits.

    Reconstructs the demeaned reduced-space PCA spectrum from the cached harvest,
    reads the fitted EV numbers from curved_feature_probes.json, and builds the
    direction / 2-block / circle-chart ladder. p = reduce_dim (decoder ambient)."""
    d = np.load(npz_path, allow_pickle=True)
    X = d[layer].astype(np.float64)
    Xd = _demean_per_template(X, d["template_idx"])
    # train-only-style reduction is irrelevant for the in-sample spectrum; use full.
    eigs_full = _pca_eigs(Xd)
    eigs = eigs_full[:reduce_dim]
    V = float(eigs.sum())                          # variance inside the reduced space
    lam1, lam2 = float(eigs[0]), float(eigs[1])
    n = int(X.shape[0])

    probe = json.loads(Path(probe_json).read_text())["results"][set_name]
    ev_chart = float(probe["insample_ev"]["curved"])
    ev_lin1 = float(probe["insample_ev"]["linear_L1"])
    ev_lin2 = float(probe["insample_ev"]["linear_L2"])
    p = reduce_dim

    direction = Featurizer(
        name=f"{set_name}:direction(1 PC)", kind="direction",
        coded_var=[lam1], n_params=1 * p, ev=ev_lin1, total_var=V,
        n_tokens=n, n_firings=n)
    block2 = Featurizer(
        name=f"{set_name}:block(2 PC)", kind="block",
        coded_var=[lam1, lam2], n_params=2 * p, ev=ev_lin2, total_var=V,
        n_tokens=n, n_firings=n)
    chart = Featurizer(
        name=f"{set_name}:circle-chart(1 coord)", kind="chart",
        coded_var=[ev_chart * V], n_params=n_basis * p, ev=ev_chart, total_var=V,
        n_tokens=n, n_firings=n)

    delta2 = chart.residual                        # task floor = chart's own residual
    rows = [score(f, delta2) for f in (direction, block2, chart)]
    cross = crossover_firings(block2, chart, delta2)
    return {
        "set": set_name, "layer": layer, "n_samples": n, "p": p, "n_basis": n_basis,
        "total_var_reduced": round(V, 4), "lambda1": round(lam1, 4),
        "lambda2": round(lam2, 4), "ev": {"direction": ev_lin1, "block2": ev_lin2,
        "chart": ev_chart}, "delta2": round(delta2, 5), "rows": rows,
        "crossover": cross,
    }


def build_synthetic_ladder(probe_json: str | Path, set_name: str,
                           reduce_dim: int = 16, n_basis: int = 4) -> dict:
    """High-SNR planted-circle ladder from synthetic_validation.json (clean 2-plane,
    tiny tail). Eigenvalues are backed out from the linear PC EVs:
        lambda1 = ev_lin1 * V,  lambda2 = (ev_lin2 - ev_lin1) * V,  V = 1 (normalized)."""
    probe = json.loads(Path(probe_json).read_text())["results"][set_name]
    ev_chart = float(probe["insample_ev"]["curved"])
    ev_lin1 = float(probe["insample_ev"]["linear_L1"])
    ev_lin2 = float(probe["insample_ev"]["linear_L2"])
    n = int(probe["n_samples"])
    V = 1.0
    lam1 = ev_lin1 * V
    lam2 = max(1e-6, (ev_lin2 - ev_lin1) * V)
    p = reduce_dim
    cyclic = bool(probe.get("cyclic", True))
    b_ext = 2 if cyclic else 1

    direction = Featurizer(f"{set_name}:direction(1 PC)", "direction", [lam1],
                           1 * p, ev_lin1, V, n, n)
    block2 = Featurizer(f"{set_name}:block({b_ext} PC)", "block",
                        [lam1, lam2][:b_ext], b_ext * p, ev_lin2, V, n, n)
    chart = Featurizer(f"{set_name}:chart(1 coord)", "chart", [ev_chart * V],
                       n_basis * p, ev_chart, V, n, n)
    delta2 = chart.residual
    rows = [score(f, delta2) for f in (direction, block2, chart)]
    cross = crossover_firings(block2, chart, delta2)
    return {"set": set_name, "synthetic": True, "cyclic": cyclic, "n_samples": n,
            "p": p, "n_basis": n_basis, "b_ext": b_ext,
            "ev": {"direction": ev_lin1, "block2": ev_lin2, "chart": ev_chart},
            "delta2": round(delta2, 5), "rows": rows, "crossover": cross}


def build_frontier_ladder(results_json: str | Path, reduce_dim: int = 9,
                          n_basis: int = 3) -> dict:
    """Rescore the planted real-shaped frontier (p=9, curved atom {1,cos,sin}).

    Uses the per-atom measured geometry: a straight top-1 atom (linear_top1_ev) vs
    the circle chart (curved_ev) on the same planted circle block. Aggregated over
    the 3 planted-curved atoms. Block eigenvalues from the top-1/top-2 EV split."""
    R = json.loads(Path(results_json).read_text())
    atoms = [a for a in R["curvature_frontier"]["atoms"] if a["planted_curved"]]
    p = reduce_dim
    b_ext = 2
    rows_all, crosses = [], []
    for a in atoms:
        ev1 = float(a["linear_top1_ev"])
        evc = float(a["curved_ev"])
        n = int(a["n_points"])
        V = 1.0
        lam1 = ev1 * V
        # a clean planted circle fills a 2-plane: top-2 ~ curved ceiling
        ev2 = min(0.999, evc)
        lam2 = max(1e-6, (ev2 - ev1) * V)
        direction = Featurizer(f"atom{a['atom']}:direction", "direction", [lam1],
                               1 * p, ev1, V, n, n)
        block2 = Featurizer(f"atom{a['atom']}:block(2)", "block", [lam1, lam2],
                            b_ext * p, ev2, V, n, n)
        chart = Featurizer(f"atom{a['atom']}:chart", "chart", [evc * V],
                           n_basis * p, evc, V, n, n)
        delta2 = chart.residual
        rows_all.append({"atom": a["atom"],
                         "scored": [score(f, delta2) for f in (direction, block2, chart)]})
        crosses.append(crossover_firings(block2, chart, delta2))
    return {"source": "frontier_out (planted real-shaped synthetic)", "p": p,
            "n_basis": n_basis, "atoms": rows_all,
            "crossover_per_atom": crosses,
            "f_star_mean": round(float(np.mean([c["f_star"] for c in crosses])), 2)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="MDL ladder scorer (bits/token).")
    ap.add_argument("--json", type=str, help="score a lane JSON payload (stdin if '-')")
    ap.add_argument("--probes", action="store_true", help="score real weekday/month probes")
    ap.add_argument("--synthetic", action="store_true", help="score synthetic planted circles")
    ap.add_argument("--frontier", action="store_true", help="score frontier_out")
    ap.add_argument("--out", type=str, default=None, help="write results.json here")
    args = ap.parse_args()

    result: dict[str, Any] = {}
    if args.json:
        text = Path(args.json).read_text() if args.json != "-" else __import__("sys").stdin.read()
        print(json.dumps(score_json(json.loads(text)), indent=2))
        return
    if args.probes:
        result["real_probes"] = [
            build_probe_ladder(EXPT / "probe_out/harvest_weekday.npz", "L14",
                               EXPT / "probe_out/curved_feature_probes.json", "weekday"),
            build_probe_ladder(EXPT / "probe_out/harvest_month.npz", "L8",
                               EXPT / "probe_out/curved_feature_probes.json", "month"),
        ]
    if args.synthetic:
        sj = EXPT / "probe_out/synthetic_validation.json"
        result["synthetic"] = [build_synthetic_ladder(sj, s)
                               for s in ("weekday", "month", "year")]
    if args.frontier:
        result["frontier"] = build_frontier_ladder(EXPT / "frontier_out/results.json")

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
