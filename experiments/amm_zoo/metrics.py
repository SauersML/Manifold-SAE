"""Per-factor, Hungarian-matched scoring for the AMM zoo.

An arm produces a list of :class:`RecoveredFactor`s (each a recovered additive
contribution + intrinsic coordinate + topology/dim guess). We match recovered ->
true factors by the assignment that MAXIMISES total contribution R² (the
replication number), then report, per matched pair:

  * **contribution R²** — fraction of the true factor's contribution variance the
    matched recovered factor captures (the paper's replication metric);
  * **coordinate circular-correlation** — recovered vs true angle (circle/arc);
  * **geodesic-Spearman** — Spearman rank corr between recovered-code Euclidean
    distances and TRUE geodesic distances (does the code respect the manifold's
    geometry, topology-agnostically);
  * **topology-ID accuracy** — did the arm name the right topology?
  * **dimension-estimation accuracy** — did it estimate the right intrinsic dim?

Plus **MDL bits/token** per arm via ``mdl_ladder.score_json``.

No SciPy on the box, so :func:`linear_sum_assignment` is a compact O(n³)
Hungarian (validated against brute force in ``__main__``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Recovered-factor contract (arms fill this in)
# --------------------------------------------------------------------------- #
@dataclass
class RecoveredFactor:
    """One factor an arm recovered, in a form the scorer can match to truth.

    contribution: (n, d) additive reconstruction contribution on the scored split
                  (0 on tokens where this factor is inactive).
    coord:        (n, r_rec) recovered intrinsic coordinate/code (NaN where the
                  factor is inactive); used for circular-corr + geodesic-Spearman.
    active:       (n,) bool — tokens where this factor fired.
    topology:     the arm's topology guess ("circle"|"arc"|"torus"|"sphere"|"linear").
    intrinsic_dim: the arm's intrinsic-dimension estimate.
    n_params:     decoder scalars for this factor (MDL n_params).
    name:         label for reporting.
    """

    contribution: np.ndarray
    coord: np.ndarray
    active: np.ndarray
    topology: str
    intrinsic_dim: int
    n_params: int
    name: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Hungarian assignment (minimise cost) — compact, no SciPy.
# --------------------------------------------------------------------------- #
def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment. Returns ``(row_ind, col_ind)``. Pads to square
    with the max cost so rectangular inputs assign every smaller-side row."""
    c = np.asarray(cost, dtype=np.float64)
    r, cc = c.shape
    n = max(r, cc)
    pad = np.full((n, n), c.max() + 1.0 if c.size else 1.0, dtype=np.float64)
    pad[:r, :cc] = c
    row_ind, col_ind = _hungarian_square(pad)
    keep = [(i, j) for i, j in zip(row_ind, col_ind) if i < r and j < cc]
    ri = np.array([i for i, _ in keep], dtype=int)
    ci = np.array([j for _, j in keep], dtype=int)
    return ri, ci


def _hungarian_square(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kuhn–Munkres on a square matrix (O(n³)); returns row->col assignment."""
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)  # p[j] = row assigned to column j
    way = np.zeros(n + 1, dtype=int)
    INF = float("inf")
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    row_of_col = p
    col_ind = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        if row_of_col[j] != 0:
            col_ind[row_of_col[j] - 1] = j - 1
    return np.arange(n), col_ind


# --------------------------------------------------------------------------- #
# Correlation helpers
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    return _pearson(_rankdata(a), _rankdata(b))


def _circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka–SenGupta circular correlation (abs), robust to phase/offset."""
    if len(a) < 3:
        return 0.0
    a0, b0 = a - _circular_mean(a), b - _circular_mean(b)
    num = float((np.sin(a0) * np.sin(b0)).sum())
    den = float(np.sqrt((np.sin(a0) ** 2).sum() * (np.sin(b0) ** 2).sum()))
    return abs(num / den) if den > 0 else 0.0


def _to_angle(coord: np.ndarray) -> np.ndarray:
    """Best 1-D angle from a recovered coordinate: use atan2 of the first two
    columns when >=2-D (a b=2 block/chart code), else the column itself."""
    coord = np.asarray(coord, dtype=np.float64)
    if coord.ndim == 1:
        return coord
    if coord.shape[1] >= 2:
        return np.arctan2(coord[:, 1], coord[:, 0])
    return coord[:, 0]


# --------------------------------------------------------------------------- #
# Geodesic distances (true side) — pairwise on a subsample.
# --------------------------------------------------------------------------- #
def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def geodesic_pairwise(coord: np.ndarray, topology: str) -> np.ndarray:
    """Pairwise geodesic distance matrix ``(m, m)`` for true intrinsic coords."""
    m = coord.shape[0]
    if topology in ("circle",):
        th = coord[:, 0]
        return np.abs(_wrap_pi(th[:, None] - th[None, :]))
    if topology == "arc":
        th = coord[:, 0]
        return np.abs(th[:, None] - th[None, :])  # open arc: no wraparound
    if topology == "torus":
        d1 = _wrap_pi(coord[:, 0][:, None] - coord[:, 0][None, :])
        d2 = _wrap_pi(coord[:, 1][:, None] - coord[:, 1][None, :])
        return np.sqrt(d1 * d1 + d2 * d2)
    if topology == "sphere":
        phi, lam = coord[:, 0], coord[:, 1]
        xyz = np.stack(
            [np.sin(phi) * np.cos(lam), np.sin(phi) * np.sin(lam), np.cos(phi)], axis=1
        )
        dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
        return np.arccos(dot)
    # linear (or fallback): Euclidean in coords.
    diff = coord[:, None, :] - coord[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def geodesic_spearman(
    true_coord: np.ndarray,
    true_topology: str,
    rec_coord: np.ndarray,
    rng: np.random.Generator,
    n_sample: int = 1200,
) -> float:
    """Spearman rank corr between recovered-code Euclidean distances and TRUE
    geodesic distances on a random subsample of shared active tokens — does the
    recovered code respect the manifold's geometry, independent of parametrisation."""
    m = true_coord.shape[0]
    if m < 5:
        return 0.0
    idx = np.arange(m) if m <= n_sample else rng.choice(m, n_sample, replace=False)
    d_true = geodesic_pairwise(true_coord[idx], true_topology)
    rc = np.atleast_2d(rec_coord[idx])
    if rc.shape[0] == 1:
        rc = rc.T
    diff = rc[:, None, :] - rc[None, :, :]
    d_rec = np.sqrt((diff * diff).sum(-1))
    iu = np.triu_indices(len(idx), k=1)
    return spearman(d_true[iu], d_rec[iu])


# --------------------------------------------------------------------------- #
# Contribution R² matrix + matching
# --------------------------------------------------------------------------- #
def _r2(true_c: np.ndarray, rec_c: np.ndarray) -> float:
    sst = float(((true_c - true_c.mean(0)) ** 2).sum())
    if sst <= 1e-12:
        return 0.0
    sse = float(((true_c - rec_c) ** 2).sum())
    return 1.0 - sse / sst


def contribution_r2_matrix(
    dataset, recovered: list[RecoveredFactor], split: str
) -> np.ndarray:
    """``(n_rec, n_true)`` matrix of contribution R² between every recovered and
    true factor. Memory-safe: streams one (true, rec) pair at a time."""
    n_true = dataset.G
    true_contribs = [dataset.contribution(split, g) for g in range(n_true)]
    n_rec = len(recovered)
    mat = np.zeros((n_rec, n_true), dtype=np.float64)
    for i, rf in enumerate(recovered):
        rc = rf.contribution
        for j in range(n_true):
            mat[i, j] = _r2(true_contribs[j], rc)
    return mat


def score_arm(
    dataset,
    recovered: list[RecoveredFactor],
    split: str = "test",
    *,
    seed: int = 0,
    delta2: float | None = None,
) -> dict[str, Any]:
    """Hungarian-match ``recovered`` to the dataset's true factors and report the
    full per-factor + aggregate metric suite (all on the held-out ``split``).

    ``delta2`` (the MDL per-token distortion floor) defaults to the dataset's
    irreducible noise energy ``d·σ²`` — the task-derived floor "reconstruct to the
    fidelity the noise allows, not beyond", so an arm that fits noise pays for it
    rather than getting infinite credit for zero residual."""
    if delta2 is None:
        delta2 = float(dataset.d * dataset.sigma ** 2)
    rng = np.random.default_rng(seed)
    r2m = contribution_r2_matrix(dataset, recovered, split)
    # Maximise total R² == minimise -R².
    ri, ci = linear_sum_assignment(-r2m)

    per_factor: list[dict[str, Any]] = []
    matched_true = set()
    for i, j in zip(ri.tolist(), ci.tolist()):
        rf = recovered[i]
        tf = dataset.factors[j]
        matched_true.add(j)
        rows, tcoord = dataset.true_intrinsic(split, j)
        # Recovered coord on the SAME true-active rows (recovered code where it
        # fired; NaN rows dropped by taking the true-active support).
        rc_rows = rf.coord[rows]
        valid = np.all(np.isfinite(rc_rows), axis=tuple(range(1, rc_rows.ndim))) if rc_rows.ndim > 1 else np.isfinite(rc_rows)
        rows_v = rows[valid]
        tcoord_v = tcoord[valid]
        rc_v = rf.coord[rows_v]
        rec = {
            "recovered": rf.name or f"rec{i}",
            "true_factor": j,
            "true_topology": tf.topology,
            "contribution_r2": round(float(r2m[i, j]), 4),
            "topology_guess": rf.topology,
            "topology_correct": bool(rf.topology == tf.topology),
            "dim_guess": int(rf.intrinsic_dim),
            "dim_correct": bool(rf.intrinsic_dim == tf.intrinsic_dim),
            "n_matched_tokens": int(rows_v.size),
        }
        if rows_v.size >= 5:
            rec["geodesic_spearman"] = round(
                geodesic_spearman(tcoord_v, tf.topology, rc_v, rng), 4
            )
            if tf.topology in ("circle", "arc"):
                rec["circular_corr"] = round(
                    circular_corr(_to_angle(rc_v), tcoord_v[:, 0]), 4
                )
        per_factor.append(rec)

    # Aggregates.
    by_topo: dict[str, list[dict]] = {}
    for rec in per_factor:
        by_topo.setdefault(rec["true_topology"], []).append(rec)

    def _mean(key: str, recs: list[dict]) -> float | None:
        vals = [r[key] for r in recs if key in r]
        return round(float(np.mean(vals)), 4) if vals else None

    topo_summary = {
        t: {
            "n": len(recs),
            "mean_contribution_r2": _mean("contribution_r2", recs),
            "topology_id_accuracy": round(
                float(np.mean([r["topology_correct"] for r in recs])), 4
            ),
            "dim_accuracy": round(float(np.mean([r["dim_correct"] for r in recs])), 4),
            "mean_circular_corr": _mean("circular_corr", recs),
            "mean_geodesic_spearman": _mean("geodesic_spearman", recs),
        }
        for t, recs in sorted(by_topo.items())
    }

    overall = {
        "mean_contribution_r2": round(float(np.mean([r["contribution_r2"] for r in per_factor])), 4),
        "topology_id_accuracy": round(float(np.mean([r["topology_correct"] for r in per_factor])), 4),
        "dim_accuracy": round(float(np.mean([r["dim_correct"] for r in per_factor])), 4),
        "n_recovered": len(recovered),
        "n_true": dataset.G,
        "n_matched": len(per_factor),
    }

    mdl = _score_mdl(dataset, recovered, per_factor, split, delta2)

    return {
        "split": split,
        "overall": overall,
        "by_topology": topo_summary,
        "per_factor": per_factor,
        "mdl": mdl,
    }


# --------------------------------------------------------------------------- #
# MDL bits/token via the mdl_ladder scorer.
# --------------------------------------------------------------------------- #
def _score_mdl(dataset, recovered, per_factor, split, delta2) -> dict | None:
    try:
        sys.path.insert(0, str(_HERE.parent / "mdl_ladder"))
        import mdl  # noqa: E402
    except Exception:
        return None
    n_tokens = dataset.test.n if split == "test" else dataset.train.n
    featurizers = []
    for i, rf in enumerate(recovered):
        n_fire = int(rf.active.sum())
        if n_fire == 0:
            continue
        # Signal variance the factor's code carries, and the EV it achieves — in
        # the ambient contribution space (consistent across arms).
        c = rf.contribution
        total_var = float((c ** 2).sum() / max(n_fire, 1))
        # ev vs its best-matched true factor (from per_factor if present).
        ev = 0.0
        for rec in per_factor:
            if rec["recovered"] == (rf.name or f"rec{i}"):
                ev = max(rec["contribution_r2"], 0.0)
                break
        featurizers.append(
            {
                "name": rf.name or f"rec{i}",
                "kind": {"linear": "block", "circle": "chart", "arc": "chart"}.get(
                    rf.topology, "block"
                ),
                "total_var": max(total_var, 1e-9),
                "n_tokens": int(n_tokens),
                "n_firings": n_fire,
                "n_params": int(rf.n_params),
                "coded_dim": max(int(rf.intrinsic_dim), 1),
                "ev": max(ev, 1e-6),
            }
        )
    if not featurizers:
        return None
    try:
        resp = mdl.score_json(
            {"delta2": delta2, "l_param_bits": None, "featurizers": featurizers}
        )
        total_bits = sum(row.get("bits_per_token", 0.0) for row in resp.get("rows", []))
        return {"bits_per_token_total": round(float(total_bits), 4), "rows": resp.get("rows", [])}
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


# --------------------------------------------------------------------------- #
# Self-test: Hungarian vs brute force + an oracle arm scores ~perfect.
# --------------------------------------------------------------------------- #
def _oracle_recovered(dataset, split: str) -> list[RecoveredFactor]:
    """A perfect arm: recovers each true factor exactly (contribution, coord,
    topology, dim). Scores contribution R²≈1, topology-ID 100%, dim 100%."""
    out = []
    sp = dataset.train if split == "train" else dataset.test
    for g, f in enumerate(dataset.factors):
        contrib = dataset.contribution(split, g)
        coord = np.full((sp.n, f.intrinsic_dim), np.nan)
        rows = np.nonzero(sp.active[:, g])[0]
        coord[rows] = sp.coords[rows, g, : f.intrinsic_dim]
        out.append(
            RecoveredFactor(
                contribution=contrib,
                coord=coord,
                active=sp.active[:, g].copy(),
                topology=f.topology,
                intrinsic_dim=f.intrinsic_dim,
                n_params=f.block_dim * dataset.d,
                name=f"oracle{g}",
            )
        )
    return out


if __name__ == "__main__":
    import itertools

    # Hungarian vs brute force on random 5x5 / 6x6 matrices.
    rng = np.random.default_rng(0)
    for n in (4, 5, 6):
        for _ in range(50):
            c = rng.random((n, n))
            ri, ci = linear_sum_assignment(c)
            got = c[ri, ci].sum()
            best = min(sum(c[i, perm[i]] for i in range(n)) for perm in itertools.permutations(range(n)))
            assert abs(got - best) < 1e-9, (n, got, best)
    # rectangular
    c = rng.random((3, 5))
    ri, ci = linear_sum_assignment(c)
    assert len(ri) == 3 and len(set(ci.tolist())) == 3
    print("Hungarian: OK (optimal vs brute force, rectangular)")

    from amm import generate_amm

    ds = generate_amm(seed=0, sigma_frac=0.02, n_train=1500, n_test=600)
    oracle = _oracle_recovered(ds, "test")
    rep = score_arm(ds, oracle, "test", seed=0)
    ov = rep["overall"]
    print(f"oracle overall: R2={ov['mean_contribution_r2']} "
          f"topoID={ov['topology_id_accuracy']} dim={ov['dim_accuracy']}")
    print(f"oracle circular (circle): {rep['by_topology']['circle']['mean_circular_corr']}, "
          f"geodesic-Spearman (sphere): {rep['by_topology']['sphere']['mean_geodesic_spearman']}")
    print(f"oracle MDL bits/token total: {rep['mdl']['bits_per_token_total'] if rep['mdl'] else None}")
    assert ov["mean_contribution_r2"] > 0.99, ov
    assert ov["topology_id_accuracy"] == 1.0 and ov["dim_accuracy"] == 1.0
    print("Oracle scores ~perfect: OK")
