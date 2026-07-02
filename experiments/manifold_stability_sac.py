"""W9 — manifold-stability half, on SAC (Sequential Atom Composition) dictionaries.

The original ``seed_stability.py`` could only fill the *linear* half of the W9 claim:
``gamfit.sae_manifold_fit`` (the joint K-atom fit) did not converge on real data in
that build (the #1784 co-collapse), so the curved manifold-SAE arm was recorded as
un-producible and only a random-init "tiling" linear SAE demonstrated the claim's
SHAPE (atoms seed-unstable, spanned subspaces seed-stable).

SAC removes that blocker: it builds K from certified K=1 fits, which DO converge on
real data. So we can now fit a genuine curved dictionary twice under two seeds and
measure the thing the manifold-SAE thesis is actually about:

    latent-level agreement  (Hungarian match of individual atom frames)   -- expected LOW
    subspace-level agreement (principal-angle cos of the spanned frames)  -- expected HIGH
    artifact content-hash    (canonical dictionary hash)                   -- exact-repro check

Seed variation enters exactly where SAC has freedom: the data **row order / subsample**
(``random_state`` + a per-seed row permutation), which changes the residual-PCA seed of
each birth. If the manifold-SAE stability claim holds, the two seeds' individual atoms
differ (latent) while the union subspace — and, on planted data, each per-chart plane —
recurs (subspace ≫ latent).

Datasets:
  * ``planted`` — n planted 1-D circle manifolds in orthogonal planes (ground-truth
    planes known, so per-chart subspace agreement is measurable).
  * ``w6`` — the real (500,128) top-128-PCA OLMo-3-7B slab (``/dev/shm/w6/cache_K8.npy``)
    that the joint K=8 fit timed out on; SAC composes it atom-by-atom.

Reuses ``seed_stability.py`` verbatim for the matching/subspace/hash math, and
``examples/sac_prototype.sac_fit`` for the composition. Run on node2 (the K=1 outer
loop oscillates near the incumbent on the CPU-starved laptop).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_GAM_EXAMPLES = Path(os.environ.get("GAM_EXAMPLES", "/models/sauers_build/gam_fable/examples"))
for _p in (_GAM_EXAMPLES, Path("/Users/user/gam/examples")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from seed_stability import (  # noqa: E402
    latent_match, union_subspace, per_chart_subspace,
    canonical_dictionary_artifact, COS_THRESH,
)
from sac_prototype import sac_fit  # noqa: E402

OUT_DIR = Path(os.environ.get("W9_OUT", str(_HERE / "stability_sac_out")))
SEEDS = tuple(int(s) for s in os.environ.get("W9_SEEDS", "0,1").split(","))
W6_CACHE = os.environ.get("W6_CACHE", "/dev/shm/w6/cache_K8.npy")


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def data_planted(N=1500, p=32, n_circ=3, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    planes = [Q[:, 2 * g:2 * g + 2] for g in range(n_circ)]
    assign = rng.integers(0, n_circ, N)
    phase = rng.uniform(0, 2 * np.pi, N)
    X = np.zeros((N, p), dtype=np.float64)
    for i in range(N):
        X[i] = planes[assign[i]] @ np.array([np.cos(phase[i]), np.sin(phase[i])])
    X += noise * rng.standard_normal((N, p))
    return np.ascontiguousarray(X.astype(np.float64)), np.stack(planes, 0)


def data_w6():
    if not os.path.exists(W6_CACHE):
        return None, None
    X = np.ascontiguousarray(np.load(W6_CACHE), dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    X /= X.std() + 1e-8
    return X, None


# --------------------------------------------------------------------------- #
# Extract per-atom decoder frames from a SAC dictionary                        #
# --------------------------------------------------------------------------- #
def sac_blocks(result):
    """Per-atom decoder frame blocks (p, d), topologies, and a flat unit direction.

    ``SacAtom.fit`` is a K=1 ``sae_manifold_fit`` result; its single atom's
    ``decoder_coefficients`` is the frame the seed_stability hashing/subspace math
    consumes (same access path as ``seed_stability._manifold_worker``).
    """
    blocks, topos, flat = [], [], []
    for atom in result.atoms:
        a0 = atom.fit.atoms[0]
        B = np.asarray(a0.decoder_coefficients, dtype=np.float64)  # (p, d) or (d, p)
        if B.ndim == 1:
            B = B[:, None]
        if B.shape[0] < B.shape[1]:      # ensure (p, d) with p = model dim
            B = B.T
        blocks.append(B)
        topos.append(str(getattr(a0, "basis", atom.topology)))
        v = B.reshape(-1)
        flat.append(v / (np.linalg.norm(v) + 1e-12))
    return blocks, topos, np.asarray(flat)


def frame_columns(blocks):
    """Stack all atom frame columns into a (sum_d, p) row set for union-subspace."""
    cols = []
    for B in blocks:
        for j in range(B.shape[1]):
            c = B[:, j]
            cols.append(c / (np.linalg.norm(c) + 1e-12))
    return np.asarray(cols)


def analyze(name, X, planes, seeds, sac_kwargs):
    per_seed = {}
    for s in seeds:
        perm = np.random.default_rng(1000 + s).permutation(X.shape[0])
        res = sac_fit(np.ascontiguousarray(X[perm]), random_state=s, verbose=False,
                      **sac_kwargs)
        blocks, topos, flat = sac_blocks(res)
        art = canonical_dictionary_artifact(
            blocks, topos, ["O(2): origin rotation + reflection"] * len(blocks),
            "python-port/v1")
        per_seed[s] = dict(blocks=blocks, topos=topos, flat=flat,
                           k=res.k, combined_ev=float(res.combined_ev),
                           ev_gain=float(res.ev_gain),
                           delta_ev=[float(a.delta_ev) for a in res.atoms],
                           content_hash=art["content_hash"])
        print(f"[{name}] seed {s}: K={res.k} combined_ev={res.combined_ev:.4f} "
              f"hash={art['content_hash'][:12]}", flush=True)

    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    lat, uni, chart, hasheq = [], [], [], []
    for a, b in pairs:
        fa, fb = per_seed[a]["flat"], per_seed[b]["flat"]
        kmin = min(len(fa), len(fb))
        lat.append(latent_match(fa[:kmin], fb[:kmin]))
        uni.append(union_subspace(frame_columns(per_seed[a]["blocks"]),
                                  frame_columns(per_seed[b]["blocks"])))
        if planes is not None:
            chart.append(per_chart_subspace(frame_columns(per_seed[a]["blocks"]),
                                            frame_columns(per_seed[b]["blocks"]), planes))
        hasheq.append(per_seed[a]["content_hash"] == per_seed[b]["content_hash"])

    latent_mean = float(np.mean([p["mean"] for p in lat]))
    subspace_mean = float(np.mean([p["mean_cos"] for p in uni]))
    return dict(
        dataset=name, n=int(X.shape[0]), p=int(X.shape[1]), seeds=list(seeds),
        per_seed={str(s): {k: v for k, v in d.items() if k not in ("blocks", "flat")}
                  for s, d in per_seed.items()},
        latent_match=lat, union_subspace=uni,
        per_chart_subspace=chart if planes is not None else None,
        hashes_equal=hasheq,
        summary=dict(latent_mean_cos=latent_mean, subspace_mean_cos=subspace_mean,
                     subspace_minus_latent=subspace_mean - latent_mean,
                     cos_threshold=COS_THRESH),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sac_kwargs = dict(
        max_atoms=int(os.environ.get("W9_MAXATOMS", "6")),
        d_atom=int(os.environ.get("W9_DATOM", "1")),
        atom_topology="circle",
        n_iter=int(os.environ.get("W9_NITER", "30")),
        backfit_sweeps=int(os.environ.get("W9_BACKFIT", "2")),
        ev_floor=float(os.environ.get("W9_EVFLOOR", "5e-3")),
    )
    print(f"[cfg] seeds={SEEDS} sac_kwargs={sac_kwargs}", flush=True)

    results = {}
    # Primary dataset: a single clean circle + several linear atoms. SAC's forward
    # births then each see a well-posed target (one circle, then rank-1 atoms), the
    # regime the K=1 fit is proven on — unlike a superimposed multi-circle mixture,
    # where the single-atom birth hits the #1784/#1026 non-PD Arrow-Schur pathology
    # (documented in the run notes). planes=None -> per-chart subspace is skipped;
    # latent-vs-union-subspace is the core claim.
    from ev_budget_frontier import make_planted  # noqa: E402
    Xm, _kind, _p, _isc, _lab = make_planted(
        n=int(os.environ.get("W9_MIX_N", "1500")),
        nc=int(os.environ.get("W9_MIX_NC", "1")),
        nl=int(os.environ.get("W9_MIX_NL", "4")), seed=0)
    print(f"[mixed] X={Xm.shape} (1 circle + linear atoms)", flush=True)
    results["mixed_circle_linear"] = analyze("mixed", Xm, None, SEEDS, sac_kwargs)

    if os.environ.get("W9_PLANTED", "0") == "1":
        Xp, planes = data_planted(
            N=int(os.environ.get("W9_PLANT_N", "1500")),
            p=int(os.environ.get("W9_PLANT_P", "32")),
            n_circ=int(os.environ.get("W9_PLANT_NC", "3")))
        print(f"[planted] X={Xp.shape} n_circ={planes.shape[0]}", flush=True)
        results["planted"] = analyze("planted", Xp, planes, SEEDS, sac_kwargs)

    Xw, _ = data_w6()
    if Xw is not None:
        print(f"[w6] X={Xw.shape}", flush=True)
        results["w6"] = analyze("w6", Xw, None, SEEDS, sac_kwargs)
    else:
        print(f"[w6] cache {W6_CACHE} absent; skipping real slab", flush=True)

    (OUT_DIR / "manifold_stability_sac.json").write_text(json.dumps(results, indent=2))
    _write_md(results, OUT_DIR)
    print(f"\n[done] {OUT_DIR}/manifold_stability_sac.json + summary.md", flush=True)
    return 0


def _write_md(results, out_dir):
    lines = ["# W9 — manifold-SAE seed stability on SAC dictionaries", "",
             "Two seeds (data row-order / subsample) → one SAC curved dictionary each. "
             "The manifold-SAE claim: individual atoms are seed-unstable (latent), but the "
             "spanned frames recur (subspace). Subspace ≫ latent is the thesis.", "",
             "| dataset | n | p | latent mean|cos| | subspace mean cos | subspace − latent | hashes equal |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, r in results.items():
        s = r["summary"]
        lines.append(f"| {name} | {r['n']} | {r['p']} | {s['latent_mean_cos']:.3f} | "
                     f"{s['subspace_mean_cos']:.3f} | {s['subspace_minus_latent']:+.3f} | "
                     f"{all(r['hashes_equal'])} |")
    lines.append("")
    for name, r in results.items():
        if r.get("per_chart_subspace"):
            pc = r["per_chart_subspace"][0]
            lines.append(f"- **{name}** per-chart subspace cos: "
                         f"mean={pc['mean']:.3f} min={pc['min']:.3f} "
                         f"over {pc['n_charts']} charts.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
