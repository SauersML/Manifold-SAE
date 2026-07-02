"""W10 — live EV-vs-budget frontier with per-atom (Theta, dEV), from SAC births.

The joint-fit version (``ev_budget_frontier.py``) had to read a per-atom
``train_loao_delta_ev`` out of one joint ``sae_manifold_fit`` — the exact call that
co-collapses on real data. SAC gives the same information *for free* and monotonically:
each forward birth reports its **marginal explained variance** ``dEV`` on the running
residual, and its decoder frame's coefficient count is its honest parameter budget
``Theta``. Sorting the births by ``dEV`` and accumulating ``Theta`` is the
``(Theta, dEV)`` frontier — and because SAC never lets atoms compete inside one
Hessian, ``combined_ev`` is strictly non-decreasing in births by construction.

We score that frontier against the reference every budget must beat: gamfit's linear /
sparse dictionaries at matched parameter budget (``K_lin = Theta / p`` rank-1 atoms),
via ``ev_budget_frontier.linear_ev_curve``. On a planted mixture of curved + linear
atoms, the curved atoms should buy EV a linear dictionary cannot reach at the same
Theta (a circle costs one curved atom's frame but two linear directions).

Held-out: the linear reference is scored on a held-out split; SAC's birth ``dEV`` is an
in-sample marginal (the same lens the original W10 used), so we ALSO replay the SAC
dictionary on the held-out split (sequential residual reconstruction) to report a
held-out combined-EV frontier where the fitted charts expose a reconstruct path.

Run on node2 (K=1 outer loop). Reuses ``ev_budget_frontier`` + ``sac_prototype``.
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
for _p in (Path(os.environ.get("GAM_EXAMPLES", "/models/sauers_build/gam_fable/examples")),
           Path("/Users/user/gam/examples")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ev_budget_frontier import make_planted, split_scale, ev, linear_ev_curve  # noqa: E402
from sac_prototype import sac_fit  # noqa: E402

OUT_DIR = Path(os.environ.get("W10_OUT", str(_HERE / "ev_budget_sac_out")))


def _atom_theta(atom, p) -> int:
    """Parameter budget of one SAC atom = size of its decoder frame (coeffs)."""
    B = np.asarray(atom.fit.atoms[0].decoder_coefficients, dtype=np.float64)
    return int(B.size)


def _heldout_frontier(result, test):
    """Replay the SAC dictionary on the held-out split: after each accepted atom,
    combined-EV of ``sum_{j<=k} recon_j(test residual)``. Returns list per k or None
    if the fitted atoms expose no reconstruct path on new data."""
    try:
        import torch  # noqa: F401
        r = test.copy()
        acc = np.zeros_like(test)
        curve = []
        for atom in result.atoms:
            sae = atom.fit
            rec = None
            for meth in ("reconstruct", "transform"):
                fn = getattr(sae, meth, None)
                if fn is None:
                    continue
                try:
                    out = fn(np.ascontiguousarray(r))
                    rec = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)
                    if rec.shape == r.shape:
                        break
                    rec = None
                except Exception:  # noqa: BLE001
                    rec = None
            if rec is None:
                return None
            acc = acc + rec
            r = test - acc
            curve.append(float(ev(test, acc)))
        return curve
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nc = int(os.environ.get("W10_NC", "3"))
    nl = int(os.environ.get("W10_NL", "3"))
    n = int(os.environ.get("W10_N", "3000"))
    X, kind, p, atom_is_curved, lab = make_planted(n=n, nc=nc, nl=nl, seed=0)
    train, test = split_scale(X, seed=1)
    print(f"[cfg] planted n={n} p={p} nc={nc} nl={nl} train={train.shape} test={test.shape}",
          flush=True)

    res = sac_fit(
        np.ascontiguousarray(train),
        max_atoms=int(os.environ.get("W10_MAXATOMS", str(nc + nl + 2))),
        d_atom=1, atom_topology="circle",
        ev_floor=float(os.environ.get("W10_EVFLOOR", "3e-3")),
        n_iter=int(os.environ.get("W10_NITER", "40")),
        backfit_sweeps=int(os.environ.get("W10_BACKFIT", "2")),
        random_state=0, verbose=True)
    print(f"[sac] K={res.k} t1_ev={res.t1_ev:.4f} combined_ev={res.combined_ev:.4f}",
          flush=True)

    # Per-atom (Theta, dEV) in birth order; cumulative frontier from the EV trace.
    per_atom = []
    cum_theta = 0
    for i, atom in enumerate(res.atoms):
        theta = _atom_theta(atom, p)
        cum_theta += theta
        per_atom.append(dict(
            birth=i, theta=theta, cum_theta=cum_theta,
            delta_ev=float(atom.delta_ev),
            combined_ev_after=float(res.ev_trace[i]) if i < len(res.ev_trace) else None,
            topology=atom.topology, hybrid_verdict=atom.hybrid_verdict,
            is_curved=atom.topology not in ("line", "linear")))
    # sorted-by-marginal frontier (the (Theta, dEV) ranking)
    ranked = sorted(per_atom, key=lambda a: -a["delta_ev"])

    # Linear reference at matched budgets (K_lin = cum_theta / p).
    k_lins = sorted(set(max(1, int(round(a["cum_theta"] / p))) for a in per_atom))
    lin_curve = linear_ev_curve(train, test, p, k_lins)

    heldout = _heldout_frontier(res, test)

    payload = dict(
        config=dict(n=n, p=p, nc=nc, nl=nl,
                    n_curved_atoms=int(atom_is_curved.sum()),
                    n_linear_atoms=int((1 - atom_is_curved).sum())),
        sac=dict(K=res.k, t1_ev=float(res.t1_ev), combined_ev=float(res.combined_ev),
                 ev_gain=float(res.ev_gain), ev_trace=[float(x) for x in res.ev_trace]),
        per_atom_birth_order=per_atom,
        per_atom_ranked_by_delta_ev=ranked,
        linear_reference_curve=lin_curve,
        heldout_combined_ev_frontier=heldout,
    )
    (OUT_DIR / "ev_budget_sac.json").write_text(json.dumps(payload, indent=2, default=float))
    _write_md(payload, OUT_DIR)
    print(f"\n[done] {OUT_DIR}/ev_budget_sac.json + summary.md", flush=True)
    return 0


def _write_md(pl, out_dir):
    lines = ["# W10 — EV-vs-budget frontier from SAC births", "",
             f"Planted mixture: {pl['config']['nc']} curved + {pl['config']['nl']} linear "
             f"atoms in p={pl['config']['p']}. SAC combined EV = "
             f"{pl['sac']['combined_ev']:.4f} at K={pl['sac']['K']} "
             f"(monotone in births by construction).", "",
             "## Per-atom (Theta, dEV), birth order",
             "| birth | topology | verdict | Theta | cum Theta | dEV | combined EV |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for a in pl["per_atom_birth_order"]:
        ce = "" if a["combined_ev_after"] is None else f"{a['combined_ev_after']:.4f}"
        lines.append(f"| {a['birth']} | {a['topology']} | {a['hybrid_verdict']} | "
                     f"{a['theta']} | {a['cum_theta']} | {a['delta_ev']:+.4f} | {ce} |")
    lines += ["", "## Linear reference at matched budget (held-out EV)",
              "| K_lin | Theta | linear_dict EV | sparse_dict EV |",
              "|---:|---:|---:|---:|"]
    for r in pl["linear_reference_curve"]:
        lines.append(f"| {r['K_lin']} | {r['theta']} | "
                     f"{r.get('linear_dict_ev', float('nan')):.4f} | "
                     f"{r.get('sparse_dict_ev', float('nan')):.4f} |")
    if pl["heldout_combined_ev_frontier"]:
        lines += ["", "## SAC held-out combined-EV frontier (sequential replay)",
                  ", ".join(f"{v:.4f}" for v in pl["heldout_combined_ev_frontier"])]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
