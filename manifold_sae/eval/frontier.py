"""Manifold-beats-linear frontier evaluation (gamfit-0.1.241-native).

The headline claim of this repo — a *curved-manifold* SAE dictionary reaches the
linear (PCA-style) reconstruction ceiling at **fewer atoms** — is exactly what
gamfit adjudicates natively via :func:`gamfit.sae_ev_vs_k_frontier` (fit a
manifold SAE and the Rust linear-dictionary baseline across a K sweep, score
held-out reconstruction EV with frozen decoders) and
:func:`gamfit._sae_manifold.wager_verdict` (does the manifold hit the linear EV
ceiling at smaller K?).

This module is a thin, opinionated wrapper around those primitives — it is the
one place the repo asserts "manifold beats linear", so it delegates the numerics
entirely to gamfit rather than re-deriving EV/knee/verdict by hand.

Usage
-----
    from manifold_sae.eval.frontier import manifold_vs_linear_frontier, format_frontier_markdown
    res = manifold_vs_linear_frontier(train_acts, test_acts, k_values=[4, 8, 16, 32])
    print(format_frontier_markdown(res))
    assert res.beats_linear            # curved dictionary is more atom-efficient
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np

import gamfit
from gamfit._sae_manifold import wager_verdict


@contextlib.contextmanager
def _fd_quiet(enabled: bool = True):
    """Silence gamfit's Rust-side inner-fit progress (written to fd 1/2)."""
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


@dataclass
class FrontierResult:
    """Outcome of a manifold-vs-linear K-sweep on held-out activations."""

    manifold_ev_by_k: dict[int, float]
    linear_ev_by_k: dict[int, float]
    verdict: dict[str, Any]
    rows: list[dict[str, Any]]
    knee: Any
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def beats_linear(self) -> bool:
        """True iff the curved dictionary reaches the linear EV ceiling at lower K."""
        return bool(self.verdict.get("confirmed", False))

    @property
    def efficiency_ratio(self) -> float | None:
        """linear_k / manifold_k — how many fewer atoms the manifold needs (>1 is good)."""
        return self.verdict.get("efficiency_ratio")

    @property
    def manifold_k(self) -> int | None:
        return self.verdict.get("manifold_k")

    @property
    def linear_k(self) -> int | None:
        return self.verdict.get("linear_k")


def manifold_vs_linear_frontier(
    train: Any,
    test: Any,
    k_values: Sequence[int],
    *,
    d_atom: int = 1,
    atom_topology: str = "circle",
    hybrid_atom_basis: Mapping[int, list[str]] | None = None,
    sae_fit_kwargs: Mapping[str, Any] | None = None,
    linear_fit_kwargs: Mapping[str, Any] | None = None,
    quiet: bool = True,
) -> FrontierResult:
    """Fit the manifold SAE and the linear-dictionary baseline across ``k_values``.

    Parameters
    ----------
    train, test : (N, D) activation matrices. Fit on ``train``, score held-out EV on ``test``.
    k_values : dictionary sizes to sweep.
    d_atom : per-atom intrinsic dimension (1 = curves; 2 = surfaces).
    atom_topology : seed topology for every atom when ``hybrid_atom_basis`` is not given.
    hybrid_atom_basis : optional per-K, per-atom basis plan ``{K: [topo, ...]}`` (len K).
    sae_fit_kwargs / linear_fit_kwargs : forwarded to gamfit's manifold / linear fits.
    quiet : suppress gamfit's Rust inner-fit progress output.

    Returns
    -------
    FrontierResult — ``.beats_linear`` is the verdict; ``.efficiency_ratio`` the atom saving.
    """
    ks = [int(k) for k in k_values]
    if not ks:
        raise ValueError("k_values must be non-empty")
    if hybrid_atom_basis is None:
        hybrid_atom_basis = {k: [atom_topology] * k for k in ks}

    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    with _fd_quiet(quiet):
        raw = gamfit.sae_ev_vs_k_frontier(
            train,
            test,
            ks,
            hybrid_atom_basis=dict(hybrid_atom_basis),
            d_atom=d_atom,
            sae_fit_kwargs=dict(sae_fit_kwargs or {}),
            linear_fit_kwargs=dict(linear_fit_kwargs or {}),
        )

    man = {int(k): float(v) for k, v in dict(raw.get("hybrid", {})).items()}
    lin = {int(k): float(v) for k, v in dict(raw.get("linear", {})).items()}
    verdict = dict(wager_verdict(man, lin)) if man and lin else {}
    return FrontierResult(
        manifold_ev_by_k=man,
        linear_ev_by_k=lin,
        verdict=verdict,
        rows=list(raw.get("rows", [])),
        knee=raw.get("knee"),
        raw=raw,
    )


def format_frontier_markdown(result: FrontierResult) -> str:
    """Render the frontier as a markdown table + one-line verdict."""
    lines = [
        "| K | manifold EV | linear EV | Δ (manifold−linear) |",
        "|---:|---:|---:|---:|",
    ]
    for k in sorted(set(result.manifold_ev_by_k) | set(result.linear_ev_by_k)):
        m = result.manifold_ev_by_k.get(k)
        l = result.linear_ev_by_k.get(k)
        if m is None or l is None:
            lines.append(f"| {k} | {m if m is not None else '—'} | {l if l is not None else '—'} | — |")
        else:
            lines.append(f"| {k} | {m:.4f} | {l:.4f} | {m - l:+.4f} |")
    v = result.verdict
    head = "✅ MANIFOLD BEATS LINEAR" if result.beats_linear else "❌ no manifold advantage"
    lines += [
        "",
        f"**{head}** — manifold_k={v.get('manifold_k')}, linear_k={v.get('linear_k')}, "
        f"efficiency_ratio={v.get('efficiency_ratio')}, target_ev={v.get('target_ev')}, "
        f"best manifold/linear EV={v.get('best_manifold_ev')}/{v.get('best_linear_ev')}.",
    ]
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Manifold-beats-linear frontier on cached activations (.npy).")
    ap.add_argument("--train", required=True, help="(N,D) float .npy of training activations")
    ap.add_argument("--test", required=True, help="(N,D) float .npy of held-out activations")
    ap.add_argument("--k", type=int, nargs="+", required=True, help="K values to sweep")
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--topology", default="circle")
    args = ap.parse_args(argv)

    res = manifold_vs_linear_frontier(
        np.load(args.train), np.load(args.test), args.k,
        d_atom=args.d_atom, atom_topology=args.topology,
    )
    print(format_frontier_markdown(res))
    return 0 if res.beats_linear else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
