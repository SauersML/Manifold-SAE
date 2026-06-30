"""Minimal repro: gaussian_reml_fit_positions_batched with periodic=True returns
status='degenerate' / lambda=NaN / fitted=0 on inputs that open mode handles fine.

The trigger is not a function of any obvious quantity (sparsity, n_nonzero, rank
of design). The same input with periodic=False produces a clean ok fit; flipping
the flag returns a zero fit. Tested at gamfit==0.1.67.
"""

from __future__ import annotations

import numpy as np
import gamfit


def trial(seed: int, n: int = 2000, K: int = 12) -> tuple[str, str, float, float]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, size=n)).astype(np.float64)
    y = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=-1).astype(np.float64)
    by = np.abs(rng.standard_normal(n)).astype(np.float64)
    centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
    penalty = np.eye(K, dtype=np.float64)
    offsets = np.array([0, n], dtype=np.uintp)

    out_open = gamfit.gaussian_reml_fit_positions_batched(
        t, y, offsets, "duchon", centers, penalty,
        basis_order=2, periodic=False, period=None, by=by, init_lambda=1.0,
    )
    out_per = gamfit.gaussian_reml_fit_positions_batched(
        t, y, offsets, "duchon", centers, penalty,
        basis_order=2, periodic=True, period=1.0, by=by, init_lambda=1.0,
    )
    return (
        list(out_open["status"])[0],
        list(out_per["status"])[0],
        float(np.linalg.norm(np.asarray(out_open["fitted"]))),
        float(np.linalg.norm(np.asarray(out_per["fitted"]))),
    )


def main() -> None:
    print("Sweeping seeds; for each, the same (t, y, by) inputs are passed to")
    print("gaussian_reml_fit_positions_batched twice — once with periodic=False")
    print("(open Duchon, control) and once with periodic=True. n=2000, K=12,")
    print("init_lambda=1.0, identity penalty, by = abs(N(0,1)).\n")
    print(f"{'seed':>6s}  {'open status':>14s}  {'periodic status':>17s}  {'open ||fit||':>14s}  {'periodic ||fit||':>18s}  bug?")
    n_bugs = 0
    for seed in range(20):
        s_open, s_per, fn_open, fn_per = trial(seed)
        is_bug = (s_open == "ok") and (s_per != "ok" or fn_per < 1e-10)
        n_bugs += int(is_bug)
        print(f"{seed:>6d}  {s_open:>14s}  {s_per:>17s}  {fn_open:>14.4e}  {fn_per:>18.4e}  {'YES' if is_bug else ''}")
    print(f"\n{n_bugs}/20 seeds reproduce the bug (periodic fails while open succeeds on the same inputs).")


if __name__ == "__main__":
    main()
