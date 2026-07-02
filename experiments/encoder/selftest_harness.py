"""Fast local self-tests for the WS-E harness parts that need no gamfit fit.

Covers the pure-numpy / torch-forward logic — decile bucketing of the
certificate-fallback rate, and the amortized-encode throughput measurement —
with a stub encoder, so the harness is exercised in seconds on the laptop
(the gamfit fit itself is validated on node2). Run:

    python selftest_harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_harness as dh          # noqa: E402


def test_decile_monotone_planted() -> None:
    """Fallback engineered to fall with frequency -> decile rates are monotone."""
    rng = np.random.default_rng(0)
    n = 10_000
    freq = rng.random(n)
    # accept-probability rises with frequency: rare tokens fall back more.
    accept_p = 0.2 + 0.75 * freq
    accepted = rng.random(n) < accept_p
    deciles = dh.fallback_by_decile(accepted, freq, n_deciles=10)
    assert len(deciles) == 10
    assert sum(d.rows for d in deciles) == n
    rates = [d.fallback_rate for d in deciles]
    # rarest decile (0) must fall back MORE than the most frequent (9).
    assert rates[0] > rates[9], (rates[0], rates[9])
    # broadly decreasing (allow small sampling wiggle): a Spearman-like check.
    inversions = sum(
        1 for i in range(len(rates)) for j in range(i + 1, len(rates)) if rates[i] < rates[j]
    )
    assert inversions <= 8, f"too many inversions ({inversions}) for a monotone plant"
    print(f"  decile rates rare->freq: {[round(r,3) for r in rates]}  inversions={inversions}")


def test_decile_edges_cover_all_rows() -> None:
    """Every row lands in exactly one decile even with heavy ties (Zipf-like)."""
    rng = np.random.default_rng(1)
    n = 5000
    freq = rng.choice([1e-4, 1e-3, 1e-2, 1e-1], size=n, p=[0.5, 0.3, 0.15, 0.05])
    accepted = rng.random(n) < 0.9
    deciles = dh.fallback_by_decile(accepted, freq, n_deciles=10)
    assert sum(d.rows for d in deciles) == n
    total_fb = sum(d.fallback_rows for d in deciles)
    assert total_fb == int(np.count_nonzero(~accepted))
    print(f"  tie-heavy freq: rows covered={sum(d.rows for d in deciles)} fb={total_fb}")


class _StubEncoder:
    """A minimal torch-MLP encoder exposing the encode_fast throughput surface."""

    def __init__(self, p: int, k: int, d: int, hidden: int = 128) -> None:
        import torch

        self.module = torch.nn.Sequential(
            torch.nn.Linear(p, hidden, dtype=torch.float64),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, k * d + k, dtype=torch.float64),
        )
        self._k, self._d, self._p = k, d, p

    def encode_fast(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = torch.as_tensor(np.asarray(X, dtype=np.float64))
            out = self.module(t).cpu().numpy()
        return out[:, : self._k]


def test_throughput_stub() -> None:
    p, k, d = 128, 3, 1
    enc = _StubEncoder(p, k, d)
    proto = np.random.default_rng(2).standard_normal((2000, p))
    thr = dh.measure_throughput(enc, proto, target_rows=200_000, warmup=1, repeats=2)
    print(f"  stub throughput: {thr['rows_per_s']:,.0f} rows/s on {thr['device']} "
          f"({thr['batch_rows']} rows)")
    assert thr["rows_per_s"] > 1.0e5, f"stub MLP should clear 1e5 rows/s; got {thr['rows_per_s']}"


def main() -> int:
    print("test_decile_monotone_planted"); test_decile_monotone_planted()
    print("test_decile_edges_cover_all_rows"); test_decile_edges_cover_all_rows()
    print("test_throughput_stub"); test_throughput_stub()
    print("ALL SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
