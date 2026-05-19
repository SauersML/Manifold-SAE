"""Phase 0 substrate smoke test for Manifold-SAE.

Verifies that gamfit's ``gaussian_reml_fit_positions_batched`` primitive and its
analytic VJP companion ``gaussian_reml_fit_positions_batched_backward`` are
callable, produce sane outputs, and behave well enough to build the Phase 1
``torch.autograd.Function`` on top of.

Run:
    .venv/bin/python scripts/phase0_substrate.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np

import gamfit
from gamfit import _rust  # used to bypass a wrapper bug, see WARNING in notes


# ---------------------------------------------------------------------------
# Test problem dimensions
# ---------------------------------------------------------------------------
F = 4     # features (= problems in the ragged batch)
B = 256   # tokens per feature segment
D = 8     # ambient response dimension
N = F * B # total packed rows
N_KNOTS = 12
DEGREE = 3
P = N_KNOTS - DEGREE - 1  # = 8 basis columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


@dataclass
class Synthetic:
    t: np.ndarray            # (N,) float64, positions in [0, 1]
    y: np.ndarray            # (N, D) float64, packed targets
    row_offsets: np.ndarray  # (F+1,) uintp
    knots: np.ndarray        # (N_KNOTS,) float64
    penalty: np.ndarray      # (P, P) float64
    amps: np.ndarray         # (F,) float64
    phases: np.ndarray       # (D,) float64
    noise_sd: float


def synth(noise_sd: float = 0.01, seed: int = 0) -> Synthetic:
    rng = np.random.default_rng(seed)
    # Distinct positions per feature, packed
    t_list = [rng.uniform(0.0, 1.0, size=B).astype(np.float64) for _ in range(F)]
    amps = np.abs(rng.standard_normal(F)).astype(np.float64)  # |N(0,1)|
    phases = rng.uniform(0.0, 2.0 * np.pi, size=D).astype(np.float64)

    ys = []
    for f in range(F):
        # Per-feature smooth curve in D ambient dims: amp_f * sin(2pi t + phase_d)
        curve = np.sin(2.0 * np.pi * t_list[f][:, None] + phases[None, :])
        ys.append(amps[f] * curve + noise_sd * rng.standard_normal((B, D)))

    t = np.concatenate(t_list).astype(np.float64)
    y = np.concatenate(ys, axis=0).astype(np.float64)
    row_offsets = np.array([f * B for f in range(F + 1)], dtype=np.uintp)
    knots = np.linspace(0.0, 1.0, N_KNOTS, dtype=np.float64)
    S, _null = gamfit.smoothness_penalty(knots, degree=DEGREE, order=2)
    penalty = np.asarray(S, dtype=np.float64)
    return Synthetic(t, y, row_offsets, knots, penalty, amps, phases, noise_sd)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def run_forward(s: Synthetic) -> dict:
    out = gamfit.gaussian_reml_fit_positions_batched(
        s.t, s.y, s.row_offsets,
        "bspline", s.knots, s.penalty,
        basis_order=DEGREE,
        periodic=False,
    )
    print("\n=== forward outputs ===")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:32s} shape={v.shape}  dtype={v.dtype}  "
                  f"range=[{v.min():.3e}, {v.max():.3e}]")
        else:
            print(f"  {k:32s} = {v!r}")
    return out


def assert_forward_sane(s: Synthetic, out: dict) -> None:
    # Expected primary outputs
    for key in ("coefficients", "fitted", "lambda", "reml_score", "edf", "sigma2"):
        if key not in out:
            fail(f"forward output missing key {key!r}")

    coefs = out["coefficients"]
    fitted = out["fitted"]
    lam = out["lambda"]
    reml = out["reml_score"]
    edf = out["edf"]
    statuses = out.get("status", ["ok"] * F)

    # Shapes
    if coefs.shape != (F, P, D):
        fail(f"coefficients shape {coefs.shape} != ({F}, {P}, {D})")
    if fitted.shape != (N, D):
        fail(f"fitted shape {fitted.shape} != ({N}, {D})")
    for arr, name in [(lam, "lambda"), (reml, "reml_score"), (edf, "edf")]:
        if arr.shape != (F,):
            fail(f"{name} shape {arr.shape} != ({F},)")

    # Per-fit status
    for i, st in enumerate(statuses):
        if st != "ok":
            fail(f"fit {i} status = {st!r}; expected 'ok' on clean synthetic data")

    # lambda finite and positive
    if not np.all(np.isfinite(lam)):
        fail(f"lambda contains non-finite values: {lam}")
    if not np.all(lam > 0.0):
        fail(f"lambda not all > 0: {lam}")

    # REML score finite
    if not np.all(np.isfinite(reml)):
        fail(f"reml_score contains non-finite values: {reml}")

    # EDF in (0, P]
    if not np.all((edf > 0) & (edf <= P + 1e-6)):
        fail(f"edf out of (0, P]: {edf}")

    # Fitted reasonably close to targets when noise is small
    resid = fitted - s.y
    rms = float(np.sqrt(np.mean(resid ** 2)))
    print(f"\n  per-fit RMS residual (noise_sd={s.noise_sd}): {rms:.4f}")
    if rms > 5.0 * s.noise_sd + 0.05:
        fail(f"fit too poor: rms={rms:.4f} vs noise_sd={s.noise_sd}")

    print("  forward checks: OK")


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
def run_backward(s: Synthetic, fwd: dict) -> dict:
    # The installed gamfit 0.1.64 _api.py wrapper omits the new ``forward_state``
    # positional argument that the underlying Rust binding (v0.1.66) expects,
    # so calling ``gamfit.gaussian_reml_fit_positions_batched_backward`` raises
    # a TypeError. We bypass the wrapper and call ``gamfit._rust`` directly.
    # See phase0_notes.md WARNING.
    grad_fitted = np.ones_like(fwd["fitted"], dtype=np.float64)
    back = _rust.gaussian_reml_fit_positions_batched_backward(
        s.t,                   # t
        s.y,                   # y
        s.row_offsets,         # row_offsets
        "bspline",             # basis_kind
        s.knots,               # knots_or_centers
        s.penalty,             # penalty
        None,                  # grad_lambda
        None,                  # grad_coefficients
        grad_fitted,           # grad_fitted
        None,                  # grad_reml_score
        None,                  # forward_state (optional cache from fwd)
        DEGREE,                # basis_order
        False,                 # periodic
        None,                  # period
        None,                  # weights
        None,                  # init_lambda
        None,                  # by
        0,                     # by_start_col
    )
    back = dict(back)
    # Coerce to numpy float64
    for k, v in list(back.items()):
        if v is not None and not isinstance(v, list):
            back[k] = np.asarray(v, dtype=np.float64) if hasattr(v, "__array__") else v
    print("\n=== backward outputs ===")
    for k, v in back.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:16s} shape={v.shape}  dtype={v.dtype}  "
                  f"range=[{v.min():.3e}, {v.max():.3e}]")
        else:
            print(f"  {k:16s} = {v!r}")
    return back


def assert_backward_sane(s: Synthetic, back: dict) -> None:
    for key in ("grad_t", "grad_y"):
        if key not in back or back[key] is None:
            fail(f"backward output missing {key!r}")
    grad_t = back["grad_t"]
    grad_y = back["grad_y"]
    if grad_t.shape != (N,):
        fail(f"grad_t shape {grad_t.shape} != ({N},)")
    if grad_y.shape != (N, D):
        fail(f"grad_y shape {grad_y.shape} != ({N}, {D})")
    if not np.all(np.isfinite(grad_t)):
        fail("grad_t contains non-finite values")
    if not np.all(np.isfinite(grad_y)):
        fail("grad_y contains non-finite values")
    # grad_fitted = ones implies d(sum fitted)/d(y) should be projection rows,
    # which are nonzero almost surely.
    if np.allclose(grad_y, 0.0):
        fail("grad_y is all zero; backward did not propagate from grad_fitted")
    print("  backward checks: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"gamfit version: {gamfit.__version__}")
    info = gamfit.build_info()
    print(f"gamfit rust crate version: {info.get('version')}")
    print(f"problem dims: F={F}  B={B}  D={D}  N={N}  P={P}")

    try:
        s = synth(noise_sd=0.01, seed=0)
        fwd = run_forward(s)
        assert_forward_sane(s, fwd)
        back = run_backward(s, fwd)
        assert_backward_sane(s, back)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("\nFAIL: unexpected exception during smoke test", file=sys.stderr)
        return 1

    print("\nPHASE 0 SUBSTRATE: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
