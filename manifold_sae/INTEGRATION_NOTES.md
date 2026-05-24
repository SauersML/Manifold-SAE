# Integration audit — patches applied

Only minimal one-line patches were applied to keep modules importable; no
refactors. Per audit rules.

## 1. `manifold_sae/transcoder.py` — guard `SkipAffineSmooth` import

**Before** (line 33):
```python
from gamfit.torch import SkipAffineSmooth
```

**After**:
```python
try:
    from gamfit.torch import SkipAffineSmooth  # type: ignore
except ImportError:  # gamfit<0.1.99 lacks SkipAffineSmooth — keep module importable
    SkipAffineSmooth = None  # type: ignore
```

**Reason**: the installed `gamfit==0.1.98` in this venv does not yet
expose `SkipAffineSmooth`. The hard import broke `manifold_sae.transcoder`
import-time, which in turn broke any downstream module that did
`from manifold_sae import transcoder`.

The fallback `None` lets the module load; the integration layer
substitutes a `_LinearTranscoderStub` when the symbol is unavailable.

## Pre-existing bugs surfaced but NOT patched

- **`manifold_sae/sae.py:357`** — `result.lambdas.reshape(())` raises when
  gamfit's joint additive REML returns one λ per atom (size > 1). The
  integration smoke test marks `ManifoldSAE` as `xfail` and leaves the
  bug for the sae.py owner to fix.

No other modules required patches — all of `adaptive_k`, `crm`,
`crosscoder`, `equivariant`, `sheaf`, `wasserstein_sae`, `scale`,
`circuit_trace`, `kernels/*`, `eval/*`, and `autointerp/*` import and
run cleanly.
