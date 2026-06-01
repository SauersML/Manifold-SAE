# Integration notes — gamfit-native state (gamfit 0.1.141)

This repo is fully cut over to the gamfit primitives. There are no import
guards, version fallbacks, or stubs. The notes below record the current truth
(verified against the installed `gamfit==0.1.141`).

## `manifold_sae/transcoder.py` — direct `SkipAffineSmooth` import

```python
from gamfit.torch import SkipAffineSmooth
```

The import is unconditional. `SkipAffineSmooth` ships in `gamfit.torch`; there
is no `try/except ImportError` guard and no `None` fallback. Likewise the
integration layer imports it directly (`_build_transcoder`) — there is no
`_LinearTranscoderStub` and no stub-substitution path anywhere.

## `manifold_sae/sae.py` — thin re-export of `gamfit.torch.ManifoldSAE`

`sae.py` no longer contains a hand-rolled SAE; it re-exports the gamfit
primitive and its config/output dataclasses and adds only Manifold-SAE-side
glue (`load_sae`, `extract_feature_curves`, `lift_atom_curve`).

* **Resolved: F=8 / per-atom λ.** The old `result.lambdas.reshape(())` crash
  belonged to the deleted hand-rolled solve. gamfit's `ManifoldSAEOutput.lambdas`
  is a per-atom vector of shape `(n_atoms,)` (one λ per atom), and forward + fit
  both run at F=8. ManifoldSAE is exercised at the shared F=8 setting in the
  integration smoke test — no xfail, no F>=16 override.

## dtype handling at the boundary

gamfit's `ManifoldSAE` is pinned to its config dtype (float64 by default for
the REML solve) and *raises* on a mismatched input dtype instead of silently
promoting. The integration driver (`integration.py::_call_loss`) casts the
input to the model's expected dtype before calling forward, so callers can pass
float32 data and the manifold variant still runs.
