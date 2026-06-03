# Integration notes — gamfit-native state (gamfit ≥ 0.1.145)

This repo is fully cut over to the gamfit primitives. There are no import
guards, version fallbacks, or stubs. The notes below record the current truth
(the SAE-glue facts below were established against `gamfit==0.1.141` and still
hold; the installed venv is now `gamfit==0.1.145`).

**Standing rule: always newest gamfit.** The joint manifold-recovery objective
(`gamfit.sae_manifold_fit`, used by `experiments/manifold_recovery.py` +
`experiments/manifold_falsifier.py`) and its verification harness depend on knobs
that ship in the *upcoming* gamfit beyond 0.1.145: cross-atom decoder incoherence
(`decoder_incoherence_weight`, #671), nuclear-norm embedding-rank selection (#672),
ScadMcp non-convex sparsity, gauge-conditional topology evidence on top of the
isometry gauge (#673), and per-atom topology/manifold **uncertainty** (posterior
shape bands, mean ± sd) + **typical coordinate range** on the fit result. The
harness resolves the incoherence knob name against the live `sae_manifold_fit`
signature at import and self-gates BLOCKED (never a false PASS) when a knob is
absent, so this repo stays correct across versions and goes green once the upcoming
release lands. Canonical assignment is `assignment="ibp"` (adaptive count, true
zeros) — the gam default — not `softmax`+`top_k`.

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
