# concept_manifold_steering

A pip-installable Python toolkit that ships the **validated cogito-L40
steering recipe** (Manifold-SAE auto_exp_38 → auto_exp_54) as a
model-agnostic library:

1. Harvest activations at any layer of any HuggingFace decoder-only LM.
2. Fit an **HSV-style gauge-fixed subspace** whose axes regress
   user-supplied targets (auto_exp_38, auto_exp_53).
3. Steer via **anchor offsets** (auto_exp_44 — not tangents!) through
   a vLLM intervention server.
4. Run pre-flight **diagnostics** so you know when *not* to ship a
   steerer (auto_exp_42 / _49 / _52 / _85).

## Install

```bash
pip install -e .                       # core: numpy + scipy
pip install -e ".[hf,plot,dev]"        # optional extras
```

## 10-line use

```python
from concept_manifold_steering import harvest_activations, GaugeFix, ManifoldSteerer

prompts = ["The color of a tomato is", "A pumpkin is", "The clear sky is", ...]
labels  = {"hue": [...], "saturation": [...], "value": [...]}      # length == len(prompts)
anchors = {"red": [0, 1, 2], "blue": [10, 11, 12]}                 # row indices per concept

X = harvest_activations("deepcogito/cogito-v1-preview-llama-8B",
                        prompts, layer=40, aggregate="mean",
                        trust_remote_code=True)
gauge   = GaugeFix(targets=["hue", "saturation", "value"]).fit(X, labels, anchor_labels=anchors)
steerer = ManifoldSteerer(gauge, server_url="http://localhost:8000", layer=40)
print(steerer.steer("My favorite color is", concept="red", alpha=2.0).completion)
```

## What generalises beyond cogito + color

* `harvest_activations` works with any `AutoModelForCausalLM`.
* `GaugeFix` is purely linear-algebra; targets can be any numeric (or
  categorical) per-prompt label, in any number of dimensions.  HSV is
  just the cogito demo — `["sentiment", "formality"]` works the same.
* `ManifoldSteerer` only assumes the server accepts an `extra_body`
  intervention payload (the same shape vLLM and several open-source
  forks already expose).
* Diagnostics are recipe-level: per-anchor curvature, permutation
  null, and locality scatter all transfer.

## Failure modes wired in as warnings

| Source experiment | Warning |
|---|---|
| auto_exp_42 | targets whose R² is not significant vs permutation null |
| auto_exp_49 | concepts whose anchor variance is mostly *outside* the gauge subspace |
| auto_exp_52 | per-anchor curvature high enough to violate the flat-affine assumption |

## Smoke test

```bash
pytest concept_manifold_steering/tests/test_smoke.py -q
```
