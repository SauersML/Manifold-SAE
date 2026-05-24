# Cross-LLM Concept Interpretability Platform

Standalone package for concept-labeled prompt ingestion, HuggingFace activation
harvesting, validated gauge fixing, diagnostics, anchor-offset steering, REST /
WebSocket serving, and cross-model atlas runs.

## Install

```bash
cd /Users/user/Manifold-SAE/cross_llm_platform
python -m pip install -e ".[all]"
```

For NumPy-only gauge fitting and diagnostics:

```bash
python -m pip install -e .
```

## Five-Line API

```python
from cross_llm_platform import ConceptSteerer, fit_gauge, harvest_activations, label_prompts
prompts = ["A red apple is", "A blue ocean is", "A green leaf is"]
X = harvest_activations("gpt2", prompts, layer=6).activations
fit = fit_gauge(X, label_prompts(prompts, "hsv"), targets=["hsv"], anchor_rows={"red": [0], "blue": [1]})
steerer = ConceptSteerer(fit, layer=6)
print(steerer.request("The object is", "red", alpha=2.0).scale)
```

Pass a loaded HuggingFace `model` and `tokenizer` to `steer_text(...)` for local
generation, or run the FastAPI server with a loaded gauge for REST requests.

## Harvest Your Model

Create a text, JSON, or JSONL prompt file:

```text
A red apple is
A blue ocean is
A formal letter states
```

Harvest activations:

```bash
cls platform harvest --model gpt2 --layer 6 --prompts prompts.txt --out harvest.npz
```

The harvester supports decoder-only `AutoModelForCausalLM` models, configurable
layer probes, batching, pooling (`last_token`, `mean`, `first_token`), and an
optional Python steering hook for intervention experiments.

## Fit A Concept Gauge

Built-in concepts:

- `hsv`: hue, saturation, value from color words
- `sentiment`: sentiment valence
- `formality`: formal/informal lexical score
- `persona`: role indicators
- `time-period`: past-to-future period score
- `geographic-region`: region indicators

Fit a BIC-selected chart:

```bash
cls platform fit-gauge --activations harvest.npz --prompts prompts.txt --concept hsv --out gauge.npz
```

Python:

```python
from cross_llm_platform import fit_gauge, label_prompts
labels = label_prompts(prompts, "hsv")
fit = fit_gauge(X, labels, targets=["hsv"])
fit.register_anchor("red", X[[0, 4, 9]].mean(axis=0))
fit.save("gauge.npz")
```

## Steer Generations

Resolve a steering vector:

```bash
cls platform steer --gauge gauge.npz --prompt "The object is" --concept red --alpha 2
```

Generate locally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cross_llm_platform import ConceptSteerer, GaugeFit

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
fit = GaugeFit.load("gauge.npz")
result = ConceptSteerer(fit, layer=6).steer_text(
    "The object is", "red", 2.0, model=model, tokenizer=tok
)
print(result.text)
```

## Server

```bash
cls platform serve --gauge gauge.npz --layer 6 --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /concepts`
- `POST /steering/request`
- `POST /steer`
- `WS /ws/steer`

`/steering/request` returns the typed vector and scale. `/steer` and
`/ws/steer` require a server created with a loaded HuggingFace model and
tokenizer.

## Diagnostics

```python
from cross_llm_platform import validated_diagnostics
report = validated_diagnostics(fit, X, labels, {"red": [0, 4, 9]}, n_perm=100)
```

Included validated checks:

- per-anchor curvature (`auto_exp_52`)
- permutation null topology control (`auto_exp_42`)
- variance-vs-concept-locality (`auto_85`)

## Atlas

```python
from cross_llm_platform.atlas import run_atlas
run_atlas(["gpt2", "distilgpt2"], ["hsv", "sentiment"], prompts, layer=6, out_dir="atlas")
```

This writes activation matrices, gauge fits, `atlas.csv`, and `atlas_r2.png`.
