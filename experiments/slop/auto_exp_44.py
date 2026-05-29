"""auto_exp_44: causal steering test on cogito-probed at L40 with color_manifold probes.

Tests whether the color-manifold tangent probes (3 Jacobian columns at base color
points on the U_3d nonlinear color manifold) causally steer generation toward
their associated color semantics. Picks pair of opposed anchors (red vs blue),
runs color-agnostic prompts at alpha in {0, 2, 5}, scores warm/cool word ratio.

Endpoint: <COGITO_API_BASE> (VPN required).
Uses /v1/chat/completions only — /v1/encode is skipped (stuck queue from exp_43).

Request schema (verified via probe_server_vllm.py source):
  {"model": "cogito-probed",
   "messages": [{"role": "user", "content": "..."}],
   "stream": false,
   "intervention": {"type": "steer",
                    "probe": "color_manifold:t1_at_red",
                    "strength": 2.0}}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

BASE = os.environ.get(
    "COGITO_API_BASE", os.environ.get("COGITO_URL", "http://localhost:8000")
)
OUT = Path(__file__).resolve().parents[1] / "runs" / "auto_exp_44_steering"
OUT.mkdir(parents=True, exist_ok=True)

WARM = {"red", "orange", "yellow", "gold", "crimson", "scarlet", "amber",
        "rust", "copper"}
COOL = {"blue", "green", "teal", "cyan", "indigo", "violet", "turquoise",
        "aqua", "mint"}

PROMPTS = [
    "Describe a sunset.",
    "Tell me about a peaceful scene.",
    "Pick three things and describe them.",
    "Paint a picture with words of a landscape.",
    "What do you see when you close your eyes?",
]

# Tangent probes: each color anchor has 3 tangent columns t1/t2/t3. Without a
# priori knowledge of which tangent corresponds to which semantic direction,
# we use t1 (first principal direction at that anchor) as default.
# Opposed semantics: red (warm) vs blue (cool); orange (warm) vs green (cool).
PROBES_TO_TEST = [
    ("color_manifold:t1_at_red", "warm"),
    ("color_manifold:t1_at_blue", "cool"),
    ("color_manifold:t1_at_orange", "warm"),
    ("color_manifold:t1_at_green", "cool"),
]
ALPHAS = [0.0, 2.0, 5.0]


def score(text: str) -> tuple[int, int, float]:
    tokens = [t.strip(".,!?;:'\"()[]").lower() for t in text.split()]
    w = sum(1 for t in tokens if t in WARM)
    c = sum(1 for t in tokens if t in COOL)
    ratio = w / (w + c) if (w + c) > 0 else float("nan")
    return w, c, ratio


def generate(prompt: str, probe: str | None, alpha: float,
             timeout: int = 90) -> str:
    body = {
        "model": "cogito-probed",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 200,
        "stream": True,
    }
    if probe is not None and alpha != 0.0:
        body["intervention"] = {
            "type": "steer",
            "probe": probe,
            "strength": float(alpha),
        }
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=timeout,
                      stream=True)
    r.raise_for_status()
    out_parts: list[str] = []
    for raw in r.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.lstrip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            choice = evt["choices"][0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or choice.get("text") or ""
            if piece:
                out_parts.append(piece)
        except (KeyError, IndexError, TypeError):
            continue
    return "".join(out_parts)


def main():
    # Sanity check
    try:
        probes = requests.get(f"{BASE}/v1/probes", timeout=10).json()
    except Exception as e:
        print(f"ABORT: cannot reach {BASE} ({e}). VPN up?")
        return
    cm = next((p for p in probes["probes"] if p["name"] == "color_manifold"), None)
    if cm is None or cm["n_probes"] != 33:
        print(f"ABORT: color_manifold probe set not loaded as expected ({cm})")
        return
    print(f"Loaded color_manifold: {cm['n_probes']} probes at L{cm['layer']}")
    print(f"Labels: {cm['labels']}")
    print()

    results = []
    rows = []  # (probe, alpha, mean_ratio, sample)
    t0 = time.time()

    for probe, expected in PROBES_TO_TEST:
        for alpha in ALPHAS:
            if time.time() - t0 > 600:
                print("Soft 10 min budget exceeded; stopping.")
                break
            warm_ratios = []
            samples = []
            for prompt in PROMPTS:
                try:
                    text = generate(prompt, probe, alpha, timeout=90)
                except requests.exceptions.RequestException as e:
                    print(f"  [{probe} a={alpha}] request failed: {e}; skip probe")
                    text = ""
                w, c, ratio = score(text)
                results.append({
                    "probe": probe, "expected": expected, "alpha": alpha,
                    "prompt": prompt, "text": text,
                    "warm": w, "cool": c, "ratio": ratio,
                })
                if not (ratio != ratio):  # not NaN
                    warm_ratios.append(ratio)
                samples.append(text[:60].replace("\n", " "))
            mean_ratio = sum(warm_ratios) / len(warm_ratios) if warm_ratios else float("nan")
            rows.append((probe, alpha, mean_ratio, samples[0]))
            print(f"  {probe:38s} a={alpha:>4.1f}  warm_ratio={mean_ratio:.3f}  "
                  f"sample={samples[0][:30]!r}")

    # Save
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps({
        "probes": [p for p, _ in PROBES_TO_TEST],
        "alphas": ALPHAS,
        "prompts": PROMPTS,
        "warm_words": sorted(WARM),
        "cool_words": sorted(COOL),
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'probe':40s} {'alpha':>6s} {'warm_ratio':>12s}  sample")
    for probe, alpha, ratio, sample in rows:
        print(f"{probe:40s} {alpha:>6.1f} {ratio:>12.3f}  {sample[:30]!r}")

    # Verdict
    print("\n=== Verdict ===")
    by_probe: dict[str, dict[float, float]] = {}
    for probe, alpha, ratio, _ in rows:
        by_probe.setdefault(probe, {})[alpha] = ratio
    for probe in by_probe:
        d = by_probe[probe]
        base = d.get(0.0, float("nan"))
        for a in (2.0, 5.0):
            if a in d:
                delta = d[a] - base
                print(f"  {probe} a={a}: Δwarm_ratio = {delta:+.3f}")


if __name__ == "__main__":
    main()
