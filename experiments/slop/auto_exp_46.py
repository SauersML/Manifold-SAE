"""auto_exp_46: anchor-offset vs tangent steering on cogito-probed (L40).

Direct followup to auto_exp_44, which found:
  - t1_at_red @ alpha=2 shifts warm-ratio +0.165 (mild push warmer)
  - t1_at_blue @ alpha=2 shifts warm-ratio +0.317 (ALSO pushes warmer — wrong sign)
  - alpha=5 collapses generation
  - t1 columns are local Jacobian tangents on the U_3d color manifold, NOT
    semantic anchors. Hypothesis: color identity lives in the ANCHOR OFFSET
    (anchor_color - manifold_center), not in tangent directions.

This experiment tests directly: if the server exposes anchor_* (or bare
color-name) probes, contrast anchor_red/anchor_blue steering vs t1_at_red/
t1_at_blue at alpha in {0, 1, 2, 3} using the same color-agnostic prompts.

Verdict:
  - Anchor-offset hypothesis WINS if anchor_red @ a=2 pushes warmer AND
    anchor_blue @ a=2 pushes cooler (warm-ratio drops vs alpha=0).
  - If anchor probes ALSO both push warm, hypothesis is falsified; color
    identity lives elsewhere (HSV-axis subspace, or non-color_manifold probes).

Endpoint: <COGITO_API_BASE> (VPN required).
Uses /v1/chat/completions with SSE streaming (non-streaming returns
"not yet implemented"). /v1/encode is avoided (stuck queue from exp_43).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

BASE = os.environ.get(
    "COGITO_API_BASE", os.environ.get("COGITO_URL", "http://localhost:8000")
)
OUT = Path(__file__).resolve().parents[1] / "runs" / "auto_exp_46_anchor_vs_tangent"
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

ALPHAS = [0.0, 1.0, 2.0, 3.0]
TARGET_COLORS = ("red", "blue")  # opposed warm/cool anchors


# ---------------------------------------------------------------------------
# Probe-label categorization
# ---------------------------------------------------------------------------
TANGENT_RE = re.compile(r"^t\d+_at_(.+)$")
ANCHOR_PATTERNS = [
    re.compile(r"^anchor_(.+)$"),
    re.compile(r"^centroid_(.+)$"),
    re.compile(r"^mean_(.+)$"),
    re.compile(r"^offset_(.+)$"),
    re.compile(r"^(.+)_anchor$"),
]


def categorize(labels: list[str]) -> dict[str, dict[str, str]]:
    """Return {category: {color: full_label}}."""
    out: dict[str, dict[str, str]] = {
        "tangent": {}, "anchor": {}, "bare_color": {}, "other": {},
    }
    # Build bare-color set: labels that exactly match a color word (e.g. "red")
    color_words = WARM | COOL | {"purple", "pink", "brown", "black", "white",
                                 "gray", "grey", "magenta", "lime", "navy"}
    for lab in labels:
        m = TANGENT_RE.match(lab)
        if m:
            # Use the first tangent (t1) only for each color
            if lab.startswith("t1_at_"):
                out["tangent"][m.group(1)] = lab
            continue
        matched = False
        for pat in ANCHOR_PATTERNS:
            m = pat.match(lab)
            if m:
                out["anchor"][m.group(1)] = lab
                matched = True
                break
        if matched:
            continue
        if lab.lower() in color_words:
            out["bare_color"][lab.lower()] = lab
            continue
        out["other"][lab] = lab
    return out


# ---------------------------------------------------------------------------
# Generation harness (lifted from auto_exp_44)
# ---------------------------------------------------------------------------
def score(text: str) -> tuple[int, int, float]:
    tokens = [t.strip(".,!?;:'\"()[]").lower() for t in text.split()]
    w = sum(1 for t in tokens if t in WARM)
    c = sum(1 for t in tokens if t in COOL)
    ratio = w / (w + c) if (w + c) > 0 else float("nan")
    return w, c, ratio


def generate(prompt: str, probe: str | None, alpha: float,
             timeout: int = 30) -> str:
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
            "type": "steer", "probe": probe, "strength": float(alpha),
        }
    r = requests.post(f"{BASE}/v1/chat/completions", json=body,
                      timeout=timeout, stream=True)
    r.raise_for_status()
    parts: list[str] = []
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
                parts.append(piece)
        except (KeyError, IndexError, TypeError):
            continue
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Step 1: discover & categorize probes ----
    try:
        resp = requests.get(f"{BASE}/v1/probes", timeout=10)
        resp.raise_for_status()
        probes = resp.json()
    except Exception as e:
        msg = (f"ABORT: cannot reach {BASE} ({e}).\n"
               "  The cogito-probed server (<COGITO_API_BASE>) is not\n"
               "  responding (connection refused as of this run). Cannot list\n"
               "  color_manifold probe labels or run the steering harness.\n"
               "  Action: ask user to restart the probe server, then re-run.")
        print(msg)
        (OUT / "abort.txt").write_text(msg + "\n")
        return

    cm = next((p for p in probes["probes"] if p["name"] == "color_manifold"),
              None)
    if cm is None:
        print(f"ABORT: color_manifold probe set not loaded. Sets: "
              f"{[p['name'] for p in probes['probes']]}")
        return
    labels = list(cm["labels"])
    print(f"Loaded color_manifold: {cm['n_probes']} probes at L{cm['layer']}")
    print(f"Sample labels: {labels[:8]}{'...' if len(labels) > 8 else ''}")

    cats = categorize(labels)
    print("\n=== Probe categorization ===")
    for category, mapping in cats.items():
        print(f"  {category:12s} ({len(mapping):3d}): "
              f"{sorted(mapping.keys())[:8]}")

    # ---- Step 1b: pick anchor + tangent probes ----
    probes_to_test: list[tuple[str, str, str]] = []  # (probe, expected, ptype)
    for color in TARGET_COLORS:
        expected = "warm" if color in WARM else "cool"
        # Tangent baseline
        if color in cats["tangent"]:
            probes_to_test.append((cats["tangent"][color], expected, "tangent"))
        # Anchor probe (try anchor_X, then bare_color X)
        if color in cats["anchor"]:
            probes_to_test.append((cats["anchor"][color], expected, "anchor"))
        elif color in cats["bare_color"]:
            probes_to_test.append(
                (cats["bare_color"][color], expected, "bare_color"))

    have_anchor = any(pt in ("anchor", "bare_color") for _, _, pt in probes_to_test)
    if not have_anchor:
        msg = ("ABORT: no anchor/centroid/bare-color probes found in "
               "color_manifold set. All labels appear to be tangent columns "
               "(t1_at_X / t2_at_X / t3_at_X).\n"
               "  Hypothesis ('color identity lives in anchor offset') CANNOT\n"
               "  be tested with the currently-loaded probes. Action: server\n"
               "  restart with a probe set that exposes anchor directions\n"
               "  (anchor_color = manifold(color) - manifold_centroid), or\n"
               "  POST a new probe via /v1/probes if that endpoint supports it.\n"
               f"  Labels seen: {labels}")
        print(msg)
        (OUT / "abort.txt").write_text(msg + "\n")
        return

    print(f"\nWill test {len(probes_to_test)} probes x {len(ALPHAS)} alphas "
          f"x {len(PROMPTS)} prompts = "
          f"{len(probes_to_test)*len(ALPHAS)*len(PROMPTS)} generations")

    # ---- Step 2: steering harness ----
    results = []
    rows = []  # (ptype, probe, alpha, mean_ratio, sample)
    t0 = time.time()

    for probe, expected, ptype in probes_to_test:
        for alpha in ALPHAS:
            if time.time() - t0 > 600:
                print("Soft 10 min budget exceeded; stopping.")
                break
            warm_ratios = []
            samples = []
            for prompt in PROMPTS:
                try:
                    text = generate(prompt, probe, alpha, timeout=30)
                except requests.exceptions.RequestException as e:
                    print(f"  [{probe} a={alpha}] req fail: {e}; skip prompt")
                    text = ""
                w, c, ratio = score(text)
                results.append({
                    "probe": probe, "ptype": ptype, "expected": expected,
                    "alpha": alpha, "prompt": prompt, "text": text,
                    "warm": w, "cool": c, "ratio": ratio,
                })
                if ratio == ratio:  # not NaN
                    warm_ratios.append(ratio)
                samples.append(text[:60].replace("\n", " "))
            mean_ratio = (sum(warm_ratios) / len(warm_ratios)
                          if warm_ratios else float("nan"))
            rows.append((ptype, probe, alpha, mean_ratio,
                         samples[0] if samples else ""))
            print(f"  [{ptype:10s}] {probe:32s} a={alpha:>4.1f}  "
                  f"warm_ratio={mean_ratio:.3f}  "
                  f"sample={(samples[0] if samples else '')[:30]!r}")

    # ---- Step 3: save & verdict ----
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps({
        "probes": [{"probe": p, "ptype": pt, "expected": ex}
                   for p, ex, pt in probes_to_test],
        "alphas": ALPHAS,
        "prompts": PROMPTS,
        "warm_words": sorted(WARM),
        "cool_words": sorted(COOL),
        "categorization": {k: v for k, v in cats.items()},
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'ptype':12s} {'probe':32s} {'alpha':>6s} {'warm_ratio':>12s} "
          f"{'shift_vs_a0':>12s}")
    by_probe: dict[str, dict[float, float]] = {}
    for ptype, probe, alpha, ratio, _ in rows:
        by_probe.setdefault(probe, {})[alpha] = ratio
    for ptype, probe, alpha, ratio, _ in rows:
        base = by_probe[probe].get(0.0, float("nan"))
        shift = ratio - base
        print(f"{ptype:12s} {probe:32s} {alpha:>6.1f} {ratio:>12.3f} "
              f"{shift:>+12.3f}")

    # Verdict
    print("\n=== Verdict ===")
    target_a = 2.0
    by_color: dict[str, dict[str, float]] = {}  # color -> {ptype: shift_at_a2}
    for ptype, probe, alpha, ratio, _ in rows:
        if alpha != target_a:
            continue
        base = by_probe[probe].get(0.0, float("nan"))
        shift = ratio - base
        # Extract color from label
        m = TANGENT_RE.match(probe)
        color = m.group(1) if m else probe.replace("anchor_", "").replace(
            "centroid_", "").replace("_anchor", "").lower()
        by_color.setdefault(color, {})[ptype] = shift

    anchor_wins = True
    for color, shifts in by_color.items():
        print(f"  color={color}: {shifts}")
        expected_sign = +1 if color in WARM else -1
        anc = shifts.get("anchor", shifts.get("bare_color"))
        tan = shifts.get("tangent")
        if anc is None or tan is None:
            anchor_wins = False
            continue
        # Anchor must (a) point in expected direction and (b) exceed tangent
        # magnitude in that direction.
        if expected_sign * anc <= 0:
            anchor_wins = False
        if abs(anc) <= abs(tan) and expected_sign * anc <= expected_sign * tan:
            anchor_wins = False
    print(f"\n  ANCHOR-OFFSET HYPOTHESIS: "
          f"{'WINS' if anchor_wins else 'FALSIFIED / inconclusive'}")


if __name__ == "__main__":
    main()
