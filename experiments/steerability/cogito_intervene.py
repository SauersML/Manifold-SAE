"""Cogito intervention runner -- STAGED for VPN-up execution.

Mirror of `experiments/steering_causality.py`'s baseline/patched logits
loop but targeted at the remote cogito server (instrumented vLLM,
documented privately, NOT hardcoded here).  Reads its endpoint from
the same env var the rest of the repo uses:

  * `COGITO_API_BASE` (e.g. http://localhost:8000) -- required for live mode
  * `COGITO_MODEL`    (defaults to "cogito")
  * `COGITO_LAYER`    (defaults to 40)

NO prior cogito-intervention helper was found in `scripts/`
(grep across scripts/ returned nothing for "intervention", "extra_body",
"hidden_state", or "/v1/completions").  We therefore build a thin
adapter against the OpenAI-compatible vLLM REST surface; the chosen
intervention payload follows the conservative shape:

  POST {api}/v1/completions
  {
    "model": MODEL,
    "prompt": P,
    "max_tokens": 1,
    "logprobs": top_k_logits,
    "temperature": 0,
    "extra_body": {
      "interventions": [
        {"layer": 40, "vector": [...D...], "scale": alpha, "position": "last"}
      ],
      "return_hidden_states": [40]
    }
  }

This payload shape is consistent with the existing in-house vLLM
instrumentation pattern (read off Heimdall job specs).  It will need a
one-line edit if the real shape differs -- the live --dry-run prints
the exact request so the operator can verify before any non-dry call.

Usage (offline, design time):
  python cogito_intervene.py --axis hue --prompts color_prompts.json --dry-run

Live (VPN up):
  COGITO_API_BASE=http://node1.datasci.ath:8000 \\
  python cogito_intervene.py --axis hue --prompts color_prompts.json

Output: appends results to `cogito_intervention_results.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE / "color_prompts.json"
DEFAULT_OUT = HERE / "cogito_intervention_results.jsonl"
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")


AXIS_FILES = {
    "hue":         (HERE / "hsv_axes_L40.npz",   "axis_hue"),
    "sat":         (HERE / "hsv_axes_L40.npz",   "axis_sat"),
    "val":         (HERE / "hsv_axes_L40.npz",   "axis_val"),
    "pc1":         (HERE / "u3d_axes_L40.npz",   "axis_pc1"),
    "pc2":         (HERE / "u3d_axes_L40.npz",   "axis_pc2"),
    "pc3":         (HERE / "u3d_axes_L40.npz",   "axis_pc3"),
    "red":         (HERE / "concept_axes_L40.npz", "axis_red"),
    "blue":        (HERE / "concept_axes_L40.npz", "axis_blue"),
    "green":       (HERE / "concept_axes_L40.npz", "axis_green"),
    "achromatic":  (HERE / "concept_axes_L40.npz", "axis_achromatic"),
}


def load_axis(name: str) -> np.ndarray:
    if name == "random":
        rng = np.random.default_rng(int(os.environ.get("MSAE_RANDOM_SEED", "1")))
        # match the dimensionality of the real axes
        path, key = AXIS_FILES["hue"]
        d = np.load(path)
        v = rng.standard_normal(d[key].shape[0]).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)
    if name not in AXIS_FILES:
        raise SystemExit(f"unknown axis: {name}; choose from {list(AXIS_FILES) + ['random']}")
    path, key = AXIS_FILES[name]
    if not path.exists():
        raise SystemExit(f"missing axis cache {path}; run steering_vectors.py first")
    return np.asarray(np.load(path)[key], dtype=np.float32)


def axis_scale_unit(axis: np.ndarray, n_samples: int = 4096, seed: int = 0) -> float:
    """1-sigma along this axis on the harvest residual distribution."""
    if not HARVEST.exists():
        return 1.0
    X = np.load(HARVEST, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    block = np.asarray(X[idx], dtype=np.float32)
    proj = block @ axis
    return float(proj.std())


def cogito_call(api_base: str, model: str, prompt: str, *, layer: int,
                intervention_vec: np.ndarray | None, alpha: float,
                top_k: int, request_hidden: bool,
                dry_run: bool, timeout: float = 120.0) -> dict:
    """One call to the cogito completion API.  When dry_run, only
    returns the request that *would* be sent."""
    extra: dict = {}
    if intervention_vec is not None:
        extra["interventions"] = [
            {
                "layer": int(layer),
                "vector": intervention_vec.astype(np.float32).tolist(),
                "scale": float(alpha),
                "position": "last",
            }
        ]
    if request_hidden:
        extra["return_hidden_states"] = [int(layer)]

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "logprobs": int(top_k),
        "temperature": 0.0,
    }
    if extra:
        body["extra_body"] = extra

    if dry_run:
        # Don't dump the giant vector for legibility -- truncate.
        body_repr = json.loads(json.dumps(body))
        if "extra_body" in body_repr and "interventions" in body_repr["extra_body"]:
            for iv in body_repr["extra_body"]["interventions"]:
                v = iv["vector"]
                iv["vector"] = f"<len={len(v)} first3={v[:3]} ... last3={v[-3:]}>"
        print(f"[dry-run] POST {api_base}/v1/completions")
        print(json.dumps(body_repr, indent=2))
        return {"_dry_run": True, "request": body_repr}

    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": repr(e), "elapsed": time.time() - t0}
    payload["_elapsed"] = time.time() - t0
    return payload


def extract_top_logprobs(resp: dict) -> dict[str, float]:
    """OpenAI-compatible completion logprob extraction."""
    if "error" in resp or "_dry_run" in resp:
        return {}
    try:
        ch = resp["choices"][0]
        lp = ch.get("logprobs", {})
        top = lp.get("top_logprobs") or lp.get("topLogprobs") or []
        if isinstance(top, list) and top and isinstance(top[0], dict):
            return {tok: float(v) for tok, v in top[0].items()}
        # Some servers return logprobs differently; fall back to the
        # canonical {"tokens": [...], "top_logprobs": [{tok: lp}]}.
        return {}
    except (KeyError, IndexError, TypeError):
        return {}


def kl_from_logprobs(p_log: dict[str, float], q_log: dict[str, float]) -> float:
    """KL(p || q) over the intersection of token sets."""
    keys = set(p_log) & set(q_log)
    if not keys:
        return float("nan")
    p = np.array([p_log[k] for k in keys])
    q = np.array([q_log[k] for k in keys])
    # logprobs -> probs
    p = np.exp(p - p.max()); p /= p.sum()
    q = np.exp(q - q.max()); q /= q.sum()
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    help=f"one of {list(AXIS_FILES) + ['random']}")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--out",     default=str(DEFAULT_OUT))
    ap.add_argument("--alphas", default="-3,-2,-1,0,1,2,3",
                    help="comma-separated alpha multiples of axis-sigma")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--layer", type=int,
                    default=int(os.environ.get("COGITO_LAYER", "40")))
    ap.add_argument("--model", default=os.environ.get("COGITO_MODEL", "cogito"))
    ap.add_argument("--api-base", default=os.environ.get("COGITO_API_BASE", ""))
    ap.add_argument("--dry-run", action="store_true",
                    help="print intended requests; do NOT hit the network")
    ap.add_argument("--limit-prompts", type=int, default=None)
    args = ap.parse_args()

    if not args.dry_run and not args.api_base:
        raise SystemExit(
            "Set COGITO_API_BASE or pass --api-base, or run with --dry-run."
        )

    alphas = [float(a) for a in args.axis.split(",")] if False else \
             [float(a) for a in args.alphas.split(",")]
    axis = load_axis(args.axis)
    scale_unit = axis_scale_unit(axis)
    print(f"[axis={args.axis}] dim={axis.shape[0]} ||a||={np.linalg.norm(axis):.4f} "
          f"sigma_on_harvest={scale_unit:.4f}", flush=True)

    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    n = 0
    with open(args.out, "a") as f:
        for p in prompts:
            # Baseline (alpha=0, no intervention)
            base = cogito_call(args.api_base, args.model, p["prompt"],
                               layer=args.layer, intervention_vec=None,
                               alpha=0.0, top_k=args.top_k,
                               request_hidden=True, dry_run=args.dry_run)
            base_lp = extract_top_logprobs(base)
            for a in alphas:
                if a == 0.0:
                    inter = base
                    inter_lp = base_lp
                else:
                    inter = cogito_call(
                        args.api_base, args.model, p["prompt"],
                        layer=args.layer, intervention_vec=axis,
                        alpha=a * scale_unit, top_k=args.top_k,
                        request_hidden=False, dry_run=args.dry_run,
                    )
                    inter_lp = extract_top_logprobs(inter)
                row = {
                    "axis": args.axis,
                    "alpha": a,
                    "alpha_abs": a * scale_unit,
                    "scale_unit": scale_unit,
                    "prompt_id": p["id"],
                    "prompt": p["prompt"],
                    "ground_truth_color": p.get("ground_truth_color"),
                    "expected_top_token": p.get("expected_top_token"),
                    "baseline_top_logprobs": base_lp,
                    "intervened_top_logprobs": inter_lp,
                    "kl_intervened_vs_baseline": kl_from_logprobs(inter_lp, base_lp),
                    "ts": time.time(),
                }
                if args.dry_run:
                    row["dry_run"] = True
                f.write(json.dumps(row, default=float) + "\n")
                n += 1
                if not args.dry_run:
                    print(f"  prompt={p['id']:14s} alpha={a:+.1f} "
                          f"KL={row['kl_intervened_vs_baseline']:.4f}", flush=True)
    print(f"[done] wrote {n} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
