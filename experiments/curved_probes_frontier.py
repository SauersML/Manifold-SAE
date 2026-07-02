"""Frontier-scale curved-feature probes on Qwen3-32B (WS-D harvest).

W7's ``curved_feature_probes.py`` proved (on Qwen2.5-0.5B) that a single curved
intrinsic coordinate reconstructs a cyclic calendar feature as well as *two* linear
PCs and recovers the cyclic token ordering a single linear direction cannot express.
This reruns the identical analysis on the frontier model: layer-{24,32,40} residual
activations of **Qwen3-32B** (d_model=5120) harvested by WS-D at real calendar/color
token sites (weekday/month/year/color).

It reuses W7's analysis verbatim (``analyze_set``, ``reml_corroborate``,
``write_reports``, ``plot_orderings``) so the frontier numbers are directly comparable
to the small-model ones; only the data loader changes — the WS-D shards are raw
``bfloat16`` residual dumps + a JSON manifest (labels / order / templates / cyclic).

Config (env):
  HARVEST_ROOT  dir of WS-D probe dirs (default /dev/shm/sauers_gpu/harvest)
  FRONTIER_OUT  output dir (default experiments/frontier_probe_out)
  FRONTIER_LAYERS   comma layers (default 24,32,40)
  FRONTIER_SETS     comma sets (default weekday,month,year,color)
  CURVED_PROBE_STEPS / CURVED_PROBE_RDIM  passed through to the W7 fit (default 600 / 16)
  FRONTIER_REML  "1" to also run the REML sae_manifold_fit cross-check (default "1")
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from curved_feature_probes import (  # noqa: E402
    analyze_set, reml_corroborate, write_reports, plot_orderings, _pca_reduce,
    _demean_per_template,
)

HARVEST_ROOT = Path(os.environ.get("HARVEST_ROOT", "/dev/shm/sauers_gpu/harvest"))
OUT_DIR = Path(os.environ.get("FRONTIER_OUT", os.path.join(_HERE, "frontier_probe_out")))
LAYERS = tuple(int(x) for x in os.environ.get("FRONTIER_LAYERS", "24,32,40").split(","))
SETS = os.environ.get("FRONTIER_SETS", "weekday,month,year,color").split(",")


def _read_bf16_shard(dir_path: Path, manifest: dict) -> np.ndarray:
    """Load a WS-D residual_shard_bf16 dir into a float64 (rows, d_model) array."""
    import torch

    d_model = int(manifest["d_model"])
    parts = []
    for sh in manifest["shards"]:
        raw = (dir_path / sh["file"]).read_bytes()
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        parts.append(t.reshape(int(sh["rows"]), d_model).float().numpy())
    X = np.concatenate(parts, 0).astype(np.float64)
    return X


def _template_idx_from_order(order) -> np.ndarray:
    """Within each ground-truth token group, number the rows 0..k-1 (template index).

    Works regardless of interleaving: each distinct token (``order`` value) contributes
    one row per template, so the running count within a group is the template index.
    """
    counts: dict = defaultdict(int)
    tidx = np.zeros(len(order), dtype=int)
    for i, o in enumerate(order):
        tidx[i] = counts[o]
        counts[o] += 1
    return tidx


def load_set(name: str) -> dict | None:
    """Build a W7 ``entry`` dict for one probe set across all requested layers."""
    entry = None
    for L in LAYERS:
        d = HARVEST_ROOT / f"qwen3_32b_probe_{name}_l{L}"
        mpath = d / "manifest.json"
        if not mpath.exists():
            print(f"[{name}] missing {mpath}; skipping layer {L}", flush=True)
            continue
        man = json.loads(mpath.read_text())
        X = _read_bf16_shard(d, man)
        if entry is None:
            order = list(man["order"])
            labels = list(man["labels"])
            cyclic = bool(man["cyclic"])
            n_labels = len(sorted(set(order)))
            entry = {
                "cyclic": cyclic,
                "rank": np.asarray(order, dtype=float),
                "label": labels,
                "template_idx": _template_idx_from_order(order),
                "n_labels": n_labels,
                "_model": man.get("model_name"),
                "_templates": man.get("templates"),
            }
        if X.shape[0] != len(entry["rank"]):
            print(f"[{name}] layer {L} row mismatch {X.shape[0]} vs {len(entry['rank'])}; skip",
                  flush=True)
            continue
        entry[L] = X
        print(f"[{name}] layer {L}: X={X.shape} cyclic={entry['cyclic']} "
              f"n_labels={entry['n_labels']}", flush=True)
    if entry is None or not any(L in entry for L in LAYERS):
        return None
    entry["_layers_present"] = [L for L in LAYERS if L in entry]
    return entry


def main() -> int:
    steps = int(os.environ.get("CURVED_PROBE_STEPS", "600"))
    reduce_dim = int(os.environ.get("CURVED_PROBE_RDIM", "16"))
    do_reml = os.environ.get("FRONTIER_REML", "1") == "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] root={HARVEST_ROOT} layers={LAYERS} sets={SETS} steps={steps} "
          f"reduce_dim={reduce_dim} reml={do_reml}", flush=True)

    results: dict = {}
    for name in SETS:
        entry = load_set(name)
        if entry is None:
            print(f"[{name}] no data; skipping", flush=True)
            continue
        present = entry["_layers_present"]
        res = analyze_set(name, entry, present, reduce_dim, steps)
        if do_reml:
            L = res["layer"]
            Xd = _demean_per_template(entry[L], entry["template_idx"])
            res["reml"] = reml_corroborate(Xd, entry["cyclic"], reduce_dim)
            print(f"[{name}] REML: {res['reml']}", flush=True)
        res["_model"] = entry.get("_model")
        results[name] = res
        print(f"[{name}] layer={res['layer']} curved_ev_insample="
              f"{res['curved_ev_full_insample']:.3f} "
              f"linear1={res['insample_ev']['linear_L1']:.3f} "
              f"linear2={res['insample_ev']['linear_L2']:.3f} "
              f"ordering={res['ordering_curved']}", flush=True)

    if not results:
        print("[FATAL] no probe set produced a result", flush=True)
        return 1

    meta = {
        "source": "WS-D Qwen3-32B residual harvest (frontier scale)",
        "model": next(iter(results.values())).get("_model", "Qwen3-32B"),
        "layers": list(LAYERS),
        "harvest_root": str(HARVEST_ROOT),
    }
    write_reports(results, OUT_DIR, meta)
    plot_orderings(results, OUT_DIR)
    print(f"\n[done] {OUT_DIR}/curved_feature_probes.json + summary.md + "
          f"recovered_orderings.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
