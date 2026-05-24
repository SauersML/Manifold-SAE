"""auto_exp_43: L20 vs L40 free-block comparison — ABORTED at sanity probe.

GOAL: harvest cogito at L20 + L40 on a mini xkcd sample, run the HSV-supervised
gauge-fix per auto_exp_38 separately at each layer, and ask whether the 2D
name-semantic free-block structure (established at L40 by auto_exp_40) is
already present at L20 (depth-invariant) or only emerges at deeper layers.

OUTCOME: the cogito-probed inference server at <COGITO_API_BASE> hooks
ONLY layer 40 (per /v1/config — every probe targets layer 40; /v1/encode
with `"layers":[20]` returns `{"error":"Layer 20 not hooked. Available: [40]"}`).
Without a server-side hook reconfiguration (which is OUT OF SCOPE per the
shared-cluster rule), the L20 harvest cannot proceed. We do the sanity
probe, persist the server's self-report, and exit cleanly.

To unblock this experiment, the cogito-probed server would need to be
restarted with additional hook layers (e.g. `--hook-layers 20,40`); see
/v1/config "probes" field which currently lists "layer": 40 for every probe.
"""
from __future__ import annotations

import json
import time
import os
import urllib.request
from pathlib import Path

ROOT = Path("/Users/user/Manifold-SAE")
OUT_DIR = ROOT / "runs" / "COLOR_COGITO_L20_L40_mini"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COGITO_URL = os.environ.get(
    "COGITO_API_BASE", os.environ.get("COGITO_URL", "http://localhost:8000")
)


def _get(path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(COGITO_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        COGITO_URL + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    t0 = time.time()
    print(f"[auto_exp_43] sanity probe of {COGITO_URL}")

    # 1) /v1/status
    try:
        status = _get("/v1/status")
    except Exception as exc:
        print(f"[ABORT] /v1/status unreachable: {exc!r}")
        (OUT_DIR / "abort.json").write_text(json.dumps(
            {"reason": "status_unreachable", "exc": repr(exc)}, indent=2))
        return 2
    print(f"[status] model={status.get('model')} "
          f"n_active={status.get('queue', {}).get('n_active')} "
          f"uptime_s={status.get('queue', {}).get('uptime_s')}")

    # 2) /v1/models
    try:
        models = _get("/v1/models")
    except Exception as exc:
        models = {"error": repr(exc)}
    print(f"[models] {models}")

    # 3) /v1/config — discover which layers are actually hooked
    try:
        cfg = _get("/v1/config")
    except Exception as exc:
        cfg = {"error": repr(exc)}
    probe_layers = sorted({p.get("layer") for p in (cfg.get("probes") or [])
                           if isinstance(p, dict) and p.get("layer") is not None})
    print(f"[config] probe layers present: {probe_layers}")

    # 4) Probe both target layers with a tiny single-prompt encode request.
    layer_availability: dict[int, dict] = {}
    for L in (20, 40):
        try:
            resp = _post("/v1/encode", {
                "texts": ["A red dress."],
                "layers": [L],
                "aggregate": "mean",
                "max_length": 32,
            }, timeout=30.0)
            ok = "error" not in resp
            err = resp.get("error")
            d = None
            if ok:
                try:
                    arr = resp["results"][0][f"layer_{L}"]
                    d = len(arr)
                except Exception as exc:
                    err = f"unparseable: {exc!r}"
                    ok = False
            layer_availability[L] = {"ok": ok, "error": err, "d": d}
            print(f"[probe L={L}] ok={ok} d={d} err={err}")
        except Exception as exc:
            layer_availability[L] = {"ok": False, "error": repr(exc), "d": None}
            print(f"[probe L={L}] EXC {exc!r}")

    # 5) Persist the server self-report and the abort metadata.
    report = {
        "experiment": "auto_exp_43",
        "cogito_url": COGITO_URL,
        "status": status,
        "models": models,
        "config_probe_layers": probe_layers,
        "layer_availability": layer_availability,
        "runtime_s": time.time() - t0,
    }
    (OUT_DIR / "sanity_probe.json").write_text(json.dumps(report, indent=2))
    print(f"[saved] {OUT_DIR / 'sanity_probe.json'}")

    if not layer_availability.get(20, {}).get("ok"):
        verdict = ("ABORTED: cogito-probed server hooks only layer 40 "
                   "(per /v1/config probes + /v1/encode probe). L20 harvest "
                   "requires a server restart with --hook-layers 20,40 — "
                   "out of scope (shared cluster, do not touch config).")
        print(f"\n[VERDICT] {verdict}")
        (OUT_DIR / "abort.json").write_text(json.dumps(
            {"reason": "layer_20_not_hooked",
             "config_probe_layers": probe_layers,
             "layer_availability": layer_availability,
             "verdict": verdict}, indent=2))
        # Stdout table (empty rows for both layers since neither was harvested)
        print("\nlayer | hsv_R2_mean | name_active | top_free_axis_max_corr")
        print("------+-------------+-------------+-----------------------")
        print(f"  20  |     N/A     |     N/A     |    N/A (not hooked)")
        print(f"  40  |     N/A     |     N/A     |    N/A (skipped — no L20 comparator)")
        return 0

    # Unreachable today, but kept for future server reconfigurations.
    print("[INFO] Both layers hooked — implement full harvest+analysis here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
