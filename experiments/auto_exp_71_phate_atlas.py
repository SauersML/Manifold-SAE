"""auto_exp_71 — PHATE / persistent-H1 / Mapper atlas of SAE atom-sets.

Falsifiable prediction
----------------------
The cogito-L40 color residual stream lives near a 1-manifold (the hue ring).
If Manifold-SAE successfully recovers this structure in its atom DICTIONARY
(not just in activations), the SET OF DECODER ATOMS — viewed as points in
the residual stream — should ALSO sit near a 1-manifold (a closed loop on
the unit sphere), so persistent H1 on the atom-atlas should show ONE
dominant cycle. By contrast, L1 and TopK do not enforce any manifold
structure on atoms, so their H1 spectra should be scattered with no
dominant cycle.

Verdict criterion:
    dominance_ratio = max_pers / runner_up_pers
    Pass for Manifold-SAE iff dominance_ratio ≥ 3.0  AND  ratio > both
    baselines' ratios.

Outputs
-------
- runs/phate_atlas/{model}_phate.png
- runs/phate_atlas/{model}_h1.png
- runs/phate_atlas/{model}_mapper.dot
- runs/phate_atlas/{model}_summary.json
- runs/auto_exp_71_phate_atlas.json   (this script's overall verdict)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phate_atlas import process_model, _color_centroids_from_harvest


BASELINE_MODELS = [
    ROOT / "runs" / "sae_comparison" / "model_topk.pt",
    ROOT / "runs" / "sae_comparison" / "model_l1.pt",
    ROOT / "runs" / "sae_comparison" / "model_manifold.pt",
]

# Opportunistically include any new SAE checkpoints that other agents have
# dropped onto disk — keep the experiment forward-compatible.
EXTRA_GLOBS = [
    "runs/DAS_SAE_COGITO_L40/*.pt",
    "runs/WASSERSTEIN_SAE_F128_M64/*.pt",
    "runs/COLOR_COGITO_MULTILAYER/crosscoder*.pt",
    "runs/adaptive_k/*.pt",
    "runs/V9_TOPK1/*.pt",
]


def main() -> None:
    out_dir = ROOT / "runs" / "phate_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)

    centroids = hues = None
    harvest = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
    if harvest.exists():
        try:
            centroids, hues = _color_centroids_from_harvest(harvest)
            print(f"[exp71] color centroids loaded: {centroids.shape}", flush=True)
        except Exception as e:
            print(f"[exp71] WARN harvest load failed: {e}", flush=True)

    models = [p for p in BASELINE_MODELS if p.exists()]
    for pat in EXTRA_GLOBS:
        models.extend(sorted(ROOT.glob(pat)))
    # dedupe while preserving order
    seen, uniq = set(), []
    for m in models:
        s = str(m.resolve())
        if s in seen:
            continue
        seen.add(s); uniq.append(m)
    print(f"[exp71] processing {len(uniq)} models", flush=True)

    summaries = []
    for m in uniq:
        try:
            s = process_model(m, out_dir, centroids, hues)
            print(f"[exp71] {s['name']}: dom_H1={s['dominant_h1_persistence']:.4f} "
                  f"runner={s['runner_up_h1_persistence']:.4f} "
                  f"ratio={s['dominance_ratio']:.2f} "
                  f"({s['n_h1_bars']} bars, {s['backend']})",
                  flush=True)
            summaries.append(s)
        except Exception as e:
            print(f"[exp71] ERROR {m}: {e}", flush=True)
            summaries.append({"name": m.stem, "error": str(e)})

    # ----- verdict -----
    by_name = {s.get("name", "?"): s for s in summaries if "error" not in s}
    man = by_name.get("model_manifold")
    l1 = by_name.get("model_l1")
    topk = by_name.get("model_topk")

    verdict: dict = {"summaries": summaries}
    if man and l1 and topk:
        man_r = man["dominance_ratio"]
        l1_r = l1["dominance_ratio"]
        topk_r = topk["dominance_ratio"]
        verdict["manifold_dominance_ratio"] = man_r
        verdict["l1_dominance_ratio"] = l1_r
        verdict["topk_dominance_ratio"] = topk_r
        verdict["manifold_has_dominant_cycle"] = bool(man_r >= 3.0)
        verdict["manifold_beats_baselines"] = bool(man_r > max(l1_r, topk_r))
        verdict["hue_ring_hypothesis_supported"] = (
            verdict["manifold_has_dominant_cycle"] and verdict["manifold_beats_baselines"]
        )
        print(
            "[exp71] VERDICT  manifold_ratio={:.2f}  l1_ratio={:.2f}  topk_ratio={:.2f}  "
            "supported={}".format(
                man_r, l1_r, topk_r, verdict["hue_ring_hypothesis_supported"]
            ),
            flush=True,
        )
    else:
        verdict["hue_ring_hypothesis_supported"] = None
        print("[exp71] WARN: missing one of {manifold,l1,topk}; verdict incomplete", flush=True)

    out_json = ROOT / "runs" / "auto_exp_71_phate_atlas.json"
    out_json.write_text(json.dumps(verdict, indent=2))
    print(f"[exp71] wrote verdict to {out_json}", flush=True)


if __name__ == "__main__":
    main()
