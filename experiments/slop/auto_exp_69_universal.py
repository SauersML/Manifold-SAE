"""auto_exp_69 — Universal-SAE cross-model HSV hue-ring test.

Falsifiable prediction
----------------------
Train a shared-encoder / per-model-decoder SAE on paired (cogito-L40,
qwen-0.5B-L12) activations on the xkcd-color × 4-template prompt set.
If both:

  (a) ≥ 50% of *alive* atoms are "universal" (decoder has ≥ 15% mass in
      both models), AND
  (b) projecting each model's per-color centroid onto the universal-atom
      subspace yields a 2D plane whose angle correlates with HSV hue
      (circular-correlation ≥ 0.3 in BOTH models),

then the hue-ring feature is model-universal. Otherwise it is cogito-
specific (or model-pair-specific in the within-model fallback).

Fallback
--------
If the Qwen harvest output is missing (CPU/MPS too slow or local
``transformers`` unavailable), we fall back to a within-cogito layer-pair:
cogito-L40 split into two disjoint template subsets, treated as
"different models" — this is NOT cross-model and is documented as such in
metrics.json["fallback"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.cross_llm.universal_sae import UniversalSAE  # noqa: E402
from scripts.train_universal_cogito_qwen import (  # noqa: E402
    load_paired, train, analyze_and_plot,
)


COGITO_X_PATH = ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"
QWEN_DEFAULT_DIR = ROOT / "runs/COLOR_QWEN_05B_L12"
OUT_DIR = ROOT / "runs/auto_exp_69_universal"
COGITO_N_TEMPLATES = 28


def real_cross_model_run(out_dir: Path) -> dict:
    print("[auto_69] cross-model mode: cogito-L40 + qwen-0.5B-L12", flush=True)
    X_by_model, t_idx, hue = load_paired(COGITO_X_PATH, QWEN_DEFAULT_DIR)
    sae = train(X_by_model, F=256, top_k=16, epochs=400, batch_size=256,
                device="cpu")
    metrics = analyze_and_plot(
        sae, X_by_model, hue_per_color=hue, n_templates=len(t_idx),
        out_dir=out_dir, universality_threshold=0.15,
    )
    metrics["fallback"] = False
    metrics["mode"] = "cogito_L40 + qwen_05B_L12"
    return metrics


def fallback_within_cogito(out_dir: Path) -> dict:
    """Within-cogito layer-pair fallback.

    Splits cogito-L40 templates into two disjoint subsets and treats each
    subset's per-color centroid as a separate "model". This is NOT a
    cross-model test — it only checks the UniversalSAE machinery and the
    per-color-centroid hue-ring metric.
    """
    print("[auto_69] FALLBACK: within-cogito template-split", flush=True)
    import colorsys
    from manifold_sae.cross_llm.harvest_local import load_xkcd_colors

    X_full = np.load(COGITO_X_PATH, mmap_mode="r")
    n_colors = X_full.shape[0] // COGITO_N_TEMPLATES

    # Pick two disjoint 4-template subsets (cosmetic separation).
    t_a = [0, 7, 16, 24]
    t_b = [1, 8, 17, 25]

    def stack(t_idx):
        rows = np.array([ci * COGITO_N_TEMPLATES + ti
                         for ci in range(n_colors) for ti in t_idx],
                        dtype=np.int64)
        return np.ascontiguousarray(X_full[rows], dtype=np.float32)

    X_a = stack(t_a)
    X_b = stack(t_b)
    X_by_model = {"cogito_L40_setA": X_a, "cogito_L40_setB": X_b}

    colors = load_xkcd_colors()[:n_colors]
    hue = np.array([colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0]
                    for _, (r, g, b) in colors], dtype=np.float32)

    sae = train(X_by_model, F=128, top_k=12, epochs=200, batch_size=256,
                device="cpu")
    metrics = analyze_and_plot(
        sae, X_by_model, hue_per_color=hue, n_templates=len(t_a),
        out_dir=out_dir, universality_threshold=0.15,
    )
    metrics["fallback"] = True
    metrics["mode"] = "WITHIN-cogito template-split (NOT cross-model)"
    return metrics


def verdict(metrics: dict, *, frac_threshold: float = 0.5,
            circ_threshold: float = 0.3) -> dict:
    n_alive = metrics["n_alive"]
    n_alive_univ = metrics["n_alive_and_universal"]
    frac = n_alive_univ / max(n_alive, 1)
    circ = metrics.get("circular_correlation_hue_vs_proj", {})
    min_circ = min((abs(v) for v in circ.values()), default=0.0)

    passes_frac = frac >= frac_threshold
    passes_ring = min_circ >= circ_threshold
    passes = passes_frac and passes_ring

    return {
        "alive_universal_fraction": frac,
        "min_abs_circ_corr": min_circ,
        "passes_universality_threshold": passes_frac,
        "passes_hue_ring_threshold": passes_ring,
        "verdict": (
            "MODEL-UNIVERSAL hue feature" if passes else
            ("MODEL-SPECIFIC: failed universality"
             if not passes_frac else "MODEL-SPECIFIC: failed hue-ring")
        ),
        "frac_threshold": frac_threshold,
        "circ_threshold": circ_threshold,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qwen_exists = (QWEN_DEFAULT_DIR / "X.npy").exists() and \
                  (QWEN_DEFAULT_DIR / "meta.json").exists()
    if qwen_exists:
        try:
            metrics = real_cross_model_run(OUT_DIR)
        except Exception as e:
            print(f"[auto_69] cross-model run failed: {e!r} — falling back",
                  flush=True)
            metrics = fallback_within_cogito(OUT_DIR)
    else:
        print(f"[auto_69] no qwen harvest at {QWEN_DEFAULT_DIR}; "
              "running within-cogito fallback", flush=True)
        metrics = fallback_within_cogito(OUT_DIR)

    v = verdict(metrics)
    metrics["verdict"] = v
    (OUT_DIR / "verdict.json").write_text(json.dumps(metrics, indent=2))
    print("=" * 60)
    print(json.dumps(v, indent=2))
    print("=" * 60)
    print(f"figures: {OUT_DIR/'universal_hue_ring.png'} , "
          f"{OUT_DIR/'affinity.png'}")


if __name__ == "__main__":
    main()
