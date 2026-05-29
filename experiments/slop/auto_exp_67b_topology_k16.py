"""auto_exp_67b: K=16 retry of auto_exp_67 topology selector.

auto_exp_67 caveat (project_topology_selector_cogito.md):
    "Holdout R^2 uniformly negative because template-noise dominates per-color
     K=64 signal; needs K<=16 for honest CV."

This re-runs the same 5-topology shootout (Euclidean / Circle / Sphere /
Torus / Cylinder) on K_PCS=16 (instead of 64) -- the higher-SNR head of the
PCA spectrum -- and reports whether Cylinder's holdout R^2 turns POSITIVE.

Everything else is identical to auto_exp_67: 3-fold CV across templates,
gamfit.gaussian_reml_fit per column, REML / BIC / TK / holdout-R^2 rankings.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/user/Manifold-SAE")
sys.path.insert(0, str(ROOT / "experiments"))

# Reuse all helpers from auto_exp_67 (load_xkcd_rgb, hsv_from_rgb,
# per_color_per_template_pcs, fourier_basis, *_design, etc.) -- only K_PCS
# differs.
spec = importlib.util.spec_from_file_location(
    "_auto_exp_67_mod", ROOT / "experiments" / "auto_exp_67_topology_selector.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Monkeypatch the global K_PCS used by per_color_per_template_pcs default arg.
mod.K_PCS = 16

OUT_DIR = ROOT / "runs" / "auto_exp_67b_topology_k16"
OUT_DIR.mkdir(parents=True, exist_ok=True)
mod.OUT_DIR = OUT_DIR
mod.OUT_JSON = OUT_DIR / "comparison_k16.json"
mod.OUT_PNG = OUT_DIR / "comparison_k16.png"

import gamfit
import json
import time
import warnings

def main():
    t_start = time.time()
    K_PCS = 16
    print(f"[auto_exp_67b] K=16 retry  gamfit={gamfit.__version__}")
    print(f"[data] mmap {mod.X_PATH}")
    X = np.load(mod.X_PATH, mmap_mode="r")
    n_c = X.shape[0] // mod.N_TEMPLATES
    print(f"[data] X={X.shape}  n_colors={n_c}")
    basis = mod.load_pc_basis(K=64)
    Z = mod.per_color_per_template_pcs(X, basis, K_PCS, mod.N_TEMPLATES)
    print(f"[stream] Z={Z.shape}  (K_PCS={K_PCS})")
    TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
    Y_global = Z[:, TOP_TEMPLATES, :].mean(axis=1)
    print(f"[stream] Y_global={Y_global.shape}")
    _names, rgb = mod.load_xkcd_rgb(n_c)
    hsv = mod.hsv_from_rgb(rgb)
    topologies = mod.build_topologies(hsv, rgb)

    rows = []
    for name, design_fn in topologies:
        t_fit = time.time()
        try:
            X_des, S_pen = design_fn()
        except Exception as exc:
            print(f"[{name}] DESIGN FAILED: {exc!r}")
            rows.append({"topology": name, "status": "design_failed", "error": repr(exc)})
            continue
        print(f"[{name}] design={X_des.shape}  penalty={S_pen.shape}")
        res = mod.fit_one_topology(X_des, S_pen, Y_global, name)
        if "error" in res:
            print(f"[{name}] FIT FAILED: {res['error']}")
            rows.append({"topology": name, "status": "fit_failed", "error": res["error"]})
            continue
        r2_mean, r2_std = mod.holdout_r2_by_template(
            Z, lambda hsv=hsv: None, design_fn, name=name, n_folds=3, seed=67,
        )
        elapsed = time.time() - t_fit
        rows.append({
            "topology": name, "status": "ok",
            "reml": res["reml"], "bic": res["bic"], "tk": res["tk"],
            "edf": res["edf"], "sse": res["sse"],
            "n_params": res["n_params"], "n_obs": res["n_obs"],
            "holdout_r2_mean": r2_mean, "holdout_r2_std": r2_std,
            "fit_failures": res["fit_failures"], "elapsed_sec": elapsed,
        })
        print(f"[{name}] REML={res['reml']:.2f}  BIC={res['bic']:.2f}  "
              f"R^2={r2_mean:.3f}+/-{r2_std:.3f}  ({elapsed:.1f}s)")

    ok = [r for r in rows if r.get("status") == "ok"]
    by_reml = sorted(ok, key=lambda r: r["reml"])
    by_bic = sorted(ok, key=lambda r: r["bic"])
    by_r2 = sorted(ok, key=lambda r: -r["holdout_r2_mean"])
    print()
    print(f"[ranking REML] " + " < ".join(f"{r['topology']}({r['reml']:.1f})" for r in by_reml))
    print(f"[ranking BIC]  " + " < ".join(f"{r['topology']}({r['bic']:.1f})" for r in by_bic))
    print(f"[ranking R^2]  " + " > ".join(f"{r['topology']}({r['holdout_r2_mean']:.3f})" for r in by_r2))

    cyl = next((r for r in ok if r["topology"] == "Cylinder"), None)
    cyl_r2 = cyl["holdout_r2_mean"] if cyl else float("nan")
    verdict = "POSITIVE" if (cyl is not None and cyl_r2 > 0) else "STILL NEGATIVE"
    print(f"[verdict] Cylinder holdout R^2 at K=16: {cyl_r2:.4f}  -> {verdict}")

    summary = {
        "experiment": "auto_exp_67b_topology_k16",
        "k_pcs": K_PCS,
        "n_colors": int(n_c),
        "rows": rows,
        "cylinder_holdout_r2": cyl_r2,
        "cylinder_verdict": verdict,
        "ranking_reml": [r["topology"] for r in by_reml],
        "ranking_bic": [r["topology"] for r in by_bic],
        "ranking_r2": [r["topology"] for r in by_r2],
        "runtime_sec": time.time() - t_start,
    }
    with open(mod.OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[json] saved {mod.OUT_JSON}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
