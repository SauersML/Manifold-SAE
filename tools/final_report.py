"""Unified findings aggregator — walks runs_cluster/ and emits a final
markdown report covering every recognized experiment schema.

Run after results have been fetched via fetch_results.sh:

    python tools/final_report.py --runs runs_cluster --output FINAL.md

Currently understands:
  llm_sweep        : matched-F vanilla vs curve EV + alive counts
  llm_probe        : holdout |ρ| per concept × layer
  realistic_scaling: synthetic curve recovery
  continuous_recovery: matched-decoder-params test
  atom_analysis    : polysemy + cross-layer + adv + probe
  atom_causality   : counterfactual + cross-SAE alignment
  axbench_lite     : steering KL + bucket-shift per method
  dbscan_manifold_fit: clusters + per-cluster manifold
  cyclic_probe     : weekday/month |ρ_circ|
  synthetic_2d_recovery: 2D atom recovery
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_runs(runs_dir: Path) -> dict[str, Path]:
    """Map run-name → directory."""
    out = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir(): continue
        if (d / "results.json").exists() or (d / "summary.json").exists() or any(d.glob("eval_*.json")):
            out[d.name] = d
    return out


def render_llm_sweep(name: str, d: Path) -> str:
    import glob
    files = sorted(glob.glob(str(d / "eval_F*.json")))
    if not files: return ""
    rows = []
    for f in files:
        try:
            r = json.load(open(f))["result"]
            rows.append(r)
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("F", 0))
    md = [f"### {name}\n",
          f"| F | TopK | vanilla EV | curve EV | locked EV | van alive | crv alive |\n",
          f"| --- | --- | --- | --- | --- | --- | --- |\n"]
    for r in rows:
        md.append(f"| {r.get('F','?')} | {r.get('top_k','?')} | "
                  f"{r.get('vanilla_explained',0):.4f} | "
                  f"{r.get('curve_explained',0):.4f} | "
                  f"{r.get('curve_locked_explained',0):.4f} | "
                  f"{r.get('vanilla_alive','?')} | {r.get('curve_alive','?')} |\n")
    md.append("\n")
    return "".join(md)


def render_llm_probe(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    md = [f"### {name}\n"]
    # Phase 2 holdout summary
    phase2 = r.get("phase2", {})
    rows = []
    for key, v in phase2.items():
        if not isinstance(v, dict): continue
        if "vanilla" in v and "curve" in v:
            vt = v["vanilla"].get("best_atom_holdout_rho")
            ct = v["curve"].get("best_atom_holdout_rho")
            if vt is None or ct is None: continue
            rows.append((key, vt, ct))
    if rows:
        md.append(f"| concept × layer | vanilla holdout \|ρ\| | curve holdout \|ρ\| |\n| --- | --- | --- |\n")
        for k, vt, ct in rows[:15]:
            md.append(f"| {k} | {vt:.3f} | **{ct:.3f}** |\n")
        md.append("\n")
    return "".join(md)


def render_realistic_scaling(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists():
        f = d / "summary.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    scenarios = r.get("scenarios") or r.get("results") or {}
    md = [f"### {name}\n",
          "| scenario | D | F | vanilla EV | curve EV | Δ EV |\n| --- | --- | --- | --- | --- | --- |\n"]
    for s_name, s in scenarios.items() if isinstance(scenarios, dict) else []:
        if not isinstance(s, dict): continue
        v = s.get("vanilla_explained") or s.get("vanilla", {}).get("explained")
        c = s.get("curve_explained") or s.get("curve", {}).get("explained")
        D = s.get("D") or s.get("d_ambient")
        F = s.get("F") or s.get("n_features")
        if v is None or c is None: continue
        md.append(f"| {s_name} | {D} | {F} | {v:.3f} | {c:.3f} | {c-v:+.3f} |\n")
    md.append("\n")
    return "".join(md)


def render_atom_analysis(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    md = [f"### {name}\n"]
    poly = r.get("polysemy", {})
    if poly:
        md.append(f"**Polysemy**: {poly.get('n_monosemantic',0)} monosemantic, "
                  f"{poly.get('n_polysemantic',0)} polysemantic, "
                  f"mean k={poly.get('mean_polysemy',0):.2f}\n\n")
    xl = r.get("cross_layer_transfer", {})
    if xl:
        md.append(f"**Cross-layer transfer**:\n\n")
        md.append(f"| layer | mean \|ρ\| | atoms > 0.3 |\n| --- | --- | --- |\n")
        for L, v in xl.items():
            md.append(f"| {L} | {v.get('mean_abs_corr',0):.3f} | "
                      f"{v.get('n_atoms_above_0.3','?')}/{v.get('n_atoms_evaluated','?')} |\n")
        md.append("\n")
    adv = r.get("adversarial_max", {}).get("per_atom", [])
    if adv:
        md.append(f"**Adversarial atom max** (top 5):\n\n")
        for entry in adv[:5]:
            tokens = " · ".join(repr(t) for t in entry.get("top5_nearest_tokens", [])[:3])
            md.append(f"- atom #{entry['atom']}: {tokens}\n")
        md.append("\n")
    probe = r.get("probe_classification", {})
    if probe:
        md.append(f"**Probe classification (magnitude bucket)**:\n\n")
        md.append(f"- raw activations: test={probe.get('accuracy_raw_test',0):.3f}\n")
        md.append(f"- SAE features:    test={probe.get('accuracy_sae_test',0):.3f}\n\n")
    return "".join(md)


def render_atom_causality(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    md = [f"### {name}\n"]
    abl = r.get("ablation", {})
    if abl:
        ratio = abl.get("target_mean_kl", 0) / max(abl.get("control_mean_kl", 1e-9), 1e-9)
        md.append(f"**Counterfactual ablation**:\n\n")
        md.append(f"- target atom {abl.get('target_atom')}: mean KL = {abl.get('target_mean_kl',0):.4f}\n")
        md.append(f"- control atom {abl.get('control_atom')}: mean KL = {abl.get('control_mean_kl',0):.4f}\n")
        md.append(f"- ratio target/control: {ratio:.2f}× ({'load-bearing' if ratio > 2 else 'no clear effect'})\n\n")
    al = r.get("alignment", {})
    if al:
        md.append(f"**Cross-SAE alignment** (Hungarian-match across seeds):\n\n")
        md.append(f"- mean |cos| of matched pairs: {al.get('mean_matched_similarity',0):.3f}\n")
        md.append(f"- pairs > 0.5 / 0.7 / 0.9: "
                  f"{al.get('n_matched_above_0.5','?')} / "
                  f"{al.get('n_matched_above_0.7','?')} / "
                  f"{al.get('n_matched_above_0.9','?')}  (of {al.get('n_atoms','?')})\n\n")
    return "".join(md)


def render_axbench(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    md = [f"### {name}\n",
          f"Target atom: {r.get('target_atom')}\n\n",
          "| method | param | mean KL | mean bucket shift |\n| --- | --- | --- | --- |\n"]
    for s in r.get("summary", []):
        md.append(f"| {s['method']} | {s['magnitude_param']:+.2f} | "
                  f"{s['mean_kl']:.4f} | {s['mean_bucket_shift']:+.3f} |\n")
    md.append("\n")
    return "".join(md)


def render_dbscan(name: str, d: Path) -> str:
    f = d / "catalog.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    db = r.get("dbscan", {})
    catalog = r.get("catalog", [])
    md = [f"### {name}\n",
          f"DBSCAN found **{db.get('n_clusters','?')} clusters** ({db.get('n_noise','?')} noise atoms)\n\n"]
    valid = [c for c in catalog if c.get("valid")]
    if valid:
        md.append(f"| cluster | n_atoms | firing tokens | arc length | PC1 var |\n| --- | --- | --- | --- | --- |\n")
        for c in valid[:10]:
            md.append(f"| {c.get('cluster_id','?')} | {c.get('n_atoms','?')} | "
                      f"{c.get('n_firing_tokens','?')} | {c.get('arc_length',0):.2f} | "
                      f"{c.get('principal_var_ratio',0):.2f} |\n")
        md.append("\n")
    return "".join(md)


def render_cyclic_probe(name: str, d: Path) -> str:
    f = d / "summary.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    results = r.get("results", {})
    md = [f"### {name}\n",
          "| task | curve best \|ρ_circ\| | atoms > 0.7 | vanilla best \|ρ_lin\| |\n| --- | --- | --- | --- |\n"]
    for task, v in results.items():
        md.append(f"| {task} | {v.get('curve_best_circ_rho',0):.3f} | "
                  f"{v.get('curve_n_above_strong','?')} | "
                  f"{v.get('vanilla_best_lin_rho',0):.3f} |\n")
    md.append("\n")
    return "".join(md)


def render_2d_recovery(name: str, d: Path) -> str:
    f = d / "results.json"
    if not f.exists(): return ""
    try: r = json.load(open(f))
    except: return ""
    rep = r.get("report", {})
    md = [f"### {name}\n",
          f"- Manifold-SAE 2D: EV = {rep.get('explained_2d',0):.3f}  "
          f"({rep.get('atoms_2d_alive','?')} alive; "
          f"{rep.get('n_atoms_2d_using_both_axes','?')} using both axes)\n",
          f"- Manifold-SAE 1D baseline: EV = {rep.get('explained_1d',0):.3f}  "
          f"({rep.get('atoms_1d_alive','?')} alive)\n\n",
          "**Per-grid recovery (2D best-atom vs 1D best-pair):**\n\n",
          "| grid | 2D score | 1D pair score | ρ_t (1D) | ρ_s (1D) |\n| --- | --- | --- | --- | --- |\n"]
    g2d = rep.get("per_grid_2d", [])
    g1d = rep.get("per_grid_1d_pair", [])
    for a, b in zip(g2d, g1d):
        md.append(f"| {a.get('grid','?')} | {a.get('score',0):.3f} | "
                  f"{b.get('pair_score',0):.3f} | {b.get('rho_t',0):.2f} | "
                  f"{b.get('rho_s',0):.2f} |\n")
    md.append("\n")
    return "".join(md)


SCHEMAS = [
    ("llm_sweep", render_llm_sweep),
    ("llm_probe", render_llm_probe),
    ("realistic_scaling", render_realistic_scaling),
    ("atom_analysis", render_atom_analysis),
    ("atom_causality", render_atom_causality),
    ("axbench", render_axbench),
    ("dbscan", render_dbscan),
    ("cyclic_probe", render_cyclic_probe),
    ("synthetic_2d_recovery", render_2d_recovery),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_cluster", help="runs root directory")
    parser.add_argument("--output", default="FINAL_RESULTS.md")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        print(f"[error] {runs_dir} not found", file=sys.stderr); return 1

    runs = find_runs(runs_dir)
    print(f"[report] found {len(runs)} run directories under {runs_dir}", flush=True)

    out = [f"# Manifold-SAE — Final results aggregator\n\n",
           f"Aggregated from `{runs_dir}/`. {len(runs)} run directories.\n\n"]

    for tag, render in SCHEMAS:
        chunk = ""
        for name, d in runs.items():
            if tag in name:
                rendered = render(name, d)
                if rendered:
                    chunk += rendered
        if chunk:
            out.append(f"## {tag}\n\n{chunk}")

    Path(args.output).write_text("".join(out))
    print(f"[report] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
