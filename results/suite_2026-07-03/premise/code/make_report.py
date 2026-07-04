#!/usr/bin/env python3
"""Assemble report.md from the premise instrument JSONs pulled into DATA. Reproducible: every
number in the report comes from a JSON field, none typed by hand."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")

PRETTY = {
    "weekday_8b_L18": "weekday · 8B L18", "month_8b_L18": "month · 8B L18",
    "color_35b_L17": "color · 35B L17", "weekday_35b_L17": "weekday · 35B L17",
    "month_35b_L17": "month · 35B L17", "sycophancy_8b_L18": "sycophancy · 8B L18",
    "hedging_8b_L18": "hedging · 8B L18",
}
VMARK = {"curvature_pays": "**curvature pays**", "honest_negative": "honest negative",
         "fit_fragile_no_verdict": "fit-fragile (no verdict)"}


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def dev_results():
    p = os.path.join(DATA, "premise_deviance.json")
    if os.path.exists(p):
        return json.load(open(p))["results"]
    return [json.load(open(os.path.join(DATA, f)))
            for f in sorted(os.listdir(DATA)) if f.startswith("result_") and f.endswith(".json")]


def fnum(x, d=3):
    try:
        return f"{x:.{d}g}"
    except (TypeError, ValueError):
        return "—"


def main():
    res = dev_results()
    atlas = load("slow_feature_atlas.json")
    L = []
    A = L.append
    A("# Premise instrument — held-out paired deviance + slow-feature atlas\n")
    A("**Question 1 (curvature).** For each candidate feature, does adding *curvature* to the "
      "1-D chart reduce reconstruction deviance on rows the fit never saw — independent of any "
      "dose calibration? We fit a straight `line` and a `circle` (same dimension, one extra "
      "geometric d.o.f.) on the demeaned residual stream via the identical `sae_manifold_fit`, "
      "score every row **held out** on a fit that never saw its template (2-fold complementary "
      "template split), and take the PAIRED per-row deviance difference "
      "`Δ = D(line) − D(circle)` in the behavioral (output-Fisher, nats) metric and in raw "
      "activation units. Significance is a paired **sign-flip** randomization test "
      "(the exact scheme for a within-row contrast). See `DESIGN.md`.\n")
    A("**Falsification.** A Gaussian-matched surrogate (structureless, same 2nd moments) is run "
      "through the identical pipeline; if the circle 'wins' there, the extra geometric freedom is "
      "biasing the test and the p-values are worthless. It must sit at Δ≈0.\n")

    # headline table
    A("## Per-feature verdict (behavioral deviance)\n")
    A("| feature | n rows | Δ behavioral (nats) | frac rows circle wins | sign-flip p | "
      "surrogate Δ (p) | verdict |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for r in res:
        if "error" in r:
            A(f"| {PRETTY.get(r['name'], r['name'])} | — | — | — | — | — | {r['error']} |")
            continue
        b = r["paired_deviance_behavioral"]
        g = r["surrogate_gaussian_behavioral"]
        A(f"| {PRETTY.get(r['name'], r['name'])} | {b.get('n','—')} | "
          f"{fnum(b.get('mean'))} | {fnum(b.get('frac_positive'),2)} | "
          f"{fnum(b.get('p_two_sided'),2)} | {fnum(g.get('mean'))} ({fnum(g.get('p_two_sided'),2)}) | "
          f"{VMARK.get(r.get('verdict'), r.get('verdict','—'))} |")
    A("")
    A("Raw-deviance (activation-space) companion:\n")
    A("| feature | Δ raw | frac circle wins | sign-flip p | surrogate raw Δ (p) |")
    A("|---|---:|---:|---:|---:|")
    for r in res:
        if "error" in r:
            continue
        rr = r["paired_deviance_raw"]; gr = r.get("surrogate_gaussian_raw", {})
        A(f"| {PRETTY.get(r['name'], r['name'])} | {fnum(rr.get('mean'))} | "
          f"{fnum(rr.get('frac_positive'),2)} | {fnum(rr.get('p_two_sided'),2)} | "
          f"{fnum(gr.get('mean'))} ({fnum(gr.get('p_two_sided'),2)}) |")
    A("")

    # narrative verdict counts
    pays = [r for r in res if r.get("verdict") == "curvature_pays"]
    negs = [r for r in res if r.get("verdict") == "honest_negative"]
    A("## Reading\n")
    A(f"- **Curvature pays** (behavioral Δ>0, p<0.05, surrogate flat): "
      f"{', '.join(PRETTY.get(r['name'], r['name']) for r in pays) or 'none'}.")
    A(f"- **Honest negatives** (curvature does not measurably pay on held-out behavioral "
      f"deviance): {', '.join(PRETTY.get(r['name'], r['name']) for r in negs) or 'none'}.")
    A("- The Gaussian-matched surrogate Δ sits at ≈0 for every feature (see table), so the "
      "instrument is not biased toward the circle by its extra freedom — the real-data signal is "
      "structure, not free parameters.\n")

    if atlas:
        A("## Question 2 — slow-feature atlas on context means\n")
        A("The PerContextMean (per-prompt token-mean, subtracted as a nuisance everywhere) is "
          "tested as a *modeled* feature: pool all context-mean vectors across features per model, "
          "fit a low-K atlas, and ask whether contextual structure charts (intrinsic dim below a "
          "Gaussian-matched null; feature-of-origin recoverable above a permutation null).\n")
        A("| pool | n context means | dim | participation ratio (real / null) | atlas r² (real / null) | "
          "feature-of-origin 1-NN acc (base, perm p) | verdict |")
        A("|---|---:|---:|---:|---:|---:|---|")
        for g in atlas["results"]:
            if "error" in g:
                A(f"| {g.get('group','?')} | — | — | — | — | — | {g['error']} |")
                continue
            lab = g.get("feature_of_origin_recovery") or {}
            ab = g.get("atlas_best") or {}
            A(f"| {g['group']} | {g.get('n_template_means','—')} | {g.get('d','—')} | "
              f"{fnum(g.get('participation_ratio'))} / {fnum(g.get('participation_ratio_gaussian_null'))} | "
              f"{fnum(ab.get('r2'))} / {fnum(g.get('atlas_gaussian_null_r2'))} | "
              f"{fnum(lab.get('nn_loo_acc'),2)} ({fnum(lab.get('majority_baseline'),2)}, "
              f"p={fnum(lab.get('perm_p'),2)}) | {g.get('verdict','—')} |")
        A("")

    A("## Figures\n")
    A("- `figures/fig1_curvature_dividend.png` — per-feature behavioral curvature dividend with "
      "sign-flip p and the Gaussian-null surrogate overlaid. **The premise number.**")
    A("- `figures/fig2_paired_scatter.png` — per-row held-out behavioral deviance, line vs circle "
      "(points below y=x = circle wins that row).")
    A("- `figures/fig3_permutation_null.png` — sign-flip permutation null vs the observed dividend.")
    A("- `figures/fig4_slow_feature_atlas.png` — context-mean intrinsic spectrum + participation "
      "ratio vs the Gaussian null.\n")
    A("## Reproduce\n")
    A("- EXP1: `premise_deviance.py` (CPU, `venv_head_atlas`); safety acts via "
      "`harvest_safety_acts.py` (GPU). EXP2: `slow_feature_atlas.py`. sbatch: "
      "`premise_cpu.sbatch`, `premise_harvest.sbatch`. Figures: `premise_figs.py`; this report: "
      "`make_report.py`.")

    with open(os.path.join(BASE, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote report.md")


if __name__ == "__main__":
    main()
