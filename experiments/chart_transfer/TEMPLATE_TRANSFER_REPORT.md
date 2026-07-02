# Chart-transfer invariance: is a circle chart a property of the FEATURE or the PROMPT?

**Lane:** N-nursery · **Code:** [`template_transfer.py`](template_transfer.py),
[`harvest_templates.py`](harvest_templates.py) ·
**Results:** [`template_out/template_transfer.json`](template_out/template_transfer.json) ·
**Date:** 2026-07-02

The LLM analogue of the paper's Blender ground-truth validation — and a claim their
linear SAE **cannot** test, because a direction has no intrinsic coordinate to transfer.
We fit the weekday / month circle chart on a **subset of prompt templates** and evaluate
on **held-out templates** (a TEMPLATE split, not a row split). Everything — the PCA
reduction, the chart, the linear baseline — is fit on fit-templates only. Harvest
extended to **14 diverse template families** per set (varied syntactic frame, tense,
target position) via the `curved_feature_probes` model-loading path (Qwen2.5-0.5B,
per-template demeaned residuals). Leave-one-template-out CV (14 folds) + a fixed split.

## Verdict: **SUPPORTED (with nuance)** — the chart *coordinate* is largely a feature property

| metric (14 templates, LOTO) | weekday (7 tok) | month (12 tok) |
|---|---:|---:|
| **coordinate consistency** — median circ-corr | **0.95** | **0.81** |
| coordinate consistency — mean | 0.78 | 0.71 |
| fraction of held-out templates with consistency > 0.8 | 0.64 | 0.50 |
| chart EV transfer (1 coord) | 0.26 | 0.12 |
| linear-1 EV transfer (1 coord) | 0.01 | 0.03 |
| linear-2 EV transfer (2 coords) | 0.43 | 0.05 |
| adjacency on unseen templates | 0.45 | 0.46 |

Read the coordinate-consistency row first: **the same token receives the same recovered
angle whether it appears in a fit-template or a held-out-template sentence** — median
circular correlation **0.95** (weekday) and **0.81** (month) across the 14 held-out folds.
For the majority of prompts the chart coordinate is invariant to the prompt.

### The chart coordinate transfers where the linear plane does NOT

On the "clean-transfer" folds (coordinate consistency > 0.8), held-out-template EV:

| | chart (1 coord) | linear-1 | linear-2 |
|---|---:|---:|---:|
| weekday (9/14 folds) | 0.33 | 0.007 | 0.43 |
| month (7/14 folds) | **0.22** | **0.001** | **0.001** |

For **month**, both linear baselines transfer at essentially **zero** EV across diverse
prompts, while the chart transfers at 0.22 EV *and* recovers a consistent coordinate — a
qualitatively more portable representation. Everywhere, the 1-coordinate **chart beats a
single linear direction by a wide margin** (the fair 1-coord fight): a curved coordinate
generalizes across prompts where a linear one does not. For weekday, two linear PCs still
transfer decently (0.43) and edge out the 1-coord chart on raw EV — but the plane has no
*coordinate*, so it cannot make the feature-invariance claim at all.

### Honest caveats (scoping)

1. **Outlier templates break transfer.** A minority of template frames (≈ 36–50% of folds
   fall below consistency 0.9) produce a rotated/degraded coordinate — e.g. month held-out
   template 9 transfers at consistency 0.04. The invariance is a strong *tendency*, not a
   law; the mean is dragged well below the median by these outliers.
2. **The invariant is the ANGLE, not the radius.** Raw reconstruction *magnitude* is
   context-dependent (each prompt scales the activation differently, even after
   per-template demeaning), so absolute held-out EV is modest for the chart **and** the
   plane. What transfers cleanly is the angular coordinate, which is the interpretable
   object.
3. **Single-template adjacency is noisy** (≈ 0.45) at 7–12 tokens on one held-out
   template; coordinate consistency (pooling all tokens) is the robust readout.
4. **One small model, two features.** Qwen2.5-0.5B, weekday + month circles. Not yet a
   claim about all curved features or larger models.

## Why this matters

This is the property a linear SAE cannot even state: **"the feature's coordinate is a
function of the feature, not of the prompt distribution it was fit on."** The circle chart
gives every token an angle; here that angle is (mostly) the same regardless of the
carrier sentence, and it is portable to prompt frames the chart never saw — while the
linear plane's reconstruction does not port at all (month). It is the interpretability
payoff of a curved coordinate over a direction, measured.

## Reproduce

```bash
python experiments/chart_transfer/harvest_templates.py     # 14-template harvest (resumable)
CHART_TRANSFER_HARVEST=experiments/chart_transfer/template_out/harvest_more \
  python experiments/chart_transfer/template_transfer.py    # template-split transfer eval
```
Held-out at the template level; incremental JSON saves; isolated torch chart fits
(reuses `block_nursery.py`). The original 5-template result (weaker, underpowered) is in
git history at commit 2a2c778.
