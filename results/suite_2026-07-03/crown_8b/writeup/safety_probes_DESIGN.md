# Safety-feature probes — design rationale (Track S)

Goal: point the manifold-SAE instrument at a feature that *matters for safety* and ask
whether it charts as cleanly as the calendar crown did — and, crucially, whether it has
an **intensity coordinate** (a curve you can move along) rather than merely a direction.

## Two graded features

Both are **graded ordinal** axes (an open intensity coordinate), unlike the calendar
features which are **cyclic categorical** (a wheel that wraps). That contrast is the
experiment: the same pipeline that gave the calendar a wraparound-correct closed circle
should give these an *open* chart (linear-competitive, or a partial arc of a circle) whose
coordinate is **monotone in the designed intensity grade**.

### Sycophancy = degree of agreement with a false claim
- **Template** (10): a user asserts a false/dubious claim with a confidence marker that
  varies across templates ("I'm certain", "Obviously", "Everyone knows", "I insist", ...).
  The assistant opens a one-word verdict.
- **Graded level** (7): the verdict, `false < mistaken < doubtful < debatable < plausible
  < correct < brilliant`. Since the claim is false, "correct"/"brilliant" *is* sycophancy
  (affirming a falsehood to please) and "false"/"mistaken" is honest correction. The level
  index is literally "degree of sycophantic agreement."
- The confidence marker is per-template context, demeaned out (a translation leaves every
  steering displacement invariant); its tier is kept as metadata for a pressure-vs-intensity
  sub-analysis.

### Refusal = compliance→refusal intensity
- **Template** (10): a borderline / petty-rule-bending request (locked out of your own
  door, sneaking snacks into a cinema, a slightly padded resume). Deliberately NOT
  operational harm — this is interpretability probe design, we never elicit harmful
  content; the axis is the *stance*, not the payload.
- **Graded level** (7): the opening stance token, `Certainly < Sure < Okay < Well < Hmm <
  Unfortunately < No`, from eager compliance to hard refusal.

## Why the intensity grading matters (the hypothesis under test)
Sycophancy is often modeled as a **direction** (a "be agreeable" vector). If that were the
whole story a `linear` atom would suffice and the levels would be equally spaced along one
axis. But sycophancy plausibly has **intensity** — a graded coordinate where mild
agreement and effusive flattery are different *distances*, possibly along a *bent* path
(e.g. flattery peels away from the correction–agreement line). So we fit both `linear` and
`circle` at matched K=1, d_atom=1 and let the data arbitrate. There is no dedicated
open-arc topology in the engine, which makes the circle a sharp probe: an open axis should
occupy only a **partial arc** of the circle (large wraparound gap, no last→first closure)
— the mirror image of the calendar's full-2π wraparound.

## Last-token convention
Every prompt ends on the graded token, so the downstream output-Fisher metric has a single
future position and the measured next-token KL (Stage B) is unambiguous — identical
convention to the calendar dose crown.

## Token-identity control
Different level words are different tokens. Controlled exactly as the calendar bank: the
same level set appears under every template, the fit runs on per-template demeaned
activations (a fixed per-word offset is shared across templates), and the ordering test
scores the per-level mean against a **permutation null over level relabelings** — a
pure-token-identity feature with no graded structure scores no better than that null.

## What "charts cleanly" means here (acceptance)
- **reconstruction_r2 > 0.9** for the preferred topology, AND
- **monotone**: Spearman(fitted coordinate, designed grade) high with permutation p small.
A negative is equally reportable: if the levels collapse to two clumps (pure agree/correct
binary, no interior) the Spearman stays near the null — sycophancy would then be a switch,
not a dial.

## Stages
- **Stage A (this dir, `fit_safety_charts.py`, acn116 CPU, forward-only):** harvest last-
  token activations, demean, fit linear+circle, report r2 + topology verdict + monotone
  ordering. No GPU, no Fisher.
- **Stage B (GPU, only if A charts cleanly):** attach the downstream output-Fisher metric,
  `steer` to a forecast Δ-nats reduction of agreement intensity, patch the residual, and
  measure the true output KL — the sycophancy analogue of the weekday dose figure.

## Files
- `safety_features.py` — the probe bank (levels, templates, pressure tiers).
- `fit_safety_charts.py` — Stage A harvest+fit+ordering+topology comparison.
- `run_fit_acn.sh` — runs Stage A on acn116.
- `fit_out/safety_charts.json` — results.
