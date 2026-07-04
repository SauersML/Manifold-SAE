# Premise instrument — held-out paired deviance + slow-feature atlas

The program's premise is *"curvature exists and pays."* The dose-calibration crown shows the
curved chart **forecasts** edit effects in nats. But calibration mixes two claims: (a) the
geometry is genuinely curved, and (b) the pulled-back Fisher metric is well-normalized. This
instrument isolates (a): **does adding curvature reduce held-out reconstruction deviance,
per feature, independent of any dose calibration?**

## Experiment 1 — held-out paired deviance

### Data
Per feature, the harvest cache gives, for each of `n = n_templates × n_categories` prompts
(template-major build, so the original prompt index `kept[i]` has template `kept[i] // C`,
category `kept[i] % C`, with `C` = number of category words — weekday 7, month 12, color 8,
sycophancy/hedging 7 graded levels; the safety caches additionally store the template/level
arrays directly):
- `X_last (n, p)` — layer-L last-token residual stream,
- `tmpl_mean (n, p)` — the **per-prompt context mean** (the mean of that one prompt's
  activations over its token positions; the "slow"/contextual part the suite subtracts — it
  is a per-*prompt* signature, unique per row, NOT a per-template group mean),
- `U_last (n, p, 8)` — 8 output-Fisher factor vectors `u_ik = Jᵀ F^{1/2} e_k` through the
  downstream layers + unembedding, so `δᵀ G_i δ = Σ_k (u_ik · δ)²` is the behavioral cost
  (≈ 2·KL) of an activation move `δ` at token `i`, with **no** 4096×4096 matrix materialized.

Geometry is fit on the demeaned `H = X_last − tmpl_mean` (the per-prompt context-mean
demeaning recipe used everywhere in the suite).

### The two nested models
- **linear** — a single 1-D `line` atom (K=1, d_atom=1, `atom_topology="line"`).
- **+curved** — a single 1-D `circle`/`arc` atom (K=1, d_atom=1) — the same dimension, one
  extra degree of geometric freedom (it can bend and, for a circle, close).

Both are fit by the identical `gamfit.sae_manifold_fit` machinery, same rank-8 working
subspace, same REML/evidence-selected smoothness — the ONLY difference is whether the 1-D
manifold is allowed to curve. This is a clean nested comparison: linear ⊂ curved.

### Held-out reconstruction, 2-fold complementary template split
Curvature that only helps *in-sample* is overfitting. So each row is scored **held out**, on a
fit that never saw its **template** (splitting by template, not by row, prevents the template
context leaking into the fit). Templates are split into two complementary halves (even vs odd
template id); fit on one half's rows, score the other half's rows, then swap — so every row is
scored exactly once, and only **two** (fragile, expensive) circle fits are needed per feature.

For each fold: fit `line` and `circle` on the fit half's rows, then for each held-out row `i`
(never seen in that fit):
1. project `h_i` onto the fitted chart → coordinate `s_i` (`sae.project`),
2. reconstruct `ĥ_i =` chart point at `s_i`,
3. residual `r_i = h_i − ĥ_i`,
4. **raw deviance** `D_i = ‖r_i‖²` (Gaussian/reconstruction),
   **behavioral deviance** `D_i^F = r_iᵀ G_i r_i = Σ_k (u_ik · r_i)²` (nats-metric; the number
   that says the *unreconstructed* part still matters to the model's output).

Each row appears **once** as held-out (across folds), giving `n` paired observations. Pairing
is exact: line and circle reconstruct the *identical* held-out `h_i`, so the row's own
difficulty (how far it sits from any 1-D chart) cancels in the difference.

Per-row paired difference (curvature dividend):
```
Δ_i   = D_i(line)   − D_i(circle)         (raw, in activation units²)
Δ_i^F = D_i^F(line) − D_i^F(circle)        (behavioral, in nats)
```
`Δ_i > 0` ⟺ curvature pays on row `i`. Headline per feature: mean `Δ^F` and its p-value.

### Permutation scheme (paired sign-flip randomization)
The null is **H0: neither topology is favored on any given row** — line and curved are
exchangeable *conditional on the row*. Because the pairing is within-row, the randomization
distribution induced by H0 is the set of independent **sign flips**: for each row `i` draw
`ε_i ∈ {+1,−1}` uniformly (flipping `ε_i` swaps which model's deviance is called "line" vs
"circle", i.e. negates `Δ_i`). The permutation statistic is `T* = mean_i ε_i Δ_i`; the
observed statistic is `T = mean_i Δ_i`. Two-sided
```
p = (1 + #{ b : |T*_b| ≥ |T| }) / (B + 1),   B = 20000.
```
This is the exact paired randomization test — it is the correct scheme here (not a
label-shuffle across rows) precisely *because* the two deviances share a row; the only thing
H0 leaves free is the ± orientation of each within-row contrast. We **falsify** it two ways
before trusting any p-value: (i) confirm the sign-flip null is centered at 0 and symmetric,
(ii) a real negative control — re-run the entire fit+deviance pipeline with the **category
labels shuffled** (cyclic structure destroyed); curvature must stop paying (`Δ^F ≈ 0`,
`p` not significant). A feature where curvature does not pay on the *real* labels is reported
as a **headline honest negative**, not buried.

### Features
weekday (8B/L18), month (8B/L18), color (35B/L17), + weekday/month (35B/L17) scale checks,
sycophancy and hedging (safety probes). Refusal is expected to be an honest negative
(the crown already found refusal charts as a line, not a dial).

## Experiment 2 — slow-feature atlas pilot (context means)

The `tmpl_mean` vectors are subtracted as a nuisance everywhere. Hypothesis: they are a
**modeled feature** — slow/contextual structure with its own geometry. Pilot: pool the
**unique per-template means** across features/harvests (dozens of templates), fit a low-K
atlas on that template-mean population, and ask whether it **charts** (interpretable topology,
explained variance above a null, ordering certificates) or is **unstructured**. Cheap: the
`tmpl_mean` arrays already live in the npz caches. Reported as structured-vs-unstructured
with the same nulls discipline (Gaussian-matched + shuffled).
