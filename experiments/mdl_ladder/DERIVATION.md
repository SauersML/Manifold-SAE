# The MDL ladder: directions → blocks → charts

**Claim.** Goodfire's *Block-Sparse Featurizers* (BSF) shows a block code describes an
activation in fewer bits than a direction (TopK) code, with an MDL optimum at block
width `b ≈ 2–4`. That analysis is one rung of a longer ladder. A **chart** — a curved
atom carrying an intrinsic coordinate map `Φ` — pays for reconstruction with its
*intrinsic* dimension `d_i` instead of the block's *extrinsic* dimension `b`. For a
cyclic/curved feature (a circle: `b = 2`, `d_i = 1`) the chart codes **one** number per
firing where the block codes **two**, at the price of storing extra harmonic decoder
vectors in the dictionary. This document makes that trade rigorous and computable, and
derives the firing count `f*` above which the chart wins.

The scorer that implements every equation here is [`mdl.py`](mdl.py); the numbers are in
[`results.json`](results.json) and [`REPORT.md`](REPORT.md).

---

## 0. Description length = negative log evidence, in bits

We use Rissanen two-part MDL. A featurizer is a *code*: a dictionary `Φ` (shared,
transmitted once) plus, per token, the active coefficients and a pointer to which atoms
fired. The description length of a corpus of `N` tokens on which a feature fires `f` times
is

```
L_total  =  L_code  +  L_dict
L_code   =  f · ( code bits per firing )        # coefficients + selection
L_dict   =  P · L_param                         # P decoder scalars, L_param bits each
bits/token = L_total / N
```

This is not a metaphor for the fit objective — it **is** the fit objective. gamfit's REML
outer criterion is a (quasi-Laplace) negative log marginal likelihood, and
`−log evidence / ln 2 = description length in bits`. The assembled criterion is a single
line, `gam/crates/gam-sae/src/manifold/construction.rs:6526`:

```
v = loss.total() + extra_penalty_energy + 0.5 * log_det − occam
```

Term by term (citations in the same crate), and its MDL reading:

| REML term | code | MDL role |
|---|---|---|
| `loss.data_fit = 0.5·Σ_i w_i‖whiten(z_i − ẑ_i)‖²` | construction.rs:3385, :3433 | **L_code, data term** — residual coded to the distortion floor (`RSS = 2·data_fit`, dispersion `φ̂ = RSS/n` at outer_objective.rs:709) |
| `assignment_sparsity` (softmax/gate prior) | construction.rs:3429 | **L_code, selection bits** — which of `G` atoms fired |
| `0.5·λ_k·⟨B_k, S_k B_k⟩` smoothness + `ard` | construction.rs:3758, :3801 | prior energy on the decoder coefficients (a code on `Φ`) |
| `0.5·log_det = 0.5·log|XᵀX + S|` | construction.rs:6444, :6471 | **L_dict** — Occam factor: effective-parameter bits of the fitted model |
| `− occam = −Σ_k 0.5·rank(S_k)·log λ_k` | construction.rs:7451 | restricted-likelihood credit (`+½log|S|₊`), the penalty null-space |

The effective degrees of freedom that `0.5·log_det` charges for is
`EDF_k = tr(S_β⁻¹ M_k)` per atom (construction.rs:7818). **The chart's `L_dict` is larger**
(its extra `Φ` harmonics inflate `log|XᵀX+S|` and the EDF), **its `L_code` is smaller**
(`d_i < b` coded coordinates). The whole question is where the trade flips.

> The live REML solver is OOM-blocked in this shared-tree build (probe_out/NOTES.md;
> frontier_out/report.md §4), so we score the **measured** artifacts against the closed
> form below rather than reading `v` off a fit. The closed form is the same accounting the
> REML criterion performs; §1–§3 derive it from scratch so it stands on its own.

---

## 1. Code bits per active firing

Adopt the paper's convention of a **task-derived distortion floor** `δ²`: the per-token
reconstruction MSE below which downstream behaviour is preserved, so we only pay to code
signal *above* `δ²`. All bits are relative to this floor.

**Rate–distortion of one coordinate.** A Gaussian coordinate of signal variance `σ²` coded
to MSE `δ²` costs

```
r(σ², δ²) = ½ log₂(σ² / δ²)          bits         (high-rate form, σ² ≫ δ²)
```

`mdl.py` uses the exact `½ log₂(1 + σ²/δ²)`, which is non-negative at all SNR and equals
the high-rate form to O(1) bit once `σ² ≫ δ²`. (This matters: the real Qwen probes sit at
SNR ≈ 1, where the naïve `½log₂(σ²/δ²)` would go negative; §5.)

**A `b`-dimensional linear block.** To pull a genuinely `b`-extrinsic feature below `δ²`, a
*linear* code must transmit all `b` coordinates — a straight atom cannot fold curvature, so
it needs the full extrinsic span. Code bits per firing:

```
L_code^block = b · ½ log₂(σ²/δ²)  +  log₂ C(G, k)
                └ coefficients ─┘     └ selection ┘
```

**A chart of intrinsic dimension `d_i`.** The chart's map `Φ` absorbs the curvature into
the *dictionary*, so per firing it transmits only `d_i` intrinsic coordinates (for a
circle, one angle). Same selection cost:

```
L_code^chart = d_i · ½ log₂(σ²/δ²)  +  log₂ C(G, k)
```

**Per-firing code saving.** Selection bits are identical (same `G`, `k`) and cancel:

```
ΔL_code = L_code^block − L_code^chart = (b − d_i) · ½ log₂(σ²/δ²)          (per firing)
```

For a circle, `b − d_i = 2 − 1 = 1`: the chart saves exactly one coordinate's worth of rate
per firing. This is the *measured* fact behind the probes — one curved coordinate does the
reconstruction work of two linear PCs (probe_out/summary.md: curved(1 coord) ≈ linear(2 PC)
≫ linear(1 PC)).

---

## 2. Dictionary bits

The dictionary is transmitted once and amortized over all `f` firings. A decoder scalar
must be stored precisely enough not to inject distortion above `δ²`; at distortion-matched
precision it costs `L_param ≈ ½ log₂(σ²/δ²) = r` bits — the same per-scalar rate as a code
coefficient. (`mdl.py` defaults to this; pass a fixed budget, e.g. 16 for fp16, to override.)

Decoder scalar counts (`p` = ambient/reduced dimension):

| featurizer | decoder | `P` |
|---|---|---|
| direction (rank-1) | one `p`-vector | `1·p` |
| `b`-block | `b` `p`-vectors | `b·p` |
| circle chart (Fourier, `n_basis` harmonics) | `n_basis` `p`-vectors | `n_basis·p` |

The circle chart's decoder is `decoder_blocks` of shape `(n_atoms, n_basis, p)` — measured
directly from the fitted `gamfit.torch.ManifoldSAE` atom (`n_basis = 4` in the probe recipe,
`p = 16` reduced → 64 generative scalars per atom, plus a scalar anchor and `log λ`).

**Extra parameters the chart pays over the matched block.** The fundamental harmonics
`{cos θ, sin θ}` span the *same* 2-plane as a 2-block. The chart's surplus is the higher
harmonics `{cos 2θ, sin 2θ, …}` — the vectors that *bend* that plane into a closed curve:

```
Φ  =  P_chart − P_block  =  (n_basis − b) · p          # extra decoder scalars
```

For the probe circle: `n_basis = 4`, `b = 2`, so `Φ = 2p`. In the pure single-winding limit
`n_basis = 2`, `Φ = 0`: a perfect circle chart stores the *same* decoder as a 2-block and
strictly dominates it (half the code, same dictionary) at every `f ≥ 1`. Real features are
approximately circular, so `Φ > 0` and there is a finite crossover.

---

## 3. The crossover

Total description length as a function of firing count `f` (selection cancels):

```
L_block(f) = f·b·r        + P_block·L_param
L_chart(f) = f·d_i·r      + P_chart·L_param
```

The chart wins when `L_chart(f) < L_block(f)`:

```
f · (b − d_i) · ½ log₂(σ²/δ²)   >   Φ · L_param
                     ⟹
        Φ · L_param                    Φ · L_param
 f*  =  ─────────────────────  =  ───────────────────────
        (b − d_i) · r             (b − d_i) · ½ log₂(σ²/δ²)
```

This is exactly the crossover in the task brief. Two clean readings:

- **Distortion-matched precision** (`L_param = r`): the rate `r` cancels and the crossover
  becomes **SNR-independent**:
  ```
  f*  =  Φ / (b − d_i)  =  (n_basis − b) · p / (b − d_i)
  ```
  For the circle (`b=2, d_i=1, Φ=2p`): **`f* = 2p`**. The chart recoups its curvature
  harmonics after ≈`2p` firings, whatever the feature's SNR.

- **General precision** `ρ = L_param / r`: `f* = ρ · 2p`. Storing the dictionary at higher
  fidelity than the code (`ρ > 1`) pushes the crossover out proportionally.

The `mdl.py` `crossover_firings` reports both the general `f*` (using the measured freed-
coordinate rate) and the matched-precision `f* = Φ/(b−d_i)`, so the regime dependence is
explicit (§5).

---

## 4. Instantiation — the circle atom, with real numbers

**Frontier planted circle** (`frontier_out/results.json`, `p = 9`, curved atom `{1,cosθ,sinθ}`
→ `n_basis = 3`, so `Φ = (3−2)·9 = 9`). Per planted-circle atom the measured geometry is a
straight top-1 atom `EV ≈ 0.61–0.72` vs a circle chart `EV ≈ 0.995` — the circle fills a
clean 2-plane, so the freed 2nd coordinate carries large variance and the rate saving is big
(`ΔL_code ≈ 2.7–3.0 bits/firing`). Crossover:

```
f* = Φ / (b − d_i) = 9 / 1 = 9 firings         (matched precision)
f* ≈ 10.6–10.9 firings                          (measured rate, mdl.py)
```

Each planted circle atom fires on ≈12–13 points → **past the crossover**: the chart's
description is already shorter than the 2-block's on the frontier data.

**Synthetic clean circles** (`probe_out/synthetic_validation.json`, `p = 16`, `n_basis = 4`
→ `Φ = 2·16 = 32`). High SNR (`δ²` tiny, `≈0.2%` of variance):

- month (12-circle, chart `EV = 0.998`): `ΔL_code ≈ 3.55 bits/firing`, `f* ≈ 37` (matched
  `2p = 32`); fires 60× → **chart wins** (9.42 vs 10.27 bits/token).
- weekday (7-circle, chart `EV = 0.93`, only 35 points): `f* ≈ 44` (matched 32); fires 35×
  → just below the measured `f*`, at/above the matched `f*`.
- year (**non-cyclic control**, a line: `d_i = 1 = b_eff`): the chart eliminates *no*
  coordinate, `ΔL_code < 0`, `f* = ∞` — the chart **never** wins. This is the control
  working: curvature only pays for genuinely curved features.

**Real Qwen probes** (§5) — the honest low-SNR case.

---

## 5. Residual-variance / distortion floor, handled honestly

Two subtleties, both handled in `mdl.py` rather than hidden:

1. **Task-derived floor.** We set `δ²` = the residual of the best single-coordinate model
   present (the chart's own residual `(1−EV_chart)·V`) — the fidelity a task that *wants*
   the cyclic structure would demand. At that floor a **direction is distortion-infeasible**
   on every circle (its residual exceeds `δ²`: a straight atom cannot reach circle fidelity
   at any rate), and a **2-block is the minimal feasible linear code** — the bits ladder
   makes the "linear can't trace a circle" fact quantitative rather than rhetorical.

2. **Low-SNR real data.** On the real Qwen weekday (L14) and month (L8) harvests the circle
   is a modest component atop a large isotropic tail (chart `EV ≈ 0.58–0.60`; the circle's
   2nd PC, `λ₂ ≈ 0.34–0.82`, can sit *below* the loose floor). Then the freed coordinate is
   nearly free (`ΔL_code ≈ 0.10–0.13 bits/firing`) yet the chart still pays for `Φ = 2p`
   harmonics → the **measured** `f* ≈ 96–122`, *above* the probe's 35–60 firings. So at
   probe scale on noisy real activations, the block wins the per-feature MDL race; the
   SNR-independent `f* = 2p = 32` only bites when the circle's two extrinsic dims are
   comparably strong (clean-circle / high-SNR regime — synthetic and frontier).

   This is not a defeat of the claim, it is its precise scope: **a curved chart wins once
   the feature fires more than `f* = Θ(p)` times at the task fidelity.** A real cyclic
   feature (weekday, month, hue) fires on *every* date/color mention across a corpus —
   millions of firings, `f ≫ f*` in any regime — so charts win at deployment scale. The
   probe's 35–60 firings straddle the crossover, which is exactly why we report `f*` rather
   than a single-`f` verdict.

---

## 6. What the scorer computes (summary)

Per featurizer, at floor `δ²`:

```
code_bits/firing  = Σ_j ½log₂(1 + v_j/δ²)  +  log₂C(G,k)     # v_j = coded-coord variances
dict_bits         = P · L_param
bits/token        = ( f·code_bits/firing + dict_bits ) / N
distortion_infeasible = residual_achieved > δ²
```

and the crossover `f* = Φ·L_param / ((b−d_i)·r)`, with the matched-precision `f* = Φ/(b−d_i)`.
The JSON interface (README.md) lets the G-bsf and N-nursery lanes score their own block /
chart compositions against the same floor without touching this file.
