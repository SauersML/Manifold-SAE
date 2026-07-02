# A response to *Block-Sparse Featurizers* — one rung further up the ladder

**Program thesis.** Goodfire's *Block-Sparse Featurizers* (BSF) makes the right move and
proves it in the right currency: replace a dictionary of single **directions** with a
dictionary of **blocks** (small subspaces), and an MDL argument shows the block code
describes activations in fewer bits, with an optimum block width `b ≈ 2–4`. Our program
takes the next rung. A block is a *flat* subspace; a **chart** is a *curved* one — a
first-class dictionary atom carrying an explicit parametric map `g(t)`, an intrinsic
coordinate, a topology type, and a downstream metric. For cyclic/curved features (weekday,
month, hue) the chart codes one intrinsic coordinate where a block codes two extrinsic
dimensions, and it comes with an intervention-dose predictor a subspace cannot supply.

This document positions our work against BSF axis by axis, presents the MDL ladder as the
shared evaluation, and reports the head-to-head numbers **as the lanes land** — empty cells
are marked `PENDING`, never filled with placeholders. Paper claims are paraphrased (we do not
have verbatim text to quote); where a number is ours it cites its artifact.

_Status: live document, updated as lanes commit. Last sync: BT1 FFI/tests commit `097b1c1de` landed (green pending R-review); N-nursery arms + P-null month/color still PENDING._

---

## 1. Axis-by-axis: BSF vs our program

| axis | BSF (paraphrased) | what we ship | evidence (artifact) | status |
|---|---|---|---|---|
| **Additive generative model** | Named by BSF as future work: move from a reconstruction autoencoder to an explicit additive generative model of activations. | A REML/IBP additive manifold-SAE is our *core*, not future work: each atom is an additive generative term with its own penalty and evidence. | `gam-sae` REML core; block-sparse Tier-1 `block.rs` + FFI/tests (gam `097b1c1de`) | **landed** — design sound & gauge-verified; FFI + gauge/recovery tests committed; R-review re-verification PENDING (§4) |
| **Curved vs flat features** | Curves/manifolds read *post-hoc* from block subspaces (PCA / Fourier detectors on top of flat blocks). | Curvature is a *first-class atom*: an explicit `circle`/`fourier` chart `g(t)` fit as a dictionary element, not detected after the fact. | `curved_feature_probes.py`, `frontier_out/`, `mdl_ladder/` | **landed** (in-sample; real-data nulls in §4) |
| **Description length (MDL)** | Block code beats direction code in bits; optimum `b ≈ 2–4`. | Ladder extends one rung: chart beats block for curved features above a firing-count crossover `f* = Θ(p)`. | `mdl_ladder/DERIVATION.md`, `results.json` | **landed** (§2) |
| **Intervention dose / calibration** | (Not addressed) — featurizer is descriptive, no forward-effect prediction. | `steer` reports `predicted_nats`: how far the output distribution moves, via a downstream output-Fisher metric on the chart, *before* the edit. | `dose_real_out/` (real llama-3.1-8b) | **landed** — strongest real-model result (§3) |
| **Model selection** | Block width / count chosen by hyperparameter sweeps. | Per-atom REML evidence / effective-DOF selects the intrinsic dimension and curved-atom count directly. | `frontier_out/` (`k_curved` recovered = 3 = planted) | **landed** (accounting; live REML OOM-blocked, §4) |
| **Typed topology** | Blocks are untyped subspaces; a subspace has no notion of "circle" vs "line". | Atoms carry a topology *type* (circle / arc / line); the type is falsifiable — a line feature must not want curvature. | `synthetic_validation.json` (year=line control), MDL year row `f*=∞` | **landed** |
| **Presence / amplitude decoupling** | Signed block codes give a full subspace, but presence and amplitude are entangled in the code norm. | Block **gate** `‖z_g‖₂` (presence) is decoupled from the **signed** code `z_g` (amplitude); gauge-invariant selection. | `block.rs` (`block_gates`, `code_row`); G-bsf signed codes | **landed** (BT1 design; compile pending) |
| **Uncertainty** | Point estimates. | REML posterior + Fisher metric yield a certified **validity radius** within which the dose prediction is trusted. | `dose_real_out/` (ratio 0.999 inside radius, n=49) | **landed** |

---

## 2. Evaluation centerpiece — the MDL ladder and the `f* = 2p` crossover

We score every featurizer in **bits/token** at a task distortion floor `δ²`, in the same
currency BSF uses. This is not a metaphor for the fit objective: gamfit's REML negative log
evidence *is* a description length (`v / ln 2` = bits), decomposing term-for-term at
`gam/crates/gam-sae/src/manifold/construction.rs:6526`
(`v = loss.total() + extra_penalty + 0.5·log|XᵀX+S| − occam`) into a code/distortion term, a
selection term, and an effective-parameter (dictionary) term.

Two-part MDL per feature that fires `f` times on `N` tokens:

```
direction  codes 1 coordinate      block codes b=2 (circle plane)      chart codes d_i=1 (angle)
chart pays Φ = (n_basis − b)·p extra harmonic decoder scalars (the curvature)
chart wins when  f·(b − d_i)·½log₂(σ²/δ²)  >  Φ·L_param
    f* = Φ·L_param / ((b − d_i)·r)      →  (distortion-matched)  f* = Φ/(b − d_i) = 2p  for a circle
```

The crossover is **SNR-independent** at matched precision and **`f* = ∞` for a line**
(a straight feature frees no coordinate) — so the ladder self-controls: curvature only pays
where curvature exists.

| regime | direction | 2-block | circle-chart | crossover `f*` | winner at `f ≫ f*` |
|---|---|---|---|---:|---|
| frontier planted circle (p=9, high SNR) | infeasible | feasible | **shortest past f≈11** | ≈9–11 | **chart** |
| synthetic 12-circle (p=16, high SNR) | infeasible | 10.27 b/tok | **9.42 b/tok** | 32–37 | **chart** |
| real weekday/month (p=16, SNR≈1) | infeasible | shortest at f=35–60 | past f≈100 | 96–122 | **chart** (only at corpus scale) |
| year / any line (control) | infeasible | **shortest** | never | ∞ | block |

A direction is **distortion-infeasible on every circle** — it cannot reach circle fidelity at
any rate — which turns "linear can't trace a circle" into a bits statement. Full derivation and
per-lane JSON interface: `mdl_ladder/DERIVATION.md`, `mdl_ladder/README.md`.

---

## 3. Head-to-head results (filled as lanes land)

| lane / component | claim under test | result | verdict | status |
|---|---|---|---|---|
| **G-bsf** — faithful BSF reimpl (`bsf_baseline/`) | block code recovers planted subspaces; beats TopK at matched budget | synthetic subspace recovery R²=0.76 (vanilla); on OLMo self-qualia L40 (a linear axis) TopK b=1 EV **0.4489** is best, BSF EV falls 0.43→0.32 as b grows (higher stable rank, lower EV) | **SOUND / faithful** (R-review verified numerically) | landed |
| **G-bsf** — cyclic block finding | one block captures a whole cycle (weekday/month) | winning block stable rank ≈2, **in-block cyclic adjacency 1.00**, 7/8 blocks active | structural claim holds | landed — **cyclic `full_ev` is IN-SAMPLE, do not publish as generalization** |
| **M-mdl** — MDL ladder (`mdl_ladder/`) | chart beats block in bits above `f*` | `f*=2p`; chart wins on frontier + synthetic; f*≈96–122 on noisy real | **scorer SOUND** (R-review hand-verified) | landed (in-sample EVs) |
| **Dose calibration** (`dose_real_out/`) | chart `predicted_nats` predicts real output KL | R²=0.951, slope 0.908, median meas/pred **0.881** (n=288); **0.999 inside validity radius** (n=49); linear baseline **~10× miscalibrated** (median ratio 10.0) | **strongest real-model result** | landed — weekday circle only (month co-collapse on pre-fix build) |
| **P-null** — matched-null battery (`null_out/`) | real weekday cyclic claims survive matched nulls | **circular correlation 0.93, p=0.010 (survives)**; discrete adjacency 0.43, p=0.43 (**fails**); C1 gap-closed p=0.48 (**fails vs matched-spectrum**) | **partially nulled** — continuous circularity real, discrete ordering at chance | landed (real weekday); month/color PENDING |
| **N-nursery** — chart-per-block vs joint-K (`block_nursery/`) | nursery beats co-collapsing joint K≥2 fit | data harness landed (n=480, p=96, 3 circles, subspace EV 0.878); `arms: {}` | `PENDING` — A/B arms not yet run | PENDING |
| **BT1** — Rust block-sparse Tier-1 (gam `097b1c1de`) | gauge-invariant block-sparse core | gauge invariance verified numerically (gate/loss invariant to O(b) rotation to 1e-15; negative control bites). Follow-up commit adds FFI + Python surface + `block_tests.rs` (gauge/recovery), addressing the two compile/test blockers R-review flagged on `a6f2c0e28`. | **DESIGN SOUND**; FFI + tests landed | PENDING R-review re-verify (`cargo test -p gam-sae` green) |

---

## 4. Honest caveats (R-review punch list, `review_checks/REVIEW.md`)

We hold ourselves to the same matched-null discipline BSF uses, and it bites some of our own
claims. Stated plainly:

1. **Live REML is OOM-blocked in this build.** `gamfit.sae_manifold_fit` (the REML curved
   solver) segfaults/OOMs in the shared-tree environment (stale ext, non-PD IBP Hessian,
   memory-leaky inner loop). Every curved result here is either the **torch-backend**
   `ManifoldSAE` fit or **parameter accounting** against measured geometry — not a live REML
   fit. The REML→bits map (§2) is the same accounting the criterion performs, cited to source,
   but not read off a converged `v`.

2. **Real cyclic-probe claims are only partially validated.** On the real weekday harvest the
   matched-null battery (P-null) **fails** the discrete cyclic-adjacency claim (p=0.43, at
   chance) and **fails** the "one curved coord = 2-PC parity is a *circle* signature" claim
   against a matched-spectrum Gaussian (p=0.48). What **survives** is the continuous circular
   correlation (0.93, p=0.010) and the dose calibration (§3). So the defensible real-model
   claims are: (a) the recovered angle is continuously circular, and (b) the chart's dose
   metric predicts interventions — **not** that the discrete token ordering beats chance on
   35 samples. Month/color nulls are still PENDING.

3. **BT1 block-sparse core: tests landed, green not yet re-verified.** R-review flagged
   `a6f2c0e28` for a missing `block_tests.rs` and a `&mut Array2` vs `ArrayView2` type error.
   The follow-up commit `097b1c1de` ("block-sparse FFI + Python surface + gauge/recovery tests")
   adds `block_tests.rs` (now present in `crates/gam-sae/src/sparse_dict/`) and the FFI/Python
   surface, addressing both blockers. R-review has **not yet re-reviewed** `097b1c1de`, so
   `cargo test -p gam-sae` green is still PENDING that verification — we do not claim it passes
   without evidence.

4. **N-nursery A/B has not run.** Only the synthetic data harness landed; the head-to-head
   (joint K≥2 co-collapse control vs the nursery arm) is `arms: {}`. No verdict yet — the
   control must be actually invoked and recorded even on non-convergence (not skipped so the
   nursery "wins" by default).

5. **G-bsf cyclic `full_ev` is in-sample**, and its real-data EV comparison is on the OLMo
   self-qualia axis, which is *linear* — so BSF not beating TopK there is expected, not a
   defeat of BSF. The block advantage is a **curvature/subspace** claim, best shown on genuinely
   multi-dimensional or curved features.

6. **MDL bits/token use in-sample EVs** (descriptive, not held-out). Fine for the crossover
   argument (`f*` depends on parameter counts and the spectrum, not on generalization), but the
   absolute bits inherit their source probe's in/out-sample status.

7. **Dose calibration shows the weekday circle only.** The 12-token month loop triggers the
   pre-fix multi-modal auto-grow/co-collapse in that build; re-run against the guard-patched
   build before claiming month/hue dose calibration.

---

### Provenance

Artifacts: `bsf_baseline/` (G-bsf), `mdl_ladder/` (M-mdl), `dose_real_out/` (dose calibration,
real llama-3.1-8b L16), `null_out/` (P-null), `block_nursery/` (N-nursery), `review_checks/REVIEW.md`
(R-review), gam `a6f2c0e28` (BT1). Reproduce the MDL ladder:
`python mdl_ladder/mdl.py --probes --synthetic --frontier`.
