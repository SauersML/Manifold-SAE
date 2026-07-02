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

_Status: live document, updated as lanes commit. Last sync: R-review verdicts encoded (weekday cyclic order QUALIFIED marginal p≈.04, month robust, C1 EV-parity safe); G-bsf synthetic recovery corrected 0.76→0.82; BT1 confirmed NOT compiling (`e01c2fd`); N-nursery arms running/uncommitted → PENDING with two mandatory caveats pre-recorded. Untracked result artifacts flagged for lane commit._

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
Publication status legend: **SAFE NOW** (verified, publishable) · **QUALIFIED** (publishable only with the stated hedge) · **PENDING** (no validated/committed number yet — never cite).

| lane / component | claim under test | result | verdict | status |
|---|---|---|---|---|
| **G-bsf** — faithful BSF reimpl (`bsf_baseline/`) | block code recovers planted subspaces; beats TopK at matched budget | synthetic subspace recovery mean **R²=0.82** (vanilla, 5 seeds, val EV 0.89); on OLMo self-qualia L40 (a *linear* axis) TopK b=1 EV **0.4489** is best, BSF EV falls 0.425→0.325 as b grows (higher stable rank, lower EV) | **SAFE NOW** — SOUND / faithful (R-review verified real numbers numerically) | landed — real/cyclic + synthetic numbers backed; `metrics.json` now committed (`337aadc`) |
| **G-bsf** — cyclic block finding | one block captures a whole cycle (weekday/month) | winning block stable rank ≈2, **in-block cyclic adjacency 1.00**, 7/8 blocks active | **QUALIFIED** — structural claim holds | landed — **cyclic `full_ev`=0.976 is IN-SAMPLE, do not publish as generalization** |
| **M-mdl** — MDL ladder (`mdl_ladder/`) | chart beats block in bits above `f*` | `f*=2p`; chart wins on frontier + synthetic; f*≈96–122 on noisy real | **SAFE NOW** — scorer SOUND (R-review hand-verified) | landed (in-sample EVs) |
| **Dose calibration** (`dose_real_out/`) | chart `predicted_nats` predicts real output KL | R²=0.951, slope 0.908, median meas/pred **0.881** (n=288); **0.999 inside validity radius** (n=49); linear baseline **~10× miscalibrated** (median ratio 10.0) | **SAFE NOW** — strongest real-model result | landed — weekday circle only (month co-collapse on pre-fix build) |
| **P-null** — matched-null battery (`null_out/`, `matched_null.py`) | real cyclic claims survive matched nulls | **month cyclic order robust** (n=12, adj 0.83); **weekday cyclic order MARGINAL** (adj 0.71 at n=7, p≈0.04, and fragile — the single-seed battery re-fit scores lower/at chance); **C1 EV-parity safe**; continuous circular correlation significant | **SAFE NOW** for month order + C1 EV-parity; **QUALIFIED** for weekday (marginal p≈.04) — R-review verified (commit 5fd1465) | landed per R-review REVIEW.md; **raw `null_out/` JSONs pending P-null commit**; month/color full battery PENDING |
| **N-nursery** — chart-per-block vs joint-K (`block_nursery/`) | nursery beats co-collapsing joint K≥2 fit | design VALID (R-review 5fd1465); real held-out arms now committed (`real_results.json`, `337aadc`: joint control incl. REML + torch, and the nursery arm) — **awaiting R-review validation of the results** | **PENDING** — results committed, not yet validated | PENDING (see §4.4 for the two mandatory caveats before any headline) |
| **BT1** — Rust block-sparse Tier-1 (gam `097b1c1de`) | gauge-invariant block-sparse core | gauge invariance verified numerically by review (gate/loss invariant to O(b) rotation to 1e-15; negative control bites). FFI + `block_tests.rs` added. **But R-review (e01c2fd) confirms `gam-sae` still does NOT compile** — 3 edition-2024 pattern errors in `block.rs`; tests cannot run. | **DESIGN SOUND, NOT GREEN** | **PENDING** — does not compile (`cargo test -p gam-sae` fails to build); no BT1 number is publishable |

**Supporting gam-core lanes (SAFE NOW, verified by R-review):** O-manifold's fleet-batch landing (`e09e6956c`, byte-identical hunks, deleted tests are pure relocations, the `reachable_dictionary_rank` correctness fix is sound) and O-solve's mixture-link gate widening (LogLog/Cauchit 5-jet Fisher weight genuinely implemented + tested to 1e-12..1e-5) underpin the "additive generative model" and REML-core axes.

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

2. **Real cyclic-probe claims are matched-null-scoped — this is a feature, not a bug.** We ran
   BSF's own matched-null discipline against our W7 circle probes (P-null, R-review-verified,
   commit `5fd1465`), and it correctly scopes what we may claim:
   - **Month (n=12, adjacency 0.83): robust cyclic order — SAFE.**
   - **Weekday (n=7): MARGINAL.** The headline adjacency 0.71 clears the label-permutation null
     only at **p≈0.04**, and it is fragile: the battery's single-seed budget re-fit scores lower
     (toward chance), pushing the effective p up. Publish weekday cyclic order as *"marginal
     (p≈.04)"*, never as a strong result.
   - **C1 EV-parity (one curved coord ≈ two linear PCs): SAFE** as a representational claim.
     The matched-spectrum null shows the *parity itself is not unique to circles* (a smooth 1-D
     curve on a matched spectrum closes a similar gap) — a scoping note on what the parity
     proves, not a refutation that it holds.
   - **Survivor:** the continuous circular correlation is significant, and the dose calibration
     (§3) is the strongest real-model result. That two of our own weekday claims are only
     marginal, honestly reported, is what makes the surviving ones (month order, C1 parity,
     continuous circularity, dose) credible.

3. **BT1 block-sparse core does NOT compile — no BT1 number is publishable.** R-review's
   re-review (`e01c2fd`) confirms `gam-sae` fails to build: 3 edition-2024 pattern errors in
   `block.rs`, so `cargo test -p gam-sae` cannot run and the gauge/recovery tests are unexecuted
   (the gauge property is verified only by review's out-of-tree numeric replica). The design is
   sound and the FFI surface (`gamfit.block_sparse_dictionary_fit`) is written, but every BT1
   result stays PENDING until the crate compiles and the tests pass.

4. **N-nursery has no validated headline yet, and two caveats are MANDATORY when it lands.**
   The design is valid (R-review): the joint-K co-collapse control is honestly bounded (REML
   attempted in a capped subprocess, recorded `TIMEOUT_BLOCKED`), and the discovered arm uses no
   labels. But the result arms are still running/uncommitted. When a headline lands it MUST carry:
   (a) **the co-collapse cure is demonstrated on a TORCH-proxy fitter, not REML** — REML, the
   production fitter the hypothesis is really about, is recorded BLOCKED here, so transfer is
   UNESTABLISHED; and (b) **the real-data Arm B blocks are set-membership-supervised** (each
   set's top-2 PCs); the fully-unsupervised `discover_blocks` result is the publishable number
   and the oracle/set-supervised arm is only the labeled ceiling. All nursery EVs are in-sample.

5. **G-bsf cyclic `full_ev` is in-sample**, and its real-data EV comparison is on the OLMo
   self-qualia axis, which is *linear* — so BSF not beating TopK there is expected, not a
   defeat of BSF. The block advantage is a **curvature/subspace** claim, best shown on genuinely
   multi-dimensional or curved features.

6. **MDL bits/token use in-sample EVs** (descriptive, not held-out). Fine for the crossover
   argument (`f*` depends on parameter counts and the spectrum, not on generalization), but the
   absolute bits inherit their source probe's in/out-sample status.

7. **Some result artifacts remain uncommitted.** `bsf_baseline/metrics.json` and
   `block_nursery/real_results.json` are now committed (`337aadc`), but `null_out/*.json`
   (P-null) and `block_nursery/synthetic_results.json` are still untracked in this shared working
   tree (not gitignored — the lanes have not `git add`-ed them). Numbers cited from an uncommitted
   file are attributed to R-review's committed verification (`REVIEW.md`); the owning lanes must
   commit the raw artifacts before they are directly citeable. This is why the P-null real-weekday
   cell carries a "pending commit" note despite the run having executed.

8. **Dose calibration shows the weekday circle only.** The 12-token month loop triggers the
   pre-fix multi-modal auto-grow/co-collapse in that build; re-run against the guard-patched
   build before claiming month/hue dose calibration.

---

### Provenance

Artifacts: `bsf_baseline/` (G-bsf), `mdl_ladder/` (M-mdl), `dose_real_out/` (dose calibration,
real llama-3.1-8b L16), `null_out/` (P-null), `block_nursery/` (N-nursery), `review_checks/REVIEW.md`
(R-review), gam `a6f2c0e28` (BT1). Reproduce the MDL ladder:
`python mdl_ladder/mdl.py --probes --synthetic --frontier`.
