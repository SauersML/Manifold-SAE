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

_Status: live document, updated as lanes commit. Last sync: **N-nursery p=96 synthetic GREEN-lit** (R-review) → SAFE NOW (QUALIFIED): 3/3 blocks discovered, 3/3 rings reconstructed (0.89–0.93 held-out EV), 2/3 pass the strict angle bar (corr>0.8); composed EV 0.834 = oracle 0.833 > joint 0.756, no joint solve, REML width-blocked; the failing ring's angle reads 0.749 in the oracle arm too (intrinsic, not discovery). NOT an EV win (linear PCA-6 0.883); REML-cure transfer unestablished (#2027 RED); discovery≈oracle only at n=480. Centerpiece unified ladder in §2; BT1 GREEN; G-bsf synthetic+cyclic held-out backed; **P-null committed** (month circle survives every null; weekday demoted — marginal/mixed); **co-collapse now GREEN on the deterministic repro** (`7a93b1d06`, #2027 3/3: EV recovers AND atoms separate onto distinct curved factors, both widths; fix = cold-start chart deflation; production-width REML validation pending); **shadow_cone** QUALIFIED (decoupling needs explicit presence supervision); **chart_transfer** QUALIFIED-positive (coordinate consistency 0.95/0.81 supported, EV does not beat the 2-PC plane); centerpiece anchored on the month flagship; all 5 arXiv IDs verified._

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
| **Presence / amplitude decoupling** | Signed block codes give a full subspace, but presence and amplitude are entangled in the code *norm* (an intensity coordinate). | Block **gate** (presence) is decoupled from the **signed** code (amplitude) — but this needs an **explicit presence signal**: on synthetic weak-vs-absent, block-norm AUC 0.47 ≈ chance and a recon-only gate 0.47 both fail; a presence-supervised gate reaches 0.999. | `block.rs` (gauge-invariant, BT1 17/0 tests); `shadow_cone/` (`db9d1e5`) | **landed** — decoupling supported *with* presence supervision (§3) |
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

### The unified real-data ladder — BSF reproduced, then extended one rung

**Rungs 1–4, BSF reproduced** (G-bsf, OLMo self-qualia L40, matched budget F=64/L0=8): the
block code's bits/token fall as the block widens, because the selection cost `log₂ C(G,k)`
collapses — the paper's "blocks beat directions" MDL result, with the mechanism visible.

| rung | bits/token | selection bits/firing | val EV |
|---|---:|---:|---:|
| direction (TopK, b=1) | **36.6** | 32.0 | 0.449 |
| block b=2 | 19.5 | 15.1 | 0.425 |
| block b=4 | 11.0 | 6.9 | 0.399 |
| block b=8 | **6.4** | 3.0 | 0.325 |

**Rung 5, one rung further** (M-mdl `score_json`, on the **month** cycle — our matched-null
flagship, §4.2 — where G-bsf's own block-finding lands: a single b≈4 block captures the cycle at
held-out EV 0.95, coord stable rank 2.4 = the circle's extrinsic dim). The circle-chart codes that
cycle from **one intrinsic coordinate** (the angle, single-coordinate cyclic order 10/12,
matched-null p=0.0004) where the block codes ~2 extrinsic dims — continuing the descent by
collapsing the *code dimension*:

| feature | block (held-out EV) | circle-chart | Φ (extra harmonics) | crossover `f*` |
|---|---|---|---:|---:|
| **month** (flagship) | b=4 (≈2 eff. dims, rank 2.4), EV 0.95 | d_i=1 (angle), order 10/12 | 12 | **≈16** (matched `2p`=12) |
| weekday (marginal, §4.2) | b=4 (≈2 eff. dims), EV 0.82 | d_i=1, order 5/7 (p≈.04) | 12 | ≈22 (matched `2p`=12) |

A month/weekday feature fires far more than ~12–22 times in any corpus (`f ≫ f*`), so the
curved chart has the shortest description. **Directions collapse the selection cost, blocks
collapse it further, and the chart collapses the code dimension (2→1 coordinate per firing) —
three rungs, each removing a different term of the description length.** (Rungs 1–4 are OLMo,
a linear axis where a chart is degenerate; the chart rung is realized on the cyclic feature.
Artifact: `mdl_ladder/unified_ladder.json`, scorer `mdl_ladder/unified_ladder.py`.)

Supporting — the crossover across regimes (single-feature, `g_dict=1`):

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
| **G-bsf** — faithful BSF reimpl (`bsf_baseline/`) | block code recovers planted subspaces; beats TopK at matched budget | synthetic subspace recovery: **Grassmannian principal-angle R²=0.986**, vanilla 0.82 (identical budget; recovered stable rank 2.9 vs planted b=4); real-data block **stable rank climbs 1.0→3.4 as b:1→8** (reproduces the paper's ≈3); on OLMo self-qualia L40 (a *linear* axis) TopK b=1 EV **0.4489** best, BSF EV falls 0.425→0.325 | **SAFE NOW** — SOUND / faithful (R-review verified) | landed — `metrics.json` committed (`33731de`) |
| **G-bsf** — cyclic block finding | one block captures a whole cycle (weekday/month) | single block captures ~80% variance, **held-out EV (leave-one-template-out) weekday 0.82 / month 0.95**, in-block **cyclic adjacency 1.00**, coord stable rank 2.4 (the circle's extrinsic dim) | **SAFE NOW** — structural claim + held-out EV | landed (`992c97b`) — in-sample number renamed `full_ev_insample`; held-out EV is the publishable one |
| **M-mdl** — MDL ladder (`mdl_ladder/`) | chart beats block in bits above `f*` | `f*=2p`; chart wins on frontier + synthetic; f*≈96–122 on noisy real | **SAFE NOW** — scorer SOUND (R-review hand-verified) | landed (in-sample EVs) |
| **Dose calibration** (`dose_real_out/`) | chart `predicted_nats` predicts real output KL | R²=0.951, slope 0.908, median meas/pred **0.881** (n=288); **0.999 inside validity radius** (n=49); linear baseline **~10× miscalibrated** (median ratio 10.0) | **SAFE NOW** — strongest real-model result | landed — weekday circle only (month co-collapse on pre-fix build) |
| **P-null** — matched-null battery (`null_out/`, committed) | real cyclic claims survive matched nulls | **Month: real, survives EVERY null** — C1 EV-parity gap-closed 1.043 vs matched-spectrum null 0.970 (**p=0.012**), C2 cyclic order 0.833 vs 0.182 (**p=0.0004**), basis-real 1.00. **Weekday: mixed** — C2 order marginal (0.714, **p=0.042** PASS) but C1 EV-parity **FAILS** the matched-spectrum null (p=0.16) and basis-real WEAK (0.22). Phase-locking n.s. for both (ordering carried by low-frequency smoothness, not higher harmonics — a diagnostic, not a gate) | **SAFE NOW** for month (all nulls pass); **QUALIFIED** for weekday (order marginal p≈.04; EV-parity does NOT survive) | landed & committed (`null_out/RESULTS.md`) |
| **N-nursery** — chart-per-discovered-block vs joint-K (`block_nursery/`) | recover curved factors without a joint K≥2 solve | **p=96 product-of-3-circles synthetic (R-review-validated):** **3/3 blocks discovered, 3/3 rings reconstructed (0.89–0.93 held-out EV), 2/3 pass the strict angle bar (corr>0.8)**; composed held-out EV 0.834 = oracle upper bound 0.833 > joint torch 0.756, no joint solve, REML width-blocked (TIMEOUT p=96). The 3rd ring's angle reads only **0.749 in the ORACLE arm too** → intrinsic angle fidelity of that ring, not a discovery failure. Clean number: one curved coord captures **0.94** of a circle's variance vs **0.52** for one linear coord. NOT an EV win — linear PCA-6 0.883 (0/3 rings recovered) | **SAFE NOW (QUALIFIED)** — factor recovery + factorized (no-joint-solve) delivery, **never** an EV win; REML transfer unestablished | landed + R-review-validated (§4.4); `missed_circle_diagnosis.json` (`4480383`) |
| **BT1** — Rust block-sparse Tier-1 (gam `4a06940cd`) | gauge-invariant block-sparse core | after the edition-2024 pattern-error fix (`4a06940cd`) `gam-sae` compiles; R-review ran `cargo test -p gam-sae --lib block` → **17 passed / 0 failed**, incl. `gauge_invariant_selection_and_loss_under_block_rotation` (with negative control), `planted_block_subspaces_recovered`, `fitted_block_frames_are_orthonormal`, utilization/stable-rank. FFI clean (no `#[allow]`, full-path prelude) | **SAFE NOW** — gauge-invariant block-sparse core verified (numeric + 17 in-repo tests green) | landed & green — hedge: block-fitter EV is in-sample; **no downstream headline EV yet** (SAFE claim = gauge-invariance + recovery, not an EV number) |

| **shadow_cone** — presence/amplitude decoupling (`shadow_cone/`) | can a block's presence be decoupled from its intensity? | synthetic weak-vs-absent presence (held-out AUC): **block norm 0.47** (≈chance — the "shadow") and a **reconstruction-only gate 0.47** both fail; only a **presence-supervised gate reaches 0.999**. Real-data η²: norm → template/context **0.67/0.82** (intensity is context-driven), angle → identity **0.89/0.87** (the coordinate carries the concept) | **SAFE NOW (QUALIFIED)** — decoupling is architecturally supported but **requires an explicit presence signal; reconstruction alone does not buy it** | landed & committed (`db9d1e5`, R-review SOUND) |
| **chart_transfer** — chart-coordinate invariance across prompts (`chart_transfer/`) | does the chart's coordinate transfer as a feature property? | **coordinate consistency SUPPORTED** (14 template families, LOTO, median circ-corr **0.95/0.81**) — the same token gets a consistent recovered angle across held-out templates, which a **directionless linear SAE cannot express**. On raw EV the 1-coord chart beats a single linear direction by a wide margin (both sets) but does **not** beat the 2-PC plane (weekday chart 0.26 < linear-2 0.43; month 0.12 > 0.05). Outlier templates rotate/degrade it (36–50% of folds <0.9) | **QUALIFIED positive** — the *coordinate* is largely a feature property (a strong tendency, not a law); it does not out-reconstruct the 2-PC plane | landed & committed (`e5662ec`/`8930bd6`; R-review) |
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

2. **The matched-null battery (now committed, `null_out/`) cleanly separates a robust claim from a
   marginal one.** Applying BSF's own discipline to the W7 circle probes:
   - **Month — real, survives every null.** C1 EV-parity beats a matched-spectrum Gaussian
     (gap-closed 1.043 vs 0.970, **p=0.012**), C2 cyclic order beats random labelling (0.833 = 10/12,
     **p=0.0004**), and basis-real corroboration is 1.00. Publishable as a genuine circle.
   - **Weekday — mixed, publish only the marginal order claim.** C2 cyclic order clears its null but
     only marginally (0.714 = 5/7, **p=0.042**, basis-real WEAK 0.22); **C1 EV-parity FAILS** the
     matched-spectrum null (p=0.16) — on weekday the "one curved coord = 2-PC parity" is *not*
     distinguishable from what a 1-D curve gets on a matched spectrum. So weekday cyclic order is
     "marginal (p≈.04)", never strong, and weekday EV-parity-as-circle-signature must not be claimed.
   - **Color-name — a clean negative (scoped).** A cheap Qwen color-NAME probe (8 hue words × 5
     templates) shows the tokens do **not** form a recovered hue circle (cyclic adjacency 1/8,
     label-perm **p=0.93**; basis-real 0.00). This is the discipline working — not every
     plausibly-cyclic token feature is a circle. Scope caveat: this is the name-token analog, **not**
     W7's big-model color-SWATCH claim (`color_geometry.py`, D=7168, unrunnable on this box), so it
     neither is nor refutes that result.
   - The phase-scramble is n.s. for both and is reported as a **diagnostic, not a gate**: the ordering
     is carried by the low-frequency power spectrum (smoothness), which does not require
     higher-harmonic phase-locking — the honest reading, not a failure.
   That our own discipline nulls one weekday claim while confirming the month circle is exactly what
   makes the surviving claims (month circle, dose calibration §3) credible.

3. **BT1 block-sparse core is now GREEN — the SAFE claim is gauge-invariance + recovery, not EV.**
   After the edition-2024 pattern-error fix (`4a06940cd`) `gam-sae` compiles and R-review ran
   `cargo test -p gam-sae --lib block` → **17 passed / 0 failed**, including the gauge-invariance
   test with a real negative control (a norm-changing map must change the loss), planted-subspace
   recovery, and orthonormal-frame checks; the FFI surface (`gamfit.block_sparse_dictionary_fit`)
   is clean. The one honest hedge: the block fitter's EV is in-sample/held-in (standard for a
   dictionary fit) and **no downstream headline EV has been produced** — so the publishable BT1
   claim is the verified gauge-invariant recovery property, not a reconstruction number.

4. **N-nursery: the publishable claim is FACTOR RECOVERY, not EV, and NOT a co-collapse cure.**
   R-review validated both the real held-out arms (`adfe50d`) and the clean p=96 synthetic
   (product of 3 circles): a chart-per-discovered-block nursery **discovers 3/3 blocks and
   reconstructs 3/3 rings (0.89–0.93 held-out EV), with 2/3 passing the strict angle bar
   (corr>0.8)** — composed held-out EV 0.834, matching the oracle-block upper bound (0.833) and
   beating the joint torch fit (0.756, which recovers 1/3), **without any joint K≥2 solve** and with
   REML width-blocked (TIMEOUT at p=96). The clean unconfounded number: one curved coordinate
   captures **0.94** of a circle's variance vs **0.52** for one linear coordinate. Discovery is
   label-free (energy-anticorrelation on X, train-only) with a consistent held-out split. Four
   hedges, all R-review-verified and MANDATORY on any headline:
   (a) **Not an EV win over linear.** Linear PCA-6 reaches 0.883 test EV (above the nursery's
   0.834) but recovers **0/3 circles**. The nursery's win is factor recovery + factorized
   (no-joint-solve) delivery, never raw EV.
   (b) **REML transfer unestablished.** REML is blocked at width; the #2027 co-collapse repro is now
   GREEN on the deterministic torch repro (fixed via cold-start chart deflation — see below), but
   production-width REML validation is still pending, so the co-collapse story is established on the
   torch repro, not yet on the production REML fitter.
   (c) **The "2/3" is an angle-bar count, not a recovery failure.** All 3 blocks are discovered and
   all 3 rings reconstructed (0.89–0.93 held-out EV); 2/3 pass the strict angle bar (corr>0.8). The
   3rd ring's angle recovers at 0.77 (just under 0.8) — and the **oracle** arm reads that same ring's
   angle at only **0.749**, so the shortfall is that ring's intrinsic angle fidelity, independent of
   discovery (`missed_circle_diagnosis.json`, `4480383`; REVIEW.md).
   (d) **The discovery≈oracle parity is N-sensitive.** It holds **at n=480**; at n=210 the
   energy-anticorrelation discovery mis-segments (blocks [3,1,2,1,1]) and drops to EV 0.60 (vs
   oracle 0.80). Phrase it "discovery matches the oracle at n=480," never a general "discovery
   loses nothing." (The 2/3 recovery itself still reproduces at reduced n.)
   The earlier real p=16 arm corroborates the recovery story (weekday adjacency 1.0 vs joint 0.429)
   but is EV-losing (linear 0.696 > joint 0.629 > nursery 0.576) and small-N noisy (circular_corr
   0.243 on ~28 rows) — suggestive, not decisive.

   **Related — the K≥2 co-collapse pathology is now reproduced AND fixed on the deterministic repro.**
   R-review independently re-ran the #2027 suite: **3/3 GREEN at HEAD (`7a93b1d06`)**, including the
   structural-separation test — on the deterministic two-circle repro, **EV recovers AND the atoms
   separate onto distinct curved factors at both widths** (per-factor even-fraction [0.998, 0.101] at
   p=16 and [0.963, 0.210] at p=96 — each atom owns one circle). The fix is cold-start sequential
   CHART deflation. Scope honestly: **not yet validated at production width on real data** (REML
   remains width-blocked in the current venv; a node REML rerun is staged) and the full manifold
   regression suite is still in progress. The evidence quality is itself worth noting: one biting
   regression test tracked the whole fix chain **broken → EV-only → separated**, so the green is
   earned, not asserted. The additive-generative-model / joint-K≥2 line accordingly softens from
   "still co-collapses" to **"fixed on the deterministic repro; production-width validation pending"**,
   and the nursery route (§4.4) **independently validates the same factorization principle the fix
   now applies at cold start.**

5. **G-bsf cyclic `full_ev` is in-sample**, and its real-data EV comparison is on the OLMo
   self-qualia axis, which is *linear* — so BSF not beating TopK there is expected, not a
   defeat of BSF. The block advantage is a **curvature/subspace** claim, best shown on genuinely
   multi-dimensional or curved features.

6. **MDL bits/token use in-sample EVs** (descriptive, not held-out). Fine for the crossover
   argument (`f*` depends on parameter counts and the spectrum, not on generalization), but the
   absolute bits inherit their source probe's in/out-sample status.

7. **Artifact-commit discipline (mostly closed).** `bsf_baseline/metrics.json`,
   `block_nursery/real_results.json`, and now the full `null_out/*.json` + `RESULTS.md` (P-null) are
   committed, so their numbers are directly citeable. The rule we adopted after an earlier miss
   (a cell cites a committed on-disk artifact or it is PENDING) still governs any late-landing lane;
   check `git ls-files` before citing a result JSON from this shared working tree.

8. **Dose calibration shows the weekday circle only.** The 12-token month loop triggers the
   pre-fix multi-modal auto-grow/co-collapse in that build; re-run against the guard-patched
   build before claiming month/hue dose calibration.

9. **Chart transfer is a coordinate result, not a reconstruction result.** Across 14 template
   families (LOTO), the chart's angular *coordinate* transfers consistently (median circ-corr
   0.95/0.81) — a genuine positive a directionless linear SAE cannot even express. On raw EV the
   1-coord chart beats a single linear direction by a wide margin but does **not** beat the 2-PC
   plane (weekday chart 0.26 < linear-2 0.43; month 0.12 > 0.05), and outlier templates
   rotate/degrade the coordinate (36–50% of folds <0.9; month template 9 at 0.04). Publish "the
   chart coordinate is largely an invariant feature property (a strong tendency, not a law)," not
   "the chart reconstructs better on held-out templates." (`chart_transfer/`, R-review.)

---

## 5. Related work — where ManifoldSAE sits in the 2026 stream

**Positioning.** SAEs discover scalar fragments; BSFs consolidate fragments into low-dimensional
supports; ManifoldSAE identifies, parameterizes, and controls the curved geometry living inside
those supports — with a metric that makes interventions calibrated.

**The model hierarchy.** SAE = scalar atom; BSF / SASA = linear *block* atom; ManifoldSAE = typed
*curved chart* atom carrying a metric. A block is the flat special case of a chart (the generator
`γ(t) = tD` — a straight line), so BSF sits one rung below on the same ladder.

The 2026 literature converges on the same diagnosis — a single direction is the wrong primitive
for a multidimensional concept — and splits on the fix:

| line of work | representative | stance vs ours |
|---|---|---|
| **Dilution / shattering diagnosis** | "Do Sparse Autoencoders Capture Concept Manifolds?" (arXiv 2604.28119); Goodfire, "Can SAEs Capture Neural Geometry?" | motivates the problem — scalar SAEs shatter a manifold into many diluted directions; we agree and give the constructive fix |
| **Subspace consolidation** (closest neighbours) | BSF (vision); SASA — "Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability" (arXiv 2606.06333, LLMs) | consolidate fragments into a *flat* block/subspace. SASA targets the same feature-splitting we diagnose; recovering the *coordinate inside* the consolidated support is our rung |
| **Region / local-geometry** | "From Directions to Regions: Decomposing Activations in Language Models via Local Geometry" (MFA; arXiv 2602.02464) | a different primitive (a Gaussian region chosen by locality, not co-presence); the BSF paper's own MFA critique applies. We select by co-presence *and* parameterize the geometry |
| **Mixture / other** | "SMIXAE: Towards Unsupervised Manifold Discovery in Language Models" (arXiv 2605.09224) | sparse mixture-of-autoencoders for multidimensional features; complementary |
| **Direction-paradigm engineering** | JumpReLU / BatchTopK / AbsTopK | orthogonal — better *scalar* dictionaries; they improve the rung below and compose with blocks/charts |
| **Geometry-aware steering** | "Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior" (arXiv 2605.05115) | corroborates our dose result — steering along the representation manifold follows the behavioural manifold, while linear steering cuts off-manifold; our chart supplies the metric that makes the dose calibrated (§3, R²=0.95) |

**What is ours, anchored on evidenced differentiators:** the measured description-length crossover
`f* ≈ 2p` (charts beat blocks once a feature fires more than ~2p times — §2); dose calibration on a
real model (R²=0.95, ~10× better than a metric-free linear latent — §3); a gauge-invariant
*certified* block/encode core (BT1, 17/0 tests — §3); typed topology (circle vs line — a line's
chart crossover is `f*=∞`, so the type self-controls); and per-atom REML evidence for model
selection instead of hyperparameter sweeps.

*(The five arXiv IDs and titles above were verified against arxiv.org, July 2026; the Goodfire
"Can SAEs Capture Neural Geometry?" and Block-Sparse Featurizers references are to the works this
document responds to. The one-line stances are our positioning, not summaries of each paper.)*

---

### Provenance

Artifacts: `bsf_baseline/` (G-bsf), `mdl_ladder/` (M-mdl), `dose_real_out/` (dose calibration,
real llama-3.1-8b L16), `null_out/` (P-null), `block_nursery/` (N-nursery), `review_checks/REVIEW.md`
(R-review), gam `a6f2c0e28` (BT1). Reproduce the MDL ladder:
`python mdl_ladder/mdl.py --probes --synthetic --frontier`.
