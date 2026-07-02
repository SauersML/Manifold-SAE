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

_Status: live document, updated as lanes commit. Last sync: **CENTERPIECE added** — the unified real-data ladder (§2): BSF reproduced (direction 36.6 → block 6.4 bits/token, selection cost 32→3) then extended one rung (circle-chart codes 1 intrinsic coord; block→chart `f*`≈12–22 on weekday/month). BT1 GREEN (`4a06940cd`, 17/0); G-bsf synthetic committed (`33731de`: Grassmannian 0.9855, vanilla 0.8185) + cyclic held-out EV (weekday 0.82, month 0.95); N-nursery = recovery-not-EV; P-null null p-values PENDING (uncommitted); K≥2 co-collapse repro RED → unpublishable._

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

**Rung 5, one rung further** (M-mdl `score_json`, cyclic weekday/month — where curvature
exists, and where G-bsf's own block-finding lands: a single b≈4 block captures the cycle at
held-out EV 0.82/0.95, cyclic adjacency 1.0, coord stable rank 2.4 = the circle's extrinsic
dim). The circle-chart codes that cycle from **one intrinsic coordinate** (the angle) where the
block codes ~2 extrinsic dims — continuing the descent by collapsing the *code dimension*:

| feature | block (held-out EV) | circle-chart | Φ (extra harmonics) | crossover `f*` |
|---|---|---|---:|---:|
| weekday | b=2, EV 0.82, adj 1.0 | d_i=1 (angle) | 12 | ≈22 (matched `2p`=12) |
| month | b=2, EV 0.95, adj 1.0 | d_i=1 (angle) | 12 | ≈16 (matched `2p`=12) |

A weekday/month feature fires far more than ~12–22 times in any corpus (`f ≫ f*`), so the
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
| **P-null** — matched-null battery (`matched_null.py`) | real cyclic claims survive matched nulls | *Descriptive W7 facts (committed `curved_feature_probes.json`):* curved(1 coord) ≈ linear(2 PC) ≫ linear(1 PC); month cyclic adjacency clean, weekday 0.714. *Matched-null p-values:* **PENDING** — `null_out/null_weekday.json` (real harvest, current `matched_null.py`) is **not committed**, so no null p-value is citeable | **code SOUND** (R-review 5fd1465); **null verdict PENDING a committed artifact** | PENDING — P-null committing `null_out/*.json`; refresh once on disk & R-review-verified |
| **N-nursery** — chart-per-block vs joint-K (`block_nursery/`) | nursery beats co-collapsing joint K≥2 fit | **structure recovery under matched budget** (real held-out L8, N=95, 70/30, R-review-validated `adfe50d`): nursery recovers the weekday circle from one curved coordinate — cyclic adjacency **1.0 vs joint 0.429** (month 0.417 vs 0.25); on **held-out EV linear beats both** — PCA-4 0.696 > joint torch 0.629 > nursery 0.576. Width diagnostic: REML converges at P=16 (61s) but hangs at P=96 | **QUALIFIED** — a *recovery* advantage at modest EV cost, **never** an EV win; **not** a co-collapse demo (REML converged at P=16) | landed + R-review-validated (§4.4) |
| **BT1** — Rust block-sparse Tier-1 (gam `4a06940cd`) | gauge-invariant block-sparse core | after the edition-2024 pattern-error fix (`4a06940cd`) `gam-sae` compiles; R-review ran `cargo test -p gam-sae --lib block` → **17 passed / 0 failed**, incl. `gauge_invariant_selection_and_loss_under_block_rotation` (with negative control), `planted_block_subspaces_recovered`, `fitted_block_frames_are_orthonormal`, utilization/stable-rank. FFI clean (no `#[allow]`, full-path prelude) | **SAFE NOW** — gauge-invariant block-sparse core verified (numeric + 17 in-repo tests green) | landed & green — hedge: block-fitter EV is in-sample; **no downstream headline EV yet** (SAFE claim = gauge-invariance + recovery, not an EV number) |

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

2. **Real cyclic-probe claims split into committed descriptive facts and a PENDING null verdict.**
   *Committed (from `curved_feature_probes.json`, on disk):* one curved coordinate reconstructs
   about as well as two linear PCs and better than one (curved(1) ≈ linear(2) ≫ linear(1)); the
   recovered angle orders month cleanly and weekday at adjacency 0.714 — these are representational
   facts and are SAFE. *PENDING:* whether these survive BSF's matched-null discipline — the
   p-values (label-permutation, matched-spectrum, phase-scramble) — is **not citeable yet**: the
   real-harvest `null_out/null_weekday.json` from the current `matched_null.py` is not committed.
   The matched-null CODE is R-review-verified sound (`5fd1465`); the numbers must wait for the
   committed run. When it lands, weekday cyclic order is expected to be only *marginal* and must be
   phrased that way, never as a strong result. We are deliberately holding these p-values to the
   doc's own rule — a cell cites an on-disk artifact or it is PENDING — rather than quoting an
   uncommitted run.

3. **BT1 block-sparse core is now GREEN — the SAFE claim is gauge-invariance + recovery, not EV.**
   After the edition-2024 pattern-error fix (`4a06940cd`) `gam-sae` compiles and R-review ran
   `cargo test -p gam-sae --lib block` → **17 passed / 0 failed**, including the gauge-invariance
   test with a real negative control (a norm-changing map must change the loss), planted-subspace
   recovery, and orthonormal-frame checks; the FFI surface (`gamfit.block_sparse_dictionary_fit`)
   is clean. The one honest hedge: the block fitter's EV is in-sample/held-in (standard for a
   dictionary fit) and **no downstream headline EV has been produced** — so the publishable BT1
   claim is the verified gauge-invariant recovery property, not a reconstruction number.

4. **N-nursery: the publishable claim is FACTOR RECOVERY, not EV, and NOT a co-collapse cure.**
   R-review validated the real held-out arms (`adfe50d`; 70/30 split, EV on test throughout,
   label-free train-only discovery — the earlier in-sample caveat is CLOSED). The enforced honest
   reading:
   (a) **The nursery does NOT win on held-out EV** — linear PCA-4 (0.696) > joint torch (0.629)
   > nursery composed (0.576). The "nursery matches-or-beats joint EV" half of the hypothesis
   FAILS on this real case. Its genuine advantage is **factor recovery**: it recovers the weekday
   circle cleanly (cyclic adjacency 1.0) where the joint fit does not (0.429), at a modest EV cost.
   (b) **This is not a co-collapse demonstration.** REML `sae_manifold_fit` CONVERGED here (small
   P=16, 61s), so this real case does not exhibit the full-width co-collapse the hypothesis is
   about — it **cannot** be cited as "REML co-collapses, nursery fixes it." The co-collapse claim
   rests **entirely on the still-PENDING synthetic P=96 arm** (where REML hangs); it remains a
   torch-proxy story, unestablished for the production REML fitter.
   (c) **Small-N noise:** weekday adjacency 1.0 but circular_corr only 0.243 on ~28 test rows —
   treat the real recovery numbers as suggestive, not decisive.

   **Related — the K≥2 co-collapse "fix" is RED end-to-end.** O-manifold's root-cause fix chain
   landed (deflation/anchor/ownership `465ad67a0`, reseed cooldown `3ddf58c03`, repro `f7991e5c8`)
   and its components are individually verified, **but the collapse detector does not fire on the
   stuck-at-null mode** — the reseed trigger never engages (reseeds=0, EV≈−0.0000), so Parts A/B/C
   never run and its own regression test is RED (R-review `10a6f56`/`649ff7d`: 2 fail / 1 pass;
   only the deflation unit guard passes; P=16 is not the P=96 hang regime). Trigger fix in progress
   (O-manifold, task #8 reopened). So "we fixed the K≥2 co-collapse" **must not be published**; any
   "additive generative model / joint K≥2 curved fit works" line must stay hedged to *K=1 +
   block-sparse Tier-1 only — the joint K≥2 curved fit still co-collapses (repro red)*.

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
