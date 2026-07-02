# R-review — adversarial review notes

Reviewer for the block-sparse + curved-chart featurizer fleet. Repos polled:
`/Users/user/gam` (main) and `/Users/user/Manifold-SAE` (main). I edit nothing
but this file.

## Baseline (established before lanes committed)

Reusable pieces BT1-block MUST build on rather than reimplement:

- `crates/gam-sae/src/frames.rs`
  - `GrassmannFrame` (l.500) with `polar_update(cross_moment)` (l.586) — polar
    factor of a p×r cross-moment via thin SVD → orthonormal frame U (the
    Grassmann/Stiefel retraction). `reconstruct_decoder`, `project_decoder`,
    `max_principal_angle`, `gauge_singular_values`.
  - `GrassmannCrossMoment` (l.709): `accumulate` + `polar_frame()`.
- `crates/gam-sae/src/sparse_dict/`
  - Current per-atom selection (`scoring.rs`): top-`s` by `|xᵀd|`, atoms
    unit-norm. `TopSSelector.offer` selects by `score.abs()`. This is the
    PER-ATOM analog; block-sparse must select by `‖z_g‖₂` over a b-dim block
    code, NOT by any single coordinate.
  - `codes.rs::solve_row_codes`: active-set ridge LS. Signed codes already
    (no ReLU). Good — block version must also keep signed codes.
  - `SparseDictFit` decoder rows unit-norm; scale identified by projection step.

Gauge-invariance target for BT1: any O(b) rotation R of block g's basis
(D_g → R D_g, z_g → R z_g) must leave BOTH selection (‖z_g‖₂ = ‖R z_g‖₂) and
the loss identical. Test must BITE: construct a fit, apply a random O(b)
rotation to one block, assert selection indices + loss unchanged; and a NEGATIVE
control that a non-orthogonal (norm-changing) map DOES change them.

## Lane status
- BT1-block: no commits yet (crates/gam-sae/src/sparse_dict/ block-* not present)
- N-nursery: experiments/block_nursery/ not present
- G-bsf: experiments/bsf_baseline/ not present
- M-mdl: experiments/mdl_ladder/ not present
- P-null: experiments/matched_null.py not present

## Prep notes (for fast review when lanes land)

N-nursery — likely forks `manifold_stability_sac.py` / `seed_stability.py`.
Checks to run when it lands:
- Planted-data leak: `data_planted` RETURNS the true `planes`. The nursery arm's
  chart seeds must come from residual-PCA / data (as SAC birth does), NOT from
  `planes`. Grep the nursery init for any use of the ground-truth planes/assign.
- Control arm (joint K≥2 `sae_manifold_fit`): must actually be invoked, bounded
  (timeout/iter cap), and its outcome recorded even on non-convergence — not
  silently skipped so the nursery arm "wins" by default. Memory says joint K≥2
  co-collapses / OOMs, so a subprocess wrapper with a recorded timeout is the
  honest control.
- Held-out EV: both arms must score EV on the SAME held-out split with the SAME
  demean/scale (per-template demean is mandatory here — memory
  [[manifold-sae-harvest-gotchas]]). Watch for train-EV in one arm vs test-EV in
  the other.
- Subprocess isolation: every `sae_manifold_fit` must be in a subprocess (memory
  [[gam-sae-manifold-fit-broken-build]] — stale-ext/non-PD/OOM in-proc). Confirm
  the wrapper exists around EVERY call, incl. the control.

P-null — `experiments/matched_null.py` (new). W7 cyclic claims.
- "Matched" = the null must preserve the stated invariants of the real statistic
  (e.g. phase-shuffle / rotation that keeps the marginal spectrum, radius) so the
  only thing destroyed is the cyclic structure. A null that also changes variance
  is not matched.
- Permutation count: p = (1+#{null ≥ obs})/(1+B); need B large enough that the
  smallest reportable p is below the claim (e.g. B≥999 for p<.001 headline).
- No peeking: null draws must not reuse the test statistic's fitted params.

## NEW lanes (scope update) — both in /Users/user/gam

Baseline of the finished fleet's DIRTY edits (what O-manifold/O-solve must land)
snapshotted at scratchpad `gam_dirty_baseline.diff`. Diffstat:
  bench/synth_sae_bench_manifold.py            +51
  crates/gam-math/src/probability.rs           +37
  crates/gam-model-kernels/src/inverse_link.rs +64
  crates/gam-sae/src/manifold/mod.rs           +3
  crates/gam-sae/src/manifold/outer_objective.rs +74
  crates/gam-sae/src/manifold/tests.rs         -144 (DELETIONS — scrutinize!)
  .../tests_collapse_bar_reachable_rank_1610.rs ±12
  crates/gam-solve/src/mixture_link.rs         +14
  crates/gam-spec/src/lib.rs                    +59
  tests/test_sae_manifold_accuracy_oos.py      ±4
  (untracked) crates/gam-sae/src/manifold/tests_frame_refresh_alpha_grad.rs

O-manifold checks:
- tests.rs shows -144 lines: a triage that DELETES 144 lines of tests is the
  prime "silently dropped intent" risk. When O-manifold commits, verify those
  deletions are (a) genuine relocations (the new tests_frame_refresh_alpha_grad.rs
  / tests_collapse_* files) not net loss of coverage, and (b) not removal of a
  test that would now fail. Diff committed tree vs `gam_dirty_baseline.diff`.
- Co-collapse repro test must BITE: it must fail/thrash on pre-fix code. Verify by
  checking out the test at the pre-fix commit and running it (in a temp build, or
  reason from the assert). If it passes before the fix, it doesn't prove anything.
- K=1 path must stay green (W7/W8 depend on it): after any seeding/anchoring
  change, `cargo test -p gam-sae` K=1 tests must pass.

O-solve checks (crates/gam-solve/ only):
- mixture_link dirty edit = widen `inverse_link_has_fisher_weight_jet` gate to
  admit LogLog + Cauchit (claims their 5-jet Fisher weight closes). VERIFY: does
  `fisher_weight_jet5_for_inverse_link` actually implement LogLog/Cauchit d1..d5?
  If the jet returns garbage/NaN for these, the gate widening enables Firth/
  Jeffreys on an unimplemented link — a real bug. Check before trusting.
- GpuRequiresDenseSystem already defined in gpu_kernels/arrow_schur.rs (l.56,200,
  220). O-solve must return it (not SchurFactorFailed) when hbb absent /
  penalty_op present, AND every caller (latent_inner.rs:371 matches
  SchurFactorFailed) must handle the new variant → CPU fallback, not panic.
- `cargo test -p gam-solve` must actually pass — run it, don't trust the message.

## Findings

### O-solve — mixture_link gate widening (PRELIMINARY: SOUND)
The dirty edit widens `inverse_link_has_fisher_weight_jet` to admit LogLog +
Cauchit. Verified against the code, NOT just the comment:
- `fisher_weight_jet5` (mixture_link.rs:286-287) routes both to
  `component_fisher_weight_jet5`, which genuinely implements LogLog (l.480/543/877)
  and Cauchit (l.499/560/902) — not a panic/fallthrough.
- Existing test `non_logit_probit_fisher_weight_jets_match_finite_differences`
  (l.2989) BITES: asserts W == mu'^2/(mu(1-mu)) to rel_err<1e-12 AND W'..W''''
  vs central FD to <1e-5..5e-4, for CLogLog/LogLog/Cauchit.
- `loglog_fifth_derivative_should_match_closed_form_sign` (l.3081) checks d5 vs a
  hand-derived closed form to 1e-15.
- `mixture_fisher_weight_jet_covers_loglog_and_cauchit_components` (l.3044)
  asserts the gate stays open + Firth-eligible for anchored mixtures.
Conclusion: gate widening is well-covered; NOT the "enable Firth on an
unimplemented link" bug I was watching for. Pending: confirm `cargo test -p
gam-solve` green on the committed tree (running now, bkjzah6n5).

### G-bsf — commit a401226 (VERDICT: SOUND / faithful)
Reviewed bsf.py + train.py AND verified numerically against the actual code
(`experiments/review_checks/check_bsf.py`, ALL PASS on the committed module):
- Tied encoder: grassmann `log_gamma` is a single 0-dim scalar (verified
  `shape==()`); `encode` computes `z = exp(log_gamma)·(x-b_dec)@Dᵀ` = one shared
  γ. bsf.py:136,154. FAITHFUL.
- Block-TopK by group ℓ2: `block_topk_mask` selects top-k of
  `vector_norm(z,dim=2)`; verified kept blocks == top-2 group-norm for every row.
  bsf.py:55-71. Signed codes (no ReLU anywhere; `z_sparse = z*mask`), mask carries
  no gradient (standard TopK STE convention). FAITHFUL.
- Grassmann projection identity: after `reproject_stiefel`, verified
  `z_g D_g == γ·P_g(x-b_dec)` to 4e-16 and rows orthonormal to 1e-10. bsf.py:203.
- Stiefel reprojection: QR of D_gᵀ with positive-R sign convention, applied to
  `decoder.data` in-place every `reproj_every` steps in grassmann mode only.
  bsf.py:204-219, cadence in `maybe_retract` (l.233). FAITHFUL.
- GAUGE INVARIANCE (the tied model's core property): rotating block 0's basis by a
  random O(b) rotation leaves ‖z_g‖ (selection) invariant to 1e-15 AND
  reconstruction (loss) invariant to 1e-15; NEGATIVE control (2× scale) DOES
  change reconstruction. So the invariance test genuinely bites.
- AuxK: targets `k_aux` LOWEST-utilization blocks (`topk(util_ema, largest=False)`)
  and reconstructs the RESIDUAL `x - x_hat.detach()` WITHOUT the decoder bias.
  bsf.py:182-200. util_ema tracks activation frequency (dead-block resurrection,
  Gao et al.). FAITHFUL.
- No synthetic leak: `make_planted` returns `true_bases` used ONLY in
  `match_blocks_to_truth` AFTER training; never enters training. train.py:130-141.
- Matched comparison: real phase holds F=64 (n_latent) and L0=8 (k·b) constant
  across b∈{1,2,4,8}; PCA + per-feature std are TRAIN-ONLY (no test leak);
  val_EV on held-out. FAIR.
MINOR (not a defect, note for headline): cyclic phase (weekday/month) trains and
evaluates on the SAME X (train_bsf(...,X,...,X)); its `full_ev` is IN-SAMPLE. The
headline there is the structural claim (one block, adjacency accuracy) which is
legitimate, but do NOT publish cyclic `full_ev` as a generalization number.

### O-manifold — commit e09e6956c (VERDICT: clean landing of fleet batch)
This is task-#7 "land fleet batch", NOT yet the co-collapse fix (#8 pending).
- Committed hunks for all 5 load-bearing files (outer_objective, mixture_link,
  gam-spec, inverse_link, probability) are BYTE-IDENTICAL to the pre-commit dirty
  baseline (md5 match) — nothing silently altered while landing.
- tests.rs -144: the 3 deleted tests (`streaming_polar_refresh_reorients_frame`,
  `small_p_zero_decoder_stays_full_b`,
  `forward_alpha_data_derivative_skips_ungated_atom_1026`) are all RELOCATED intact
  into the new `tests_frame_refresh_alpha_grad.rs` (+163). Pure relocation (cfg-test
  scanner pattern), NOT coverage loss. Verified by name presence.
- The substantive math change (reachable_dictionary_rank → rank of CONCATENATED
  chart design instead of Σ per-atom ranks, #C5) is a genuine correctness fix:
  removes an upward bias in the collapse null floor (double-counting shared
  directions). Sound.
PENDING: `cargo test -p gam-sae` green + the co-collapse repro/fix (#8) when it
lands — the repro test must BITE on pre-fix code.

### M-mdl — commit 75f6304 (VERDICT: scorer SOUND, one latent caveat)
Hand-verified mdl.py scorer against a toy (all terms exact match):
code_coeff=0.5·log2(1+v/δ²), dict=n_params·l_param, total, bits/token — all match.
- Units consistent (bits throughout); rate term is exact scalar R(D)=0.5log2(1+SNR).
- Selection bits `log2 C(G,k)` present in `score()` for EVERY featurizer (both arms).
- Crossover `dcode = code_b−code_c` correctly OMITS selection (verified == hand
  code_b−code_c); this cancels ONLY when block & chart share (G,k). In every built
  ladder both use g_dict=1,k_active=1 → sel=0, so harmless. CAVEAT: if a lane feeds
  a JSON payload with DIFFERENT g_dict/k_active for block vs chart, crossover_firings
  silently ignores the selection difference (mdl.py:185-192). Not triggered by the
  committed ladders; flag if any lane passes asymmetric (G,k).
- NOTE: all built ladders use IN-SAMPLE ev (insample_ev / same-X spectra). The MDL
  bits/token are descriptive, not held-out — fine for the crossover argument, but
  the underlying EVs inherit whatever in/out-sample status their source probe has.

### BT1-block — commit a6f2c0e28 (VERDICT: DESIGN SOUND, but DOES NOT COMPILE + tests absent)
Design reviewed in full (block.rs, 1108 lines) and the gauge math verified
numerically (replicating the exact Rust `block_gates`+`reconstruct_row`):
gate error 0.0 under a random O(b) rotation, selection identical, loss error
3.5e-15; negative control (2× scale) DOES change the loss (property bites).
- Gauge invariance CORRECT BY CONSTRUCTION: routing/report see a block only through
  `‖w_g‖₂=‖x D_gᵀ‖₂` (block_gates l.216, invariant to w_g→w_gRᵀ); reconstruction is
  `γ·x D_gᵀD_g` (reconstruct_row l.238) and `D_gᵀD_g` is invariant to D_g→RD_g since
  RᵀR=I. Holds for ANY left-O(b), orthonormal or not.
- Signed codes, no ReLU (code_row l.495 `gamma*wr`, can be negative). Presence
  (`‖z_g‖₂` gate) vs amplitude (signed z_g) decoupled.
- ONE shared scalar γ, closed-form LS refresh (refresh_gamma l.557). 
- REUSES GrassmannFrame::polar_update for the Stiefel step (orthonormalize_block
  l.298, refresh_frames l.664) — NOT a reimplementation. GS fallback only for
  rank-deficient seeds.
- Revival seeds from worst-residual ROWS (never PCs, house rule), distinct row
  groups so revived blocks don't duplicate (revive_dead_blocks l.693).
- stable_rank_symmetric is a gauge invariant (trace/λmax, l.889). Consistent.
- Fit loop: seed(farthest-point, reused)→[γ→frames→revive→re-encode→EV]. EV is
  HELD-IN (in-sample) — standard for a dict fit, but any headline comparing BT1 EV
  must keep that consistent.
ISSUES:
  [HIGH] Commit a6f2c0e28 DOES NOT COMPILE. (1) block.rs ends with
    `#[cfg(test)] #[path="block_tests.rs"] mod block_tests;` but block_tests.rs does
    NOT exist → `cargo test -p gam-sae` fails to build the whole crate's tests.
    (2) refresh_frames (l.597) & revive_dead_blocks (l.696) hold `decoder: &mut
    Array2` and the committed code called `reconstruct_row(xi, decoder, …)` where
    the param is `ArrayView2` — a type error that breaks even `cargo build`. BT1 is
    fixing #2 in the working tree now (decoder→decoder.view()); #1 (the tests) is
    task #10, still pending.
  [HIGH] The load-bearing gauge/parity/recovery tests DO NOT EXIST yet
    (block_tests.rs absent). The gauge invariance is currently code-correct + my
    numeric check passes, but there is NO in-repo test. When it lands it MUST:
    rotate D_g→R·D_g and RE-ENCODE (the tied code follows automatically — there is
    no separately-stored code to rotate), assert gate+loss identical, AND include
    the negative control (a norm-changing map must change loss) or it won't bite.
    Also assert frames stay orthonormal after refresh_frames.

### P-null — commit 5fd1465 (VERDICT: SOUND, one honesty caveat on weekday)
matched_null.py — genuinely careful; catches its own subtle traps.
- N2 label-perm null is CORRECT where a naive version would be broken:
  `cyclic_adjacency_accuracy(angle, order)` relabels BOTH adjacency sets when
  `order` is permuted (permuting order = no-op), so the null instead randomises the
  recovered ORDERING (_adj_null_sample, l.205). Verified numerically: null adj mean
  0.33 (n=7) / 0.18 (n=12); perfect circle p=0.0028/0.0002; null-mean→p≈0.4. The
  p-machinery bites and is well-calibrated.
- N3 matched-spectrum preserves the per-PC eigenspectrum (red is PCA coords → indep
  Gaussian reproduces the covariance); primary statistic is fit-quality-normalised
  gap-closed (cev−l1)/(l2−l1). Matched. (Caveat: if a lane passes a NON-PCA
  fitted_basis, red.std drops cross-covariance — degrades the match. Not triggered
  by the W7 CLI path, which uses PCA.)
- N4 phase-scramble: HONESTLY gated — a pure fundamental-mode circle is invariant to
  per-column phase scrambling, so they compute fundamental_mode_fraction and only
  let phase-scramble count toward the verdict when applicable (FMF<0.9). Exactly
  right; a lesser version would falsely pass/fail here.
- Empirical p uses (hits+1)/(B+1), never 0. B: n_perm=5000 (min p 2e-4),
  n_gauss=128 (min p 7.7e-3) — enough for the α=0.05 verdicts.
- NO PEEKING: observed-in-battery stats recomputed at the SAME single-seed/600-step
  budget as the nulls; the best-of-2 headline is recorded as `canonical` for
  reference ONLY, not used for p. Correct discipline.
- Subprocess-isolated retried CLI; reuses torch-backend curved_fit, NOT REML
  sae_manifold_fit (OOM/segfault-safe). Consistent with house rules.
CAVEAT [MED, honesty]: weekday C2 (cyclic order) is MARGINAL — adjacency 0.71 at
n=7 gives p≈0.039 against this null (and the battery's single-seed budget may score
BELOW the best-of-2 headline 0.71, pushing p up further). Publish weekday cyclic
order as "marginal (p≈.04)", not a strong result. Month (n=12, adj 0.83) is robust.
C1 EV-parity and the month cyclic claim are safe.

### N-nursery — commits f797213, 2a4834a (VERDICT: design VALID w/ caveats; NO RESULTS YET)
block_nursery.py reviewed against the 4 experimental-validity checks. The committed
result JSONs (synthetic_results.json / real_results.json) contain ONLY the initial
data stub — `arms` is empty — so the arms have NOT run; there are no N-nursery
headline numbers to validate yet, only the design.
- Control arm (joint K≥2) HONEST: REML joint attempted in a capped (120s) subprocess
  and recorded as TIMEOUT_BLOCKED/OOM even on failure (reml_joint_isolated l.228);
  torch joint (target_k=K, additive) run + recorded with per-circle recovery; plus an
  over-complete K=2·ncirc arm. Bounded and recorded. GOOD.
- NO planted-leak in the honest arm: the DISCOVERED arm (discover_blocks l.258) uses
  NO labels/planes — sparse_dictionary_fit on X + coactivation-affinity clustering.
  `theta` (true angles) enters run_nursery ONLY for scoring (best_planted_circle_corr,
  l.375), NEVER the fit (fit is on Z=Xc@Q). The ORACLE arm DOES use planted planes
  (oracle_blocks l.333) but is explicitly labeled the "factorization upper bound" and
  reported separately from the discovered arm. Clean separation. GOOD — publish the
  DISCOVERED arm as the result, oracle as ceiling.
- Subprocess isolation on EVERY curved fit AND the REML fit (fit_curved_isolated /
  reml_joint_isolated both subprocess.run with timeout; workers reset sys.excepthook).
  sparse_dictionary_fit runs in-process, which is FINE — it's the stable linear lane,
  not the OOM-prone sae_manifold_fit. GOOD.
- EV consistency: ALL arms use IN-SAMPLE ev() on the same X (no held-out split). This
  is CONSISTENT across arms (joint in-sample vs nursery in-sample = fair comparison),
  but it is NOT a generalization number — especially real (N=95, P=16, charts can
  overfit). Report as in-sample factorization EV, not held-out.
CAVEATS to enforce on any eventual N-nursery headline:
  [HIGH] The co-collapse is demonstrated on a TORCH proxy fitter, not REML. REML —
    the production fitter the hypothesis "is really about" — is recorded BLOCKED. So
    "nursery cures co-collapse" is a torch-joint-vs-torch-nursery result; transfer to
    the REML fitter is UNESTABLISHED. State this explicitly.
  [MED] Real Arm B headline uses SET-membership-supervised 2-planes (each set's top-2
    PCs) as blocks; fully-unsupervised discover_blocks is only a side cross-check
    there. Set identity is legit metadata (not the cyclic answer), but say so.
  [MED] All EVs in-sample (above).

### BSF_RESPONSE.md synthesis — commit 605e716 (VERDICT: honest structure, TWO UNBACKED numbers)
The doc is unusually honest (cites this review as §4) and its stated rule is "empty
cells marked PENDING, never filled with placeholders." Most cited numbers DO trace to
artifacts — BUT two "landed" head-to-head numbers trace to NO artifact and must be
pulled to PENDING or corrected:

  [HIGH] G-bsf "synthetic subspace recovery R²=0.76 (vanilla)" (§3 row 1) is UNBACKED.
    bsf_baseline/metrics.json contains only keys ['real','cyclic'] — the SYNTHETIC
    phase never ran/saved. "0.76" appears nowhere in bsf_baseline/. (The G-bsf REAL
    numbers ARE backed and correct: TopK b=1 EV 0.4489 ✓, BSF b=2..8 0.425→0.325 ✓;
    cyclic adjacency 1.00 ✓, 7/8 blocks active ✓, winning stable rank 1.98 ✓,
    full_ev 0.976 correctly flagged in-sample.)

  [HIGH] P-null REAL-weekday results (§3 row 5 + §4.2): "circular correlation 0.93,
    p=0.010", "discrete adjacency 0.43, p=0.43", "C1 gap-closed p=0.48" trace to NO
    artifact. null_out/ contains ONLY null_synthetic_weekday.json, which is (a) the
    SYNTHETIC planted-circle set (name="weekday" but n=35 planted), NOT the real
    harvest, and (b) STALE — emitted by a PRE-5fd1465 matched_null.py (its
    matched_spectrum has old keys p_parity_vs_null / p_curved_beats_lin1_vs_null; the
    committed code's `p_gap_closed_vs_null` does NOT appear, yet the doc cites a
    "gap-closed" p — a statistic only the NEW code emits, from a run whose output is
    not on disk). The stale artifact's actual values (circ 0.965/p 0.0037, adjacency
    0.714/p 0.039, curved_beats_lin1 p 0.41) do NOT match the doc's (0.93/0.010,
    0.43/0.43, 0.48). So the "real weekday nulls" in the doc are unverifiable — either
    an unsaved run or fabricated. Per the doc's own standard → PENDING until a
    committed null_out/null_weekday.json (real, current-code) exists.

Everything else in the doc I can vouch for: MDL f*=2p / crossover table cites
mdl_ladder/results.json; the caveats §4.1/3/4/5/6/7 are accurate and match my review.

### BT1-block FIX — commit 097b1c1de (VERDICT: RESOLVES the compile+tests HIGH issues)
- block_tests.rs now EXISTS + committed (328 lines); the two decoder.view() fixes +
  gate-packing tidy landed (block.rs +7/-5). FFI (geometry_ffi.rs) + Python surface
  (gamfit/_sparse_dictionary.py, pure-numpy OOS transform) added (task #11).
- Gauge test `gauge_invariant_selection_and_loss_under_block_rotation` (l.101) is
  CORRECT and bites the right property: it rotates block g_rot's basis by a random
  O(b) `random_orthogonal(b)`, RE-ENCODES (block_projections_row→gates→route→row_loss
  recomputed on the rotated decoder — the tied code follows automatically, exactly the
  property lead asked for), and asserts selection + per-block gate + loss invariant.
  Would fail if block_gates keyed on a coordinate instead of the ℓ₂ norm.
- Also tests: γ-invariant routing (l.181), presence/amplitude decoupling (l.154),
  planted-block subspace recovery via principal angle (l.192), fitted frames
  orthonormal (l.247), utilisation∈[0,1] + stable-rank reports (l.300+).
- MINOR: the gauge test has no explicit NEGATIVE control (a norm-changing map that
  SHOULD change loss). The degenerate "constant loss" breakage is covered indirectly
  by the recovery test, and my own numeric check included the neg-control (it bit).
  Nice-to-have, not a blocker.
PENDING: confirm `cargo test -p gam-sae block` green on this commit (running,
b0eonwlv5) and that gam-pyffi still builds (FFI touched ffi_prelude — watch the
include!/prelude gotchas).

### M-mdl selection-bits FIX — commit 1d9f843
crossover_firings now accounts for the selection-bits delta (addresses my MED caveat).
Re-check when I re-pull: verify dcode/f* now include Δ selection bits and the built
ladders (all g_dict=1 → Δsel=0) are unchanged.
