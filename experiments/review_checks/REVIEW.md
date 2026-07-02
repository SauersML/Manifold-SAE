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
