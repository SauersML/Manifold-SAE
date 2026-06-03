# AUTO_EXP_PLAN — composition-engine coverage map

## Installed gamfit

- **Installed gamfit** (`/Users/user/Manifold-SAE/.venv`): `0.1.145`
  (standing rule: always track newest gamfit). The joint manifold-recovery
  objective below depends on knobs shipping in the *upcoming* release beyond
  0.1.145.

The composition primitives (`LatentCoord`, `TopologyAutoSelector`,
`IBPAssignmentPenalty`, `ARDPenalty`, `OrthogonalityPenalty`,
`TotalVariationPenalty`, `GumbelTemperatureSchedule`, `Circle` / `Torus`
/ `Cylinder` / `Sphere` / `EuclideanPatch`, `glm_reml_fit_latent`, etc.)
are first-class Rust primitives in this release; the experiments were
migrated off the hand-rolled numpy fallbacks (see the gamfit-0.1.123
migration commits). The coverage map below records which corners of each
primitive an existing experiment exercises vs. which are still untested.

## Primitive → existing-experiment coverage

| Primitive | Existing experiment(s) | Status |
|---|---|---|
| LatentCoord (Euclidean d) | auto_exp_17, 21, 23 | covered |
| LatentCoord (manifold=Circle) | auto_exp_21, 23 | covered |
| LatentCoord (manifold=Sphere/Torus/Cylinder) | none | untested |
| Fisher-Rao W (per-row IRLS weight) | auto_exp_21, 23 | covered |
| ARDPenalty | auto_exp_21, 23 | covered |
| OrthogonalityPenalty | auto_exp_23 | covered |
| IsometryPenalty | none | untested |
| SparsityPenalty (smoothed L1) | auto_exp_24 | covered |
| TotalVariationPenalty (forward_1d) | auto_exp_24 | covered |
| TotalVariationPenalty (graph_edges) | none | untested |
| IBPAssignmentPenalty / sae_manifold_fit | auto_exp_18, 20, 22; manifold_recovery, manifold_falsifier | covered |
| SoftmaxAssignmentSparsityPenalty | none | untested |
| decoder_incoherence_weight (#671, separability lever) | manifold_recovery (check 2, self-gates until knob ships) | upcoming |
| nuclear-norm embedding-rank selection (#672) | none yet | upcoming |
| ScadMcp non-convex sparsity | none yet | upcoming |
| gauge-conditional topology evidence (#673) | manifold_recovery (topology menu select) | upcoming |
| per-atom uncertainty (posterior bands) + typical coordinate range | none yet | upcoming |
| GumbelTemperatureSchedule | auto_exp_22 | covered |
| select_topology() one-shot | auto_exp_19 | covered |
| select_topology(score="laml") / "bic" | none | untested |
| select_topology + custom BasisSpec list | none | untested |
| `penalties=` kwarg with ≥3 stacked penalties | none | untested |
| NegBin / Tweedie / Gamma `family=` on glm_reml_fit_latent | none | untested |

## Canonical recovery harness (joint `sae_manifold_fit`)

Separate from the color/composition experiments above, the canonical multi-atom
recovery verification lives in:

- `experiments/manifold_recovery.py` — the gate: (1) K=2 superposed-circle
  recovery under IBP (PASS if recon R² > 0.9), (2) headline incoherence ON vs OFF
  across a coherence sweep (ON must raise recovered-tangent σ_min AND lower the
  cross-atom decoder cross-Gram / improve coord recovery), (3) single-atom
  out-of-class specification margin (K=1, runs today).
- `experiments/manifold_falsifier.py` — keystone falsifier + shared scoring
  (`circ_procrustes_r2`, `tangent_sigma_min` = the σ_min identifiability metric).
  `--selftest` validates the scoring before the fit unblocks.

Both use canonical `assignment="ibp"` and self-gate BLOCKED (never false PASS)
when a fit diverges or the installed gamfit lacks the incoherence knob. The K ≥ 2
joint solve currently diverges upstream (fix in progress); the single-atom check
runs today and the rest go green once the solver fix + incoherence knob land.

## Prioritized new experiments

| # | Slot | One-line hypothesis |
|---|---|---|
| P1 | auto_exp_26 | TotalVariationPenalty with **graph_edges** built from hue-kNN on cogito 949 colors gives strictly fewer atom transitions than the 1D-hue-ordered forward-diff baseline at matched per-hue R² (tests `difference_op=graph_edges`). |
| P2 | auto_exp_27 | Three-penalty stack `[Orthogonality, ARD, TV-graph]` on a Circle-manifold LatentCoord recovers the perceptual-hue axis with **fewer effective dims (ARD prunes) AND fewer transitions (TV bands) AND stable rotation gauge** than any pairwise subset (tests `penalties=` with 3 stacked analytic penalties). |
| P3 | auto_exp_28 | NegBin GLM on **per-prompt color-token count** as response (with PC-16 latent as predictor) yields strictly lower AIC than Gaussian on the same predictors, justifying `family="negbin"` for count-like cogito signals (tests `glm_reml_fit_latent` non-Gaussian families). |
| P4 | auto_exp_29 | `select_topology(score="laml" vs "reml" vs "bic")` with a custom 7-candidate basis pool (default 5 + EuclideanPatch(d=3) + Sphere) picks **the same winner under all three scores** on cogito PC-16, demonstrating evidence-criterion robustness. |
| P5 | auto_exp_30 | Joint Riemannian-Circle LatentCoord + Fisher-Rao W from **local kNN-covariance behavioral metric** (not just per-row residual variance like auto_exp_21) recovers smoother θ vs HSV-hue than the diagonal-W variant (tests behavioral-metric sourcing). |

## Selection for PHASE 3

Pick the two most contrasting + informative:

- **P1 (auto_exp_26)** — directly exercises an UNCOVERED corner of an
  already-tested primitive (`TotalVariationPenalty.difference_op =
  graph_edges`). Cheap, clean falsifiable.
- **P2 (auto_exp_27)** — first 3-penalty stack on the `penalties=`
  kwarg; the composition-engine claim "penalties just compose"
  needs an empirical test beyond pairs.

Skipping P3 (count response requires harvest reload of per-prompt token
counts not in cached results.json), P4 (auto_exp_19 already covers
`select_topology` end-to-end fallback; LAML/BIC variation can be a
follow-up), P5 (Fisher-Rao W sourcing variant is incremental to
auto_exp_21).
