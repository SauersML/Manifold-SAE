# Creative experiment ideas

A backlog of experiments worth running once we have time + compute.
Sorted by "would generate surprising result" weight rather than ease.

## High-novelty causal + interpretability tests

### 1. Counterfactual atom ablation

For a prompt that strongly activates atom k, set atom k's amplitude to
zero, re-run the LM, measure how much the output distribution shifts.
Stronger than correlational steering — tests whether the atom is
*necessary* for encoding its concept. If ablating a magnitude atom on
"There were 500 apples." causes the LM to predict a generic
number-token rather than the right magnitude, the atom is causally
load-bearing.

### 2. Cross-SAE alignment

Train two Manifold-SAEs with different random seeds on the same
activations. Hungarian-match their atoms by direction similarity in
residual stream. If the matched pairs encode the same concepts (same
top-firing tokens), the architecture is *identifying universal*
features — not artifacts of optimization.

### 3. Atom-as-direction probe

For each atom k, the gradient `∂t_k/∂x` at a firing token gives the
local "atom direction" in input space. Compare these gradient
directions across atoms — do they cluster? Do they match known
interpretable directions from probing literature?

### 4. Recursive Manifold-SAE

Train an SAE-2 on the atom firings (positions, amplitudes) of an
SAE-1. What does the meta-atom encode? If concepts are hierarchically
organized (e.g. "size + color + magnitude" → "object descriptor"),
meta-atoms should pick this up.

### 5. Atom mortality curve

Track per-atom firing rate over training. Which atoms die early?
Which come alive late? Plot the distribution. If certain atoms always
die under random init, that's evidence of architectural over-allocation.

### 6. Adversarial atom hardening

Given a trained SAE, find a perturbation of an input token that
maximally CHANGES which atoms fire (max ‖Δfiring‖). How big does the
perturbation need to be? Robust atoms are perturbation-stable.

### 7. Polysemy-induced atom split

Identify polysemantic atoms (from `atom_analysis.py` cluster count).
Force a single such atom into TWO atoms during training (architectural
intervention: clone the atom, perturb its W_k slightly, retrain). Do
the two atoms split the polysemy?

## Architectural / representation tests

### 8. Topology benchmark

Plant 2D manifolds with KNOWN topology: sphere (non-square,
non-periodic), torus (both axes periodic), cylinder (one periodic),
plane. Train Manifold-SAE 2D variants with each topology choice;
report recovery. Tests whether the architecture's basis choice
(periodic vs non-periodic Duchon) actually matters in practice.

### 9. Active learning sample efficiency

Instead of uniform sampling from wikitext, pick the next training
token to maximize EXPECTED ATOM COVERAGE (atoms not-yet-active under
the current model). Compare convergence to uniform sampling. Does
the architecture benefit from coverage-aware curricula?

### 10. Atom temperature transition

Add a temperature parameter τ on TopK gating. At τ → 0, hard TopK;
at τ → ∞, all atoms fire. Sweep τ and watch metrics. Is there a phase
transition where atoms suddenly become interpretable?

### 11. Per-atom intrinsic-rank decision

Currently `R` is global config. Let each atom learn its own intrinsic
rank via a gated softmax over R values per atom. Atoms with simple
1D structure use R=1; complex curves use R=4. Hypothesis: the
architecture self-discovers feature complexity.

### 12. Manifold-SAE on vision residuals

Apply the architecture to ViT or U-Net activations. Do the same
manifolds appear (color = continuous; size = continuous)? Tests
whether the "continuous feature on a manifold" framing is
cross-modal.

### 13. Multilingual concept transfer

Qwen is multilingual. Train the SAE on English text. Does the
magnitude atom still fire on numbers in Chinese, German, etc.? Test
the universality of the recovered manifold.

## Benchmark constructions

### 14. AxBench-style atom-driven steering

Use `experiments/steering_causality.py` as the bones; sweep multiple
atoms × multiple t-values; report which steering paths produce
COHERENT semantic shifts in the LM output. Compare to direction-only
steering on a vanilla SAE atom of similar firing pattern.

### 15. Down-stream task probe

Train SAE on layer L of Qwen. For tasks (sentiment, magnitude,
entailment), train a linear probe on SAE features vs raw activations.
If SAE features outperform raw, the architecture genuinely
*decomposes* useful structure.

### 16. Token-level naturalness benchmark

For a corpus of natural text, measure each token's likelihood under
the unperturbed LM vs under a Manifold-SAE-reconstructed forward.
If the architecture's reconstruction is faithful, likelihoods match.
(This is patched-residual fidelity, not just MSE.)

### 17. Compositionality test

Train SAE. Find atoms encoding orthogonal concepts (e.g. "color" via
holdout test + "size" via holdout test). Present prompts like "big
red apple", "tiny green grape" — do both atoms fire as predicted by
their single-concept baselines? Test linear superposition of concepts.

### 18. Information-theoretic atom quality

For each atom, compute mutual information between its firing pattern
and a known concept (via histogram or KDE). MI captures non-monotone
relationships that Spearman misses. Predicts atoms that encode
relational/categorical structure rather than ordinal.

## Self-supervised discovery tests

### 19. Atom-pair co-firing structure

Compute pairwise co-firing matrix C_{ij} across the corpus. Spectral
embed C; clusters of correlated atoms reveal compositional structure
("atoms-of-atoms"). Compare to vanilla SAE — does Manifold-SAE
co-firing structure tell us more about LM internals?

### 20. Atom-level information bottleneck

How much can we compress an LM's intermediate activations through the
SAE's atom-firing bottleneck before downstream task accuracy drops?
Vs. how much we can compress through raw PCA at matched dimensionality.
Tests structural superiority of the decomposition.

## Implementation status

| # | Idea | Status |
| --- | --- | --- |
| 1 | Counterfactual ablation | not implemented |
| 2 | Cross-SAE alignment | not implemented |
| 3 | Atom-as-direction probe | partial (in atom_analysis.py adv_max) |
| 4 | Recursive SAE | not implemented |
| 5 | Atom mortality curve | not implemented |
| 6 | Adversarial perturbation | not implemented |
| 7 | Polysemy-induced atom split | not implemented |
| 8 | Topology benchmark | partial (synthetic_2d_recovery.py — only plane) |
| 14 | AxBench atom steering | partial (`steering_causality.py`) |
| 15 | Downstream task probe | partial (`atom_analysis.py` probe_classification) |
| 17 | Compositionality | not implemented |
| 18 | Information-theoretic MI | not implemented |
| 19 | Atom co-firing structure | not implemented |
| 20 | Information bottleneck | not implemented |
