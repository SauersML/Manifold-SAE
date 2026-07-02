# Matched-null battery (`experiments/matched_null.py`)

A reusable, BSF-style **matched-null** validator for cyclic / curve-manifold
claims. Goodfire's BSF paper validates its curve-manifold Fourier claims with a
*matched null* — a synthetic control that preserves some structure of the data
while destroying the specific structure the claim rests on. If the observed
statistic survives the null (small empirical p), the structure is real; if the
null reproduces it, the "finding" is an artifact of the reading, the basis, or
the raw spectrum.

This module retrofits that discipline onto the **W7 circle-probe claims**
(`experiments/curved_feature_probes.py`), which previously shipped with no null
baseline, and packages it so the **G-bsf** (BSF baseline) and **N-nursery**
lanes can run the same battery on their cyclic-block findings.

## The claims under test (W7)

- **C1 — EV parity:** a *curved* atom reconstructs weekday/month residual
  activations from **one** intrinsic coordinate about as well as a *linear*
  **2-PC** reconstruction, and beats a linear **1-PC** reconstruction
  (`curved ≈ linear-L2 ≫ linear-L1`). "One curved coordinate does the work of
  two linear PCs."
- **C2 — cyclic order:** the single recovered angular coordinate orders the
  tokens correctly around the circle (weekday cyclic-adjacency 0.71, month 0.83;
  month 2D-PCA angle 1.00).

## The four nulls (each preserves some structure, destroys the claimed one)

| null | what it preserves | what it destroys | tests |
|---|---|---|---|
| **rotation** | the r-dim subspace + its spectrum | the PCA axes (random orthonormal basis of the *same* subspace, re-fit) | is the coordinate/ordering **basis-real**, or an artifact of reading PC1/PC2 as the circle plane? (C1+C2) |
| **label-permutation** | the recovered angles | the token→label correspondence | does the recovered angle order tokens **better than a random labelling**? empirical p over ≥1000 permutations (C2) |
| **matched-spectrum Gaussian** | the per-PC eigenspectrum | all cyclic structure (isotropic Gaussian per PC) | is "one curved coord reaches 2-PC parity" a **circle signature**, or what any 1-D curve gets on a matched spectrum? (C1) |
| **phase-scramble** | the power spectrum over the cycle | the Fourier phases (per-column) | does the circle rest on **higher-harmonic phase-locking**? |

### Statistics & verdict

- **C1 (matched-spectrum).** Primary statistic is the **fraction of the 2nd-PC
  gap that curved(1) closes**, `(curved − lin1)/(lin2 − lin1)`. This is
  *fit-quality-normalised*: on a true 1-D-in-2-D circle both PCs *are* the atom's
  plane, so curved closes ≈100% of the gap; on a genuine 2-D Gaussian blob a 1-D
  curve closes only a fraction. PASS ⇔ observed gap-closed exceeds the
  matched-spectrum null at p<0.05.
- **C2 (label-permutation).** Empirical p that the observed cyclic-adjacency
  exceeds a random recovered ordering (the correct null: `cyclic_adjacency_
  accuracy` is invariant to relabelling *both* sides, so we randomise the
  recovered ordering, not the labels). PASS ⇔ p<0.05. Corroborated by the
  **rotation** null's *basis-real fraction* (share of random rotations whose
  ordering still beats the label-perm 95th-percentile chance ceiling).
- **phase-scramble caveat (honest).** A *pure* circle is entirely
  fundamental-mode power, and any sum of frequency-1 components is still an
  ellipse — so per-column phase randomisation of a single-harmonic signal leaves
  the circle intact. Phase-scramble is therefore **only discriminative for claims
  that rest on higher-harmonic phase-locking** (BSF curve manifolds, N-nursery
  blocks). The battery reports the **fundamental-mode fraction (FMF)**; when
  FMF ≳ 0.9 it marks the null *non-applicable* rather than emitting a spurious
  FAIL. This is verified on the planted-circle sanity check (FMF ≈ 1.00, null
  adjacency ≈ observed by construction).

## Reusable API

```python
from matched_null import null_battery

out = null_battery(
    X,                 # (N, D) analysis-ready activations for ONE layer
                       #        (already per-frame demeaned if that is your recipe)
    labels,            # (N,) int cyclic rank of each sample in [0, n_labels)
    n_labels=7,        # number of distinct cyclic tokens
    cyclic=True,       # circles (weekday/month/color); False = ordered curve (year)
    fitted_basis=None, # optional (D, r) reading basis; else train-PCA to reduce_dim
    claims=("rotation", "label_perm", "matched_spectrum", "phase_scramble"),
    n_rot=48, n_gauss=128, n_perm=5000, n_phase=5000, seed=0,
)
# out["observed"], out["nulls"][<name>] (obs stat + p + compact null histogram),
# out["verdict"]  -> per-claim pass/fail + p-values
```

Every curved fit reuses the **torch-backend** `ManifoldSAE` recipe from
`curved_feature_probes.curved_fit` (K=1, intrinsic_rank=1, circle+fourier atom) —
**not** the REML `sae_manifold_fit`, which is OOM/segfault-prone here. The CLI
still isolates each set in a retried subprocess with incremental JSON saves.

## CLI

```bash
python matched_null.py --set weekday    # one cached W7 set -> null_out/null_weekday.json + .png
python matched_null.py --set month
python matched_null.py                  # orchestrate both, retried subprocesses
python matched_null.py --synthetic      # planted-circle sanity check (C2 should PASS)
```

Env: `MATCHED_NULL_STEPS` (fit steps, default 600), `MATCHED_NULL_SEEDS`
(default 1 — single seed so the null distribution carries the fit's own seed
noise; the observed-in-battery statistic uses the same budget so every p is
internally apples-to-apples), `MATCHED_NULL_RETRIES`, `MATCHED_NULL_OUT`.

## Outputs

- `null_<set>.json` — observed statistics + per-null distribution (compact
  histogram) + p-values + pass/fail verdict.
- `null_<set>.png` — one panel per null: the null distribution with the observed
  statistic marked and its p-value.
- `RESULTS.md` — the pass/fail summary table across sets (generated by the
  driver / written up by the P-null lane).

## For the G-bsf and N-nursery lanes

Call `null_battery(X, labels, n_labels=..., cyclic=True)` directly on your
cyclic-block token means / activations. If your claim involves **higher
harmonics** (BSF curve manifolds), the phase-scramble null becomes discriminative
— check `out["nulls"]["phase_scramble"]["applicable"]` and its p-value. For a
custom reading basis (a fitted manifold chart rather than PCA), pass it as
`fitted_basis=(D, r)`.
