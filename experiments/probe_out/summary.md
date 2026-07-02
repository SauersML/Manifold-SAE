# Curved-feature probes: weekday / month / year

- source: **REAL residual-stream harvest**  |  model: `Qwen/Qwen2.5-0.5B`
- curved atom: gamfit manifold SAE (torch backend `gamfit.torch.ManifoldSAE`), K=1, intrinsic_rank=1, periodic circle+fourier atom (n_basis=4)
- preprocessing: per-template demean (isolate the token feature from sentence context) then train-only PCA-whiten
- NOTE: `year` set is absent here — its real harvest was OOM-blocked on this box (see NOTES.md); the non-periodic branch is validated in `synthetic_validation.json`

## Matched-budget held-out EV (leave-one-template-out CV)

Curved uses **1** intrinsic coordinate; linear-L1 uses 1 PC, linear-L2 uses 2 PCs. A circle is intrinsically 1-D but needs 2 linear dims — so curved(1) should match linear(2) and beat linear(1).

| set | layer | tokens | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |
|---|---:|---:|---:|---:|---:|
| weekday | 14 | 7 | 0.335 | 0.490 | 0.541 |
| month | 8 | 12 | 0.487 | 0.281 | 0.519 |

In-sample EV (full fit, no held-out — the cleaner 'can a 1-coord curved atom *represent* this set' view; CV above is noisy at these tiny sample counts):

| set | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |
|---|---:|---:|---:|
| weekday | 0.584 | 0.444 | 0.580 |
| month | 0.598 | 0.316 | 0.586 |

## Ordering accuracy (does the recovered chart order the tokens?)

| set | curved metric | curved value | linear best single-PC (Spearman) | linear 2D-PCA-angle |
|---|---|---:|---:|---:|
| weekday | circular | adj=0.714 / circ_r=0.545 | 0.821 | 0.571 |
| month | circular | adj=0.833 / circ_r=0.597 | 0.818 | 1.000 |

