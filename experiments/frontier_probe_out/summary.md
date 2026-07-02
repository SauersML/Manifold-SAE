# Curved-feature probes: weekday / month / year

- source: **WS-D Qwen3-32B residual harvest (frontier scale)**  |  model: `/models/Qwen3-32B`
- curved atom: gamfit manifold SAE (torch backend), K=1, intrinsic_rank=1, periodic circle+fourier atom (years use the same circle atom on a monotone arc; the torch backend has no open-interval manifold at rank 1)

## Matched-budget held-out EV (leave-one-template-out CV)

Curved uses **1** intrinsic coordinate; linear-L1 uses 1 PC, linear-L2 uses 2 PCs. A circle is intrinsically 1-D but needs 2 linear dims — so curved(1) should match linear(2) and beat linear(1).

| set | layer | tokens | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |
|---|---:|---:|---:|---:|---:|
| weekday | 32 | 7 | 0.053 | 0.062 | 0.254 |
| month | 40 | 12 | 0.049 | 0.076 | 0.316 |
| year | 40 | 15 | 0.151 | 0.212 | 0.448 |
| color | 24 | 8 | nan | nan | nan |

In-sample EV (full fit, no held-out — the cleaner 'can a 1-coord curved atom *represent* this set' view; CV above is noisy at these tiny sample counts):

| set | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |
|---|---:|---:|---:|
| weekday | 0.288 | 0.407 | 0.527 |
| month | 0.192 | 0.125 | 0.240 |
| year | 0.118 | 0.179 | 0.333 |
| color | 0.289 | 0.536 | 0.627 |

## Ordering accuracy (does the recovered chart order the tokens?)

| set | curved metric | curved value | linear best single-PC (Spearman) | linear 2D-PCA-angle |
|---|---|---:|---:|---:|
| weekday | circular | adj=0.286 / circ_r=0.643 | 0.929 | 0.571 |
| month | circular | adj=0.417 / circ_r=0.481 | 0.846 | 0.833 |
| year | monotone | spearman=0.257 / tau=0.192 | 0.950 | n/a |
| color | circular | adj=1.000 / circ_r=0.000 | 0.000 | 1.000 |

## REML corroboration (gamfit.sae_manifold_fit)

| set | REML EV (in-sample) | note |
|---|---:|---|
| weekday | 0.525 | n_iter=60 |
| month | 0.243 | n_iter=60 |
| year | 0.335 | n_iter=60 |
| color | 0.627 | n_iter=60 |

