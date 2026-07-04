# FLOP accounting for the compute-matched frontier

The frontier plots price the curved (manifold) dictionary against the linear/block
TopK dictionary at **matched compute**. "Compute" is counted as **multiply-accumulates
(MACs)**, analytically, from the *realized* fit — not wall-clock, which is thread- and
hardware-dependent. Wall-clock (`fit_seconds`) is recorded per fit as a cross-check;
a large wall/MAC divergence is flagged in the writeup, not used as the frontier axis.

This is deliberately the accounting a skeptical reviewer would demand: it is
reproducible from the JSON alone, it counts *both* training and inference, and it
prices each atom at the width the fit actually kept (a chart the optimizer collapsed
to linear is billed as linear).

## Symbols

| symbol | meaning |
|---|---|
| `p` | activation dimension (ambient) |
| `K` | dictionary width (number of atoms) |
| `s` | routing sparsity requested per token (target L0) |
| `mean_l0` | **measured** mean number of atoms that actually fire per held-out token |
| `dv_k` | **realized** decoder-vector count of atom `k`, read from the fit (`fit.decoder_blocks`). A block/TopK direction atom = 1; a circle chart at `d_atom=1` = 3 (`{1, cos t, sin t}`); the same-lane affine `"linear"` atom = 2 (`{1, t}`). We never assume these — they are read back, so a chart the optimizer collapsed is billed at its collapsed width. |
| `N` | training rows |
| `passes` | training passes over N: block-lane `epochs`, manifold-lane outer-REML `n_iter` |

## Inference MACs / token

```
encode = Σ_k dv_k · p                     # score every atom against the row
       + [curved] Σ_k (dv_k³ + dv_k · p)  # per-scored-atom intrinsic-coordinate solve
decode = mean_l0 · mean_k(dv_k) · p       # sum the active atoms' curves
infer  = encode + decode
```

The encode term dominates at large `K` (it touches all `K` atoms), so the curved lane's
per-token inference cost is ≈ `dv̄` × the linear lane's at equal `K` — the chart's
constant-factor tax. The whole point of the frontier is to ask whether the chart still
wins *after* paying it: we plot EV and bits against `infer` (and against training MACs),
so "matched compute" is read off at equal x.

## Training MACs (total)

```
train = passes · N · 3 · infer            # forward + ~2× backward (standard AD factor)
```

`passes` is taken from the fit itself (block-lane epochs actually run; manifold-lane
`n_iter`). This *undercounts* the manifold lane's true optimizer work: each REML outer
iteration also does a Laplace/Hessian evidence solve whose cost is not a simple `N·infer`
matmul. We do **not** hide this — the training-MAC axis is therefore a **lower bound**
on the manifold lane's training cost, i.e. it is *charitable to us*. The claim we defend
("curved never loses EV at matched compute") is only strengthened if the true manifold
training cost is higher than plotted and it *still* matches or beats linear on EV; the
claim we would need to be careful about is the reverse, and it is not the one we assert.
Inference MACs (the deployment-relevant number) are exact.

## Parameters (dictionary storage)

```
decoder_params = Σ_k dv_k · p             # circle chart: 3p/atom; block/TopK: p/atom
```

Feeds `L_dict = decoder_params · param_bits / N` in the bits-per-token frontier
(`param_bits` = 16 for bf16 by default). This is the one-time cost the chart pays *extra*
(3p vs p per atom); the code-bit win has to beat it, and whether it does is set by firing
frequency — hence the heavy-tailed-firing axis.

## Why the bits can move opposite to the EV

At **matched distortion** δ² (the smallest held-out MSE any lane reaches; all lanes must
be able to hit it), the description length is

```
bits/token = mean_l0 · R(δ²)                 # code the active coordinates to the floor
           + selection_bits                  # which atoms fired
           + decoder_params · param_bits / N # amortized dictionary
```

with `R(δ²) = ½ log₂(1 + signal_var/δ²)`. On **curved** data a rank-1 atom cannot trace
a circle, so the linear lane's *measured* `mean_l0` inflates (it spends ≈2 atoms per
circle) — that inflation is the code-bit gap. `selection_bits` is reported in **two
currencies**: the combinatorial bound `log₂ C(K, mean_l0)` (always available) and the
empirical **support entropy** `H(S)` of each lane's own realized active-set distribution
(the currency the MDL lane is migrating to). We publish both so the number is robust to
that migration.

On a **pure-linear DGP** there is no curvature to find: `mean_l0` does not inflate for
the linear lane, the chart's extra `dv_k=2` decoder is wasted, and the chart can only
*lose* the selection/dictionary overhead. That row is the honest cost line.
