# Manifold-SAE paper

NeurIPS-style arxiv draft.

## Abstract

Sparse autoencoders (SAEs) decompose a language model's residual stream into a
dictionary of point-like "features." On continuous concept families --- hue,
magnitude, sentiment polarity, time-of-day --- this point-atom inductive bias
is wrong: a single curved concept is *shattered* into many redundant atoms.
We introduce **Manifold-SAE**, a dictionary whose atoms are closed curves on
S^1: each atom i produces a per-token angle theta_i via an input-linear
projection, then reconstructs a Fourier curve D_i * phi(theta_i) in
residual-stream space, dispatched through an IBP-Gumbel concrete gate with
ARD on the per-atom decoder norm. On Deep Cogito 671B (L40) color-token
activations (26572 prompts x 7168 dims, 949 xkcd colors x 28 templates),
Manifold-SAE reaches **val R^2 = 0.913** versus 0.882 (L1) and 0.874 (TopK)
at matched dictionary size F=512, with **zero dead atoms**. Atoms recover
continuous color arcs (e.g. *night blue -> deep sky blue -> light sky blue*)
rather than shattered point features. We further introduce a *gauge-fix
recipe*: supervising d_aux=3 axes with HSV and leaving d_free=3 free
recovers name-semantic structure (modifier count, monoword) unsupervisedly
on the free block.

## Build

```
make            # build PDF (requires pdflatex / latexmk)
make clean
```

If pdflatex is not installed, the .tex source is self-contained; any LaTeX
distribution (MacTeX, TeXLive, MikTeX) can build it.

## Files

- `manifold_sae.tex`  -- single-file paper source
- `refs.bib`           -- 23 BibTeX entries
- `figures/comparison_4panel.png` -- main result figure
- `figures/training_curves.{png,pdf}` -- per-method training-loss figure (fig 2)
- `Makefile`
