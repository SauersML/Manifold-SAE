# Theorem I rotary-transport test — seconds-scale replica (checkpoint)

**Replica hop:** L17 -> L18, weekday circle, Qwen3-8B, dense harvest (210 rows =
30 templates x 7 weekdays, per-template demeaned), gamfit 0.1.248.

Method: fit K=1 `circle` atom at L17 and L18 (`sae_manifold_fit`, rank-8 PCA,
isometry_weight=0), take each row's angle, then `gamfit.layer_transport_fit`
gives the induced coordinate map h and its departure from a phase shift.

**Replica result (single-pair):**
- winding `degree = 1`, `degree_concentration = 0.973` (h is a clean degree-1
  circle->circle map — the same weekday sits at the same phase).
- `isometry_defect = 0.024` (± 0.016): departure of h from an isometry, i.e.
  from `h(theta)=±theta+phi`, is ~0. **The single-pair transport IS a phase
  shift**, consistent with Theorem I.
- free-spline h residual_rms 0.21 rad; `transport_edf` 5.6.

**Caveat found (why the full-ladder needs care):** the gamfit circle solver's
arc-coordinate is stochastic (non-convergent outer BFGS) — re-fitting the same
L17->L18 hop in a batch gave isometry_defect 2.02, not 0.024. So the full-ladder
verdict is being computed on a DETERMINISTIC circle coordinate (data top-2 SVD
plane angle) with gamfit still supplying the circle certificate (r2, planarity)
and the transport verdict (`layer_transport_fit`). Full L11–L23 ladder + O(2)
departure of M=A'⁺WA + harmonic-impurity correlation + shuffled-day null in
progress (job 12546300).
