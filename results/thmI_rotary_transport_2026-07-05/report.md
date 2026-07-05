# Theorem I — rotary transport rigidity of the weekday circle

**Prediction under test (Superposed-Geometry P2 / Theorem I).** For a weekday
feature that charts as an elliptical (circle) atom `g_l(θ)=c_l+A_l e(θ)`,
`e(θ)=(cosθ,sinθ)`, ANY *linear* transport `W` carrying `im g_l` onto `im g_{l+1}`
is FORCED to induce a coordinate map that is a pure phase shift / reflection
`h(θ)=±θ+φ`. Rotary structure is a *theorem*, not a design choice. Mechanism:
enforcing `‖e(h)‖²≡1` forces the pulled-back operator `M = A'⁺ W A` into `O(2)`
(conformal, `MᵀM=λI`); deviations from `±θ+φ` should concentrate where the atom's
harmonic spectrum departs from a pure ellipse.

**Verdict: Theorem I CONFIRMED.** Across every consecutive hop L11→L23 of
Qwen3-8B, real cross-layer transport of the weekday circle is a phase shift to
within a few degrees, the induced 2×2 operator `M` is conformal to ~2%, and a
shuffled-day null destroys the structure. The proof's *mechanism* is directly
observed: the per-hop deviation from `±θ+φ` correlates at **r = 0.83** with the
conformal departure of `M` — `h` leaves a phase shift exactly, and only, insofar
as `M` leaves `O(2)`. The P3 "deviations track harmonic impurity" refinement is
NOT confirmed — untestable here (below).

---

## Data & method

- **Acts.** DOSE weekday battery, dense harvest: 30 templates × 7 weekdays = 210
  last-token residuals at each of L11…L23 (d=4096), Qwen3-8B, per-template
  demeaned. MSI cache `weekday_acts_8b_L11to23_dense.npz`.
- **Circle certificate (gamfit).** `sae_manifold_fit(K=1, atom_topology="circle",
  rank-8 PCA, isometry_weight=0)` per layer: reconstruction r² 0.52–0.61.
- **Circle coordinate.** Deterministic `θ_l = atan2` of demeaned data in layer
  `l`'s top-2 SVD plane (the elliptical/fundamental coordinate; see reproducibility note).
- **Transport verdict (gamfit).** `layer_transport_fit(θ_l, θ_{l+1})`: winding
  `degree`, `degree_concentration`, `isometry_defect` (departure of `h` from an
  isometry, i.e. from `±θ+φ`) + SE.
- **Induced operator (numpy).** `M` from `e(θ_b) ≈ M e(θ_a)` least squares = the
  pullback `A'⁺ W A`; conformal departure `‖MᵀM−λI‖/λ`, anisotropy `(σ₁−σ₂)/σ₁`.
- **Rigid cross-check (numpy).** Best `±θ+φ` circular-RMS residual (corroborates gamfit).
- **Null.** Shuffle the token↔token correspondence, 50 permutations/hop.
- Compute: gamfit 0.1.248 on MSI (`venv_xport`), amdsmall dedicated cores.

## Results — the L11→L23 ladder

| hop | deg | conc | rigid ±θ+φ resid (deg) | gamfit iso-defect | conformal dep. M | anisotropy M | null conc |
|---|---|---|---|---|---|---|---|
| L11→L12 | 1 | 1.00 | 5.1 | 0.036 | 0.106 | 0.07 | 0.19 |
| L12→L13 | 1 | 1.00 | 1.7 | 0.004 | 0.014 | 0.01 | 0.19 |
| L13→L14 | 1 | 1.00 | 1.7 | 0.002 | 0.023 | 0.02 | 0.20 |
| L14→L15 | 1 | 1.00 | 1.3 | 0.001 | 0.009 | 0.01 | 0.19 |
| L15→L16 | 1 | 1.00 | 0.8 | 0.000 | 0.002 | 0.00 | 0.19 |
| L16→L17 | 1 | 1.00 | 1.9 | 0.003 | 0.040 | 0.03 | 0.18 |
| L17→L18 | 1 | 1.00 | 1.2 | 0.001 | 0.015 | 0.01 | 0.18 |
| L18→L19 | 1 | 1.00 | 3.6 | 0.005 | 0.056 | 0.04 | 0.14 |
| L19→L20 | 1 | 1.00 | 3.3 | 0.016 | 0.024 | 0.02 | 0.18 |
| L20→L21 | 1 | 1.00 | 1.8 | 0.003 | 0.053 | 0.04 | 0.18 |
| L21→L22 | 1 | 1.00 | 1.8 | 0.003 | 0.045 | 0.03 | 0.18 |
| L22→L23 | 1 | 1.00 | 2.4 | 0.001 | 0.054 | 0.04 | 0.18 |

**Medians:** degree-concentration 0.9995, rigid residual 1.8° (max 5.1°), gamfit
isometry defect 0.003, conformal departure 0.032 (max 0.106), anisotropy 0.022,
real vs null concentration gap 0.82.

### 1. Transport IS a phase shift (Theorem I headline)
Every hop is winding-degree **1**, concentration **≥ 0.998** (same day, same
phase). Best `±θ+φ` fit residual **1.8° median, 5.1° worst**; gamfit isometry
defect **≈ 0** everywhere. `h` is a pure phase shift throughout L11–L23.

### 2. Mechanism: `M` is conformal, and that is *why* `h` is a phase shift
The free 2×2 `M = A'⁺ W A` comes out conformal: anisotropy median **2.2%**
(max 7%), `‖MᵀM−λI‖/λ` median **0.032**. A free 4-parameter map turning out
orthogonal-conformal is not automatic. Decisively, per-hop rigid-residual vs
conformal-departure of `M` **correlate at r = 0.83** (fig a): `h` departs from
`±θ+φ` precisely and only to the extent `M` departs from `O(2)` — the proof's
mechanism observed on the real model, tying an independent gamfit statistic to a
numpy linear-algebra quantity.

### 3. P3 harmonic-impurity refinement — NOT confirmed (honest)
Atoms are not pure ellipses: harmonic impurity (energy k≥2 / fundamental) is
**~1.1–1.3 at every layer**. Yet the phase-shift deviation is uniformly near
zero. With no spread in deviation (all < 5°, at the noise floor) and near-constant
impurity, P3's concentration claim is **not differentiable in this data**:
r = −0.26 (fig b), i.e. noise. The deviations track conformal-departure of `M`
(r = 0.83), not impurity. Rigidity holds *despite* substantial non-ellipticity —
the fundamental part transports rigidly regardless of harmonics riding on top.
P3 is neither confirmed nor cleanly refuted; it has no lever in the weekday circle.

### 4. Null control — PASSED
Breaking the day↔day correspondence collapses concentration from ≥0.998 to
**0.18 ± 0.02** (median, n=50/hop): the surrogate shows no phase-shift structure
(fig d). Rigidity is a property of the genuine token correspondence, not of "both
layers host a circle."

## Caveats

- **Reproducibility note.** gamfit's honest arc coordinate `u(θ)` comes from a
  non-convergent outer BFGS and is not reproducible: the same L17→L18 hop gave
  isometry defect 0.024 one-off vs 2.02 in a batch re-fit, and using it produced
  a spuriously messy ladder (degree flips, defects up to 97). The verdict uses the
  deterministic top-2 SVD-plane angle — exactly the elliptical coordinate the
  theorem is stated over; gamfit still supplies the circle certificate and the
  transport verdict.
- **Fundamental-plane test surface.** The top-2 plane captures ~35–48% of demeaned
  variance (circle r² in rank-8 is 0.52–0.61); the rest is harmonic/off-plane.
  Measuring `h` on the fundamental plane is correct for a theorem about the
  *elliptical* atom. The non-trivial content: the 7 days keep identical relative
  angular spacing across layers (isometry defect ≈ 0) and the free `M` is
  conformal — neither forced by the projection, both fail under the shuffle null.

## Relation to prior XPORT work
The earlier chart-transport study (`../suite_2026-07-03/transport/`) called the
circle "carried in topology but continuously re-encoded in geometry" (ambient
plane tilts 15–43°/hop) using the ambient parallel-transport gauge and the
stochastic arc coordinate on 70 rows. Not in tension: that measured the *ambient
2-plane's* motion; here, on the deterministic in-plane angle with 210 rows, the
*coordinate map itself* is a clean phase shift. A tilting-but-conformal frame is
exactly a rotary re-encoding — what Theorem I predicts.

## Files
- `thmI_v3.py` — verdict driver (deterministic angle + gamfit `layer_transport_fit` + induced `M` + null).
- `thmI_v3_results.json` — full per-hop numbers.
- `thmI_figure.png` — (a) mechanism r=0.83, (b) P3 null, (c) ladder, (d) null control.
- `thmI_rotary_transport.py`, `thmI_v2.py` — v1 (arc-coord, superseded) / v2 (adds gamfit certificate + stability diagnostic).
- `REPLICA.md` — seconds-scale single-pair checkpoint.
