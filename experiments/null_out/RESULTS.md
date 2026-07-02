# Matched-null battery — W7 circle-probe results

Retrofit of a BSF-style **matched-null** baseline onto the W7 circle-probe claims
(`experiments/curved_feature_probes.py`, `probe_out/`). Module + method:
`experiments/matched_null.py`, `null_out/README.md`. Every curved fit uses the
**exact W7 recipe** (torch `ManifoldSAE`, K=1, intrinsic_rank=1, circle+fourier,
best-of-2 seeds by train EV, steps=600), so the observed statistic reproduces the
**claimed** value. Fits ran on the cached real harvests (`probe_out/harvest_*.npz`,
`Qwen/Qwen2.5-0.5B`) through a checkpointed, OOM-reaper-resilient harness (the box
was memory-starved by a concurrent fleet — ~58 MB free — and SIGKILLed every long
torch process; draws are fsync-checkpointed per fit and resumed by index).

Nulls: **32** rotations, **80** matched-spectrum Gaussians, **5000** label
permutations, **5000** phase scrambles per set. Empirical p uses the +1
correction. `weekday`/`month` are the cached W7 harvests; `color` is a cheap
Qwen color-NAME harvest added here (see the scope caveat below — it is **not**
W7's big-model color-swatch claim). `year` was OOM-blocked in W7. Any set runs
via `null_battery(X, labels, n_labels=..., cyclic=True)`.

## Pass/fail per claim

| set | claim | statistic | observed | null (mean / p) | verdict |
|---|---|---|---:|---:|:--:|
| **month** | C1 EV parity (curved(1) does the work of 2 PCs) | gap-closed `(cev−l1)/(l2−l1)` vs matched-spectrum Gaussian | **1.043** | 0.970 / **p=0.012** | **PASS** |
| **month** | C2 cyclic order | cyclic-adjacency vs random labelling | **0.833** (10/12) | 0.182 / **p=0.0004** | **PASS** |
| **month** | C2 basis-real (corrob.) | adjacency-under-rotation ≥ chance95 | — | basis-real **1.00** | **PASS** |
| **weekday** | C1 EV parity | gap-closed vs matched-spectrum Gaussian | 1.024 | 0.952 / **p=0.16** | **FAIL** |
| **weekday** | C2 cyclic order | cyclic-adjacency vs random labelling | 0.714 (5/7) | 0.334 / **p=0.042** | **PASS (marginal)** |
| **weekday** | C2 basis-real (corrob.) | adjacency-under-rotation ≥ chance95 | — | basis-real **0.22** | **WEAK** |
| **color-name** | C2 cyclic order (hue) | cyclic-adjacency vs random labelling | 0.125 (1/8) | 0.286 / **p=0.93** | **FAIL** (no circle) |
| **color-name** | C2 basis-real (corrob.) | adjacency-under-rotation ≥ chance95 | — | basis-real **0.00** | **FAIL** |
| **color-name** | C1 EV parity | gap-closed vs matched-spectrum Gaussian | 0.999 | 0.650 / **p=0.11** | **n.s.** |

Supplementary phase-locking diagnostic (NOT a gate on C2 — see below):

| set | fundamental-mode fraction | phase-scramble p | reading |
|---|---:|---:|---|
| month | 0.62 | 0.29 (n.s.) | ordering carried by low-frequency **power spectrum** (smoothness); no higher-harmonic phase-locking required |
| weekday | 0.65 | 0.61 (n.s.) | same |

## Headline findings

**1. The month circle is real and survives every null.** One curved coordinate
reaches linear-2-PC EV parity *beyond* what a 1-D curve gets on a matched-spectrum
Gaussian (gap-closed 1.043 vs 0.970, **p=0.012**); the recovered angular order
matches the true 12-cycle far beyond chance (adjacency 10/12, **p=0.0004**); and it
is **fully basis-real** — every one of 32 random orthonormal rotations of the
reading subspace keeps the ordering above the chance ceiling (basis-real 1.00,
curved-EV rotation-CV 0.05). The month claims are solid.

**2. HEADLINE — the weekday EV-parity claim FAILS its matched-spectrum null.**
Weekday's "one curved coordinate ≈ two linear PCs" (gap-closed 1.024) is **not**
distinguishable from what a 1-D curve achieves on a Gaussian with the *same
eigenspectrum but no circle* (null 0.952, **p=0.16**; secondary curved-beats-lin1
p=0.62). For weekday, the EV parity is largely a **property of the spectrum, not of
circular structure** — it is not evidence for a curved feature over a linear one.

**3. HEADLINE — the weekday cyclic-order claim is marginal and basis-fragile.**
It clears the label-permutation null only at **p=0.042** (vs month's 0.0004), and
under random rotation of the reading basis the ordering collapses toward chance
(**basis-real 0.22**, rotation-mean adjacency 0.41 vs observed 0.71). So the
weekday "circle" depends substantially on reading it in the PC1/PC2 plane — a
basis artifact risk the rotation null is designed to expose. Weekday should be
reported with this caveat, not as a co-equal success with month.

**4. The circular ORDERING (both sets) is a smoothness / power-spectrum effect,
not higher-harmonic phase-locking.** The phase-scramble null (preserve the
per-cycle power spectrum, randomise phases) does not falsify the ordering for
either set (p=0.29, 0.61, n.s.), because the fundamental harmonic dominates
(FMF 0.62–0.65) and any fundamental-heavy signal orders cyclically regardless of
phase. This is **not** a failure of the W7 cyclic-order claim (that is a claim about
*order*, owned by the label-perm null, which month passes decisively). It is an
honest statement of *what kind* of structure it is: low-frequency smoothness, the
same for both. The sharp higher-harmonic form of this null is the right tool for
the **G-bsf** curve-manifold and **N-nursery** multi-harmonic claims, where FMF is
lower — for a single circle it is supplementary.

## Color (name-token analog — important scope caveat)

W7's color claim came from a **big-model color-SWATCH harvest** (`color_geometry.py`,
D=7168) that cannot run on this box. `null_color.json` here is a **cheap
Qwen2.5-0.5B color-NAME probe** — 8 hue-wheel words (red…magenta) × 5 templates,
the same harvest that gave weekday/month. It is a *related analog, not W7's color
claim*, and it does **not** refute W7's swatch result.

Finding: color-NAME tokens do **not** form a recovered hue circle. The recovered
angle orders the 8 colors no better than chance (adjacency 1/8, label-perm
**p=0.93**, i.e. below the chance mean 0.29), no random rotation keeps it above
chance (**basis-real 0.00**), and FMF is low (0.29 — no fundamental-mode circle).
The high curved EV (0.82, gap-closed 0.999) is the same low-rank-blob effect the
matched-spectrum null controls for (C1 p=0.11, n.s.). Interpretation: a small LM's
color-*word* embeddings do not carry a clean continuous hue cycle the way visual
swatches in a large model do — the cheap name-token probe is the wrong instrument
for the hue-circle claim, and says nothing against the swatch-based W7 result.

## Caveats (honest)

- **C1 is stringent by design and conservative.** On a *noiseless planted* circle
  the linear-2-PC reconstruction is near-perfect (the circle is exactly a 2-D
  plane), so a 1-basis fourier curve cannot beat it and the matched-spectrum null
  matches curved too (synthetic sanity: gap-closed 0.83, p=0.76 — a deliberate
  non-trivial bar). Month **passing** C1 (p=0.012) on *real* data is therefore a
  genuine, non-trivial signal, not a gimme.
- **Synthetic sanity** (`null_synthetic_weekday.*`): a planted 7-circle passes C2
  decisively (label-perm p=0.039, basis-real 1.00, FMF 1.00), confirming the C2
  machinery is correct.
- Sample sizes are tiny (N=35/60, 7/12 tokens); the recovered-angle ordering is
  seed-sensitive, so the battery matches W7's honest best-of-2-seed selection for
  the observed statistic. p-values on 32/80 refit draws are coarse (~±0.03) but the
  month/weekday separation is well outside that.

## Files

- `null_weekday.json`, `null_month.json` — observed stats + per-null distributions
  (compact histograms) + p-values + verdict.
- `null_weekday.png`, `null_month.png` — one panel per null, distribution + observed.
- `null_synthetic_weekday.{json,png}` — planted-circle sanity anchor.
- `parts/` — checkpoint jsonl/npz (resumable draws; safe to delete after assembly).
- `README.md` — method, reusable API, and how the G-bsf / N-nursery lanes call it.
