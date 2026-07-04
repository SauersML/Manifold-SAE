# Spec-specificity empty-splice floor: what was bugged, what it is now

**Symptom (v1 report):** the empty-edit "noise floor" printed as `nan` in the report
text and `1e-30` in the table, and every band showed **100% "detectable collateral"** —
obviously wrong.

**Root cause (analysis-side, not the data).** The empty-splice control patches a *zero*
delta through the identical splice hook (`stack`/`reshape` + `+0`). That operation is
**bit-exact**: the patched forward reproduces the clean logits bitwise, so the control
returns KL = **exactly 0.0** on all 19,800 cells (verified directly in the saved JSON:
`off_floor`, `on_floor`, `bleed_floor` are all `0.0`). That is the control *passing* — it
proves the hook is faithful. The bug was purely downstream handling of a zero floor:

- report median filtered `off_floor > 0` → empty list → `np.median([])` = **nan**;
- table geomean did `exp(mean(log(clip(floor, 1e-30, None))))` → **1e-30**;
- `frac_detectable = mean(off > 2*floor)` = `mean(off > 0)` → **~100%**.

**Fix (`code/spec_specificity.py`, `_band_summary` + `make_report`).** Report the floor
honestly as bit-exact 0 (a hook-faithfulness result), and flag "perceptible collateral"
against an explicit absolute threshold `DETECT_EPS = 1e-4` nats instead of against the
(zero) floor. No GPU rerun was needed — the raw per-edit rows were already correct; only
the summary/plot were regenerated from the existing `spec_specificity.json`.

**Corrected headline.** With the floor fixed, the surgical figure of merit is the
specificity ratio (off-target / on-target KL) at matched on-target effect:

| on-target band | manifold chart | linear latent | random dir |
|---|---:|---:|---:|
| ~0.01 | **0.0051** | 0.0203 | 0.0125 |
| ~0.1  | **0.0030** | 0.0189 | 0.0040 |
| ~0.3  | **0.0034** | —      | 0.0081 |

The manifold chart is the most surgical: ~4x lower off/on ratio than the matched-norm
linear latent across bands, and it moves the fewest unrelated tasks above the 1e-4-nat
perceptibility threshold. This is a reconstruction-independent, causal specificity claim.

The full raw JSON (rows_B/rows_A, 19,800/15,840 rows) stays on MSI at
`$ROOT/spec_out/spec_specificity.json`; the copy here is trimmed to config + consistency
+ bands + floor audit.
