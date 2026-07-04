# Results suite — 2026-07-03 (Qwen3-8B / Qwen3.6-35B, manifold-SAE program)

Everything needed to analyze or hand to another agent. Each subdirectory is one
experiment with its raw data, logs, and (where finished) figures. Repro details for
the flagship are in `crown_8b/README.md`; all fits use gamfit (SauersML/gam) with
SHAs noted per experiment.

## Index & one-line verdicts

| dir | experiment | verdict |
|---|---|---|
| `crown_8b/` | dose-calibration crown, Qwen3-8B L18 weekday circle | **PASS, headline**: held-out slope 0.945, R²=0.999, median ratio 1.10; 97% of edits within 2× of forecast. Full reproducible bundle (figures, 504-row raw data, harvest npz, driver code, analyses incl. honest negatives). |
| `crown_35b/` | same recipe, Qwen3.6-35B-A3B L17, bf16 2×A40 | **FAIL v1 (honest miss)**: held-out slope 0.559, R²=0.69, median ratio 5.8×. n=42 harvest rows (fewer than 8B's 70), bf16 numerics in the Fisher harvest, and L17-by-depth-analogy are the suspect list. Retry job (12486012) running. Raw JSON + PNG + full log included. |
| `safety/` | Track S: sycophancy & refusal charts + sycophancy dose loop | **Sycophancy = graded open-arc intensity dial** (circle r²=0.693, Spearman(coord, designed grade)=0.910, partial 294° arc, no wraparound); **refusal = switch-like** (circle boosts r² 0.83→0.87 but collapses ordering 0.63→0.32 — honest negative). **Dose loop (n=336): slope 0.934, median ratio 1.01, bite rate 1.0, R²=0.59** — absolutely calibrated on average, noisier per-edit than the calendar crown (fit-quality caveat r²=0.69 carried). Behavioral effect real: ±agreement doses shift the agreement pole by +0.24 / −0.13. |
| `scale_evidence/` | gam-side scale runs on real 35B L17 activations | **Block lane at K=32,000 explained variance 0.9895** with 0 dead blocks, orthonormality dev ≤4e-8, 55 min (t1_frame_health.json — rows for K=4k EV 0.707, K=16k EV 0.906, K=32k EV 0.990). The old K≥64 co-collapse ceiling (EV 0.199) is dead. qwen_kscale.log: dense-width joint fit at p=2048 hits the #1995 width-wall grind (K=2 timeout) while PCA-128 arm is healthy — corroborates the grind class. real_l17_ab.log: 150k×2048 stagewise rank-charge A/B in flight. |
| `nulls/` | hallucinated-structure controls (matched Gaussian + shuffled real) | IN FLIGHT (jobs 12484519/20, ~1h in at snapshot). PASS criterion: ~0 accepted curved atoms on noise, Θ mass at 0. Verdict JSONs will be added when scored. |

## The one-paragraph story

An unsupervised curved atom fitted to a real LLM's residual stream carries a
behavioral metric good enough to *forecast* interventions: on Qwen3-8B, "move this
token N days along the weekday circle" comes with a predicted output shift in nats
that the model then matches to ~10% (median) across four orders of magnitude — while
the field-standard metric-free steering is 6× miscalibrated. The same instrument
pointed at sycophancy finds a graded intensity dial (and honestly reports that
refusal is a switch, not a dial), and its dose loop is calibrated in scale on the
first try. The first 35B attempt failed calibration — recorded as such. Meanwhile the
block-sparse dictionary lane scales to K=32k on real 35B activations at EV 0.99 with
zero dead atoms, which is the substrate the full Atlas run composes on. Controls
(noise nulls) and the composed run are in flight.

## Provenance

- Models: Qwen3-8B (`$ROOT/models/qwen3-8b`), Qwen3.6-35B-A3B (bf16, 2×A40).
- gamfit: 0.1.247 (SHA 67735d1f4) for crown_8b; 0.1.248 (6f89ebf84) for scale runs.
- Cluster: MSI, partitions msismall (CPU) + preempt-gpu (A40/L40S); $ROOT =
  /projects/standard/hsiehph/sauer354.
- All prompts/templates are generated in code (see crown_8b/code/ and
  safety/safety_probes_DESIGN.md) — no external datasets.
- Repos: SauersML/gam (engine), SauersML/Manifold-SAE (this bundle).
