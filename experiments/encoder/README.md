# WS-E — amortized encoder + corpus sweep

Distill an **amortized encoder** from **certified exact** solves against a frozen
composed dictionary (T1 linear atoms + T2 curved atoms), then sweep a corpus for
per-token features. The exact solve is always the teacher and the fallback; the
encoder only proposes, and a certificate keeps it honest — no approximation
enters silently (issue #1010).

## The machinery this wraps (already in the tree — WS-E builds the harness, not the encoder)

Encoding a row `x` against the FROZEN dictionary is, per atom `k`, the
coordinate-only Newton problem `min_t ½‖x − z_k·B_kᵀΦ_k(t)‖² + prior_k(t)`. That
solve, its basin warm-up, wrapped distances, resident GPU kernel and per-row
exactness certificate live in Rust:

- `crates/gam-sae/src/encode.rs` — the **Kantorovich-certified encode atlas**.
  Offline: certified charts (`h = β·η·L ≤ ½` solved for a Newton radius per
  chart). Online: route to nearest charts, 1–2 Newton steps, per-row certificate
  at the start; uncertified rows are FLAGGED (`EncodeResult.encode_uncertified_count`)
  and routed to the exact multi-start solve.
- `crates/gam-sae/src/gpu_kernels/sae_encode_resident.rs` — the device-resident
  exact per-row certified encode (#988), with a byte-faithful CPU emulator/oracle.
- `crates/gam-sae/src/manifold/amortized_routing.rs` — chart-geometry routing
  logits from an amortized predictor; uncertified rows keep their existing route.
- `gamfit/_sae_manifold.py` — `ManifoldSAE.converged_latents(X)` is the **certified
  exact teacher** (the frozen-decoder OOS Newton solve, cold-started).
- `gamfit/distill.py` — `distill_encoder` (torch MLP from exact teacher solves),
  `DistilledEncoder.encode_fast` (the amortized forward pass, throughput path),
  `encode_with_fallback` (cold-gated rowwise fallback → `EncoderFallbackStats`).

The Python torch-MLP encoder is the amortized encoder distilled from the certified
exact teacher. The Rust `EncodeAtlas` is the certified fast path; it is not yet
wired to the Python FFI, so the Python honesty gate uses a **cold exact probe**
(the #1166 self-referential-gate trap avoided). Both are honest fallback measures;
when the atlas FFI lands, the certificate `h ≤ ½` becomes the cheap in-kernel gate.

## Files (this harness)

| file | role |
|---|---|
| `distill_harness.py` | core: teacher solve → distill → per-row gate → agreement / **certificate-fallback (overall + by token-freq decile)** / **throughput**; `EncoderReport` |
| `synth_dictionary.py` | synthetic composed dictionaries: planted circles (separated variances) + linear atoms; Zipf token-frequency metadata |
| `run_synthetic.py` | end-to-end on synthetic; `--scale local` (K=1 smoke) / `--scale node` (real scale) |
| `run_real.py` | end-to-end on a **real SAC dictionary** (`--dictionary`) + **real corpus** (`--manifest`, WS-D) |
| `heimdall_submit.py` | submit node2 jobs through Heimdall, wrapped to always exit 0 (rc/log to files) |

## Gates (SAC_PLAN Part 3, WS-E)

- **throughput ≥ 1e5 rows/s** of the amortized forward pass (`encode_fast`) on
  node hardware — reported in `EncoderReport.throughput_rows_per_s`.
- **certificate-fallback rate by token-frequency decile** — the fraction of rows
  whose amortized guess does not match the cold exact solve inside the calibrated
  gate, bucketed by corpus token frequency (needs WS-D token metadata; wired and
  ready, activates automatically when `token_freq` is present).

## Running

Local smoke (tiny K=1; the 8 GB laptop is memory-contended under the fleet, so
prefer node2):

    /Users/user/gam/.venv/bin/python run_synthetic.py --scale local

Node2 (real scale) — always through Heimdall, wrapped to exit 0:

    python heimdall_submit.py --name enc_synth_node --command \
      "cd /dev/shm/sauers_gpu/encoder; export RAYON_NUM_THREADS=32 OMP_NUM_THREADS=32; \
       nice -19 ionice -c3 /models/sauers_build/venv_fable/bin/python -u run_synthetic.py \
       --scale node --out /dev/shm/sauers_gpu/encoder/synth_node.json"

Real dictionary + corpus (once WS-A/WS-D land):

    ... run_real.py --dictionary <sac_artifact.json> --manifest <MANIFEST.json> \
        --layer 0 --out /dev/shm/sauers_gpu/encoder/real_report.json

`venv_fable` on node2 has `gamfit` (develop) + `torch==2.12.1+cpu`; a tiny MLP
clears 1e5 rows/s on CPU. A CUDA torch wheel gives the B200 headline if wanted
(the harness auto-detects the module's device and syncs).
