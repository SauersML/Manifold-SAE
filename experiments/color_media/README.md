# Color-manifold media (figures, animations, sonifications)

Generators for the across-training color-manifold visualizations and sonifications
(data: per-checkpoint 30-color L44 reps extracted from the Blob into /tmp/colall/*.npz).

**Video**: `umap_movie2.py` (minimal-total-distance-aligned UMAP evolution),
`fig_umap_all.py` (57-panel UMAP grid), `synced_{timeline,video,audio}.py` (the final
synced+faster audio+video at a shared checkpoint->time timeline -> synced_color.mp4).

**Sonifications** (all driven by the real per-checkpoint data, morphing pretrain->RL):
- `huenote_*`     — hue -> just-scale pitches (carrier-based).
- `native_*`      — graph-Laplacian eigenmodes as resonant partials (carrier-based, data-driven freqs).
- `harmonic_*`    — circular-harmonic decomposition of the recovered loop, fixed-f0 partials (carrier-based).
- `datafreq_*`    — CARRIER-FREE: white noise shaped by the manifold's own Laplacian magnitude-filter
                    (spectral overlap-add iFFT; no oscillators/placed frequencies). The honest one.

Through-line: color structure crystallizes in the first few pretraining checkpoints (participation
ratio 28->~13) and holds steady through SFT/DPO/RL.
