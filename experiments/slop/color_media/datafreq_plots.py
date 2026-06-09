#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, welch

SR, stereo = wavfile.read('/tmp/datafreq_emergence.wav')
x = stereo.astype(np.float64).mean(axis=1) / 32768.0
dur = len(x) / SR

# ---- spectrogram of the ACTUAL rendered wav (sane dynamic range) ----
f, tt, Sxx = spectrogram(x, fs=SR, nperseg=8192, noverlap=6144, window='hann')
Sdb = 10 * np.log10(Sxx + 1e-12)
# sane dynamic range: anchor vmax at a high percentile (ignore rare transient spikes),
# show only the top 60 dB so the quiet noise floor stays dark, not lit up.
vmax = np.percentile(Sdb, 99.8)
vmin = vmax - 60
Sdb = np.clip(Sdb, vmin, vmax)

fig, ax = plt.subplots(figsize=(13, 6))
m = ax.pcolormesh(tt, f, Sdb, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
ax.set_ylim(0, 6500)
ax.set_xlabel('time (s)  ->  training progression (stage1 .. stage2 .. stage3 .. SFT .. DPO .. RL3.0 .. RL3.1)')
ax.set_ylabel('frequency (Hz)')
ax.set_title('datafreq_emergence.wav — carrier-free spectral synthesis; resonances emerge from the manifold (NO fixed carriers)')
fig.colorbar(m, ax=ax, label='dB (top 65 dB)')
fig.tight_layout()
fig.savefig('/tmp/datafreq_spectrogram.png', dpi=130)
print("wrote /tmp/datafreq_spectrogram.png")

# ---- data-derived spectral-envelope trajectory across checkpoints ----
A = np.load('/tmp/datafreq_analysis.npz', allow_pickle=True)
env_all = A['env_all']; ff = A['freqs_fft']; NC = int(A['NC'])
# show the magnitude envelope as a heatmap over checkpoints (the FILTER trajectory)
env_db = 20 * np.log10(env_all + 1e-6)
fig, ax = plt.subplots(figsize=(13, 6))
mm = ax.pcolormesh(np.arange(NC), ff, env_db.T, shading='gouraud', cmap='viridis',
                   vmin=env_db.max() - 50, vmax=env_db.max())
ax.set_ylim(40, 6500)
ax.set_yscale('log')
ax.set_xlabel('checkpoint index (chronological)')
ax.set_ylabel('frequency (Hz, log)')
ax.set_title('Data-derived magnitude/filter envelope per checkpoint — resonances MOVE and SHARPEN as structure emerges')
fig.colorbar(mm, ax=ax, label='dB')
# overlay the laplacian-mode resonance centers
mode_logpos = A['mode_logpos']
for k in range(mode_logpos.shape[1]):
    ax.plot(np.arange(NC), np.exp(mode_logpos[:, k]), color='white', lw=0.6, alpha=0.5)
fig.tight_layout()
fig.savefig('/tmp/datafreq_modes.png', dpi=130)
print("wrote /tmp/datafreq_modes.png")

# ---- VERIFICATION ----
def analyze(seg):
    fw, Pxx = welch(seg, fs=SR, nperseg=8192)
    Pxx = Pxx + 1e-15
    centroid = np.sum(fw * Pxx) / np.sum(Pxx)
    order = np.argsort(Pxx)[::-1]
    peaks = []
    for idx in order:
        fr = fw[idx]
        if fr < 40: continue
        if all(abs(fr - pp) > 60 for pp in peaks):
            peaks.append(fr)
        if len(peaks) >= 5: break
    return centroid, sorted(peaks)

n = len(x)
segs = {'start': x[:int(6*SR)],
        'middle': x[n//2 - int(3*SR): n//2 + int(3*SR)],
        'end': x[-int(6*SR):]}
print("\n=== FREQUENCY-MOVEMENT VERIFICATION (from the rendered wav) ===")
cents = {}
for k, s in segs.items():
    c, pk = analyze(s)
    cents[k] = c
    print(f"{k:7s}: spectral centroid = {c:7.1f} Hz | dominant resonances (Hz) = {['%.0f'%p for p in pk]}")
print(f"\ncentroid shift start->middle->end: {cents['start']:.1f} -> {cents['middle']:.1f} -> {cents['end']:.1f} Hz")

# fixed-carrier check: any freq bin 'on' (within 25 dB of max) for >85% of the file?
on = Sdb > (vmax - 25)
frac_on = on.mean(axis=1)
persistent = f[frac_on > 0.85]
print(f"\nfreq bins 'on' >85% of whole file (would be fixed carriers): {len(persistent)}")
peakfreq = f[np.argmax(Sdb, axis=0)]
print(f"per-frame dominant freq: min={peakfreq.min():.0f} max={peakfreq.max():.0f} "
      f"std={peakfreq.std():.0f} Hz (large spread+std => content moves)")
# spectral centroid trajectory over the whole file
cent_traj = (f[:, None] * Sxx).sum(0) / (Sxx.sum(0) + 1e-15)
print(f"spectral-centroid trajectory over file: min={cent_traj.min():.0f} "
      f"max={cent_traj.max():.0f} std={cent_traj.std():.0f} Hz")
