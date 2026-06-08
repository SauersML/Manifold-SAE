#!/usr/bin/env python3
"""
Carrier-free audio on the SHARED timeline (synced_timeline.CP_TIME).

Identical carrier-free principle as datafreq_build.py: white noise (fresh random
phase per STFT frame) shaped by a MAGNITUDE-FREQUENCY envelope H(f) built from each
checkpoint's normalized-graph-Laplacian eigen-spectrum (+ covariance tilt). Spectral
overlap-add iFFT, COLA, fade, normalize, tanh soft-limit. NO oscillators / placed
frequencies / scale -- the resonances EMERGE from the data and only MOVE/SHARPEN.

Sync: each STFT frame's center time (seconds) is mapped to a fractional checkpoint
index through the SAME CP_TIME inversion the video uses, so the descriptor heard at
time t is the descriptor for the checkpoint shown at time t. Descriptors are smoothly
(smoothstep) interpolated between bracketing checkpoints on that shared timeline.

Tuning (allowed, no carriers added): slightly higher resonance Q when organized,
gentle excitation low-shelf, wider organization-driven stereo, soft downward
compression of the noise floor for a cleaner late-tone.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import synced_timeline as T

SR = 44100
DUR = T.DUR
FMIN, FMAX = 60.0, 6000.0
NMODES = 8
NFFT = 4096
HOP = NFFT // 4

files = T.FILES
NC = T.NC

def manifold_features(V):
    norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    U = V / norm
    C = U @ U.T
    W = np.clip(C, 0.0, None); np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1); dinv = 1.0 / np.sqrt(d + 1e-12)
    Wn = (W * dinv[:, None]) * dinv[None, :]
    L = np.eye(W.shape[0]) - Wn
    lap = np.clip(np.linalg.eigvalsh(L), 0.0, 2.0)
    gaps = np.diff(np.concatenate([lap, [2.0]]))
    Cov = (V @ V.T) / V.shape[1]
    cov = np.clip(np.linalg.eigvalsh(Cov), 0.0, None)[::-1]
    return lap, gaps, cov

feats = [manifold_features(np.load(f)['V'].astype(np.float64)) for f in files]
lap_all = np.array([x[0] for x in feats])
gap_all = np.array([x[1] for x in feats])
cov_all = np.array([x[2] for x in feats])

freqs_fft = np.fft.rfftfreq(NFFT, 1.0 / SR)
logf = np.log(np.clip(freqs_fft, 1.0, None))
lF, hF = np.log(FMIN), np.log(FMAX)

om_all = np.sqrt(np.clip(lap_all[:, 1:1 + NMODES], 1e-6, 2.0))
om_lo = om_all.min(axis=0, keepdims=True); om_hi = om_all.max(axis=0, keepdims=True)
om_frac = (om_all - om_lo) / (om_hi - om_lo + 1e-12)
band_lo = lF + (hF - lF) * (np.arange(NMODES) / NMODES)
band_hi = lF + (hF - lF) * ((np.arange(NMODES) + 0.9) / NMODES)
band_hi = np.clip(band_hi, None, hF)
mode_logpos = band_lo[None, :] + (band_hi - band_lo)[None, :] * om_frac

g_sel = np.clip(gap_all[:, 1:1 + NMODES], 0, None)
g_norm = g_sel / (g_sel.max() + 1e-12)

cn = cov_all / (cov_all.sum(axis=1, keepdims=True) + 1e-12)
cov_pr = 1.0 / np.sum(cn ** 2, axis=1)
org = 1.0 - (cov_pr - cov_pr.min()) / (np.ptp(cov_pr) + 1e-12)

def build_envelope(ci):
    o = org[ci]
    floor = 0.012 + 0.16 * (1.0 - o)                  # slightly lower floor -> cleaner tone
    H = np.full_like(freqs_fft, floor)
    for m in range(NMODES):
        c = mode_logpos[ci, m]; sharp = g_norm[ci, m]
        # higher Q when organized & sharp (tuned narrower than original 0.30 base)
        bw = (0.27 * (1.0 - 0.62 * sharp)) * (1.0 - 0.84 * o) + 0.009
        height = (0.15 + 0.9 * sharp) * (0.25 + 1.9 * o)
        H += height * np.exp(-0.5 * ((logf - c) / bw) ** 2)
    # gentle excitation low-shelf: a touch more body in the low mids (no placed tone)
    shelf = 1.0 + 0.25 * np.exp(-0.5 * ((logf - np.log(220.0)) / 0.9) ** 2)
    H *= shelf
    H[freqs_fft < FMIN * 0.7] *= 0.04
    H[freqs_fft > FMAX * 1.05] *= 0.08
    return H

env_all = np.array([build_envelope(ci) for ci in range(NC)])

def concentration(H):
    p = H / (H.sum() + 1e-12); return 1.0 / np.sum(p ** 2)
conc = np.array([concentration(env_all[ci]) for ci in range(NC)])
print("envelope concentration (lower=tonal): start %.0f mid %.0f end %.0f"
      % (conc[0], conc[NC // 2], conc[-1]))

# ---- carrier-free overlap-add iFFT on the SHARED timeline ----
N = int(DUR * SR)
win = get_window('hann', NFFT)
nframes = 1 + (N - NFFT) // HOP
norm = np.zeros(N + NFFT)
rng = np.random.default_rng(7)

# frame-center TIME (s) -> fractional checkpoint via SAME CP_TIME inversion as video
frame_t = (np.arange(nframes) * HOP + NFFT / 2) / SR
frame_t = np.clip(frame_t, 0.0, DUR)
cpos = np.interp(frame_t, T.CP_TIME, np.arange(NC))
i0 = np.clip(np.floor(cpos).astype(int), 0, NC - 2)
fr = cpos - i0
fr_s = fr * fr * (3 - 2 * fr)

rgbs = np.array([np.load(f)['rgb'].mean(axis=0) for f in files])
rgbs = rgbs / (rgbs.max() + 1e-9)
pan_cp = rgbs[:, 0] - rgbs[:, 2]
pan_frame = pan_cp[i0] + (pan_cp[i0 + 1] - pan_cp[i0]) * fr_s
# wider stereo when organized (org-driven width), still rgb-derived pan only
org_frame = org[i0] + (org[i0 + 1] - org[i0]) * fr_s
pan_frame = np.clip(pan_frame * (0.6 + 0.5 * org_frame), -1, 1)

outL = np.zeros(N + NFFT); outR = np.zeros(N + NFFT)
nbins = len(freqs_fft)
for fi in range(nframes):
    H = env_all[i0[fi]] + (env_all[i0[fi] + 1] - env_all[i0[fi]]) * fr_s[fi]
    phase = rng.uniform(0, 2 * np.pi, nbins)
    spec = H * np.exp(1j * phase); spec[0] = 0.0
    frame = np.fft.irfft(spec, n=NFFT) * win
    a = fi * HOP
    theta = (pan_frame[fi] + 1) * (np.pi / 4)
    outL[a:a + NFFT] += frame * np.cos(theta)
    outR[a:a + NFFT] += frame * np.sin(theta)
    norm[a:a + NFFT] += win ** 2

norm[norm < 1e-6] = 1e-6
outL = outL[:N] / norm[:N]; outR = outR[:N] / norm[:N]

# soft downward compression of quiet noise floor (cleaner late tone, no added carriers)
def soft_comp(x, thr=0.12, ratio=2.0):
    a = np.abs(x); s = np.sign(x)
    over = a > thr
    a2 = a.copy(); a2[over] = thr + (a[over] - thr) / ratio
    return s * a2
outL = soft_comp(outL); outR = soft_comp(outR)

fade = int(0.6 * SR)
fenv = np.ones(N); fenv[:fade] = np.linspace(0, 1, fade); fenv[-fade:] = np.linspace(1, 0, fade)
outL *= fenv; outR *= fenv

peak = max(np.abs(outL).max(), np.abs(outR).max()) + 1e-9
outL /= peak; outR /= peak
drive = 1.4
outL = np.tanh(outL * drive) / np.tanh(drive)
outR = np.tanh(outR * drive) / np.tanh(drive)
g = 0.92 / max(np.abs(outL).max(), np.abs(outR).max())
outL *= g; outR *= g

stereo = np.stack([outL, outR], axis=1).astype(np.float32)
wavfile.write('/tmp/synced_audio.wav', SR, (stereo * 32767).astype(np.int16))
print("wrote /tmp/synced_audio.wav  dur=%.3fs samples=%d" % (N / SR, N))

# audio checkpoint-transition times: time at which active checkpoint's stage flips.
# active checkpoint per audio frame = nearest bracket on shared timeline.
near = np.where(fr_s < 0.5, i0, i0 + 1)
stg = T.STAGES
trans = []
prev = stg[near[0]]; prev_t = frame_t[0]
for k in range(1, nframes):
    s = stg[near[k]]
    if s != prev:
        trans.append((prev, s, round(frame_t[k], 3))); prev = s
np.savez('/tmp/synced_audio_meta.npz',
         frame_t=frame_t, cpos=cpos, near=near,
         transition_times=np.array([x[2] for x in trans]),
         transition_labels=np.array([f"{a}->{b}" for a, b, _ in trans]))
print("audio stage transitions (s):", trans)
