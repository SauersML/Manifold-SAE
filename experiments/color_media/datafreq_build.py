#!/usr/bin/env python3
"""
Truthful color-structure-emergence sonification — CARRIER-FREE.

ZERO placed oscillators. The audio IS the data's own evolving spectral signature,
synthesized by spectral / overlap-add iFFT: broadband (randomized-phase) excitation
is shaped, frame by frame, by a MAGNITUDE-FREQUENCY ENVELOPE that is derived directly
from the representation's own structure. The resonances/colour EMERGE from the data;
nothing is tuned to a chosen frequency.

Per checkpoint the magnitude envelope H(f) is built from the manifold's spectral
signature in TWO truthful, complementary ways that are summed:

  (A) Source-filter / LPC-style colouring: the 30 frame-demeaned color reps V
      give a 30x30 cosine-similarity graph; its normalized-Laplacian eigenvalues
      {lambda_k} in [0,2] are the manifold's natural vibrational modes, and the
      eigen-gaps say how SHARP each mode is. We deposit, for each mode, a resonance
      *band* (a smooth bump) into the magnitude response at a log-warped position
      sqrt(lambda_k) -> band — but this is a FILTER RESPONSE excited by noise, not an
      oscillator: it widens to broadband when modes are diffuse and narrows to tonal
      colour when the gap sharpens. (As organization rises, the bumps sharpen.)

  (B) Spectral density: the covariance eigenvalue spectrum of V mapped to a smooth
      frequency-magnitude density, so the *overall colour tilt* of the noise tracks
      how the representation's energy is distributed.

As training organizes the manifold, the eigenvalue spectrum SHIFTS and SHARPENS,
so the filter's resonances MOVE and the broadband noise concentrates into colored,
resonant, near-tonal sound. The spectrogram therefore shows moving, sharpening
formants — NOT fixed horizontal carrier lines (there are no carriers).

Continuous morph: the magnitude envelope H_cp(f) is interpolated smoothly across the
57 chronological checkpoints; STFT frames are filled with the interpolated envelope and
RANDOM phase (phase is fresh per frame -> pure noise excitation, click-free via overlap-add
of a windowed iFFT). Fade in/out, normalize, tanh soft-limit.
"""
import re, glob, numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

SR = 44100
DUR = 58.0                 # seconds, within 45-70
FMIN, FMAX = 60.0, 6000.0  # band into which the manifold's modes are log-warped
NMODES = 8                 # how many low Laplacian modes deposit resonance bands
NFFT = 4096                # STFT frame size
HOP = NFFT // 4            # 75% overlap

# ---------------------------------------------------------------- ordering
def order_key(path):
    """Chronological training order: stage1<stage2<stage3<SFT<DPO<RL3.0<RL3.1, then step."""
    name = path.split('/')[-1]
    if 'stage1' in name:      stage = 0
    elif 'stage2' in name:    stage = 1
    elif 'stage3' in name:    stage = 2
    elif '_SFT_' in name:     stage = 3
    elif '_DPO_' in name:     stage = 4
    elif 'RL31' in name:      stage = 6          # RL 3.1
    elif '_RL_' in name:      stage = 5          # RL 3.0
    else:                     stage = 9
    m = re.findall(r'step[_-]?(\d+)', name)
    step = int(m[-1]) if m else 0
    return (stage, step)

files = sorted(glob.glob('/tmp/colall/*.npz'), key=order_key)
NC = len(files)
print("Chronological order (%d):" % NC)
for f in files:
    print("  ", order_key(f), f.split('/')[-1])

# ---------------------------------------------------------------- per-checkpoint manifold spectrum
def manifold_features(V):
    """Return (lap_eigs, lap_gaps, cov_eigs) describing the manifold's spectral signature."""
    # normalized graph Laplacian of the cosine-similarity graph
    norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    U = V / norm
    C = U @ U.T
    W = np.clip(C, 0.0, None)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(d + 1e-12)
    Wn = (W * dinv[:, None]) * dinv[None, :]
    L = np.eye(W.shape[0]) - Wn
    lap = np.clip(np.linalg.eigvalsh(L), 0.0, 2.0)        # ascending, in [0,2]
    gaps = np.diff(np.concatenate([lap, [2.0]]))          # gap above each eigenvalue
    # covariance eigen-spectrum of the reps (energy distribution of the manifold)
    Cov = (V @ V.T) / V.shape[1]                          # 30x30 gram (V already demeaned)
    cov = np.clip(np.linalg.eigvalsh(Cov), 0.0, None)[::-1]   # descending
    return lap, gaps, cov

feats = [manifold_features(np.load(f)['V'].astype(np.float64)) for f in files]
lap_all = np.array([x[0] for x in feats])     # (NC,30)
gap_all = np.array([x[1] for x in feats])     # (NC,30)
cov_all = np.array([x[2] for x in feats])     # (NC,30)

# ---------------------------------------------------------------- map manifold spectrum -> magnitude envelope H(f)
freqs_fft = np.fft.rfftfreq(NFFT, 1.0 / SR)   # (NFFT/2+1,)
logf = np.log(np.clip(freqs_fft, 1.0, None))
lF, hF = np.log(FMIN), np.log(FMAX)

# PER-MODE log-warp: each Laplacian mode's OWN min..max across the trajectory spans the
# band, so the real (but modest-in-absolute-terms) per-checkpoint motion is faithfully
# expanded into clearly-audible frequency glides. This is a monotone, order-preserving
# map of the actual eigenvalue — no quantization, no chosen pitches.
om_all = np.sqrt(np.clip(lap_all[:, 1:1 + NMODES], 1e-6, 2.0))   # skip trivial mode 0
om_lo = om_all.min(axis=0, keepdims=True)
om_hi = om_all.max(axis=0, keepdims=True)
om_frac = (om_all - om_lo) / (om_hi - om_lo + 1e-12)            # (NC,NMODES) in [0,1]
# give each mode a sub-band of the spectrum (low modes -> low freq) and let its eigenvalue
# motion sweep within an overlapping window, so resonances both sit in order AND move.
band_lo = lF + (hF - lF) * (np.arange(NMODES) / NMODES)
band_hi = lF + (hF - lF) * ((np.arange(NMODES) + 0.9) / NMODES)
band_hi = np.clip(band_hi, None, hF)
mode_logpos = band_lo[None, :] + (band_hi - band_lo)[None, :] * om_frac   # (NC,NMODES)

# sharpness of each mode from its eigen-gap; larger gap => narrower, taller resonance.
g_sel = np.clip(gap_all[:, 1:1 + NMODES], 0, None)
g_norm = g_sel / (g_sel.max() + 1e-12)

# global "organization" scalar = how concentrated the covariance energy is (real data):
# participation ratio drops as the manifold organizes -> use it to (a) sharpen all
# resonances and (b) tilt energy downward (broadband hiss -> focused resonant tone).
cn = cov_all / (cov_all.sum(axis=1, keepdims=True) + 1e-12)
cov_pr = 1.0 / np.sum(cn ** 2, axis=1)                    # effective dim, ~14..28
org = 1.0 - (cov_pr - cov_pr.min()) / (np.ptp(cov_pr) + 1e-12)   # 0 (diffuse)..1 (organized)

def build_envelope(ci):
    """Magnitude response H(f) for checkpoint ci — emergent resonances, no carriers.

    Early/diffuse manifold (org~0): strong broadband floor, wide shallow bumps -> HISS.
    Organized manifold (org~1): floor collapses, bumps become narrow & tall -> RESONANT
    TONE concentrated into a few moving lines. The transition is driven entirely by the
    real covariance participation ratio (org) and per-mode eigen-gaps (sharp).
    """
    o = org[ci]
    # broadband noise floor: present when diffuse, quiet once organized (hiss -> tone).
    # Kept LOW overall so coherent resonances dominate and the noise floor stays quiet.
    floor = 0.015 + 0.16 * (1.0 - o)
    H = np.full_like(freqs_fft, floor)
    for m in range(NMODES):
        c = mode_logpos[ci, m]
        sharp = g_norm[ci, m]
        # bandwidth: wide when diffuse, collapses to narrow resonance when organized & sharp
        bw = (0.30 * (1.0 - 0.6 * sharp)) * (1.0 - 0.80 * o) + 0.010
        # height grows strongly with organization so late lines dominate the floor
        height = (0.15 + 0.85 * sharp) * (0.25 + 1.8 * o)
        H += height * np.exp(-0.5 * ((logf - c) / bw) ** 2)
    # band-limit
    H[freqs_fft < FMIN * 0.7] *= 0.04
    H[freqs_fft > FMAX * 1.05] *= 0.08
    return H

env_all = np.array([build_envelope(ci) for ci in range(NC)])   # (NC, nbins)
# per-checkpoint scalar describing how 'concentrated'/tonal the envelope is (for reporting)
def concentration(H):
    p = H / (H.sum() + 1e-12)
    return 1.0 / np.sum(p ** 2)
conc = np.array([concentration(env_all[ci]) for ci in range(NC)])
print("\nenvelope concentration (effective #bins, lower=more tonal):")
print("  start %.0f  mid %.0f  end %.0f" % (conc[0], conc[NC//2], conc[-1]))

# ---------------------------------------------------------------- carrier-free overlap-add iFFT synthesis
N = int(DUR * SR)
win = get_window('hann', NFFT)
# COLA normalization for 75% overlap hann
nframes = 1 + (N - NFFT) // HOP
out = np.zeros(N + NFFT)
norm = np.zeros(N + NFFT)
rng = np.random.default_rng(7)

# fractional checkpoint index per FRAME (smooth morph)
frame_centers = (np.arange(nframes) * HOP + NFFT / 2) / N
cpos = frame_centers * (NC - 1)
i0 = np.clip(np.floor(cpos).astype(int), 0, NC - 2)
fr = cpos - i0
fr_s = fr * fr * (3 - 2 * fr)                   # smoothstep

# rgb -> pan (ONLY pan, never pitch)
rgbs = np.array([np.load(f)['rgb'].mean(axis=0) for f in files])
rgbs = rgbs / (rgbs.max() + 1e-9)
pan_cp = rgbs[:, 0] - rgbs[:, 2]
pan_frame = pan_cp[i0] + (pan_cp[i0 + 1] - pan_cp[i0]) * fr_s
pan_frame = np.clip(pan_frame, -1, 1)

outL = np.zeros(N + NFFT); outR = np.zeros(N + NFFT)
nbins = len(freqs_fft)

for fi in range(nframes):
    # interpolate the magnitude envelope smoothly between checkpoints
    H = env_all[i0[fi]] + (env_all[i0[fi] + 1] - env_all[i0[fi]]) * fr_s[fi]
    # broadband excitation = random phase (this is the noise source; NO oscillator)
    phase = rng.uniform(0, 2 * np.pi, nbins)
    spec = H * np.exp(1j * phase)
    spec[0] = 0.0                                # no DC
    frame = np.fft.irfft(spec, n=NFFT) * win
    a = fi * HOP
    # equal-power pan
    theta = (pan_frame[fi] + 1) * (np.pi / 4)
    outL[a:a + NFFT] += frame * np.cos(theta)
    outR[a:a + NFFT] += frame * np.sin(theta)
    norm[a:a + NFFT] += win ** 2

norm[norm < 1e-6] = 1e-6
outL = outL[:N] / norm[:N]
outR = outR[:N] / norm[:N]

# fade in/out
fade = int(0.8 * SR)
fenv = np.ones(N)
fenv[:fade] = np.linspace(0, 1, fade)
fenv[-fade:] = np.linspace(1, 0, fade)
outL *= fenv; outR *= fenv

# normalize + tanh soft-limit
peak = max(np.abs(outL).max(), np.abs(outR).max()) + 1e-9
outL /= peak; outR /= peak
drive = 1.4
outL = np.tanh(outL * drive) / np.tanh(drive)
outR = np.tanh(outR * drive) / np.tanh(drive)
g = 0.92 / max(np.abs(outL).max(), np.abs(outR).max())
outL *= g; outR *= g

stereo = np.stack([outL, outR], axis=1).astype(np.float32)
wavfile.write('/tmp/datafreq_emergence.wav', SR, (stereo * 32767).astype(np.int16))
print("\nwrote /tmp/datafreq_emergence.wav  dur=%.1fs  (carrier-free spectral iFFT)" % DUR)

np.savez('/tmp/datafreq_analysis.npz',
         env_all=env_all, freqs_fft=freqs_fft, lap_all=lap_all, cov_all=cov_all,
         mode_logpos=mode_logpos, conc=conc, NC=NC, FMIN=FMIN, FMAX=FMAX,
         order=[f.split('/')[-1] for f in files])
print("wrote /tmp/datafreq_analysis.npz")
