import json, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

d = json.load(open('/tmp/harmonic_records.json'))
recs = d['records']
STAGE = d['stage_names']
K = d['K']
N = len(recs)

# ---- per-checkpoint descriptors (truthful, from data) ----
# harmonic amplitudes ~ sqrt(energy) so power = data energy
Hen = np.array([r['harm_energy'] for r in recs])           # (N,K) energies
noise = np.array([r['noise_energy'] for r in recs])        # (N,)
sig   = np.array([r['sig_energy'] for r in recs])          # (N,)
HNR   = np.array([r['HNR'] for r in recs])

# normalize harmonic energies per-checkpoint to a spectrum (share), keep total signal too
Hshare = Hen / Hen.sum(1, keepdims=True)                   # (N,K)
amp = np.sqrt(Hshare)                                       # partial amplitudes
# "noise"/inharmonicity descriptor: how DIFFUSE the harmonic spectrum is.
# participation ratio PR = effective # of harmonics carrying energy (1..K).
# Early training: energy smeared across many k -> PR high -> buzzy/inharmonic haze.
# Late: energy locks into low harmonics -> PR low -> clean tone. This is the
# strongest, most honest emergent trend in the data (k=1 share 0.30->0.40, PR 6->4.5).
PR = 1.0/(Hshare**2).sum(1)                                 # (N,) 1..K
noise_frac = (PR - 1.0)/(K - 1.0)                           # 0..1 diffuseness
# overall "structure" loudness: keep modest dynamic range tied to total signal energy (log)
logsig = np.log10(sig)
loud = (logsig - logsig.min()) / (logsig.max() - logsig.min() + 1e-9)
loud = 0.6 + 0.4*loud                                      # 0.6..1.0, gentle

# ---- audio params ----
SR = 44100
DUR = 58.0                                                 # seconds
n_samples = int(DUR*SR)
f0 = 98.0                                                  # fixed low fundamental (no scale imposed; arbitrary anchor)
t = np.arange(n_samples)/SR

# light temporal smoothing across checkpoints (3-tap) so descriptors evolve cleanly,
# preserving the trend while removing single-checkpoint jitter (still data-driven)
def smooth(M, ax=0):
    M = np.asarray(M, float)
    pad = np.pad(M, ((1,1),)+((0,0),)*(M.ndim-1), mode='edge')
    return (pad[:-2]+2*pad[1:-1]+pad[2:])/4.0
amp = smooth(amp)   # relative partial amplitudes; absolute scale set later by normalization
noise_frac = smooth(noise_frac)
loud = smooth(loud)

# map sample time -> continuous checkpoint position [0, N-1]
# allocate slightly more time to pretrain emergence (where structure forms) -> use uniform; honest & simple
pos = np.linspace(0, N-1, n_samples)
i0 = np.floor(pos).astype(int); i0 = np.clip(i0, 0, N-2)
frac = pos - i0

def interp_rows(M):
    # smooth (cosine) interpolation between consecutive checkpoint rows -> no clicks
    w = 0.5 - 0.5*np.cos(np.pi*frac)
    return M[i0]*(1-w[:,None]) + M[i0+1]*w[:,None]

def interp_vec(v):
    w = 0.5 - 0.5*np.cos(np.pi*frac)
    return v[i0]*(1-w) + v[i0+1]*w

amp_t   = interp_rows(amp)        # (n_samples, K)
nf_t    = interp_vec(noise_frac)
loud_t  = interp_vec(loud)

# ---- additive harmonic synthesis: partials at k*f0, amplitude from data ----
# integrate phase (fundamental fixed so trivial, but keep general form)
out = np.zeros(n_samples)
for k in range(1, K+1):
    fk = k*f0
    phase = 2*np.pi*fk*t
    # slight per-partial detune-free; amplitude rolloff already in data
    out += amp_t[:, k-1] * np.sin(phase)

# normalize harmonic part
out /= np.sqrt((amp_t**2).sum(1) + 1e-9)

# ---- noise haze: filtered noise, loud early (high noise_frac), quiet late ----
rng = np.random.default_rng(0)
white = rng.standard_normal(n_samples)
# bandpass the noise around the harmonic region so it reads as "breathiness" not hiss
sos = butter(2, [f0*0.7/(SR/2), f0*K*1.2/(SR/2)], btype='band', output='sos')
haze = sosfilt(sos, white)
haze /= (np.abs(haze).max()+1e-9)
# noise amplitude tracks measured residual share; scaled so early is audible haze, late near-silent
haze_gain = 0.55 * (nf_t**1.5)
# normalize nf to its own range so the *change* is what's heard
nf_lo, nf_hi = noise_frac.min(), noise_frac.max()
nf_norm = (nf_t - nf_lo)/(nf_hi-nf_lo+1e-9)
haze_gain = 0.45 * nf_norm**1.3

tone = out * (1.0 - 0.25*nf_norm)        # harmonic tone slightly ducked when noisy
mix = tone + haze*haze_gain

# overall loudness envelope from data signal energy
mix *= loud_t

# global fade in/out to avoid clicks
fade = int(0.4*SR)
env = np.ones(n_samples)
env[:fade] = np.linspace(0,1,fade)
env[-fade:] = np.linspace(1,0,fade)
mix *= env

# normalize + soft limit (tanh)
mix /= (np.abs(mix).max()+1e-9)
mix = np.tanh(1.6*mix)/np.tanh(1.6)
mix /= (np.abs(mix).max()+1e-9)
mix *= 0.92

# ---- stereo: pan by mean hue of the checkpoint's true colors (cosmetic only, allowed) ----
# load rgb per checkpoint -> mean hue angle -> pan; does NOT affect pitch
import glob, re
def sort_key(path):
    lab = path.split('/')[-1].replace('.npz','')
    if 'stage1' in lab: s=0; st=int(re.search(r'step(\d+)',lab).group(1))
    elif 'stage2' in lab: s=1; st=int(re.search(r'step(\d+)',lab).group(1))
    elif 'stage3' in lab: s=2; st=int(re.search(r'step(\d+)',lab).group(1))
    elif '_SFT_' in lab: s=3; st=int(re.search(r'step(\d+)',lab).group(1))
    elif '_DPO_' in lab: s=4; st=0
    elif 'RL31' in lab: s=6; st=int(re.search(r'step_?(\d+)',lab).group(1))
    elif '_RL_' in lab: s=5; st=int(re.search(r'step_?(\d+)',lab).group(1))
    return (s,st)
files = sorted(glob.glob('/tmp/colall/*.npz'), key=sort_key)
import colorsys
pan = []
for f in files:
    rgb = np.load(f)['rgb']
    rgb = np.clip(rgb/255.0, 0, 1)
    # mean saturation-weighted hue -> single pan value, just for spatial interest
    hs = np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb])
    pan.append(np.sin(2*np.pi*hs).mean())   # -1..1
pan = np.array(pan)
pan = (pan - pan.min())/(np.ptp(pan)+1e-9)*1.4 - 0.7   # gentle -0.7..0.7
pan_t = interp_vec(pan)
left  = mix * np.sqrt(0.5*(1-pan_t*0.5))
right = mix * np.sqrt(0.5*(1+pan_t*0.5))
stereo = np.column_stack([left, right])
stereo = (stereo/np.abs(stereo).max()*0.92*32767).astype(np.int16)

wavfile.write('/tmp/harmonic_emergence.wav', SR, stereo)
print('wrote /tmp/harmonic_emergence.wav  dur=%.1fs  f0=%.0fHz  K=%d' % (DUR, f0, K))
print('diffuseness(noise) early(mean5)=%.3f late(mean7)=%.3f' % (noise_frac[:5].mean(), noise_frac[-7:].mean()))
print('PR early=%.2f late=%.2f' % (PR[:5].mean(), PR[-7:].mean()))
