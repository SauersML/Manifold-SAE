import numpy as np, json
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
import colorsys, glob, os, re

# ---- reload trajectory + mean hue for panning ----
rows = json.load(open('/tmp/native_traj.json'))
N = len(rows)
K = len(rows[0]['lap_eig'])

lap = np.array([r['lap_eig'] for r in rows])        # (N,K) in [0,2]
cov = np.array([r['cov_top'] for r in rows])        # (N,K) relative partial energy
se  = np.array([r['spec_entropy'] for r in rows])   # (N,)
pr  = np.array([r['participation'] for r in rows])  # (N,)

# mean hue per ckpt (rgb panning only)
files = sorted(glob.glob('/tmp/colall/*.npz'), key=lambda p:int(os.path.basename(p).split('__')[0]))
# rebuild same chrono order as decomp script
def sort_key(p):
    label = os.path.basename(p).split('__',1)[1].rsplit('.npz',1)[0]
    if 'RL31' in label: stage=6
    elif re.search(r'TRAJ_RL__',label): stage=5
    elif 'DPO' in label: stage=4
    elif 'SFT' in label: stage=3
    elif 'stage1' in label: stage=0
    elif 'stage2' in label: stage=1
    elif 'stage3' in label: stage=2
    else: stage=99
    m=re.search(r'step[_-]?(\d+)',label); step=int(m.group(1)) if m else (10**9 if 'DPO' in label else 0)
    return (stage,step)
files = sorted(glob.glob('/tmp/colall/*.npz'), key=sort_key)
hue=[]
for f in files:
    rgb=np.load(f)['rgb']; m=rgb.mean(0); m=np.clip(m,0,1)
    hue.append(colorsys.rgb_to_hsv(*m)[0])
hue=np.array(hue)

# ================= SYNTH PARAMS =================
sr = 44100
DUR = 58.0
n = int(sr*DUR)
t = np.arange(n)/sr

# per-checkpoint time anchors, equal spacing along the morph
anchors = np.linspace(0, DUR, N)

def interp_smooth(vals_per_ckpt):
    # vals_per_ckpt: (N,) -> (n,) smooth interpolation in time
    x = np.interp(t, anchors, vals_per_ckpt)
    # mild smoothing to kill any residual kinks (~12ms)
    return gaussian_filter1d(x, sigma=int(sr*0.012))

def slow_noise(seed, smooth_s):
    # band-limited noise, cheap: low control-rate noise -> gaussian-smooth -> upsample to audio rate
    cr = 300  # control-rate Hz
    m = int(DUR*cr)+4
    g = np.random.default_rng(seed).standard_normal(m)
    g = gaussian_filter1d(g, sigma=max(1.0, smooth_s*cr))
    g = g/(np.abs(g).max()+1e-9)
    return np.interp(t, np.linspace(0,DUR,m), g)

# Map Laplacian eigenvalues (0..2) -> audible frequencies, log-spaced register.
# Mode k gets a base register; the eigenvalue modulates it. Lower modes = lower pitch.
base_f = np.geomspace(110, 1760, K)   # A2 .. A6 spread across the K modes
# eigenvalue (graph connectivity) nudges each partial: more structure -> partials separate & ring
freq_ck = np.zeros((N,K))
for k in range(K):
    # eigenvalue scaled into a +/- ratio around base; sqrt(eig) ~ vibrational freq of a mode
    freq_ck[:,k] = base_f[k] * (0.6 + 0.9*np.sqrt(np.clip(lap[:,k],0,2)/2))

# Amplitudes from covariance spectrum (energy concentration) * a global ring envelope
amp_ck = cov.copy()
amp_ck = amp_ck / (amp_ck.max(axis=1,keepdims=True)+1e-9)

# inharmonicity / noise from spectral entropy: high entropy(early)=more noise & detune jitter
# normalize se to 0..1 over observed range for expressive control
se_n = (se - se.min())/(se.max()-se.min()+1e-9)         # 0=most ordered(early surge), 1=flat
# but the *flattest* is index0 (random). We want early random -> noisy. se index0 is max(0.985)->se_n~1 good.
noise_amt = interp_smooth(se_n)            # 0..1
detune = interp_smooth(se_n)               # frequency jitter scale

# brightness / lowpass from participation ratio (high eff-dim = darker/diffuse)
pr_n = (pr - pr.min())/(pr.max()-pr.min()+1e-9)   # 1 early(diffuse) .. 0 concentrated
bright = interp_smooth(1.0 - pr_n)                # higher = brighter

# ============ ADDITIVE MODAL SYNTH (phase-integrated) ============
out = np.zeros(n)
rng = np.random.default_rng(0)
# slow random detune signal (one per mode), shaped by detune amount -> inharmonic shimmer early
for k in range(K):
    f_t = interp_smooth(freq_ck[:,k])
    a_t = interp_smooth(amp_ck[:,k])
    # detune jitter: band-limited (~few Hz) noise * detune amount * a few % of freq
    jit = slow_noise(100+k, 0.05)
    f_inst = f_t * (1.0 + 0.06*detune*jit)
    phase = 2*np.pi*np.cumsum(f_inst)/sr          # integrate inst freq -> continuous phase
    # timbre: add a touch of 2nd partial that strengthens as structure forms (1-noise)
    tone = np.sin(phase) + 0.25*(1-noise_amt)*np.sin(2*phase)
    out += a_t * tone

# normalize modal sum
out = out/(K*0.9)

# broadband membrane noise (the un-resonant 'drum head'), gated by spectral entropy
noise = rng.standard_normal(n)
# bandpass-ish: emphasize a moving band by simple smoothing diff (cheap), scaled by brightness
noise_lp = gaussian_filter1d(noise, sigma=2)
noise_band = noise - gaussian_filter1d(noise, sigma=8)   # rough hi-mid band
membrane = 0.5*noise_amt * (0.4*noise_lp + 0.6*noise_band)
membrane *= (0.5+0.5*bright)
out = out + 0.6*membrane

# global low-pass that opens with brightness (one-pole, time-varying)
# implement as morph between a smoothed (dark) and raw (bright) version
dark = gaussian_filter1d(out, sigma=4)
out = dark + bright*(out-dark)

# ---- overall amplitude envelope: gentle, plus the early 'crystallization' swell is data-driven already ----
# fade in/out
fade = int(sr*0.6)
env = np.ones(n)
env[:fade] = np.linspace(0,1,fade)**2
env[-fade:] = np.linspace(1,0,fade)**2
out *= env

# ---- stereo pan from mean hue (allowed) ----
pan = interp_smooth((hue-0.5))      # -0.5..0.5
pan = np.clip(pan*1.4,-0.9,0.9)
left  = out*np.sqrt(0.5*(1-pan))
right = out*np.sqrt(0.5*(1+pan))

# ---- normalize + soft limit (tanh) ----
stereo = np.stack([left,right],1)
stereo = stereo/(np.abs(stereo).max()+1e-9)
stereo = np.tanh(1.6*stereo)
stereo = stereo/(np.abs(stereo).max()+1e-9)*0.95
wavfile.write('/tmp/native_manifold_drum.wav', sr, (stereo*32767).astype(np.int16))
print("WROTE /tmp/native_manifold_drum.wav", round(DUR,1),"s", stereo.shape)
print("noise_amt early/late:", round(float(noise_amt[100]),3), round(float(noise_amt[-100]),3))
print("bright early/late:", round(float(bright[100]),3), round(float(bright[-100]),3))
