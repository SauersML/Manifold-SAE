import numpy as np, json, colorsys
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

data = json.load(open('/tmp/color_data.json'))
records = data['records']
true_hue = np.array(data['true_hue'])     # (30,) in [0,1)
rgb = np.array(data['rgb'])
N = len(records)                          # 57 checkpoints
NC = len(true_hue)                        # 30 colors

SR = 44100
DUR = 60.0                                 # seconds total
total_samples = int(SR*DUR)

# ---------- per-color TARGET pitch from TRUE hue (musical, just-intoned pentatonic over ~1.5 octaves) ----------
# pentatonic just ratios across ~1.5 octaves
base = 196.0  # G3
ratios = []
penta = [1, 9/8, 5/4, 3/2, 5/3]  # just pentatonic
for octv in range(2):
    for r in penta:
        ratios.append(r*(2**octv))
ratios = np.array(sorted(ratios))
ratios = ratios[ratios <= 3.0]  # ~1.5 octaves
def hue_to_target_freq(h):
    idx = int(np.clip(h,0,0.9999)*len(ratios))
    return base*ratios[idx]
target_freq = np.array([hue_to_target_freq(h) for h in true_hue])  # (30,)

# a "disordered" reference pitch per color: pull from its index spread across a dissonant cluster around base
rng = np.random.default_rng(42)
disorder_freq = base*(2**(rng.uniform(-0.15,1.35,size=NC)))  # scattered, inharmonic

# ---------- build per-checkpoint per-color descriptors ----------
# pitch = blend between disorder_freq (when manifold unorganized) and target_freq (when organized)
# organization driver = circularity (global) blended with per-color on_manifold (local)
# detune (beating roughness) added when on_manifold low
ck_freq = np.zeros((N,NC))
ck_amp  = np.zeros((N,NC))
ck_detune = np.zeros((N,NC))
ck_pan = np.zeros((N,NC))   # by true hue for stereo width
for i,r in enumerate(records):
    circ = r['circularity']
    onm = np.array(r['on_manifold'])      # (30,)
    h1 = r['h1']
    # organization weight per color: how locked this partial is to its hue pitch
    org = np.clip(0.5*circ + 0.5*onm, 0, 1)
    org = org**1.5  # sharpen so emergence is audible
    # pitch in log domain for smooth glide
    lf = np.log(disorder_freq)*(1-org) + np.log(target_freq)*org
    ck_freq[i] = np.exp(lf)
    # detune: random beating offset, large when disordered -> roughness/dissonance
    ck_detune[i] = (1-org)*disorder_freq*0.02*rng.standard_normal(NC)
    # amplitude: on-manifold colors louder; slight global swell with circularity
    ck_amp[i] = (0.3 + 0.7*onm) * (0.6+0.4*circ)
    ck_pan[i] = true_hue  # 0..1 -> L..R
# normalize amp rows so total energy ~ stable but consonant later sounds fuller
for i in range(N):
    ck_amp[i] /= (np.sqrt(np.sum(ck_amp[i]**2))+1e-9)

# ---------- timeline: place checkpoints, interpolate per-sample (cosine) ----------
# slightly more time on pretrain so emergence breathes; but keep monotone chronological
seg_w = np.ones(N)
seg_bounds = np.linspace(0, total_samples, N)  # checkpoint i at sample seg_bounds[i]
t = np.arange(total_samples)
# for each sample find bracketing checkpoints
seg_idx = np.searchsorted(seg_bounds, t, side='right')-1
seg_idx = np.clip(seg_idx, 0, N-2)
left = seg_bounds[seg_idx]; right = seg_bounds[seg_idx+1]
frac = (t-left)/np.maximum(right-left,1)
w = 0.5*(1-np.cos(np.pi*np.clip(frac,0,1)))  # cosine crossfade weight

# interpolate descriptors per sample (in log-freq for pitch)
def interp(arr):  # arr (N,NC) -> (total_samples,NC)
    a = arr[seg_idx]; b = arr[seg_idx+1]
    return a*(1-w[:,None]) + b*w[:,None]

logf = interp(np.log(ck_freq))
freq_s = np.exp(logf)
detune_s = interp(ck_detune)
amp_s = interp(ck_amp)
pan_s = interp(ck_pan)

# instantaneous freq with detune; integrate phase to avoid clicks
inst_f = freq_s + detune_s
phase = 2*np.pi*np.cumsum(inst_f, axis=0)/SR  # (total,NC)
# add a slight detuned second voice per partial for beating when disordered (already via detune); keep simple
voices = np.sin(phase)  # (total,NC)

# amplitude smoothing (extra) to kill any residual zipper noise
from scipy.ndimage import uniform_filter1d
amp_s = uniform_filter1d(amp_s, size=441, axis=0, mode='nearest')
pan_s = uniform_filter1d(pan_s, size=441, axis=0, mode='nearest')

# stereo pan (equal power) by hue
panL = np.cos(pan_s*np.pi/2); panR = np.sin(pan_s*np.pi/2)
L = np.sum(voices*amp_s*panL, axis=1)
R = np.sum(voices*amp_s*panR, axis=1)

# gentle global low-pass to smooth highs
sos = butter(4, 6000, 'lp', fs=SR, output='sos')
L = sosfilt(sos, L); R = sosfilt(sos, R)

# global fade in/out
fade = int(SR*1.5)
env = np.ones(total_samples)
env[:fade] = np.linspace(0,1,fade)
env[-fade:] = np.linspace(1,0,fade)
L*=env; R*=env

# normalize + soft limit (tanh)
stereo = np.stack([L,R],axis=1)
stereo /= (np.max(np.abs(stereo))+1e-9)
stereo = np.tanh(stereo*1.6)/np.tanh(1.6)
stereo /= (np.max(np.abs(stereo))+1e-9)
stereo *= 0.95
pcm = (stereo*32767).astype(np.int16)
wavfile.write('/tmp/color_emergence.wav', SR, pcm)
print(f"Wrote /tmp/color_emergence.wav  dur={DUR}s stereo {SR}Hz")
print(f"target_freq range {target_freq.min():.1f}-{target_freq.max():.1f} Hz")
print(f"early mean freq spread (std) {ck_freq[2].std():.1f}  late {ck_freq[-1].std():.1f}")
