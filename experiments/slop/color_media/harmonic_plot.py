import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

d = json.load(open('/tmp/harmonic_records.json'))
recs = d['records']; STAGE = d['stage_names']; K = d['K']
N = len(recs)
Hen = np.array([r['harm_energy'] for r in recs])
Hshare = Hen/Hen.sum(1,keepdims=True)
PR = 1.0/(Hshare**2).sum(1)
h1 = Hshare[:,0]
stages = np.array([r['stage'] for r in recs])

# stage band boundaries
bounds = []
for s in range(7):
    idx = np.where(stages==s)[0]
    if len(idx): bounds.append((idx[0], idx[-1], STAGE[s]))
cmap = plt.cm.viridis(np.linspace(0.1,0.9,7))

fig, axs = plt.subplots(3,1, figsize=(13,11), sharex=True)

# (1) harmonic spectrum heatmap (share per k vs checkpoint)
ax=axs[0]
im=ax.imshow(Hshare.T, aspect='auto', origin='lower', cmap='magma',
             extent=[0,N-1,0.5,K+0.5], vmin=0, vmax=0.45)
ax.set_ylabel('circular harmonic k')
ax.set_title('Color manifold circular-harmonic spectrum across OLMo-3-32B training\n(share of harmonic energy in harmonic k of the unsupervised color loop)')
plt.colorbar(im, ax=ax, label='energy share', pad=0.01)
for a in axs:
    for (lo,hi,nm) in bounds:
        a.axvline(hi+0.5, color='w' if a is ax else '0.5', lw=0.8, ls=':')

# (2) k=1 share + participation ratio
ax=axs[1]
ax.plot(h1, color='crimson', lw=2, label='1st-harmonic energy share (purity)')
ax.set_ylabel('1st-harmonic share', color='crimson')
ax.tick_params(axis='y', labelcolor='crimson')
ax2=ax.twinx()
ax2.plot(PR, color='steelblue', lw=2, label='participation ratio (eff. # harmonics)')
ax2.set_ylabel('eff. # harmonics (diffuseness)', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax.set_title('Emergence of harmonic structure: energy concentrating into the fundamental loop harmonic')

# (3) harmonic-to-noise ratio (HNR) computed in analysis
ax=axs[2]
HNR=np.array([r['HNR'] for r in recs])
ax.plot(HNR, color='darkgreen', lw=2)
ax.set_ylabel('HNR (harmonic / residual)')
ax.set_xlabel('checkpoint index (chronological)')
ax.set_title('Harmonic-to-residual energy ratio')

# stage labels on top axis (alternate vertical offset to avoid overlap on short stages)
for j,(lo,hi,nm) in enumerate(bounds):
    yo = K+0.7 + (1.1 if (j%2 and (hi-lo)<4) else 0)
    axs[0].text((lo+hi)/2, yo, nm, ha='center', va='bottom', fontsize=8.5, rotation=0)
axs[0].set_ylim(0.5, K+2.8)

plt.tight_layout()
plt.savefig('/tmp/harmonic_trajectory.png', dpi=130)
print('wrote /tmp/harmonic_trajectory.png')

# ---- spectrogram of the rendered audio ----
sr,x = wavfile.read('/tmp/harmonic_emergence.wav')
x = x.astype(float).mean(1)/32768
fig,ax=plt.subplots(figsize=(13,5))
Pxx,freqs,bins,im = ax.specgram(x, NFFT=8192, Fs=sr, noverlap=6144, cmap='inferno', vmin=-110, vmax=-30)
ax.set_ylim(0,1100)
ax.set_xlabel('time (s)  —  pretrain → stage2 → stage3 → SFT → DPO → RL3.0 → RL3.1')
ax.set_ylabel('frequency (Hz)')
ax.set_title('Spectrogram of harmonic_emergence.wav  (f0=98 Hz; partials = data harmonic amplitudes; haze = spectral diffuseness)')
# stage time markers
T=bins[-1]
for (lo,hi,nm) in bounds:
    tt = (hi+0.5)/(N-1)*T
    ax.axvline(tt, color='cyan', lw=0.6, ls=':')
plt.colorbar(im, ax=ax, label='dB')
plt.tight_layout()
plt.savefig('/tmp/harmonic_spectrogram.png', dpi=130)
print('wrote /tmp/harmonic_spectrogram.png')

# companion: true colors row (cosmetic context)
import glob,re
def sk(p):
    l=p.split('/')[-1]
    if 'stage1' in l:return(0,int(re.search(r'step(\d+)',l).group(1)))
    if 'stage2' in l:return(1,int(re.search(r'step(\d+)',l).group(1)))
    if 'stage3' in l:return(2,int(re.search(r'step(\d+)',l).group(1)))
    if '_SFT_' in l:return(3,int(re.search(r'step(\d+)',l).group(1)))
    if '_DPO_' in l:return(4,0)
    if 'RL31' in l:return(6,int(re.search(r'step_?(\d+)',l).group(1)))
    return(5,int(re.search(r'step_?(\d+)',l).group(1)))
print('done')
