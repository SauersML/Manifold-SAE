import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

rows = json.load(open('/tmp/native_traj.json'))
N=len(rows); K=len(rows[0]['lap_eig'])
lap=np.array([r['lap_eig'] for r in rows])
cov=np.array([r['cov_top'] for r in rows])
se=np.array([r['spec_entropy'] for r in rows])
pr=np.array([r['participation'] for r in rows])
ts=np.array([r['top_share'] for r in rows])

# stage boundaries (chrono index) for annotation
labels=[r['label'] for r in rows]
def stage_of(l):
    if 'RL31' in l: return 'RL3.1'
    if 'TRAJ_RL__' in l: return 'RL3.0'
    if 'DPO' in l: return 'DPO'
    if 'SFT' in l: return 'SFT'
    if 'stage1' in l: return 'pretrain-1'
    if 'stage2' in l: return 'pretrain-2'
    if 'stage3' in l: return 'pretrain-3'
    return '?'
stages=[stage_of(l) for l in labels]
bounds=[i for i in range(1,N) if stages[i]!=stages[i-1]]

fig,axs=plt.subplots(3,1,figsize=(13,11),sharex=True)
x=np.arange(N)

# 1: Laplacian eigenmode trajectory (the manifold's vibrational modes)
ax=axs[0]
base_f=np.geomspace(110,1760,K)
for k in range(K):
    fk=base_f[k]*(0.6+0.9*np.sqrt(np.clip(lap[:,k],0,2)/2))
    ax.plot(x,fk,lw=1.6,label=f'mode {k+1}')
ax.set_ylabel('modal freq (Hz)')
ax.set_yscale('log')
ax.set_title('OLMo-3-32B color manifold sonified — graph-Laplacian eigenmodes -> resonant partials')
ax.legend(ncol=4,fontsize=7,loc='upper right')

# 2: covariance partial amplitudes (energy concentration)
ax=axs[1]
im=ax.imshow(cov.T,aspect='auto',origin='lower',extent=[0,N-1,0.5,K+0.5],cmap='magma',interpolation='nearest')
ax.set_ylabel('PCA partial #')
ax.set_title('covariance spectrum -> partial amplitudes (variance concentration)')
plt.colorbar(im,ax=ax,pad=0.01,label='rel. energy')

# 3: scalar descriptors
ax=axs[2]
ax.plot(x,se,'-o',ms=3,color='tab:red',label='spectral entropy (-> noise/inharmonicity)')
ax.plot(x,ts,'-o',ms=3,color='tab:blue',label='top-mode share')
ax2=ax.twinx()
ax2.plot(x,pr,'-s',ms=3,color='tab:green',label='participation ratio (-> darkness)')
ax.set_ylabel('entropy / top-share'); ax2.set_ylabel('participation (eff dim)')
ax.set_xlabel('checkpoint (chronological)')
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,fontsize=8,loc='center right')

for ax in axs:
    for b in bounds: ax.axvline(b-0.5,color='k',ls=':',alpha=0.35,lw=0.8)
# stage labels along top
seen={}
for i,s in enumerate(stages):
    if s not in seen: seen[s]=i
for s,i in seen.items():
    axs[0].text(i+0.3, axs[0].get_ylim()[1]*0.9, s, fontsize=7, rotation=90, va='top', alpha=0.7)

plt.tight_layout()
plt.savefig('/tmp/native_decomp_trajectory.png',dpi=120)
print('wrote /tmp/native_decomp_trajectory.png')

# spectrogram of the rendered audio
sr,a=wavfile.read('/tmp/native_manifold_drum.wav')
mono=a.astype(float).mean(1)
f,tt,Sxx=spectrogram(mono,sr,nperseg=8192,noverlap=6144)
plt.figure(figsize=(13,5))
plt.pcolormesh(tt,f,10*np.log10(Sxx+1e-12),shading='auto',cmap='inferno',vmin=-90,vmax=-20)
plt.ylim(0,3500); plt.ylabel('Hz'); plt.xlabel('time (s)')
plt.title('Spectrogram: noisy/flat (random init) -> resonant pitched partials (post-pretrain) -> stable')
plt.colorbar(label='dB')
plt.tight_layout(); plt.savefig('/tmp/native_spectrogram.png',dpi=120)
print('wrote /tmp/native_spectrogram.png')
