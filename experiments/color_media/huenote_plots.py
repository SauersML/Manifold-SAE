import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

data = json.load(open('/tmp/color_data.json'))
recs = data['records']
N = len(recs)
circ = np.array([r['circularity'] for r in recs])
h1 = np.array([r['h1'] for r in recs])
h2 = np.array([r['h2'] for r in recs])
h3 = np.array([r['h3'] for r in recs])
stages = [r['stage'] for r in recs]

# stage band colors
stage_col = {'stage1':'#cfe8ff','stage2':'#aee0c0','stage3':'#ffe9a8',
             'SFT':'#ffd0a8','DPO':'#f3c0e0','RL30':'#d8c0ff','RL31':'#c0c0ff'}
stage_lbl = {'stage1':'Pretrain s1','stage2':'s2','stage3':'s3',
             'SFT':'SFT','DPO':'DPO','RL30':'RL3.0','RL31':'RL3.1'}

fig, axes = plt.subplots(2,1, figsize=(13,8), sharex=True)
x = np.arange(N)

# shade stage bands
def shade(ax):
    i=0
    while i<N:
        j=i
        while j<N and stages[j]==stages[i]: j+=1
        ax.axvspan(i-0.5, j-0.5, color=stage_col[stages[i]], alpha=0.55, lw=0)
        ax.text((i+j-1)/2, ax.get_ylim()[1]*0.97, stage_lbl[stages[i]],
                ha='center', va='top', fontsize=8, rotation=0, color='#333')
        i=j

ax=axes[0]
ax.set_ylim(0,1.0); shade(ax)
ax.plot(x, circ, '-o', color='#111', ms=4, lw=1.8, label='Circularity (|circ-corr| recovered vs true hue)')
ax.axhline(circ[:5].mean(), color='#888', ls=':', lw=1)
ax.set_ylabel('Circularity'); ax.legend(loc='lower right', fontsize=9)
ax.set_title('Emergence of the hue circle across OLMo-3-32B training (chronological)')

ax=axes[1]
ax.set_ylim(0,1.0); shade(ax)
ax.plot(x, h1, '-o', color='#c0392b', ms=4, lw=1.8, label='1st harmonic fraction (clean circle)')
ax.plot(x, h2, '-s', color='#2980b9', ms=3, lw=1.2, label='2nd harmonic')
ax.plot(x, h3, '-^', color='#27ae60', ms=3, lw=1.2, label='3rd harmonic')
ax.set_ylabel('Harmonic power fraction'); ax.set_xlabel('Checkpoint (chronological index)')
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('/tmp/color_harmonics.png', dpi=130)
print("Wrote /tmp/color_harmonics.png")
print(f"early circ {circ[:5].mean():.3f} h1 {h1[:5].mean():.3f}")
print(f"late  circ {circ[-10:].mean():.3f} h1 {h1[-10:].mean():.3f}")

# ---- spectrogram of the wav ----
try:
    from scipy.io import wavfile
    from scipy.signal import spectrogram
    sr, w = wavfile.read('/tmp/color_emergence.wav')
    mono = w.mean(1) if w.ndim>1 else w
    mono = mono/32768.0
    f, t, Sxx = spectrogram(mono, sr, nperseg=8192, noverlap=6144)
    fig2, a2 = plt.subplots(figsize=(13,5))
    sel = f<3500
    a2.pcolormesh(t, f[sel], 10*np.log10(Sxx[sel]+1e-10), shading='auto', cmap='magma')
    a2.set_ylabel('Hz'); a2.set_xlabel('time (s)')
    a2.set_title('Spectrogram of color_emergence.wav — scattered/beating early -> locked hue partials late')
    plt.tight_layout(); plt.savefig('/tmp/color_emergence_spectrogram.png', dpi=130)
    print("Wrote /tmp/color_emergence_spectrogram.png")
except Exception as e:
    print("spectrogram skipped:", e)
