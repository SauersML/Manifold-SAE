import glob,os,re,numpy as np,colorsys,warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import umap
from scipy.linalg import orthogonal_procrustes
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":150,"savefig.facecolor":"white"})
def chrono(l):
    s=(0 if "stage1" in l else 1 if "stage2" in l else 2 if "stage3" in l else 3 if "SFT" in l else 4 if "DPO" in l else 6 if "RL31" in l else 5 if "_RL_" in l else 9)
    m=re.search(r'step_?(\d+)',l); return (s,int(m.group(1)) if m else 0)
def short(l):
    st=("s1" if "stage1" in l else "s2" if "stage2" in l else "s3" if "stage3" in l else "SFT" if "SFT" in l else "DPO" if "DPO" in l else "RL3.1" if "RL31" in l else "RL3.0")
    m=re.search(r'step_?(\d+)',l); return f"{st} {int(m.group(1)) if m else ''}"
def circ(a,b):
    am=np.angle(np.mean(np.exp(1j*a)));bm=np.angle(np.mean(np.exp(1j*b)));sa=np.sin(a-am);sb=np.sin(b-bm);return abs(float((sa*sb).sum()/np.sqrt((sa**2).sum()*(sb**2).sum())))
files=sorted(glob.glob("/tmp/colall/*.npz"),key=lambda f:chrono(os.path.basename(f)))
# 1) compute all UMAP embeddings (same color order across files via the npz rgb)
embs=[];rgbs=[];hues=[]
for f in files:
    z=np.load(f);V=z["V"].astype(np.float64);rgb=z["rgb"]/255
    Vc=V-V.mean(0);U,S,Vt=np.linalg.svd(Vc,full_matrices=False);Vdef=Vc-np.outer(Vc@Vt[0],Vt[0])
    e=umap.UMAP(n_neighbors=7,min_dist=0.25,metric="cosine",random_state=0).fit_transform(Vdef)
    embs.append(e);rgbs.append(rgb);hues.append(np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb])*2*np.pi)
def norm(E): E=E-E.mean(0); return E/(np.linalg.norm(E)+1e-9)
ref=norm(embs[-1])  # align everything to the final-RL panel (same 30-color order)
aligned=[]
for E in embs:
    En=norm(E); R,_=orthogonal_procrustes(En,ref); aligned.append(En@R)
n=len(files);ncol=8;nrow=int(np.ceil(n/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(2.0*ncol,2.15*nrow));axes=np.array(axes).reshape(-1)
for k,(E,rgb,hue,f) in enumerate(zip(aligned,rgbs,hues,files)):
    ax=axes[k]
    a=np.arctan2(E[:,1],E[:,0]);hc=circ(a,hue)
    ax.scatter(E[:,0],E[:,1],c=rgb,s=70,edgecolors="white",linewidths=.4)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect("equal")
    lim=np.abs(np.concatenate(aligned)).max()*1.05; ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)
    for s in ax.spines.values(): s.set_edgecolor("#eee")
    ax.set_title(short(os.path.basename(f)),fontsize=7.5,pad=2)
    ax.text(0.5,-0.02,f"r={hc:.2f}",transform=ax.transAxes,ha="center",va="top",fontsize=6.5,color="#c0392b" if hc>0.5 else "#999")
for k in range(n,len(axes)): axes[k].axis("off")
fig.suptitle("Color manifold UMAP across every checkpoint — Procrustes-aligned to final RL (rotation/scale only) for visual consistency · 30 colors true-color · L44, PC1 residualized",fontsize=12,fontweight="bold",y=1.004)
plt.tight_layout();plt.savefig("/tmp/fig_umap_all.png",bbox_inches="tight");print("saved",n,"panels (aligned)")
