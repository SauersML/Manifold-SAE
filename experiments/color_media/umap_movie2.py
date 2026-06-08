import glob,os,re,numpy as np,colorsys,warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.animation as animation; from scipy.linalg import orthogonal_procrustes; import umap
plt.rcParams.update({"font.family":"DejaVu Sans"})
BG="#0c0e14"; FG="#e8e8ee"
SC={"pretrain":"#5b8def","SFT":"#f2a34a","DPO":"#b06cf0","RL 3.0":"#2fd089","RL 3.1":"#19c6b0"}
def chrono(l):
    s=(0 if "stage1" in l else 1 if "stage2" in l else 2 if "stage3" in l else 3 if "SFT" in l else 4 if "DPO" in l else 6 if "RL31" in l else 5 if "_RL_" in l else 9)
    m=re.search(r'step_?(\d+)',l); return (s,int(m.group(1)) if m else 0)
def stage_of(l): return ("pretrain" if "stage" in l else "SFT" if "SFT" in l else "DPO" if "DPO" in l else "RL 3.1" if "RL31" in l else "RL 3.0")
def step_of(l): m=re.search(r'step_?(\d+)',l); return int(m.group(1)) if m else 0
files=sorted(glob.glob("/tmp/colall/*.npz"),key=lambda f:chrono(os.path.basename(f)))
embs=[];stg=[];stp=[]
for f in files:
    z=np.load(f);V=z["V"].astype(np.float64)
    Vc=V-V.mean(0);U,S,Vt=np.linalg.svd(Vc,full_matrices=False);Vdef=Vc-np.outer(Vc@Vt[0],Vt[0])
    embs.append(umap.UMAP(n_neighbors=7,min_dist=0.25,metric="cosine",random_state=0).fit_transform(Vdef))
    lab=os.path.basename(f);stg.append(stage_of(lab));stp.append(step_of(lab))
RGB=np.load(files[0])["rgb"]/255
def nrm(E): E=E-E.mean(0); return E/(np.linalg.norm(E)+1e-9)
# SEQUENTIAL alignment: each frame Procrustes-aligned to the PREVIOUS frame -> minimize frame-to-frame motion
norms=[nrm(E) for E in embs]; A=[norms[0]]
for En in norms[1:]:
    R,_=orthogonal_procrustes(En,A[-1]); A.append(En@R)
A=np.array(A)*8.0; lim=np.abs(A).max()*1.12
F=22;HOLD=7;frames=[];fstage=[];fstep=[];fci=[]
for i in range(len(A)):
    for _ in range(HOLD): frames.append(A[i]);fstage.append(stg[i]);fstep.append(stp[i]);fci.append(i)
    if i<len(A)-1:
        for fr in range(1,F+1):
            w=0.5-0.5*np.cos(np.pi*fr/F);frames.append((1-w)*A[i]+w*A[i+1]);j=i if w<0.5 else i+1;fstage.append(stg[j]);fstep.append(stp[j]);fci.append(i+w)
frames=np.array(frames);T=len(frames)
fig=plt.figure(figsize=(7.2,7.8),facecolor=BG);ax=fig.add_axes([0.04,0.10,0.92,0.80]);ax.set_facecolor(BG)
tax=fig.add_axes([0.08,0.045,0.84,0.022]);tax.set_xlim(0,len(A)-1);tax.set_ylim(0,1);tax.axis("off")
seg={}
for i,s in enumerate(stg): seg.setdefault(s,[i,i]); seg[s][1]=i
for s,(a,b) in seg.items(): tax.add_patch(plt.Rectangle((a,0),b-a+0.9,1,color=SC[s],alpha=.5,lw=0))
play=tax.scatter([0],[0.5],s=70,color="white",zorder=5,edgecolors=SC["pretrain"],lw=2)
title=fig.text(0.5,0.955,"",ha="center",va="center",fontsize=21,fontweight="bold",color=FG)
sub=fig.text(0.5,0.917,"",ha="center",va="center",fontsize=11,color="#9aa0b0")
fig.text(0.5,0.012,"OLMo-3-32B  ·  color manifold (UMAP, sequential-aligned)  ·  pretrain → SFT → DPO → RL",ha="center",fontsize=8.5,color="#666")
TR=9
def draw(t):
    ax.clear();ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect("equal");ax.axis("off");ax.set_facecolor(BG)
    for h in range(TR,0,-1):
        if t-h>=0: ax.scatter(frames[t-h][:,0],frames[t-h][:,1],c=RGB,s=260*(1-h/TR*0.6),alpha=0.10*(1-h/TR),edgecolors="none",zorder=1)
    P=frames[t]
    ax.scatter(P[:,0],P[:,1],c=RGB,s=900,alpha=0.16,edgecolors="none",zorder=2)
    ax.scatter(P[:,0],P[:,1],c=RGB,s=360,edgecolors="white",linewidths=1.1,zorder=3)
    s=fstage[t];title.set_text(s);title.set_color(SC[s]);sub.set_text(f"step {fstep[t]:,}")
    play.set_offsets([[fci[t],0.5]]);play.set_edgecolor(SC[s])
    return ()
ani=animation.FuncAnimation(fig,draw,frames=T,blit=False)
ani.save("/tmp/umap_evolution.mp4",writer=animation.FFMpegWriter(fps=30,bitrate=4200),dpi=120,savefig_kwargs={"facecolor":BG})
print("saved",T,"frames (sequential align)")
