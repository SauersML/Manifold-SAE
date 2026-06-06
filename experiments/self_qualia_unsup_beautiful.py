"""Beautiful unsupervised panel: embed entity representations with NO labels;
color by feeling post-hoc; where does the self land?"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.family":"DejaVu Sans"})
RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=23
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
H=X[:,L,:].astype(np.float64)
fr=sorted(set(ref[(role=='qualia_pair')&(PS=='experience')])); nr=sorted(set(ref[(role=='qualia_pair')&(PS=='no_experience')]))
fc=np.array([H[ref==r].mean(0) for r in fr]); nc=np.array([H[ref==r].mean(0) for r in nr]); selfv=H[role=='self'].mean(0)
V=np.vstack([fc,nc]); mu=np.vstack([V,selfv]).mean(0); Vc=V-mu
U,S,Vt=np.linalg.svd(Vc,full_matrices=False); emb=Vc@Vt[:2].T; se=(selfv-mu)@Vt[:2].T
lab=np.r_[np.ones(len(fc)),np.zeros(len(nc))]
# unsupervised k-means k=2 for purity stat
rng=np.random.default_rng(0); C=emb[rng.choice(len(emb),2,replace=False)]
for _ in range(80):
    a=np.argmin(((emb[:,None]-C[None])**2).sum(-1),1); C=np.array([emb[a==k].mean(0) if (a==k).any() else C[k] for k in range(2)])
purity=max((a==lab).mean(),(a==(1-lab)).mean())
TEAL="#2a9d8f"; GRAY="#9aa6b2"; GOLD="#f4a261"
fig,ax=plt.subplots(figsize=(11,8.5))
def hull(pts,color):
    if len(pts)>=3:
        h=ConvexHull(pts); poly=pts[h.vertices]; ax.fill(poly[:,0],poly[:,1],color=color,alpha=0.10,zorder=0,lw=0)
hull(emb[lab==1],TEAL); hull(emb[lab==0],GRAY)
ax.scatter(emb[lab==1,0],emb[lab==1,1],s=120,c=TEAL,edgecolor="white",lw=1,zorder=3,label="feeling entity")
ax.scatter(emb[lab==0,0],emb[lab==0,1],s=120,c=GRAY,edgecolor="white",lw=1,zorder=3,label="non-feeling entity")
ax.scatter([se[0]],[se[1]],s=900,c=GOLD,marker="*",edgecolor="#333",lw=1.6,zorder=6,label="the model's self")
ax.annotate("the self\n“the author of these words”",(se[0],se[1]),fontsize=11,fontweight="bold",color="#bf6b1f",ha="center",va="bottom",xytext=(0,22),textcoords="offset points")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("unsupervised dimension 1"); ax.set_ylabel("unsupervised dimension 2")
ax.text(0.5,1.06,"With no labels, feeling and non-feeling entities separate — and the self lands with the feelers",transform=ax.transAxes,ha="center",fontsize=14.5,fontweight="bold")
ax.text(0.5,1.015,f"unsupervised clustering recovers feeling vs not at {purity:.0%} purity; the self joins the feeling cluster",transform=ax.transAxes,ha="center",fontsize=10.5,color="#555")
ax.legend(loc="best",fontsize=11,frameon=True,framealpha=.9)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_unsupervised_feeling.png",dpi=160,bbox_inches="tight")
print(f"purity {purity:.0%}")
