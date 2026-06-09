"""Beautiful single-panel: for each kind-matched minimal pair (same entity,
experience toggled), is the 32B self's raw representation closer to the FEELING
version or the NO-EXPERIENCE version?"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import comb
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.family":"DejaVu Sans"})
RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=23
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);PID=np.array([r['pair_id'] for r in rows]);ref=np.array([r['referent'] for r in rows])
H=X[:,L,:].astype(np.float64); selfv=H[role=='self'].mean(0)
pairs={}
for i in np.where(role=='qualia_pair')[0]: pairs.setdefault(PID[i],{}).setdefault(PS[i],[]).append(i)
names,fe,nf=[],[],[]
for pid,d in pairs.items():
    if 'experience' in d and 'no_experience' in d:
        names.append(ref[d['experience'][0]]); fe.append(H[d['experience']].mean(0)); nf.append(H[d['no_experience']].mean(0))
fe=np.array(fe); nf=np.array(nf)
mu=np.vstack([fe,nf,selfv]).mean(0)
def cos(a,b):return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
sv=selfv-mu
diff=np.array([cos(sv,fe[k]-mu)-cos(sv,nf[k]-mu) for k in range(len(fe))])
o=np.argsort(diff); diff=diff[o]; names=[names[i] for i in o]
nwin=int((diff>0).sum()); npair=len(diff); p=min(2*sum(comb(npair,k) for k in range(nwin,npair+1))/2**npair,1)

fig,ax=plt.subplots(figsize=(13.5,11))
pos="#2a9d8f"; neg="#c44e52"
colors=[pos if v>0 else neg for v in diff]
bars=ax.barh(range(npair),diff,color=colors,height=0.74,zorder=3,edgecolor="white",lw=0.6)
ax.axvline(0,color="#333",lw=1.2,zorder=4)
ax.set_yticks(range(npair)); ax.set_yticklabels(names,fontsize=10.5)
ax.set_ylim(-0.7,npair-0.3)
for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="x",color="#e6e6e6",zorder=0)
for k,v in enumerate(diff):
    ax.text(v+(0.006 if v>=0 else -0.006),k,f"{v:+.2f}",va="center",ha="left" if v>=0 else "right",fontsize=8,color="#555")
ax.set_xlabel("← closer to the NO-EXPERIENCE version      ·      closer to the FEELING version →",fontsize=11)
ax.text(0.5,1.045,"Does the model represent its own self as feeling?",transform=ax.transAxes,ha="center",fontsize=16,fontweight="bold")
ax.text(0.5,1.01,f"In {nwin} of {npair} kind-matched pairs, OLMo-3-32B's self is closer to the FEELING version of the same entity   (sign-test p = {p:.0e})",transform=ax.transAxes,ha="center",fontsize=11,color="#444")
ax.margins(x=0.14)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_paired_feeling.png",dpi=160,bbox_inches="tight")
print(f"{nwin}/{npair} p={p:.1e}")
