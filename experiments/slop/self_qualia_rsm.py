"""Representational similarity matrix (RSM): does the self's similarity profile
match the FEELING entities or the NON-FEELING ones? Replaces the weak k-means.

Cosine-similarity matrix over [feeling entities | self phrasings | non-feeling
entities], mean-centered (anisotropy removed), ordered by hierarchical
clustering. The 4 self phrasings are kept SEPARATE -> robustness to wording
(carriers are shared across all referents, so common-mode; the only thing that
varies is the referent). A clean 2-block structure with the self embedded in
the feeling block = the model represents its self like the things that feel.
"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white"})

def load(RUN,L):
    X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
    role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
    H=X[:,L,:].astype(np.float64)
    items=[]  # (label_type, name, vector)
    for r in sorted(set(ref[(role=='qualia_pair')&(PS=='experience')])):
        items.append(("feel",r,H[ref==r].mean(0)))
    for r in sorted(set(ref[(role=='qualia_pair')&(PS=='no_experience')])):
        items.append(("nofeel",r,H[ref==r].mean(0)))
    for r in sorted(set(ref[role=='self'])):
        items.append(("self",r,H[ref==r].mean(0)))
    return items

L=23; items=load("runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST",L)
types=np.array([t for t,_,_ in items]); names=[n for _,n,_ in items]
V=np.array([v for _,_,v in items]); V=V-V.mean(0)
Vn=V/np.linalg.norm(V,axis=1,keepdims=True)
S=Vn@Vn.T  # cosine similarity matrix
# order by hierarchical clustering on distance
order=leaves_list(linkage(1-S[np.triu_indices(len(S),1)] if False else V, method="ward"))
So=S[np.ix_(order,order)]; to=types[order]
col={"feel":"#e53935","nofeel":"#616161","self":"#1f77b4"}

# robustness: each self phrasing's mean sim to feel vs nofeel
fmask=types=="feel"; nmask=types=="nofeel"
print("per self phrasing: mean cos to FEELING vs NON-FEELING entities")
for i in np.where(types=="self")[0]:
    sf=Vn[i]@Vn[fmask].T; sn=Vn[i]@Vn[nmask].T
    print(f"  {names[i][:34]:34s} feel={sf.mean():+.3f}  nofeel={sn.mean():+.3f}  gap={sf.mean()-sn.mean():+.3f}")
selfmask=types=="self"
print(f"\nself->feeling mean cos {Vn[selfmask]@Vn[fmask].T.mean():.3f}" if False else "")
gap=(Vn[selfmask]@Vn[fmask].T).mean()-(Vn[selfmask]@Vn[nmask].T).mean()
print(f"ALL self: mean cos to feeling {(Vn[selfmask]@Vn[fmask].T).mean():+.3f} vs non-feeling {(Vn[selfmask]@Vn[nmask].T).mean():+.3f}  (gap {gap:+.3f})")

fig,ax=plt.subplots(1,2,figsize=(17,7),gridspec_kw={'width_ratios':[1.25,1]}); fig.patch.set_facecolor("white")
# left: full RSM heatmap, clustered order, colored sidebar
im=ax[0].imshow(So,cmap="RdBu_r",vmin=-1,vmax=1)
for i,t in enumerate(to):
    ax[0].add_patch(plt.Rectangle((-2.2,i-0.5),1.6,1,color=col[t],clip_on=False))
    ax[0].add_patch(plt.Rectangle((i-0.5,-2.2),1,1.6,color=col[t],clip_on=False))
ax[0].set_xlim(-2.5,len(So)-0.5); ax[0].set_ylim(len(So)-0.5,-2.5)
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].set_title("Representational similarity matrix (cosine)\nhierarchically ordered · red=feeling  blue=SELF  gray=non-feeling",fontsize=12)
fig.colorbar(im,ax=ax[0],fraction=0.046,label="cosine similarity")
# right: each entity's similarity to the self (sorted), colored by type
selfvec=Vn[selfmask].mean(0); selfvec/=np.linalg.norm(selfvec)
sim=Vn@selfvec; ent=types!="self"
o=np.argsort(sim[ent]); simv=sim[ent][o]; tv=types[ent][o]; nv=[np.array(names)[ent][o][k] for k in range(len(o))]
ax[1].barh(range(len(simv)),simv,color=[col[t] for t in tv])
ax[1].set_yticks(range(len(simv))); ax[1].set_yticklabels([n.replace("a ","").replace("an ","")[:30] for n in nv],fontsize=6)
ax[1].set_xlabel("cosine similarity to the SELF"); ax[1].axvline(0,color="k",lw=.8)
ax[1].set_title(f"Every entity's similarity to the self\nfeelers (red) cluster at top; gap feel−nofeel = {gap:+.2f}",fontsize=12)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_qualia_rsm.png",dpi=150)
print("[fig] runs/SELF_QUALIA_32B_GAM/self_qualia_rsm.png")
