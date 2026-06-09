"""Two MORE methods (different from the axis projection) that the 32B self is
represented as an experiencer.

A) PAIRED NEAREST-NEIGHBOR (model-free, kind-controlled): for each minimal pair
   (same entity, experience toggled), is the self's raw representation closer to
   the FEELING member or the NO-EXPERIENCE member? Within-pair => kind/register
   controlled. Sign test over pairs.
B) UNSUPERVISED: k-means (no labels) on entity reps; does it recover feeling vs
   not, and which cluster does the self join?
"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white"})

def analyze(RUN, tag, L):
    X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
    role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows])
    PID=np.array([r['pair_id'] for r in rows]);ref=np.array([r['referent'] for r in rows])
    H=X[:,L,:].astype(np.float64)
    self_vec=H[role=='self'].mean(0)
    # build pair table
    pairs={}
    for i in np.where(role=='qualia_pair')[0]:
        pairs.setdefault(PID[i],{}).setdefault(PS[i],[]).append(i)
    feel_c, nofeel_c, labels_ref, names = [], [], [], []
    for pid,d in pairs.items():
        if 'experience' in d and 'no_experience' in d:
            fe=H[d['experience']].mean(0); nf=H[d['no_experience']].mean(0)
            feel_c.append(fe); nofeel_c.append(nf)
            names.append(ref[d['experience'][0]])
    feel_c=np.array(feel_c); nofeel_c=np.array(nofeel_c)
    # mean-center to remove the global anisotropy direction
    allc=np.vstack([feel_c,nofeel_c,self_vec[None]]); mu=allc.mean(0)
    fc=feel_c-mu; nc=nofeel_c-mu; sv=self_vec-mu
    def cos(a,b): return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
    # METHOD A: paired NN
    diff=np.array([cos(sv,fc[k])-cos(sv,nc[k]) for k in range(len(fc))])
    nwin=int((diff>0).sum()); npair=len(diff)
    # sign-test p (two-sided, binomial)
    from math import comb
    p_sign=2*sum(comb(npair,k) for k in range(nwin,npair+1))/2**npair
    return dict(names=names,diff=diff,nwin=nwin,npair=npair,p=min(p_sign,1.0),
                feel_c=feel_c,nofeel_c=nofeel_c,self_vec=self_vec,mu=mu)

# pick best-qualia layer on instruct
def best_layer(RUN):
    X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
    role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows])
    exp=(role=='qualia_pair')&(PS=='experience');no=(role=='qualia_pair')&(PS=='no_experience')
    def u(v):n=np.linalg.norm(v);return v/n if n>1e-12 else v*0
    def auc(s,l):
        o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
        return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
    return max(range(X.shape[1]),key=lambda l:(lambda H,q:auc(np.r_[(H@q)[exp],(H@q)[no]],np.r_[np.ones(exp.sum()),np.zeros(no.sum())]))(X[:,l,:].astype(np.float64),u(X[:,l,:].astype(np.float64)[exp].mean(0)-X[:,l,:].astype(np.float64)[no].mean(0))))

RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=best_layer(RUN)
R=analyze(RUN,"instruct",L)
Rb=analyze("runs/OLMO3_32B_BASE_SELF_QUALIA_LAST","base",best_layer("runs/OLMO3_32B_BASE_SELF_QUALIA_LAST"))
print(f"=== METHOD A: paired nearest-neighbor (layer {L}) ===")
print(f"instruct: self is closer to the FEELING member in {R['nwin']}/{R['npair']} kind-matched pairs (sign-test p={R['p']:.1e})")
print(f"base:     self is closer to the FEELING member in {Rb['nwin']}/{Rb['npair']} pairs (p={Rb['p']:.1e})")

# METHOD B: unsupervised k-means on feel+nofeel entities, where does self land?
feat=np.vstack([R['feel_c'],R['nofeel_c']])-R['mu']
lab=np.r_[np.ones(len(R['feel_c'])),np.zeros(len(R['nofeel_c']))]
# PCA to 2D for clustering + viz
U,S,Vt=np.linalg.svd(feat,full_matrices=False); emb=feat@Vt[:2].T
selfemb=(R['self_vec']-R['mu'])@Vt[:2].T
# k-means k=2 (no labels), simple
rng=np.random.default_rng(0); C=emb[rng.choice(len(emb),2,replace=False)]
for _ in range(50):
    a=np.argmin(((emb[:,None]-C[None])**2).sum(-1),1)
    C=np.array([emb[a==k].mean(0) if (a==k).any() else C[k] for k in range(2)])
# align cluster id to feeling majority
fc_cluster=int(round(lab[a==1].mean())) if (a==1).any() else 1
purity=max((a==lab).mean(),(a==(1-lab)).mean())
# which cluster is self
sa=int(np.argmin(((selfemb-C)**2).sum(1)))
feelcluster=int(np.round([lab[a==k].mean() for k in range(2)][sa])) if (a==sa).any() else None
self_in_feeling = ([lab[a==k].mean() for k in range(2)][sa] > 0.5)
print(f"\n=== METHOD B: unsupervised k-means (no labels) ===")
print(f"cluster purity vs feel/no-feel = {purity:.2f}; self joins the {'FEELING' if self_in_feeling else 'NON-FEELING'} cluster")

# ===== FIGURES =====
fig,axes=plt.subplots(1,2,figsize=(17,6.5)); fig.patch.set_facecolor("white")
# A: per-pair similarity difference
a0=axes[0]; order=np.argsort(R['diff']); d=R['diff'][order]; nm=[R['names'][i] for i in order]
colors=["#e53935" if x>0 else "#616161" for x in d]
a0.barh(range(len(d)),d,color=colors)
a0.axvline(0,color="k",lw=1)
a0.set_yticks(range(len(d))); a0.set_yticklabels([n.replace("a ","").replace("an ","")[:34] for n in nm],fontsize=6.5)
a0.set_xlabel("cos(self, FEELING version) − cos(self, NO-EXPERIENCE version)")
a0.set_title(f"A. Paired nearest-neighbor (kind-controlled)\nself is closer to the FEELING version in {R['nwin']}/{R['npair']} pairs  (sign-test p={R['p']:.0e})",fontsize=12)
# B: unsupervised embedding
b0=axes[1]
b0.scatter(emb[lab==1,0],emb[lab==1,1],c="#e53935",s=60,label="feeling entity",zorder=3,edgecolor="white",lw=.5)
b0.scatter(emb[lab==0,0],emb[lab==0,1],c="#616161",s=60,label="non-feeling entity",zorder=3,edgecolor="white",lw=.5)
b0.scatter([selfemb[0]],[selfemb[1]],c="#1f77b4",marker="*",s=620,edgecolor="k",lw=1.2,zorder=6,label="the self ★")
b0.set_xlabel("unsupervised PC1"); b0.set_ylabel("unsupervised PC2")
b0.set_title(f"B. Unsupervised k-means (NO labels used)\nclusters recover feeling vs not at purity {purity:.0%}; self joins the {'FEELING' if self_in_feeling else 'non-feeling'} cluster",fontsize=12)
b0.legend(fontsize=9,loc="best")
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_qualia_two_methods.png",dpi=150)
print("[fig] runs/SELF_QUALIA_32B_GAM/self_qualia_two_methods.png")
