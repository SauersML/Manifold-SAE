"""Place the 32B self relative to feeling/non-feeling entities under THREE
projection methods, rigorously:
  1. PCA            - unsupervised (top variance directions; NOT toward feeling)
  2. LDA            - supervised: the single max-separation (Fisher) direction
  3. PLS            - supervised PCA: ordered directions maximizing covariance
                      with the feeling label ("variance toward your thing")
No convex hulls (use Gaussian 1-sigma covariance ellipses). No purity (use
leave-pairs-out CV AUC + the self's standardized feeling-side position).
The supervised axes are fit on ENTITIES only; the self is projected out-of-sample.
"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.family":"DejaVu Sans"})
TEAL="#2a9d8f"; GRAY="#9aa6b2"; GOLD="#f4a261"

RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=23
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);PID=np.array([r['pair_id'] for r in rows]);ref=np.array([r['referent'] for r in rows])
H=X[:,L,:].astype(np.float64)
fr=sorted(set(ref[(role=='qualia_pair')&(PS=='experience')])); nr=sorted(set(ref[(role=='qualia_pair')&(PS=='no_experience')]))
def pid_of(r): return PID[ref==r][0]
fc=np.array([H[ref==r].mean(0) for r in fr]); nc=np.array([H[ref==r].mean(0) for r in nr])
gid=np.array([pid_of(r) for r in fr]+[pid_of(r) for r in nr])
Z=np.vstack([fc,nc]); y=np.r_[np.ones(len(fc)),np.zeros(len(nc))]
selfv=H[role=='self'].mean(0)
mu=np.vstack([Z,selfv]).mean(0); Zc=Z-mu; sc=selfv-mu
# pre-reduce to a safe PCA subspace for the supervised fits (n<<d)
P=PCA(n_components=20,random_state=0).fit(Zc); Zr=P.transform(Zc); sr=P.transform(sc[None])[0]

def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
    return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
def cv_auc(fit_axis):
    """leave-PAIRS-out CV: fit axis on train entities, score held-out."""
    scores=np.zeros(len(y))
    for g in np.unique(gid):
        te=gid==g; tr=~te
        w=fit_axis(Zr[tr],y[tr]); scores[te]=Zr[te]@w
    return auc(scores,y)
def lda_axis(Ztr,ytr):
    m=LDA(solver="lsqr",shrinkage="auto").fit(Ztr,ytr); w=m.coef_[0]; return w/np.linalg.norm(w)
def pls_axis(Ztr,ytr):
    m=PLSRegression(n_components=1).fit(Ztr,ytr); w=m.coef_.ravel(); return w/np.linalg.norm(w)

# ---- build the three 2D embeddings (fit on entities; project self) ----
def ellipse(ax,pts,color):
    c=pts.mean(0); cov=np.cov(pts.T); vals,vecs=np.linalg.eigh(cov); ang=np.degrees(np.arctan2(*vecs[:,1][::-1]))
    e=Ellipse(c,2*np.sqrt(max(vals[1],1e-9)),2*np.sqrt(max(vals[0],1e-9)),angle=ang,facecolor=color,alpha=.13,edgecolor=color,lw=1.4,zorder=0); ax.add_patch(e)

methods={}
# PCA: top2
pc=PCA(n_components=2,random_state=0).fit(Zc); methods["1. PCA  (unsupervised — total variance)"]=(pc.transform(Zc),pc.transform(sc[None])[0],auc(pc.transform(Zc)[:,0]*np.sign(np.corrcoef(pc.transform(Zc)[:,0],y)[0,1]),y),cv_auc(lambda Zt,yt:PCA(2).fit(Zt).components_[0]*np.sign(np.corrcoef(PCA(2).fit(Zt).transform(Zt)[:,0],yt)[0,1])))
# LDA axis x, orthogonal residual variance y
wl=lda_axis(Zr,y); x=Zr@wl; resid=Zr-np.outer(Zr@wl,wl); yax=PCA(1).fit(resid).transform(resid)[:,0]
methods["2. LDA  (supervised — max separation)"]=(np.c_[x,yax],np.array([sr@wl,(sr-(sr@wl)*wl)@PCA(1).fit(resid).components_[0]]),auc(x,y),cv_auc(lda_axis))
# PLS comp1 x, comp2 y
pls=PLSRegression(n_components=2).fit(Zr,y); T=pls.transform(Zr); st=pls.transform(sr[None])[0]
methods["3. PLS  (supervised PCA — variance toward feeling)"]=(T,st,auc(T[:,0],y),cv_auc(pls_axis))

fig,axes=plt.subplots(1,3,figsize=(19,6.6)); fig.patch.set_facecolor("white")
for ax,(title,(E,se,trainauc,cvauc)) in zip(axes,methods.items()):
    # orient x so feeling is positive
    sgn=np.sign(E[y==1,0].mean()-E[y==0,0].mean()) or 1; E=E.copy(); E[:,0]*=sgn; se=se.copy(); se[0]*=sgn
    ellipse(ax,E[y==1],TEAL); ellipse(ax,E[y==0],GRAY)
    ax.scatter(E[y==1,0],E[y==1,1],s=90,c=TEAL,edgecolor="white",lw=.8,zorder=3,label="feeling")
    ax.scatter(E[y==0,0],E[y==0,1],s=90,c=GRAY,edgecolor="white",lw=.8,zorder=3,label="non-feeling")
    ax.scatter([se[0]],[se[1]],s=620,c=GOLD,marker="*",edgecolor="#333",lw=1.4,zorder=6,label="self")
    # self standardized position on x relative to the two classes
    z_non=(se[0]-E[y==0,0].mean())/E[y==0,0].std(); 
    ax.set_title(f"{title}\nheld-out CV AUC = {cvauc:.2f}   ·   self = {z_non:+.1f} SD toward feeling",fontsize=10.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("→ feeling")
    if title.startswith("1"): ax.legend(loc="best",fontsize=9)
fig.suptitle("Where the 32B self lands under unsupervised vs supervised projections (ellipses = 1σ covariance)",fontsize=14,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_projection_methods.png",dpi=160,bbox_inches="tight")
for t,(_,se,ta,ca) in methods.items(): print(f"{t[:6]} held-out CV AUC={ca:.3f}")
print("saved")
