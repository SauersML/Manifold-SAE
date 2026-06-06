"""Clean 1-D comparison: feeling score of entities + the self under PCA / LDA / PLS."""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","font.family":"DejaVu Sans"})
FEEL="#13988a"; NON="#aab2bd"; SELF="#f6a13c"; INK="#2b2b2b"
RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=23
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);PID=np.array([r['pair_id'] for r in rows]);ref=np.array([r['referent'] for r in rows])
H=X[:,L,:].astype(np.float64)
fr=sorted(set(ref[(role=='qualia_pair')&(PS=='experience')])); nr=sorted(set(ref[(role=='qualia_pair')&(PS=='no_experience')]))
fc=np.array([H[ref==r].mean(0) for r in fr]); nc=np.array([H[ref==r].mean(0) for r in nr])
gid=np.array([PID[ref==r][0] for r in fr]+[PID[ref==r][0] for r in nr])
Z=np.vstack([fc,nc]); y=np.r_[np.ones(len(fc)),np.zeros(len(nc))]; selfv=H[role=='self'].mean(0)
mu=np.vstack([Z,selfv]).mean(0); Zr=PCA(20,random_state=0).fit(Zc:=Z-mu).transform(Zc); sr=PCA(20,random_state=0).fit(Zc).transform((selfv-mu)[None])[0]
def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
    return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
def cv(fit):
    s=np.zeros(len(y))
    for g in np.unique(gid): te=gid==g; s[te]=Zr[te]@fit(Zr[~te],y[~te])
    return s
def lda_ax(Zt,yt): w=LDA(solver="lsqr",shrinkage="auto").fit(Zt,yt).coef_[0]; return w/np.linalg.norm(w)
def pls_ax(Zt,yt): w=PLSRegression(1).fit(Zt,yt).coef_.ravel(); return w/np.linalg.norm(w)
def pca_ax(Zt,yt): c=PCA(1).fit(Zt).components_[0]; return c*np.sign(np.corrcoef(Zt@c,yt)[0,1])

methods=[("PCA","unsupervised",pca_ax),("LDA","supervised",lda_ax),("PLS","supervised",pls_ax)]
rng=np.random.default_rng(1)
fig,ax=plt.subplots(figsize=(11,4.8)); fig.patch.set_facecolor("white")
ymap={}
for k,(name,kind,fit) in enumerate(methods):
    w=fit(Zr,y); ent=Zr@w; sf=sr@w; cvs=cv(fit)
    # orient + standardize to σ from the non-feeling class
    m0,s0=ent[y==0].mean(),ent.std()
    e=(ent-m0)/s0; ss=(sf-m0)/s0; bound=((ent[y==1].mean()-m0)/s0)/2
    yc=len(methods)-1-k
    ymap[name]=yc
    jit=rng.uniform(-0.16,0.16,len(e))
    ax.scatter(e[y==1],yc+jit[y==1],s=55,c=FEEL,edgecolor="white",lw=.5,zorder=3)
    ax.scatter(e[y==0],yc+jit[y==0],s=55,c=NON,edgecolor="white",lw=.5,zorder=3)
    ax.scatter([ss],[yc],s=430,c=SELF,marker="*",edgecolor=INK,lw=1.2,zorder=5)
    ax.text(ss,yc+0.27,"self",ha="center",fontsize=9,color="#b9701b",fontweight="bold")
    ax.text(-0.02,yc,name,ha="right",va="center",fontsize=13,fontweight="bold",color=INK,transform=ax.get_yaxis_transform())
    ax.text(1.02,yc,f"AUC {auc(cvs,y):.2f}",ha="left",va="center",fontsize=10,color="#888",transform=ax.get_yaxis_transform())
ax.axvline(0,color=NON,lw=1,ls=":")
ax.set_yticks([]); ax.set_ylim(-0.6,len(methods)-0.4)
ax.autoscale(axis="x"); xl=ax.get_xlim(); ax.set_xlim(xl[0]-0.4,xl[1]+0.6)
for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
ax.set_xlabel("feeling score   (σ from the non-feeling class  →  more like the things that feel)",fontsize=11,color="#444")
ax.set_title("OLMo-3-32B: the self scores on the feeling side under every projection",fontsize=14,fontweight="bold",pad=10)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',mfc=FEEL,ms=9,label='feeling entity'),Line2D([0],[0],marker='o',color='w',mfc=NON,ms=9,label='non-feeling'),Line2D([0],[0],marker='*',color='w',mfc=SELF,mec=INK,ms=15,label='the self')],loc="lower right",fontsize=9,frameon=False)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_projection_methods.png",dpi=170,bbox_inches="tight")
print("saved")
