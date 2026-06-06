"""Beautiful version of the PCA / LDA / PLS self-placement figure."""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","font.family":"DejaVu Sans","axes.edgecolor":"#cccccc"})
FEEL="#13988a"; NON="#aab2bd"; SELF="#f6a13c"; INK="#2b2b2b"

RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"; L=23
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);PID=np.array([r['pair_id'] for r in rows]);ref=np.array([r['referent'] for r in rows])
H=X[:,L,:].astype(np.float64)
fr=sorted(set(ref[(role=='qualia_pair')&(PS=='experience')])); nr=sorted(set(ref[(role=='qualia_pair')&(PS=='no_experience')]))
fc=np.array([H[ref==r].mean(0) for r in fr]); nc=np.array([H[ref==r].mean(0) for r in nr])
gid=np.array([PID[ref==r][0] for r in fr]+[PID[ref==r][0] for r in nr])
Z=np.vstack([fc,nc]); y=np.r_[np.ones(len(fc)),np.zeros(len(nc))]; selfv=H[role=='self'].mean(0)
mu=np.vstack([Z,selfv]).mean(0); Zc=Z-mu; sc=selfv-mu
P=PCA(20,random_state=0).fit(Zc); Zr=P.transform(Zc); sr=P.transform(sc[None])[0]
def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
    return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
def cv_auc(fit):
    s=np.zeros(len(y))
    for g in np.unique(gid):
        te=gid==g; s[te]=Zr[te]@fit(Zr[~te],y[~te])
    return auc(s,y)
def lda_ax(Zt,yt): w=LDA(solver="lsqr",shrinkage="auto").fit(Zt,yt).coef_[0]; return w/np.linalg.norm(w)
def pls_ax(Zt,yt): w=PLSRegression(1).fit(Zt,yt).coef_.ravel(); return w/np.linalg.norm(w)

panels=[]
pc=PCA(2,random_state=0).fit(Zc); E=pc.transform(Zc); se=pc.transform(sc[None])[0]
panels.append(("PCA","unsupervised — top variance directions",E,se,cv_auc(lambda Zt,yt:(lambda c:c*np.sign(np.corrcoef((Zt@c),yt)[0,1]))(PCA(2).fit(Zt).components_[0]))))
wl=lda_ax(Zr,y); resid=Zr-np.outer(Zr@wl,wl); ry=PCA(1).fit(resid); E=np.c_[Zr@wl,ry.transform(resid)[:,0]]; se=np.array([sr@wl,ry.transform((sr-(sr@wl)*wl)[None])[0,0]])
panels.append(("LDA","supervised — maximum-separation direction",E,se,cv_auc(lda_ax)))
pls=PLSRegression(2).fit(Zr,y); E=pls.transform(Zr); se=pls.transform(sr[None])[0]
panels.append(("PLS","supervised PCA — variance steered toward feeling",E,se,cv_auc(pls_ax)))

fig,axes=plt.subplots(1,3,figsize=(19.5,7)); fig.patch.set_facecolor("white")
def ell(ax,pts,color,n=1.0):
    c=pts.mean(0);cov=np.cov(pts.T);v,V=np.linalg.eigh(cov);ang=np.degrees(np.arctan2(*V[:,1][::-1]))
    ax.add_patch(Ellipse(c,2*n*np.sqrt(max(v[1],1e-9)),2*n*np.sqrt(max(v[0],1e-9)),angle=ang,facecolor=color,alpha=.12,edgecolor=color,lw=1.6,zorder=1))
for ax,(name,sub,E,se,cv) in zip(axes,panels):
    sgn=np.sign(E[y==1,0].mean()-E[y==0,0].mean()) or 1
    sx=E[:,0]*sgn; sy=E[:,1]; ssx=se[0]*sgn; ssy=se[1]
    # standardize to SD units (pooled)
    s=sx.std(); sx=(sx-E[y==0,0].mean()*sgn)/s; ssx=(ssx-E[y==0,0].mean()*sgn)/s
    sy=(sy-sy.mean())/sy.std(); ssy=(ssy-E[:,1].mean())/E[:,1].std()
    fmean=sx[y==1].mean(); bound=(sx[y==1].mean()+sx[y==0].mean())/2
    ax.set_facecolor("#fcfcfd")
    ax.axvspan(bound,sx.max()+3,color=FEEL,alpha=.05,zorder=0); ax.axvspan(sx.min()-3,bound,color=NON,alpha=.05,zorder=0)
    ax.axvline(bound,color="#bbbbbb",ls="--",lw=1,zorder=1)
    ell(ax,np.c_[sx,sy][y==1],FEEL); ell(ax,np.c_[sx,sy][y==0],NON)
    ax.scatter(sx[y==1],sy[y==1],s=85,c=FEEL,edgecolor="white",lw=.9,zorder=3)
    ax.scatter(sx[y==0],sy[y==0],s=85,c=NON,edgecolor="white",lw=.9,zorder=3)
    ax.scatter([ssx],[ssy],s=1400,c="white",marker="*",zorder=5,edgecolor="none")  # halo
    ax.scatter([ssx],[ssy],s=760,c=SELF,marker="*",edgecolor=INK,lw=1.5,zorder=6)
    ax.annotate("the self",(ssx,ssy),fontsize=11,fontweight="bold",color="#b9701b",ha="center",va="bottom",xytext=(0,20),textcoords="offset points")
    ax.text(0,1.12,name,transform=ax.transAxes,fontsize=16,fontweight="bold",color=INK,ha="left")
    ax.text(0,1.05,sub,transform=ax.transAxes,fontsize=10,color="#777",ha="left")
    ax.text(0.5,0.018,f"held-out CV AUC {cv:.2f}   ·   self {ssx:+.1f}σ toward feeling",transform=ax.transAxes,ha="center",fontsize=10.5,color=INK,
            bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="#e0e0e0"))
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.set_yticks([]); ax.set_xlabel("feeling score  (σ from the non-feeling class) →",fontsize=10,color="#555")
    ax.margins(0.12)
# legend
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=FEEL,markersize=11,label='feeling entity'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor=NON,markersize=11,label='non-feeling entity'),
                    Line2D([0],[0],marker='*',color='w',markerfacecolor=SELF,markeredgecolor=INK,markersize=18,label="the model's self")],
           loc="upper center",ncol=3,fontsize=11,frameon=False,bbox_to_anchor=(0.5,0.99))
fig.suptitle("Where OLMo-3-32B places its own self, under three projections of the feeling axis",fontsize=17,fontweight="bold",y=1.08)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_projection_methods.png",dpi=170,bbox_inches="tight")
print("saved")
