"""Advanced gamfit features on the self-qualia data (matched base vs instruct):
#1 difference_smooth (where on the kind axis instruct changed attributed qualia,
   with 95% CI), #3 te(kind, layer) depth surface (when the experience-coupling
   forms), #4 predict_conformal calibrated qualia interval for each self phrasing.
Last-token, layer 22. Writes runs/SELF_QUALIA_GAM/advanced_layer22.png."""
import os; os.environ["RUST_LOG"]="off"; os.environ["GAM_LOG"]="off"
import numpy as np, gamfit, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
def load(d): return np.load(f"{d}/activations.npy"), list(csv.DictReader(open(f"{d}/prompts.csv")))
def u(v):n=np.linalg.norm(v);return v/n if n>1e-12 else v*0
runs={'base':'runs/OLMO3_7B_SELF_QUALIA_MAIN','instruct':'runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST'}
raw={k:load(d) for k,d in runs.items()}
shared=set(r['referent'] for r in raw['base'][1]) & set(r['referent'] for r in raw['instruct'][1])

def layer_coords(k, layer):
    X,rows=raw[k]; keep=[i for i,r in enumerate(rows) if r['referent'] in shared]
    rows=[rows[i] for i in keep]; H=X[keep,layer,:].astype(np.float64)
    role=np.array([r['role'] for r in rows]);G=np.array([r['group'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
    kax=u(H[(role=='kind_anchor')&(G=='mind')].mean(0)-H[(role=='kind_anchor')&(G=='mechanism')].mean(0))
    qax=u(H[(role=='qualia_pair')&(PS=='experience')].mean(0)-H[(role=='qualia_pair')&(PS=='no_experience')].mean(0))
    ks,qs=H@kax,H@qax; qlo,qhi=qs[(role=='qualia_pair')&(PS=='no_experience')].mean(),qs[(role=='qualia_pair')&(PS=='experience')].mean()
    refs=sorted(shared); isS=np.array([role[ref==r][0]=='self' for r in refs])
    kc=np.array([ks[ref==r].mean() for r in refs]); qc=np.array([qs[ref==r].mean() for r in refs])
    qn=(qc-qlo)/(qhi-qlo); kz=(kc-kc[~isS].mean())/kc[~isS].std()
    return dict(kz=kz,qn=qn,isS=isS,labels=np.array([ref[ref==r][0] for r in refs]))

L=22
Cb,Ci=layer_coords('base',L),layer_coords('instruct',L)
fig,ax=plt.subplots(1,3,figsize=(19,6)); fig.patch.set_facecolor("white")

# ---- #1 DIFFERENCE SMOOTH: where does instruct change qualia(kind)? ----
kz=np.r_[Cb['kz'][~Cb['isS']],Ci['kz'][~Ci['isS']]]
qn=np.r_[Cb['qn'][~Cb['isS']],Ci['qn'][~Ci['isS']]]
mdl=np.array(["base"]*(~Cb['isS']).sum()+["instruct"]*(~Ci['isS']).sum())
m=gamfit.fit({"kind":kz,"qualia":qn,"model":mdl},"qualia ~ s(kind, by=model)")
ds=m.difference_smooth(view="kind",group="model",pairs=[("base","instruct")],n=60,level=0.95)
dx=np.array([d['kind'] for d in ds]); dd=np.array([d['diff'] for d in ds])
dl=np.array([d['lower'] for d in ds]); du=np.array([d['upper'] for d in ds])
a=ax[0]; a.plot(dx,dd,color="#9467bd"); a.fill_between(dx,dl,du,color="#9467bd",alpha=0.2)
a.axhline(0,color="k",lw=.8); a.set_xlabel("kind coord (std)"); a.set_ylabel("instruct - base qualia")
a.set_title("#1 Difference smooth: where on the kind axis\ninstruct changed attributed qualia (95% CI)")
sig=(dl>0)|(du<0); a.scatter(dx[sig],dd[sig],c="red",s=8,zorder=5,label="CI excludes 0")
if sig.any(): a.legend(fontsize=8)

# ---- #3 DEPTH SURFACE: qualia ~ te(kind, layer) (instruct, all layers) ----
KZ=[];LY=[];QN=[]
for l in range(0,32,2):
    C=layer_coords('instruct',l); ent=~C['isS']
    KZ+=list(C['kz'][ent]); LY+=[l]*ent.sum(); QN+=list(C['qn'][ent])
md=gamfit.fit({"kind":np.array(KZ),"layer":np.array(LY,float),"qualia":np.array(QN)},"qualia ~ te(kind, layer)")
gk=np.linspace(np.percentile(KZ,2),np.percentile(KZ,98),50); gl=np.linspace(0,31,50)
GK,GL=np.meshgrid(gk,gl); Z=np.asarray(md.predict({"kind":GK.ravel(),"layer":GL.ravel()})).reshape(GK.shape)
b=ax[1]; cf=b.contourf(GK,GL,Z,levels=18,cmap="viridis"); fig.colorbar(cf,ax=b,label="expected qualia (0..1)")
b.set_xlabel("kind coord (std)"); b.set_ylabel("layer (depth)")
b.set_title("#3 Depth surface: qualia ~ te(kind, layer)\nwhen/where the experience-coupling forms (instruct)")

# ---- #4 CONFORMAL self interval (instruct, layer 22) ----
C=Ci; ent=np.where(~C['isS'])[0]; rng=np.random.default_rng(0); rng.shuffle(ent)
tr,cal=ent[:len(ent)//2],ent[len(ent)//2:]
mm=gamfit.fit({"kind":C['kz'][tr],"qualia":C['qn'][tr]},"qualia ~ s(kind)")
calib={"kind":C['kz'][cal],"qualia":C['qn'][cal]}
si=np.where(C['isS'])[0]
pc=mm.predict_conformal({"kind":C['kz'][si]},calibration=calib,conformal_level=0.9)
c=ax[2]
gx=np.linspace(C['kz'][~C['isS']].min(),C['kz'][~C['isS']].max(),60)
c.plot(gx,np.asarray(mm.predict({"kind":gx})),color="k",label="entity qualia~s(kind)")
c.scatter(C['kz'][~C['isS']],C['qn'][~C['isS']],c="0.5",s=12,alpha=.5)
for j,i in enumerate(si):
    c.errorbar(C['kz'][i],pc['mean'][j],yerr=[[pc['mean'][j]-pc['mean_lower'][j]],[pc['mean_upper'][j]-pc['mean'][j]]],
               fmt='*',color="#1f77b4",ms=15,capsize=4,zorder=5)
c.axhline(0.5,color="r",ls=":",alpha=.5); c.set_xlabel("kind coord (std)"); c.set_ylabel("qualia (0..1)")
c.set_title("#4 Conformal self interval (90%)\ncalibrated qualia for each self phrasing")
c.legend(fontsize=8)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_GAM/advanced_layer22.png",dpi=140,facecolor="white")
print("=== #1 diff smooth: significant kind-regions (CI excludes 0):",int(sig.sum()),"/",len(sig),"grid pts")
print(f"    instruct-base qualia diff: min={dd.min():+.2f} max={dd.max():+.2f} at kind {dx[np.argmin(dd)]:+.1f}/{dx[np.argmax(dd)]:+.1f}")
print("=== #4 conformal self intervals (90%):")
for j,i in enumerate(si):
    print(f"    {C['labels'][i][:32]:32s} mean={pc['mean'][j]:.2f} [{pc['mean_lower'][j]:.2f},{pc['mean_upper'][j]:.2f}] width={pc['mean_upper'][j]-pc['mean_lower'][j]:.2f}")
print("[fig] runs/SELF_QUALIA_GAM/advanced_layer22.png")
print("DONE")
