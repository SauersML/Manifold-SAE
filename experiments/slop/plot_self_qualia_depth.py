import os; os.environ["RUST_LOG"]="off"; os.environ["GAM_LOG"]="off"
import numpy as np, gamfit, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white"})
def load(d): return np.load(f"{d}/activations.npy"), list(csv.DictReader(open(f"{d}/prompts.csv")))
def u(v):n=np.linalg.norm(v);return v/n if n>1e-12 else v*0
def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
    return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
OUT="runs/SELF_QUALIA_GAM"; C_SELF="#1f77b4"
RUNS={"base":"runs/OLMO3_7B_SELF_QUALIA_MAIN","instruct":"runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST"}
RAW={k:load(d) for k,d in RUNS.items()}

def per_layer(k,layer):
    X,rows=RAW[k]; H=X[:,layer,:].astype(np.float64)
    role=np.array([r['role'] for r in rows]);G=np.array([r['group'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
    mind=(role=='kind_anchor')&(G=='mind');mech=(role=='kind_anchor')&(G=='mechanism')
    exp=(role=='qualia_pair')&(PS=='experience');no=(role=='qualia_pair')&(PS=='no_experience')
    kax=u(H[mind].mean(0)-H[mech].mean(0));qax=u(H[exp].mean(0)-H[no].mean(0))
    ks,qs=H@kax,H@qax
    klo,khi=ks[mech].mean(),ks[mind].mean();qlo,qhi=qs[no].mean(),qs[exp].mean()
    refs=sorted(set(ref)); isS=np.array([role[ref==r][0]=='self' for r in refs])
    kc=np.array([(ks[ref==r].mean()-klo)/(khi-klo) for r in refs])
    qc=np.array([(qs[ref==r].mean()-qlo)/(qhi-qlo) for r in refs])
    kAUC=auc(np.r_[ks[mind],ks[mech]],np.r_[np.ones(mind.sum()),np.zeros(mech.sum())])
    qAUC=auc(np.r_[qs[exp],qs[no]],np.r_[np.ones(exp.sum()),np.zeros(no.sum())])
    sv=u(H[np.isin(ref,np.array(refs)[isS])].mean(0)) if isS.any() else None
    gap=np.dot(sv,u(H[(role=='landmark')&(G=='human_author')].mean(0)))-np.dot(sv,u(H[(role=='landmark')&(G=='ai_author')].mean(0)))
    return dict(kc=kc,qc=qc,isS=isS,kAUC=kAUC,qAUC=qAUC,self_k=kc[isS].mean(),self_q=qc[isS].mean(),gap=gap)

LYS=list(range(0,32))
prof={k:[per_layer(k,l) for l in LYS] for k in RUNS}

# ---- FIG 1: qualia ~ te(kind, layer) surface, base | instruct ----
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,k in zip(axes,["base","instruct"]):
    KZ=[];LY=[];QN=[]
    for l in range(0,32,2):
        P=prof[k][l];ent=~P['isS'];KZ+=list(P['kc'][ent]);LY+=[l]*ent.sum();QN+=list(P['qc'][ent])
    m=gamfit.fit({"kind":np.array(KZ),"layer":np.array(LY,float),"qualia":np.array(QN)},"qualia ~ te(kind, layer)")
    gk=np.linspace(0,1,50);gl=np.linspace(0,31,50);GK,GL=np.meshgrid(gk,gl)
    Z=np.asarray(m.predict({"kind":GK.ravel(),"layer":GL.ravel()})).reshape(GK.shape)
    cf=ax.contourf(GK,GL,Z,levels=18,cmap="viridis");fig.colorbar(cf,ax=ax,label="expected qualia")
    ax.set_xlabel("kind coordinate (0=mech,1=mind)");ax.set_ylabel("layer");ax.set_title(f"{k}: qualia ~ te(kind, layer)")
fig.suptitle("Depth surface #1 — expected QUALIA over (kind, depth): when the experience gradient forms",fontsize=13)
fig.tight_layout();fig.savefig(f"{OUT}/depth1_qualia_surface.png",dpi=150);plt.close(fig);print("[1] depth1_qualia_surface.png")

# ---- FIG 2: kind ~ te(qualia, layer) surface (symmetric), base | instruct ----
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,k in zip(axes,["base","instruct"]):
    QZ=[];LY=[];KN=[]
    for l in range(0,32,2):
        P=prof[k][l];ent=~P['isS'];QZ+=list(P['qc'][ent]);LY+=[l]*ent.sum();KN+=list(P['kc'][ent])
    m=gamfit.fit({"qualia":np.array(QZ),"layer":np.array(LY,float),"kind":np.array(KN)},"kind ~ te(qualia, layer)")
    gq=np.linspace(0,1,50);gl=np.linspace(0,31,50);GQ,GL=np.meshgrid(gq,gl)
    Z=np.asarray(m.predict({"qualia":GQ.ravel(),"layer":GL.ravel()})).reshape(GQ.shape)
    cf=ax.contourf(GQ,GL,Z,levels=18,cmap="magma");fig.colorbar(cf,ax=ax,label="expected kind")
    ax.set_xlabel("qualia coordinate (0=no-exp,1=exp)");ax.set_ylabel("layer");ax.set_title(f"{k}: kind ~ te(qualia, layer)")
fig.suptitle("Depth surface #2 — expected KIND over (qualia, depth)",fontsize=13)
fig.tight_layout();fig.savefig(f"{OUT}/depth2_kind_surface.png",dpi=150);plt.close(fig);print("[2] depth2_kind_surface.png")

# ---- FIG 3: axis quality (AUC) vs depth, base & instruct ----
fig,ax=plt.subplots(figsize=(10,6))
for k,ls in [("base","--"),("instruct","-")]:
    ax.plot(LYS,[prof[k][l]['kAUC'] for l in LYS],ls,color="#7e57c2",label=f"{k} kind AUC")
    ax.plot(LYS,[prof[k][l]['qAUC'] for l in LYS],ls,color="#e53935",label=f"{k} qualia AUC")
ax.axhline(0.5,color="0.7",lw=.8);ax.set_xlabel("layer (depth)");ax.set_ylabel("anchor separation AUC")
ax.set_title("Depth profile #3 — when the kind & qualia axes become decodable");ax.legend(fontsize=9);ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(f"{OUT}/depth3_axis_quality.png",dpi=150);plt.close(fig);print("[3] depth3_axis_quality.png")

# ---- FIG 4: self trajectory through depth (kind, qualia, human-AI gap) ----
fig,axes=plt.subplots(1,3,figsize=(18,5))
for k,ls in [("base","--"),("instruct","-")]:
    col="#5c6bc0" if k=="base" else "#9467bd"
    axes[0].plot(LYS,[prof[k][l]['self_k'] for l in LYS],ls,color=col,label=k)
    axes[1].plot(LYS,[prof[k][l]['self_q'] for l in LYS],ls,color=col,label=k)
    axes[2].plot(LYS,[prof[k][l]['gap'] for l in LYS],ls,color=col,label=k)
for a,t,yl in zip(axes,["self KIND coord vs depth","self QUALIA coord vs depth","self human−AI author gap vs depth"],
                  ["kind coord","qualia coord","cos(self,human)−cos(self,AI)"]):
    a.set_xlabel("layer");a.set_ylabel(yl);a.set_title(t);a.legend(fontsize=9);a.grid(alpha=.3)
axes[2].axhline(0,color="k",lw=.8)
fig.suptitle("Depth profile #4 — the indexical self's trajectory through the network",fontsize=13)
fig.tight_layout();fig.savefig(f"{OUT}/depth4_self_trajectory.png",dpi=150);plt.close(fig);print("[4] depth4_self_trajectory.png")
print("DONE")
