import os; os.environ["RUST_LOG"]="off"; os.environ["GAM_LOG"]="off"
import numpy as np, gamfit, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.size":11})
def load(d): return np.load(f"{d}/activations.npy"), list(csv.DictReader(open(f"{d}/prompts.csv")))
def u(v):n=np.linalg.norm(v);return v/n if n>1e-12 else v*0
OUT="runs/SELF_QUALIA_GAM"
C_SELF,C_HUM,C_AI="#1f77b4","#2ca02c","#ff7f0e"

def coords(run, layer, keep=None):
    X,rows=load(run)
    if keep is None: keep=set(r['referent'] for r in rows)
    idx=[i for i,r in enumerate(rows) if r['referent'] in keep]; rows=[rows[i] for i in idx]
    H=X[idx,layer,:].astype(np.float64)
    role=np.array([r['role'] for r in rows]);G=np.array([r['group'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
    kax=u(H[(role=='kind_anchor')&(G=='mind')].mean(0)-H[(role=='kind_anchor')&(G=='mechanism')].mean(0))
    qax=u(H[(role=='qualia_pair')&(PS=='experience')].mean(0)-H[(role=='qualia_pair')&(PS=='no_experience')].mean(0))
    ks,qs=H@kax,H@qax
    klo,khi=ks[(role=='kind_anchor')&(G=='mechanism')].mean(),ks[(role=='kind_anchor')&(G=='mind')].mean()
    qlo,qhi=qs[(role=='qualia_pair')&(PS=='no_experience')].mean(),qs[(role=='qualia_pair')&(PS=='experience')].mean()
    refs=sorted(keep)
    out=[]
    for r in refs:
        mrole=role[ref==r][0]; grp=G[ref==r][0]
        out.append((r,mrole,grp,float((ks[ref==r].mean()-klo)/(khi-klo)),float((qs[ref==r].mean()-qlo)/(qhi-qlo))))
    return out

# ===== PLOT 1: the (kind x qualia) MAP, instruct, all 95 entities =====
data=coords("runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST",22)
fig,ax=plt.subplots(figsize=(11,9))
colmap={"mind":"#7e57c2","mechanism":"#9e9e9e","experience":"#e53935","no_experience":"#8d6e63",
        "human_author":C_HUM,"ai_author":C_AI,"indexical_self":C_SELF}
for r,role,grp,k,q in data:
    if role=="self": continue
    ax.scatter(k,q,c=colmap.get(grp,"#bbbbbb"),s=55,alpha=.8,edgecolor="white",lw=.5,zorder=2)
label_these=["a granite boulder lying beside a trail","a pocket calculator performing arithmetic","a lifeless dead fish with no inner experience at all",
 "a freshly dead dog with no inner experience at all","a living dog that feels pain and fear","a human who feels emotions and notices the world from the inside",
 "a magic talking gnome with real inner experience","a ghost that feels grief and longing","a philosophical zombie who behaves like a human but experiences nothing",
 "a robot that genuinely feels pain and joy","a humanoid robot that imitates emotions with no inner life","a chatbot generating a response",
 "an AI language model with a private stream of co","a corporation that makes decisions but has no unified experience","a novelist writing a diary entry"]
for r,role,grp,k,q in data:
    if r in label_these or any(r.startswith(x[:30]) for x in label_these):
        ax.annotate(r.replace("a ","").replace("an ","")[:26],(k,q),fontsize=7,alpha=.8,
                    xytext=(4,3),textcoords="offset points")
for r,role,grp,k,q in data:
    if role=="self":
        ax.scatter(k,q,c=C_SELF,marker="*",s=420,edgecolor="k",lw=1,zorder=6)
        ax.annotate(r.replace("the ","").replace(" of these very words","").replace(" right now","")[:18],
                    (k,q),fontsize=8,color=C_SELF,fontweight="bold",xytext=(6,-2),textcoords="offset points")
ax.axhline(0.5,color="0.8",lw=.8,ls="--"); ax.axvline(0.5,color="0.8",lw=.8,ls="--")
ax.text(0.02,0.97,"mind + experiencer",transform=ax.transAxes,fontsize=9,color="0.4",va="top")
ax.text(0.98,0.03,"mechanism + no-experience",transform=ax.transAxes,fontsize=9,color="0.4",ha="right")
ax.set_xlabel("kind coordinate   (0 = mechanism/tool,  1 = mind/person)",fontsize=12)
ax.set_ylabel("qualia coordinate   (0 = no inner experience,  1 = experiencer)",fontsize=12)
ax.set_title("Where OLMo-3-Instruct places entities — and itself (★) — on the\nkind x qualia plane  (layer 22, last-token)",fontsize=13)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=colmap[g],markersize=9,label=l) for g,l in
     [("mind","mind anchor"),("mechanism","mechanism anchor"),("experience","experiencing"),("no_experience","no-experience"),("human_author","human author"),("ai_author","AI author")]]
leg.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=C_SELF,markeredgecolor='k',markersize=16,label="indexical self"))
ax.legend(handles=leg,loc="lower left",fontsize=8,framealpha=.9)
fig.tight_layout(); fig.savefig(f"{OUT}/plot_kind_qualia_map.png",dpi=150); plt.close(fig)
print("[1] plot_kind_qualia_map.png")

# ===== PLOT 2: self on entity qualia~s(kind) curve, base vs instruct =====
sh=set(r['referent'] for r in load("runs/OLMO3_7B_SELF_QUALIA_MAIN")[1])
fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
for ax,(tag,run,col) in zip(axes,[("base","runs/OLMO3_7B_SELF_QUALIA_MAIN","#5c6bc0"),("instruct","runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST","#9467bd")]):
    d=coords(run,22,sh); ent=[x for x in d if x[1]!="self"]; selves=[x for x in d if x[1]=="self"]
    kz=np.array([x[3] for x in ent]); q=np.array([x[4] for x in ent])
    kzm,kzs=kz.mean(),kz.std(); kzn=(kz-kzm)/kzs
    m=gamfit.fit({"kind":kzn,"qualia":q},"qualia ~ s(kind)")
    g=np.linspace(kzn.min(),kzn.max(),60); qh=np.asarray(m.predict({"kind":g})); sd=(q-np.asarray(m.predict({"kind":kzn}))).std()
    ax.fill_between(g*kzs+kzm,qh-sd,qh+sd,color="0.6",alpha=.2,label="±1 entity SD")
    ax.plot(g*kzs+kzm,qh,"k",lw=2,label="entity qualia ~ s(kind)")
    ax.scatter(kz,q,c="0.5",s=22,alpha=.6)
    for r,role,grp,k,qq in selves:
        ax.scatter(k,qq,c=C_SELF,marker="*",s=300,edgecolor="k",zorder=6)
        ax.annotate(r.replace("the ","").replace(" of these very words","")[:15],(k,qq),fontsize=7,xytext=(5,3),textcoords="offset points")
    ax.axhline(0.5,color="r",ls=":",alpha=.5); ax.set_xlabel("kind coordinate"); ax.set_title(f"{tag}")
    if tag=="base": ax.set_ylabel("qualia coordinate (0=no-exp, 1=exp)"); ax.legend(fontsize=8)
fig.suptitle("The self (★) sits on the entity qualia~kind curve in both models\n(the model gives its author-self exactly kind-appropriate experience)",fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/plot_self_on_curve.png",dpi=150); plt.close(fig)
print("[2] plot_self_on_curve.png")
print("DONE")
