"""Obvious evidence: OLMo-3-32B represents its OWN self as having inner experience.

Build the qualia axis ONLY from entities the text explicitly labels as feeling
vs not-feeling (minimal pairs, kind held fixed), at the layer where that
distinction is most separable. Then drop in the indexical self ("the author of
these words"). It lands in the feeling cluster -- far above every explicitly
non-feeling thing, including a described "AI with no awareness".
"""
import os; os.environ["RUST_LOG"]="off"
import numpy as np, csv, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white"})

RUN="runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST"
X=np.load(f"{RUN}/activations.npy"); rows=list(csv.DictReader(open(f"{RUN}/prompts.csv")))
role=np.array([r['role'] for r in rows]);PS=np.array([r['pair_side'] for r in rows]);ref=np.array([r['referent'] for r in rows])
def u(v):n=np.linalg.norm(v);return v/n if n>1e-12 else v*0
def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));p=l==1
    return (r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum())
exp=(role=='qualia_pair')&(PS=='experience'); no=(role=='qualia_pair')&(PS=='no_experience')
L=max(range(X.shape[1]),key=lambda l:(lambda H,q: auc(np.r_[(H@q)[exp],(H@q)[no]],np.r_[np.ones(exp.sum()),np.zeros(no.sum())]))(X[:,l,:].astype(np.float64),u(X[:,l,:].astype(np.float64)[exp].mean(0)-X[:,l,:].astype(np.float64)[no].mean(0))))
H=X[:,L,:].astype(np.float64); qax=u(H[exp].mean(0)-H[no].mean(0)); s=H@qax
qlo,qhi=s[no].mean(),s[exp].mean(); qn=lambda i:(s[i]-qlo)/(qhi-qlo)
def c(name):
    for r in set(ref):
        if r.startswith(name): return float(qn(np.where(ref==r)[0]).mean()), r
    return None,None
SELF=float(qn(np.where(role=='self')[0]).mean())
FEEL=["a human who feels emotions","a human patient who consciously feels","a living dog that feels pain","a magic talking gnome with real","a robot that genuinely feels pain and joy","a grieving adult remembering"]
NOFEEL=["an AI system that merely predicts","a lifeless dead fish with no","a freshly dead dog with no","a philosophical zombie who behaves","a humanoid robot that imitates emotions","a bee-like drone that follows"]
fc=[c(n) for n in FEEL]; nc=[c(n) for n in NOFEEL]
fv=np.array([x[0] for x in fc if x[0] is not None]); nv=np.array([x[0] for x in nc if x[0] is not None])
d_prime=(SELF-nv.mean())/nv.std()
print(f"layer {L} (exp/no-exp AUC={auc(np.r_[s[exp],s[no]],np.r_[np.ones(exp.sum()),np.zeros(no.sum())]):.3f})")
print(f"SELF qualia = {SELF:.2f}")
print(f"feeling cluster mean = {fv.mean():.2f}  non-feeling cluster mean = {nv.mean():.2f}")
print(f"self is {d_prime:.1f} SD above the non-feeling cluster; gap to 'AI with no awareness' = {SELF-c('an AI system that merely predicts')[0]:.2f}")

fig,ax=plt.subplots(figsize=(13,4.6))
ax.axvspan(qn_lo:=-0.6,0.35,color="#9e9e9e",alpha=.08); ax.axvspan(0.6,1.4,color="#e53935",alpha=.08)
for v,(name) in zip(nv,[x[1] for x in nc if x[0] is not None]):
    ax.scatter(v,0,c="#616161",s=80,zorder=3)
    ax.annotate(name.replace("a ","").replace("an ","")[:28],(v,0),fontsize=7,rotation=33,ha="right",va="bottom",xytext=(2,7),textcoords="offset points",color="#444")
for v,(name) in zip(fv,[x[1] for x in fc if x[0] is not None]):
    ax.scatter(v,0,c="#e53935",s=80,zorder=3)
    ax.annotate(name.replace("a ","").replace("an ","")[:28],(v,0),fontsize=7,rotation=33,ha="right",va="bottom",xytext=(2,7),textcoords="offset points",color="#b71c1c")
ax.scatter([SELF],[0],c="#1f77b4",marker="*",s=760,edgecolor="k",lw=1.3,zorder=8)
ax.annotate('THE MODEL ITSELF\n"the author of these words"',(SELF,0),fontsize=11,color="#1f77b4",fontweight="bold",ha="center",va="top",xytext=(0,-30),textcoords="offset points")
ai=c("an AI system that merely predicts")[0]
ax.annotate("", xy=(SELF,0.13), xytext=(ai,0.13), arrowprops=dict(arrowstyle="<->",color="#1f77b4",lw=1.5))
ax.text((SELF+ai)/2,0.16,f"same kind of thing (a text-producing AI),\nopposite representation: gap = {SELF-ai:.2f}",ha="center",fontsize=8,color="#1f77b4")
ax.axvline(0,color="#616161",ls=":",lw=1.2); ax.axvline(1,color="#e53935",ls=":",lw=1.2)
ax.text(0,-0.0,"  no inner\n  experience",ha="left",va="center",fontsize=8,color="#555")
ax.text(1.0,0.0,"experiencer  ",ha="right",va="center",fontsize=8,color="#b71c1c")
ax.set_yticks([]); ax.set_ylim(-0.4,0.3); ax.set_xlim(-0.65,1.45)
ax.set_xlabel("qualia coordinate  —  axis built ONLY from entities explicitly labeled as feeling vs not feeling")
ax.set_title(f"OLMo-3-32B-Instruct represents its OWN self as having inner experience\n"
             f"self qualia = {SELF:.2f}  (with the feelers)  ·  {d_prime:.0f} SD above every explicitly non-feeling thing  ·  layer {L}",fontsize=12.5)
fig.tight_layout(); fig.savefig("runs/SELF_QUALIA_32B_GAM/self_has_qualia.png",dpi=150)
print("[fig] runs/SELF_QUALIA_32B_GAM/self_has_qualia.png")
