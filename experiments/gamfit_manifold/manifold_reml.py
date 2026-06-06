import warnings, json, os; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import colorsys
import gamfit
print("gamfit", gamfit.__version__, flush=True)

base="runs/OLMO3_32B_TRAJ_RL31/step_2300/extra"
acts=np.load(base+"/activations.npy")          # (180,64,5120)
rows=[json.loads(l) for l in open(base+"/prompts.jsonl")]
L=44
X=acts[:,L,:].astype(np.float64)               # (180,5120)
colors=[r["color"] for r in rows]
frames=np.array([r["frame"] for r in rows])
rgb=np.array([r["rgb"] for r in rows],float)/255.0

uniq=sorted(set(colors))                        # 30 colors (alphabetical)
cidx={c:i for i,c in enumerate(uniq)}
ci=np.array([cidx[c] for c in colors])

# ---- frame-demean: subtract per-frame mean across colors, then avg 6 frames per color ----
Xd=X.copy()
for f in sorted(set(frames)):
    m=frames==f
    Xd[m]=Xd[m]-Xd[m].mean(axis=0,keepdims=True)
# average frames per color -> 30 vectors
C=np.zeros((len(uniq),X.shape[1]))
for c,i in cidx.items():
    C[i]=Xd[ci==i].mean(axis=0)
# representative RGB per color (frames share same rgb per color)
RGB=np.zeros((len(uniq),3))
for c,i in cidx.items():
    RGB[i]=rgb[ci==i][0]

# ---- reduce reps to top rep-PCs (response columns) ----
Cc=C-C.mean(axis=0,keepdims=True)
U,S,Vt=np.linalg.svd(Cc,full_matrices=False)
nPC=5
PCs=(U[:,:nPC]*S[:nPC])                          # (30, nPC) scores
ev=(S**2)/np.sum(S**2)
print("rep-PC explained var ratio (top8):", np.round(ev[:8],4))
# standardize each PC response to unit variance for comparable REML across responses
PCs=PCs/PCs.std(axis=0,keepdims=True)

# ---- latent coordinates ----
# HSV hue/sat/val
hsv=np.array([colorsys.rgb_to_hsv(*RGB[i]) for i in range(len(uniq))])
hue=hsv[:,0]; sat=hsv[:,1]; val=hsv[:,2]
light=RGB.mean(axis=1)
# classical MDS on rep distances (frame-demeaned color vectors)
D=np.linalg.norm(C[:,None,:]-C[None,:,:],axis=2)
n=len(uniq); J=np.eye(n)-np.ones((n,n))/n
B=-0.5*J@(D**2)@J
w,V=np.linalg.eigh(B); order=np.argsort(w)[::-1]
w=w[order]; V=V[:,order]
mds=V*np.sqrt(np.clip(w,0,None))
mds1,mds2,mds3=mds[:,0],mds[:,1],mds[:,2]
# 2D sheet coords from hue->cos/sin (circle embed) collapsed via sat-weighting -> use (mds1,mds2) too
hx=np.cos(2*np.pi*hue); hy=np.sin(2*np.pi*hue)

df=pd.DataFrame(dict(hue=hue,sat=sat,val=val,light=light,
                     R=RGB[:,0],G=RGB[:,1],B=RGB[:,2],
                     mds1=mds1,mds2=mds2,mds3=mds3,hx=hx,hy=hy))
for k in range(nPC):
    df[f"pc{k}"]=PCs[:,k]

# normalize MDS coords to [0,1] for stable splines
for c in ["mds1","mds2","mds3"]:
    df[c]=(df[c]-df[c].min())/(df[c].max()-df[c].min()+1e-12)

candidates={
 "CIRCLE_hue_cyclic":      "{y} ~ s(hue, bs='cc')",
 "LINE_lightness":         "{y} ~ s(light)",
 "LINE_mds1":              "{y} ~ s(mds1)",
 "SHEET_te_hue_sat":       "{y} ~ te(hx,hy)",
 "SHEET_te_mds12":         "{y} ~ te(mds1,mds2)",
 "VOLUME_te_RGB":          "{y} ~ te(R,G,B)",
 "VOLUME_te_mds123":       "{y} ~ te(mds1,mds2,mds3)",
}

results=[]
percol=[]
for name,ftpl in candidates.items():
    tot_reml=0.0; tot_edf=0.0; ok=True; errs=[]
    for k in range(nPC):
        y=f"pc{k}"
        try:
            m=gamfit.fit(df, ftpl.format(y=y))
            s=m.summary()
            tot_reml+=s.reml_score; tot_edf+=s.edf_total
            percol.append(dict(topology=name,response=y,reml_score=s.reml_score,
                               edf=s.edf_total,deviance=s.deviance))
        except Exception as e:
            ok=False; errs.append(f"{y}:{type(e).__name__}:{str(e)[:80]}")
    results.append(dict(topology=name,formula=ftpl,total_reml=tot_reml,
                        total_edf=tot_edf,mean_edf=tot_edf/nPC,ok=ok,errors=";".join(errs)))
    print(f"{name:24s} total_reml={tot_reml:12.3f} total_edf={tot_edf:7.2f} ok={ok} {';'.join(errs)[:120]}",flush=True)

res=pd.DataFrame(results).sort_values("total_reml")  # lower REML cost = better evidence
os.makedirs("runs/ANALYSIS",exist_ok=True)
res.to_csv("runs/ANALYSIS/gamfit_manifold_reml.csv",index=False)
pd.DataFrame(percol).to_csv("runs/ANALYSIS/gamfit_manifold_reml_percol.csv",index=False)
print("\n=== RANKING (lower REML cost = better marginal-likelihood evidence) ===")
print(res[["topology","total_reml","total_edf","mean_edf","ok"]].to_string(index=False))
res.to_json("runs/ANALYSIS/gamfit_manifold_reml.json",orient="records",indent=2)
print("\nSaved runs/ANALYSIS/gamfit_manifold_reml.{csv,json}")
