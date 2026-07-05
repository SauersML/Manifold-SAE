"""Theorem I verdict, deterministic-angle, FAST (no stochastic circle fits).
Reuses gamfit.layer_transport_fit as the transport verdict engine on the
deterministic data-plane angle. Writes JSON for the report + plot."""
import argparse, json, math
import numpy as np
TWO_PI=2*math.pi
LAYERS=[f"L{l}" for l in range(11,24)]
def demean(X,t):
    Xc=X.copy()
    for u in np.unique(t): m=t==u; Xc[m]-=X[m].mean(0,keepdims=True)
    return Xc
def plane_angle(Xc):
    _,sv,vt=np.linalg.svd(Xc,full_matrices=False); P=vt[:2].T; pr=Xc@P
    return np.mod(np.arctan2(pr[:,1],pr[:,0]),TWO_PI), float((sv[:2]**2).sum()/max((sv**2).sum(),1e-30))
def impurity(Xc,phi,H=5):
    cols=[np.ones_like(phi)]; idx={}
    for k in range(1,H+1): idx[k]=(len(cols),len(cols)+1); cols+=[np.cos(k*phi),np.sin(k*phi)]
    B=np.stack(cols,1); c,*_=np.linalg.lstsq(B,Xc,rcond=None)
    e={k:float((c[i:j]**2).sum()) for k,(i,j) in idx.items()}
    return sum(e[k] for k in range(2,H+1))/max(e[1],1e-30), e
def induced(pa,pb):
    Ea=np.stack([np.cos(pa),np.sin(pa)],1); Eb=np.stack([np.cos(pb),np.sin(pb)],1)
    Mt,*_=np.linalg.lstsq(Ea,Eb,rcond=None); M=Mt.T
    lin=float(np.sqrt(((Eb-Ea@M.T)**2).sum(1).mean()))
    MtM=M.T@M; lam=float(np.trace(MtM)/2)
    conf=float(np.linalg.norm(MtM-lam*np.eye(2))/max(lam,1e-30))
    w,_=np.linalg.eigh(MtM); s=np.sqrt(np.clip(w,0,None))
    return dict(lin_resid_rms=lin,conformal_departure=conf,anisotropy=float((s.max()-s.min())/max(s.max(),1e-30)),det=float(np.linalg.det(M)),singular_values=s.tolist())
def rigid(pa,pb):
    o={}
    for s in (1.,-1.):
        r=np.angle(np.exp(1j*(pb-s*pa))); ph=math.atan2(np.sin(r).mean(),np.cos(r).mean())
        rr=np.angle(np.exp(1j*(pb-(s*pa+ph)))); o[s]=(float(np.sqrt((rr**2).mean())),float(ph))
    s=1. if o[1.][0]<=o[-1.][0] else -1.
    return o[s][0], int(s), o[s][1]
ap=argparse.ArgumentParser(); ap.add_argument("--acts",required=True); ap.add_argument("--out",default="thmI_v3_results.json"); ap.add_argument("--n-perm",type=int,default=50)
a=ap.parse_args()
import gamfit
d=np.load(a.acts); t=d["template_ids"]
Xc={L:demean(d["acts_"+L],t) for L in LAYERS}
phi={}; plan={}; imp={}
for L in LAYERS:
    phi[L],plan[L]=plane_angle(Xc[L]); imp[L]=impurity(Xc[L],phi[L])[0]
hops=[]; rng=np.random.default_rng(0)
for A,B in zip(LAYERS[:-1],LAYERS[1:]):
    pa,pb=phi[A],phi[B]; tr=gamfit.layer_transport_fit(pa,pb); ind=induced(pa,pb); rg,sgn,ph=rigid(pa,pb)
    nd=[]; nc=[]
    for _ in range(a.n_perm):
        p=rng.permutation(len(pb)); tn=gamfit.layer_transport_fit(pa,pb[p]); nd.append(tn["isometry_defect"]); nc.append(tn["degree_concentration"])
    nd=np.array(nd); nc=np.array(nc)
    hops.append(dict(hop=f"{A}->{B}",degree=tr["degree"],degree_concentration=tr["degree_concentration"],
        isometry_defect=tr["isometry_defect"],isometry_defect_se=tr["isometry_defect_se"],transport_edf=tr["transport_edf"],
        residual_rms=tr["residual_rms"],rotation_offset=tr["rotation_offset"],conformal_departure=ind["conformal_departure"],
        anisotropy=ind["anisotropy"],lin_resid_rms=ind["lin_resid_rms"],det=ind["det"],rigid_circ_rms=rg,sign=sgn,phase=ph,
        impurity_src=imp[A],impurity_dst=imp[B],impurity_mean=0.5*(imp[A]+imp[B]),
        null_defect_mean=float(nd.mean()),null_conc_mean=float(nc.mean()),null_conc_sd=float(nc.std()),
        conc_gap=float(tr["degree_concentration"]-nc.mean())))
    print("[hop]",hops[-1]["hop"],"deg",hops[-1]["degree"],"conc",round(hops[-1]["degree_concentration"],3),"rigidRMS",round(rg,3),"confdep",round(ind["conformal_departure"],3),flush=True)
rig=np.array([h["rigid_circ_rms"] for h in hops]); conf=np.array([h["conformal_departure"] for h in hops]); impm=np.array([h["impurity_mean"] for h in hops]); dev=np.array([h["isometry_defect"] for h in hops])
co=lambda x,y:0. if x.std()<1e-12 or y.std()<1e-12 else float(np.corrcoef(x,y)[0,1])
S=dict(planarity=plan,impurity=imp,hops=hops,
    corr_rigid_impurity=co(rig,impm),corr_confdep_impurity=co(conf,impm),corr_rigid_confdep=co(rig,conf),corr_isodefect_impurity=co(dev,impm),
    median_rigid_circ_rms=float(np.median(rig)),median_conformal_departure=float(np.median(conf)),median_anisotropy=float(np.median([h["anisotropy"] for h in hops])),
    median_isometry_defect=float(np.median(dev)),median_degree_concentration=float(np.median([h["degree_concentration"] for h in hops])),
    median_null_conc=float(np.median([h["null_conc_mean"] for h in hops])),median_conc_gap=float(np.median([h["conc_gap"] for h in hops])),
    all_degree_one=all(h["degree"]==1 for h in hops),n_hops_phaseshift=int((rig<0.35).sum()),n_hops=len(hops),
    max_rigid_circ_rms=float(rig.max()),max_conformal_departure=float(conf.max()))
json.dump(S,open(a.out,"w"),indent=2)
print("[done]",json.dumps({k:S[k] for k in ["median_rigid_circ_rms","max_rigid_circ_rms","median_conformal_departure","max_conformal_departure","median_anisotropy","median_degree_concentration","median_null_conc","median_conc_gap","all_degree_one","n_hops_phaseshift","corr_rigid_impurity","corr_rigid_confdep"]},indent=2))
