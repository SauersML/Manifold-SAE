import numpy as np, torch, gamfit
from experiments.recover_bench import shape_library, make_data, orthogonal_blocks
import experiments.method_ot as m
from experiments.shape_metrics import rel_rms
shapes=shape_library();names=list(shapes)
def sw(A,B,np_,gen):
  d=A.shape[1];dr=torch.randn(d,np_,generator=gen,dtype=A.dtype);dr/=dr.norm(dim=0,keepdim=True)
  pa=(A@dr).sort(0).values;pb=(B@dr).sort(0).values
  if pa.shape[0]!=pb.shape[0]:
    n=max(pa.shape[0],pb.shape[0]);q=torch.linspace(0,1,n,dtype=A.dtype);pa=m._qi(pa,q);pb=m._qi(pb,q)
  return ((pa-pb)**2).mean()
def interp(cp,t,closed):
  P=cp.shape[0]
  if closed: pos=t*P;lo=pos.floor().long()%P;hi=(lo+1)%P;w=(pos-pos.floor()).view(-1,1)
  else: pos=t*(P-1);lo=pos.floor().long().clamp(0,P-1);hi=(lo+1).clamp(0,P-1);w=(pos-lo).view(-1,1)
  return cp[lo]*(1-w)+cp[hi]*w
def fit2d(P2,seed,closed,K=8,ns=600):
  N=P2.shape[0];gen=torch.Generator().manual_seed(seed)
  ctr=P2.median(0).values;nrm=(P2-ctr).norm(dim=1);sc=float(nrm.quantile(0.95));P=80
  if closed:
    a=torch.linspace(0,2*np.pi,P+1,dtype=torch.float64)[:-1];cp=(ctr+0.6*sc*torch.stack([torch.cos(a),torch.sin(a)],1)).clone()
  else:
    Pc=P2-P2.mean(0);U,S,Vt=torch.linalg.svd(Pc,full_matrices=False);ax=Vt[0]
    s=torch.linspace(-1,1,P,dtype=torch.float64).unsqueeze(1);cp=(ctr+s*ax*sc).clone()
  cp=cp+0.02*sc*torch.randn(cp.shape[0],2,generator=gen,dtype=torch.float64);cp.requires_grad_()
  opt=torch.optim.Adam([cp],lr=0.03);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ns)
  near=P2[(P2-ctr).norm(dim=1)<0.3*sc];bstd=near.std(0).detach() if near.shape[0]>10 else torch.ones(2,dtype=torch.float64)*0.01
  for step in range(ns):
    idx=torch.randint(0,N,(1800,),generator=gen);Xb=P2[idx];ncurve=1800//K
    t=torch.rand(ncurve,generator=gen,dtype=torch.float64);cl=interp(cp,t,closed)
    nb=1800-ncurve;blobpts=ctr+bstd*torch.randn(nb,2,generator=gen,dtype=torch.float64)
    L=sw(Xb,torch.cat([cl,blobpts],0),256,gen)
    d2=(cp.roll(-1,0)-2*cp+cp.roll(1,0)) if closed else (cp[2:]-2*cp[1:-1]+cp[:-2]);smv=(d2**2).sum(-1).mean()
    seg=((cp.roll(-1,0)-cp) if closed else (cp[1:]-cp[:-1])).pow(2).sum(-1);spd=seg.var()
    loss=L+(0.003+0.08*(1-step/ns))*smv+0.03*spd
    opt.zero_grad();loss.backward();opt.step();sched.step()
  return cp.detach()
def detect_closed(P2np):
  P2=P2np;nrm=np.linalg.norm(P2-np.median(P2,0),axis=1);arm=P2[nrm>0.4*np.quantile(nrm,0.95)]
  c=arm.mean(0);rel=arm-c;ang=np.arctan2(rel[:,1],rel[:,0]);rad=np.linalg.norm(rel,axis=1)
  asort=np.sort(ang);maxgap=np.degrees(np.diff(np.r_[asort,asort[0]+2*np.pi]).max())
  bins=np.linspace(-np.pi,np.pi,13);bi=np.digitize(ang,bins);cvs=[]
  for b in range(1,13):
    rr=rad[bi==b]
    if len(rr)>3: cvs.append(rr.std()/(rr.mean()+1e-9))
  cv=np.median(cvs) if cvs else 0.0
  return (maxgap<60) and (cv<0.15)
def reml_smooth(curve,closed):
  d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
  t=d/(d[-1]+np.linalg.norm(curve[0]-curve[-1])) if closed else d/d[-1]
  outc=np.zeros_like(curve)
  for j in range(curve.shape[1]):
    rr=gamfit.gaussian_reml_fit_positions(t,curve[:,j],basis='duchon',basis_order=2,periodic=closed,period=(1.0 if closed else None))
    outc[:,j]=np.asarray(rr['fitted']).ravel()
  return outc
X,gts,names=make_data(shapes,seed=0)
mu=X.mean(0);Xc=X-mu
U,S,Vt=np.linalg.svd(Xc,full_matrices=False);r=16;Vr=Vt[:r];Xr=Xc@Vr.T
Btrue=orthogonal_blocks(8,20,100);Btrue_r=[B@Vr.T for B in Btrue]
es=[]
for i in range(8):
  Q=np.linalg.qr(Btrue_r[i].T)[0];P2=Xr@Q
  cl=detect_closed(P2)
  print(f'{names[i]:14s} closed={cl}',flush=True)
  cp=fit2d(torch.tensor(P2,dtype=torch.float64),i,cl)
  tt=torch.linspace(0,1,201,dtype=torch.float64)[:-1] if cl else torch.linspace(0,1,200,dtype=torch.float64)
  out=interp(cp,tt,cl).numpy();c2=reml_smooth(out,cl);amb=c2@Q.T@Vr+mu
  e=100*rel_rms(gts[i],amb);es.append(e);print(f'   {names[i]:14s} rel_rms={e:.2f}%',flush=True)
print('MEAN %.2f%%'%np.mean(es))
