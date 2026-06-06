import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, gamfit
print("gamfit", gamfit.__version__)
# gam#795: RemlConvergenceError on a single circle. Minimal repro:
rng=np.random.default_rng(0)
N=120; D=8
t=np.sort(rng.uniform(0,2*np.pi,N))
# embed a circle in D dims + noise
A=rng.standard_normal((2,D))
X=np.c_[np.cos(t),np.sin(t)]@A + 0.05*rng.standard_normal((N,D))
print("X shape", X.shape)
t0=time.time()
try:
    res=gamfit.sae_manifold_fit(X=X, K=1, d_atom=1, atom_topology="circle")
    print("OK sae_manifold_fit single circle in %.2fs"%(time.time()-t0))
    print("result type:", type(res), [a for a in dir(res) if not a.startswith("_")][:20])
except Exception as e:
    print("RAISED %s after %.2fs: %s"%(type(e).__name__, time.time()-t0, str(e)[:300]))
