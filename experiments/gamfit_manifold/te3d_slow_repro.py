import time, numpy as np, pandas as pd, gamfit
print("gamfit", gamfit.__version__)
rng = np.random.default_rng(0)
n = 30
df = pd.DataFrame(dict(x1=rng.uniform(0,1,n), x2=rng.uniform(0,1,n), x3=rng.uniform(0,1,n)))
df["y"] = np.sin(3*df.x1) + df.x2**2 - df.x3 + 0.05*rng.standard_normal(n)
for f in ["y ~ s(x1)", "y ~ te(x1,x2)", "y ~ te(x1,x2,x3)"]:
    t0=time.time(); m=gamfit.fit(df, f); dt=time.time()-t0
    print("%-22s %7.2fs  reml=%.2f edf=%.2f" % (f, dt, m.summary().reml_score, m.summary().edf_total))
