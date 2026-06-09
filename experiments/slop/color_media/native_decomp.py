import numpy as np, re, glob, os, json
np.seterr(all='ignore')

files = glob.glob('/tmp/colall/*.npz')

def sort_key(p):
    name = os.path.basename(p)
    # strip leading "NN__"
    label = name.split('__',1)[1].rsplit('.npz',1)[0]
    # stage rank
    if 'RL31' in label: stage=6
    elif 'TRAJ_RL_' in label or re.search(r'TRAJ_RL__',label): stage=5
    elif 'DPO' in label: stage=4
    elif 'SFT' in label: stage=3
    elif 'stage1' in label: stage=0
    elif 'stage2' in label: stage=1
    elif 'stage3' in label: stage=2
    else: stage=99
    # step number
    m = re.search(r'step[_-]?(\d+)', label)
    step = int(m.group(1)) if m else (10**9 if 'DPO' in label else 0)
    return (stage, step)

files.sort(key=sort_key)
print("ORDER:")
for f in files:
    print(" ", sort_key(f), os.path.basename(f))

K = 8  # number of modal/laplacian frequencies
rows=[]
for f in files:
    d = np.load(f); V = d['V'].astype(np.float64)  # (30,5120)
    # normalize each color rep to unit norm for similarity geometry
    Vn = V/ (np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
    S = Vn@Vn.T                       # cosine similarity (30,30)
    A = np.clip(S,0,None); np.fill_diagonal(A,0)   # affinity (nonneg)
    deg = A.sum(1)
    Dinv = 1/np.sqrt(deg+1e-12)
    Lsym = np.eye(30) - (Dinv[:,None]*A*Dinv[None,:])  # normalized Laplacian
    lap_eig = np.sort(np.linalg.eigvalsh(Lsym))        # 0..2, ascending
    # covariance spectrum of the rep (PCA energy)
    cov_eig = np.linalg.eigvalsh(V@V.T)[::-1]          # (30,) descending
    cov_eig = np.clip(cov_eig,0,None)
    p = cov_eig/cov_eig.sum()
    spectral_entropy = -(p*np.log(p+1e-12)).sum()/np.log(len(p))   # 0..1
    participation = (cov_eig.sum()**2)/(np.sum(cov_eig**2))        # eff dim
    top_share = p[0]
    rows.append(dict(
        label=os.path.basename(f).split('__',1)[1].rsplit('.npz',1)[0],
        lap_eig=lap_eig[1:1+K].tolist(),   # skip trivial 0 mode -> K nontrivial modes
        cov_top=(cov_eig[:K]/cov_eig[0]).tolist(),  # relative partial amps
        spec_entropy=float(spectral_entropy),
        participation=float(participation),
        top_share=float(top_share),
    ))

json.dump(rows, open('/tmp/native_traj.json','w'))
import numpy as np
se=[r['spec_entropy'] for r in rows]; ts=[r['top_share'] for r in rows]; pr=[r['participation'] for r in rows]
print("\nN ckpts", len(rows))
print("spec_entropy first/last:", round(se[0],3), round(se[-1],3), " min/max", round(min(se),3), round(max(se),3))
print("top_share first/last:", round(ts[0],3), round(ts[-1],3))
print("participation first/last:", round(pr[0],3), round(pr[-1],3))
print("lap_eig[1] (Fiedler) first/last:", round(rows[0]['lap_eig'][0],4), round(rows[-1]['lap_eig'][0],4))
