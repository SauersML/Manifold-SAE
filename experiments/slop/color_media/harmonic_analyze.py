import numpy as np, glob, re, json
from numpy.linalg import svd

# ---------- chronological ordering ----------
def sort_key(path):
    lab = path.split('/')[-1].replace('.npz','')
    # stage ordering
    if 'stage1' in lab:
        stage = 0; step = int(re.search(r'step(\d+)', lab).group(1))
    elif 'stage2' in lab:
        stage = 1; step = int(re.search(r'step(\d+)', lab).group(1))
    elif 'stage3' in lab:
        stage = 2; step = int(re.search(r'step(\d+)', lab).group(1))
    elif '_SFT_' in lab:
        stage = 3; step = int(re.search(r'step(\d+)', lab).group(1))
    elif '_DPO_' in lab:
        stage = 4; step = 0
    elif 'RL31' in lab:
        stage = 6; step = int(re.search(r'step_?(\d+)', lab).group(1))
    elif '_RL_' in lab:
        stage = 5; step = int(re.search(r'step_?(\d+)', lab).group(1))
    else:
        raise ValueError(lab)
    return (stage, step)

STAGE_NAMES = ['pretrain(stage1)','stage2','stage3','SFT','DPO','RL3.0','RL3.1']

files = sorted(glob.glob('/tmp/colall/*.npz'), key=sort_key)
print("ordered:")
for f in files:
    print(' ', sort_key(f), f.split('/')[-1])

# ---------- harmonic decomposition per checkpoint ----------
# For each checkpoint:
#   V (30,5120) frame-demeaned color reps; rows = 30 colors
#   1. center, reduce to leading PCs (scores S, 30 x r)
#   2. dominant non-cyclic direction = PC1. Deflate it -> residual plane = PC2,PC3.
#   3. angle theta_i = atan2(score3, score2) gives unsupervised loop ordering.
#   4. decompose ALL pc-scores' variation around theta into circular harmonics k=1..K:
#         basis B_k = [cos k theta, sin k theta]; regress each PC-score column on
#         the full harmonic basis; harmonic-k amplitude = sqrt of variance explained
#         by [cos k th, sin k th] across all retained PC dims.
#   5. residual after all harmonics = "noise". HNR = harmonic energy / residual energy.

K = 8           # number of circular harmonics
R = 8           # PCs retained for harmonic fit

records = []
for f in files:
    d = np.load(f)
    V = d['V'].astype(np.float64)          # (30,5120)
    rgb = d['rgb']
    Vc = V - V.mean(0, keepdims=True)
    # PCA via SVD
    U, sv, Vt = svd(Vc, full_matrices=False)   # U(30,30) sv(30) Vt(30,5120)
    scores = U * sv                            # (30,30) PC scores
    total_var = (sv**2).sum()

    # cyclic plane: deflate PC1 (dominant, often a brightness/non-cyclic axis)
    # use PC2,PC3 as the loop plane
    s2, s3 = scores[:,1], scores[:,2]
    theta = np.arctan2(s3, s2)

    # retain R pcs for harmonic fit (these carry the manifold's shape)
    Y = scores[:, :R]                          # (30,R)
    Yc = Y - Y.mean(0, keepdims=True)
    sig_energy = (Yc**2).sum()                 # total signal energy in retained PCs

    # build harmonic design over theta
    cols = [np.ones_like(theta)]
    for k in range(1, K+1):
        cols.append(np.cos(k*theta)); cols.append(np.sin(k*theta))
    Bfull = np.column_stack(cols)              # (30, 1+2K)
    # least squares fit of each PC-score col on full basis
    coef, *_ = np.linalg.lstsq(Bfull, Yc, rcond=None)   # (1+2K, R)
    fit = Bfull @ coef
    resid = Yc - fit
    resid_energy = (resid**2).sum()

    # per-harmonic energy: energy contributed by [cos k, sin k] columns.
    # Use orthogonalized contribution: refit incrementally to attribute energy.
    # Simpler & honest: project residual-of-lower onto each harmonic sequentially.
    harm_energy = np.zeros(K)
    Bcur = Bfull[:, :1]                        # start with constant
    coef0, *_ = np.linalg.lstsq(Bcur, Yc, rcond=None)
    cur_resid = Yc - Bcur @ coef0
    for k in range(1, K+1):
        Bk = Bfull[:, 1+2*(k-1):1+2*k]         # the two cols for harmonic k
        # additional variance explained by adding harmonic k (orthogonalize Bk vs Bcur)
        # regress Bk out of current basis space: fit Bk from Bcur
        proj, *_ = np.linalg.lstsq(Bcur, Bk, rcond=None)
        Bk_orth = Bk - Bcur @ proj
        ck, *_ = np.linalg.lstsq(Bk_orth, cur_resid, rcond=None)
        contrib = Bk_orth @ ck
        harm_energy[k-1] = (contrib**2).sum()
        cur_resid = cur_resid - contrib
        Bcur = np.column_stack([Bcur, Bk])

    noise_energy = (cur_resid**2).sum()
    harm_total = harm_energy.sum()
    HNR = harm_total / (noise_energy + 1e-12)

    records.append(dict(
        label=f.split('/')[-1].replace('.npz',''),
        stage=sort_key(f)[0],
        harm_energy=harm_energy.tolist(),
        harm_total=float(harm_total),
        noise_energy=float(noise_energy),
        sig_energy=float(sig_energy),
        HNR=float(HNR),
        pc1_share=float(sv[0]**2/total_var),
        sv=sv[:R].tolist(),
    ))

with open('/tmp/harmonic_records.json','w') as fp:
    json.dump(dict(stage_names=STAGE_NAMES, K=K, R=R, records=records), fp, indent=1)

# summary
print("\n n  stage            HNR    h1share  pc1share  label")
for i,r in enumerate(records):
    he = np.array(r['harm_energy'])
    h1 = he[0]/(he.sum()+1e-12)
    print(f"{i:2d} {STAGE_NAMES[r['stage']]:14s} {r['HNR']:7.3f} {h1:7.3f} {r['pc1_share']:7.3f}  {r['label']}")

# early vs late aggregate
import numpy as np
early = [r for r in records if r['stage']==0][:5]
late  = [r for r in records if r['stage']>=5][-8:]
def agg(rs, key):
    return np.mean([r[key] for r in rs])
def h1agg(rs):
    return np.mean([np.array(r['harm_energy'])[0]/(np.array(r['harm_energy']).sum()+1e-12) for r in rs])
print(f"\nEARLY pretrain: HNR={agg(early,'HNR'):.3f} h1share={h1agg(early):.3f}")
print(f"LATE RL:        HNR={agg(late,'HNR'):.3f} h1share={h1agg(late):.3f}")
print("wrote /tmp/harmonic_records.json")
