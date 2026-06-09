import numpy as np, glob, re, os, colorsys, json

files = sorted(glob.glob('/tmp/colall/*.npz'))

# ---- chronological sort ----
STAGE_ORDER = {'stage1':0,'stage2':1,'stage3':2,'SFT':3,'DPO':4,'RL30':5,'RL31':6}
def stage_of(fn):
    b = os.path.basename(fn)
    if 'stage1' in b: return 'stage1'
    if 'stage2' in b: return 'stage2'
    if 'stage3' in b: return 'stage3'
    if '_SFT_' in b: return 'SFT'
    if '_DPO_' in b: return 'DPO'
    if 'RL31' in b: return 'RL31'
    if '_RL_' in b: return 'RL30'
    raise ValueError(b)
def step_of(fn):
    b = os.path.basename(fn)
    m = re.search(r'step[_-]?(\d+)', b)
    return int(m.group(1)) if m else 0

meta = []
for fn in files:
    st = stage_of(fn); meta.append((STAGE_ORDER[st], step_of(fn), st, fn))
meta.sort(key=lambda x:(x[0],x[1]))

print("Chronological order:")
for i,(so,stp,st,fn) in enumerate(meta):
    print(f"{i:2d} {st:7s} step={stp:7d}  {os.path.basename(fn)}")

# ---- true hue of the 30 colors (constant across files) ----
d0 = np.load(meta[0][3]); rgb = d0['rgb']
true_hue = np.array([colorsys.rgb_to_hsv(*(c/255.0))[0] for c in rgb])  # [0,1)
true_theta = 2*np.pi*true_hue

# ---- per-checkpoint hue recovery + harmonics ----
def circ_corr(a, b):
    # circular correlation between two angle arrays
    a = a - np.angle(np.mean(np.exp(1j*a)))
    b = b - np.angle(np.mean(np.exp(1j*b)))
    num = np.sum(np.sin(a)*np.sin(b))
    den = np.sqrt(np.sum(np.sin(a)**2)*np.sum(np.sin(b)**2))
    return num/den if den>0 else 0.0

records = []
for so,stp,st,fn in meta:
    V = np.load(fn)['V'].astype(np.float64)   # (30,5120) already frame-demeaned
    Vc = V - V.mean(0, keepdims=True)
    # PCA
    U,S,Wt = np.linalg.svd(Vc, full_matrices=False)
    scores = U*S   # (30, k) coords in PC space
    # deflate PC1 (dominant nuisance): use PC2,PC3,PC4 candidate planes, pick plane whose angle best matches true hue
    best = None
    cand_planes = [(1,2),(1,3),(2,3),(1,4),(2,4)]
    for (a,b) in cand_planes:
        if b >= scores.shape[1]: continue
        ang = np.arctan2(scores[:,b], scores[:,a])
        # try both orientations (sign flip) via circ corr magnitude
        cc = circ_corr(ang, true_theta)
        # also allow reflection (reverse hue direction)
        cc_r = circ_corr(-ang, true_theta)
        if abs(cc_r) > abs(cc):
            cc = cc_r; ang = -ang
        if best is None or abs(cc) > abs(best[0]):
            best = (cc, ang, (a,b))
    cc, rec_ang, plane = best
    circularity = abs(cc)

    # circular harmonics: regress each of PC2..PC5 scores onto [cos k th, sin k th] for k=1,2,3 using TRUE theta
    # fraction of explained structure in 1st harmonic vs higher
    th = true_theta
    Xh = {k: np.column_stack([np.cos(k*th), np.sin(k*th)]) for k in (1,2,3)}
    # Use the recovered-plane 2 coords stacked as the signal to explain
    sig = scores[:, list(plane)]  # (30,2)
    sig = sig - sig.mean(0)
    pwr = {}
    for k in (1,2,3):
        X = Xh[k]; X = X - X.mean(0)
        # projection power of sig onto harmonic-k subspace
        beta, *_ = np.linalg.lstsq(X, sig, rcond=None)
        pred = X@beta
        pwr[k] = np.sum(pred**2)
    tot = pwr[1]+pwr[2]+pwr[3] + 1e-12
    h1_frac = pwr[1]/tot
    h2_frac = pwr[2]/tot
    h3_frac = pwr[3]/tot

    # per-color "on-manifold" amplitude: how well its recovered angle fits the circle
    # use cos of angular residual between recovered and (sign-aligned) true
    # align recovered to true by best rotation
    rot = np.angle(np.mean(np.exp(1j*(true_theta - rec_ang))))
    rec_aligned = rec_ang + rot
    ang_err = np.angle(np.exp(1j*(rec_aligned - true_theta)))  # in [-pi,pi]
    on_manifold = 0.5*(1+np.cos(ang_err))  # 1 = perfect, per color

    records.append(dict(stage=st, step=stp, circularity=circularity,
                        h1=h1_frac, h2=h2_frac, h3=h3_frac,
                        rec_ang=rec_ang.tolist(), rec_aligned=rec_aligned.tolist(),
                        on_manifold=on_manifold.tolist(), plane=plane))

# save
out = dict(true_hue=true_hue.tolist(), rgb=rgb.tolist(), records=records,
           order=[(r['stage'],r['step']) for r in records])
json.dump(out, open('/tmp/color_data.json','w'))
print("\nSaved /tmp/color_data.json")
print("\ncircularity / h1_frac by checkpoint:")
for i,r in enumerate(records):
    print(f"{i:2d} {r['stage']:6s} step={r['step']:7d} circ={r['circularity']:.3f} h1={r['h1']:.3f}")
# early vs late summary
early = records[:5]; late = records[-10:]
print(f"\nEARLY (first 5) mean circularity={np.mean([r['circularity'] for r in early]):.3f} h1={np.mean([r['h1'] for r in early]):.3f}")
print(f"LATE (last 10) mean circularity={np.mean([r['circularity'] for r in late]):.3f} h1={np.mean([r['h1'] for r in late]):.3f}")
