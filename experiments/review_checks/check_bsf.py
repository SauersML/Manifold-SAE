"""R-review numerical checks against the ACTUAL committed BSF code (a401226)."""
import sys, numpy as np, torch
sys.path.insert(0, "/Users/user/Manifold-SAE/experiments/bsf_baseline")
from bsf import BSF, BSFConfig, block_topk_mask
torch.set_default_dtype(torch.float64)

fail = []
def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" | "+extra if extra else ""))
    if not cond: fail.append(name)

# ---- 1. single shared scalar gamma -------------------------------------
cfg = BSFConfig(d_model=10, n_blocks=4, block_size=3, k_blocks=2, mode="grassmann", seed=1)
m = BSF(cfg)
chk("gamma is a single 0-dim scalar", m.log_gamma.shape == torch.Size([]),
    f"shape={tuple(m.log_gamma.shape)}")
chk("vanilla-only params absent in grassmann", m.encoder is None and m.enc_bias is None)

# ---- 2. block-TopK selects by group L2 norm ----------------------------
z = torch.randn(5, 4, 3)
norms = torch.linalg.vector_norm(z, dim=2)
mask = block_topk_mask(z, 2)
# the 2 kept blocks per row must be exactly the 2 largest-norm blocks
for i in range(5):
    kept = set(torch.where(mask[i] > 0)[0].tolist())
    want = set(torch.topk(norms[i], 2).indices.tolist())
    chk(f"row{i} block-TopK == top-2 group-norm", kept == want, f"{kept} vs {want}")
chk("mask carries no gradient", not mask.requires_grad)

# ---- 3. grassmann projection identity: z_g D_g == gamma * P_g x ---------
m.reproject_stiefel()  # put decoder on Stiefel
x = torch.randn(7, 10)
z = m.encode(x)                      # (N,G,b)
g, b, d = 4, 3, 10
gamma = float(torch.exp(m.log_gamma))
dec = m.decoder.detach()             # (G,b,d)
# reconstruct block 0 contribution from code, compare to gamma * projection
for gi in range(g):
    Dg = dec[gi]                     # (b,d), orthonormal rows after reproject
    contrib = z[:, gi, :] @ Dg       # (N,d)
    Pg = Dg.T @ Dg                   # (d,d) projector onto row-space
    proj = gamma * (x - m.b_dec) @ Pg
    chk(f"block{gi}: z_g D_g == gamma*P_g(x-bdec)",
        torch.allclose(contrib, proj, atol=1e-9),
        f"maxerr={float((contrib-proj).abs().max()):.2e}")
    # orthonormality of rows
    chk(f"block{gi} rows orthonormal after reproject",
        torch.allclose(Dg @ Dg.T, torch.eye(b), atol=1e-10))

# ---- 4. GAUGE INVARIANCE: rotate block basis by O(b), tied code follows -
# In grassmann mode encoder IS the decoder, so rotating D_g -> R D_g must
# leave BOTH ||z_g|| (selection) and z_g D_g (reconstruction) invariant.
rng = np.random.default_rng(0)
R = np.linalg.qr(rng.standard_normal((b, b)))[0]     # random O(b)
Rt = torch.tensor(R)
out0 = m(x, update_util=False)
z0 = m.encode(x)
norms0 = torch.linalg.vector_norm(z0, dim=2)
xhat0 = out0.x_hat.clone()
with torch.no_grad():
    m.decoder.data[0] = Rt @ m.decoder.data[0]       # rotate block 0's basis
z1 = m.encode(x)
norms1 = torch.linalg.vector_norm(z1, dim=2)
out1 = m(x, update_util=False)
chk("gauge: ||z_g|| invariant under O(b) rotation (selection)",
    torch.allclose(norms0, norms1, atol=1e-9),
    f"maxerr={float((norms0-norms1).abs().max()):.2e}")
chk("gauge: reconstruction invariant under O(b) rotation (loss)",
    torch.allclose(xhat0, out1.x_hat, atol=1e-9),
    f"maxerr={float((xhat0-out1.x_hat).abs().max()):.2e}")
# NEGATIVE control: a norm-CHANGING (non-orthogonal) map must change things
with torch.no_grad():
    m.decoder.data[0] = 2.0 * m.decoder.data[0]      # scale != orthogonal
out2 = m(x, update_util=False)
chk("neg-control: non-orthogonal map DOES change reconstruction",
    not torch.allclose(xhat0, out2.x_hat, atol=1e-6))

print("\n" + ("ALL PASS" if not fail else f"FAILURES: {fail}"))
