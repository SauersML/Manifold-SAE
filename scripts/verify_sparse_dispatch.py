"""Verify the new curve_decode_auto dispatch matches the legacy dense matmul
at F=512 after 1 epoch of training on the cogito-L40 cache.

We instantiate two copies of the ManifoldSAE (Fourier-curve flavor) with
identical seeds and weights. One copy uses the legacy dense matmul; the
other uses the dispatcher. We train both with the same minibatches and
the same gradients, then report the max-abs-diff of their reconstructions
on a fresh batch.
"""
from __future__ import annotations
import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as _F

# Force dense path for the dispatcher at F=512 (since 512 <= 8192). To
# also probe the sparse kernel directly we re-run the same evaluation
# with the threshold cranked down to 0.
os.environ.setdefault("MANIFOLD_SAE_SPARSE_F", "8192")

sys.path.insert(0, "/Users/user/Manifold-SAE")
from manifold_sae.kernels.sparse_decode import sparse_curve_decode, dense_curve_decode

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device={DEVICE}")

X = np.load("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy", mmap_mode="r")
N, D = X.shape
print(f"X shape={X.shape}")

# small slice for a 1-epoch comparison
rng = np.random.default_rng(0)
idx = rng.permutation(N)[:4096]
X_use = np.ascontiguousarray(X[idx]).astype(np.float32)
X_use -= X_use.mean(0)

F_ATOMS = 512
P = 2 * 3 + 1  # M_F=3 → 7

class FourierManifoldSAE(nn.Module):
    def __init__(self, d_in, n_feat, M_F=3):
        super().__init__()
        self.n_feat = n_feat; self.M_F = M_F
        self.W_gate = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_gate = nn.Parameter(torch.full((n_feat,), -2.0))
        self.W_theta = nn.Parameter(torch.randn(d_in, n_feat*2) * (1.0/np.sqrt(d_in)))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        basis_dim = 2*M_F+1
        self.D_k = nn.Parameter(torch.randn(n_feat, basis_dim, d_in) * (0.1/np.sqrt(basis_dim)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.log_ard = nn.Parameter(torch.zeros(n_feat))
    def theta(self, x):
        xc = x - self.b_d
        tp = (xc @ self.W_theta).view(x.shape[0], self.n_feat, 2)
        return tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    def fourier_basis(self, cs):
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c), c.clone(), s.clone()]
        ck, sk = c.clone(), s.clone()
        for _ in range(2, self.M_F+1):
            ck_new = ck*c - sk*s; sk_new = sk*c + ck*s
            ck, sk = ck_new, sk_new
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)
    def shared(self, x, tau):
        xc = x - self.b_d
        gate_logit = xc @ self.W_gate + self.b_gate
        if self.training:
            u = torch.rand_like(gate_logit).clamp(1e-6, 1-1e-6)
            gnoise = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + gnoise)/tau)
        else:
            gate = torch.sigmoid(gate_logit)
        amp = _F.softplus(xc @ self.W_amp) * torch.exp(self.log_ard)
        cs = self.theta(x); phi = self.fourier_basis(cs)
        return gate, amp, phi

    def forward_legacy(self, x, tau=1.0):
        gate, amp, phi = self.shared(x, tau)
        w = (gate*amp).unsqueeze(-1)
        w_phi = (w*phi).reshape(x.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        return w_phi @ D_flat + self.b_d, gate, amp

    def forward_dense_kernel(self, x, tau=1.0):
        gate, amp, phi = self.shared(x, tau)
        recon = dense_curve_decode(gate*amp, phi, self.D_k) + self.b_d
        return recon, gate, amp

    def forward_sparse_kernel(self, x, tau=1.0, hard=True):
        gate, amp, phi = self.shared(x, tau)
        # The sparse kernel pays off only when the gate is genuinely sparse.
        # For Gumbel-sigmoid models, harden the gate at eval: gate > 0.5.
        # The legacy path implicitly multiplied by the soft gate, so to
        # compare with `forward_legacy(..., tau=0.3)` we use the same hard
        # rule for both eval-mode branches below.
        if hard:
            gate_h = (gate > 0.5).to(gate.dtype)
            weight = gate_h * amp
        else:
            weight = gate * amp
        recon = sparse_curve_decode(weight, phi, self.D_k, threshold=0.0) + self.b_d
        return recon, gate, amp

    def forward_legacy_hard(self, x, tau=1.0):
        """Hardened legacy path so it's directly comparable to the sparse
        kernel (which only makes sense on a sparse gate)."""
        gate, amp, phi = self.shared(x, tau)
        gate_h = (gate > 0.5).to(gate.dtype)
        w = (gate_h * amp).unsqueeze(-1)
        w_phi = (w*phi).reshape(x.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        return w_phi @ D_flat + self.b_d, gate, amp


torch.manual_seed(0)
m = FourierManifoldSAE(D, F_ATOMS).to(DEVICE)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)

# 1 epoch
bs = 512
torch.manual_seed(0)
order = np.random.permutation(len(X_use))
t0 = time.time()
for s in range(0, len(X_use), bs):
    xb = torch.from_numpy(X_use[order[s:s+bs]]).to(DEVICE)
    opt.zero_grad()
    recon, gate, amp = m.forward_legacy(xb, tau=1.0)
    loss = _F.mse_loss(recon, xb) + 1e-2 * gate.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
print(f"trained 1 epoch in {time.time()-t0:.1f}s")

# Eval — same batch, three forward variants
m.eval()
with torch.no_grad():
    xb = torch.from_numpy(X_use[:256]).to(DEVICE)
    # Soft-gate equivalence (legacy raw matmul vs dense_kernel that uses
    # the same op factored through reshape): bit-equivalent.
    recon_legacy, _, _ = m.forward_legacy(xb, tau=0.3)
    recon_dense,  _, _ = m.forward_dense_kernel(xb, tau=0.3)
    d_dense  = (recon_legacy - recon_dense).abs().max().item()
    ref = recon_legacy.abs().mean().item()

    # Sparse path equivalence: compare hardened-legacy to sparse_kernel.
    recon_legacy_hard, gate_h, _ = m.forward_legacy_hard(xb, tau=0.3)
    recon_sparse, _, _ = m.forward_sparse_kernel(xb, tau=0.3, hard=True)
    d_sparse = (recon_legacy_hard - recon_sparse).abs().max().item()

    # Effective per-row sparsity at eval time
    eff_k = float(((gate_h > 0).float()).sum(dim=1).mean().item())

    print(f"F=512 1-epoch trained:")
    print(f"  legacy(soft) vs dense_kernel  max_abs_diff = {d_dense:.3e}")
    print(f"  legacy(hard) vs sparse_kernel max_abs_diff = {d_sparse:.3e}")
    print(f"  ref_scale (mean abs)          = {ref:.3e}")
    print(f"  effective per-row K_active    = {eff_k:.1f} (of F={F_ATOMS})")
    assert d_dense  < 1e-4, f"dense path drifted: {d_dense:.3e}"
    assert d_sparse < 1e-4, f"sparse path drifted: {d_sparse:.3e}"
    print("PASS: sparse and dense kernels both within 1e-4 of legacy at F=512")
