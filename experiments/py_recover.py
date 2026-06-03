"""Pure-torch joint manifold-SAE recovery (Run 0/1), bypassing the Rust solver.

Plants 2 circle atoms in ambient D (same generator as manifold_falsifier.plant),
fits each atom as a plane P_k in R^{D x 2} + per-token angle + per-token amp by
gradient descent, with a decoder-incoherence penalty on the cross-plane Gram
||P0^T P1||_F^2 (the lever). Oracle gate (known support) to isolate the per-token
SPLIT / coordinate recovery — the headline question. Scores with the SAME
circ_procrustes_r2 metric and reports the coherence sweep, incoherence ON vs OFF.
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from manifold_falsifier import Config, plant, circ_procrustes_r2, tangent_sigma_min

DEV = "cpu"  # tiny problem; CPU is fine and deterministic


def fit_py(X, gate, D, lam_inc, steps=3000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    n = X.shape[0]
    Xt = torch.tensor(X, dtype=torch.float64, device=DEV)
    G = torch.tensor(gate.astype(np.float64), device=DEV)  # (n,2)
    # params: planes (2,D,2), angle phi (2,n), log-amp (2,n)
    P = torch.nn.Parameter(0.1 * torch.randn(2, D, 2, dtype=torch.float64))
    phi = torch.nn.Parameter(2 * np.pi * torch.rand(2, n, dtype=torch.float64))
    logamp = torch.nn.Parameter(torch.zeros(2, n, dtype=torch.float64))
    opt = torch.optim.Adam([P, phi, logamp], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        emb = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (2,n,2)
        amp = torch.exp(logamp)  # (2,n)
        # contrib_k = amp_k * gate_k * (emb_k @ P_k^T)  -> (n,D)
        xhat = torch.zeros(n, D, dtype=torch.float64, device=DEV)
        for k in range(2):
            ck = (emb[k] @ P[k].T)  # (n,D)
            xhat = xhat + (amp[k] * G[:, k]).unsqueeze(1) * ck
        mse = ((Xt - xhat) ** 2).mean()
        # decoder-incoherence on the column SPACES (normalize columns first)
        inc = torch.tensor(0.0, dtype=torch.float64, device=DEV)
        if lam_inc > 0:
            Pn = P / (P.norm(dim=1, keepdim=True) + 1e-9)  # unit columns
            cross = Pn[0].T @ Pn[1]  # (2,2)
            inc = (cross ** 2).sum()
        # mild orthonormality reg keeps each plane well-conditioned
        orth = sum(((P[k].T @ P[k]) / (P[k].norm() ** 2 / 2 + 1e-9)
                    - torch.eye(2, dtype=torch.float64)).pow(2).sum() for k in range(2))
        loss = mse + lam_inc * inc + 1e-3 * orth
        loss.backward()
        opt.step()
    with torch.no_grad():
        emb = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
        amp = torch.exp(logamp)
        xhat = torch.zeros(n, D, dtype=torch.float64)
        for k in range(2):
            xhat = xhat + (amp[k] * G[:, k]).unsqueeze(1) * (emb[k] @ P[k].T)
        recon_r2 = 1 - ((Xt - xhat) ** 2).sum().item() / (Xt ** 2).sum().item()
        phi_hat = (phi.detach().numpy() / (2 * np.pi)) % 1.0  # back to [0,1]
        P_hat = P.detach().numpy()
    return phi_hat, P_hat, recon_r2


def median_coactive_sigma_min(gt):
    t, planes, co = gt["t"], gt["planes"], gt["coactive"]
    vals = [tangent_sigma_min(t[0, i], t[1, i], planes[0], planes[1])
            for i in np.flatnonzero(co)]
    return float(np.median(vals)) if vals else float("nan")


def run_cell(coherence, D=64, n=2000, n_dis=200, coact=0.5, noise=0.05, seed=0):
    cfg = Config(n=n, d_ambient=D, noise=noise, coherence=coherence,
                 coactive_fraction=coact, n_disambiguating=n_dis, seed=seed)
    gt = plant(cfg)
    X, gate, t_true = gt["X"], gt["gate"], gt["t"]
    smin = median_coactive_sigma_min(gt)
    out = {}
    for tag, lam in (("OFF", 0.0), ("ON", 1.0)):
        phi_hat, _, r2 = fit_py(X, gate, D, lam, seed=seed)
        # score coord recovery per atom over its active tokens (gauge-robust)
        r2coord = []
        for k in range(2):
            act = gate[:, k].astype(bool)
            r2coord.append(circ_procrustes_r2(phi_hat[k][act], t_true[k][act]))
        out[tag] = (float(np.mean(r2coord)), r2)
    return smin, out


if __name__ == "__main__":
    print(f"pure-torch manifold-SAE recovery | D=64, n=2000, 2 circles, oracle gate")
    print(f"{'coh':>5} {'sigmin':>7} | {'coordR2_OFF':>11} {'reconR2_OFF':>11} | "
          f"{'coordR2_ON':>10} {'reconR2_ON':>10}")
    sweep = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]
    for coh in sweep:
        c_off, c_on, s = [], [], []
        for seed in (0, 1, 2):
            smin, out = run_cell(coh, seed=seed)
            c_off.append(out["OFF"][0]); c_on.append(out["ON"][0]); s.append(smin)
        smin = float(np.mean(s))
        # recon from last seed for reference
        _, out = run_cell(coh, seed=0)
        print(f"{coh:5.2f} {smin:7.3f} | {np.mean(c_off):11.3f} {out['OFF'][1]:11.3f} | "
              f"{np.mean(c_on):10.3f} {out['ON'][1]:10.3f}")
    print("\nclaim: coordR2 ~1 at coh=0 (orthogonal=easy); ON should hold up better "
          "than OFF as coh->1 (planes colinear); sigma_min should track the cliff.")
