"""End-to-end behavioral-probe training on local behavioral activations.

Pipeline:
  1. Load harvested behavioral activations X (N, D) + labels.
  2. Train a lightweight TopK SAE on X to get atom activations a (N, F).
     - We use TopK (not the full Manifold-SAE gamfit path) because N is
       tiny (60 prompts) and gamfit's REML needs more samples per atom
       than that. The TopK SAE is the Manifold-SAE atom layer with a
       trivial (per-atom-constant) curve, which is the right baseline for
       a behavioral-interp first pass.
  3. Train 3 BehavioralProbes (refusal, sycophancy, hedging) on a.
  4. Run causal-steer evaluation on each.
  5. Save runs/behavioral_probes/report.md.

CAVEAT: This is a sanity-scale evaluation. With N=60 the val-set per probe
is ~15 examples. Accuracies are reported with the val-N alongside them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.behavioral.probes import BehavioralProbe, cross_correlation
from manifold_sae.behavioral.causal_steer import causal_steer_eval


# ----------------------------------------------------------------------
# Tiny TopK SAE (numpy/torch only). One file so we can ship without
# touching the heavy gamfit ManifoldSAE training loop.
# ----------------------------------------------------------------------

class TopKSAE(nn.Module):
    def __init__(self, D: int, F: int, top_k: int = 32) -> None:
        super().__init__()
        self.D = D
        self.F = F
        self.top_k = min(top_k, F)
        self.W_enc = nn.Parameter(torch.randn(D, F) * (1.0 / np.sqrt(D)))
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.W_dec = nn.Parameter(torch.randn(F, D) * (1.0 / np.sqrt(F)))
        self.b_dec = nn.Parameter(torch.zeros(D))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # pre-activations, then TopK
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.top_k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return out

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a


def train_topk_sae(X: np.ndarray, F: int = 128, top_k: int = 16,
                   epochs: int = 1500, lr: float = 1e-2,
                   device: str = "cpu", seed: int = 0) -> TopKSAE:
    torch.manual_seed(seed)
    D = X.shape[1]
    sae = TopKSAE(D=D, F=F, top_k=top_k).to(device)
    # Init b_dec to mean.
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor(X.mean(0), dtype=torch.float32, device=device))
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        x_hat, a = sae(Xt)
        loss = ((x_hat - Xt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 250 == 0:
            with torch.no_grad():
                ev = 1.0 - ((x_hat - Xt) ** 2).mean() / Xt.var()
            print(f"  [sae] ep={ep+1:4d}  loss={loss.item():.4e}  EV={ev.item():.3f}  t={time.time()-t0:.1f}s", flush=True)
    return sae


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=None, help="path to run dir with X.npy + labels.json")
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--out", default=str(ROOT / "runs" / "behavioral_probes"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # auto-discover most recent BEHAVIORAL_* if not given
    if args.harvest is None:
        cands = sorted((ROOT / "runs").glob("BEHAVIORAL_*"))
        if not cands:
            sys.exit("no BEHAVIORAL_* run dir; run scripts/harvest_behavioral_local.py first")
        args.harvest = str(cands[-1])
    h = Path(args.harvest)
    X = np.load(h / "X.npy")
    meta = json.loads((h / "labels.json").read_text())
    print(f"[main] X={X.shape} from {h.name} (synthetic={meta.get('synthetic_fallback')})", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train SAE ----
    sae = train_topk_sae(X, F=args.F, top_k=args.top_k, epochs=args.epochs, device=args.device)
    with torch.no_grad():
        A = sae.encode(torch.tensor(X, dtype=torch.float32, device=args.device)).cpu().numpy()
    torch.save({"state_dict": sae.state_dict(), "D": int(X.shape[1]), "F": args.F, "top_k": args.top_k},
               out_dir / "sae.pt")
    np.save(out_dir / "atoms.npy", A)
    print(f"[main] SAE trained, atom-acts shape={A.shape}, mean-active={(A>0).mean()*A.shape[1]:.1f}", flush=True)

    # ---- Build labels ----
    labels_list = meta["labels"]
    n = len(labels_list)
    assert n == A.shape[0], (n, A.shape)
    targets = ["refusal", "sycophancy", "hedging"]
    y_by_target = {t: np.array([int(d.get(t, 0)) for d in labels_list], dtype=np.float32) for t in targets}

    # Held-out 25 % per probe (the probe.fit does its own stratified split,
    # but we also keep a second held-out set for steering — by holding out
    # the LAST 25 % of each class. Tiny N → just take alternating indices).
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_steer = max(8, n // 4)
    steer_idx = perm[:n_steer]
    fit_idx = perm[n_steer:]

    probes: dict[str, BehavioralProbe] = {}
    reports = {}
    steer_results = {}
    for t in targets:
        y_fit = y_by_target[t][fit_idx]
        if y_fit.sum() < 2 or (len(y_fit) - y_fit.sum()) < 2:
            # not enough positives or negatives — fall back to all data for fit, no steer-holdout.
            X_fit, y_fit_full = A, y_by_target[t]
            X_steer = A
        else:
            X_fit, y_fit_full = A[fit_idx], y_by_target[t][fit_idx]
            X_steer = A[steer_idx]
        probe = BehavioralProbe(n_atoms=args.F, target=t, l2=1e-3)
        rep = probe.fit(X_fit, y_fit_full, val_split=0.25, epochs=200, lr=0.5, device=args.device, seed=0)
        probes[t] = probe
        reports[t] = rep
        steer = causal_steer_eval(probe, X_steer, top_k=10, alphas=(1.0, 2.0, 5.0), device=args.device)
        steer_results[t] = steer
        print(f"[probe:{t}] train_acc={rep.train_acc:.2f} val_acc={rep.val_acc:.2f} "
              f"val_auc={rep.val_auc:.2f}  steer Δp@α=1: {steer.delta_p[1.0]:+.3f}", flush=True)

    # ---- Cross-correlation ----
    xc = cross_correlation(probes, top_k=10)
    print("[main] top-10 atom Jaccard:", json.dumps(xc, indent=2), flush=True)

    # ---- Write report.md ----
    lines = []
    lines.append("# Behavioral-Probe Report\n")
    lines.append(f"Source: `{h}`")
    lines.append(f"Activations: shape={list(X.shape)}, synthetic_fallback={meta.get('synthetic_fallback')}")
    lines.append(f"SAE: TopK F={args.F} top_k={args.top_k} epochs={args.epochs}\n")
    lines.append(f"Total prompts={n}, fit-set={len(fit_idx)}, steer-holdout={len(steer_idx)}\n")
    lines.append("## Per-behavior\n")
    for t in targets:
        rep = reports[t]
        steer = steer_results[t]
        pos = int(y_by_target[t].sum())
        lines.append(f"### {t}  (positives={pos}/{n})")
        lines.append(f"- train_acc = **{rep.train_acc:.3f}**  (n_train={rep.n_train})")
        lines.append(f"- val_acc   = **{rep.val_acc:.3f}**   (n_val={rep.n_val})")
        lines.append(f"- val_auc   = **{rep.val_auc:.3f}**")
        lines.append(f"- top-10 atoms (idx, signed weight):")
        for i, w in rep.top_atoms[:10]:
            firing_rate = float((A[:, i] > 0).mean())
            lines.append(f"    - atom {i:4d}  w={w:+.3f}  firing_rate={firing_rate:.2f}")
        lines.append(f"- causal steering (top-10 atoms, push +α along probe-positive sign):")
        lines.append(f"    - baseline P(behavior) = {steer.baseline_p_mean:.3f}")
        for a in steer.alphas:
            lines.append(f"    - α={a:.1f}: P={steer.steered_p_mean[a]:.3f}  ΔP={steer.delta_p[a]:+.3f}  flip_rate={steer.flip_rate[a]:.2f}")
        lines.append("")

    lines.append("## Cross-correlation (top-10 atom Jaccard)\n")
    keys = list(probes.keys())
    lines.append("| | " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for a in keys:
        lines.append(f"| **{a}** | " + " | ".join(f"{xc[a][b]:.2f}" for b in keys) + " |")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))
    # Also dump machine-readable.
    json.dump(
        {
            "harvest": str(h),
            "synthetic_fallback": meta.get("synthetic_fallback"),
            "F": args.F, "top_k": args.top_k, "epochs": args.epochs,
            "probes": {
                t: {
                    "train_acc": reports[t].train_acc,
                    "val_acc": reports[t].val_acc,
                    "val_auc": reports[t].val_auc,
                    "n_train": reports[t].n_train,
                    "n_val": reports[t].n_val,
                    "top_atoms": reports[t].top_atoms[:10],
                }
                for t in targets
            },
            "steer": {
                t: {
                    "baseline_p": steer_results[t].baseline_p_mean,
                    "delta_p": steer_results[t].delta_p,
                    "flip_rate": steer_results[t].flip_rate,
                    "top_atoms": steer_results[t].top_atoms,
                }
                for t in targets
            },
            "cross_correlation_jaccard_top10": xc,
        },
        open(out_dir / "report.json", "w"),
        indent=2,
    )
    print(f"[main] wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
