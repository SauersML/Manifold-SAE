"""auto_exp_70 — End-to-end behavioral-probe sweep on Manifold-SAE atoms.

Steps:
  0. (optional) Harvest behavioral activations from Qwen 2.5 1.5B Instruct
     locally. Skipped if runs/BEHAVIORAL_*/X.npy already exists.
  1. Train a small TopK SAE on the harvested activations.
  2. Train BehavioralProbes: refusal, sycophancy, hedging.
  3. Causal-steer evaluation per probe (α=1, 2, 5).
  4. Cross-correlation table:
       - refusal vs sycophancy vs hedging  (do they share atoms?)
       - behavior atoms vs "color-mention" content atoms  (do behavior atoms
         live in an orthogonal subspace from content atoms?)
     "color-mention" is a content baseline: probe label = 1 if the prompt
     mentions a color word (red, blue, green, ...). This is the local
     equivalent of "do behavior atoms overlap with the cogito-hue atoms?" —
     we cannot directly compare to the L40 hue probes because the behavioral
     model + layer + dimension differs, but the in-same-SAE color content
     probe is the analogous test.

Saves to runs/auto_exp_70/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from manifold_sae.behavioral.probes import BehavioralProbe, cross_correlation  # noqa: E402
from manifold_sae.behavioral.causal_steer import causal_steer_eval  # noqa: E402


OUT = ROOT / "runs" / "auto_exp_70"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 0. Harvest (only if missing)
# ----------------------------------------------------------------------

def ensure_harvest() -> Path:
    existing = sorted((ROOT / "runs").glob("BEHAVIORAL_*"))
    if existing:
        print(f"[exp70] reusing harvest {existing[-1].name}", flush=True)
        return existing[-1]
    print("[exp70] no harvest found — running scripts/harvest_behavioral_local.py", flush=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "harvest_behavioral_local.py"),
           "--n", "10", "--layer", "12"]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"[exp70] harvest failed/timed out ({e!r}); retrying with --synthetic",
              flush=True)
        subprocess.run(cmd + ["--synthetic"], check=True)
    return sorted((ROOT / "runs").glob("BEHAVIORAL_*"))[-1]


# ----------------------------------------------------------------------
# Tiny TopK SAE (duplicated here so the experiment is self-contained)
# ----------------------------------------------------------------------

class TopKSAE(torch.nn.Module):
    def __init__(self, D: int, F: int, top_k: int) -> None:
        super().__init__()
        self.D, self.F, self.top_k = D, F, min(top_k, F)
        self.W_enc = torch.nn.Parameter(torch.randn(D, F) * (1 / np.sqrt(D)))
        self.b_enc = torch.nn.Parameter(torch.zeros(F))
        self.W_dec = torch.nn.Parameter(torch.randn(F, D) * (1 / np.sqrt(F)))
        self.b_dec = torch.nn.Parameter(torch.zeros(D))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        tv, ti = pre.topk(self.top_k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, ti, torch.relu(tv))
        return out

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        a = self.encode(x)
        return self.decode(a), a


def train_sae(X: np.ndarray, F: int = 128, top_k: int = 16, epochs: int = 1500,
              lr: float = 1e-2, device: str = "cpu") -> TopKSAE:
    torch.manual_seed(0)
    sae = TopKSAE(X.shape[1], F, top_k).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor(X.mean(0), dtype=torch.float32, device=device))
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for ep in range(epochs):
        x_hat, a = sae(Xt)
        loss = ((x_hat - Xt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sae


# ----------------------------------------------------------------------
# content-baseline: color-mention label
# ----------------------------------------------------------------------

COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "violet", "indigo", "turquoise", "scarlet",
}


def color_mention_labels(labels_list: list[dict]) -> np.ndarray:
    out = []
    for d in labels_list:
        toks = set(d["prompt"].lower().split())
        # crude — strip punctuation
        toks = {t.strip(".,!?;:'\"") for t in toks}
        out.append(1 if toks & COLOR_WORDS else 0)
    return np.array(out, dtype=np.float32)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    harvest = ensure_harvest()
    X = np.load(harvest / "X.npy")
    meta = json.loads((harvest / "labels.json").read_text())
    print(f"[exp70] X={X.shape}  synthetic={meta.get('synthetic_fallback')}", flush=True)

    F, top_k, epochs = 128, 16, 1500
    sae = train_sae(X, F=F, top_k=top_k, epochs=epochs)
    with torch.no_grad():
        A = sae.encode(torch.tensor(X, dtype=torch.float32)).numpy()
    print(f"[exp70] SAE trained — atom-acts {A.shape}  density={(A>0).mean():.3f}", flush=True)

    labels_list = meta["labels"]
    n = len(labels_list)
    targets = ["refusal", "sycophancy", "hedging"]
    y_all = {t: np.array([int(d.get(t, 0)) for d in labels_list], dtype=np.float32)
             for t in targets}
    y_all["color_mention"] = color_mention_labels(labels_list)

    # Steer-holdout: ~25 %
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    nh = max(8, n // 4)
    steer_idx, fit_idx = perm[:nh], perm[nh:]

    probes: dict[str, BehavioralProbe] = {}
    steer_res = {}
    rep_dump = {}
    for t in y_all:
        y = y_all[t]
        if y[fit_idx].sum() < 2 or (len(fit_idx) - y[fit_idx].sum()) < 2:
            Xf, yf = A, y
            Xs = A
        else:
            Xf, yf = A[fit_idx], y[fit_idx]
            Xs = A[steer_idx]
        p = BehavioralProbe(n_atoms=F, target=t, l2=1e-3)
        rep = p.fit(Xf, yf, val_split=0.25, epochs=200, lr=0.5, seed=0)
        probes[t] = p
        rep_dump[t] = {
            "train_acc": rep.train_acc, "val_acc": rep.val_acc, "val_auc": rep.val_auc,
            "n_train": rep.n_train, "n_val": rep.n_val,
            "top_atoms": rep.top_atoms[:10],
            "positives": int(y.sum()), "n": int(n),
        }
        if t in targets:
            steer = causal_steer_eval(p, Xs, top_k=10, alphas=(1.0, 2.0, 5.0))
            steer_res[t] = {
                "baseline_p": steer.baseline_p_mean,
                "delta_p": steer.delta_p,
                "flip_rate": steer.flip_rate,
                "top_atoms": steer.top_atoms,
            }

    # Cross-correlation: include content baseline.
    xc = cross_correlation(probes, top_k=10)
    print("[exp70] cross-corr (top-10 Jaccard):", json.dumps(xc, indent=2), flush=True)

    # Save
    out = {
        "harvest": str(harvest),
        "synthetic_fallback": meta.get("synthetic_fallback"),
        "n_total": int(n),
        "n_fit": int(len(fit_idx)),
        "n_steer_holdout": int(len(steer_idx)),
        "F": F, "top_k": top_k, "epochs": epochs,
        "probes": rep_dump,
        "steer": steer_res,
        "cross_correlation_jaccard_top10": xc,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))

    # human-readable report
    L = []
    L.append("# auto_exp_70: Behavioral Probes on Manifold-SAE Atoms\n")
    L.append(f"Harvest: `{harvest}`")
    L.append(f"Activations: {X.shape}  synthetic_fallback={meta.get('synthetic_fallback')}")
    L.append(f"TopK SAE: F={F} top_k={top_k} epochs={epochs}")
    L.append(f"N total={n}, fit={len(fit_idx)}, steer-holdout={len(steer_idx)}\n")
    L.append("## Probes\n| target | positives | val_acc | val_auc | ΔP@α=1 | ΔP@α=5 |")
    L.append("|---|---|---|---|---|---|")
    for t in targets:
        s = steer_res[t]
        L.append(f"| {t} | {rep_dump[t]['positives']}/{n} | "
                 f"{rep_dump[t]['val_acc']:.2f} | {rep_dump[t]['val_auc']:.2f} | "
                 f"{s['delta_p'][1.0]:+.3f} | {s['delta_p'][5.0]:+.3f} |")
    L.append("")
    L.append("## Cross-correlation (Jaccard of top-10 atoms)\n")
    keys = list(probes.keys())
    L.append("| | " + " | ".join(keys) + " |")
    L.append("|" + "---|" * (len(keys) + 1))
    for a in keys:
        L.append(f"| **{a}** | " + " | ".join(f"{xc[a][b]:.2f}" for b in keys) + " |")
    L.append("")
    L.append("## Orthogonality finding\n")
    pair_means = []
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            pair_means.append(xc[a][b])
    mean_behavior_overlap = float(np.mean(pair_means)) if pair_means else float("nan")
    color_overlap = float(np.mean([xc[t]["color_mention"] for t in targets]))
    L.append(f"- mean behavior-vs-behavior Jaccard = **{mean_behavior_overlap:.2f}**")
    L.append(f"- mean behavior-vs-color_mention Jaccard = **{color_overlap:.2f}**")
    if mean_behavior_overlap > color_overlap + 0.05:
        verdict = "Behavior atoms share more with each other than with content atoms — partial orthogonality of behavior vs content."
    elif mean_behavior_overlap < color_overlap - 0.05:
        verdict = "Behavior atoms overlap more with content than with each other — at this scale the probes appear to be picking up content confounds."
    else:
        verdict = "Behavior- and content-atom overlap are comparable; no strong orthogonality signal at this sample size."
    L.append(f"- verdict: {verdict}")
    (OUT / "report.md").write_text("\n".join(L))
    print(f"[exp70] wrote {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
