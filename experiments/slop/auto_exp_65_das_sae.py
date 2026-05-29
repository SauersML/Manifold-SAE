"""auto_exp_65: DAS-SAE vs vanilla L1 SAE on the hue-swap identification test.

Hypothesis: DAS-SAE produces a SMALL set of features that score > 0.7 on the
per-feature hue-swap test (decode(swap(z_b, z_a, e_i)) ≈ target_swap), while
a vanilla L1 SAE produces FEWER such features — its hue information is
smeared across many entangled atoms, none of which causally encode hue
alone.

We train two F=256 SAEs on the cogito-L40 color manifold for the same number
of epochs:
  1) DAS-SAE with interchange-intervention loss (λ_intv = 1.0)
  2) vanilla L1 SAE (λ_intv = 0, all other hyperparams identical)

Both are evaluated on the same held-out color-pair set. Reported metrics:
  - val reconstruction R²
  - val interchange R² (using the LEARNED hue_mask for DAS-SAE; using the
    mask of features ABOVE 0.5 swap-score for the vanilla SAE)
  - per-feature swap-test R² distribution
  - # features above τ=0.7 ("identified hue features")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors

from manifold_sae.das_sae import (
    DASSAE,
    DASSAEConfig,
    build_target_swap,
    fit_hue_direction,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_DIR = ROOT / "runs" / "auto_exp_65_das_sae"
N_TEMPLATES = 28

# Smaller / faster regime so the experiment finishes quickly.
F = 256
TOP_K = 32
EPOCHS = 8
BATCH = 96
LR = 3e-4
SEED = 65
THRESHOLD = 0.7


def load_xkcd_rgb(n_colors: int):
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((
                int(hexs[0:2], 16) / 255.0,
                int(hexs[2:4], 16) / 255.0,
                int(hexs[4:6], 16) / 255.0,
            ))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float32)


def per_color_centroids(X, n_t):
    n_rows, d = X.shape
    n_c = n_rows // n_t
    out = np.empty((n_c, d), dtype=np.float32)
    block = 64
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        chunk = np.asarray(X[cs * n_t : ce * n_t], dtype=np.float32)
        chunk = chunk.reshape(ce - cs, n_t, d).mean(axis=1)
        out[cs:ce] = chunk
    return out


def r2(pred, target):
    res = (pred - target).pow(2).sum().item()
    var = (target - target.mean(0, keepdim=True)).pow(2).sum().item()
    return 1.0 - res / max(var, 1e-12)


def make_pairs(n, n_pairs, rng):
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    same = a == b
    while np.any(same):
        b[same] = rng.integers(0, n, size=same.sum())
        same = a == b
    return a, b


def train_one(name, lambda_intv, X_t, hue_t, v_hue, tr_idx, val_idx, device, rng):
    cfg = DASSAEConfig(input_dim=X_t.shape[1], n_features=F, top_k=TOP_K)
    sae = DASSAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    n_steps = max(BATCH * 8, len(tr_idx) * 4) // BATCH

    hist = []
    for ep in range(EPOCHS):
        sae.train()
        agg = {"loss": 0.0, "recon": 0.0, "intv": 0.0}
        for _ in range(n_steps):
            a_idx, b_idx = make_pairs(len(tr_idx), BATCH, rng)
            x_a = X_t[tr_idx[a_idx]]
            x_b = X_t[tr_idx[b_idx]]
            ha = hue_t[tr_idx[a_idx]]
            hb = hue_t[tr_idx[b_idx]]
            tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
            losses = sae.compute_loss(
                x_a, x_b, tgt,
                lambda_intv=lambda_intv,
                lambda_gate=5e-4 if lambda_intv > 0 else 0.0,
                lambda_l1=1e-3,
                lambda_gate_entropy=1e-2 if lambda_intv > 0 else 0.0,
            )
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder_columns_()
            for k in agg:
                agg[k] += float(losses[k].item())
        for k in agg:
            agg[k] /= n_steps

        sae.eval()
        with torch.no_grad():
            x_val = X_t[val_idx]
            r2_recon = r2(sae(x_val).x_hat, x_val)
        hist.append({"epoch": ep, **agg, "val_recon_r2": r2_recon})
        print(f"[{name} ep{ep:02d}] recon={agg['recon']:.4f} intv={agg['intv']:.4f} "
              f"val_R²={r2_recon:.3f}")

    # Per-feature swap test on a fresh batch of val pairs.
    with torch.no_grad():
        a_idx, b_idx = make_pairs(len(val_idx), min(128, len(val_idx) * 4), rng)
        x_a = X_t[val_idx[a_idx]]
        x_b = X_t[val_idx[b_idx]]
        ha = hue_t[val_idx[a_idx]]
        hb = hue_t[val_idx[b_idx]]
        tgt = build_target_swap(x_a, x_b, ha, hb, v_hue)
        scores = sae.per_feature_swap_score(x_a, x_b, tgt).cpu().numpy()
        n_above = int((scores > THRESHOLD).sum())
        # Evaluate group-swap R² (use learned mask for DAS, threshold-pass set for vanilla).
        if lambda_intv > 0:
            mask = sae.hue_mask()
        else:
            mask_np = (scores > THRESHOLD).astype(np.float32)
            mask = torch.from_numpy(mask_np).to(device)
        za = sae(x_a).z; zb = sae(x_b).z
        z_swap = sae.swap(zb, za, mask=mask)
        x_swap_hat = sae.decode(z_swap)
        group_intv_r2 = r2(x_swap_hat, tgt)
        final_val_r2 = r2(sae(X_t[val_idx]).x_hat, X_t[val_idx])

    return {
        "name": name,
        "history": hist,
        "scores": scores,
        "n_identified": n_above,
        "group_intv_r2": group_intv_r2,
        "final_val_recon_r2": final_val_r2,
        "hue_mask": sae.hue_mask().detach().cpu().numpy() if lambda_intv > 0 else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[auto_exp_65] device={device}")

    X_mmap = np.load(X_PATH, mmap_mode="r")
    centroids = per_color_centroids(X_mmap, N_TEMPLATES)
    n_c, D = centroids.shape
    names, rgb = load_xkcd_rgb(n_c)
    hsv = np.stack([mcolors.rgb_to_hsv(rgb[i]) for i in range(n_c)], 0).astype(np.float32)
    hue_np = hsv[:, 0]

    mu = centroids.mean(0, keepdims=True)
    sd = centroids.std(0, keepdims=True).clip(min=1e-6)
    X_std = (centroids - mu) / sd
    X_t = torch.from_numpy(X_std).to(device)
    hue_t = torch.from_numpy(hue_np).to(device)

    perm = rng.permutation(n_c)
    n_val = max(48, n_c // 6)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    v_hue = fit_hue_direction(X_t[tr_idx], hue_t[tr_idx])
    print(f"[setup] n_c={n_c} D={D} F={F} train={len(tr_idx)} val={len(val_idx)}")

    t0 = time.time()
    print("\n=== Training DAS-SAE (λ_intv=1.0) ===")
    das = train_one("DAS", 1.0, X_t, hue_t, v_hue, tr_idx, val_idx, device, np.random.default_rng(SEED))
    print("\n=== Training vanilla L1 SAE (λ_intv=0) ===")
    vanilla = train_one("VAN", 0.0, X_t, hue_t, v_hue, tr_idx, val_idx, device, np.random.default_rng(SEED + 1))
    runtime = time.time() - t0

    # Hypothesis check.
    delta = das["n_identified"] - vanilla["n_identified"]
    hyp_das_more = das["n_identified"] > vanilla["n_identified"]
    hyp_das_small = das["n_identified"] <= max(32, F // 8)
    hyp_das_intv = das["group_intv_r2"] > vanilla["group_intv_r2"]

    summary = {
        "config": {"F": F, "top_k": TOP_K, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "seed": SEED, "threshold": THRESHOLD},
        "n_colors": int(n_c), "D": int(D),
        "das": {
            "n_identified": das["n_identified"],
            "group_intv_r2": das["group_intv_r2"],
            "final_val_recon_r2": das["final_val_recon_r2"],
            "history": das["history"],
            "scores_top10": np.sort(das["scores"])[-10:].tolist(),
            "hue_mask_n_above_0.5": int((das["hue_mask"] > 0.5).sum()),
        },
        "vanilla": {
            "n_identified": vanilla["n_identified"],
            "group_intv_r2": vanilla["group_intv_r2"],
            "final_val_recon_r2": vanilla["final_val_recon_r2"],
            "history": vanilla["history"],
            "scores_top10": np.sort(vanilla["scores"])[-10:].tolist(),
        },
        "delta_identified_das_minus_vanilla": int(delta),
        "hypotheses": {
            "das_finds_more_hue_features": bool(hyp_das_more),
            "das_hue_set_is_small": bool(hyp_das_small),
            "das_group_swap_better": bool(hyp_das_intv),
        },
        "runtime_seconds": runtime,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT_DIR / "scores.npz",
             das_scores=das["scores"], van_scores=vanilla["scores"],
             hue_mask=das["hue_mask"])

    # Plot: score distributions + identification counts + interchange R².
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ax = axs[0]
    bins = np.linspace(-0.5, 1.0, 40)
    ax.hist(vanilla["scores"], bins=bins, alpha=0.55, label="vanilla L1", color="#1f77b4")
    ax.hist(das["scores"], bins=bins, alpha=0.55, label="DAS-SAE", color="#d62728")
    ax.axvline(THRESHOLD, color="k", ls="--", lw=1, label=f"τ={THRESHOLD}")
    ax.set_xlabel("per-feature hue-swap R²"); ax.set_ylabel("# features")
    ax.set_title("Distribution of per-feature swap scores")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[1]
    ax.bar(["vanilla L1", "DAS-SAE"],
           [vanilla["n_identified"], das["n_identified"]],
           color=["#1f77b4", "#d62728"])
    ax.set_ylabel(f"# features with swap R² > {THRESHOLD}")
    ax.set_title("Identified hue features")
    for i, v in enumerate([vanilla["n_identified"], das["n_identified"]]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y")

    ax = axs[2]
    ep = [h["epoch"] for h in das["history"]]
    ax.plot(ep, [h["intv"] for h in das["history"]], label="DAS train intv loss", color="#d62728")
    ax.plot(ep, [h["recon"] for h in das["history"]], label="DAS train recon loss", color="#d62728", ls=":")
    ax.plot(ep, [h["recon"] for h in vanilla["history"]], label="vanilla train recon", color="#1f77b4", ls=":")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training curves")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_exp_65: DAS-SAE vs vanilla L1 | identified hue features: "
        f"DAS={das['n_identified']} vs VAN={vanilla['n_identified']} | "
        f"group-swap R²: DAS={das['group_intv_r2']:.3f} vs VAN={vanilla['group_intv_r2']:.3f}",
        fontsize=11, y=1.04,
    )
    fig.savefig(OUT_DIR / "auto_exp_65.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Summary ===")
    print(json.dumps(summary["hypotheses"], indent=2))
    print(f"DAS identified: {das['n_identified']} | vanilla identified: {vanilla['n_identified']}")
    print(f"DAS group-swap R²: {das['group_intv_r2']:.3f} | vanilla: {vanilla['group_intv_r2']:.3f}")
    print(f"runtime={runtime:.1f}s")
    return summary


if __name__ == "__main__":
    main()
