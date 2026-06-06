"""Statistical-rigor + manifold analysis across training checkpoints.

COLOR manifold (frame-demeaned, L44 by default):
  * Mantel test rep-distance(1-cos) vs true-RGB-distance -> Pearson r + permutation p.
  * Mantel test rep-distance vs CIRCULAR-HUE distance -> does it recover the hue circle?
  * participation ratio (intrinsic dimensionality) of the 30-color manifold.
SELF / QUALIA axis (L~40%):
  * self_qualia_coord + bootstrap 95% CI; one-sample vs 0.5 (is the self experiencer-side?).
  * qualia AUC + permutation p (shuffle exp/noexp labels).
  * null-control AUC (should bracket 0.5).
Runs across all stages in model-flow order and writes a CSV + trajectory plot.
Read-only.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

STAGE_ORDER = [("pretrain", "OLMO3_32B_TRAJ"), ("SFT", "OLMO3_32B_TRAJ_SFT"),
               ("DPO", "OLMO3_32B_TRAJ_DPO"), ("RL3.0", "OLMO3_32B_TRAJ_RL"),
               ("RL3.1", "OLMO3_32B_TRAJ_RL31")]
RNG = np.random.RandomState(0)


def _key(n):
    m = re.search(r"stage(\d+).*?step(\d+)", n) or re.search(r"step[_-]?(\d+)", n)
    return (int(m.group(1)), int(m.group(2))) if (m and m.lastindex == 2) else ((0, int(m.group(1))) if m else (0, 0))


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def _mantel(Da, Db, perms=499):
    n = Da.shape[0]; iu = np.triu_indices(n, 1)
    a = Da[iu]; r = _pearson(a, Db[iu])
    cnt = 1
    for _ in range(perms):
        p = RNG.permutation(n)
        if abs(_pearson(a, Db[np.ix_(p, p)][iu])) >= abs(r):
            cnt += 1
    return r, cnt / (perms + 1)


def _auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def color_stats(extra, layer):
    X = np.load(Path(extra) / "activations.npy"); recs = [json.loads(l) for l in open(Path(extra) / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.asarray(r["rgb"], float)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    Drep = 1 - Vn @ Vn.T
    truth = np.stack([rgb[c] for c in cols]); Drgb = np.sqrt(((truth[:, None] - truth[None]) ** 2).sum(-1))
    hues = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255.0))[0] for c in cols])
    Dhue = np.abs(hues[:, None] - hues[None]); Dhue = np.minimum(Dhue, 1 - Dhue)
    r_rgb, p_rgb = _mantel(Drep, Drgb)
    r_hue, p_hue = _mantel(Drep, Dhue)
    Vc = V - V.mean(0); ev = np.linalg.svd(Vc, compute_uv=False) ** 2
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())   # participation ratio
    return {"color_layer": L, "mantel_rgb_r": r_rgb, "mantel_rgb_p": p_rgb,
            "mantel_hue_r": r_hue, "mantel_hue_p": p_hue, "color_participation_ratio": pr}


def self_stats(ckpt, lpct=0.40):
    X = np.load(Path(ckpt) / "activations.npy"); recs = [json.loads(l) for l in open(Path(ckpt) / "prompts.jsonl") if l.strip()]
    L = int(round(lpct * (X.shape[1] - 1))); H = X[:, L, :].astype(np.float64)
    role = np.array([r["role"] for r in recs]); side = np.array([r["side"] for r in recs])
    kind = np.array([r["kind"] for r in recs])
    ie = np.where((role == "pair") & (side == "exp"))[0]; ino = np.where((role == "pair") & (side == "noexp"))[0]
    isf = np.where((role == "self") & (side == "-") & (kind == "self"))[0]
    if len(isf) == 0:
        isf = np.where(role == "self")[0]
    axis = H[ie].mean(0) - H[ino].mean(0); axis /= max(np.linalg.norm(axis), 1e-9)
    s = H @ axis; lo, hi = s[ino].mean(), s[ie].mean()
    coord = (s - lo) / (hi - lo + 1e-12)
    selfc = coord[isf]
    boot = [selfc[RNG.randint(0, len(selfc), len(selfc))].mean() for _ in range(999)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # qualia AUC + permutation p
    lab = np.r_[np.ones(len(ie)), np.zeros(len(ino))]; sc = np.r_[s[ie], s[ino]]
    auc = _auc(sc, lab)
    cnt = 1; P = 199
    for _ in range(P):
        pl = RNG.permutation(lab)
        # rebuild axis under permuted labels (proper null)
        ax2 = sc_perm = None
        idx = np.r_[ie, ino]; e2 = idx[pl == 1]; n2 = idx[pl == 0]
        ax2 = H[e2].mean(0) - H[n2].mean(0); ax2 /= max(np.linalg.norm(ax2), 1e-9)
        s2 = H @ ax2; a2 = _auc(np.r_[s2[e2], s2[n2]], np.r_[np.ones(len(e2)), np.zeros(len(n2))])
        if a2 >= auc:
            cnt += 1
    auc_p = cnt / (P + 1)
    inu = np.where(role == "null_pair")[0]
    null_auc = float("nan")
    if len(inu):
        a = side[inu]; aa = inu[a == "a"]; bb = inu[a == "b"]
        if len(aa) and len(bb):
            null_auc = _auc(np.r_[s[aa], s[bb]], np.r_[np.ones(len(aa)), np.zeros(len(bb))])
    return {"self_layer": L, "self_coord": float(selfc.mean()), "self_ci_lo": ci[0], "self_ci_hi": ci[1],
            "self_gt_half": bool(ci[0] > 0.5), "qualia_auc": auc, "qualia_auc_perm_p": auc_p,
            "null_auc": null_auc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs"); ap.add_argument("--out", default="runs/ANALYSIS")
    ap.add_argument("--color-layer", type=int, default=44)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []; gi = 0
    for label, sd in STAGE_ORDER:
        base = Path(args.runs_root) / sd
        if not base.exists():
            continue
        for d in sorted([x for x in glob.glob(f"{base}/*/") if os.path.exists(os.path.join(x, "done.json"))],
                        key=lambda x: _key(Path(x).name)):
            row = {"global_idx": gi, "stage": label, "checkpoint": Path(d).name}
            try:
                row.update(self_stats(d))
            except Exception as e:
                print("self_stats fail", d, e)
            ex = Path(d) / "extra"
            if (ex / "activations.npy").exists():
                try:
                    row.update(color_stats(ex, args.color_layer))
                except Exception as e:
                    print("color_stats fail", d, e)
            rows.append(row); gi += 1
            print("  %-26s self=%.2f[%.2f,%.2f] aucP=%.3f | color rgb_r=%.2f(p=%.3f) hue_r=%.2f PR=%.1f"
                  % (row["checkpoint"], row.get("self_coord", float('nan')), row.get("self_ci_lo", float('nan')),
                     row.get("self_ci_hi", float('nan')), row.get("qualia_auc_perm_p", float('nan')),
                     row.get("mantel_rgb_r", float('nan')), row.get("mantel_rgb_p", float('nan')),
                     row.get("mantel_hue_r", float('nan')), row.get("color_participation_ratio", float('nan'))))
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "global_idx", k))
    with open(out / "stats_manifold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[stats] {len(rows)} ckpts -> {out/'stats_manifold.csv'}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["global_idx"] for r in rows]; stages = [r["stage"] for r in rows]
        bounds = [i for i in range(1, len(rows)) if stages[i] != stages[i - 1]]
        fig, a = plt.subplots(3, 1, figsize=(max(9, len(rows) * 0.28), 10), sharex=True, constrained_layout=True)
        a[0].plot(x, [r.get("self_coord") for r in rows], "o-", ms=3, color="#d62728", label="self coord")
        a[0].fill_between(x, [r.get("self_ci_lo") for r in rows], [r.get("self_ci_hi") for r in rows], color="#d62728", alpha=0.2)
        a[0].axhline(0.5, color="0.7", lw=0.8, ls="--"); a[0].set_ylabel("self qualia coord\n(95% CI)"); a[0].legend(fontsize=7)
        a[0].set_title("Statistical trajectory: self coord (CI), color↔RGB/HUE Mantel, color intrinsic-dim")
        a[1].plot(x, [r.get("mantel_rgb_r") for r in rows], "o-", ms=3, color="#6a0dad", label="rep↔RGB Mantel r")
        a[1].plot(x, [r.get("mantel_hue_r") for r in rows], "s-", ms=3, color="#1f6feb", label="rep↔HUE Mantel r")
        a[1].axhline(0, color="0.8", lw=0.7); a[1].set_ylabel("Mantel r"); a[1].legend(fontsize=7)
        a[2].plot(x, [r.get("color_participation_ratio") for r in rows], "o-", ms=3, color="#e08e0b", label="color participation ratio (dim)")
        a[2].set_ylabel("intrinsic dim"); a[2].legend(fontsize=7)
        for b in bounds:
            for ax in a:
                ax.axvline(b - 0.5, color="0.6", lw=0.8, ls="--")
        a[2].set_xticks(x); a[2].set_xticklabels([r["checkpoint"][:12] for r in rows], rotation=90, fontsize=5)
        fig.savefig(out / "stats_manifold.png", dpi=160); plt.close(fig)
        print(f"[stats] wrote {out/'stats_manifold.png'}")
    except Exception as e:
        print("plot skip", e)


if __name__ == "__main__":
    main()
