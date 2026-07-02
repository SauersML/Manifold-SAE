"""Train + evaluate the BSF baseline: synthetic recovery, real-activation EV
sweep, and the weekday/month cyclic-feature block finding.

Run everything (writes metrics.json + REPORT.md into this directory):

    .venv/bin/python experiments/bsf_baseline/train.py

Individual phases:

    ... train.py --phase synthetic
    ... train.py --phase real
    ... train.py --phase cyclic

Everything is laptop-scale CPU float64. The real-activation phase reuses the
cached OLMo self/qualia residual harvest under ``runs/`` (no re-harvest); the
cyclic phase reuses ``experiments/probe_out/harvest_{weekday,month}.npz``.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from bsf import (  # noqa: E402
    BSF,
    BSFConfig,
    TrainConfig,
    block_diagnostics,
    ev,
    pca_reduce,
    reconstruct,
    stable_rank,
    train_bsf,
)

torch.set_default_dtype(torch.float64)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "6")
torch.set_num_threads(6)

OUT = HERE


# ==========================================================================
# subspace / ordering helpers
# ==========================================================================
def orthonormal_block_basis(dec_g: np.ndarray) -> np.ndarray:
    """Orthonormal basis (d, b) spanning block g's decoder row space."""
    q, _ = np.linalg.qr(dec_g.T)  # dec_g is (b, d) -> (d, b)
    return q


def subspace_r2(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean cos² of the principal angles between two orthonormal subspaces.

    ``basis_*`` are ``(d, k)`` with orthonormal columns. Principal-angle cosines
    are the singular values of ``Aᵀ B``; ``mean(cos²θ)`` ∈ [0,1] is an alignment
    R² (1 = identical subspaces). Averaged over ``min(k_a, k_b)`` angles.
    """
    s = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float((s ** 2).mean())


def match_blocks_to_truth(model: BSF, true_bases: list[np.ndarray]) -> dict:
    """Optimally match recovered blocks to planted subspaces by principal-angle R².

    Builds the ``(n_true, G)`` alignment-R² matrix and solves the assignment with
    the Hungarian algorithm (maximizing total matched R²). Returns per-truth
    matched R² and the mean.
    """
    from scipy.optimize import linear_sum_assignment

    dec = model.decoder.detach().cpu().numpy()  # (G, b, d)
    rec_bases = [orthonormal_block_basis(dec[g]) for g in range(model.cfg.n_blocks)]
    overlap = np.array(
        [[subspace_r2(tb, rb) for rb in rec_bases] for tb in true_bases]
    )  # (n_true, G)
    ti, gi = linear_sum_assignment(-overlap)
    matches = [{"truth": int(t), "block": int(g), "r2": float(overlap[t, g])}
               for t, g in zip(ti, gi)]
    return {"matches": matches, "mean_r2": float(np.mean([m["r2"] for m in matches]))}


def circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def cyclic_adjacency_accuracy(recovered_angle: np.ndarray, true_order: np.ndarray) -> float:
    """Fraction of true cyclic adjacencies preserved by the recovered angular
    order (rotation- and reflection-invariant). 1.0 = perfect circle ordering.
    Transcribed from experiments/curved_feature_probes.py."""
    n = len(recovered_angle)
    true_adj = {frozenset((int(true_order[i]), int(true_order[(i + 1) % n]))) for i in range(n)}
    seq = list(np.argsort(recovered_angle % (2 * np.pi)))
    rec_adj = {frozenset((int(true_order[seq[i]]), int(true_order[seq[(i + 1) % n]]))) for i in range(n)}
    return len(true_adj & rec_adj) / n


# ==========================================================================
# Phase 1 — synthetic planted-subspace recovery
# ==========================================================================
def make_planted(seed: int, d: int, n_true: int, b: int, k_true: int, n: int, noise: float):
    """N points, each a sparse sum of ``k_true`` random points on ``k_true`` of
    ``n_true`` planted ``b``-dim subspaces, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    bases = [np.linalg.qr(rng.standard_normal((d, b)))[0] for _ in range(n_true)]
    X = np.zeros((n, d))
    for i in range(n):
        for a in rng.choice(n_true, k_true, replace=False):
            X[i] += bases[a] @ rng.standard_normal(b)
    X += noise * rng.standard_normal((n, d))
    return X.astype(np.float64), bases


def phase_synthetic() -> dict:
    print("[synthetic] planted-subspace recovery", flush=True)
    d, n_true, b, k_true, n, noise = 48, 8, 4, 2, 4000, 0.05
    X, true_bases = make_planted(0, d, n_true, b, k_true, n, noise)
    ntr = int(0.85 * n)
    Xtr, Xval = torch.tensor(X[:ntr]), torch.tensor(X[ntr:])

    results = {"setup": {"d": d, "n_true_subspaces": n_true, "block_size": b,
                         "k_true": k_true, "n": n, "noise": noise}}
    for mode in ("vanilla", "grassmann"):
        cfg = BSFConfig(d_model=d, n_blocks=n_true, block_size=b, k_blocks=k_true,
                        mode=mode, aux_k_blocks=1, seed=0)
        model = BSF(cfg)
        train_bsf(model, Xtr, TrainConfig(steps=6000, batch_size=512, lr=4e-3), Xval)
        match = match_blocks_to_truth(model, true_bases)
        diag = block_diagnostics(model, Xval)
        results[mode] = {
            "val_ev": ev(model, Xval),
            "recovery_mean_r2": match["mean_r2"],
            "matches": match["matches"],
            "mean_stable_rank": diag["mean_stable_rank"],
            "mean_utilization": diag["mean_utilization"],
        }
        print(f"  {mode:10s} val_EV={results[mode]['val_ev']:.4f} "
              f"recovery_R2={match['mean_r2']:.4f} "
              f"stable_rank={diag['mean_stable_rank']:.2f}", flush=True)
    return results


# ==========================================================================
# Phase 2 — real-activation EV sweep at matched budget & sparsity
# ==========================================================================
def _load_real_activations(layer: int, reduce_dim: int, seed: int = 0):
    """Load an OLMo self/qualia residual cache, one layer, train-only PCA-reduced.

    Reuses the cached harvest under runs/ (no re-harvest). Returns
    (Xtr_reduced, Xval_reduced) as float64 torch tensors.
    """
    candidates = [
        REPO / "runs/OLMO3_32B_BASE_SELF_QUALIA_LAST/activations.npy",
        REPO / "runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST/activations.npy",
        REPO / "runs/OLMO3_7B_SELF_QUALIA_MAIN/activations.npy",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("no cached OLMo activation harvest found under runs/")
    A = np.load(path, mmap_mode="r")  # (n_prompts, n_layers, d)
    L = min(layer, A.shape[1] - 1)
    X = np.asarray(A[:, L, :], dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    ntr = int(0.85 * X.shape[0])
    tr, te, _, _ = pca_reduce(X[:ntr], X[ntr:], reduce_dim)
    # per-feature standardize on train (whiten scale so MSE isn't dominated by a
    # few high-variance PCs) — a common SAE preprocessing.
    sd = tr.std(0, keepdims=True).clip(min=1e-8)
    return torch.tensor(tr / sd), torch.tensor(te / sd), {"path": str(path), "layer": int(L),
                                                          "n": int(X.shape[0]), "d_reduced": tr.shape[1]}


def phase_real(F: int = 64, L0: int = 8, layer: int = 40, reduce_dim: int = 128) -> dict:
    print(f"[real] OLMo activation EV sweep (F={F}, L0={L0})", flush=True)
    Xtr, Xval, meta = _load_real_activations(layer, reduce_dim)
    print(f"  data: {meta['path']} L{meta['layer']} n={meta['n']} d->{meta['d_reduced']}", flush=True)
    d = Xtr.shape[1]

    rows = []
    for b in (1, 2, 4, 8):
        G = F // b
        k = max(1, L0 // b)
        modes = ("vanilla", "grassmann") if b > 1 else ("vanilla",)  # b=1 grassmann == vanilla
        for mode in modes:
            cfg = BSFConfig(d_model=d, n_blocks=G, block_size=b, k_blocks=k,
                            mode=mode, aux_k_blocks=max(1, G // 8), seed=0)
            model = BSF(cfg)
            train_bsf(model, Xtr, TrainConfig(steps=4000, batch_size=512, lr=3e-3), Xval)
            diag = block_diagnostics(model, Xval)
            label = "TopK-SAE" if b == 1 else f"BSF-{mode}"
            rows.append({
                "model": label, "mode": mode, "b": b, "G": G, "k_blocks": k,
                "L0_nonzeros": k * b, "n_latent": G * b, "dec_params": G * b * d,
                "val_ev": ev(model, Xval),
                "mean_stable_rank": diag["mean_stable_rank"],
                "mean_utilization": diag["mean_utilization"],
                "n_active_blocks": diag["n_active_blocks"],
            })
            print(f"  b={b} {label:14s} G={G:3d} k={k} L0={k*b} "
                  f"EV={rows[-1]['val_ev']:.4f} sr={diag['mean_stable_rank']:.2f} "
                  f"util={diag['mean_utilization']:.2f}", flush=True)
    return {"meta": meta, "budget": {"F": F, "L0": L0, "d_reduced": d}, "rows": rows}


# ==========================================================================
# Phase 3 — cyclic-feature block finding (weekday / month)
# ==========================================================================
def _demean_per_template(X: np.ndarray, tidx: np.ndarray) -> np.ndarray:
    Xd = X.copy()
    for t in np.unique(tidx):
        m = tidx == t
        Xd[m] = X[m] - X[m].mean(0, keepdims=True)
    return Xd


def _load_probe_set(name: str, layer: int, reduce_dim: int):
    z = np.load(REPO / f"experiments/probe_out/harvest_{name}.npz", allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    L = layer if layer in layers else layers[len(layers) // 2]
    X = z[f"L{L}"].astype(np.float64)
    tidx = z["template_idx"]
    rank = z["rank"].astype(int)
    n_labels = int(z["n_labels"])
    Xd = _demean_per_template(X, tidx)
    red, _, _, _ = pca_reduce(Xd, Xd, min(reduce_dim, Xd.shape[0] - len(np.unique(tidx)) - 1))
    return red, rank, n_labels, L


def phase_cyclic(reduce_dim: int = 16) -> dict:
    print("[cyclic] weekday / month single-block curve capture", flush=True)
    out = {}
    for name in ("weekday", "month"):
        Xred, rank, n_labels, L = _load_probe_set(name, layer=8, reduce_dim=reduce_dim)
        d = Xred.shape[1]
        X = torch.tensor(Xred)
        # G blocks, b=4 (a circle needs 2 dims; 4 gives slack), k=1 active block
        # per token: BSF must route each (token,template) to ONE block. Paper's
        # curve-detector claim = one block captures the whole cycle.
        G, b = 8, 4
        cfg = BSFConfig(d_model=d, n_blocks=G, block_size=b, k_blocks=1,
                        mode="grassmann", aux_k_blocks=2, seed=0)
        model = BSF(cfg)
        train_bsf(model, X, TrainConfig(steps=3000, batch_size=len(X), lr=6e-3), X)

        m_out = model(X, update_util=False)
        mask = m_out.mask.detach().cpu().numpy()  # (N, G)
        active_freq = mask.mean(0)
        win = int(np.argmax(active_freq))  # the block that captures the most tokens
        diag = block_diagnostics(model, X)
        win_sr = diag["per_block"][win]["stable_rank"]

        # in-block PCA of the winning block's contributions -> 2D angle -> ordering
        z_sparse = m_out.z_sparse.detach().cpu().numpy()
        dec = model.decoder.detach().cpu().numpy()
        active = mask[:, win] > 0
        contrib = z_sparse[active, win, :] @ dec[win]  # (n_active, d)
        cc = contrib - contrib.mean(0)
        _, _, vt = np.linalg.svd(cc, full_matrices=False)
        p2 = cc @ vt[:2].T  # (n_active, 2)
        angle_all = np.arctan2(p2[:, 1], p2[:, 0])
        rk = rank[active]
        uniq = sorted(set(rk.tolist()))
        tok_ang = np.array([circular_mean(angle_all[rk == u]) for u in uniq])
        adj = cyclic_adjacency_accuracy(tok_ang, np.array(uniq))

        out[name] = {
            "layer": int(L), "n_labels": n_labels, "d_reduced": int(d),
            "G": G, "block_size": b, "k_blocks": 1,
            "winning_block": win,
            "winning_block_active_freq": float(active_freq[win]),
            "winning_block_captures_frac": float(active.mean()),
            "winning_block_stable_rank": float(win_sr),
            "n_active_blocks": diag["n_active_blocks"],
            "cyclic_adjacency_accuracy": float(adj),
            "full_ev": ev(model, X),
            "block_active_freqs": active_freq.round(3).tolist(),
        }
        print(f"  {name}: block {win} captures {active.mean()*100:.0f}% of tokens, "
              f"stable_rank={win_sr:.2f}, adjacency_acc={adj:.2f}, "
              f"n_active_blocks={diag['n_active_blocks']}/{G}", flush=True)
    return out


# ==========================================================================
# Report
# ==========================================================================
def write_report(metrics: dict):
    L = ["# BSF baseline — Block-Sparse Featurizers (torch reimplementation)", ""]
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · CPU float64 · "
             f"our faithful reimplementation of Goodfire's BSF as the head-to-head baseline._")
    L.append("")
    L.append("Models: **Vanilla BSF** (free encoder + free block decoder, unit-norm rows), "
             "**Grassmannian BSF** (tied encoder `z=γ·xDᵀ`, one scalar γ, block decoders "
             "held column-orthonormal on the Stiefel manifold via periodic QR), and the "
             "**TopK-SAE** baseline (= vanilla BSF at block size b=1). Sparsity is per-block "
             "top-k on ‖z_g‖₂; codes are **signed** (no ReLU) so each block is a full subspace. "
             "AuxK resurrects dead blocks from the residual.")
    L.append("")

    if "synthetic" in metrics:
        s = metrics["synthetic"]
        st = s["setup"]
        L.append("## 1. Synthetic planted-subspace recovery")
        L.append("")
        L.append(f"{st['n_true_subspaces']} random {st['block_size']}-dim subspaces in "
                 f"d={st['d']}; each of {st['n']} points is a sparse sum of {st['k_true']} of "
                 f"them + noise σ={st['noise']}. Recovery = mean cos² of principal angles "
                 f"between each planted subspace and its matched recovered block "
                 f"(1.0 = perfect).")
        L.append("")
        L.append("| model | val EV | recovery R² (principal angles) | mean stable rank | mean utilization |")
        L.append("|---|---:|---:|---:|---:|")
        for mode in ("vanilla", "grassmann"):
            r = s[mode]
            L.append(f"| BSF-{mode} | {r['val_ev']:.4f} | **{r['recovery_mean_r2']:.4f}** | "
                     f"{r['mean_stable_rank']:.2f} | {r['mean_utilization']:.2f} |")
        L.append("")
        L.append(f"_(planted block size = {st['block_size']}; recovered stable rank ≈ "
                 f"{s['grassmann']['mean_stable_rank']:.1f} confirms each block spans its "
                 f"full {st['block_size']}-D subspace.)_")
        L.append("")

    if "real" in metrics:
        r = metrics["real"]
        bud = r["budget"]
        L.append("## 2. Real activations — EV at matched budget & sparsity")
        L.append("")
        L.append(f"Data: `{r['meta']['path'].split('/')[-2]}` layer {r['meta']['layer']}, "
                 f"n={r['meta']['n']} prompts, PCA-reduced to d={bud['d_reduced']}. "
                 f"**Matched decoder budget** (latent width F={bud['F']} → dec params "
                 f"F·d constant) and **matched sparsity** (L0 = k·b = {bud['L0']} nonzeros) "
                 f"across block sizes. b=1 is the TopK-SAE baseline.")
        L.append("")
        L.append("| block b | model | G | k | L0 | val EV | mean stable rank | mean util |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in r["rows"]:
            L.append(f"| {row['b']} | {row['model']} | {row['G']} | {row['k_blocks']} | "
                     f"{row['L0_nonzeros']} | {row['val_ev']:.4f} | "
                     f"{row['mean_stable_rank']:.2f} | {row['mean_utilization']:.2f} |")
        L.append("")
        topk = next(x for x in r["rows"] if x["b"] == 1)
        best = max(r["rows"], key=lambda x: x["val_ev"])
        L.append(f"_TopK-SAE (b=1) EV = {topk['val_ev']:.4f}; best BSF = "
                 f"{best['model']} b={best['b']} EV = {best['val_ev']:.4f} "
                 f"(Δ = {best['val_ev']-topk['val_ev']:+.4f})._")
        L.append("")

    if "cyclic" in metrics:
        c = metrics["cyclic"]
        L.append("## 3. Cyclic-feature block finding (weekday / month)")
        L.append("")
        L.append("Per-template-demeaned residuals for the weekday (7-circle) and month "
                 "(12-circle) token sets, fit with Grassmannian BSF (G=8 blocks, b=4, k=1: "
                 "each token routes to ONE block). Does a single block capture the whole "
                 "cycle (the paper's curve-detector result)? A circle is extrinsically 2-D, "
                 "so its block's stable rank should be ≈2; in-block 2-D PCA should order the "
                 "tokens correctly around the circle (adjacency accuracy → 1.0).")
        L.append("")
        L.append("| set | tokens | winning block | % tokens captured | block stable rank | in-block cyclic adjacency acc | active blocks |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, cc in c.items():
            L.append(f"| {name} | {cc['n_labels']} | #{cc['winning_block']} | "
                     f"{cc['winning_block_captures_frac']*100:.0f}% | "
                     f"{cc['winning_block_stable_rank']:.2f} | "
                     f"**{cc['cyclic_adjacency_accuracy']:.2f}** | "
                     f"{cc['n_active_blocks']}/{cc['G']} |")
        L.append("")

    L.append("## Files")
    L.append("")
    L.append("- `bsf.py` — models (vanilla / Grassmannian BSF, TopK-SAE baseline), block-TopK, "
             "AuxK, Stiefel retraction, metrics, gam shard-format loader.")
    L.append("- `train.py` — this driver (synthetic / real / cyclic phases + report).")
    L.append("- `metrics.json` — all numbers above, machine-readable.")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"[report] wrote {OUT/'REPORT.md'}", flush=True)


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "synthetic", "real", "cyclic"], default="all")
    args = ap.parse_args()

    mfile = OUT / "metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}

    t0 = time.time()
    if args.phase in ("all", "synthetic"):
        metrics["synthetic"] = phase_synthetic()
        mfile.write_text(json.dumps(metrics, indent=2))
    if args.phase in ("all", "real"):
        metrics["real"] = phase_real()
        mfile.write_text(json.dumps(metrics, indent=2))
    if args.phase in ("all", "cyclic"):
        metrics["cyclic"] = phase_cyclic()
        mfile.write_text(json.dumps(metrics, indent=2))

    write_report(metrics)
    print(f"[done] {time.time()-t0:.0f}s  ->  {mfile}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
