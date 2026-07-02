"""Targeted curved-feature probes: weekday / month / year token geometry.

The canonical circular/cyclic features in an LM residual stream are:
  * day-of-week  — a 7-point CIRCLE (Monday..Sunday, wraps)
  * month        — a 12-point CIRCLE (January..December, wraps)
  * year         — a 1-D ORDERED CURVE (1950..2020, does NOT wrap)

This probe is the cleanest demonstration that a *curved* dictionary atom captures
what a *linear* SAE / PCA shreds. A circle is intrinsically 1-D but extrinsically
2-D: no single linear direction can order all 7 weekdays cyclically (a line folds
the circle — two days project to the same coordinate). One curved (periodic) atom
represents the whole set with a single angular coordinate and orders the tokens
correctly around the circle.

Method (cheap dedicated harvest — not a corpus job)
===================================================
1. Harvest residual-stream activations for each token set in a handful of natural
   template sentences (reusing the model-loading / hook idioms from llm_probe.py).
   Readout = residual at the target token's last sub-token position.
2. Choose the analysis layer by the LINEAR PCA diagnostic (conservative — we pick
   the layer where the *linear* baseline looks best, then ask whether curved still
   wins).
3. For each set, in a shared train-only-PCA-reduced space, compare held-out
   reconstruction EV at MATCHED intrinsic budget:
     * linear PCA with L components  (optimal linear L-dim reconstruction)
     * curved gamfit manifold SAE, K=1, intrinsic_rank=1
       (periodic 'circle' topology for weekday/month, non-periodic 'product' for years)
   via leave-one-template-out cross-validation.
4. Ordering: from the curved atom's recovered chart coordinate (angle for circles),
   quantify whether tokens order correctly around the circle:
     * circles  — circular correlation + cyclic adjacency accuracy (rotation- and
       reflection-invariant), plus Kendall tau of the unwrapped angle.
     * years    — |Spearman| / |Kendall tau| of recovered position vs year.
   Contrast with the best single linear direction (a linear atom), which folds a
   circle and cannot order it.
5. (optional, CURVED_PROBE_REML=1) corroborate the curved EV with the REML
   gamfit.sae_manifold_fit on the reduced space, with the standard retry/higher-iter
   guard (it is slower and can fail to converge on thin data).

The gamfit manifold SAE used for the headline is gamfit.torch.ManifoldSAE — the same
curved dictionary as gamfit.sae_manifold_fit, fit by backprop, which exposes the
per-sample chart coordinate we need for the ordering test.

Runs on CPU. Self-test on synthetic planted circles/curve:  python curved_feature_probes.py --synthetic
Real harvest:                                               python curved_feature_probes.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Keep BLAS/rayon polite on a shared box.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("CURVED_PROBE_OUT", HERE / "probe_out"))


# ---------------------------------------------------------------------------
# Token sets + templates
# ---------------------------------------------------------------------------


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
YEARS = [str(y) for y in range(1950, 2021, 5)]  # 1950,1955,...,2020  (15 tokens, ordered)


def _token_sets() -> dict[str, dict]:
    """Each set: labels, integer ground-truth order, whether it is cyclic, templates."""
    return {
        "weekday": {
            "labels": WEEKDAYS,
            "order": list(range(len(WEEKDAYS))),
            "cyclic": True,
            "templates": [
                "I will see you on {x}.",
                "The meeting is scheduled for {x}.",
                "She was born on a {x}.",
                "We always rest on {x}.",
                "By {x}, everything was ready.",
            ],
        },
        "month": {
            "labels": MONTHS,
            "order": list(range(len(MONTHS))),
            "cyclic": True,
            "templates": [
                "It happened in {x}.",
                "We got married in {x}.",
                "The festival is held every {x}.",
                "By {x} the snow had melted.",
                "Her birthday is in {x}.",
            ],
        },
        "year": {
            "labels": YEARS,
            "order": [int(y) for y in YEARS],
            "cyclic": False,
            "templates": [
                "It happened in {x}.",
                "The book was published in {x}.",
                "She was born in {x}.",
                "By the year {x} everything had changed.",
                "The war ended in {x}.",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Harvest (reuses llm_probe.py idioms: forward hooks / output_hidden_states)
# ---------------------------------------------------------------------------


@dataclass
class HarvestConfig:
    model_name: str = os.environ.get("CURVED_PROBE_MODEL", "Qwen/Qwen2.5-0.5B")
    layers: tuple[int, ...] = tuple(
        int(x) for x in os.environ.get("CURVED_PROBE_LAYERS", "5,8,11,14").split(",")
    )


def _target_span_last_pos(tok, template: str, label: str) -> tuple[str, int]:
    """Build the prompt and return (prompt, index of the LAST sub-token of {x}).

    Uses offset mapping so we robustly locate the target token span regardless of
    how the label tokenizes (weekday/month are one token; a year is several digits).
    """
    prefix, suffix = template.split("{x}")
    prompt = prefix + label + suffix
    char_lo = len(prefix)
    char_hi = len(prefix) + len(label)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc["offset_mapping"]
    last = None
    for i, (a, b) in enumerate(offs):
        if a == b:  # special token / empty span
            continue
        # token overlaps the target character range
        if a >= char_lo and b <= char_hi:
            last = i
    if last is None:  # fall back to last non-empty token
        for i, (a, b) in enumerate(offs):
            if a != b:
                last = i
    return prompt, last


def harvest(cfg: HarvestConfig, sets: dict[str, dict], out_dir: Path) -> list[str]:
    """Harvest activations and save each set's npz to out_dir, resumably.

    Saves each set immediately after building it and SKIPS sets already cached, so a
    re-run (after an external OOM kill during harvest) resumes instead of recomputing.
    Returns the list of set names that are now cached.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    todo = [s for s in sets if not (out_dir / f"harvest_{s}.npz").exists()]
    if not todo:
        return list(sets)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bfloat16 + low_cpu_mem_usage halves the resident model (~2GB fp32 -> ~1GB) to
    # shrink the OOM-kill window during load on this memory-pressured shared box.
    # CPU bf16 forward is fine here; we upcast the harvested residuals to float32.
    model = AutoModel.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    model.to(torch.device("cpu"))

    torch.set_grad_enabled(False)
    for set_name in todo:
        spec = sets[set_name]
        labels, order, templates = spec["labels"], spec["order"], spec["templates"]
        per_layer: dict[int, list] = {L: [] for L in cfg.layers}
        ranks, used_labels, template_idx = [], [], []
        for ti, template in enumerate(templates):
            for lbl, rnk in zip(labels, order):
                prompt, pos = _target_span_last_pos(tok, template, lbl)
                inputs = tok(prompt, return_tensors="pt")
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states  # len n_layers+1, [0] = embeddings
                for L in cfg.layers:
                    per_layer[L].append(hs[L + 1][0, pos, :].cpu().float().numpy())
                ranks.append(float(rnk))
                used_labels.append(lbl)
                template_idx.append(ti)
        entry = {
            "cyclic": bool(spec["cyclic"]), "rank": np.array(ranks),
            "label": used_labels, "template_idx": np.array(template_idx),
            "n_labels": len(labels),
        }
        for L in cfg.layers:
            entry[L] = np.stack(per_layer[L], 0).astype(np.float64)
        save_harvest({set_name: entry}, cfg.layers, out_dir)  # checkpoint this set
        print(f"[harvest] {set_name}: {len(ranks)} samples "
              f"({len(labels)} tokens x {len(templates)} templates), D={entry[cfg.layers[0]].shape[1]} "
              f"[saved]", flush=True)
    torch.set_grad_enabled(True)
    del model
    gc.collect()
    return list(sets)


# ---------------------------------------------------------------------------
# Spearman / Kendall / circular stats (no scipy)
# ---------------------------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _ranks(x) - (len(x) - 1) / 2, _ranks(y) - (len(y) - 1) / 2
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    tot = c + d
    return float((c - d) / tot) if tot > 0 else 0.0


def circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka-SenGupta circular correlation (rotation-invariant)."""
    a0, b0 = a - circular_mean(a), b - circular_mean(b)
    num = float((np.sin(a0) * np.sin(b0)).sum())
    den = float(np.sqrt((np.sin(a0) ** 2).sum() * (np.sin(b0) ** 2).sum()))
    return num / den if den > 0 else 0.0


def cyclic_adjacency_accuracy(recovered_angle: np.ndarray, true_order: np.ndarray) -> float:
    """Fraction of true cyclic adjacencies preserved by the recovered angular order.

    Rotation- and reflection-invariant: sort tokens by recovered angle to get a
    cyclic sequence, form its unordered adjacent pairs, and compare to the true
    cyclic adjacency set. 1.0 = perfect circular ordering (up to rotation/flip).
    """
    n = len(recovered_angle)
    true_adj = {frozenset((int(true_order[i]), int(true_order[(i + 1) % n]))) for i in range(n)}
    seq = list(np.argsort(recovered_angle % (2 * np.pi)))
    rec_adj = {frozenset((int(true_order[seq[i]]), int(true_order[seq[(i + 1) % n]]))) for i in range(n)}
    return len(true_adj & rec_adj) / n


# ---------------------------------------------------------------------------
# Curved fit (gamfit manifold SAE, torch backend) + linear PCA baseline
# ---------------------------------------------------------------------------


def _pca_reduce(train: np.ndarray, test: np.ndarray, r: int):
    """Train-only centering + PCA to r components. Returns (train_red, test_red, mu, Vt, evr)."""
    mu = train.mean(0)
    tc = train - mu
    U, S, Vt = np.linalg.svd(tc, full_matrices=False)
    r = min(r, Vt.shape[0])
    Vt = Vt[:r]
    tr = tc @ Vt.T
    te = (test - mu) @ Vt.T
    var_total = float((tc ** 2).sum())
    evr = float((tr ** 2).sum() / var_total) if var_total > 0 else 1.0
    return tr, te, mu, Vt, evr


def _ev(x: np.ndarray, xhat: np.ndarray) -> float:
    sst = float(((x - x.mean(0)) ** 2).sum())
    return float(1 - ((x - xhat) ** 2).sum() / sst) if sst > 0 else float("nan")


def linear_pca_ev(train_red: np.ndarray, test_red: np.ndarray, L: int) -> float:
    """Held-out EV of the optimal linear L-dim reconstruction (PCA fit on train_red)."""
    mu = train_red.mean(0)
    tc = train_red - mu
    _, _, Vt = np.linalg.svd(tc, full_matrices=False)
    Vt = Vt[:L]
    te_c = test_red - mu
    recon = (te_c @ Vt.T) @ Vt + mu
    return _ev(test_red, recon)


# Tuned recipe (see scratch sweeps): a rank-1 circle atom must be a *single-winding*
# curve for its recovered angle to order the tokens. Keep n_basis LOW (a near-pure
# circle) — high basis lets the fourier curve double back (high EV but scrambled
# angle). A wide encoder + moderate init_scale + moderate lr spreads points around
# the full circle; too-high lr collapses it to antipodal points. No sparsity penalty
# (pointless with a single atom).
_CURVE_N_BASIS = int(os.environ.get("CURVED_PROBE_NBASIS", "4"))
_CURVE_LR = float(os.environ.get("CURVED_PROBE_LR", "8e-3"))
_CURVE_ENC_HIDDEN = 64
_CURVE_INIT_SCALE = 0.2
_CURVE_N_SEEDS = int(os.environ.get("CURVED_PROBE_SEEDS", "2"))


def _fit_one(train_red, cyclic, steps, n_basis, seed):
    import torch
    from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

    torch.manual_seed(seed)
    D = train_red.shape[1]
    # rank-1: the torch backend only offers 'circle' at intrinsic_rank==1. For years
    # (non-periodic) the circle atom parameterizes a monotone ARC (no wrap) — the
    # recovered coordinate still orders the years; we read it un-wrapped.
    cfg = ManifoldSAEConfig(
        input_dim=D, n_atoms=1, intrinsic_rank=1,
        atom_manifold="circle", atom_basis="fourier", n_basis_per_atom=int(n_basis),
        sparsity={"kind": "softmax_topk", "target_k": 1,
                  "tau_start": 4.0, "tau_min": 1.0, "tau_steps": steps},
        encoder_hidden=_CURVE_ENC_HIDDEN, init_scale=_CURVE_INIT_SCALE,
        dtype=torch.float64,
    )
    sae = ManifoldSAE(cfg)
    x = torch.tensor(train_red, dtype=torch.float64)
    opt = torch.optim.Adam(sae.parameters(), lr=_CURVE_LR)
    sae.train()
    for _ in range(steps):
        out = sae(x)  # tiny N — full batch each step
        loss = ((out.x_hat - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.sparsity.advance_temperature()
    sae.eval()
    with torch.no_grad():
        ev = _ev(train_red, sae(x).x_hat.numpy())
    return sae, ev


def curved_fit(train_red: np.ndarray, cyclic: bool, steps: int = 800,
               n_basis: int | None = None, seed: int = 0):
    """Fit gamfit's manifold SAE (torch backend): K=1, intrinsic_rank=1, circle atom.

    Best-of-N-seeds selected by *train* reconstruction EV (unsupervised — never uses
    the ordering labels), so the reported ordering metric stays honest.
    """
    nb = _CURVE_N_BASIS if n_basis is None else n_basis
    best_sae, best_ev = None, -np.inf
    for s in range(seed, seed + _CURVE_N_SEEDS):
        sae, ev = _fit_one(train_red, cyclic, steps, nb, s)
        if ev > best_ev:
            best_sae, best_ev = sae, ev
        else:
            del sae
        gc.collect()
    return best_sae


def curved_ev_and_positions(sae, X_red: np.ndarray):
    """Forward X_red through fitted sae; return (EV, angle_per_sample)."""
    import torch
    with torch.no_grad():
        out = sae(torch.tensor(X_red, dtype=torch.float64))
    xhat = out.x_hat.numpy()
    pos = out.positions[:, 0, 0].numpy()  # single atom, single intrinsic coord
    return _ev(X_red, xhat), pos


# ---------------------------------------------------------------------------
# Per-set analysis
# ---------------------------------------------------------------------------


def choose_layer(entry: dict, layers) -> tuple[int, dict]:
    """Pick the layer with the strongest LINEAR structure (best |Spearman| of a top-PC
    with the ground-truth order), so the layer choice is conservative for linear."""
    rank = entry["rank"]
    diag = {}
    best_L, best_score = layers[0], -1.0
    for L in layers:
        X = entry[L]
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:8].T
        rhos = [abs(spearman(proj[:, k], rank)) for k in range(proj.shape[1])]
        score = max(rhos)
        diag[L] = {"best_abs_spearman_top8pc": score}
        if score > best_score:
            best_L, best_score = L, score
    return best_L, diag


def _demean_per_template(X: np.ndarray, tidx: np.ndarray) -> np.ndarray:
    """Subtract each template's own mean activation (over its tokens). Real residual
    activations are dominated by the sentence/context; the token-of-interest (weekday /
    month / year) is a small component. Removing the per-template mean isolates the
    token-varying signal — the same 'frame-demean before geometry' recipe the color
    harvest uses in DATA_README. Without this the geometry is swamped by context and
    held-out EV goes negative."""
    Xd = X.copy()
    for t in np.unique(tidx):
        m = tidx == t
        Xd[m] = X[m] - X[m].mean(0, keepdims=True)
    return Xd


def analyze_set(name: str, entry: dict, layers, reduce_dim: int, steps: int) -> dict:
    rank = entry["rank"]
    cyclic = entry["cyclic"]
    n_labels = entry["n_labels"]
    templates = np.unique(entry["template_idx"])
    labels = entry["label"]

    # per-template demean every layer, then pick the layer with the strongest LINEAR
    # structure on the demeaned signal (conservative for the linear baseline).
    entry = {**entry, **{L: _demean_per_template(entry[L], entry["template_idx"]) for L in layers}}
    L, layer_diag = choose_layer(entry, layers)
    X = entry[L]
    N = X.shape[0]
    r = min(reduce_dim, N - len(templates) - 1)  # keep < n_train to avoid degeneracy

    # ---- leave-one-template-out CV: matched-budget EV (linear L=1,2 vs curved rank-1)
    cv = {"linear_L1": [], "linear_L2": [], "curved": [], "reduce_evr": []}
    for h in templates:
        te_mask = entry["template_idx"] == h
        tr_mask = ~te_mask
        tr_red, te_red, _, _, evr = _pca_reduce(X[tr_mask], X[te_mask], r)
        cv["reduce_evr"].append(evr)
        cv["linear_L1"].append(linear_pca_ev(tr_red, te_red, 1))
        cv["linear_L2"].append(linear_pca_ev(tr_red, te_red, 2))
        sae = curved_fit(tr_red, cyclic, steps=steps, seed=0)
        ev_test, _ = curved_ev_and_positions(sae, te_red)
        cv["curved"].append(ev_test)
    cv_mean = {k: float(np.mean(v)) for k, v in cv.items()}
    cv_std = {k: float(np.std(v)) for k, v in cv.items()}

    # ---- full-data fit for ordering: recover chart coordinate per sample
    full_red, _, _, _, evr_full = _pca_reduce(X, X, r)
    sae_full = curved_fit(full_red, cyclic, steps=steps, seed=0)
    ev_full, angle = curved_ev_and_positions(sae_full, full_red)
    insample = {
        "curved": float(ev_full),
        "linear_L1": float(linear_pca_ev(full_red, full_red, 1)),
        "linear_L2": float(linear_pca_ev(full_red, full_red, 2)),
    }

    # per-token mean of recovered coordinate (circular mean for cyclic sets)
    rank_int = rank.astype(int)
    uniq = sorted(set(rank_int.tolist()))
    tok_true = np.array(uniq, dtype=float)
    if cyclic:
        # circle position is a fraction of a turn in [0,1); map to radians directly
        # (NO min-max rescale — that would distort the circle's metric).
        ang = angle.astype(float) * 2 * np.pi
        tok_ang = np.array([circular_mean(ang[rank_int == u]) for u in uniq])
        true_ang = np.array([2 * np.pi * (u / n_labels) for u in uniq])
        ccorr = abs(circular_corr(tok_ang, true_ang))
        adj_acc = cyclic_adjacency_accuracy(tok_ang, np.array(uniq))
        # unwrapped kendall along recovered angular order vs true order (both directions)
        seq = np.argsort(tok_ang % (2 * np.pi))
        ktau = max(abs(kendall_tau(np.arange(len(seq)), np.argsort(np.array(uniq)[seq]))), 0.0)
        ordering = {
            "metric": "circular",
            "circular_corr_abs": float(ccorr),
            "cyclic_adjacency_accuracy": float(adj_acc),
            "n_tokens": int(n_labels),
        }
    else:
        # circle position parameterizes a monotone ARC for the (non-periodic) year
        # curve; if the arc straddles the 0/1 seam the raw coord folds, so also try a
        # half-turn rotation (unwrapping the seam) and keep the better Spearman.
        raw = np.array([float(angle[rank_int == u].mean()) for u in uniq])
        rot = (raw + 0.5) % 1.0
        sp_raw, sp_rot = abs(spearman(raw, tok_true)), abs(spearman(rot, tok_true))
        tok_coord = raw if sp_raw >= sp_rot else rot
        sp = max(sp_raw, sp_rot)
        kt = abs(kendall_tau(tok_coord, tok_true))
        ordering = {
            "metric": "monotone",
            "spearman_abs": float(sp),
            "kendall_tau_abs": float(kt),
            "n_tokens": int(n_labels),
        }

    # ---- linear "atom folds the circle" contrast: best single PC ordering
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:8].T
    # per-token mean of each PC, ordering vs truth
    if cyclic:
        # a single linear direction cannot give cyclic order; report its best monotone
        # Spearman with rank as the (folded) linear ceiling
        best_lin = max(abs(spearman(np.array([proj[rank_int == u, k].mean() for u in uniq]),
                                    tok_true)) for k in range(proj.shape[1]))
        # 2D PCA angle ordering (fair linear upper bound: needs 2 dims for a circle)
        p2 = np.array([[proj[rank_int == u, 0].mean(), proj[rank_int == u, 1].mean()] for u in uniq])
        lin_angle = np.arctan2(p2[:, 1] - p2[:, 1].mean(), p2[:, 0] - p2[:, 0].mean())
        lin_adj = cyclic_adjacency_accuracy(lin_angle, np.array(uniq))
        linear_ordering = {
            "best_single_PC_spearman_abs": float(best_lin),
            "pca2d_angle_cyclic_adjacency_accuracy": float(lin_adj),
        }
    else:
        best_lin = max(abs(spearman(np.array([proj[rank_int == u, k].mean() for u in uniq]),
                                    tok_true)) for k in range(proj.shape[1]))
        linear_ordering = {"best_single_PC_spearman_abs": float(best_lin)}

    return {
        "layer": int(L),
        "n_samples": int(N),
        "n_tokens": int(n_labels),
        "cyclic": cyclic,
        "reduce_dim": int(r),
        "reduce_evr_full": float(evr_full),
        "layer_diag": {str(k): v for k, v in layer_diag.items()},
        "cv_ev_mean": cv_mean,
        "cv_ev_std": cv_std,
        "insample_ev": insample,
        "curved_ev_full_insample": float(ev_full),
        "ordering_curved": ordering,
        "ordering_linear": linear_ordering,
        "_tok_true": tok_true.tolist(),
        "_tok_coord": (tok_ang.tolist() if cyclic else tok_coord.tolist()),
    }


# ---------------------------------------------------------------------------
# Optional REML corroboration (gamfit.sae_manifold_fit)
# ---------------------------------------------------------------------------


def reml_corroborate(X: np.ndarray, cyclic: bool, reduce_dim: int) -> dict:
    import gamfit
    red, _, _, _, _ = _pca_reduce(X, X, min(reduce_dim, X.shape[0] - 2))
    topo = "circle" if cyclic else "line"
    err = None
    for n_iter in (60, 120):
        try:
            kw = dict(K=1, d_atom=1, n_iter=n_iter)
            # only pass atom_topology where it is meaningful/periodic
            if cyclic:
                kw["atom_topology"] = "circle"
            fit = gamfit.sae_manifold_fit(red, **kw)
            return {"reml_ev_insample": float(fit.reconstruction_r2), "n_iter": n_iter,
                    "topology": topo}
        except Exception as exc:  # noqa: BLE001 — REML can fail to converge on thin data
            err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    return {"reml_ev_insample": None, "error": err}


# ---------------------------------------------------------------------------
# Synthetic self-test (validates the fit+ordering pipeline without a model)
# ---------------------------------------------------------------------------


def synthetic_sets(seed: int = 0) -> dict:
    """Plant a 7-circle, a 12-circle, and a 15-point line in noisy high-D space,
    each with 5 'templates' (random per-template offsets) so the harness is exercised
    exactly like the real harvest."""
    rng = np.random.default_rng(seed)
    D = 48
    out = {}
    specs = [("weekday", 7, True), ("month", 12, True), ("year", 15, False)]
    for name, n, cyclic in specs:
        b = rng.standard_normal((3, D))
        ranks, labels, tmpl, rows = [], [], [], []
        for ti in range(5):
            off = 0.15 * rng.standard_normal(D)
            for u in range(n):
                if cyclic:
                    th = 2 * np.pi * u / n
                    v = np.cos(th) * b[0] + np.sin(th) * b[1]
                else:
                    s = u / (n - 1)
                    v = s * b[0] + (s ** 2) * b[1]
                v = v + off + 0.05 * rng.standard_normal(D)
                rows.append(v)
                ranks.append(float(u if cyclic else 1950 + 5 * u))
                labels.append(f"{name}{u}")
                tmpl.append(ti)
        entry = {"cyclic": cyclic, "rank": np.array(ranks), "label": labels,
                 "template_idx": np.array(tmpl), "n_labels": n}
        X = np.stack(rows, 0)
        for L in (0,):
            entry[L] = X
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def save_harvest(harvested: dict, layers, out_dir: Path):
    """Persist each set's activations so per-set analysis can run in a fresh process
    (the shared box has an external OOM killer; one process per set is the safe unit)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in harvested.items():
        arrs = {f"L{L}": entry[L] for L in layers if L in entry}
        np.savez(
            out_dir / f"harvest_{name}.npz",
            rank=entry["rank"], template_idx=entry["template_idx"],
            labels=np.array(entry["label"]), cyclic=np.array(entry["cyclic"]),
            n_labels=np.array(entry["n_labels"]), layers=np.array(list(layers)),
            **arrs,
        )


def load_harvest_set(name: str, out_dir: Path):
    z = np.load(out_dir / f"harvest_{name}.npz", allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    entry = {
        "rank": z["rank"], "template_idx": z["template_idx"],
        "label": [str(s) for s in z["labels"]], "cyclic": bool(z["cyclic"]),
        "n_labels": int(z["n_labels"]),
    }
    for L in layers:
        entry[L] = z[f"L{L}"]
    return entry, layers


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"


def write_reports(results: dict, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curved_feature_probes.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, default=float))

    lines = ["# Curved-feature probes: weekday / month / year", ""]
    lines.append(f"- source: **{meta.get('source')}**  |  model: `{meta.get('model')}`")
    lines.append(f"- curved atom: gamfit manifold SAE (torch backend), K=1, intrinsic_rank=1, "
                 f"periodic circle+fourier atom (years use the same circle atom on a monotone arc; "
                 f"the torch backend has no open-interval manifold at rank 1)")
    lines.append("")
    lines.append("## Matched-budget held-out EV (leave-one-template-out CV)")
    lines.append("")
    lines.append("Curved uses **1** intrinsic coordinate; linear-L1 uses 1 PC, linear-L2 uses 2 PCs. "
                 "A circle is intrinsically 1-D but needs 2 linear dims — so curved(1) should match "
                 "linear(2) and beat linear(1).")
    lines.append("")
    lines.append("| set | layer | tokens | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        cv = r["cv_ev_mean"]
        lines.append(f"| {name} | {r['layer']} | {r['n_tokens']} | "
                     f"{_fmt(cv['curved'])} | {_fmt(cv['linear_L1'])} | {_fmt(cv['linear_L2'])} |")
    lines.append("")
    lines.append("In-sample EV (full fit, no held-out — the cleaner 'can a 1-coord curved atom "
                 "*represent* this set' view; CV above is noisy at these tiny sample counts):")
    lines.append("")
    lines.append("| set | curved EV (1 coord) | linear EV (1 PC) | linear EV (2 PC) |")
    lines.append("|---|---:|---:|---:|")
    for name, r in results.items():
        ins = r.get("insample_ev", {})
        lines.append(f"| {name} | {_fmt(ins.get('curved'))} | "
                     f"{_fmt(ins.get('linear_L1'))} | {_fmt(ins.get('linear_L2'))} |")
    lines.append("")
    lines.append("## Ordering accuracy (does the recovered chart order the tokens?)")
    lines.append("")
    lines.append("| set | curved metric | curved value | linear best single-PC (Spearman) | linear 2D-PCA-angle |")
    lines.append("|---|---|---:|---:|---:|")
    for name, r in results.items():
        oc = r["ordering_curved"]
        olin = r["ordering_linear"]
        if oc["metric"] == "circular":
            cval = f"adj={_fmt(oc['cyclic_adjacency_accuracy'])} / circ_r={_fmt(oc['circular_corr_abs'])}"
            lin2 = _fmt(olin.get("pca2d_angle_cyclic_adjacency_accuracy"))
        else:
            cval = f"spearman={_fmt(oc['spearman_abs'])} / tau={_fmt(oc['kendall_tau_abs'])}"
            lin2 = "n/a"
        lines.append(f"| {name} | {oc['metric']} | {cval} | "
                     f"{_fmt(olin['best_single_PC_spearman_abs'])} | {lin2} |")
    lines.append("")
    if any("reml" in r for r in results.values()):
        lines.append("## REML corroboration (gamfit.sae_manifold_fit)")
        lines.append("")
        lines.append("| set | REML EV (in-sample) | note |")
        lines.append("|---|---:|---|")
        for name, r in results.items():
            rm = r.get("reml", {})
            ev = rm.get("reml_ev_insample")
            note = rm.get("error", f"n_iter={rm.get('n_iter')}") if ev is None else f"n_iter={rm.get('n_iter')}"
            lines.append(f"| {name} | {_fmt(ev)} | {note} |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def plot_orderings(results: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        true = np.array(r["_tok_true"])
        coord = np.array(r["_tok_coord"])
        if r["ordering_curved"]["metric"] == "circular":
            ax.scatter(np.cos(coord), np.sin(coord), c=true, cmap="hsv", s=90)
            for k in range(len(true)):
                ax.annotate(str(int(true[k])), (np.cos(coord[k]), np.sin(coord[k])),
                            fontsize=8, ha="center", va="center")
            ax.set_aspect("equal")
            ax.set_title(f"{name}: recovered circle\nadj={r['ordering_curved']['cyclic_adjacency_accuracy']:.2f}")
        else:
            ax.scatter(true, coord, c=true, cmap="viridis", s=60)
            ax.set_xlabel("true year")
            ax.set_ylabel("recovered coord")
            ax.set_title(f"{name}: recovered curve\nspearman={r['ordering_curved']['spearman_abs']:.2f}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovered_orderings.png", dpi=120)
    plt.close(fig)


SET_ORDER = ("year", "weekday", "month")  # smaller / independent first


def _cap_threads():
    import torch
    torch.set_num_threads(int(os.environ.get("CURVED_PROBE_THREADS", "4")))
    from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
    bypass_gamfit_cuda_check()


def _meta(args, source, model, do_reml):
    return {"source": source, "model": model, "steps": args.steps,
            "reduce_dim": args.reduce_dim, "reml": do_reml}


def phase_harvest(args) -> tuple[list, str, str, list]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        sets = synthetic_sets()
        layers = [0]
        source, model = "SYNTHETIC planted circles/curve", "(none)"
        # synthetic data is in-memory; save all sets now (cheap, no model)
        save_harvest(sets, layers, OUT_DIR)
        cached = [k for k in SET_ORDER if k in sets]
    else:
        cfg = HarvestConfig()
        layers = list(cfg.layers)
        source, model = "REAL residual-stream harvest", cfg.model_name

        def _write_meta():
            cached = [k for k in SET_ORDER if (OUT_DIR / f"harvest_{k}.npz").exists()]
            (OUT_DIR / "harvest_meta.json").write_text(json.dumps(
                {"layers": layers, "source": source, "model": model, "sets": cached}))
            return cached
        _write_meta()  # exists before the (kill-prone) model harvest
        harvest(cfg, _token_sets(), OUT_DIR)  # resumable, saves per-set
        cached = _write_meta()
        print(f"[harvest] cached sets: {cached}", flush=True)
        return layers, source, model, cached
    # write meta up front so it survives even a mid-harvest kill on a later re-run
    (OUT_DIR / "harvest_meta.json").write_text(json.dumps(
        {"layers": layers, "source": source, "model": model, "sets": cached}))
    print(f"[harvest] cached sets: {cached}", flush=True)
    return layers, source, model, cached


def phase_analyze_one(name: str, args) -> int:
    """Analyze a single set from the cached harvest and MERGE into results.json."""
    _cap_threads()
    hmeta = json.loads((OUT_DIR / "harvest_meta.json").read_text())
    entry, layers = load_harvest_set(name, OUT_DIR)
    do_reml = os.environ.get("CURVED_PROBE_REML", "0") == "1"
    print(f"[analyze] {name} ...", flush=True)
    res = analyze_set(name, entry, layers, args.reduce_dim, args.steps)
    if do_reml:
        # REML must see the SAME preprocessing as the torch fit (per-template demean).
        Xdm = _demean_per_template(entry[res["layer"]], entry["template_idx"])
        res["reml"] = reml_corroborate(Xdm, entry["cyclic"], args.reduce_dim)
        print(f"[reml] {name}: {res['reml']}", flush=True)
    oc, cv = res["ordering_curved"], res["cv_ev_mean"]
    print(f"  -> layer {res['layer']} | curved EV {cv['curved']:.3f} vs "
          f"lin1 {cv['linear_L1']:.3f} / lin2 {cv['linear_L2']:.3f} | ordering {oc}", flush=True)
    # merge-write (read existing, update this set, rewrite json + summary)
    jf = OUT_DIR / "curved_feature_probes.json"
    prev = json.loads(jf.read_text())["results"] if jf.exists() else {}
    prev[name] = res
    ordered = {k: prev[k] for k in SET_ORDER if k in prev}
    meta = {"source": hmeta["source"], "model": hmeta["model"], "steps": args.steps,
            "reduce_dim": args.reduce_dim, "reml": do_reml}
    write_reports(ordered, OUT_DIR, meta)
    plot_orderings(ordered, OUT_DIR)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run on planted synthetic circles/curve (pipeline self-test, no model)")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("CURVED_PROBE_STEPS", "600")))
    ap.add_argument("--reduce-dim", type=int, default=int(os.environ.get("CURVED_PROBE_RDIM", "16")))
    ap.add_argument("--harvest", action="store_true", help="(phase) harvest + cache only")
    ap.add_argument("--analyze", type=str, default=None, help="(phase) analyze one cached set")
    args = ap.parse_args()

    if args.analyze:
        return phase_analyze_one(args.analyze, args)

    if args.harvest:
        _cap_threads()
        phase_harvest(args)
        return 0

    # Default: orchestrate everything as fresh, retried subprocesses. The shared box
    # has an external OOM reaper that SIGKILLs large processes at random, so every heavy
    # phase (model harvest, per-set fit) runs isolated and is retried; partial progress
    # (cached npz / merged results.json) always survives a kill.
    import subprocess
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).resolve()),
            "--steps", str(args.steps), "--reduce-dim", str(args.reduce_dim)]
    if args.synthetic:
        base.append("--synthetic")
    max_tries = int(os.environ.get("CURVED_PROBE_RETRIES", "5"))
    target = list(SET_ORDER)

    # --- Phase A: harvest (resumable; retry until every set is cached) ---
    def cached_sets():
        return [s for s in target if (OUT_DIR / f"harvest_{s}.npz").exists()]
    for attempt in range(1, max_tries + 1):
        if len(cached_sets()) == len(target):
            break
        print(f"\n[driver] === harvest (attempt {attempt}/{max_tries}) ===", flush=True)
        rc = subprocess.run(base + ["--harvest"], env=os.environ).returncode
        print(f"[driver] harvest rc={rc}, cached={cached_sets()}", flush=True)
    ready = cached_sets()
    if not ready:
        print("[driver] FATAL: harvest produced no cached sets after retries", flush=True)
        return 1

    # --- Phase B: analyze each cached set in a fresh, retried process ---
    for name in ready:
        for attempt in range(1, max_tries + 1):
            print(f"\n[driver] === analyzing {name} (attempt {attempt}/{max_tries}) ===", flush=True)
            rc = subprocess.run(base + ["--analyze", name], env=os.environ).returncode
            if rc == 0:
                break
            print(f"[driver] set {name} exited rc={rc}; retrying" if attempt < max_tries
                  else f"[driver] set {name} failed after {max_tries} attempts (rc={rc})", flush=True)
    print(f"\n[done] {OUT_DIR}/curved_feature_probes.json + summary.md + recovered_orderings.png",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
