"""Slow-feature atlas pilot on context means.

Everywhere in the suite the PerContextMean (template mean) is SUBTRACTED as a nuisance. The
hypothesis under test: it is a MODELED FEATURE — slow/contextual structure with its own
geometry. Pilot: pool the UNIQUE per-template means from the existing dose harvests (dozens
of templates across several features), fit a low-K manifold-SAE atlas on that template-mean
population, and ask whether contextual structure CHARTS (interpretable topology, explained
variance above a null, structure certificate, feature-of-origin recoverable above a
permutation null) or is unstructured.

Template means from different dims (8B=4096, 35B=2048) cannot be pooled, so we pool per model.
Data already exists in the npz caches (tmpl_mean arrays). Forward-free, CPU.

Nulls: a Gaussian-matched surrogate of the pooled template-mean population (same 2nd moments)
tells us how much "chart" is free — real structure must beat it on EV and on label recovery.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
OUT = os.environ.get("PREMISE_OUT", os.path.join(ROOT, "premise_out"))
os.makedirs(OUT, exist_ok=True)
FIT_ALARM = int(os.environ.get("ATLAS_FIT_ALARM", "240"))

# (feature_label, npz, model_group)
CACHES = [
    ("weekday", f"{ROOT}/dose_qwen8b_out/harvest_cache_weekday_L18_n70.npz", "8b_L18"),
    ("month", f"{ROOT}/dose_month_out/harvest_cache_month_L18_n120.npz", "8b_L18"),
    ("sycophancy", f"{OUT}/harvest_cache_sycophancy_L18.npz", "8b_L18"),
    ("hedging", f"{OUT}/harvest_cache_hedging_L18.npz", "8b_L18"),
    ("color", f"{ROOT}/dose_qwen36b_out/harvest_cache_color_L17_n48.npz", "35b_L17"),
    ("weekday", f"{ROOT}/dose_qwen36b_out/harvest_cache_month_L17_n72.npz", "35b_L17"),
    ("month", f"{ROOT}/dose_qwen36b_out/harvest_cache_weekday_L17_n42.npz", "35b_L17"),
]


class _TO(Exception):
    pass


def _h(s, f):
    raise _TO()


def unique_template_means(tm):
    key = np.round(tm, 5)
    uniq, idx = np.unique(key, axis=0, return_inverse=True)
    # return the true-precision mean per unique group (not the rounded key)
    means = np.zeros((len(uniq), tm.shape[1]))
    for g in range(len(uniq)):
        means[g] = tm[idx == g].mean(0)
    return means


def participation_ratio(S):
    """Intrinsic dimensionality proxy from singular values: (sum s^2)^2 / sum s^4."""
    ev = S ** 2
    return float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-30))


def atlas_fit(M, Ks=(1, 2, 3), seconds=FIT_ALARM):
    """Fit manifold-SAE atlas over template-mean population M (n, d). Reduced frame, small K
    search by evidence. Returns best fit summary + certificate."""
    import gamfit
    mu = M.mean(0)
    Mc = M - mu
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rdim = min(int(os.environ.get("ATLAS_RDIM", "10")), Vt.shape[0], len(M) - 1)
    Vt = np.ascontiguousarray(Vt[:rdim])
    Mr = np.ascontiguousarray(Mc @ Vt.T)
    old = signal.signal(signal.SIGALRM, _h)
    best = None
    tried = []
    try:
        for K in Ks:
            if K >= len(M):
                continue
            signal.alarm(int(seconds))
            t0 = time.time()
            try:
                sae = gamfit.sae_manifold_fit(Mr, K=K, d_atom=1, n_iter=50, random_state=0)
                signal.alarm(0)
                r2 = float(sae.reconstruction_r2)
                cert = None
                try:
                    cert = sae.structure_certificate_json()
                    if isinstance(cert, str):
                        cert = json.loads(cert)
                except Exception:  # noqa: BLE001
                    cert = None
                rec = dict(K=K, r2=r2, topologies=list(sae.atom_topologies),
                           chosen_k=int(getattr(sae, "chosen_k", K) or K),
                           seconds=time.time() - t0,
                           certificate=cert, coords=np.asarray(sae.coords).tolist()
                           if hasattr(sae, "coords") else None)
                tried.append({k: v for k, v in rec.items() if k not in ("coords", "certificate")})
                if best is None or r2 > best["r2"]:
                    best = rec
                    best["_coords"] = np.asarray(sae.coords) if hasattr(sae, "coords") else None
            except _TO:
                tried.append(dict(K=K, error=f"timeout>{seconds}s"))
            except Exception as exc:  # noqa: BLE001
                tried.append(dict(K=K, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return best, tried, S, rdim, Vt, mu


def label_recovery(coords, labels, B=5000, seed=0):
    """1-NN leave-one-out accuracy in coordinate space vs majority baseline + label
    permutation null. Categorical feature-of-origin label."""
    from numpy.linalg import norm
    C = coords - coords.mean(0)
    C = C / (norm(C, axis=1, keepdims=True) + 1e-9)
    D = 1.0 - C @ C.T
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    y = np.asarray(labels)
    acc = float(np.mean(y[nn] == y))
    vals, cnts = np.unique(y, return_counts=True)
    maj = float(cnts.max() / len(y))
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        yp = rng.permutation(y)
        null[b] = np.mean(yp[nn] == yp)
    p = (1 + int(np.sum(null >= acc))) / (B + 1)
    return dict(nn_loo_acc=acc, majority_baseline=maj, perm_p=float(p),
                null_mean=float(null.mean()), n_classes=int(len(vals)))


def gaussian_surrogate(M, seed=0):
    rng = np.random.default_rng(seed)
    mu = M.mean(0)
    Mc = M - mu
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    Z = rng.standard_normal((len(M), len(S)))
    return mu + (Z * (S / np.sqrt(max(len(M) - 1, 1)))) @ Vt


def analyze_group(group, entries):
    means = []
    feat_label = []
    tpl_label = []
    for feat, npz, _ in entries:
        if not os.path.exists(npz):
            print(f"[atlas:{group}] MISSING {npz}", flush=True)
            continue
        z = np.load(npz)
        m = unique_template_means(z["tmpl_mean"].astype(np.float64))
        means.append(m)
        feat_label += [feat] * len(m)
        tpl_label += list(range(len(m)))
        print(f"[atlas:{group}] {feat}: {len(m)} unique template means", flush=True)
    if not means:
        return dict(group=group, error="no caches present")
    M = np.vstack(means)
    feat_label = np.asarray(feat_label)
    n, d = M.shape
    print(f"[atlas:{group}] pooled {n} template means, d={d}, features={sorted(set(feat_label))}",
          flush=True)

    mu = M.mean(0); Mc = M - mu
    _, S, _ = np.linalg.svd(Mc, full_matrices=False)
    ev = S ** 2
    ev_frac = (ev / ev.sum()).tolist()
    pr = participation_ratio(S)

    best, tried, _, rdim, _, _ = atlas_fit(M)
    lab = None
    if best is not None and best.get("_coords") is not None and len(set(feat_label)) > 1:
        lab = label_recovery(best["_coords"], feat_label)

    # Gaussian-matched null: intrinsic dim + atlas EV
    Mg = gaussian_surrogate(M, seed=7)
    _, Sg, _ = np.linalg.svd(Mg - Mg.mean(0), full_matrices=False)
    pr_g = participation_ratio(Sg)
    bestg, _, _, _, _, _ = atlas_fit(Mg, Ks=(best["K"],) if best else (2,))

    rec = dict(
        group=group, n_template_means=int(n), d=int(d),
        features=sorted(set(feat_label.tolist())),
        pca_ev_fraction_top10=[float(x) for x in ev_frac[:10]],
        participation_ratio=pr, participation_ratio_gaussian_null=pr_g,
        atlas_best=({k: v for k, v in best.items() if not k.startswith("_")} if best else None),
        atlas_tried=tried,
        atlas_gaussian_null_r2=(bestg["r2"] if bestg else None),
        feature_of_origin_recovery=lab,
    )
    # verdict
    charts = False
    if best is not None and bestg is not None:
        charts = (best["r2"] - bestg["r2"] > 0.05) or (pr < 0.6 * pr_g)
    if lab is not None and lab["perm_p"] < 0.05 and lab["nn_loo_acc"] > lab["majority_baseline"] + 0.1:
        charts = True
    rec["verdict"] = "contextual_structure_charts" if charts else "unstructured_or_weak"
    with open(os.path.join(OUT, f"atlas_{group}.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    np.savez(os.path.join(OUT, f"atlas_means_{group}.npz"), M=M, feat=feat_label,
             coords=(best["_coords"] if best and best.get("_coords") is not None else np.zeros((n, 1))),
             ev_frac=np.asarray(ev_frac))
    print(f"[atlas:{group}] VERDICT={rec['verdict']} pr={pr:.2f} (null {pr_g:.2f}) "
          f"atlas_r2={best['r2'] if best else None} null_r2={bestg['r2'] if bestg else None} "
          f"feat_recovery={lab}", flush=True)
    return rec


def main():
    groups = {}
    for feat, npz, grp in CACHES:
        groups.setdefault(grp, []).append((feat, npz, grp))
    results = [analyze_group(g, e) for g, e in groups.items()]
    with open(os.path.join(OUT, "slow_feature_atlas.json"), "w") as fh:
        json.dump(dict(pilot="slow_feature_atlas_on_template_means", results=results), fh, indent=2)
    print("[done] slow_feature_atlas.json", flush=True)


if __name__ == "__main__":
    main()
