"""auto_exp_42: NULL CONTROL for auto_exp_41's Sphere-wins result.

auto_exp_41 found Sphere beats Euclidean by BIC margin ~5844 on the 2D
post-HSV-gauge-fix free block of cogito-L40. The lift-Jacobian
stereographic R^2 -> S^2 may inflate the absolute BIC of Sphere on any
data with non-Gaussian tails. Test this by running the SAME 4-topology
comparison on:
  (a) iid N(0, Sigma) with Sigma = empirical cov of real T2
  (b) iid N(0, I)
  (c) real T2 with each coordinate shuffled independently across colors
       (preserves marginals, destroys joint structure)

If Sphere still beats Euclidean by ~5844 on iid nulls, the result is a
lift artifact. If nulls give ~0 gap, auto_exp_41's finding is real.

Reuses the EXACT scorer functions from auto_exp_41 (import).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_exp_41 import (
    fit_score_euclidean, fit_score_circle, fit_score_sphere, fit_score_cylinder,
    bic, tk_score, SCORERS, N_FOLDS,
)

ROOT = Path("/Users/user/Manifold-SAE")
IN_NPZ = ROOT / "runs" / "auto_exp_41_results.npz"
OUT_NPZ = ROOT / "runs" / "auto_exp_42_results.npz"
MEMO = Path("/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/"
            "project_cogito_recovery_at_d_aux_3.md")

N_REPS = 20
RNG = np.random.default_rng(42)


def cv_scores(T2: np.ndarray, n_folds: int, rng) -> dict:
    n = T2.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    cv_ll = {name: [] for name in SCORERS}
    for te_idx in folds:
        tr_idx = np.setdiff1d(idx, te_idx)
        tr, te = T2[tr_idx], T2[te_idx]
        for name, fn in SCORERS.items():
            ll, _ = fn(tr, te)
            cv_ll[name].append(ll / len(te))
    return {name: float(np.mean(v)) for name, v in cv_ll.items()}


def in_sample(T2: np.ndarray) -> dict:
    n = T2.shape[0]
    out = {}
    for name, fn in SCORERS.items():
        ll, k = fn(T2, T2)
        out[name] = {"ll": ll, "k": k, "bic": bic(ll, k, n)}
    return out


def make_null(kind: str, T2_real: np.ndarray, rng) -> np.ndarray:
    n, d = T2_real.shape
    if kind == "iid_cov":
        Sigma = np.cov(T2_real.T)
        L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
        z = rng.standard_normal(size=(n, d))
        return z @ L.T
    elif kind == "iid_I":
        return rng.standard_normal(size=(n, d))
    elif kind == "shuffled":
        out = T2_real.copy()
        # shuffle each column independently — destroys joint dependence,
        # preserves marginals exactly
        for j in range(d):
            perm = rng.permutation(n)
            out[:, j] = T2_real[perm, j]
        return out
    else:
        raise ValueError(kind)


def main():
    t0 = time.time()
    print("[auto_exp_42] NULL CONTROL for auto_exp_41 Sphere-wins finding")
    d41 = np.load(IN_NPZ, allow_pickle=True)
    T2_real = d41["T2"].astype(np.float64)
    print(f"[real] T2 shape={T2_real.shape}; "
          f"BIC_real={dict(zip(d41['topology_names'], d41['bic']))}")
    real_bic = dict(zip(d41["topology_names"], d41["bic"]))
    real_cv = dict(zip(d41["topology_names"], d41["cv_ll_mean"]))
    real_gap_bic = real_bic["Euclidean"] - real_bic["Sphere"]
    real_gap_cv = real_cv["Sphere"] - real_cv["Euclidean"]
    print(f"[real] Sphere-vs-Eucl: BIC gap={real_gap_bic:.2f} "
          f"(Sphere lower by {real_gap_bic:.2f}); "
          f"CV gap={real_gap_cv:.4f} nats/pt (Sphere higher)")

    NULL_KINDS = ["iid_cov", "iid_I", "shuffled"]
    # storage: for each kind, list-of-reps of dicts
    results = {kind: {"bic": {nm: [] for nm in SCORERS},
                      "cv": {nm: [] for nm in SCORERS}}
               for kind in NULL_KINDS}

    for kind in NULL_KINDS:
        print(f"\n=== NULL: {kind} ({N_REPS} reps) ===")
        for r in range(N_REPS):
            sub_rng = np.random.default_rng(RNG.integers(0, 2**31 - 1))
            X = make_null(kind, T2_real, sub_rng)
            ins = in_sample(X)
            cvs = cv_scores(X, N_FOLDS, sub_rng)
            for nm in SCORERS:
                results[kind]["bic"][nm].append(ins[nm]["bic"])
                results[kind]["cv"][nm].append(cvs[nm])
        # summarize this null
        bic_gap = (np.array(results[kind]["bic"]["Euclidean"])
                   - np.array(results[kind]["bic"]["Sphere"]))
        cv_gap = (np.array(results[kind]["cv"]["Sphere"])
                  - np.array(results[kind]["cv"]["Euclidean"]))
        frac_match = float(np.mean(bic_gap >= real_gap_bic))
        print(f"  BIC gap (Eucl - Sphere): "
              f"mean={bic_gap.mean():.2f} sd={bic_gap.std():.2f} "
              f"(real={real_gap_bic:.2f})")
        print(f"  CV gap (Sphere - Eucl) nats/pt: "
              f"mean={cv_gap.mean():.4f} sd={cv_gap.std():.4f} "
              f"(real={real_gap_cv:.4f})")
        print(f"  fraction(null >= real BIC gap): {frac_match:.3f}")
        # also print bic per topology
        print("  per-topology mean BIC:")
        for nm in SCORERS:
            arr = np.array(results[kind]["bic"][nm])
            print(f"    {nm:>10}: {arr.mean():>10.2f} +/- {arr.std():.2f}")

    # save
    out = {"real_T2": T2_real,
           "real_bic_eucl": real_bic["Euclidean"],
           "real_bic_sphere": real_bic["Sphere"],
           "real_gap_bic": real_gap_bic,
           "real_gap_cv": real_gap_cv,
           "null_kinds": np.array(NULL_KINDS),
           "topology_names": np.array(list(SCORERS.keys())),
           "n_reps": N_REPS}
    for kind in NULL_KINDS:
        for nm in SCORERS:
            out[f"bic_{kind}_{nm}"] = np.array(results[kind]["bic"][nm])
            out[f"cv_{kind}_{nm}"] = np.array(results[kind]["cv"][nm])
    np.savez(OUT_NPZ, **out)
    print(f"\n[npz] {OUT_NPZ}")

    # memo
    verdict_lines = []
    for kind in NULL_KINDS:
        bic_gap = (np.array(results[kind]["bic"]["Euclidean"])
                   - np.array(results[kind]["bic"]["Sphere"]))
        cv_gap = (np.array(results[kind]["cv"]["Sphere"])
                  - np.array(results[kind]["cv"]["Eucl"
                                                 "idean"]))
        frac = float(np.mean(bic_gap >= real_gap_bic))
        ratio = bic_gap.mean() / real_gap_bic if real_gap_bic != 0 else float("nan")
        verdict_lines.append(
            f"| {kind} | {bic_gap.mean():.1f} +/- {bic_gap.std():.1f} | "
            f"{cv_gap.mean():.3f} +/- {cv_gap.std():.3f} | "
            f"{ratio:.3f} | {frac:.2f} |")

    # decide verdict
    iid_cov_gap = (np.array(results["iid_cov"]["bic"]["Euclidean"])
                   - np.array(results["iid_cov"]["bic"]["Sphere"]))
    iid_I_gap = (np.array(results["iid_I"]["bic"]["Euclidean"])
                 - np.array(results["iid_I"]["bic"]["Sphere"]))
    shuf_gap = (np.array(results["shuffled"]["bic"]["Euclidean"])
                - np.array(results["shuffled"]["bic"]["Sphere"]))
    avg_iid = 0.5 * (iid_cov_gap.mean() + iid_I_gap.mean())
    if avg_iid > 0.5 * real_gap_bic:
        verdict = ("**LIFT-ARTIFACT VERDICT**: iid Gaussian nulls reproduce "
                   "a Sphere>Euclidean BIC gap comparable to real data. "
                   "auto_exp_41's Sphere-wins finding is dominated by the "
                   "stereographic-lift Jacobian / vMF tails, not by real "
                   "structure on the cogito free block.")
    elif avg_iid < 0.1 * real_gap_bic and shuf_gap.mean() < 0.5 * real_gap_bic:
        verdict = ("**SIGNAL-CONFIRMED VERDICT**: iid nulls give small "
                   "Sphere>Euclidean gap; real data far exceeds nulls. "
                   "auto_exp_41's Sphere-wins finding survives null control.")
    else:
        verdict = ("**MIXED VERDICT**: null gaps non-trivial but not at "
                   "real-data magnitude. Some lift bias + some real signal.")

    snippet = (
        "\n## auto_exp_42: Sphere null control (2026-05-23)\n"
        f"Tested auto_exp_41's Sphere-wins-by-5844-BIC finding against three nulls "
        f"({N_REPS} reps each), using IDENTICAL scorer functions (imported from "
        f"auto_exp_41).\n\n"
        f"| null type | BIC gap (Eucl-Sphere) | CV gap (Sphere-Eucl) nats/pt | "
        f"ratio vs real | frac >= real |\n"
        f"|---|---|---|---|---|\n"
        + "\n".join(verdict_lines)
        + f"\n\nReal data: BIC gap = {real_gap_bic:.2f}, "
        f"CV gap = {real_gap_cv:.4f} nats/pt.\n\n"
        f"{verdict}\n\n"
        f"Archive: `runs/auto_exp_42_results.npz`.\n"
    )
    with open(MEMO, "a") as f:
        f.write(snippet)
    print(f"[memo] appended to {MEMO}")
    print(f"\n{verdict}")
    print(f"[runtime] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
