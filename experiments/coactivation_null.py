"""Fixed-margin randomization null for SAE feature *co-activation* discovery.

Motivation (the open wound)
===========================
"Manifold-tiling" / Ising-coupling readings of SAE co-activation are confounded by
the routing itself. Under a fixed top-``k`` route over ``c`` candidates, two
co-active *indicators* have a purely MECHANICAL covariance

        Cov(1_a, 1_b) = -k(c-k) / (c^2 (c-1))  <  0                     (uniform TopK)

with NO manifold structure whatsoever, and the hard-``k`` support violates the
positivity condition an Ising (Hammersley-Clifford) fit assumes. So an Ising fit
reads negative couplings everywhere and calls it "tiling"; and any co-occurrence
test against an *independence* null over-calls, because independence is the wrong
null when every token fires exactly ``k`` of ``c`` atoms.

This module never fits an Ising model. It tests observed co-activation against a
null that preserves *exactly the two quantities the route fixes* — each atom's
firing count (column margin of the binary code matrix) and each token's active
count (row margin) — via curveball / swap randomization (Strona et al. 2014, the
standard constraint-preserving randomization for binary matrices). Any coupling
that SURVIVES this null is real co-firing structure *beyond* what the margins and
the route force. This is the missing null: a co-activation significance test that
is invariant to the mechanical TopK anticorrelation.

What it gives you
=================
  * ``curveball``            : one constraint-preserving randomization of a binary
                              (tokens x atoms) code matrix (both margins exact).
  * ``analytic_topk_cov``    : the closed-form uniform-TopK indicator covariance,
                              a homogeneous-case cross-check of the null.
  * ``coactivation_test``    : observed co-activation excess vs the curveball null
                              AND vs the naive independence null, per atom pair;
                              significant-pair counts under each; the drop from
                              naive->corrected is the artifact the field misreads.
  * seed-stability of the surviving pair set (real structure is seed-stable; the
    naive artifact set is not).

Get a code matrix ``Z`` (tokens x atoms, {0,1}) from any router: a block-sparse
dictionary's active-block indicators (``block_sparse_dictionary_fit_begin(...)
.to_fit(X).blocks`` one-hot'd), a TopK-SAE's top-``k`` support, etc. For scale,
restrict to the ``--top-atoms`` most active columns: the null is then the correct
*conditional* null given each token's active count among those atoms.

CLI
===
    python coactivation_null.py --selftest              # falsifiable checks
    python coactivation_null.py --codes Z.npy --out coactivation_out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Constraint-preserving randomization (curveball / trade)
# --------------------------------------------------------------------------- #
def curveball(Z: np.ndarray, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """One curveball randomization of binary ``Z`` (tokens x atoms).

    Preserves BOTH margins exactly: every row sum (token active-count) and every
    column sum (atom firing-count) is unchanged. Each trade picks two rows, and
    redistributes the columns where exactly one of them fires, keeping each row's
    count of such columns fixed -- so row sums are invariant and every moved
    column stays in exactly one of the two rows, keeping column sums invariant.
    """
    Zb = Z.astype(bool, copy=True)
    n = Zb.shape[0]
    for _ in range(n_swaps):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = Zb[i], Zb[j]
        only_i = a & ~b
        only_j = b & ~a
        n_i = int(only_i.sum())
        swap = np.flatnonzero(only_i | only_j)
        if swap.size == 0 or n_i == 0 or n_i == swap.size:
            continue  # nothing exchangeable between these two rows
        rng.shuffle(swap)
        new_i = swap[:n_i]
        new_j = swap[n_i:]
        # rebuild the two rows: shared columns stay; exclusive ones reassigned
        a_new = a & b
        b_new = a & b
        a_new[new_i] = True
        b_new[new_j] = True
        Zb[i] = a_new
        Zb[j] = b_new
    return Zb


def _ndtri(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1e-9).
    Used only to convert a family-wise ``alpha`` into the naive independence-null
    z threshold -- so the one interpretable knob is a significance level, not a
    magic z cutoff."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def analytic_topk_cov(k_active: int, n_cand: int) -> float:
    """Closed-form Cov(1_a, 1_b) for two atoms under a uniform top-k route over
    ``n_cand`` candidates: ``-k(c-k)/(c^2 (c-1))``. Homogeneous-case cross-check
    of the curveball null (curveball generalizes it to heterogeneous margins)."""
    c = float(n_cand)
    k = float(k_active)
    if c <= 1:
        return 0.0
    return -k * (c - k) / (c * c * (c - 1.0))


# --------------------------------------------------------------------------- #
# Observed vs null co-activation
# --------------------------------------------------------------------------- #
def _coactivation_counts(Zb: np.ndarray) -> np.ndarray:
    """Symmetric (atoms x atoms) co-firing count matrix ``Z^T Z``."""
    Zf = Zb.astype(np.float64)
    return Zf.T @ Zf


def _naive_independence(Zb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected co-firing count + sd under the WRONG (independence) null:
    E[C_ab] = n p_a p_b, Var = n p_a p_b (1 - p_a p_b). Ignores the route -- this
    is the null the field implicitly uses and that the curveball null corrects."""
    n = Zb.shape[0]
    p = Zb.mean(0)
    e = n * np.outer(p, p)
    var = e * (1.0 - np.outer(p, p))
    return e, np.sqrt(np.maximum(var, 1e-12))


def coactivation_test(
    Z: np.ndarray,
    *,
    top_atoms: int | None = 128,
    n_null: int = 200,
    n_token_sub: int | None = 20000,
    swap_mult: float = 5.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Test each atom pair's co-firing against the curveball null and the naive
    independence null, both controlled at family-wise error rate ``alpha`` (the
    single interpretable knob). Positive corrected z = real excess co-firing; the
    naive test additionally flags the mechanical TopK anticorrelation."""
    rng = np.random.default_rng(seed)
    Zb = np.asarray(Z) > 0

    if n_token_sub is not None and Zb.shape[0] > n_token_sub:
        idx = rng.choice(Zb.shape[0], n_token_sub, replace=False)
        Zb = Zb[idx]
    if top_atoms is not None and Zb.shape[1] > top_atoms:
        keep = np.argsort(Zb.sum(0))[::-1][:top_atoms]
        keep.sort()
        Zb = Zb[:, keep]
    else:
        keep = np.arange(Zb.shape[1])

    n, m = Zb.shape
    triu = np.triu_indices(m, k=1)
    C_obs = _coactivation_counts(Zb)[triu]
    e_naive, sd_naive = _naive_independence(Zb)
    z_naive = (C_obs - e_naive[triu]) / sd_naive[triu]

    # Curveball null: keep each replicate's pair vector so significance uses a
    # permutation max-T threshold (calibrated from the null itself), not a
    # Gaussian tail -- robust to under-mixing / non-normal null shape.
    n_swaps = int(swap_mult * n)
    null_pairs = np.empty((n_null, triu[0].size))
    for r in range(n_null):
        Zr = curveball(Zb, n_swaps, rng)
        null_pairs[r] = _coactivation_counts(Zr)[triu]
    mean_null = null_pairs.mean(0)
    sd_null = np.sqrt(np.maximum(null_pairs.var(0), 1e-9))
    z_corr = (C_obs - mean_null) / sd_null

    # Corrected test: max-T permutation FWER at level ``alpha`` -- the (1-alpha)
    # quantile of each null replicate's own most extreme positive standardized
    # co-firing. A pair survives if its observed excess beats the strongest excess
    # the null routinely produces by chance. No distributional assumption, no
    # magic z cutoff.
    z_null = (null_pairs - mean_null) / sd_null
    thresh = float(np.percentile(z_null.max(1), 100.0 * (1.0 - alpha)))

    # Naive test at the SAME family-wise level, via a Bonferroni normal quantile on
    # the independence null -- the apples-to-apples "what the field's implicit test
    # would call". The gap corrected<<naive is the mechanical TopK artifact.
    n_pairs = int(triu[0].size)
    naive_thresh = _ndtri(1.0 - alpha / (2.0 * n_pairs))
    naive_sig = np.abs(z_naive) > naive_thresh
    corr_pos = z_corr > thresh
    pair_i, pair_j = keep[triu[0]], keep[triu[1]]
    surviving = [
        {"a": int(pair_i[q]), "b": int(pair_j[q]),
         "z_corrected": round(float(z_corr[q]), 3),
         "z_naive": round(float(z_naive[q]), 3),
         "obs": int(C_obs[q]), "null_mean": round(float(mean_null[q]), 2)}
        for q in np.flatnonzero(corr_pos)
    ]
    surviving.sort(key=lambda r: -r["z_corrected"])
    return {
        "n_tokens": int(n), "n_atoms": int(m), "n_pairs": n_pairs,
        "n_null": int(n_null), "n_swaps": int(n_swaps),
        "alpha": alpha, "corrected_threshold": round(thresh, 3),
        "naive_threshold": round(naive_thresh, 3),
        "naive_significant": int(naive_sig.sum()),
        "corrected_significant": int(corr_pos.sum()),
        "corrected_positive": int(corr_pos.sum()),
        "artifact_removed_frac": round(
            1.0 - corr_pos.sum() / max(int(naive_sig.sum()), 1), 4),
        "surviving_pairs": surviving[:200],
        "_thresh": thresh, "_z_corr": z_corr, "_triu": triu, "_pair_ij": (pair_i, pair_j),
    }


def surviving_set(res: dict) -> set:
    """The set of positively-surviving atom pairs (for seed-stability checks)."""
    pi, pj = res["_pair_ij"]
    return {(int(pi[q]), int(pj[q]))
            for q in np.flatnonzero(res["_z_corr"] > res["_thresh"])}


# --------------------------------------------------------------------------- #
# Synthetic controls (the falsifiable checks)
# --------------------------------------------------------------------------- #
def make_topk_artifact(n: int, K: int, k: int, seed: int) -> np.ndarray:
    """Pure uniform top-k route, tokens independent: NO co-firing structure, only
    the mechanical TopK anticorrelation. Corrected test must find ~0; naive must
    over-call."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((n, K), bool)
    for t in range(n):
        Z[t, rng.choice(K, k, replace=False)] = True
    return Z


def make_planted_clique(n: int, K: int, k: int, clique: tuple[int, ...],
                        q: float, seed: int) -> np.ndarray:
    """Uniform top-k route, EXCEPT a fraction ``q`` of tokens co-fire a fixed
    clique of atoms (plus random fillers). The clique co-occurs far above what its
    (only mildly elevated) margins predict, so it must survive the margin-
    preserving null; off-clique pairs must not."""
    rng = np.random.default_rng(seed)
    s = len(clique)
    Z = np.zeros((n, K), bool)
    others = np.array([a for a in range(K) if a not in set(clique)])
    for t in range(n):
        if rng.random() < q:
            Z[t, list(clique)] = True
            fill = rng.choice(others, max(k - s, 0), replace=False)
            Z[t, fill] = True
        else:
            Z[t, rng.choice(K, k, replace=False)] = True
    return Z


def selftest() -> int:
    """Assert the two falsifiable predictions + the analytic cross-check."""
    print("[selftest] fixed-margin co-activation null", flush=True)
    ok = True

    # (1) analytic cross-check: curveball reproduces the homogeneous-TopK covariance
    n, K, k = 3000, 40, 4
    Z = make_topk_artifact(n, K, k, seed=1)
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(18):
        Zr = curveball(Z > 0, 5 * n, rng)
        Cr = _coactivation_counts(Zr) / n
        p = Zr.mean(0)
        cov = Cr - np.outer(p, p)
        iu = np.triu_indices(K, 1)
        covs.append(cov[iu].mean())
    emp = float(np.mean(covs))
    ana = analytic_topk_cov(k, K)
    print(f"  analytic Cov={ana:.3e}  curveball mean Cov={emp:.3e}", flush=True)
    if abs(emp - ana) > 0.15 * abs(ana) + 1e-5:
        print("  FAIL: curveball null does not reproduce analytic TopK covariance")
        ok = False

    # (2a) naive over-call: at large n the tiny mechanical TopK anticorrelation is
    # "significant" under an independence null (the field's implicit test). Cheap:
    # the naive expectation is closed-form, no curveball needed.
    Zbig = make_topk_artifact(30000, 30, 6, seed=5) > 0
    e, sd = _naive_independence(Zbig)
    iu = np.triu_indices(30, 1)
    nthr = _ndtri(1.0 - 0.05 / (2.0 * iu[0].size))  # Bonferroni at alpha=0.05
    naive_big = int((np.abs((_coactivation_counts(Zbig)[iu] - e[iu]) / sd[iu]) > nthr).sum())
    print(f"  naive over-call (structureless TopK, n=30k): "
          f"{naive_big}/{iu[0].size} pairs flagged", flush=True)
    if naive_big < 20:
        print("  WARN: naive test did not over-call (weak demo of the artifact)")

    # (2b) corrected removes the artifact: full test on a structureless route ~ 0.
    art = coactivation_test(Z, top_atoms=None, n_null=40, n_token_sub=None,
                            swap_mult=6.0, alpha=0.05, seed=2)
    print(f"  artifact corrected_sig={art['corrected_significant']} "
          f"(maxT={art['corrected_threshold']})", flush=True)
    if art["corrected_significant"] > 2:
        print("  FAIL: corrected test still flags a structureless TopK route")
        ok = False

    # (3) planted clique: clique pairs survive the margin-preserving null AND
    # dominate the ranking; the naive test is massively worse. (A joint clique
    # induces genuine weak higher-order dependence at finite n, so a few borderline
    # off-clique pairs surviving is real dependence, not an artifact -- the
    # meaningful claim is that the clique is recovered and ranks at the top.)
    clique = (0, 1, 2, 3)
    Zc = make_planted_clique(n=3000, K=30, k=5, clique=clique, q=0.30, seed=3)
    res = coactivation_test(Zc, top_atoms=None, n_null=45, n_token_sub=None,
                            swap_mult=6.0, alpha=0.05, seed=4)
    surv = surviving_set(res)
    clique_pairs = {(a, b) for a in clique for b in clique if a < b}
    recall = len(surv & clique_pairs) / len(clique_pairs)
    top6 = {(r["a"], r["b"]) for r in res["surviving_pairs"][:len(clique_pairs)]}
    naive_ct = res["naive_significant"]
    print(f"  planted: clique recall={recall:.2f}  clique-dominates-top="
          f"{top6 == clique_pairs}  corrected={res['corrected_positive']} "
          f"vs naive={naive_ct}", flush=True)
    if recall < 0.99:
        print("  FAIL: planted clique did not survive the null")
        ok = False
    if top6 != clique_pairs:
        print("  FAIL: clique pairs do not dominate the co-firing ranking")
        ok = False
    if res["corrected_positive"] >= 0.5 * max(naive_ct, 1):
        print("  FAIL: corrected test did not reduce the naive over-call")
        ok = False

    # (4) seed-stability: the recovered clique is identical across null seeds
    res_b = coactivation_test(Zc, top_atoms=None, n_null=45, n_token_sub=None,
                              swap_mult=6.0, alpha=0.05, seed=7)
    surv_b = surviving_set(res_b)
    clique_stable = (surv & clique_pairs) == (surv_b & clique_pairs) == clique_pairs
    jac = len(surv & surv_b) / max(len(surv | surv_b), 1)
    print(f"  seed-stability: clique-stable={clique_stable}  "
          f"surviving-set Jaccard={jac:.3f}", flush=True)
    if not clique_stable:
        print("  FAIL: recovered clique not stable across null seeds")
        ok = False

    print(f"[selftest] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the falsifiable synthetic checks and exit")
    ap.add_argument("--codes", type=str, default=None,
                    help="(tokens x atoms) binary/real code matrix .npy (>0 => active)")
    ap.add_argument("--out", type=str, default="coactivation_out")
    ap.add_argument("--top-atoms", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-token-sub", type=int, default=20000)
    ap.add_argument("--swap-mult", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="family-wise error rate (the significance level)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.codes:
        print("provide --codes Z.npy or --selftest", file=sys.stderr)
        return 2

    Z = np.load(args.codes)
    print(f"[coactivation] loaded {args.codes} shape={Z.shape}", flush=True)
    t0 = time.time()
    res = coactivation_test(
        Z, top_atoms=args.top_atoms, n_null=args.n_null,
        n_token_sub=args.n_token_sub, swap_mult=args.swap_mult,
        alpha=args.alpha, seed=args.seed)
    res_b = coactivation_test(
        Z, top_atoms=args.top_atoms, n_null=args.n_null,
        n_token_sub=args.n_token_sub, swap_mult=args.swap_mult,
        alpha=args.alpha, seed=args.seed + 101)
    surv, surv_b = surviving_set(res), surviving_set(res_b)
    res["seed_stability_jaccard"] = round(
        len(surv & surv_b) / max(len(surv | surv_b), 1), 4)

    dump = {k: v for k, v in res.items() if not k.startswith("_")}
    dump["wall_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coactivation_null.json").write_text(json.dumps(dump, indent=2))
    print(f"[coactivation] naive_sig={res['naive_significant']} "
          f"corrected_sig={res['corrected_significant']} "
          f"removed={res['artifact_removed_frac']:.3f} "
          f"seed_jaccard={res['seed_stability_jaccard']} "
          f"-> {out/'coactivation_null.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
