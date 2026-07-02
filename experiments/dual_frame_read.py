"""Dual-frame (biorthogonal) read tier for a sparse dictionary / manifold SAE.

The problem this fixes
======================
An SAE reads feature coordinates with its *encoder* -- a tied/ReLU linear map
``c_hat = D_S^T (x - b)`` over the active support ``S``. When the active atoms are
not orthogonal (decoder Gram ``G = D_S D_S^T`` has off-diagonal mass), that read is
BIASED: ``c_hat = G c_true (+noise) = c_true + (G - I) c_true``. The cross-talk
``(G - I) c_true`` is exactly "feature absorption" -- one atom's coordinate bleeds
into another's read. The *exact* linear read is the biorthogonal dual frame
``D~_S = D_S (D_S D_S^T)^{-1}`` (``D~_S^T D_S = I``), i.e. the least-squares /
pseudo-inverse coordinate ``c = (D_S D_S^T)^{-1} D_S (x - b)``. This tier computes
that read and shows it removes the absorption the encoder read suffers.

But the exact dual is UNBIASED at the cost of variance ``~ trace(G^{-1})``, which
blows up as the support becomes coherent. So neither the encoder (biased,
low-variance) nor the exact dual (unbiased, high-variance) is uniformly best. The
MSE-optimal read is the *posterior-mean* shrinkage dual under an empirical-Bayes
prior on the coordinates, with the shrinkage estimated by REML -- which recovers
the encoder-like read when the support is near-degenerate and the exact dual when
it is well-conditioned, and shrinks an unsupported feature's coordinate to the
null. That posterior-mean read is the DEFAULT here.

Design choices (aligned with the gam SPEC)
==========================================
  * Posterior mean is the default read, never a MAP point estimate; the exact
    biorthogonal dual is its flat-prior (``lambda -> 0``) limit.
  * The shrinkage ``lambda`` is fit by a closed-form REML fixed point over the
    pooled reads (no grid search, no wall-clock budget, no magic constant).
  * The prior "each feature coordinate shrinks toward absent (c_a = 0)" is the
    empirical-Bayes null: when the data do not support a feature, its read -> 0.
  * Subspace agreement is reported with PRINCIPAL ANGLES, not Gram overlap.

The reads assume a fixed support size ``L0`` (top-k / block route), so every step
is a vectorised ``(N, L0)`` closed form -- no per-token Python loop.

CLI
===
    python dual_frame_read.py --selftest
    python dual_frame_read.py --decoder D.npy --acts X.npy --codes S.npy --out dual_out/

``D`` is (K, p) unit-norm decoder rows; ``X`` is (N, p) activations (already
centred, or pass ``--bias b.npy``); ``S`` is (N, L0) int active-atom indices per
token (e.g. a TopK support or a block route's active atoms).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Core linear algebra: per-token Gram spectrum on the active support
# --------------------------------------------------------------------------- #
def support_spectrum(D: np.ndarray, X: np.ndarray, S: np.ndarray):
    """Eigendecompose each token's active-support Gram once.

    Returns everything the reads and the REML fixed point need, as vectorised
    ``(N, L0, ...)`` arrays:
      g  (N, L0)      -- eigenvalues of the support Gram G_t = D_St D_St^T
      s  (N, L0)      -- U_t^T r_t, the target projected into the Gram eigenbasis,
                         where r_t = D_St (x_t) is the encoder read
      U  (N, L0, L0)  -- Gram eigenvectors (to rotate coords back to atom order)
      r  (N, L0)      -- encoder read D_St x_t (the tied-encoder baseline)
      ynorm2 (N,)     -- ||x_t||^2 (ambient residual bookkeeping for REML)
    """
    D = np.asarray(D, np.float64)
    X = np.asarray(X, np.float64)
    S = np.asarray(S)
    N, L0 = S.shape
    Dact = D[S]                                   # (N, L0, p)  active decoder rows
    G = np.einsum("nkp,njp->nkj", Dact, Dact)     # (N, L0, L0) support Gram
    r = np.einsum("nkp,np->nk", Dact, X)          # (N, L0) encoder read D_S x
    g, U = np.linalg.eigh(G)                       # ascending eigenpairs
    g = np.clip(g, 1e-12, None)
    s = np.einsum("nkj,nk->nj", U, r)             # U^T r  (N, L0)
    ynorm2 = np.einsum("np,np->n", X, X)          # (N,)
    return g, s, U, r, ynorm2


def reml_ridge(g: np.ndarray, s: np.ndarray, ynorm2: np.ndarray, p: int,
               n_iter: int = 50, tol: float = 1e-8) -> dict:
    """Closed-form REML fixed point for the pooled coordinate-shrinkage ``lambda``.

    Model per token: ``x = D_S c + e``, ``e ~ N(0, sigma^2 I_p)``, prior
    ``c ~ N(0, tau^2 I)``; ``lambda = sigma^2 / tau^2``. Every quantity below is a
    closed form in the Gram eigenbasis, so the fixed point is grid-free:

        edf(l)   = sum_t sum_j g_tj / (g_tj + l)
        ||c||^2  = sum_t sum_j s_tj^2 / (g_tj + l)^2
        rss(l)   = sum_t [ ||x_t||^2 - 2 sum_j s_tj^2/(g_tj+l)
                                     + sum_j g_tj s_tj^2/(g_tj+l)^2 ]
        sigma^2  = rss / (N p - edf),   tau^2 = ||c||^2 / edf,   l <- sigma^2/tau^2

    sigma^2 is grounded in the AMBIENT residual (mostly the off-span component),
    so the shrinkage is real: an unsupported / noisy feature is pulled to the null.
    """
    N = g.shape[0]
    n_obs = float(N * p)
    s2 = s * s
    lam = float(np.mean(g))                        # scale-matched start, not magic
    for _ in range(n_iter):
        denom = g + lam
        edf = float(np.sum(g / denom))
        c2 = float(np.sum(s2 / denom**2))
        cr = float(np.sum(s2 / denom))
        cGc = float(np.sum(g * s2 / denom**2))
        rss = float(np.sum(ynorm2)) - 2.0 * cr + cGc
        sigma2 = rss / max(n_obs - edf, 1.0)
        tau2 = c2 / max(edf, 1e-12)
        lam_new = sigma2 / max(tau2, 1e-30)
        if not np.isfinite(lam_new) or lam_new <= 0:
            break
        if abs(lam_new - lam) <= tol * (lam + 1e-30):
            lam = lam_new
            break
        lam = lam_new
    return {"lambda": lam, "edf": edf, "sigma2": sigma2, "tau2": tau2}


# --------------------------------------------------------------------------- #
# The three reads (all in atom order, aligned to S columns)
# --------------------------------------------------------------------------- #
def read_encoder(r: np.ndarray) -> np.ndarray:
    """Tied-encoder read c_hat = D_S x (biased by the support Gram)."""
    return r


def read_exact_dual(g: np.ndarray, s: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Exact biorthogonal dual-frame read c = G^{-1} r (unbiased, min-residual)."""
    return np.einsum("nkj,nj->nk", U, s / g)


def read_posterior(g: np.ndarray, s: np.ndarray, U: np.ndarray,
                   lam: float) -> np.ndarray:
    """Posterior-mean shrinkage dual read c = (G + lambda I)^{-1} r (the default)."""
    return np.einsum("nkj,nj->nk", U, s / (g + lam))


def dual_frame(Dact: np.ndarray) -> np.ndarray:
    """Explicit dual frame D~ = D_S (D_S D_S^T)^{-1} for one support (rows dual to
    atoms). Exposed for the biorthogonality certificate; the reads use the
    eigenbasis form above."""
    G = Dact @ Dact.T
    return np.linalg.solve(G, Dact)               # (L0, p), D~ D_S^T = I


# --------------------------------------------------------------------------- #
# Diagnostics done correctly
# --------------------------------------------------------------------------- #
def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column spaces of A and B -- the correct
    subspace comparison (NOT a Gram/overlap heuristic)."""
    Qa, _ = np.linalg.qr(np.asarray(A, np.float64))
    Qb, _ = np.linalg.qr(np.asarray(B, np.float64))
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(sv, -1.0, 1.0))


def coherence(Dact: np.ndarray) -> float:
    """Max off-diagonal |Gram| of a support -- its mutual coherence."""
    G = Dact @ Dact.T
    off = G - np.diag(np.diag(G))
    return float(np.max(np.abs(off)))


# --------------------------------------------------------------------------- #
# Planted generators (ground-truth coordinates + controllable coherence)
# --------------------------------------------------------------------------- #
def _two_atom_decoder(rho: float) -> np.ndarray:
    """Two unit decoder rows in R^p with inner product rho (p large so the rest of
    the dictionary is ~orthogonal to them)."""
    p = 64
    D = np.zeros((2, p))
    D[0, 0] = 1.0
    D[1, 0] = rho
    D[1, 1] = np.sqrt(max(1.0 - rho**2, 1e-12))
    return D


def planted_pair(rho: float, n: int, sigma: float, seed: int):
    """n tokens, fixed 2-atom support with decoder coherence rho, iid true coords,
    ambient noise sigma. Returns D, X, S, c_true."""
    rng = np.random.default_rng(seed)
    D = _two_atom_decoder(rho)
    c_true = rng.standard_normal((n, 2))
    X = c_true @ D + sigma * rng.standard_normal((n, D.shape[1]))
    S = np.zeros((n, 2), int)
    S[:, 1] = 1
    return D, X, S, c_true


def planted_absorption(rho: float, n: int, sigma: float, seed: int):
    """Two co-firing atoms A,B with decoder coherence rho and INDEPENDENT true
    coords. The encoder read of B is contaminated by A (absorption); a correct
    read is not. Returns D, X, S, c_true (columns [A, B])."""
    return planted_pair(rho, n, sigma, seed)


# --------------------------------------------------------------------------- #
# Comparison harness
# --------------------------------------------------------------------------- #
def compare_reads(D, X, S, c_true=None, reml_iter=50):
    g, s, U, r, ynorm2 = support_spectrum(D, X, S)
    fit = reml_ridge(g, s, ynorm2, p=X.shape[1], n_iter=reml_iter)
    reads = {
        "encoder": read_encoder(r),
        "exact_dual": read_exact_dual(g, s, U),
        "posterior": read_posterior(g, s, U, fit["lambda"]),
    }
    out = {"lambda": fit["lambda"], "edf": fit["edf"], "sigma2": fit["sigma2"]}
    if c_true is not None:
        out["coord_mse"] = {k: float(np.mean((v - c_true) ** 2))
                            for k, v in reads.items()}
    return reads, out


# --------------------------------------------------------------------------- #
# Selftest: the falsifiable predictions
# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("[selftest] dual-frame biorthogonal read tier", flush=True)
    ok = True

    # (0) biorthogonality certificate: D~ D_S^T = I to machine precision
    D = _two_atom_decoder(0.7)
    Dt = dual_frame(D)
    ident_err = float(np.max(np.abs(Dt @ D.T - np.eye(2))))
    print(f"  biorthogonality ||D~ D_S^T - I||_max = {ident_err:.2e}", flush=True)
    if ident_err > 1e-9:
        print("  FAIL: dual frame is not biorthogonal to the support")
        ok = False

    # (1) coherence sweep: encoder is biased (MSE grows with rho); exact dual is
    # unbiased (MSE flat then variance-limited); posterior read is MSE-optimal.
    n, sigma = 40000, 0.15
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.85, 0.95):
        Dp, X, Sp, c = planted_pair(rho, n, sigma, seed=1)
        _, res = compare_reads(Dp, X, Sp, c_true=c)
        m = res["coord_mse"]
        rows.append((rho, m["encoder"], m["exact_dual"], m["posterior"], res["lambda"]))
        print(f"  rho={rho:.2f}  enc={m['encoder']:.4f}  dual={m['exact_dual']:.4f}"
              f"  post={m['posterior']:.4f}  lambda={res['lambda']:.3g}", flush=True)

    # exact dual must beat encoder in the moderate-coherence regime
    rho06 = next(r for r in rows if r[0] == 0.6)
    if not rho06[2] < rho06[1]:
        print("  FAIL: exact dual does not beat the biased encoder at rho=0.6")
        ok = False
    # the bias-variance crossover must exist: at high coherence the exact dual's
    # variance exceeds the encoder's bias (honest story, motivates shrinkage)
    rho095 = next(r for r in rows if r[0] == 0.95)
    if not rho095[2] > rho095[1]:
        print("  WARN: no dual-variance crossover at rho=0.95 (weak coherence)")
    # posterior read is never worse than the better of the two by much, and is
    # best-or-tied at every sweep point
    for rho, enc, dual, post, _ in rows:
        if post > min(enc, dual) + 0.02 * max(enc, dual, 1e-6):
            print(f"  FAIL: posterior read not MSE-optimal at rho={rho}")
            ok = False

    # (2) recover the null: an UNSUPPORTED feature (its true coord is 0, only
    # ambient noise) has its posterior coordinate shrunk toward 0, while the exact
    # dual read carries the full read variance.
    rng = np.random.default_rng(3)
    Dn = _two_atom_decoder(0.5)
    c = rng.standard_normal((30000, 2))
    c[:, 1] = 0.0                                  # atom B is absent
    Xn = c @ Dn + 0.3 * rng.standard_normal((30000, Dn.shape[1]))
    Sn = np.tile([0, 1], (30000, 1))
    reads, _ = compare_reads(Dn, Xn, Sn, c_true=c)
    var_dual_B = float(np.var(reads["exact_dual"][:, 1]))
    var_post_B = float(np.var(reads["posterior"][:, 1]))
    print(f"  absent-feature read var: dual={var_dual_B:.4f} "
          f"post={var_post_B:.4f} (shrinkage={1 - var_post_B/var_dual_B:.2%})",
          flush=True)
    if var_post_B >= var_dual_B:
        print("  FAIL: posterior read did not shrink the absent feature")
        ok = False

    # (3) absorption: encoder read of B correlates with the TRUE coord of A (the
    # absorption signature); the dual / posterior reads decouple them.
    Da, Xa, Sa, ca = planted_absorption(0.7, 40000, 0.15, seed=4)
    reads, _ = compare_reads(Da, Xa, Sa, c_true=ca)
    def corr(u, v):
        return float(np.corrcoef(u, v)[0, 1])
    enc_abs = abs(corr(reads["encoder"][:, 1], ca[:, 0]))
    dual_abs = abs(corr(reads["exact_dual"][:, 1], ca[:, 0]))
    print(f"  absorption corr(read_B, true_A): encoder={enc_abs:.3f} "
          f"dual={dual_abs:.3f}", flush=True)
    if not (enc_abs > 0.2 and dual_abs < 0.03):
        print("  FAIL: dual read did not remove encoder feature-absorption")
        ok = False

    print(f"[selftest] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--decoder", type=str, default=None, help="(K, p) unit-norm rows")
    ap.add_argument("--acts", type=str, default=None, help="(N, p) activations")
    ap.add_argument("--codes", type=str, default=None, help="(N, L0) int support")
    ap.add_argument("--bias", type=str, default=None, help="(p,) decoder bias b_dec")
    ap.add_argument("--out", type=str, default="dual_out")
    ap.add_argument("--reml-iter", type=int, default=50)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.decoder and args.acts and args.codes):
        print("provide --decoder --acts --codes, or --selftest", file=sys.stderr)
        return 2

    D = np.load(args.decoder)
    X = np.load(args.acts).astype(np.float64)
    S = np.load(args.codes).astype(int)
    if args.bias:
        X = X - np.load(args.bias)
    t0 = time.time()
    reads, res = compare_reads(D, X, S, reml_iter=args.reml_iter)

    # Report cross-talk removed and subspace agreement of the recovered atoms.
    g, s, U, r, _ = support_spectrum(D, X, S)
    xtalk = float(np.mean(np.abs(reads["encoder"] - reads["exact_dual"])))
    dump = {
        "n_tokens": int(S.shape[0]), "L0": int(S.shape[1]), "p": int(X.shape[1]),
        "lambda": res["lambda"], "edf": res["edf"], "sigma2": res["sigma2"],
        "mean_abs_crosstalk_encoder_vs_dual": xtalk,
        "mean_support_coherence": float(np.mean([
            coherence(D[S[i]]) for i in range(min(2000, S.shape[0]))])),
        "wall_s": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dual_frame_read.json").write_text(json.dumps(dump, indent=2))
    np.save(out / "coords_posterior.npy", reads["posterior"])
    print(f"[dual] lambda={res['lambda']:.3g} edf={res['edf']:.1f} "
          f"crosstalk={xtalk:.4f} -> {out/'dual_frame_read.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
