"""Matched-null battery for the W7 curved-circle-probe claims.

Goodfire's BSF paper validates its curve-manifold Fourier claims with a *matched
null*: a synthetic control that preserves some structure of the data while
destroying the specific structure the claim rests on. If the observed statistic
survives the null (empirical p small), the structure is real; if the null
reproduces it, the "finding" is an artifact of the reading, the basis, or the
raw spectrum. Our W7 circle-probe claims (`curved_feature_probes.py`) currently
ship with NO null baseline. This module retrofits one, as a reusable battery.

The W7 claims under test (see `probe_out/curved_feature_probes.json`)
================================================================================
  C1  EV parity:      a *curved* atom reconstructs weekday/month residuals from
                      ONE intrinsic coordinate about as well as a *linear* 2-PC
                      reconstruction, and beats a linear 1-PC reconstruction
                      (curved ~= linear-L2 >> linear-L1). "One curved coordinate
                      does the work of two linear PCs."
  C2  Cyclic order:   the single recovered angular coordinate orders the tokens
                      correctly around the circle (weekday cyclic-adjacency 0.71,
                      month 0.83; month 2D-PCA angle 1.0).

The null battery (each null preserves some structure, destroys the claimed one)
================================================================================
  N1 rotation           replace the PCA reading basis with a random orthonormal
                        basis of the SAME r-dim subspace, re-fit, re-read. Tests
                        whether the coordinate/ordering structure is basis-real
                        or an artifact of reading PC1/PC2 as the circle plane.
                        (PCA reconstruction EV is rotation-invariant by
                        construction, so this is a stability/robustness null.)
  N2 label-permutation  permute the cyclic labels; recompute cyclic-adjacency /
                        circular-correlation of the *fixed* recovered angle.
                        Empirical p that the recovered ordering matches the TRUE
                        cyclic order better than a random labelling. -> C2.
  N3 matched-spectrum   synthesize Gaussian data with the SAME per-PC eigen-
     Gaussian           spectrum but NO cyclic structure; re-fit curved + linear.
                        Tests whether "curved(1) ~= linear(2) >> linear(1)" is a
                        circle signature or something any 1-D curve gets on a
                        matched spectrum. -> C1.  (headline: can FAIL)
  N4 phase-scramble     preserve the power spectrum of the cyclically-ordered
                        token means but randomise the per-column Fourier phases
                        (destroys cross-dimension mode-locking, keeps each dim's
                        autocorrelation). Recompute 2D-PCA-angle cyclic adjacency.
                        Tests whether the circle is genuine fundamental-mode
                        phase-locking or merely band-limited smoothness. -> C2.

Reusable entry point
====================
    from matched_null import null_battery
    res = null_battery(X, labels, n_labels=7, cyclic=True,
                       claims=("rotation","label_perm","matched_spectrum",
                               "phase_scramble"))
`X` is (N, D) analysis-ready activations (already per-frame demeaned if that is
your recipe); `labels` is the (N,) integer cyclic rank in [0, n_labels). The
G-bsf and N-nursery lanes can call this directly on their cyclic-block findings.
See README.md in this directory.

Every curved fit reuses the *torch-backend* ManifoldSAE recipe from
`curved_feature_probes.curved_fit` (K=1, intrinsic_rank=1, circle+fourier atom) —
NOT the REML `sae_manifold_fit`, which is OOM/segfault-prone in this environment.
The CLI still isolates each set in a retried subprocess with incremental saves,
per house rules.

CLI
===
    python matched_null.py --set weekday      # one cached set -> null_out/
    python matched_null.py --set month
    python matched_null.py                    # orchestrate both, retried subprocs
    python matched_null.py --synthetic        # planted-circle sanity check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Keep BLAS/rayon polite on a shared box (same as curved_feature_probes).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import curved_feature_probes` works from anywhere
import curved_feature_probes as C  # noqa: E402  (reuse the exact W7 pipeline)

OUT_DIR = Path(os.environ.get("MATCHED_NULL_OUT", HERE / "null_out"))

# Fit budget for every curved fit in the battery. We MATCH the W7 headline recipe
# exactly (steps=600, best-of-2 seeds selected by train EV) so the observed-in-
# battery statistic reproduces the CLAIMED value — otherwise a single unlucky seed
# reads out a worse ordering than the claim reports and we would test a straw man.
# The nulls use the same recipe, so every p-value is internally apples-to-apples.
# (The recovered-angle ordering is somewhat seed-fragile at these tiny N; matching
# the claim's best-of-2 selection is what keeps the null test honest.)
_STEPS = int(os.environ.get("MATCHED_NULL_STEPS", "600"))
_NSEEDS = int(os.environ.get("MATCHED_NULL_SEEDS", "2"))


# ---------------------------------------------------------------------------
# Core statistics on a reduced representation
# ---------------------------------------------------------------------------


def _token_means(red: np.ndarray, labels: np.ndarray, uniq) -> np.ndarray:
    """Per-token mean row in reduced space, rows in the given (true cyclic) order."""
    return np.stack([red[labels == u].mean(0) for u in uniq], 0)


def _pca2d_angle_adjacency(M: np.ndarray, n: int) -> float:
    """Cyclic-adjacency accuracy of the 2D-PCA angle of the token-mean rows M
    (M already in true cyclic order 0..n-1). This is the W7 'linear 2D-PCA angle'
    circle readout — the fair linear upper bound (a circle needs two dims)."""
    Mc = M - M.mean(0)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    p2 = Mc @ Vt[:2].T
    ang = np.arctan2(p2[:, 1] - p2[:, 1].mean(), p2[:, 0] - p2[:, 0].mean())
    return C.cyclic_adjacency_accuracy(ang, np.arange(n))


def _curved_read(red: np.ndarray, labels: np.ndarray, uniq, n_labels: int,
                 cyclic: bool, seed: int):
    """Fit the curved atom on `red`; return (curved_ev, tok_angle_per_token)."""
    C._CURVE_N_SEEDS = _NSEEDS
    sae = C.curved_fit(red, cyclic, steps=_STEPS, seed=seed)
    ev, angle = C.curved_ev_and_positions(sae, red)
    if cyclic:
        ang = angle.astype(float) * 2 * np.pi
        tok = np.array([C.circular_mean(ang[labels == u]) for u in uniq])
    else:
        tok = np.array([float(angle[labels == u].mean()) for u in uniq])
    del sae
    return float(ev), tok


def _adjacency(tok_angle: np.ndarray, order: np.ndarray) -> float:
    return C.cyclic_adjacency_accuracy(tok_angle, order)


def _circ_corr_to_truth(tok_angle: np.ndarray, uniq, n_labels: int) -> float:
    true_ang = np.array([2 * np.pi * (u / n_labels) for u in uniq])
    return abs(C.circular_corr(tok_angle, true_ang))


def _hist(arr: np.ndarray, bins: int = 40) -> dict:
    """Compact histogram of a null distribution for the JSON + figure."""
    arr = np.asarray(arr, float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _empirical_p(null: np.ndarray, observed: float, tail: str = "greater") -> float:
    """Empirical p with the standard +1 correction (never reports p=0)."""
    null = np.asarray(null, float)
    B = len(null)
    if tail == "greater":
        hits = int((null >= observed - 1e-12).sum())
    else:
        hits = int((null <= observed + 1e-12).sum())
    return (hits + 1) / (B + 1)


# ---------------------------------------------------------------------------
# The four nulls
# ---------------------------------------------------------------------------


def _rotation_draw(red, labels, uniq, n_labels, cyclic, seed):
    """One rotation-null draw: random orthonormal basis of the same subspace,
    re-fit, re-read. Returns (adjacency, circ_corr, curved_ev). Deterministic in
    `seed` so a killed-and-resumed collector never repeats a draw."""
    r = red.shape[1]
    rng = np.random.default_rng(1000 + seed)
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    ev, tok = _curved_read(red @ Q, labels, uniq, n_labels, cyclic, seed=seed)
    return (_adjacency(tok, np.array(uniq)),
            _circ_corr_to_truth(tok, uniq, n_labels), ev)


def _agg_rotation(adjs, circs, evs, obs, perm_chance95):
    adjs, circs, evs = map(np.asarray, (adjs, circs, evs))
    # basis-real fraction: rotations whose recovered ordering still beats the
    # label-permutation 95th-percentile chance ceiling.
    basis_real_frac = float((adjs >= perm_chance95 - 1e-12).mean())
    return {
        "kind": "rotation (random orthonormal basis of the same subspace)",
        "tests": "C2 ordering + C1 EV are basis-real, not PC1/PC2 artifacts",
        "n_null": int(len(adjs)),
        "observed_adjacency": obs["adjacency"],
        "observed_curved_ev": obs["curved_ev"],
        "rot_adjacency_mean": float(adjs.mean()),
        "rot_adjacency_std": float(adjs.std()),
        "rot_adjacency_min": float(adjs.min()),
        "rot_circ_corr_mean": float(circs.mean()),
        "rot_curved_ev_mean": float(evs.mean()),
        "rot_curved_ev_std": float(evs.std()),
        "label_perm_chance95_adjacency": float(perm_chance95),
        "basis_real_fraction": basis_real_frac,
        "curved_ev_cv": float(evs.std() / (abs(evs.mean()) + 1e-12)),
        "null_hist": _hist(adjs, bins=min(20, max(5, len(adjs) // 2))),
    }


def _null_rotation(red, labels, uniq, n_labels, cyclic, obs, n_rot, rng, perm_chance95):
    """N1 (in-memory): random orthonormal rotations of the reading subspace."""
    draws = [_rotation_draw(red, labels, uniq, n_labels, cyclic, i) for i in range(n_rot)]
    adjs, circs, evs = zip(*draws)
    return _agg_rotation(adjs, circs, evs, obs, perm_chance95)


def _adj_null_sample(n: int, rng) -> float:
    """One draw of cyclic-adjacency accuracy under a RANDOM recovered ordering vs
    the true cyclic order arange(n). This is the correct label-permutation null:
    `cyclic_adjacency_accuracy(angle, order)` relabels BOTH its adjacency sets when
    `order` is permuted (so permuting `order` is a no-op) — the real null must
    randomise the recovered *ordering* of the tokens instead."""
    true_adj = {frozenset((i, (i + 1) % n)) for i in range(n)}
    perm = rng.permutation(n)
    rec_adj = {frozenset((int(perm[i]), int(perm[(i + 1) % n]))) for i in range(n)}
    return len(true_adj & rec_adj) / n


def _null_label_perm(tok_angle, uniq, n_labels, obs, n_perm, rng):
    """N2: permute cyclic labels; recompute adjacency / circ-corr of fixed angle."""
    uniq_arr = np.array(uniq)
    n = len(uniq_arr)
    adj_null, circ_null = np.empty(n_perm), np.empty(n_perm)
    for j in range(n_perm):
        adj_null[j] = _adj_null_sample(n, rng)
        # circ-corr null: permute the TRUE angle assignment (well-posed here — the
        # circular correlation is not relabel-invariant the way adjacency is).
        perm = rng.permutation(n)
        true_ang = np.array([2 * np.pi * (uniq_arr[perm][k] / n_labels) for k in range(n)])
        circ_null[j] = abs(C.circular_corr(tok_angle, true_ang))
    return {
        "kind": "label-permutation (shuffle cyclic label assignment)",
        "tests": "C2 the recovered angle orders tokens better than chance",
        "n_null": int(n_perm),
        "observed_adjacency": obs["adjacency"],
        "p_adjacency": _empirical_p(adj_null, obs["adjacency"], "greater"),
        "null_hist": _hist(adj_null),
        "null_adjacency_mean": float(adj_null.mean()),
        "null_adjacency_95": float(np.quantile(adj_null, 0.95)),
        "observed_circ_corr": obs["circ_corr"],
        "p_circ_corr": _empirical_p(circ_null, obs["circ_corr"], "greater"),
        "null_circ_corr_mean": float(circ_null.mean()),
    }


def _gap_closed(cev, l1, l2):
    return (cev - l1) / max(l2 - l1, 1e-9)


def _matched_spectrum_draw(red, cyclic, sigma, seed):
    """One matched-spectrum draw: Gaussian with the same per-PC eigenspectrum but
    NO cyclic structure; re-fit curved + linear. Returns (cev, l1, l2). Determin-
    istic in `seed`."""
    N, r = red.shape
    rng = np.random.default_rng(2000 + seed)
    Y = rng.standard_normal((N, r)) * sigma
    l1 = C.linear_pca_ev(Y, Y, 1)
    l2 = C.linear_pca_ev(Y, Y, 2)
    C._CURVE_N_SEEDS = _NSEEDS
    sae = C.curved_fit(Y, cyclic, steps=_STEPS, seed=seed)
    cev, _ = C.curved_ev_and_positions(sae, Y)
    del sae
    return float(cev), float(l1), float(l2)


def _agg_matched_spectrum(cev_null, l1_null, l2_null, obs):
    cev_null, l1_null, l2_null = map(np.asarray, (cev_null, l1_null, l2_null))
    g1_null = cev_null - l1_null
    gc_null = (cev_null - l1_null) / np.maximum(l2_null - l1_null, 1e-9)
    obs_gc = _gap_closed(obs["curved_ev"], obs["linear_L1"], obs["linear_L2"])
    obs_g1 = obs["curved_ev"] - obs["linear_L1"]
    n_gauss = len(cev_null)
    return {
        "kind": "matched-spectrum Gaussian (same eigenspectrum, no circle)",
        "tests": "C1 that ONE curved coord reaches 2-PC EV parity is a circle "
                 "signature, not what a 1-D curve gets on any spectrum",
        "n_null": int(n_gauss),
        # PRIMARY: fraction of the (lin2 - lin1) gap that curved(1) closes. This is
        # fit-quality-normalised: on a true 1-D-in-2-D circle both PCs ARE the atom's
        # plane so curved closes ~100% of the gap; on a genuine 2-D Gaussian blob a
        # 1-D curve closes only a fraction. obs ~1.0 >> null => circle signature.
        "observed_gap_closed": float(obs_gc),
        "p_gap_closed_vs_null": _empirical_p(gc_null, obs_gc, "greater"),
        "null_hist": _hist(gc_null),
        "null_gap_closed_mean": float(gc_null.mean()),
        "null_gap_closed_95": float(np.quantile(gc_null, 0.95)),
        # secondary: raw curved-minus-lin1 advantage vs the matched Gaussian.
        "observed_curved_minus_lin1": float(obs_g1),
        "p_curved_beats_lin1_vs_null": _empirical_p(g1_null, obs_g1, "greater"),
        "null_curved_minus_lin1_mean": float(g1_null.mean()),
        "observed_curved_minus_lin2": float(obs["curved_ev"] - obs["linear_L2"]),
        "null_curved_ev_mean": float(cev_null.mean()),
        "null_lin1_ev_mean": float(l1_null.mean()),
        "null_lin2_ev_mean": float(l2_null.mean()),
    }


def _null_matched_spectrum(red, labels, uniq, n_labels, cyclic, obs, n_gauss, rng):
    """N3 (in-memory): matched-spectrum Gaussian null over n_gauss draws."""
    sigma = red.std(0, keepdims=True)
    draws = [_matched_spectrum_draw(red, cyclic, sigma, i) for i in range(n_gauss)]
    cev, l1, l2 = zip(*draws)
    return _agg_matched_spectrum(cev, l1, l2, obs)


def _fundamental_mode_fraction(M: np.ndarray, n: int) -> float:
    """Fraction of the cyclically-ordered token-mean power carried by the
    fundamental harmonic (k=1 and its conjugate k=n-1), over all non-DC harmonics.
    A pure circle is FMF ~ 1 (all fundamental); higher-harmonic curve manifolds
    (BSF-style) carry a meaningful tail at k>=2."""
    F = np.fft.fft(M, axis=0)                 # (n, r) complex
    P = (np.abs(F) ** 2).sum(1)               # power per harmonic, summed over dims
    total = float(P[1:].sum())
    fund = float(P[1] + P[n - 1])
    return fund / total if total > 0 else float("nan")


def _null_phase_scramble(M, n, obs, n_phase, rng):
    """N4: phase-scramble the cyclically-ordered token means (preserve power
    spectrum, randomise per-column phases); recompute 2D-PCA-angle adjacency.

    IMPORTANT (honest caveat, verified on the planted-circle sanity check): a
    *pure* circle is entirely fundamental-mode power, and any sum of frequency-1
    components is still an ellipse — so per-column phase randomisation of a
    single-harmonic signal leaves the circle (and its 2D-PCA ordering) intact.
    Phase-scramble is therefore only *discriminative* for claims that rest on
    HIGHER-harmonic phase-locking (BSF curve manifolds, the N-nursery blocks). We
    report the fundamental-mode fraction so callers can see whether the null even
    applies; for a pure circle we expect null adjacency ~ observed (non-informative
    by construction, NOT a claim failure)."""
    fmf = _fundamental_mode_fraction(M, n)
    F = np.fft.rfft(M, axis=0)          # (n//2+1, r) complex
    mag = np.abs(F)
    adj_null = np.empty(n_phase)
    for j in range(n_phase):
        phases = rng.uniform(0, 2 * np.pi, size=F.shape)
        phases[0] = 0.0                  # keep DC real (preserves the mean)
        if n % 2 == 0:                   # keep Nyquist real for even n
            phases[-1] = 0.0
        Fs = mag * np.exp(1j * phases)
        Ms = np.fft.irfft(Fs, n=n, axis=0)
        adj_null[j] = _pca2d_angle_adjacency(Ms, n)
    obs_adj = obs["pca2d_adjacency"]
    null_mean = float(adj_null.mean())
    p = _empirical_p(adj_null, obs_adj, "greater")
    # This is a SUPPLEMENTARY diagnostic, not a test of the W7 cyclic-ORDER claim
    # (that is owned by label-perm). It asks the STRONGER question: does the circular
    # ordering require higher-harmonic phase-locking beyond the fundamental? The
    # 2D-PCA-angle adjacency is driven by the fundamental harmonic, whose power
    # phase-scramble preserves — so whenever the fundamental dominates (as it does for
    # any circle) a large fraction of phase-randomised power-matched signals order
    # cyclically too. p<0.05 => phase-locking beyond the power spectrum (BSF-style,
    # multi-harmonic); p not significant => the ordering is carried by the low-
    # frequency POWER SPECTRUM (smoothness), which is NOT a failure of the cyclic-
    # order claim, just a statement that it is a fundamental-mode / smoothness effect.
    significant = bool(p < 0.05)
    interp = ("phase-locking beyond the power spectrum (higher-harmonic structure)"
              if significant else
              "ordering carried by the low-frequency power spectrum (smoothness); "
              "does not require higher-harmonic phase-locking — fundamental-dominated")
    return {
        "kind": "phase-scramble (preserve power spectrum, randomise phases)",
        "tests": "SUPPLEMENTARY: does the circular ordering require higher-harmonic "
                 "phase-locking beyond the fundamental? (BSF-style; not the W7 order claim)",
        "n_null": int(n_phase),
        "fundamental_mode_fraction": float(fmf),
        "phase_locking_significant": significant,
        "interpretation": interp,
        "observed_pca2d_adjacency": obs_adj,
        "p_pca2d_adjacency": p,
        "null_hist": _hist(adj_null),
        "null_pca2d_adjacency_mean": null_mean,
        "null_pca2d_adjacency_95": float(np.quantile(adj_null, 0.95)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def null_battery(X, labels, *, n_labels, cyclic=True, fitted_basis=None,
                 reduce_dim=16, claims=("rotation", "label_perm",
                                        "matched_spectrum", "phase_scramble"),
                 n_rot=48, n_gauss=128, n_perm=5000, n_phase=5000,
                 seed=0, name="set", canonical=None):
    """Run the matched-null battery for the W7 circle claims on one token set.

    Parameters
    ----------
    X : (N, D) array
        Analysis-ready activations for ONE layer (already per-frame demeaned if
        that is your recipe — this function does not demean).
    labels : (N,) int array
        Cyclic rank of each sample in [0, n_labels). (For the color lane: the
        integer hue bin.)
    n_labels : int
        Number of distinct cyclic tokens (7 weekday, 12 month, ...).
    cyclic : bool
        True for circles (weekday/month/color). Non-cyclic curves (year) are
        supported for the fit but the circular-ordering nulls are less meaningful.
    fitted_basis : (D, r) array, optional
        The reading basis (e.g. the PCA loadings the W7 result used). If given,
        X is reduced through it; else train-PCA to `reduce_dim`.
    claims : tuple[str]
        Which nulls to run (subset of the four).
    canonical : dict, optional
        Reference best-of-2 headline numbers to record alongside (not used for p).

    Returns a JSON-able dict: observed statistics + per-null distribution + p.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    labels = np.asarray(labels).astype(int)
    uniq = sorted(set(labels.tolist()))
    n = len(uniq)

    # ---- reduce to the reading subspace -------------------------------------
    if fitted_basis is not None:
        Vt = np.asarray(fitted_basis, float)
        if Vt.shape[0] != X.shape[1]:      # accept (r, D) too
            Vt = Vt.T
        red = (X - X.mean(0)) @ Vt
        r = red.shape[1]
    else:
        r = min(reduce_dim, X.shape[0] - 2)
        red, _, _, _, _ = C._pca_reduce(X, X, r)

    # ---- observed statistics (battery-internal, same fit budget as nulls) ---
    l1 = C.linear_pca_ev(red, red, 1)
    l2 = C.linear_pca_ev(red, red, 2)
    cev, tok_angle = _curved_read(red, labels, uniq, n_labels, cyclic, seed=0)
    M = _token_means(red, labels, uniq)
    obs = {
        "curved_ev": float(cev),
        "linear_L1": float(l1),
        "linear_L2": float(l2),
        "adjacency": float(_adjacency(tok_angle, np.array(uniq))) if cyclic else None,
        "circ_corr": float(_circ_corr_to_truth(tok_angle, uniq, n_labels)) if cyclic else None,
        "pca2d_adjacency": float(_pca2d_angle_adjacency(M, n)) if cyclic else None,
    }

    out = {
        "name": name, "n_samples": int(X.shape[0]), "n_tokens": int(n_labels),
        "cyclic": bool(cyclic), "reduce_dim": int(r),
        "fit_budget": {"steps": _STEPS, "n_seeds": _NSEEDS},
        "canonical_headline": canonical or {},
        "observed": obs, "nulls": {},
    }

    # label-permutation first: its chance ceiling feeds the rotation null ------
    perm_chance95 = None
    if cyclic and ("label_perm" in claims or "rotation" in claims):
        lp = _null_label_perm(tok_angle, uniq, n_labels, obs, n_perm, rng)
        perm_chance95 = lp["null_adjacency_95"]
        if "label_perm" in claims:
            out["nulls"]["label_perm"] = lp

    if cyclic and "rotation" in claims:
        out["nulls"]["rotation"] = _null_rotation(
            red, labels, uniq, n_labels, cyclic, obs, n_rot, rng, perm_chance95)

    if "matched_spectrum" in claims:
        out["nulls"]["matched_spectrum"] = _null_matched_spectrum(
            red, labels, uniq, n_labels, cyclic, obs, n_gauss, rng)

    if cyclic and "phase_scramble" in claims:
        out["nulls"]["phase_scramble"] = _null_phase_scramble(M, n, obs, n_phase, rng)

    out["verdict"] = _verdict(out)
    return out


def _verdict(out: dict, alpha: float = 0.05) -> dict:
    """Pass/fail per claim from the assembled nulls."""
    nulls = out["nulls"]
    v = {}
    # C2 ordering: label-perm p small AND phase-scramble p small AND rotation stable
    if "label_perm" in nulls:
        lp = nulls["label_perm"]
        c2 = {"label_perm_p_adjacency": lp["p_adjacency"],
              "pass_label_perm": lp["p_adjacency"] < alpha}
        if "rotation" in nulls:
            c2["basis_real_fraction"] = nulls["rotation"]["basis_real_fraction"]
        # C2 verdict rests on label-perm (primary) corroborated by rotation basis-
        # real fraction; phase-scramble is a separate supplementary diagnostic below.
        v["C2_cyclic_order"] = c2
    # supplementary phase-locking diagnostic (NOT a gate on C2)
    if "phase_scramble" in nulls:
        ps = nulls["phase_scramble"]
        v["phase_locking_diagnostic"] = {
            "fundamental_mode_fraction": ps["fundamental_mode_fraction"],
            "p": ps["p_pca2d_adjacency"],
            "phase_locking_significant": ps["phase_locking_significant"],
            "interpretation": ps["interpretation"],
        }
    # C1 EV parity: does ONE curved coord reach 2-PC parity beyond the matched-
    # spectrum Gaussian? (primary) — plus curved-beats-1PC as a secondary view.
    if "matched_spectrum" in nulls:
        ms = nulls["matched_spectrum"]
        v["C1_ev_parity"] = {
            "gap_closed_p_vs_matched_spectrum": ms["p_gap_closed_vs_null"],
            "pass": ms["p_gap_closed_vs_null"] < alpha,
            "observed_gap_closed": ms["observed_gap_closed"],
            "null_gap_closed_mean": ms["null_gap_closed_mean"],
            "secondary_curved_beats_lin1_p": ms["p_curved_beats_lin1_vs_null"],
        }
    return v


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_battery(out: dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    nulls = out["nulls"]
    panels = [k for k in ("label_perm", "matched_spectrum", "phase_scramble", "rotation")
              if k in nulls]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.3))
    if len(panels) == 1:
        axes = [axes]

    def _bars(ax, hist, color="0.7"):
        edges = np.array(hist["edges"]); counts = np.array(hist["counts"])
        ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=(edges[1] - edges[0]) * 0.95,
               color=color, edgecolor="none")

    for ax, k in zip(axes, panels):
        nd = nulls[k]
        ax.set_title(f"{out['name']} — {k}", fontsize=9)
        if k == "label_perm":
            _bars(ax, nd["null_hist"])
            ax.axvline(nd["observed_adjacency"], color="crimson", lw=2,
                       label=f"obs adj={nd['observed_adjacency']:.2f}\np={nd['p_adjacency']:.3f}")
            ax.set_xlabel("cyclic adjacency (null: random order)")
        elif k == "matched_spectrum":
            _bars(ax, nd["null_hist"])
            ax.axvline(nd["observed_gap_closed"], color="crimson", lw=2,
                       label=f"obs gap-closed={nd['observed_gap_closed']:.2f}\n"
                             f"p={nd['p_gap_closed_vs_null']:.3f}")
            ax.set_xlabel("(curved−lin1)/(lin2−lin1)")
        elif k == "phase_scramble":
            _bars(ax, nd["null_hist"])
            ax.axvline(nd["observed_pca2d_adjacency"], color="crimson", lw=2,
                       label=f"obs adj={nd['observed_pca2d_adjacency']:.2f}\n"
                             f"p={nd['p_pca2d_adjacency']:.3f} "
                             f"{'phase-lock' if nd['phase_locking_significant'] else '(spectrum)'}\n"
                             f"FMF={nd['fundamental_mode_fraction']:.2f}")
            ax.set_xlabel("2D-PCA-angle adjacency")
        elif k == "rotation":
            _bars(ax, nd["null_hist"], color="steelblue")
            ax.axvline(nd["observed_adjacency"], color="crimson", lw=2,
                       label=f"obs adj={nd['observed_adjacency']:.2f}\n"
                             f"basis-real={nd['basis_real_fraction']:.2f}")
            ax.axvline(nd["label_perm_chance95_adjacency"], color="gray", ls="--",
                       label=f"chance95={nd['label_perm_chance95_adjacency']:.2f}")
            ax.set_xlabel("adjacency under random rotation")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Matched-null battery: {out['name']} "
                 f"(n_tokens={out['n_tokens']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI: load a cached W7 set and run the battery (subprocess-isolated per set)
# ---------------------------------------------------------------------------


_SETS = ("weekday", "month")  # cached real harvests in probe_out/


def _load_set(name: str):
    """Load a cached W7 set, pick the same analysis layer as the headline json,
    per-template demean it, and return (X, labels, n_labels, canonical)."""
    entry, layers = C.load_harvest_set(name, C.HERE / "probe_out")
    jf = C.HERE / "probe_out" / "curved_feature_probes.json"
    res = json.loads(jf.read_text())["results"][name]
    L = int(res["layer"])
    X = C._demean_per_template(entry[L], entry["template_idx"])
    labels = entry["rank"].astype(int)
    canonical = {
        "layer": L,
        "curved_ev_insample": res["insample_ev"]["curved"],
        "linear_L1_insample": res["insample_ev"]["linear_L1"],
        "linear_L2_insample": res["insample_ev"]["linear_L2"],
        "cyclic_adjacency": res["ordering_curved"]["cyclic_adjacency_accuracy"],
        "pca2d_adjacency": res["ordering_linear"]["pca2d_angle_cyclic_adjacency_accuracy"],
    }
    return X, labels, int(entry["n_labels"]), bool(entry["cyclic"]), canonical


def _synthetic_set(name="weekday", n=7, seed=0):
    """Planted noisy circle (sanity check: all C2 nulls should PASS decisively)."""
    rng = np.random.default_rng(seed)
    D = 48
    b = rng.standard_normal((2, D))
    rows, labels = [], []
    for ti in range(5):
        off = 0.15 * rng.standard_normal(D)
        for u in range(n):
            th = 2 * np.pi * u / n
            v = np.cos(th) * b[0] + np.sin(th) * b[1] + off + 0.05 * rng.standard_normal(D)
            rows.append(v); labels.append(u)
    return np.stack(rows), np.array(labels), n, True, {"synthetic": True}


def run_one(name: str, synthetic=False, **kw) -> dict:
    """One-shot in-memory battery (fine when the box is not memory-starved).
    On the shared box use the checkpointed harness (`main` default) instead."""
    C._cap_threads()
    if synthetic:
        X, labels, n_labels, cyclic, canonical = _synthetic_set(name)
    else:
        X, labels, n_labels, cyclic, canonical = _load_set(name)
    print(f"[null] {name}: N={X.shape[0]} D={X.shape[1]} n_tokens={n_labels} "
          f"cyclic={cyclic}", flush=True)
    out = null_battery(X, labels, n_labels=n_labels, cyclic=cyclic, name=name,
                       canonical=canonical, **kw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "synthetic_" if synthetic else ""
    (OUT_DIR / f"null_{tag}{name}.json").write_text(json.dumps(out, indent=2, default=float))
    plot_battery(out, OUT_DIR / f"null_{tag}{name}.png")
    _print_verdict(out)
    return out


# ---------------------------------------------------------------------------
# Checkpointed, fit-granular harness (survives an aggressive external OOM reaper)
#
# The shared box SIGKILLs any ~300MB torch process within seconds under fleet
# memory pressure, so a whole-set battery never completes. We split it into tiny,
# resumable units: reduce+observed once (`prep`), the refit-free nulls in one shot
# (`fast`), and the two refit nulls as APPEND-ONLY jsonl collectors that record
# every single draw immediately — a kill loses at most the in-flight draw, and the
# next attempt resumes from the saved count. Draws are deterministic in their index
# so resume never repeats or biases. `assemble` stitches the parts into the final
# json + figure. All of this is orchestrated by many retried subprocesses.
# ---------------------------------------------------------------------------


PARTS = OUT_DIR / "parts"


def _prep_paths(name):
    return PARTS / f"{name}_prep.npz", PARTS / f"{name}_observed.json"


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def _jsonl_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def phase_prep(name: str, reduce_dim=16) -> int:
    """Reduce + observed fit (1 curved fit). Idempotent; skips if already done."""
    C._cap_threads()
    PARTS.mkdir(parents=True, exist_ok=True)
    prep_npz, obs_json = _prep_paths(name)
    if prep_npz.exists() and obs_json.exists():
        print(f"[prep] {name}: already done", flush=True)
        return 0
    X, labels, n_labels, cyclic, canonical = _load_set(name)
    uniq = sorted(set(labels.tolist()))
    r = min(reduce_dim, X.shape[0] - 2)
    red, _, _, _, _ = C._pca_reduce(X, X, r)
    l1 = C.linear_pca_ev(red, red, 1)
    l2 = C.linear_pca_ev(red, red, 2)
    cev, tok_angle = _curved_read(red, labels, uniq, n_labels, cyclic, seed=0)
    M = _token_means(red, labels, uniq)
    obs = {
        "curved_ev": float(cev), "linear_L1": float(l1), "linear_L2": float(l2),
        "adjacency": float(_adjacency(tok_angle, np.array(uniq))) if cyclic else None,
        "circ_corr": float(_circ_corr_to_truth(tok_angle, uniq, n_labels)) if cyclic else None,
        "pca2d_adjacency": float(_pca2d_angle_adjacency(M, len(uniq))) if cyclic else None,
    }
    np.savez(prep_npz, red=red, labels=labels, tok_angle=tok_angle, M=M,
             n_labels=n_labels, cyclic=cyclic, reduce_dim=r, n_samples=X.shape[0])
    obs_json.write_text(json.dumps(
        {"observed": obs, "canonical": canonical, "n_labels": int(n_labels),
         "cyclic": bool(cyclic), "reduce_dim": int(r), "n_samples": int(X.shape[0]),
         "fit_budget": {"steps": _STEPS, "n_seeds": _NSEEDS}}, indent=2, default=float))
    print(f"[prep] {name}: obs curved_ev={cev:.3f} adj={obs['adjacency']} "
          f"pca2d={obs['pca2d_adjacency']} [saved]", flush=True)
    return 0


def _load_prep(name):
    prep_npz, obs_json = _prep_paths(name)
    z = np.load(prep_npz, allow_pickle=False)
    meta = json.loads(obs_json.read_text())
    return z, meta


def phase_fast(name: str, n_perm=5000, n_phase=5000) -> int:
    """The two refit-FREE nulls (label-permutation + phase-scramble) in one shot."""
    C._cap_threads()
    z, meta = _load_prep(name)
    red, labels = z["red"], z["labels"]
    uniq = sorted(set(labels.tolist()))
    n_labels, cyclic = meta["n_labels"], meta["cyclic"]
    obs, tok_angle, M = meta["observed"], z["tok_angle"], z["M"]
    rng = np.random.default_rng(7)
    if cyclic:
        lp = _null_label_perm(tok_angle, uniq, n_labels, obs, n_perm, rng)
        ps = _null_phase_scramble(M, len(uniq), obs, n_phase, rng)
        (PARTS / f"{name}_label_perm.json").write_text(json.dumps(lp, default=float))
        (PARTS / f"{name}_phase_scramble.json").write_text(json.dumps(ps, default=float))
        print(f"[fast] {name}: label_perm p={lp['p_adjacency']:.4f} | "
              f"phase FMF={ps['fundamental_mode_fraction']:.2f}", flush=True)
    return 0


def phase_collect(name: str, which: str, target: int, max_per_call=200) -> int:
    """Append draws for a refit null to its jsonl until `target` reached. Each draw
    is flushed immediately (crash-safe). Returns 0 when target met, 2 if more work
    remains (driver re-invokes). `max_per_call` bounds the kill window per process."""
    C._cap_threads()
    z, meta = _load_prep(name)
    red, labels = z["red"], z["labels"]
    uniq = sorted(set(labels.tolist()))
    n_labels, cyclic = meta["n_labels"], meta["cyclic"]
    path = PARTS / f"{name}_{which}.jsonl"
    done = _jsonl_count(path)
    if done >= target:
        print(f"[collect] {name}/{which}: {done}/{target} done", flush=True)
        return 0
    sigma = red.std(0, keepdims=True) if which == "matched_spectrum" else None
    made = 0
    with path.open("a") as f:
        for i in range(done, target):
            if which == "rotation":
                adj, circ, ev = _rotation_draw(red, labels, uniq, n_labels, cyclic, i)
                row = {"i": i, "adj": adj, "circ": circ, "ev": ev}
            else:  # matched_spectrum
                cev, l1, l2 = _matched_spectrum_draw(red, cyclic, sigma, i)
                row = {"i": i, "cev": cev, "l1": l1, "l2": l2}
            f.write(json.dumps(row, default=float) + "\n")
            f.flush(); os.fsync(f.fileno())
            made += 1
            if made >= max_per_call:
                break
    now = _jsonl_count(path)
    print(f"[collect] {name}/{which}: +{made} -> {now}/{target}", flush=True)
    return 0 if now >= target else 2


def phase_assemble(name: str, n_rot: int, n_gauss: int) -> int:
    """Stitch prep + observed + all null parts into the final json + figure."""
    z, meta = _load_prep(name)
    labels = z["labels"]
    uniq = sorted(set(labels.tolist()))
    obs, cyclic, n_labels = meta["observed"], meta["cyclic"], meta["n_labels"]
    out = {
        "name": name, "n_samples": meta["n_samples"], "n_tokens": int(n_labels),
        "cyclic": bool(cyclic), "reduce_dim": meta["reduce_dim"],
        "fit_budget": meta["fit_budget"], "canonical_headline": meta["canonical"],
        "observed": obs, "nulls": {},
    }
    lp_path = PARTS / f"{name}_label_perm.json"
    ps_path = PARTS / f"{name}_phase_scramble.json"
    if lp_path.exists():
        out["nulls"]["label_perm"] = json.loads(lp_path.read_text())
    perm_chance95 = out["nulls"].get("label_perm", {}).get("null_adjacency_95", 1.0)
    rot_rows = _jsonl_rows(PARTS / f"{name}_rotation.jsonl")[:n_rot]
    if rot_rows:
        out["nulls"]["rotation"] = _agg_rotation(
            [r["adj"] for r in rot_rows], [r["circ"] for r in rot_rows],
            [r["ev"] for r in rot_rows], obs, perm_chance95)
    ms_rows = _jsonl_rows(PARTS / f"{name}_matched_spectrum.jsonl")[:n_gauss]
    if ms_rows:
        out["nulls"]["matched_spectrum"] = _agg_matched_spectrum(
            [r["cev"] for r in ms_rows], [r["l1"] for r in ms_rows],
            [r["l2"] for r in ms_rows], obs)
    if ps_path.exists():
        out["nulls"]["phase_scramble"] = json.loads(ps_path.read_text())
    out["verdict"] = _verdict(out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"null_{name}.json").write_text(json.dumps(out, indent=2, default=float))
    plot_battery(out, OUT_DIR / f"null_{name}.png")
    _print_verdict(out)
    return 0


def _print_verdict(out: dict):
    v = out["verdict"]
    print(f"\n=== {out['name']} verdict ===", flush=True)
    if "C1_ev_parity" in v:
        c1 = v["C1_ev_parity"]
        print(f"  C1 EV parity (curved(1) closes the 2nd-PC gap vs matched-spectrum): "
              f"p={c1['gap_closed_p_vs_matched_spectrum']:.4f} "
              f"[{'PASS' if c1['pass'] else 'FAIL'}]  "
              f"obs gap-closed={c1['observed_gap_closed']:.3f} "
              f"vs null {c1['null_gap_closed_mean']:.3f}  "
              f"(2ndary curved>lin1 p={c1['secondary_curved_beats_lin1_p']:.4f})", flush=True)
    if "C2_cyclic_order" in v:
        c2 = v["C2_cyclic_order"]
        print(f"  C2 cyclic order: label-perm p={c2.get('label_perm_p_adjacency'):.4f} "
              f"[{'PASS' if c2.get('pass_label_perm') else 'FAIL'}]", flush=True)
        if "basis_real_fraction" in c2:
            print(f"     rotation basis-real fraction={c2['basis_real_fraction']:.2f}", flush=True)
    if "phase_locking_diagnostic" in v:
        pl = v["phase_locking_diagnostic"]
        print(f"  [diagnostic] phase-locking beyond power spectrum: p={pl['p']:.4f} "
              f"{'significant' if pl['phase_locking_significant'] else 'n.s.'} "
              f"(FMF={pl['fundamental_mode_fraction']:.2f}) — {pl['interpretation']}", flush=True)


def _retry(base, argv, tag, tries):
    """Run a phase subprocess up to `tries` times; treat rc 0 as done, rc 2 as
    'progress made, call again', anything else (incl. -9 SIGKILL) as retry."""
    import subprocess
    for attempt in range(1, tries + 1):
        rc = subprocess.run(base + argv, env=os.environ).returncode
        if rc == 0:
            return True
        if rc != 2:
            print(f"[driver] {tag} rc={rc} (attempt {attempt}/{tries})", flush=True)
    return False


def _drive(name: str, n_rot, n_gauss, n_perm, n_phase) -> None:
    """Orchestrate one set through the checkpointed phases with heavy retries."""
    base = [sys.executable, str(Path(__file__).resolve())]
    tries = int(os.environ.get("MATCHED_NULL_RETRIES", "40"))
    common = ["--n-rot", str(n_rot), "--n-gauss", str(n_gauss),
              "--n-perm", str(n_perm), "--n-phase", str(n_phase)]
    prep_npz, obs_json = _prep_paths(name)
    # 1) prep (1 fit) — retry until the observed json exists
    for _ in range(tries):
        if prep_npz.exists() and obs_json.exists():
            break
        _retry(base, ["--phase", "prep", "--set", name] + common, f"{name}/prep", 1)
    # 2) refit-free nulls
    for _ in range(tries):
        if (PARTS / f"{name}_label_perm.json").exists():
            break
        _retry(base, ["--phase", "fast", "--set", name] + common, f"{name}/fast", 1)
    # 3) refit collectors — call repeatedly until the jsonl reaches target
    for which, target in (("rotation", n_rot), ("matched_spectrum", n_gauss)):
        path = PARTS / f"{name}_{which}.jsonl"
        for attempt in range(tries * 4):
            if _jsonl_count(path) >= target:
                break
            _retry(base, ["--phase", "collect", "--set", name, "--which", which] + common,
                   f"{name}/{which}", 1)
    # 4) assemble
    _retry(base, ["--phase", "assemble", "--set", name] + common, f"{name}/assemble", tries)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=str, default=None, help="which cached set")
    ap.add_argument("--phase", type=str, default=None,
                    choices=["prep", "fast", "collect", "assemble", "oneshot"])
    ap.add_argument("--which", type=str, default=None,
                    choices=["rotation", "matched_spectrum"])
    ap.add_argument("--synthetic", action="store_true", help="planted-circle sanity check")
    ap.add_argument("--n-rot", type=int, default=32)
    ap.add_argument("--n-gauss", type=int, default=80)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-phase", type=int, default=5000)
    ap.add_argument("--max-per-call", type=int, default=200)
    args = ap.parse_args()
    kw = dict(n_rot=args.n_rot, n_gauss=args.n_gauss, n_perm=args.n_perm, n_phase=args.n_phase)

    # ---- individual phases (invoked by the driver as isolated subprocesses) ----
    if args.phase == "prep":
        return phase_prep(args.set)
    if args.phase == "fast":
        return phase_fast(args.set, n_perm=args.n_perm, n_phase=args.n_phase)
    if args.phase == "collect":
        target = args.n_rot if args.which == "rotation" else args.n_gauss
        return phase_collect(args.set, args.which, target, max_per_call=args.max_per_call)
    if args.phase == "assemble":
        return phase_assemble(args.set, args.n_rot, args.n_gauss)

    # ---- synthetic / one-shot in-memory (only when the box is not starved) ----
    if args.synthetic or args.phase == "oneshot":
        if args.set:
            run_one(args.set, synthetic=args.synthetic, **kw)
        else:
            for name in _SETS:
                run_one(name, synthetic=args.synthetic, **kw)
        return 0

    # ---- default: checkpointed driver over both sets --------------------------
    PARTS.mkdir(parents=True, exist_ok=True)
    targets = [args.set] if args.set else list(_SETS)
    for name in targets:
        print(f"\n[driver] ===== {name} =====", flush=True)
        _drive(name, args.n_rot, args.n_gauss, args.n_perm, args.n_phase)
    print(f"\n[done] {OUT_DIR}/null_*.json + null_*.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
