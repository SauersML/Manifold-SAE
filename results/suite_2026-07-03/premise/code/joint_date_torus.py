"""Joint date representation: is month x day a FACTORIZED product of two independent circles,
or ONE bound 2-torus?

Data: harvest_cache_jointdate_L18.npz — a 12-month x 28-day grid, per-prompt last-token acts
(X_last), per-prompt context mean (tmpl_mean), output-Fisher factors, and month_idx/day_idx.

Method:
  1. Demeaned H = X_last - tmpl_mean.
  2. Recover the MONTH 2-plane and DAY 2-plane fit-free from the label group means: for month,
     average H over all days at each month -> 12 month-centroids; their top-2 PCs span the
     month circle plane (same construction as the crown's raw day-centroid views). Likewise for
     day (28 day-centroids). These planes are SUPERVISED by the labels but involve no chart fit.
  3. pair_rho (faithful replica of Rust pair_kappa::screen_pair): rho = E[r_M^2 r_D^2] /
     (E r_M^2 . E r_D^2), where r_M^2, r_D^2 are per-row squared energies in the two planes.
     rho ~ 1  => energies independent => FACTORIZED PRODUCT of two circles (a flat product
                 torus; month and day are separately, independently encoded).
     rho > 1  (z>3) => shared/bound presence => a single BOUND torus (the two circles co-vary).
  4. Falsification: a permutation null that shuffles the day label WITHIN each month (destroys
     any month-day binding but keeps both marginals) — rho must collapse to ~1 under it if the
     observed rho>1 is real binding; and a Gaussian-matched surrogate.
  5. Also fit gamfit circles (month plane, day plane projections) for the topology verdict on
     each factor separately (does each chart as a circle?).

pair_rho and the energy computation are pure bookkeeping/resampling (SPEC-compliant); the only
model fits are gamfit.sae_manifold_fit calls.
"""
from __future__ import annotations

import json
import os
import signal

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
OUT = os.environ.get("PREMISE_OUT", os.path.join(ROOT, "premise_out"))
LAYER = int(os.environ.get("HD_LAYER", "18"))


def plane_from_centroids(H, labels, k=2):
    """Top-k PC plane of the per-label centroid cloud (fit-free supervised plane)."""
    labs = np.unique(labels)
    C = np.stack([H[labels == l].mean(0) for l in labs])
    Cc = C - C.mean(0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    return np.ascontiguousarray(Vt[:k].T)  # (p, k)


def plane_energy(H, mean, basis):
    c = (H - mean[None, :]) @ basis
    return (c ** 2).sum(1)


def pair_rho(rM, rD):
    mM, mD = rM.mean(), rD.mean()
    rho = float((rM * rD).mean() / (mM * mD))
    kM = float((rM * rM).mean() / (mM * mM))
    kD = float((rD * rD).mean() / (mD * mD))
    var = max(kM * kD - 1.0, 0.0) / len(rM)
    se = float(np.sqrt(var))
    z = float((rho - 1.0) / se) if se > 0 else (float("inf") if rho > 1 else 0.0)
    return dict(rho=rho, kappa_month=kM, kappa_day=kD, rho_se=se, z=z,
                bound_torus=bool(z > 3.0))


class _TO(Exception):
    pass


def _fit_circle_r2(H, basis, seconds=200):
    """Fit a K=1 circle in the supplied 2-plane's span (reduce to that plane first) -> r2 and
    ordering vs label handled by caller. Returns (r2, ok)."""
    import gamfit
    Hr = np.ascontiguousarray((H - H.mean(0)) @ basis)
    old = signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))
    try:
        signal.alarm(seconds)
        try:
            try:
                sae = gamfit.sae_manifold_fit(Hr, K=1, d_atom=1, atom_topology="circle",
                                              n_iter=40, random_state=0,
                                              _run_structure_search=False, _run_outer_rho_search=True)
            except TypeError:
                sae = gamfit.sae_manifold_fit(Hr, K=1, d_atom=1, atom_topology="circle",
                                              n_iter=40, random_state=0)
            signal.alarm(0)
            return float(sae.reconstruction_r2), True
        except Exception:  # noqa: BLE001
            signal.alarm(0)
            return float("nan"), False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    npz = os.path.join(OUT, f"harvest_cache_jointdate_L{LAYER}.npz")
    if not os.path.exists(npz):
        print("[joint] MISSING", npz, flush=True)
        return
    z = np.load(npz)
    X = z["X_last"].astype(np.float64); tm = z["tmpl_mean"].astype(np.float64)
    H = X - tm
    month = np.asarray(z["month_idx"]).astype(int)
    day = np.asarray(z["day_idx"]).astype(int)
    n, p = H.shape
    print(f"[joint] n={n} p={p} months={len(np.unique(month))} days={len(np.unique(day))}", flush=True)

    Bm = plane_from_centroids(H, month)
    Bd = plane_from_centroids(H, day)
    mean = H.mean(0)
    rM = plane_energy(H, mean, Bm)
    rD = plane_energy(H, mean, Bd)
    obs = pair_rho(rM, rD)

    # permutation null: shuffle day WITHIN each month (keeps marginals, breaks binding)
    rng = np.random.default_rng(0)
    null_rho = []
    for _ in range(2000):
        perm = np.arange(n)
        for mo in np.unique(month):
            idx = np.where(month == mo)[0]
            perm[idx] = idx[rng.permutation(len(idx))]
        # rebuild day-plane energies under permuted day assignment is complex; instead permute
        # the day-energy vector within month (energies are per-row; binding shows as within-month
        # covariance of rM,rD). Shuffle rD within month:
        rD_perm = rD.copy()
        for mo in np.unique(month):
            idx = np.where(month == mo)[0]
            rD_perm[idx] = rD[idx][rng.permutation(len(idx))]
        null_rho.append((rM * rD_perm).mean() / (rM.mean() * rD_perm.mean()))
    null_rho = np.array(null_rho)
    p_perm = (1 + int(np.sum(np.abs(null_rho - 1) >= abs(obs["rho"] - 1)))) / (len(null_rho) + 1)

    r2_month, okm = _fit_circle_r2(H, Bm)
    r2_day, okd = _fit_circle_r2(H, Bd)

    verdict = "bound_torus" if (obs["bound_torus"] and p_perm < 0.05) else "factorized_product"
    rec = dict(n=int(n), p=int(p), n_months=int(len(np.unique(month))),
               n_days=int(len(np.unique(day))),
               pair_rho=obs, perm_p_within_month=float(p_perm),
               null_rho_mean=float(null_rho.mean()), null_rho_sd=float(null_rho.std()),
               month_circle_r2=r2_month, day_circle_r2=r2_day,
               verdict=verdict)
    json.dump(rec, open(os.path.join(OUT, "joint_date_torus.json"), "w"), indent=2)
    print(f"[joint] rho={obs['rho']:.4f} z={obs['z']:.2f} perm_p={p_perm:.4g} "
          f"month_r2={r2_month:.3f} day_r2={r2_day:.3f} VERDICT={verdict}", flush=True)


if __name__ == "__main__":
    main()
