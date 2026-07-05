"""Aggregate the thmG grid JSONs into the decisive figure for Theorem G / P1.

Design (confound-guarded): rho off everywhere + a fixed smoothing penalty; occupancy varies
n_eff at a FIXED base rank; fidelity is swept two independent ways at FULL occupancy — working
rank (model complexity) and additive isotropic noise sigma (data SNR). Every plotted number is a
gamfit-sourced held-out paired-deviance dividend from thmG_sweep.py. Non-converged cells (fewer
than 4/4 fold-topology fits scored) are EXCLUDED from the verdict (missing data, NOT a weak
verdict) and tallied separately.

Verdict strength per cell = matched-null corrected: magnitude -log10(real sign-test p), sign
from (real median - surrogate median), damped to 0 when the real signal does not clear the
per-cell Gaussian-surrogate freedom floor. Each feature is read in the metric where the premise
found its signal (weekday->behavioral nats; graded->raw activation); measured fidelity (held-out
circle R2) is annotated so axes are calibrated in achieved SNR.

Panels per feature: (1) OCCUPANCY marginal [expect SHARPEN]; (2) FIDELITY-noise marginal
[expect ~FLAT within range]; (3) FIDELITY-rank marginal [expect ~FLAT]; (4) the decisive
dissociation scatter — verdict strength vs MEASURED fidelity, colored by axis, size ~ n_eff.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, ".."))
METRIC_FOR = {"weekday_8b_L18": "behav", "month_8b_L18": "behav",
              "sycophancy_8b_L18": "raw", "hedging_8b_L18": "raw"}


def load(name, indir):
    with open(os.path.join(indir, f"thmG_{name}.json")) as fh:
        return json.load(fh)


def strength_cell(cell, metric):
    """Matched-null-corrected verdict strength (see module docstring)."""
    real = cell[metric]
    sur = cell["surrogate_" + metric]
    p, med, smed = real.get("sign_test_p"), real.get("median"), sur.get("median")
    if p is None or med is None or not np.isfinite(p) or not np.isfinite(med):
        return np.nan
    corr = med - (smed if smed is not None and np.isfinite(smed) else 0.0)
    if med == 0 or (np.sign(corr) != np.sign(med)):
        return 0.0
    return -np.log10(max(p, 1e-6)) * np.sign(corr)


def conv(cell):
    return bool(cell.get("converged"))


def axis_cells(rec, axis, converged_only=True):
    cs = [c for c in rec["grid"] if c["axis"] == axis]
    if converged_only:
        cs = [c for c in cs if conv(c)]
    return cs


def marginal(rec, axis, xkey, metric):
    cs = axis_cells(rec, axis)
    by = {}
    for c in cs:
        by.setdefault(c[xkey], []).append(c)
    xs, ys, r2s = [], [], []
    for k in sorted(by):
        g = by[k]
        xs.append(rec["C"] * k if xkey == "n_tpl" else k)
        ys.append(np.nanmean([strength_cell(c, metric) for c in g]))
        r2s.append(np.nanmean([c["measured_fidelity"]["r2_circle"] for c in g]))
    return np.array(xs), np.array(ys), np.array(r2s)


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "data")
    features = sys.argv[2].split(",") if len(sys.argv) > 2 else ["weekday_8b_L18", "sycophancy_8b_L18"]

    nf = len(features)
    fig, axes = plt.subplots(nf, 4, figsize=(20, 4.6 * nf), squeeze=False)
    summary = {}
    for r, name in enumerate(features):
        rec = load(name, indir)
        metric = METRIC_FOR.get(name, "behav")
        ncells = len(rec["grid"]); nconv = sum(conv(c) for c in rec["grid"])
        summary[name] = dict(metric=metric, n_cells=ncells, n_converged=nconv,
                             n_nonconverged=ncells - nconv)

        ox, oy, or2 = marginal(rec, "occupancy", "n_tpl", metric)
        nx, ny, nr2 = marginal(rec, "fidelity_noise", "sigma_frac", metric)
        rx, ry, rr2 = marginal(rec, "fidelity_rank", "rdim", metric)

        for col, (X, Y, R2, c0, mk, xl, tl) in enumerate([
            (ox, oy, or2, "#c0392b", "o", "occupancy  n_eff (rows firing)", "OCCUPANCY (base rank, σ=0)"),
            (nx, ny, nr2, "#2471a3", "s", "added noise σ (× signal RMS) → lower fidelity", "FIDELITY-noise (full n_eff)"),
            (rx, ry, rr2, "#7d3c98", "D", "working rank (model complexity) → fidelity", "FIDELITY-rank (full n_eff)"),
        ]):
            ax = axes[r][col]
            if len(X):
                ax.plot(X, Y, marker=mk, color=c0, lw=2)
                for x, y, q in zip(X, Y, R2):
                    ax.annotate(f"R²={q:.2f}", (x, y), fontsize=7, xytext=(0, 6),
                                textcoords="offset points", ha="center")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel(xl)
            if col == 0:
                ax.set_ylabel("verdict strength\nsign(med−sur)·−log10 p")
            ax.set_title(f"{name} [{metric}]\n{tl}")

        # Decisive dissociation scatter
        ax = axes[r][3]
        series = (("occupancy", "#c0392b", "occupancy sweep", "o"),
                  ("fidelity_noise", "#2471a3", "fidelity: noise", "s"),
                  ("fidelity_rank", "#7d3c98", "fidelity: rank", "D"))
        for axis, col, lab, mk in series:
            cs = axis_cells(rec, axis)
            xr = [c["measured_fidelity"]["r2_circle"] for c in cs]
            yv = [strength_cell(c, metric) for c in cs]
            szs = [20 + 2.5 * c["n_eff"] for c in cs]
            ax.scatter(xr, yv, s=szs, c=col, marker=mk, alpha=0.75, edgecolor="k", lw=0.4, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("MEASURED fidelity: held-out circle R²")
        ax.set_ylabel("verdict strength")
        ax.set_title(f"{name} [{metric}] — DISSOCIATION\n(size ~ n_eff; converged cells only)")
        ax.legend(fontsize=7)

    fig.suptitle("Theorem G / P1 — topology verdict set by OCCUPANCY, not FIDELITY "
                 "(rho off, fixed smoothing; non-converged cells excluded)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    outpng = os.path.join(RES, "thmG_occupancy_vs_fidelity.png")
    fig.savefig(outpng, dpi=130)
    print("wrote", outpng)

    # Quantitative dissociation: across ALL converged cells, regress |verdict strength| on
    # standardized (n_eff, measured circle R2). Theorem G predicts the n_eff (occupancy) partial
    # slope dominates and stays significant while the fidelity partial slope is weak. This turns
    # the decisive panel into a number rather than an eyeball.
    for name in features:
        rec = load(name, indir)
        metric = METRIC_FOR.get(name, "behav")
        cs = [c for c in rec["grid"] if conv(c)]
        y = np.array([abs(strength_cell(c, metric)) for c in cs], float)
        occ = np.array([c["n_eff"] for c in cs], float)
        fid = np.array([c["measured_fidelity"]["r2_circle"] for c in cs], float)
        keep = np.isfinite(y) & np.isfinite(occ) & np.isfinite(fid)
        y, occ, fid = y[keep], occ[keep], fid[keep]
        # Does this channel carry a real topology verdict at all? Require a consistently-signed,
        # significant preference somewhere (a null channel makes the OLS fit noise).
        signed = np.array([strength_cell(c, metric) for c in cs], float)
        signed = signed[np.isfinite(signed)]
        has_verdict = bool(np.sum(np.abs(signed) >= 1.30) >= 3 and
                           abs(np.sign(signed[np.abs(signed) >= 1.30]).mean()) > 0.8)
        if not has_verdict:
            summary[name]["dissociation_OLS"] = dict(reading="null channel — no topology verdict to test")
        elif len(y) >= 6 and occ.std() > 0 and fid.std() > 0:
            zocc = (occ - occ.mean()) / occ.std()
            zfid = (fid - fid.mean()) / fid.std()
            Xd = np.column_stack([np.ones_like(y), zocc, zfid])
            beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            resid = y - Xd @ beta
            dof = max(len(y) - 3, 1)
            s2 = float(resid @ resid) / dof
            cov = s2 * np.linalg.inv(Xd.T @ Xd)
            se = np.sqrt(np.diag(cov))
            t_occ, t_fid = beta[1] / se[1], beta[2] / se[2]
            # G-consistent iff occupancy carries the effect and fidelity is inert (|beta| and |t|).
            g_ok = abs(beta[1]) > 3 * abs(beta[2]) and beta[1] > 0
            summary[name]["dissociation_OLS"] = dict(
                n=int(len(y)),
                beta_occupancy=float(beta[1]), t_occupancy=float(t_occ),
                beta_fidelity=float(beta[2]), t_fidelity=float(t_fid),
                reading=("occupancy carries verdict strength, fidelity inert (G-consistent)" if g_ok
                         else "fidelity-dominant (G-falsified)" if abs(beta[2]) > abs(beta[1]) and abs(t_fid) > 2
                         else "occupancy-leaning but sub-threshold"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
