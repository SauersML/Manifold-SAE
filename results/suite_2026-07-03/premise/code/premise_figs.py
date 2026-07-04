#!/usr/bin/env python3
"""Figures + report for the premise instrument (held-out paired deviance) and the
slow-feature atlas pilot. Reads the JSONs + delta npz pulled from MSI into DATA, writes
PNGs into FIGS. dataviz palette (light+ink), matches the suite."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "figures")
os.makedirs(FIGS, exist_ok=True)

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED = "#d64545"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

PRETTY = {
    "weekday_8b_L18": "weekday · 8B L18", "month_8b_L18": "month · 8B L18",
    "color_35b_L17": "color · 35B L17", "weekday_35b_L17": "weekday · 35B L17",
    "month_35b_L17": "month · 35B L17", "sycophancy_8b_L18": "sycophancy · 8B L18",
    "hedging_8b_L18": "hedging · 8B L18",
}


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def dev_json():
    # prefer the ROBUST aggregated file (aggregate_premise.py); fall back to per-feature results
    p = os.path.join(DATA, "premise_deviance.json")
    if os.path.exists(p):
        d = json.load(open(p))
        if d.get("instrument") == "held_out_paired_deviance_robust":
            return d["results"]
    return [json.load(open(os.path.join(DATA, f))) for f in sorted(os.listdir(DATA))
            if f.startswith("result_") and f.endswith(".json")]


def _sig(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))


def _signed_log(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


def fig_hero(results):
    """THE premise figure: per-feature ROBUST median curvature dividend, in BOTH the behavioral
    (output-Fisher, nats) metric and the raw activation metric, with the distribution-free
    sign-test significance. Positive ⇒ the closed circle reconstructs unseen rows better than a
    straight line. The split (raw pays for some features while behavioral stays flat) is the
    'geometry real in activations, behaviorally inert' story; behavioral-flat everywhere sharpens
    the crown: the metric buys dose FORECASTING, not per-row behavioral likelihood."""
    rs = [r for r in results if r.get("paired_deviance_behavioral", {}).get("n")]
    if not rs:
        return
    rs = list(reversed(rs))
    labels = [PRETTY.get(r["name"], r["name"]) for r in rs]
    beh_med = np.array([r["paired_deviance_behavioral"]["median"] for r in rs])
    beh_p = [r["paired_deviance_behavioral"].get("sign_test_p", 1) for r in rs]
    raw_med = np.array([r["paired_deviance_raw"]["median"] for r in rs])
    raw_p = [r["paired_deviance_raw"].get("sign_test_p", 1) for r in rs]
    y = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.66 * len(rs) + 2.8))
    for ax, med, ps, title, unit in (
        (axes[0], beh_med, beh_p, "Behavioral dividend  (output-Fisher, nats)", "nats"),
        (axes[1], raw_med, raw_p, "Activation-space dividend  (raw)", "act. units²")):
        ax.axvline(0, color=INK2, lw=1.0, ls="--")
        xs = _signed_log(med)
        cols = [BLUE if (m > 0 and p < 0.05) else (YELLOW if (m < 0 and p < 0.05) else INK2)
                for m, p in zip(med, ps)]
        ax.barh(y, xs, color=cols, alpha=0.88, height=0.6)
        span = float(np.nanmax(np.abs(xs))) or 1.0
        ax.set_xlim(-span * 1.7, span * 1.7)
        for i, (m, p) in enumerate(zip(med, ps)):
            ax.text(xs[i] + np.sign(xs[i] or 1) * span * 0.04, i,
                    f"{m:+.3g} {_sig(p)}", va="center",
                    ha="left" if xs[i] >= 0 else "right", fontsize=8.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_xlabel(f"signed log10(1+|median Δ|)   [{unit}]")
        ax.set_title(title, loc="left", fontsize=11.5, pad=10)
    fig.suptitle("Does curvature pay?  Held-out paired dividend Δ = deviance(line) − deviance(circle), median + sign test",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    # legend
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=BLUE, label="circle wins (p<.05)"),
                            Patch(color=YELLOW, label="line wins (p<.05)"),
                            Patch(color=INK2, label="n.s.")],
                   frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig1_curvature_dividend.png"), dpi=200)
    plt.close(fig)


def fig_paired_scatter(results):
    """Per-row held-out deviance: line vs circle, for each feature with data. Points below
    y=x ⇒ circle wins that row."""
    rs = [r for r in results if os.path.exists(os.path.join(DATA, f"deltas_{r['name']}.npz"))]
    if not rs:
        return
    ncol = min(3, len(rs)); nrow = int(np.ceil(len(rs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, r in enumerate(rs):
        ax = axes.flat[k]; ax.set_visible(True)
        z = np.load(os.path.join(DATA, f"deltas_{r['name']}.npz"))
        xl, xc = z["beh_lin"], z["beh_cir"]
        good = np.isfinite(xl) & np.isfinite(xc) & (xl > 0) & (xc > 0)
        xl, xc = xl[good], xc[good]
        if len(xl) == 0:
            continue
        lo = min(xl.min(), xc.min()) * 0.7; hi = max(xl.max(), xc.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=INK2)
        win = xc < xl
        ax.scatter(xl[win], xc[win], s=26, facecolors="none", edgecolors=BLUE, lw=1.2,
                   label="circle wins")
        ax.scatter(xl[~win], xc[~win], s=26, facecolors="none", edgecolors=YELLOW, lw=1.2,
                   label="line wins")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_title(PRETTY.get(r["name"], r["name"]), fontsize=11)
        ax.text(0.04, 0.96, f"{win.mean()*100:.0f}% rows\ncircle wins",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec=GRID, boxstyle="round,pad=0.3"))
        if k == 0:
            ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.set_xlabel("held-out behavioral deviance — LINE")
        ax.set_ylabel("held-out behavioral deviance — CIRCLE")
    fig.suptitle("Paired held-out reconstruction: every point is one row scored by both charts",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_paired_scatter.png"), dpi=200)
    plt.close(fig)


def fig_null(results):
    """Permutation null histogram vs observed, for the top-signal features (validates p)."""
    rs = [r for r in results if os.path.exists(os.path.join(DATA, f"deltas_{r['name']}.npz"))
          and "paired_deviance_behavioral" in r]
    rs = sorted(rs, key=lambda r: r["paired_deviance_behavioral"].get("p_two_sided", 1))[:3]
    if not rs:
        return
    fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 4.0), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, r in zip(axes.flat, rs):
        z = np.load(os.path.join(DATA, f"deltas_{r['name']}.npz"))
        d = z["beh_delta"]; d = d[np.isfinite(d)]
        T = d.mean()
        signs = rng.integers(0, 2, size=(20000, len(d))) * 2 - 1
        null = (signs * d[None, :]).mean(1)
        ax.hist(null, bins=60, color=INK2, alpha=0.35, label="sign-flip null")
        ax.axvline(T, color=BLUE, lw=2.2, label="observed mean Δ")
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        p = r["paired_deviance_behavioral"]["p_two_sided"]
        ax.set_title(f"{PRETTY.get(r['name'], r['name'])}\np={p:.1e}", fontsize=11)
        ax.set_xlabel("mean paired Δ (behavioral, nats)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Paired sign-flip permutation: observed dividend vs the null it must beat",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_permutation_null.png"), dpi=200)
    plt.close(fig)


def fig_atlas():
    a = load("slow_feature_atlas.json")
    if not a:
        return
    groups = [g for g in a["results"] if "pca_ev_fraction_top10" in g]
    if not groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    ax = axes[0]
    for gi, g in enumerate(groups):
        ev = np.array(g["pca_ev_fraction_top10"])
        ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), marker="o",
                color=[BLUE, AQUA][gi % 2],
                label=f"{g['group']} (n={g.get('n_context_means','?')}, PC1={g.get('pc1_fraction',0):.2f})")
    ax.set(xlabel="PC index", ylabel="cumulative EV fraction", ylim=(0, 1.02))
    ax.set_title("Context-mean population: one dominant common-mode axis", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax2 = axes[1]
    labs, real, base, null = [], [], [], []
    for g in groups:
        lab = g.get("feature_of_origin_recovery") or {}
        labg = g.get("feature_of_origin_recovery_gaussian_null") or {}
        labs.append(g["group"]); real.append(lab.get("nn_loo_acc", np.nan))
        base.append(lab.get("majority_baseline", np.nan)); null.append(labg.get("nn_loo_acc", np.nan))
    y = np.arange(len(labs))
    ax2.barh(y + 0.24, real, height=0.22, color=BLUE, label="real context means")
    ax2.barh(y, null, height=0.22, color=INK2, alpha=0.5, label="Gaussian-matched null")
    ax2.barh(y - 0.24, base, height=0.22, color=YELLOW, alpha=0.8, label="majority baseline")
    ax2.set_yticks(y); ax2.set_yticklabels(labs); ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("feature-of-origin 1-NN LOO accuracy")
    ax2.set_title("Context mean carries contextual identity ≫ null", loc="left", fontsize=11.5)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Slow-feature atlas pilot: the subtracted PerContextMean is a modeled feature",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig4_slow_feature_atlas.png"), dpi=200)
    plt.close(fig)


def main():
    results = dev_json()
    if results:
        fig_hero(results); fig_paired_scatter(results); fig_null(results)
    fig_atlas()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
