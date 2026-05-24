"""
auto_exp_56: Causal-inference benchmark for cogito-L40 concept-manifold steering.

================================================================================
FRAMING: Potential outcomes + do-calculus for steering interventions.
================================================================================

UNITS:     prompts p ∈ P (|P|=5 in cached data — woefully underpowered)
TREATMENT: A := (concept C ∈ {red, blue, orange, green}, strength α ∈ {0, 2, 5})
OUTCOME:   Y_p(C, α) := warm/(warm+cool) color-token ratio in generation
                       (categorical KL and perplexity dropped — text-only cache)

POTENTIAL-OUTCOMES OBJECTS
--------------------------
  Y_p(C, α)        : potential outcome had prompt p been steered with (C, α)
  Y_p(C, 0)        : baseline potential outcome (no steering — α=0 collapses C)
  Δ_p(C, α) := Y_p(C, α) − Y_p(C, 0)       Individual Treatment Effect (ITE)
  ATE(C, α) := E_p[ Δ_p(C, α) ]            Average Treatment Effect
  CATE(C, α | g(p)) := E_p[ Δ_p(C,α) | p∈g ]   Conditional ATE by prompt class

DO-CALCULUS ESTIMAND
--------------------
The causal target is the interventional distribution of the LLM's generation
under a forced steer:

  E[ Y(do(steer = (C, α))) ] = ∫ Y(C, α | p) dP(p)            (*)

Because steering is an *internal* intervention on the residual stream, classical
backdoor confounding by prompt content is eliminated *by construction* — the
intervention is applied AFTER prompt encoding to a specific hidden layer (L40).
In Pearl's notation, the SCM is:

         Prompt ──► Hidden_L40 ─(+ α·v_C)─► Hidden_L41..L_end ──► Y
                       ▲
                       └── exogenous = sampling noise ε

So there is no open backdoor path from (C, α) to Y given Prompt. The estimand
(*) is identified by a simple g-formula reducing to the within-prompt mean:

  ATE(C, α) = E_p[ E_ε[ Y_p(C, α) ] − E_ε[ Y_p(C, 0) ] ]

Confounders that DO matter (left uncontrolled in this cache):
  • sampling temperature drift across requests (no seed pinning recorded)
  • queue-position / model-warmup effects (single-trajectory cache)
  • prompt-class imbalance (5 prompts, no stratification)
  • Monte-Carlo noise from ε (n=1 sample per cell — fatal)

PROTOCOL THIS BENCHMARK WOULD REQUIRE (not satisfied by cached data):
  P1. ≥200 prompts stratified by domain (nature/object/abstract/...)
  P2. ≥30 sampling seeds per (p, C, α) cell — needed for ε marginalization
  P3. Randomized presentation order of (C, α) within a prompt block
  P4. A placebo concept axis (random unit vector) as negative control
  P5. Pre-registered α grid with α=0 anchored both head and tail

What we have: 1 sample/cell, 5 prompts, 4 concepts, 3 α's = 35 observations
(missing 25/60 cells — likely warm-only probes had cool-baseline dropped).
This is a *demonstration of the analysis pipeline*, not a valid estimate.

================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/user/Manifold-SAE")
CACHE = ROOT / "runs/auto_exp_44_steering/results.json"
OUT = ROOT / "runs/auto_exp_56_results.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_cache():
    return json.loads(CACHE.read_text())


def organize(cache):
    """Index outcomes as Y[prompt, concept, alpha] with NaN for missing cells."""
    prompts = cache["prompts"]
    probes = cache["probes"]
    alphas = cache["alphas"]

    P = {p: i for i, p in enumerate(prompts)}
    C = {c: i for i, c in enumerate(probes)}
    A = {a: i for i, a in enumerate(alphas)}

    Y = np.full((len(prompts), len(probes), len(alphas)), np.nan)
    W = np.full_like(Y, np.nan)  # warm counts
    K = np.full_like(Y, np.nan)  # cool counts

    for r in cache["results"]:
        i, j, k = P[r["prompt"]], C[r["probe"]], A[r["alpha"]]
        Y[i, j, k] = r["ratio"]
        W[i, j, k] = r["warm"]
        K[i, j, k] = r["cool"]
    return Y, W, K, prompts, probes, alphas


def ite(Y):
    """Δ_p(C, α) = Y_p(C, α) − Y_p(C, 0). Shape (P, C, A)."""
    base = Y[:, :, 0:1]  # α=0 baseline per (prompt, concept)
    # Many baseline cells are NaN — fall back to per-prompt baseline averaged
    # across concepts where present (legitimate under the SCM: α=0 ⇒ steering
    # vector is the zero vector, identical across concepts).
    prompt_baseline = np.nanmean(Y[:, :, 0], axis=1, keepdims=True)[:, :, None]
    base_filled = np.where(np.isnan(base), prompt_baseline, base)
    return Y - base_filled


def ate_with_ci(delta, n_boot=2000, seed=0):
    """ATE(C, α) with percentile bootstrap CIs over prompts."""
    rng = np.random.default_rng(seed)
    P, C_, A = delta.shape
    point = np.nanmean(delta, axis=0)  # (C, A)
    boots = np.empty((n_boot, C_, A))
    for b in range(n_boot):
        idx = rng.integers(0, P, P)
        boots[b] = np.nanmean(delta[idx], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return point, lo, hi


def main():
    cache = load_cache()
    Y, W, K, prompts, probes, alphas = organize(cache)
    delta = ite(Y)
    point, lo, hi = ate_with_ci(delta)

    print("=" * 70)
    print("auto_exp_56 — causal steering benchmark (cached, underpowered)")
    print("=" * 70)
    print(f"n_prompts={len(prompts)}  n_concepts={len(probes)}  alphas={alphas}")
    print(f"cells observed: {int(np.isfinite(Y).sum())}/{Y.size}")
    print()
    print("ATE(C, α) := E_p[ Y_p(C,α) − Y_p(C,0) ]   (warm-ratio shift)")
    print("-" * 70)
    for j, c in enumerate(probes):
        for k, a in enumerate(alphas):
            if a == 0.0:
                continue
            print(
                f"  {c:35s}  α={a:>3}  ATE={point[j,k]:+.3f}  "
                f"95%CI=[{lo[j,k]:+.3f}, {hi[j,k]:+.3f}]"
            )

    # Headline ITE table for one concept
    print()
    print("Per-prompt ITE for color_manifold:t1_at_red, α=5")
    j = probes.index("color_manifold:t1_at_red")
    k = alphas.index(5.0)
    for i, p in enumerate(prompts):
        print(f"  Δ={delta[i,j,k]:+.3f}   {p}")

    np.savez(
        OUT,
        Y=Y, W=W, K=K, delta=delta,
        ate=point, ate_lo=lo, ate_hi=hi,
        prompts=np.array(prompts), probes=np.array(probes),
        alphas=np.array(alphas),
    )
    print(f"\nwrote {OUT}")

    # Stretch: forest plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(probes), figsize=(4 * len(probes), 4),
                                  sharey=True)
        if len(probes) == 1:
            axes = [axes]
        nonzero_alphas = [a for a in alphas if a != 0.0]
        x = np.arange(len(nonzero_alphas))
        for j, (ax, c) in enumerate(zip(axes, probes)):
            pts = [point[j, alphas.index(a)] for a in nonzero_alphas]
            los = [lo[j, alphas.index(a)] for a in nonzero_alphas]
            his = [hi[j, alphas.index(a)] for a in nonzero_alphas]
            err = np.array([np.array(pts) - np.array(los),
                            np.array(his) - np.array(pts)])
            ax.errorbar(x, pts, yerr=err, fmt="o", capsize=4, color="C3")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([f"α={a}" for a in nonzero_alphas])
            ax.set_title(c.split(":")[-1])
            if j == 0:
                ax.set_ylabel("ATE on warm-ratio (95% CI)")
        fig.suptitle("Causal effect of steering on warm-color ratio "
                     "(N=5 prompts — illustrative only)")
        fig.tight_layout()
        figpath = ROOT / "runs/auto_exp_56_forest.png"
        fig.savefig(figpath, dpi=120)
        print(f"wrote {figpath}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
