# W9 / W10 on SAC output — harnesses ready, blocked on the SAC engine

**Status (2026-07-02):** the two harnesses (`manifold_stability_sac.py`,
`ev_budget_sac.py`) are written, import-clean, and correctly consume the SAC
dictionary interface (`SacResult.atoms[k].fit.atoms[0].decoder_coefficients`,
`.delta_ev`, `.coords`). They run up to the first atom fit and then hit a **hard
crash in gamfit's multi-atom manifold fit** on the current node build. This is the
WS-A / WS-B lane (the SAC engine + gauge quotient), not a WS-F harness bug.

## The precise blocker

The dose experiment (WS-F, W8) proves `gamfit.sae_manifold_fit(K=1, d_atom=1,
atom_topology="circle")` is **stable on a clean, unimodal single loop** (the weekday
calendar circle fits to r²=0.966, `atom_topologies=['circle']`). But:

1. **`sae_manifold_fit` does not hold K=1 on multi-modal data — it auto-grows the
   dictionary.** A stable direct call on a planted 3-circle mixture (n=900, p=12)
   returned a **6-atom** dictionary (`topo=['circle']×6`) that co-collapsed
   (`reconstruction EV≈0.35` at the collapse bar, `reseeding all 5/6 atoms`), r²=0.77.
   The month calendar circle (12 tokens) did the same inside the dose run
   (`topo=['circle','circle','circle']`, r²=0.28), which then crashed the OOS path:
   `sae_manifold_predict_oos: decoder_blocks[1] has M=2 but rebuilt basis has M=3`.
2. **The SAC prototype's stabilised path crashes on the node build.**
   `sac_prototype.sac_fit` calls the K=1 fit with `structured_residual_passes=2,
   isometry_weight=1.0, assignment="ibp_map"` (the whitened forward-birth that is
   *meant* to keep each birth unimodal). On node2's `venv_fable` gamfit this
   **segfaults / aborts** mid-fit (W10 left a binary-corrupted log; W9 exited rc=1
   with no Python traceback) on both a 3-circle mixture and a 1-circle+linear mixture.

So the multi-atom composition W9/W10 need — several clean, separable atoms — cannot
currently be produced: the plain fit co-collapses/grows, and SAC's whitened
composition (which exists to fix exactly this) crashes on the frozen node build.

## What is ready to run the moment the SAC engine is stable

* `manifold_stability_sac.py` — two seeds (data row-order/subsample) → SAC dictionary
  each → Hungarian latent match vs principal-angle union-subspace vs canonical
  content-hash, reusing `seed_stability.py` verbatim. Datasets: 1-circle+linear
  mixture (primary), planted multi-circle (opt-in `W9_PLANTED=1`), real W6 slab
  (`/dev/shm/w6/cache_K8.npy`, opt-in).
* `ev_budget_sac.py` — SAC birth ledger → per-atom `(Θ, ΔEV)` frontier vs the gamfit
  linear/sparse dictionary reference at matched Θ (`linear_ev_curve`), plus a
  held-out sequential-replay EV frontier.

Both are validated to the point of the fit; only the SAC engine's stability gates
them. Retry once A1's `fit_stagewise` (or the prototype's structured-residual path)
runs without co-collapse/segfault on multi-atom real-shaped data.
