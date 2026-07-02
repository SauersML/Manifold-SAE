# mdl_ladder — description-length scoring for the featurizer ladder

Scores any featurizer (TopK/direction SAE, block BSF, curved chart) in **bits/token** at a
task distortion floor, and computes the firing count `f*` where a curved chart's description
becomes shorter than a block's. This is the MDL lane's answer to Goodfire's *Block-Sparse
Featurizers*: their MDL argument (blocks beat directions, optimum `b≈2–4`) extends one rung —
**charts beat blocks for cyclic/curved features**.

- **[DERIVATION.md](DERIVATION.md)** — the two-part MDL account, the crossover `f*`, and the
  term-for-term map onto gamfit's REML negative-log-evidence (a description length).
- **[mdl.py](mdl.py)** — the scorer (library + CLI + JSON interface).
- **[results.json](results.json)** / **[REPORT.md](REPORT.md)** — the bits/token ladder on
  weekday / month / (synthetic year control) / frontier planted circles, with measured `f*`.

## Reproduce

```bash
# from experiments/mdl_ladder/
python mdl.py --probes --synthetic --frontier --out results.json   # rescore all artifacts
python mdl.py --json payload.json                                   # score a lane's own ladder
```

Uses the repo's `gamfit` venv (`/Users/user/gam/.venv/bin/python`) only for the `--probes`
path (it reloads cached `probe_out/*.npz`; **no model load, no GPU** — safe under the box's
OOM reaper). `--synthetic` and `--frontier` read only JSON. The `--json` path is pure numpy.

## JSON interface (for the G-bsf and N-nursery lanes)

Call `mdl.score_json(payload)` or `python mdl.py --json payload.json`. You describe each rung
of your ladder; the scorer returns bits/token and the crossover. **You do not need to touch
`mdl.py`.**

### Payload

```jsonc
{
  "delta2": null,            // per-token distortion floor (MSE). null -> use the residual
                             //   of the best chart present (task-derived matched floor).
  "l_param_bits": null,      // bits per stored dictionary scalar. null -> distortion-matched
                             //   (same per-scalar precision as the code). Or e.g. 16 for fp16.
  "featurizers": [
    {
      "name": "topk-dir",
      "kind": "direction",   // "direction" | "block" | "chart" (labels only; math is uniform)
      "total_var": 1.0,      // V: per-token variance to reconstruct (in your working space)
      "n_tokens": 100000,    // N: corpus tokens (denominator of bits/token)
      "n_firings": 8000,     // f: times THIS feature/atom fires
      "n_params": 16,        // P: dictionary DECODER scalars (direction: 1*p, b-block: b*p,
                             //     circle chart: n_basis*p)
      "g_dict": 4096,        // dictionary size (selection bits log2 C(G,k)); optional, default 1
      "k_active": 1,         // atoms active per firing; optional, default 1

      // ---- provide the coded coefficients ONE of two ways ----
      "coded_var": [0.42],   // (A) per-active-coefficient signal variances. len = coded dim m.
                             //     direction: [lambda1]; 2-block: [lambda1, lambda2];
                             //     chart: [ev_chart * V]  (all signal through the intrinsic coords)
      // -- or --
      "ev": 0.58,            // (B) explained-variance fraction achieved, plus:
      "coded_dim": 2         //     m; we split ev*V equally across m coords.
    }
    // ... more rungs ...
  ],
  "block_name": "block-2",   // optional: names of the two rungs to compute f* between
  "chart_name": "circle-chart"
}
```

### Response

```jsonc
{
  "delta2": 0.42,
  "rows": [
    {
      "name": "...", "kind": "...", "coded_dim_m": 2,
      "code_bits_per_firing": 0.73,       // coefficients + selection
      "code_coeff_bits_per_firing": 0.73,
      "selection_bits_per_firing": 0.0,
      "n_params": 32, "l_param_bits": 0.36, "dict_bits": 11.7,
      "code_bits_total": 25.5, "total_bits": 37.1,
      "bits_per_token": 1.061,
      "residual_achieved": 1.07, "distortion_floor": 1.03,
      "distortion_infeasible": false      // true = cannot reach the floor with its m coords
    }
  ],
  "crossover": {                          // present iff block_name & chart_name resolve
    "delta_code_bits_per_firing": 0.096,  // (b - d_i) * r, per-firing rate the chart frees
    "phi_extra_params": 32,               // Phi = P_chart - P_block (the curvature harmonics)
    "f_star": 121.6,                      // firings above which the chart's DL is shorter
    "f_star_matched_simple": 32.0,        // SNR-independent Phi/(m_block - m_chart)
    "chart_wins_at_actual_f": false,      // n_firings >= f_star ?
    "actual_firings": 35
  }
}
```

### Conventions you must match

- **`total_var` V and `coded_var` are in the same space** (raw, whitened, or PCA-reduced —
  your choice, just be consistent). The probes work in a `reduce_dim=16` whitened space; the
  frontier in `p=9` ambient.
- **`n_params` is decoder scalars only** (the generative map intrinsic→ambient), *not* the
  encoder/amortization MLP. For a `gamfit` circle atom that is `decoder_blocks.numel()` =
  `n_basis * p` (measured: `(1, n_basis, p)`), optionally +1 anchor +1 `log λ`.
- **`distortion_infeasible: true`** means the rung's own residual exceeds the floor — it
  cannot represent the feature to task fidelity at any rate (a direction on a circle). Such a
  rung is excluded from a fair matched-distortion comparison.
- **The floor `delta2` is task-derived.** Default = the best chart's residual, i.e. "reach the
  fidelity a single curved coordinate achieves." Override with your task's MSE budget.

### Programmatic use

```python
from mdl import Featurizer, score, crossover_firings, score_json
block = Featurizer("b2", "block", coded_var=[1.10, 0.34], n_params=32, ev=0.58,
                   total_var=2.55, n_tokens=35, n_firings=35)
chart = Featurizer("circle", "chart", coded_var=[1.49], n_params=64, ev=0.584,
                   total_var=2.55, n_tokens=35, n_firings=35)
delta2 = chart.residual
print(score(block, delta2))
print(crossover_firings(block, chart, delta2))
```

Questions to the MDL lane (M-mdl) only if a convention above is ambiguous for your featurizer;
otherwise the interface is self-contained.
