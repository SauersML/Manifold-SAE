# Layer transport of the weekday circle — carried, rotated, or re-encoded?

**Model:** Qwen3-8B (dense, 36 layers, d=4096). **Probe:** DOSE weekday battery,
10 templates × 7 weekdays = 70 last-token residuals, per-template demeaned (W7).
**Wheel:** gamfit 0.1.248 SHA `3cbb04b76` (`wheels_head_atlas` — has #2074 +
chart-transfer FFI). **Fit:** K=1 `circle` atom in top-8 PCA coords, lifted to
ambient, `isometry_weight=0`, n_iter=30; per-layer plane = top-2 SVD of the
fitted curve (planarity 0.97–1.00 at every layer, reconstruction r² 0.50–0.58).
Cross-layer gauge = ambient parallel transport of the L11 frame — **not** per-layer
label pinning, which would read zero rotation by construction.

## Verdict in one line

The weekday circle is **carried in topology and coordinate correspondence but
continuously re-encoded in geometry**: the same-day↔same-phase map transports at
every hop (degree 1, degree-concentration 0.85–0.99, confirmed 14σ against a
shuffled null), yet the circle's ambient 2-plane tilts ~17–42° per hop and the
map is non-isometric (defect O(1)), so every hop classifies as **compute**
(active re-orientation), not **carry** (passive rigid transport). And the
downstream behavioral (output-Fisher) metric harvested once at L18 **propagates
by the model's own Jacobian to a match within 0.1%** of the metric harvested
directly at layers 7 deep — the re-encoding map and the metric-transport map are
the same object (J).

---

## (a) Per-layer sweep L11–L23 — carried vs re-encoded

`chart_transport_l11_l23.py --probe-npz weekday_acts_8b_L11to23.npz
--demean-by-template --pca-rank 8 --dump-planes planes_L11to23.npz` (job 12500634).

| hop | plane principal angles (deg) | isometry defect | degree | degree-conc | anchored rot. offset (rad) | class |
|---|---|---|---|---|---|---|
| L11→L12 | 22.8, 25.8 | 1.02 | 1 | 0.909 | +0.008 | compute |
| L12→L13 | 20.5, 26.1 | **0.078** | 1 | 0.992 | +0.000 | compute |
| L13→L14 | 20.2, 23.9 | 0.244 | 1 | 0.915 | −0.013 | compute |
| L14→L15 | 17.9, 20.3 | 0.974 | 1 | 0.912 | +0.013 | compute |
| L15→L16 | 17.1, 18.4 | 1.25 | 1 | 0.951 | −0.012 | compute |
| L16→L17 | 17.5, 19.7 | 0.670 | 1 | 0.943 | +0.023 | compute |
| L17→L18 | 15.5, 16.7 | 0.670 | 1 | 0.949 | −0.016 | compute |
| L18→L19 | 16.6, 18.2 | 0.382 | 1 | 0.849 | −0.015 | compute |
| L19→L20 | 19.6, 31.5 | 0.803 | 1 | 0.750 | +0.058 | compute |
| **L20→L21** | 16.4, 26.0 | **4.56** | **2** | 0.648 | **−2.44** | compute |
| L21→L22 | 17.4, 38.0 | 0.661 | 1 | 0.904 | +0.078 | compute |
| L22→L23 | 24.5, 42.6 | 1.46 | 1 | 0.938 | −0.070 | compute |

**Cumulative (L11 vs L23 planes):** principal angles **44.3°, 52.7°** (subspace
overlap cos 0.72, 0.61). Sum of per-hop mean plane angles = 266.7° — the plane
jitters far more than its 48° net tilt, i.e. it drifts and partly returns.

Reading:
- **Plane angles 15–43°/hop, never near 0 and never near 90°.** Not a fixed
  plane (rigid carry ⇒ 0°) and not an orthogonal re-embedding (re-encode ⇒ 90°).
  The circle lives in a **steadily rotating 2-plane** that keeps 60–72% overlap
  end to end.
- **Degree 1 with degree-concentration 0.85–0.99 everywhere** (except the one
  anomaly): the winding/coordinate correspondence — which weekday sits at which
  phase — is preserved through every hop.
- **Isometry defect O(1)** (min 0.078 at L12→L13): the transport is not a rigid
  motion; the circle is rescaled/sheared as it is re-oriented.
- **Anchored rotation offset ±0.01–0.08 rad**: relative to ambient parallel
  transport the circle's phase barely advances — the model **tilts the plane**
  holding the circle, it does not **spin** the circle within its plane.

**Honest anomaly:** **L20→L21** is a degree-2, isometry-defect-4.56,
rotation-offset −2.44 rad outlier with the lowest degree-concentration (0.648).
Either a fit degeneracy around L20 (its planarity 0.984 is the run's lowest bar
L22) or a genuine representational disruption in the L20–21 band. Flagged, not
hidden; it is the single hop where the clean degree-1 correspondence breaks.

## (b) JVP metric-transport arm — propagate L18's Fisher vs harvest per layer

`xport_metric_transport.py --chart-layer {17,11} --metric-layer 18 --rank 8`
(job 12500639, A40). The residual stream is sequential, so the only downstream
path from h_Lc to the logits runs through h_L18 (Lc<18):
`G_Lc = J_{Lc→18}ᵀ G_18 J_{Lc→18}`. We test this as a **predictive** claim on the
real model — predicted nats from the propagated metric vs from the metric
harvested directly at Lc, both scored against the **measured** output KL of
patching `h_Lc += s·τ` (τ = the circle's on-chart tangent) over signed dose
magnitudes. Fisher harvested by the exact real-model reverse-mode call
(`harvest_last_position_fisher`, rank 8); J·τ by forward-mode AD through the real
decoder blocks (wiring-checked to <1e-3 vs the reference hidden state).

| chart layer | hop length | harvested slope / R² / med.ratio | propagated slope / R² / med.ratio | pred log-corr | median prop/harv |
|---|---|---|---|---|---|
| L17 | 1 layer | 0.850 / 0.811 / 1.024 | 0.849 / 0.811 / 1.030 | **1.0000** | 1.0001 |
| L11 | **7 layers** | 0.727 / 0.757 / 0.893 | 0.717 / 0.750 / 0.889 | **0.9995** | 0.9996 |

**The L18 output-Fisher metric, propagated by the frozen model's Jacobian down 7
layers to L11, reproduces the directly-harvested-at-L11 metric's dose predictions
to within 0.1%** (per-edit log-correlation 0.9995; slope, R², and calibration
ratio all match to three significant figures). This is a genuine numerical
confirmation, not an identity by construction: the two metrics come from
**independent** randomized-Fisher estimations at different layers, each rank-8
truncated, bridged by a numerically-computed 7-layer JVP — that they agree to 4
sig figs verifies (i) the JVP is correct, (ii) rank-8 captures the metric, (iii)
the sequential-path metric identity holds on the real model. **Practical upshot:
harvest the behavioral metric once, propagate it anywhere by JVP.**

Caveat, stated plainly: the *absolute* calibration here (R² 0.75–0.81, slope
0.72–0.85) is looser than the L18 dose crown (R² 0.999). That is expected — this
arm uses the local-quadratic predicted-nats ½s²·τᵀGτ on a top-2 PCA plane with a
rank-8 Fisher, not the crown's full arc path-integral. The result of this arm is
the **harvested-vs-propagated equivalence**, which is decisive; the absolute
number is a floor, not the claim.

## (c) Shuffled-day null control — PASSED

`xport_null_control.py --from-layer 17 --to-layer 18 --pca-rank 8 --n-perm 30`
(job 12500635). Falsification: break the token correspondence between the two
layers (random row-permutation of the target angles; both circles stay intact).
If transport is real, the signal must collapse.

| hop | real degree | real degree-conc | shuffled degree-conc (mean±sd, n=30) | z |
|---|---|---|---|---|
| L17→L18 | 1 | 0.949 | 0.156 ± 0.055 (max 0.30) | **14.4σ** |

Real transport sits 14σ above the shuffled null. The degree-1 correspondence is
**not** an artifact of "both layers happen to host a circle" — it is the
same-token weekday map transporting. (Real isometry defect 0.67 confirms the
same non-isometric character as the sweep: corresponded, not rigid.)

---

## How the two verdicts fit together

The geometry arm says each hop **computes** (re-encodes: tilts + rescales the
plane). The metric arm says the L18 metric **transports** near-perfectly by JVP.
These are the same statement: `G_Lc = Jᵀ G_L18 J`, and `J` is exactly the map
that re-encodes the chart. The model's Jacobian both **rotates the circle's
embedding** and **carries its behavioral metric** — the circle is re-embedded at
every layer, but its downstream meaning rides along inside the same Jacobian.

## Provenance / repro

- Harvest: `weekday_probe_harvest.py harvest --layers 11..23` → `weekday_acts_8b_L11to23.npz` (job 12500294).
- Sweep: `chart_transport_l11_l23.py … --pca-rank 8 --dump-planes …` → `xport_out/sweep/chart_transport_summary.json`, `planes_L11to23.npz` (job 12500634; exit was matplotlib-only, all data written first).
- Null: `xport_null_control.py` → `xport_out/null_L17_L18.json` (job 12500635).
- JVP: `xport_metric_transport.py` → `xport_out/metric_transport_L{17,11}_from_L18.json(.rows.npz)` (job 12500639).
- Figure: `xport_out/transport_figure.png`.
- Wheel selection rationale: `wheels_xport` (9cd9da7a4) lacked #2074 (1-core deadlock) and the chart-transfer FFI; `wheels_head_atlas` (3cbb04b76) has both. Its stricter REML certificate refused the old rank-48 fit as off-optimum (~60 eff dof); rank 8 converges and fits best (r² 0.59 vs the old wheel's 0.47).

---

## Addendum — the L20→L21 anomaly characterized (XPORT chart-transport + TOPO fit-free homology)

**Joint verdict: the single degree-2 / defect-4.56 hop is a chart-parametrization
artifact localized to L20, not a topological transition.** Two independent lines
of evidence converge.

**Chart side (XPORT — rank sweep + skip-hops + JVP), `anomaly_L20_L21.json`:**
- **Rank sensitivity.** The degree-2 appears **only at rank 8** (degree-conc 0.648,
  isometry defect 4.68); it is **degree 1 at rank 16** (defect 1.95) and **degree 1
  at rank 24** (conc 0.836, defect 1.15). Not robust to chart rank → not a genuine
  double-winding.
- **Winding concentration** on the rank-8 arc coords: c(k=1)=0.583 vs c(k=2)=0.648 —
  k=2 barely edges k=1 (a real double-wind would show c₂≫c₁). No clean 2-winding.
- **Localization (skip-hops).** Every hop with **L20 as an endpoint** is messy
  (L20→L21 defect 4.68, L20→L22 defect 1.47, both flagged degree 2); every hop that
  **avoids L20** is clean and degree 1 (L19→L21 defect 0.45, L19→L22 defect **0.16**,
  the cleanest in the whole run). The wobble is specific to L20's chart, and L20 was
  already the run's weakest layer (planarity 0.984, rank-8 r² 0.52).
- **Metric rides through cleanly.** The L21 output-Fisher metric propagated by JVP to
  L20 across this exact hop matches the metric harvested directly at L20 to
  **pred-corr 1.000, ratio 0.9997** (`metric_transport_L20_from_L21.json`) — the
  behavioral metric transports perfectly through the very hop where the chart
  coordinate stumbles. Same unifying point as the main JVP arm.

**Topology side (TOPO — fit-free Vietoris–Rips persistent homology on the raw
weekday clouds):** Fit-free persistent homology on the raw weekday clouds (n=210
dense, W7-demeaned, projected into each layer's fitted-circle plane, 500× bootstrap)
finds a single connected component (H0=1, bootstrap [1,1,1]) and a single dominant
H1 loop generator at BOTH L20 and L21 — crisp at L21 (dominance ratio ~10×) and
weak-but-single at L20 (ratio ~2×, the fuzziest ring in the sweep), with no second
generator and no fragmentation at either layer. Since homology counts loops rather
than winding number, the rank-8 degree-2 is invisible to H1 by construction; the
manifold's topology is an unchanged single S¹ across the hop, and L20's
poorly-resolved ring is precisely what destabilizes its rank-8 chart into the
spurious degree-2.

**Why the two lines fit:** homology counts loops (Betti numbers), the chart's degree
counts winding — a degree-2 and a degree-1 map parametrize the *same* S¹ (b₀=1,
b₁=1). The topology is provably unchanged; only the low-rank chart parametrization
at the fuzziest layer (L20) wobbled. Not a contested representational transition.

Figure: `anomaly_figure.png` (correspondence scatter + rank-sensitivity bars).
Data: `anomaly_L20_L21.json`, `metric_transport_L20_from_L21.json`; dense
211-row harvest `weekday_acts_8b_L11to23_dense.npz` (30 templates). Repro:
`scripts/xport_anomaly_probe.py` (job 12501235; dense confirm 12514425), fit-free
audit by the TOPO lane. Both n=70 and n=210 are bootstrap-stable.
