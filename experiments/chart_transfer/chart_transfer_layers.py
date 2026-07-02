"""Cross-layer chart transport vs. compute — the "which block rotates the circle" figure.

The claim
---------
A curved manifold-SAE atom fit at layer L is an explicit 1-manifold chart ``g_L(t)``
through residual-stream space. The *same* semantic feature (weekday circle, month
circle) is present at several layers of the same model. Question (Pillar 2): as the
residual stream passes from layer L to layer L', does the block **transport** the
chart — carry it rigidly, a degree-±1 isometric rotation/relabel of the SAME circle —
or does it **compute** a genuinely new coordinatization (a reshape the transport map
cannot capture)?

We answer it two ways, both from cached multi-layer harvests of the same tokens:

1. **Coordinate-space transport (the headline).** Fit a K=1 circle chart per layer,
   recover each token's angle ``t_L``. Fit the transport map ``t_{L'} = h(t_L)`` with
   ``gamfit.layer_transport`` (all math in Rust): winding **degree**, **isometry
   defect** (‖‑‑‖ of h from a rigid rotation), topology-preserved flag, transport EDF,
   and — across the ladder L5→L8→L11→L14 — the **composition law**
   ``h_{L→L''} =? h_{L'→L''} ∘ h_{L→L'}``. Degree ±1 + tiny isometry defect + tiny
   composition defect ⇒ the blocks *rotate the circle* (pure transport). A large
   isometry/composition defect on some hop ⇒ that block *recomputes* the chart.

2. **Ambient reconstruction transport vs. compute.** Native chart at L' reconstructs
   the L' activations with R²_native (coordinates free). The *transported* prediction
   decodes L''s own chart frame at the transported coordinates ``ĥ = h(t_L)`` (carried
   from the previous layer, not re-estimated). R²_native − R²_transported is the
   fraction of L''s structure that rigid transport of the previous layer's coordinate
   ordering does NOT explain: the block's *compute* contribution.

Data: the D-data multi-layer probe harvests (``harvest_weekday.npz`` /
``harvest_month.npz``): Qwen2.5-0.5B residual stream, layers {5,8,11,14}, D=896, the
weekday (7) and month (12) cyclic token sets, per-template demeaned (the recipe the
probe uses; raw activation is context-dominated). REML ``sae_manifold_fit`` is the
canonical fit (needs real RAM → node2); torch backend is a fallback.

Outputs (under ``$CT_OUT``, default this dir/out):
  * ``chart_transfer.json`` — per-set, per-layer fits + per-hop transport + ladder.
  * ``report.md`` — the transport-vs-compute tables and interpretation.
"""

from __future__ import annotations

import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np


# --------------------------------------------------------------------------- #
# Preprocessing                                                               #
# --------------------------------------------------------------------------- #
def per_template_demean(X: np.ndarray, template_idx: np.ndarray) -> np.ndarray:
    """Subtract each template sentence's mean over its tokens (probe recipe).

    The raw residual is dominated by sentence context; the token-of-interest is a
    small component. Demeaning per template exposes the cyclic feature (without it
    held-out EV goes strongly negative — see probe_out/NOTES.md).
    """
    out = X.copy()
    for t in np.unique(template_idx):
        m = template_idx == t
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def pca_reduce(X: np.ndarray, dim: int):
    """Reduce ambient dim to ``dim`` PCs (the circle lives in a 2-D subspace; this

    keeps the per-layer fit inside the small-fit budget without touching geometry).
    Returns ``(Z, basis)`` with ``Z = X @ basis``, ``basis`` shape ``(D, dim)``.
    """
    Xc = X - X.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = vt[:dim].T
    Z = Xc @ basis
    return np.ascontiguousarray(Z - Z.mean(0, keepdims=True)), np.ascontiguousarray(basis)


# --------------------------------------------------------------------------- #
# Per-layer K=1 circle chart                                                  #
# --------------------------------------------------------------------------- #
def fit_circle_chart(Z: np.ndarray, n_iter: int, seed: int):
    """Fit a single circle atom (K=1) — the proven robust regime — to one layer.

    Tries the REML solver (canonical, SPEC-compliant) with escalating iters/seed;
    returns ``(sae, seconds)`` or ``None`` if every attempt raised.
    """
    import gamfit

    for kw in (dict(n_iter=n_iter, random_state=seed),
               dict(n_iter=n_iter + 20, random_state=seed + 101),
               dict(n_iter=n_iter + 40, random_state=seed + 202)):
        try:
            t0 = time.time()
            sae = gamfit.sae_manifold_fit(Z, K=1, d_atom=1,
                                          atom_topology="circle", **kw)
            return sae, time.time() - t0, kw
        except Exception as exc:  # noqa: BLE001 - solver raises several kinds
            print(f"[fit] circle attempt {kw} failed: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}", flush=True)
    return None


def chart_decode(sae, coords: np.ndarray):
    """Decode a K=1 circle chart at arbitrary coordinates ``coords`` (N,).

    Reuses the SAME Rust basis kernel ``basis_with_jet`` the fit uses; the ambient
    image is ``phi(coords) @ B  (+ anchor)``. Self-checked by the caller against
    ``sae.fitted`` at the model's own coordinates; returns ``(recon, anchor_used)``
    where ``anchor_used`` records whether the stored anchor had to be added to match.
    """
    from gamfit._binding import rust_module

    B = np.asarray(sae.atoms[0].decoder_coefficients, dtype=np.float64)  # (M, p)
    H = int(sae._n_harmonics[0]) if sae._n_harmonics[0] else (B.shape[0] - 1) // 2
    H = max(1, H)
    ct = np.ascontiguousarray(np.asarray(coords, dtype=np.float64).reshape(-1, 1))
    phi, _jet, _pen = rust_module().basis_with_jet("periodic", ct, {"n_harmonics": H})
    recon = np.asarray(phi, dtype=np.float64) @ B
    anchors = [np.asarray(a, dtype=np.float64) for a in sae.get_anchors()]
    anchor = anchors[0] if anchors else np.zeros(B.shape[1])
    return recon, anchor


def _r2(target: np.ndarray, recon: np.ndarray) -> float:
    from gamfit._binding import rust_module
    return float(rust_module().sae_manifold_reconstruction_r2(
        np.ascontiguousarray(target), np.ascontiguousarray(recon)))


def decode_selfcheck(sae, Z: np.ndarray):
    """Return (ok, rel_err, use_anchor). Decode at the model's own coords must

    reproduce ``sae.fitted``; pick the anchor convention (add vs not) that matches.
    """
    coords0 = np.asarray(sae.coords[0]).ravel()
    recon, anchor = chart_decode(sae, coords0)
    fit = np.asarray(sae.fitted, dtype=np.float64)
    e_no = np.linalg.norm(recon - fit) / (np.linalg.norm(fit) + 1e-30)
    e_an = np.linalg.norm(recon + anchor - fit) / (np.linalg.norm(fit) + 1e-30)
    if e_an <= e_no:
        return (e_an < 1e-6), float(e_an), True
    return (e_no < 1e-6), float(e_no), False


# --------------------------------------------------------------------------- #
# One token set                                                               #
# --------------------------------------------------------------------------- #
def run_set(name: str, npz_path: str, pca_dim: int, n_iter: int, seed: int):
    import gamfit
    from gamfit.layer_transport import fit_transport, layer_transport_ladder

    d = np.load(npz_path, allow_pickle=True)
    layers = [int(x) for x in d["layers"]]
    template_idx = np.asarray(d["template_idx"])
    labels = [str(x) for x in d["labels"]]
    print(f"\n=== {name}: layers={layers} n={len(labels)} ===", flush=True)

    per_layer = {}
    for L in layers:
        Xraw = np.asarray(d[f"L{L}"], dtype=np.float64)
        Xd = per_template_demean(Xraw, template_idx)
        Z, _basis = pca_reduce(Xd, min(pca_dim, Xd.shape[0] - 1))
        got = fit_circle_chart(Z, n_iter, seed)
        if got is None:
            print(f"[{name}] L{L}: FIT FAILED", flush=True)
            continue
        sae, secs, kw = got
        ok, rel, use_anchor = decode_selfcheck(sae, Z)
        coords = np.asarray(sae.project(Z, 0)).ravel()
        per_layer[L] = dict(
            sae=sae, Z=Z, coords=coords, r2=float(sae.reconstruction_r2),
            topo=list(sae.atom_topologies), secs=secs, kw=kw,
            decode_ok=bool(ok), decode_rel=float(rel), use_anchor=bool(use_anchor))
        print(f"[{name}] L{L}: {secs:.1f}s r2={sae.reconstruction_r2:.4f} "
              f"topo={sae.atom_topologies} decode_ok={ok} rel={rel:.1e}", flush=True)

    fitted_layers = [L for L in layers if L in per_layer]

    # ---- coordinate-space transport per adjacent hop ----
    hops = []
    for a, b in zip(fitted_layers[:-1], fitted_layers[1:]):
        ta, tb = per_layer[a]["coords"], per_layer[b]["coords"]
        try:
            tr = fit_transport(ta, tb, "circle", "circle")
            rep = tr.report(layer_from=a, layer_to=b)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] transport L{a}->L{b} failed: {exc}", flush=True)
            hops.append(dict(layer_from=a, layer_to=b, error=str(exc)))
            continue

        # ---- ambient reconstruction: native vs transported coords ----
        recon_stats = {}
        if per_layer[b]["decode_ok"]:
            saeb, Zb = per_layer[b]["sae"], per_layer[b]["Z"]
            use_anchor = per_layer[b]["use_anchor"]
            # transported coords: carry layer-a angles through h
            t_hat = np.asarray(tr.eval(ta), dtype=np.float64).ravel()
            rec_native, anchor = chart_decode(saeb, per_layer[b]["coords"])
            rec_trans, _ = chart_decode(saeb, t_hat)
            if use_anchor:
                rec_native = rec_native + anchor
                rec_trans = rec_trans + anchor
            recon_stats = dict(
                r2_native=_r2(Zb, rec_native),
                r2_transported=_r2(Zb, rec_trans),
            )
            recon_stats["compute_gap"] = (
                recon_stats["r2_native"] - recon_stats["r2_transported"])

        hop = dict(
            layer_from=a, layer_to=b,
            degree=rep.get("degree"),
            degree_concentration=rep.get("degree_concentration"),
            isometry_defect=rep.get("isometry_defect"),
            isometry_defect_se=rep.get("isometry_defect_se"),
            topology_preserved=rep.get("topology_preserved"),
            transport_edf=rep.get("transport_edf"),
            smoothing_lambda=rep.get("smoothing_lambda"),
            **recon_stats,
        )
        hops.append(hop)
        print(f"[{name}] L{a}->L{b}: degree={hop['degree']} "
              f"iso_defect={hop['isometry_defect']} topo_ok={hop['topology_preserved']} "
              f"edf={hop['transport_edf']} recon={recon_stats}", flush=True)

    # ---- composition law across the ladder ----
    ladder = None
    if len(fitted_layers) >= 3:
        try:
            coords_list = [per_layer[L]["coords"] for L in fitted_layers]
            ladder = layer_transport_ladder(coords_list, "circle", layers=fitted_layers)
            for th in ladder.get("two_hop", []):
                print(f"[{name}] ladder two-hop {th.get('layer_from')}->"
                      f"{th.get('layer_to')}: comp_defect={th.get('composition_defect')} "
                      f"p={th.get('composition_p_value')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] ladder failed: {exc}", flush=True)
            ladder = dict(error=str(exc))

    return dict(
        name=name, layers=layers, fitted_layers=fitted_layers, labels=labels,
        per_layer={L: dict(r2=per_layer[L]["r2"], topo=per_layer[L]["topo"],
                           secs=per_layer[L]["secs"], decode_ok=per_layer[L]["decode_ok"],
                           decode_rel=per_layer[L]["decode_rel"])
                   for L in fitted_layers},
        hops=hops, ladder=ladder,
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    probe = os.environ.get("CT_PROBE",
                           os.path.join(here, "..", "probe_out"))
    out = os.environ.get("CT_OUT", os.path.join(here, "out"))
    os.makedirs(out, exist_ok=True)
    # 8 PCs: the circle is 2-D extrinsically, so ~8 dims capture it with margin
    # while keeping n/p ~ 4 (n=35/60 tokens) — p≈n overfits the chart and makes the
    # K=1 inner solve oscillate (#1026 EV-degrade/restore churn) with noisy coords.
    pca_dim = int(os.environ.get("CT_PCA", "8"))
    n_iter = int(os.environ.get("CT_NITER", "30"))
    seed = int(os.environ.get("CT_SEED", "0"))

    import gamfit  # noqa: F401  surface a clean error if missing
    print(f"[cfg] gamfit={gamfit.__version__ if hasattr(gamfit,'__version__') else '?'} "
          f"pca_dim={pca_dim} n_iter={n_iter} seed={seed} probe={probe}", flush=True)

    results = {}
    for name in ("weekday", "month"):
        p = os.path.join(probe, f"harvest_{name}.npz")
        if not os.path.exists(p):
            print(f"[skip] {p} missing", flush=True)
            continue
        results[name] = run_set(name, p, pca_dim, n_iter, seed)

    payload = dict(
        config=dict(pca_dim=pca_dim, n_iter=n_iter, seed=seed,
                    model="Qwen2.5-0.5B residual (D=896)", probe=probe),
        results=results,
    )
    jpath = os.path.join(out, "chart_transfer.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    md = _report_md(payload)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(md)
    print("\n" + md, flush=True)
    print(f"[out] {jpath}\n[out] {os.path.join(out, 'report.md')}", flush=True)
    return 0


def _report_md(payload) -> str:
    lines = [
        "# Cross-layer chart transport vs. compute — which block rotates the circle\n",
        "**Claim:** a curved manifold-SAE atom is an explicit chart `g_L(t)`; the same "
        "cyclic feature lives at several layers. We fit a K=1 circle chart per layer, "
        "recover each token's angle, and fit the coordinate transport `t_{L'}=h(t_L)`. "
        "Degree ±1 + small isometry defect + small composition defect ⇒ the blocks "
        "*rotate* the circle (pure transport); a large defect on a hop ⇒ that block "
        "*recomputes* the chart. The ambient `compute_gap = R²_native − R²_transported` "
        "is the fraction of a layer's structure that rigid transport of the previous "
        "layer's coordinates does not explain.\n",
        f"- model: {payload['config']['model']}; PCA dim {payload['config']['pca_dim']}; "
        f"REML K=1 circle per layer.\n",
    ]
    for name, r in payload.get("results", {}).items():
        lines.append(f"\n## {name} (layers {r['fitted_layers']})\n")
        lines.append("| layer | chart R² | topology | decode ok |")
        lines.append("|---|---:|---|---|")
        for L in r["fitted_layers"]:
            pl = r["per_layer"][L]
            lines.append(f"| L{L} | {pl['r2']:.4f} | {pl['topo']} | {pl['decode_ok']} |")
        lines.append("\n| hop | degree | isometry defect | topo preserved | "
                     "R² native | R² transported | compute gap |")
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for h in r["hops"]:
            if "error" in h:
                lines.append(f"| L{h['layer_from']}→L{h['layer_to']} | ERROR: {h['error']} |")
                continue
            def f(x):
                return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
            lines.append(
                f"| L{h['layer_from']}→L{h['layer_to']} | {f(h.get('degree'))} | "
                f"{f(h.get('isometry_defect'))} | {f(h.get('topology_preserved'))} | "
                f"{f(h.get('r2_native'))} | {f(h.get('r2_transported'))} | "
                f"{f(h.get('compute_gap'))} |")
        if isinstance(r.get("ladder"), dict) and "two_hop" in r["ladder"]:
            lines.append("\n**Composition law** `h_{L→L''} =? h_{L'→L''} ∘ h_{L→L'}`:\n")
            lines.append("| two-hop | composition defect | p-value |")
            lines.append("|---|---:|---:|")
            for th in r["ladder"]["two_hop"]:
                lines.append(f"| L{th.get('layer_from')}→L{th.get('layer_to')} | "
                             f"{th.get('composition_defect')} | {th.get('composition_p_value')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
