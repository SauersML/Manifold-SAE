"""Cross-layer chart transport vs. compute — torch-backend fit, frontier + small model.

Companion to ``chart_transfer_layers.py``. Same science (see that file's docstring),
but the per-layer K=1 circle chart is fit with the **torch backend**
``gamfit.torch.ManifoldSAE`` (backprop), which is the robust fit for real demeaned
residual probes — the REML ``sae_manifold_fit`` joint solve does not converge on this
data (outer cost-stall; also recorded in probe_out/NOTES.md). The transport itself is
the Rust ``gamfit.layer_transport`` machinery (a robust 1-D REML transport fit),
independent of how the per-layer angles were obtained.

Answers, per adjacent layer hop and across the ladder:
  * winding **degree** + **isometry defect** — degree ±1 & near-zero defect ⇒ the block
    *rotates* the circle (TRANSPORT layer); large defect ⇒ it *recomputes* the chart
    (COMPUTE layer) — the exact TRANSPORT/COMPUTE criterion of layer_transport.rs.
  * **composition law** ``h_{L→L''} =? h_{L'→L''} ∘ h_{L→L'}`` (ladder gauge-quotient test).
  * ambient **compute gap** ``R²_native − R²_transported`` — decoding L''s own chart at
    the transported coordinates ``h(t_L)`` vs at its natively-fit coordinates.

Also reports a 2-D-PCA-angle transport as a coordinate-source robustness cross-check.

Data sources (``CT_SOURCE``):
  * ``qwen25`` (default): local ``probe_out/harvest_{set}.npz`` (Qwen2.5-0.5B, D=896,
    layers 5/8/11/14, sets weekday/month).
  * ``qwen3``: node2 shard dirs ``$CT_HARVEST/qwen3_32b_probe_{set}_l{L}`` (Qwen3-32B,
    D=5120, layers 24/32/40, sets weekday/month/year/color) via load_shards.
"""

from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_v, "8")

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def _load_shard_dir(path: str):
    """Load one Qwen3-32B probe shard dir → (X (n,D), labels, order, cyclic, templates)."""
    sys.path.insert(0, os.environ.get("CT_GAM_EXAMPLES",
                                      "/models/sauers_build/gam_fable/examples"))
    import residual_shard_io as rio  # noqa: E402

    reader = rio.load_shards(path)
    rows = np.concatenate([b for b in reader.batches(100000)], 0).astype(np.float64)
    man = reader.manifest
    return (rows, [str(x) for x in man["labels"]], np.asarray(man["order"]),
            bool(man["cyclic"]), list(man.get("templates", [])))


def load_set(source: str, harvest: str, probe: str, name: str):
    """Return dict{L: X} plus (labels, order, cyclic, n_templates, layers)."""
    if source == "qwen25":
        d = np.load(os.path.join(probe, f"harvest_{name}.npz"), allow_pickle=True)
        layers = [int(x) for x in d["layers"]]
        template_idx = np.asarray(d["template_idx"])
        order = np.asarray(d["template_idx"]) * 0  # placeholder; real order below
        labels = [str(x) for x in d["labels"]]
        # order = the cyclic label index; recover from labels
        uniq = list(dict.fromkeys(labels))
        order = np.asarray([uniq.index(x) for x in labels])
        n_templates = int(len(labels) // max(1, len(uniq)))
        Xs = {L: np.asarray(d[f"L{L}"], dtype=np.float64) for L in layers}
        return Xs, labels, order, bool(d["cyclic"]), n_templates, layers
    # qwen3 shard dirs
    layers = [int(x) for x in os.environ.get("CT_LAYERS", "24,32,40").split(",")]
    Xs, labels, order, cyclic, templates = {}, None, None, None, None
    for L in layers:
        p = os.path.join(harvest, f"qwen3_32b_probe_{name}_l{L}")
        X, labels, order, cyclic, templates = _load_shard_dir(p)
        Xs[L] = X
    n_templates = max(1, len(templates))
    return Xs, labels, order, bool(cyclic), n_templates, layers


def per_template_demean(X: np.ndarray, n_templates: int) -> np.ndarray:
    """Subtract each template sentence's mean (probe recipe). Rows are label-major:

    label block of ``n_templates`` rows each, so template index = row % n_templates.
    """
    out = X.copy()
    if n_templates <= 1:
        return out - out.mean(0, keepdims=True)
    tid = np.arange(len(X)) % n_templates
    for t in np.unique(tid):
        m = tid == t
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def pca_reduce(X: np.ndarray, dim: int):
    Xc = X - X.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    dim = min(dim, vt.shape[0])
    Z = Xc @ vt[:dim].T
    return np.ascontiguousarray(Z - Z.mean(0, keepdims=True))


def _ev(X, Xhat):
    sse = float(((X - Xhat) ** 2).sum())
    sst = float(((X - X.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Torch K=1 circle chart                                                      #
# --------------------------------------------------------------------------- #
def fit_circle_torch(Z: np.ndarray, steps: int, n_basis: int, n_seeds: int, seed: int):
    """Best-of-seeds torch ManifoldSAE circle fit; returns (sae, ev, cfg)."""
    import torch
    from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

    D = Z.shape[1]
    best = None
    for s in range(seed, seed + n_seeds):
        torch.manual_seed(s)
        cfg = ManifoldSAEConfig(
            input_dim=D, n_atoms=1, intrinsic_rank=1,
            atom_manifold="circle", atom_basis="fourier", n_basis_per_atom=int(n_basis),
            sparsity={"kind": "softmax_topk", "target_k": 1,
                      "tau_start": 4.0, "tau_min": 1.0, "tau_steps": steps},
            encoder_hidden=64, init_scale=0.2, dtype=torch.float64)
        sae = ManifoldSAE(cfg)
        x = torch.tensor(Z, dtype=torch.float64)
        opt = torch.optim.Adam(sae.parameters(), lr=8e-3)
        sae.train()
        for _ in range(steps):
            out = sae(x)
            loss = ((out.x_hat - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.sparsity.advance_temperature()
        sae.eval()
        with torch.no_grad():
            ev = _ev(Z, sae(x).x_hat.numpy())
        if best is None or ev > best[1]:
            best = (sae, float(ev), cfg)
    return best


def torch_positions_ev(sae, Z):
    """(ev, angle01, z, native_xhat) for Z under fitted sae."""
    import torch
    with torch.no_grad():
        out = sae(torch.tensor(Z, dtype=torch.float64))
    xhat = out.x_hat.numpy()
    ang = out.positions[:, 0, 0].numpy()  # single atom/coord, in [0,1)
    z = out.z.numpy()                      # (n, 1) gate
    return _ev(Z, xhat), ang, z, xhat


def torch_decode_at(sae, angle01: np.ndarray, z: np.ndarray):
    """Ungated-then-gated decode of the fitted chart at arbitrary angles ``angle01``.

    Reuses the model's OWN basis kernel (``_eval_basis_on_manifold`` → Rust
    ``basis_with_jet``) and learned ``decoder_blocks`` with the supplied per-token
    gate ``z`` (the native gate of that token). Self-checked by the caller against the
    native ``x_hat`` at the native angles.
    """
    import torch
    from gamfit.torch.manifold_sae import _eval_basis_on_manifold

    pos = torch.tensor(np.asarray(angle01, dtype=np.float64).reshape(-1, 1, 1))
    curves = _eval_basis_on_manifold(pos, sae.cfg, sae._forward_centers)
    per_atom = torch.einsum("nfk,fkd->nfd", curves, sae.decoder_blocks)
    zt = torch.tensor(np.asarray(z, dtype=np.float64).reshape(pos.shape[0], 1))
    xhat = (zt.unsqueeze(-1) * per_atom).sum(dim=1)
    return xhat.detach().numpy()


def pca2_angle(Z: np.ndarray) -> np.ndarray:
    """2-D-PCA angle (radians) — the robust linear circle-coordinate cross-check."""
    Zc = Z - Z.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Zc, full_matrices=False)
    p2 = Zc @ vt[:2].T
    return np.arctan2(p2[:, 1] - p2[:, 1].mean(), p2[:, 0] - p2[:, 0].mean())


# --------------------------------------------------------------------------- #
# One set                                                                     #
# --------------------------------------------------------------------------- #
def run_set(name, Xs, order, cyclic, n_templates, layers, pca_dim, steps, n_basis,
            n_seeds, seed):
    from gamfit.layer_transport import fit_transport, layer_transport_ladder

    topo = "circle" if cyclic else "interval"
    print(f"\n=== {name}: layers={layers} n={len(order)} cyclic={cyclic} "
          f"n_templates={n_templates} topo={topo} ===", flush=True)

    per_layer = {}
    for L in layers:
        Zd = per_template_demean(np.asarray(Xs[L], dtype=np.float64), n_templates)
        Z = pca_reduce(Zd, min(pca_dim, Zd.shape[0] - 1))
        t0 = time.time()
        got = fit_circle_torch(Z, steps, n_basis, n_seeds, seed)
        secs = time.time() - t0
        if got is None:
            print(f"[{name}] L{L}: torch fit failed", flush=True)
            continue
        sae, ev, _cfg = got
        ev2, ang01, z, xhat_native = torch_positions_ev(sae, Z)
        # self-check the decode helper reproduces native x_hat at native angles
        xhat_chk = torch_decode_at(sae, ang01, z)
        rel = float(np.linalg.norm(xhat_chk - xhat_native) /
                    (np.linalg.norm(xhat_native) + 1e-30))
        rad = ang01 * 2.0 * np.pi
        per_layer[L] = dict(sae=sae, Z=Z, ang01=ang01, rad=rad, z=z, ev=float(ev2),
                            decode_ok=bool(rel < 1e-6), decode_rel=rel, secs=secs,
                            pca2_rad=pca2_angle(Z))
        print(f"[{name}] L{L}: {secs:.1f}s torch_ev={ev2:.4f} decode_ok={rel < 1e-6} "
              f"rel={rel:.1e}", flush=True)

    fit_layers = [L for L in layers if L in per_layer]

    def transport_hops(coord_key, topo_key):
        hops = []
        for a, b in zip(fit_layers[:-1], fit_layers[1:]):
            ca, cb = per_layer[a][coord_key], per_layer[b][coord_key]
            try:
                tr = fit_transport(ca, cb, topo, topo)
                rep = tr.report(layer_from=a, layer_to=b)
            except Exception as exc:  # noqa: BLE001
                hops.append(dict(layer_from=a, layer_to=b, error=str(exc)))
                print(f"[{name}] {topo_key} L{a}->L{b} transport failed: {exc}", flush=True)
                continue
            recon = {}
            if coord_key == "rad" and per_layer[b]["decode_ok"]:
                t_hat_rad = np.asarray(tr.eval(ca), dtype=np.float64).ravel()
                saeb = per_layer[b]["sae"]; Zb = per_layer[b]["Z"]; zb = per_layer[b]["z"]
                rec_native = torch_decode_at(saeb, per_layer[b]["ang01"], zb)
                rec_trans = torch_decode_at(saeb, (t_hat_rad / (2 * np.pi)) % 1.0, zb)
                recon = dict(r2_native=_ev(Zb, rec_native),
                             r2_transported=_ev(Zb, rec_trans))
                recon["compute_gap"] = recon["r2_native"] - recon["r2_transported"]
            hop = dict(layer_from=a, layer_to=b, degree=rep.get("degree"),
                       degree_concentration=rep.get("degree_concentration"),
                       isometry_defect=rep.get("isometry_defect"),
                       isometry_defect_se=rep.get("isometry_defect_se"),
                       topology_preserved=rep.get("topology_preserved"),
                       transport_edf=rep.get("transport_edf"),
                       residual_rms=rep.get("residual_rms"), **recon)
            hops.append(hop)
            print(f"[{name}] {topo_key} L{a}->L{b}: degree={hop['degree']} "
                  f"iso_defect={hop['isometry_defect']} topo_ok={hop['topology_preserved']} "
                  f"resid_rms={hop['residual_rms']} recon={recon}", flush=True)
        return hops

    hops_sae = transport_hops("rad", "sae")
    hops_pca = transport_hops("pca2_rad", "pca2")

    ladder = None
    if len(fit_layers) >= 3:
        try:
            ladder = layer_transport_ladder([per_layer[L]["rad"] for L in fit_layers],
                                            topo, layers=fit_layers)
            for th in ladder.get("two_hop", []):
                print(f"[{name}] ladder {th.get('layer_from')}->{th.get('layer_to')}: "
                      f"comp_defect={th.get('composition_defect')} "
                      f"p={th.get('composition_p_value')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            ladder = dict(error=str(exc))
            print(f"[{name}] ladder failed: {exc}", flush=True)

    return dict(name=name, layers=layers, fit_layers=fit_layers, cyclic=cyclic,
                per_layer={L: dict(ev=per_layer[L]["ev"], decode_ok=per_layer[L]["decode_ok"],
                                   decode_rel=per_layer[L]["decode_rel"], secs=per_layer[L]["secs"])
                           for L in fit_layers},
                hops_sae=hops_sae, hops_pca=hops_pca, ladder=ladder)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    source = os.environ.get("CT_SOURCE", "qwen25")
    probe = os.environ.get("CT_PROBE", os.path.join(here, "..", "probe_out"))
    harvest = os.environ.get("CT_HARVEST", "/dev/shm/sauers_gpu/harvest")
    out = os.environ.get("CT_OUT", os.path.join(here, f"out_{source}"))
    os.makedirs(out, exist_ok=True)
    pca_dim = int(os.environ.get("CT_PCA", "16"))
    steps = int(os.environ.get("CT_STEPS", "800"))
    n_basis = int(os.environ.get("CT_NBASIS", "4"))
    n_seeds = int(os.environ.get("CT_SEEDS", "3"))
    seed = int(os.environ.get("CT_SEED", "0"))
    sets = os.environ.get("CT_SETS", "weekday,month").split(",")

    import gamfit  # noqa: F401
    print(f"[cfg] source={source} gamfit={getattr(gamfit,'__version__','?')} pca_dim={pca_dim} "
          f"steps={steps} n_basis={n_basis} seeds={n_seeds} sets={sets}", flush=True)

    results = {}
    for name in sets:
        try:
            Xs, labels, order, cyclic, n_templates, layers = load_set(
                source, harvest, probe, name)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: load failed: {exc}", flush=True)
            continue
        results[name] = run_set(name, Xs, order, cyclic, n_templates, layers,
                                pca_dim, steps, n_basis, n_seeds, seed)

    payload = dict(config=dict(source=source, pca_dim=pca_dim, steps=steps,
                               n_basis=n_basis, n_seeds=n_seeds, seed=seed), results=results)
    jpath = os.path.join(out, "chart_transfer.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    md = _report_md(payload)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(md)
    print("\n" + md, flush=True)
    print(f"[out] {jpath}", flush=True)
    return 0


def _report_md(payload) -> str:
    cfg = payload["config"]
    L = [f"# Cross-layer chart transport vs. compute ({cfg['source']})\n",
         "Per-layer K=1 circle chart (torch backend), coordinate transport "
         "`t_{L'}=h(t_L)` via `gamfit.layer_transport`. Degree ±1 + small isometry "
         "defect ⇒ block **rotates** the circle (TRANSPORT); large defect ⇒ "
         "**recomputes** (COMPUTE). `compute_gap = R²_native − R²_transported`.\n"]
    for name, r in payload.get("results", {}).items():
        L.append(f"\n## {name} (layers {r['fit_layers']}, cyclic={r['cyclic']})\n")
        L.append("| layer | chart EV | decode ok |")
        L.append("|---|---:|---|")
        for lyr in r["fit_layers"]:
            pl = r["per_layer"][lyr]
            L.append(f"| L{lyr} | {pl['ev']:.4f} | {pl['decode_ok']} |")
        L.append("\n**SAE-chart transport** | hop | degree | isometry defect | "
                 "topo ok | R² native | R² transported | compute gap |")
        L.append("|---|---:|---:|---|---:|---:|---:|")
        def f(x):
            return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
        for h in r["hops_sae"]:
            if "error" in h:
                L.append(f"| L{h['layer_from']}→L{h['layer_to']} | ERR {h['error'][:40]} |")
                continue
            L.append(f"| L{h['layer_from']}→L{h['layer_to']} | {f(h.get('degree'))} | "
                     f"{f(h.get('isometry_defect'))} | {f(h.get('topology_preserved'))} | "
                     f"{f(h.get('r2_native'))} | {f(h.get('r2_transported'))} | "
                     f"{f(h.get('compute_gap'))} |")
        L.append("\n**2D-PCA-angle transport (robustness cross-check)** | hop | degree | "
                 "isometry defect | topo ok |")
        L.append("|---|---:|---:|---|")
        for h in r["hops_pca"]:
            if "error" in h:
                L.append(f"| L{h['layer_from']}→L{h['layer_to']} | ERR |")
                continue
            L.append(f"| L{h['layer_from']}→L{h['layer_to']} | {f(h.get('degree'))} | "
                     f"{f(h.get('isometry_defect'))} | {f(h.get('topology_preserved'))} |")
        if isinstance(r.get("ladder"), dict) and "two_hop" in r["ladder"]:
            L.append("\n**Composition law:** | two-hop | composition defect | p |")
            L.append("|---|---:|---:|")
            for th in r["ladder"]["two_hop"]:
                L.append(f"| L{th.get('layer_from')}→L{th.get('layer_to')} | "
                         f"{th.get('composition_defect')} | {th.get('composition_p_value')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
