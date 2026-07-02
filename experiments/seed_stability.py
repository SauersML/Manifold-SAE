"""Seed-stability of sparse dictionaries: latent-level vs subspace-level.

The SAE-instability critique: refit a linear sparse autoencoder with a different
seed and the individual latent atoms come out different. The manifold-SAE claim
is that a curved dictionary is stable at the SUBSPACE level (the frame each chart
spans recurs across seeds) even where the individual latent parameterization
differs. This script quantifies latent-level vs subspace-level agreement for the
dictionaries we can actually fit in this environment, and records the one we
cannot.

WHAT RUNS HERE, AND WHY. Two hard environment facts were established empirically
(see the module-level NOTES and the run log):

  (1) gamfit.sae_manifold_fit does NOT converge in this build (gamfit 0.1.242)
      for any atom topology tried (circle / euclidean / linear), on real OLMo
      activations OR planted-circle synthetic OR the repo's own
      whitened_convergence_probe.make_synthetic data with its own recipe. Every
      fit hits the documented #1784 regime: dictionary co-collapse (EV below the
      data-derived collapse bar, "no atom carries material signal") followed by a
      non-PD Arrow-Schur Schur-complement Cholesky in the outer REML. So the
      curved manifold-SAE arm cannot be produced on this machine; we attempt one
      fit and record the failure verbatim.

  (2) gamfit.sparse_dictionary_fit is FULLY DETERMINISTIC -- a farthest-point
      seed_decoder plus fixed minibatch order, no RNG (see
      crates/gam-sae/src/sparse_dict/). It has no random_state, so its only seed
      knob is data ROW ORDER, and its farthest-point init is order-robust: two
      refits on row-permuted copies of identical data produce the SAME atoms up
      to reordering, hence a byte-identical canonical artifact hash. This lane is
      therefore seed-stable BY CONSTRUCTION -- a real (if negative) answer to the
      instability critique for this specific collapsed-linear lane.

Because neither gamfit function can exhibit latent-level seed instability here
(one won't fit, the other has no seed), we ALSO fit a transparent random-init
"tiling" linear SAE (plain numpy: random Gaussian init + top-1 matching pursuit
+ centroid atom updates) whose seed genuinely randomizes the initialization.
This is the reference that lets the analysis harness demonstrate the claim's
SHAPE quantitatively -- on planted-circle data it reproduces the predicted
pattern (individual atoms seed-unstable, spanned subspaces seed-stable). It is
clearly labeled as an auxiliary random-init dictionary, not a gamfit fit.

CONTENT ADDRESSING. Every fitted dictionary is canonicalized and SHA-256 hashed
with a faithful Python port of gam-sae crates/gam-sae/src/dictionary_artifact.rs
(HASH_VERSION v1): each atom frame is Frobenius-normalized to 1, sign-oriented on
its max-abs entry, epsilon-snapped; atoms are sorted by their per-atom hash, then
hashed with the version + gauge-certificate prefix. The Rust module carries no
PyO3 binding, so this is a byte-faithful reimplementation (deviations: the
residual-gauge certificate string and the topology spelling are supplied by this
port rather than read from the Rust identifiability pass). Identical decoder
blocks -> identical hash, so runs are content-addressed and reproducible.

DATA. Real harvested residual-stream activations from
runs/OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST/activations.npy (OLMo-3.1-32B-Instruct,
last-token residual, 760 prompts x 64 layers x 5120); a narrow 3-layer mid slab
(24,25,26) around the self/qualia analysis layer, projected onto its top PCA
components. Plus fixed real-shaped synthetic: planted 1-D circle manifolds (each
a smooth loop in its own 2-plane of R^p) -- the curved substrate the manifold-SAE
claim is about and the regime where latent-vs-subspace actually diverges.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment

import gamfit
from gamfit import sae_manifold_fit, sparse_dictionary_fit

HERE = Path(__file__).resolve().parent
OUT = HERE / "stability_out"
OUT.mkdir(exist_ok=True)
ACT_SRC = HERE.parent / "runs" / "OLMO3_32B_INSTRUCT_SELF_QUALIA_LAST" / "activations.npy"

SEEDS = (0, 1, 2)
COS_THRESH = 0.90

# ============================ content addressing =============================
# Faithful Python port of gam-sae crates/gam-sae/src/dictionary_artifact.rs (v1).
_HASH_VERSION = b"gam-sae-dictionary-artifact-v1"
_EPS = 1.0e-12


def _canonical_zero(v):
    v = np.asarray(v, dtype=np.float64).copy()
    v[np.abs(v) < _EPS] = 0.0
    return v


def _canonical_decoder_block(frame):
    frame = np.asarray(frame, dtype=np.float64)
    norm = float(np.sqrt((frame * frame).sum()))
    scale = 1.0 / norm if (norm > 0.0 and np.isfinite(norm)) else 1.0
    out = _canonical_zero(frame * scale)
    flat = out.ravel()
    if flat.size and flat[int(np.argmax(np.abs(flat)))] < 0.0:
        out = -out
    return out, norm


def _atom_hash_bytes(topology, block, residual_gauge):
    h = hashlib.sha256()
    h.update(f"{topology}|{block.shape[0]}|{block.shape[1]}|".encode())
    for v in _canonical_zero(block).ravel(order="C"):
        h.update(np.float64(v).tobytes())
    h.update(residual_gauge.encode())
    return h.digest()


def canonical_dictionary_artifact(blocks, topologies, residual_gauges, gauge_cert):
    atoms = []
    for topo, frame, rg in zip(topologies, blocks, residual_gauges):
        cblock, norm = _canonical_decoder_block(frame)
        atoms.append({"topology": topo, "block": cblock, "rg": rg,
                      "_key": _atom_hash_bytes(topo, cblock, rg)})
    atoms.sort(key=lambda a: a["_key"])
    h = hashlib.sha256()
    h.update(_HASH_VERSION)
    h.update(gauge_cert.encode())
    for a in atoms:
        h.update(_atom_hash_bytes(a["topology"], a["block"], a["rg"]))
    return {"content_hash": h.hexdigest(), "gauge_certificate": gauge_cert,
            "n_atoms": len(atoms), "atom_shapes": [list(a["block"].shape) for a in atoms]}


def linear_artifact(decoder):
    """Content artifact for a K x p unit-norm linear dictionary."""
    blocks = [np.asarray(decoder[i:i + 1], dtype=np.float64) for i in range(decoder.shape[0])]
    return canonical_dictionary_artifact(
        blocks, ["linear"] * len(blocks),
        ["flat 1-D atom: sign convention-fixed"] * len(blocks),
        "python-port/v1: residual-gauge certificate not read from Rust")


# ============================ geometry helpers ===============================
def _span(D, tol=1e-8):
    _, s, Vt = np.linalg.svd(np.asarray(D, dtype=np.float64), full_matrices=False)
    r = max(int((s > tol * (s[0] if s.size else 1.0)).sum()), 1)
    return np.ascontiguousarray(Vt[:r].T)


def _subspace_cos(Q1, Q2):
    cs = np.cos(subspace_angles(Q1, Q2))
    return float(cs.mean()), float(cs.min())


def latent_match(D1, D2):
    """Hungarian match unit-norm atoms on |cosine|; matched-cosine distribution."""
    C = np.abs(np.asarray(D1) @ np.asarray(D2).T)
    r, c = linear_sum_assignment(-C)
    m = C[r, c]
    return {"matched_cos": m.tolist(), "mean": float(m.mean()),
            "median": float(np.median(m)), "frac_above": float((m >= COS_THRESH).mean())}


def union_subspace(D1, D2):
    mean, mn = _subspace_cos(_span(D1), _span(D2))
    return {"mean_cos": mean, "min_cos": mn,
            "dim1": int(_span(D1).shape[1]), "dim2": int(_span(D2).shape[1])}


def per_chart_subspace(D1, D2, planes):
    """Group atoms by nearest ground-truth plane; principal-angle cos per chart."""
    def groups(D):
        asg = [int(np.argmax([np.linalg.norm(P.T @ D[k]) ** 2 for P in planes]))
               for k in range(len(D))]
        return [_span(D[[k for k in range(len(D)) if asg[k] == g]])
                for g in range(len(planes)) if any(a == g for a in asg)]
    g1, g2 = groups(D1), groups(D2)
    cos = [_subspace_cos(a, b)[0] for a, b in zip(g1, g2)]
    return {"per_chart_cos": [float(x) for x in cos],
            "mean": float(np.mean(cos)), "min": float(np.min(cos)), "n_charts": len(cos)}


# ============================ data ===========================================
def data_real(width=32, n_rows=600, layers=(24, 25, 26), seed=7):
    A = np.load(ACT_SRC, mmap_mode="r")
    slab = np.asarray(A[:, list(layers), :], dtype=np.float64)
    X = slab.reshape(-1, slab.shape[-1])
    rng = np.random.default_rng(seed)
    if X.shape[0] > n_rows:
        X = X[rng.choice(X.shape[0], size=n_rows, replace=False)]
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Xr = (U[:, :width] * S[:width]).astype(np.float32)
    Xr /= Xr.std() + 1e-8
    prov = {"kind": "real", "model": "allenai/Olmo-3.1-32B-Instruct",
            "readout": "last-token residual stream", "layers": list(layers),
            "projection": f"top-{width} PCA scores (unwhitened)", "N": int(Xr.shape[0]),
            "p": int(Xr.shape[1]), "sha256": hashlib.sha256(Xr.tobytes()).hexdigest()}
    return np.ascontiguousarray(Xr), prov, None


def data_planted(N=3000, p=30, n_circ=3, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    planes = [Q[:, 2 * g:2 * g + 2] for g in range(n_circ)]
    assign = rng.integers(0, n_circ, N)
    phase = rng.uniform(0, 2 * np.pi, N)
    X = np.zeros((N, p), dtype=np.float64)
    for i in range(N):
        X[i] = planes[assign[i]] @ np.array([np.cos(phase[i]), np.sin(phase[i])])
    X += noise * rng.standard_normal((N, p))
    X = X.astype(np.float32)
    prov = {"kind": "planted 1-D circle manifolds (fixed real-shaped synthetic)",
            "N": int(N), "p": int(p), "n_circles": int(n_circ), "noise": noise,
            "sha256": hashlib.sha256(X.tobytes()).hexdigest()}
    return np.ascontiguousarray(X), prov, np.stack(planes, 0)


# ============================ fitters ========================================
def fit_gamfit_linear(X, seed, K=24):
    """Deterministic gamfit sparse_dictionary_fit; seed = row permutation only."""
    perm = np.random.default_rng(1000 + seed).permutation(X.shape[0])
    s = sparse_dictionary_fit(np.ascontiguousarray(X[perm]), K=K, active=1, max_epochs=30)
    return s.decoder.astype(np.float64), {"ev": float(s.explained_variance),
                                          "converged": bool(s.converged)}


def fit_tiling_sae(X, seed, K=18, epochs=50):
    """Auxiliary random-init linear SAE (numpy). Seed randomizes initialization."""
    rng = np.random.default_rng(seed)
    Xf = np.asarray(X, dtype=np.float64)
    N, p = Xf.shape
    D = rng.standard_normal((K, p))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    for _ in range(epochs):
        proj = Xf @ D.T
        idx = np.argmax(np.abs(proj), 1)
        sgn = np.sign(proj[np.arange(N), idx])
        for k in range(K):
            m = idx == k
            if m.sum() < 2:
                j = rng.integers(N)
                D[k] = Xf[j] / (np.linalg.norm(Xf[j]) + 1e-9)
                continue
            c = (Xf[m] * sgn[m][:, None]).mean(0)   # sector centroid -> seed-dependent angle
            nrm = np.linalg.norm(c)
            if nrm > 1e-9:
                D[k] = c / nrm
    ev = 1.0 - ((Xf - (Xf @ D.T)[np.arange(N), np.argmax(np.abs(Xf @ D.T), 1)][:, None]
                 * D[np.argmax(np.abs(Xf @ D.T), 1)]) ** 2).sum() / (Xf ** 2).sum()
    return D, {"ev": float(ev), "converged": True}


MANIFOLD_TIMEOUT_S = 90


def _manifold_worker(npy_path):
    """Child-process worker: one curved manifold-SAE fit, prints one JSON line."""
    import json as _json
    X = np.ascontiguousarray(np.load(npy_path), dtype=np.float64)
    t = time.time()
    try:
        m = sae_manifold_fit(X=X, K=6, d_atom=1, atom_topology="circle",
                             assignment="ibp_map", isometry_weight=1.0,
                             n_iter=25, random_state=0)
        blocks = [np.asarray(a.decoder_coefficients, dtype=np.float64) for a in m.atoms]
        art = canonical_dictionary_artifact(
            blocks, [str(a.basis) for a in m.atoms],
            ["O(2): origin rotation + reflection"] * len(m.atoms), "python-port/v1")
        print(_json.dumps({"status": "CONVERGED", "seconds": round(time.time() - t, 1),
                           "r2": float(m.reconstruction_r2), "K": len(m.atoms),
                           "content_hash": art["content_hash"]}))
    except BaseException as exc:  # noqa: BLE001
        print(_json.dumps({"status": "NON_CONVERGENT", "seconds": round(time.time() - t, 1),
                           "error_type": type(exc).__name__, "error": str(exc)[:300]}))


def attempt_manifold(X):
    """Attempt a curved manifold-SAE fit in a subprocess with a hard SIGKILL
    timeout (the Rust FFI ignores SIGTERM mid-call, so a bounded subprocess is
    the only way to keep the #1784 non-convergence from hanging the run)."""
    import subprocess
    import sys
    import tempfile
    t = time.time()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, np.ascontiguousarray(X, dtype=np.float64))
        path = f.name
    try:
        proc = subprocess.run([sys.executable, str(HERE / "seed_stability.py"),
                               "--manifold-worker", path],
                              capture_output=True, text=True, timeout=MANIFOLD_TIMEOUT_S)
        for ln in reversed(proc.stdout.splitlines()):
            ln = ln.strip()
            if ln.startswith("{"):
                return json.loads(ln)
        return {"status": "NON_CONVERGENT", "seconds": round(time.time() - t, 1),
                "error_type": "NoResult", "error": (proc.stderr or proc.stdout)[-300:]}
    except subprocess.TimeoutExpired:
        return {"status": "NON_CONVERGENT", "seconds": MANIFOLD_TIMEOUT_S,
                "error_type": "Timeout",
                "error": f"exceeded {MANIFOLD_TIMEOUT_S}s wall clock (co-collapse / "
                         "non-PD Arrow-Schur thrash, #1784) -- SIGKILLed"}


# ============================ per-dataset analysis ===========================
def analyze_dictionary(name, fit_fn, X, planes, want_hash=True):
    fits, meta = {}, {}
    for s in SEEDS:
        D, info = fit_fn(X, s)
        # normalize rows for cosine comparisons
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        fits[s] = Dn
        meta[s] = info
        if want_hash:
            meta[s]["content_hash"] = linear_artifact(Dn)["content_hash"]
    pairs = [(a, b) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
    lat, uni, chart = [], [], []
    per_pair = {}
    for a, b in pairs:
        L = latent_match(fits[a], fits[b])
        Umatch = union_subspace(fits[a], fits[b])
        rec = {"latent": L, "union": Umatch}
        if planes is not None:
            Cc = per_chart_subspace(fits[a], fits[b], list(planes))
            rec["per_chart"] = Cc
            chart.append(Cc["mean"])
        lat.append(L["mean"]); uni.append(Umatch["mean_cos"])
        per_pair[f"{a}-{b}"] = rec
    hashes = {int(s): meta[s].get("content_hash") for s in SEEDS} if want_hash else {}
    headline = {"latent_matched_cos_mean": float(np.mean(lat)),
                "latent_frac_above_mean": float(np.mean(
                    [per_pair[f"{a}-{b}"]["latent"]["frac_above"] for a, b in pairs])),
                "union_span_cos_mean": float(np.mean(uni)),
                "hashes_identical": (len(set(hashes.values())) == 1) if want_hash else None}
    if chart:
        headline["subspace_per_chart_cos_mean"] = float(np.mean(chart))
    return {"name": name, "fit_meta": meta, "content_hashes": hashes,
            "per_pair": per_pair, "headline": headline}


def main():
    print(f"[env] gamfit {gamfit.__version__}", flush=True)
    results = {"gamfit_version": gamfit.__version__, "seeds": list(SEEDS),
               "cos_threshold": COS_THRESH, "datasets": {}}

    for dname, loader in (("real_olmo32b", data_real), ("planted_circles", data_planted)):
        X, prov, planes = loader()
        print(f"\n[data:{dname}] {prov}", flush=True)
        block = {"provenance": prov, "dictionaries": {}}

        # (1) gamfit deterministic linear lane
        d = analyze_dictionary("gamfit_sparse_dictionary_fit", fit_gamfit_linear, X, planes)
        block["dictionaries"]["gamfit_linear"] = d
        h = d["headline"]
        print(f"  [gamfit_linear] latent_cos={h['latent_matched_cos_mean']:.3f} "
              f"union={h['union_span_cos_mean']:.3f} hashes_identical={h['hashes_identical']}",
              flush=True)

        # (2) auxiliary random-init tiling SAE (genuine seed randomness)
        d = analyze_dictionary("aux_random_init_tiling_sae", fit_tiling_sae, X, planes)
        block["dictionaries"]["aux_random_tiling"] = d
        h = d["headline"]
        line = (f"  [aux_random_tiling] latent_cos={h['latent_matched_cos_mean']:.3f} "
                f"latent_frac>=thr={h['latent_frac_above_mean']:.2f} "
                f"union={h['union_span_cos_mean']:.3f}")
        if "subspace_per_chart_cos_mean" in h:
            line += f" per_chart={h['subspace_per_chart_cos_mean']:.3f}"
        print(line, flush=True)

        # (3) gamfit curved manifold-SAE -- attempt + record
        man = attempt_manifold(X)
        block["dictionaries"]["gamfit_manifold_sae"] = man
        print(f"  [gamfit_manifold_sae] {man['status']} "
              f"({man.get('error_type','')}) {man['seconds']}s", flush=True)

        results["datasets"][dname] = block

    (OUT / "seed_stability_results.json").write_text(json.dumps(results, indent=2))

    # ---- markdown table ----
    md = ["# Seed stability: latent-level vs subspace-level", "",
          f"gamfit {gamfit.__version__}. Seeds {list(SEEDS)}. Threshold cos>={COS_THRESH}. "
          "Higher = more stable across seeds.", "",
          "Latent = individual atom directions (Hungarian-matched); "
          "Subspace = principal-angle agreement of the spans (union, and per planted chart).", ""]
    for dname, block in results["datasets"].items():
        prov = block["provenance"]
        md += [f"## {dname}", "",
               f"Data: {prov.get('kind')}, N={prov['N']}, p={prov['p']}, "
               f"sha256 `{prov['sha256'][:24]}`.", "",
               "| dictionary | latent cos | latent frac>=thr | union span cos | per-chart cos | hashes identical |",
               "|------------|-----------|------------------|----------------|---------------|------------------|"]
        for key, label in (("gamfit_linear", "gamfit sparse_dictionary_fit (deterministic)"),
                           ("aux_random_tiling", "random-init tiling SAE (aux)")):
            h = block["dictionaries"][key]["headline"]
            pc = h.get("subspace_per_chart_cos_mean")
            md.append(f"| {label} | {h['latent_matched_cos_mean']:.3f} | "
                      f"{h['latent_frac_above_mean']:.2f} | {h['union_span_cos_mean']:.3f} | "
                      f"{pc:.3f} | {h['hashes_identical']} |" if pc is not None else
                      f"| {label} | {h['latent_matched_cos_mean']:.3f} | "
                      f"{h['latent_frac_above_mean']:.2f} | {h['union_span_cos_mean']:.3f} | n/a | "
                      f"{h['hashes_identical']} |")
        man = block["dictionaries"]["gamfit_manifold_sae"]
        md.append(f"| gamfit sae_manifold_fit (circle) | {man['status']} "
                  f"({man.get('error_type','')}) | - | - | - | - |")
        md.append("")
    md += ["## Content-addressed hashes (dictionary_artifact v1 port)", ""]
    for dname, block in results["datasets"].items():
        for key in ("gamfit_linear", "aux_random_tiling"):
            hs = block["dictionaries"][key]["content_hashes"]
            hh = " ".join(f"s{s}={hs[s][:12]}" for s in SEEDS)
            md.append(f"- {dname} / {key}: {hh}")
    (OUT / "seed_stability_table.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md), flush=True)
    print(f"\n[done] wrote {OUT/'seed_stability_results.json'} and table.md", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--manifold-worker":
        _manifold_worker(sys.argv[2])
    else:
        main()
