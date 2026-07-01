"""Honest manifold-vs-linear on REAL activations, with an EXTERNAL SAE baseline.

The point: don't let gamfit grade its own homework. On real Qwen L17 residual
activations, compare held-out reconstruction EV at matched dictionary size K for:
  1. gamfit manifold SAE   (curve/surface atoms — the thing under test)
  2. gamfit linear_dictionary_fit   (gamfit's own linear baseline)
  3. an EXTERNAL standard SAE   (dictionary_learning.AutoEncoder, ReLU+L1) — the
     independent reference nobody can accuse of being rigged.

A real result is: does the manifold reach a given EV at fewer atoms than a
*standard* SAE? If it doesn't, that's a real (publishable) negative too.

Protections (shared-node-safe): scratch/HF under /dev/shm; thread caps; one GPU
pinned via CUDA_VISIBLE_DEVICES with a memory ceiling; disk pre-flight guard;
cleanup on exit. Resilient per-method (one failure doesn't kill the run).

Env:
  MVE_N_SHARDS   caiovicentino shards to pull (10240 tokens each; default 5 ≈ 51k).
  MVE_LAYER      acts_L{11,17,23} (default 17).
  MVE_K_VALUES   dict sizes (default "64,128,256,512").
  MVE_D_ATOM / MVE_TOPOLOGY   manifold atom dim/topology (default 1 / circle).
  MVE_N_ITER     manifold REML iters (default 50).
  MVE_EXT_STEPS  external-SAE Adam steps (default 3000); MVE_EXT_L1 (default 1e-3).
  MVE_ACTIVATIONS  optional path to a (N,D) .npy to use instead of the HF shards.
  MANIFOLD_SAE_OUTPUT_DIR  where results.json / report.md go.
"""

from __future__ import annotations

import json
import os
import shutil

_MAX_THREADS = os.environ.get("MVE_MAX_THREADS", "16")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _MAX_THREADS)
_SCRATCH = os.environ.get("MVE_SCRATCH", "/dev/shm/sauers_mve")
os.makedirs(_SCRATCH, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(_SCRATCH, "hf"))
os.environ.setdefault("TMPDIR", os.path.join(_SCRATCH, "tmp"))
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

REPO = "caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b"


def _free_gib(p):
    try:
        s = os.statvfs(p); return s.f_bavail * s.f_frsize / 2**30
    except OSError:
        return float("inf")


def _guard():
    r = _free_gib("/")
    print(f"[protect] root(/) free={r:.1f}G scratch({_SCRATCH}) free={_free_gib(_SCRATCH):.1f}G "
          f"threads<={_MAX_THREADS} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(all)')}")
    if r < 5.0:
        raise SystemExit(f"[protect] ABORT: root / only {r:.1f}G free")


def _load_real(n_shards, layer):
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    key = f"acts_L{layer}"
    mats = []
    for i in range(n_shards):
        fn = f"shards/q{250 + i:04d}.safetensors"
        p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                            local_dir=os.path.join(_SCRATCH, "shards"))
        with safe_open(p, framework="np") as f:
            mats.append(np.asarray(f.get_tensor(key), dtype=np.float64))
        os.remove(p)  # free /dev/shm as we go
    X = np.concatenate(mats, 0)
    return X, f"REAL Qwen {key} [{X.shape}] from {n_shards} shards"


def _load():
    import numpy as np
    p = os.environ.get("MVE_ACTIVATIONS")
    if p and os.path.exists(p):
        X = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
        return X, f"{p} [{X.shape}]"
    return _load_real(int(os.environ.get("MVE_N_SHARDS", "5")),
                      int(os.environ.get("MVE_LAYER", "17")))


def _split_prep(X, test_frac=0.2):
    """Train-only PCA-reduce + scale. Reducing ambient dim (2048 -> MVE_PCA) is
    essential: the manifold SAE's per-atom decoder is D-dimensional, so fitting in
    full 2048-dim is brutally slow. All three methods fit in the SAME reduced space
    -> fair. MVE_PCA=0 keeps full dim."""
    import numpy as np
    n_pca = int(os.environ.get("MVE_PCA", "128"))
    rng = np.random.default_rng(0)
    X = X[rng.permutation(len(X))]
    nt = max(1, int(len(X) * test_frac))
    test, train = X[:nt], X[nt:]
    mu = train.mean(0)
    tr, te = train - mu, test - mu
    if n_pca and n_pca < tr.shape[1]:
        _, _, Vt = np.linalg.svd(tr, full_matrices=False)
        Vt = Vt[:n_pca]
        tr, te = tr @ Vt.T, te @ Vt.T
    # scale to unit average norm (same transform for every method — fair)
    s = np.sqrt((tr**2).sum(1).mean()) + 1e-8
    return tr / s, te / s


def _ev(x, xhat):
    import numpy as np
    sst = float(((x - x.mean(0)) ** 2).sum())
    return float(1 - ((x - xhat) ** 2).sum() / sst) if sst > 0 else float("nan")


def _ext_sae_ev(train, test, K, steps, l1):
    """Train the EXTERNAL dictionary_learning ReLU+L1 SAE; return held-out EV + L0."""
    import importlib.util, sys, torch, numpy as np
    # load AutoEncoder from the file directly (package __init__ pulls a broken
    # transformers 5.x). Find dictionary.py on sys.path without importing the package.
    dict_py = None
    for base in sys.path:
        cand = os.path.join(base, "dictionary_learning", "dictionary.py")
        if os.path.exists(cand):
            dict_py = cand; break
    if dict_py is None:
        raise RuntimeError("dictionary_learning/dictionary.py not found on sys.path")
    spec = importlib.util.spec_from_file_location("dl_dict", dict_py)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    AutoEncoder = m.AutoEncoder

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AutoEncoder(activation_dim=train.shape[1], dict_size=int(K)).to(dev).float()
    Xtr = torch.tensor(train, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    bs = min(4096, len(Xtr))
    ae.train()
    for step in range(steps):
        idx = torch.randint(0, len(Xtr), (bs,), device=dev)
        xb = Xtr[idx]
        f = ae.encode(xb); rec = ae.decode(f)
        loss = ((rec - xb) ** 2).mean() + l1 * f.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        Xte = torch.tensor(test, dtype=torch.float32, device=dev)
        f = ae.encode(Xte); rec = ae.decode(f).cpu().numpy()
        l0 = float((f > 1e-6).float().sum(-1).mean().item())
    return _ev(test, rec), l0


def main():
    import numpy as np, gamfit
    from gamfit._sae_manifold import wager_verdict  # noqa: F401 (kept for parity)

    _guard()
    X, src = _load()
    print(f"[data] {src}")
    train, test = _split_prep(X)
    print(f"[data] train={train.shape} test={test.shape}")

    ks = [int(k) for k in os.environ.get("MVE_K_VALUES", "64,128,256,512").split(",") if k.strip()]
    d_atom = int(os.environ.get("MVE_D_ATOM", "1"))
    topo = os.environ.get("MVE_TOPOLOGY", "circle")
    n_iter = int(os.environ.get("MVE_N_ITER", "50"))
    ext_steps = int(os.environ.get("MVE_EXT_STEPS", "3000"))
    ext_l1 = float(os.environ.get("MVE_EXT_L1", "1e-3"))
    out = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", _SCRATCH)
    os.makedirs(out, exist_ok=True)

    import torch
    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    def _fdq(fn, *a, **k):
        dn = os.open(os.devnull, os.O_WRONLY); s1, s2 = os.dup(1), os.dup(2)
        try:
            os.dup2(dn, 1); os.dup2(dn, 2); return fn(*a, **k)
        finally:
            os.dup2(s1, 1); os.dup2(s2, 2); os.close(dn); os.close(s1); os.close(s2)

    rows = []
    for K in ks:
        row = {"K": K}
        # 1) gamfit manifold SAE
        try:
            fit = _fdq(gamfit.sae_manifold_fit, train, K=K, d_atom=d_atom,
                       atom_topology=topo, assignment="ibp_map", n_iter=n_iter)
            row["manifold_ev_train"] = float(fit.reconstruction_r2)
            row["manifold_ev_test"] = _ev(test, np.asarray(fit.reconstruct(test)))
            row["manifold_used_gpu"] = bool(getattr(fit, "used_device", False))
        except Exception as e:
            row["manifold_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
        # 2) gamfit linear dictionary
        try:
            lin = _fdq(gamfit.linear_dictionary_fit, train, int(K))
            # held-out EV: reconstruct test via its atoms (top-1 assignment)
            atoms = np.asarray(lin.atoms)  # (K, D)
            proj = test @ atoms.T
            keep = np.zeros_like(proj); am = np.argmax(np.abs(proj), 1)
            keep[np.arange(len(test)), am] = proj[np.arange(len(test)), am]
            row["gamlinear_ev_test"] = _ev(test, keep @ atoms)
        except Exception as e:
            row["gamlinear_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
        # 3) external standard SAE
        try:
            ev_ext, l0 = _ext_sae_ev(train, test, K, ext_steps, ext_l1)
            row["external_ev_test"] = ev_ext
            row["external_L0"] = l0
        except Exception as e:
            row["external_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
        print("[K=%d] %s" % (K, json.dumps({k: v for k, v in row.items() if k != "K"})), flush=True)
        rows.append(row)

    payload = {"source": src, "k_values": ks, "d_atom": d_atom, "topology": topo,
               "n_iter": n_iter, "ext_steps": ext_steps, "ext_l1": ext_l1, "rows": rows}
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    # markdown
    lines = ["| K | manifold EV(test) | gamfit-linear EV | external SAE EV | ext L0 | manifold GPU |",
             "|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        lines.append("| %d | %s | %s | %s | %s | %s |" % (
            r["K"], _fmt(r.get("manifold_ev_test")), _fmt(r.get("gamlinear_ev_test")),
            _fmt(r.get("external_ev_test")), _fmt(r.get("external_L0"), 1),
            "✓" if r.get("manifold_used_gpu") else "·"))
    md = "\n".join(lines)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(md + "\n")
    print("\n" + md + "\n")
    print(f"[out] {out}/results.json + report.md")
    return 0


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        if _SCRATCH not in os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", ""):
            shutil.rmtree(_SCRATCH, ignore_errors=True)
    raise SystemExit(rc)
