"""Manifold-beats-linear on activations — protected, gamfit-native run.

Fits the gamfit-native manifold SAE across a K sweep and reports whether the
curved dictionary reaches the linear-baseline reconstruction ceiling at fewer
atoms (via :mod:`manifold_sae.eval.frontier`). Designed to run as a well-behaved
guest on a SHARED cluster node — it must never disturb a co-tenant.

Protections (all enforced here, before any heavy import):
  * Thread caps (OMP/MKL/OPENBLAS/NUMEXPR/RAYON) so we never hog all cores.
  * Scratch + HF cache under ``$MBL_SCRATCH`` (default /dev/shm — RAM tmpfs, huge)
    so we never write to the small shared root disk; cleaned up on exit.
  * GPU memory hard-capped (``set_per_process_memory_fraction``) so we cannot OOM
    a co-resident model. (The manifold REML fit is CPU/Rust-bound anyway.)
  * Pre-flight disk guard: abort if the root filesystem is dangerously full.
  * Bounded token count + K sweep; writes only to ``$MANIFOLD_SAE_OUTPUT_DIR``.

Config (env; all optional):
  MBL_ACTIVATIONS   path to a (N, D) float .npy of activations. If unset, uses a
                    synthetic planted mixture-of-circles (pipeline validation).
  MBL_N_TOKENS      cap on rows used (default 200000).
  MBL_K_VALUES      comma list of dictionary sizes (default "4,8,16,32").
  MBL_D_ATOM        per-atom intrinsic dim (default 1).
  MBL_TOPOLOGY      seed atom topology (default "circle").
  MBL_TEST_FRAC     held-out fraction (default 0.2).
  MBL_DROP_TOP_PCS  PCs to drop after whitening (default 0).
  MBL_GPU_MEM_FRACTION  cap of total GPU mem our process may use (default 0.15).
  MBL_MAX_THREADS   thread cap (default 16).
  MBL_SCRATCH       scratch root (default /dev/shm/sauers_msae).
  MANIFOLD_SAE_OUTPUT_DIR  where to write results.json / report.md.
"""

from __future__ import annotations

import json
import os
import shutil
import sys


# --- protections that MUST be set before numpy/torch/gamfit import ------------
_MAX_THREADS = os.environ.get("MBL_MAX_THREADS", "16")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _MAX_THREADS)

_SCRATCH = os.environ.get("MBL_SCRATCH", "/dev/shm/sauers_msae")
os.makedirs(_SCRATCH, exist_ok=True)
# Keep HF + tmp off the shared root disk.
os.environ.setdefault("HF_HOME", os.path.join(_SCRATCH, "hf"))
os.environ.setdefault("TMPDIR", os.path.join(_SCRATCH, "tmp"))
os.makedirs(os.environ["TMPDIR"], exist_ok=True)


def _free_gib(path: str) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 2**30
    except OSError:
        return float("inf")


def _preflight_disk_guard(min_root_gib: float = 5.0) -> None:
    root_free = _free_gib("/")
    print(f"[protect] root(/) free={root_free:.1f} GiB, scratch({_SCRATCH}) free={_free_gib(_SCRATCH):.1f} GiB")
    if root_free < min_root_gib:
        raise SystemExit(
            f"[protect] ABORT: root filesystem only {root_free:.1f} GiB free "
            f"(< {min_root_gib} GiB). Refusing to run so we cannot fill a shared disk."
        )


def _cap_gpu_memory() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu (no CUDA)"
    frac = float(os.environ.get("MBL_GPU_MEM_FRACTION", "0.15"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"{name} (capped to {frac:.0%} of {total:.0f} GiB ≈ {frac*total:.0f} GiB)"


def _load_activations() -> "tuple[Any, str]":  # noqa: F821
    import numpy as np

    n_tokens = int(os.environ.get("MBL_N_TOKENS", "200000"))
    path = os.environ.get("MBL_ACTIVATIONS")
    if path and os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr[:n_tokens], dtype=np.float64)
        return arr, f"{path} [{arr.shape}]"
    # Synthetic planted mixture-of-circles: the canonical manifold-beats-linear case.
    rng = np.random.default_rng(0)
    d, n_circles = 64, 6
    n = min(n_tokens, 6000)
    per = n // n_circles
    parts = []
    for _c in range(n_circles):
        t = rng.uniform(0.0, 2.0 * np.pi, size=per)
        basis = rng.standard_normal((2, d))
        parts.append(np.cos(t)[:, None] * basis[0] + np.sin(t)[:, None] * basis[1])
    x = np.concatenate(parts, axis=0) + 0.05 * rng.standard_normal((per * n_circles, d))
    return x.astype(np.float64), f"SYNTHETIC mixture-of-{n_circles}-circles [{x.shape}]"


def _preprocess(x, test_frac: float, drop_top_pcs: int):
    """Train-only PCA whiten (the recommended real-activation recipe)."""
    import numpy as np

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(x))
    x = x[perm]
    n_test = max(1, int(len(x) * test_frac))
    test, train = x[:n_test], x[n_test:]
    mu = train.mean(0)
    tr_c, te_c = train - mu, test - mu
    # economy PCA on TRAIN
    _, _, Vt = np.linalg.svd(tr_c, full_matrices=False)
    if drop_top_pcs > 0:
        Vt = Vt[drop_top_pcs:]
    tr_p, te_p = tr_c @ Vt.T, te_c @ Vt.T
    sd = tr_p.std(0) + 1e-8
    return tr_p / sd, te_p / sd


def main() -> int:
    _preflight_disk_guard()

    k_values = [int(k) for k in os.environ.get("MBL_K_VALUES", "4,8,16,32").split(",") if k.strip()]
    d_atom = int(os.environ.get("MBL_D_ATOM", "1"))
    topology = os.environ.get("MBL_TOPOLOGY", "circle")
    test_frac = float(os.environ.get("MBL_TEST_FRAC", "0.2"))
    drop_top_pcs = int(os.environ.get("MBL_DROP_TOP_PCS", "0"))
    out_dir = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", _SCRATCH)
    os.makedirs(out_dir, exist_ok=True)

    gpu = _cap_gpu_memory()
    print(f"[protect] threads<= {_MAX_THREADS} | gpu: {gpu}")

    x, src = _load_activations()
    print(f"[data] {src}")
    train, test = _preprocess(x, test_frac, drop_top_pcs)
    print(f"[data] train={train.shape} test={test.shape} | K sweep={k_values} d_atom={d_atom} topo={topology}")

    from manifold_sae.eval.frontier import manifold_vs_linear_frontier, format_frontier_markdown

    import gamfit

    base_iter = int(os.environ.get("MBL_N_ITER", "50"))
    # Resilient: gamfit's REML solve can fail to converge on hard/thin data.
    # Retry once with more iterations; if it still fails, record the miss and
    # exit cleanly (0) rather than crash — a cluster job must always tidy up.
    res = None
    err = None
    for n_iter in (base_iter, max(base_iter * 2, 80)):
        try:
            res = manifold_vs_linear_frontier(
                train, test, k_values, d_atom=d_atom, atom_topology=topology,
                quiet=True, sae_fit_kwargs={"n_iter": n_iter},
            )
            break
        except (getattr(gamfit, "GamError", Exception),) as exc:
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"[fit] n_iter={n_iter} did not converge: {err}")

    if res is None:
        payload = {"source": src, "k_values": k_values, "d_atom": d_atom,
                   "topology": topology, "beats_linear": None,
                   "error": "manifold fit failed to converge", "detail": err}
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[out] non-convergence recorded to {out_dir}/results.json")
        return 0

    report = format_frontier_markdown(res)
    print("\n" + report + "\n")

    payload = {
        "source": src,
        "k_values": k_values,
        "d_atom": d_atom,
        "topology": topology,
        "manifold_ev_by_k": res.manifold_ev_by_k,
        "linear_ev_by_k": res.linear_ev_by_k,
        "verdict": res.verdict,
        "beats_linear": res.beats_linear,
        "efficiency_ratio": res.efficiency_ratio,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(report + "\n")
    print(f"[out] wrote results.json + report.md to {out_dir}")
    print(f"[verdict] beats_linear={res.beats_linear} efficiency_ratio={res.efficiency_ratio}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        # Clean our RAM-backed scratch so we never leave tmpfs pinned.
        keep_out = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "")
        if _SCRATCH not in keep_out:
            shutil.rmtree(_SCRATCH, ignore_errors=True)
    sys.exit(rc)
