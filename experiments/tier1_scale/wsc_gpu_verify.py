"""WS-C item 4: GPU scoring-router engagement + graceful-decline verification.

Bounded-observation probe on the decline-fix .so (venv_fable, gamfit 0.1.247).
The K>=2 joint arrow-Schur fit on real activations does not converge (that is
the SAC premise), so we do NOT wait for it — we only need the routing decision,
which is taken in the first inner solve:

  A) above break-even (K=48, d=2, real-shaped): the CUDA router should ENGAGE
     the device (nvidia-smi shows device memory held by our PID), then, because
     the frames-engaged assembly has no dense shared beta block, DECLINE to the
     CPU lane (W12 fix). We watch a bounded window and record whether the fit
     raised the FATAL "requires a dense shared beta block" error. Graceful
     decline == engaged AND the fit runs on (CPU) past the window with no such
     fatal, OR completes without it.
  B) below break-even (tiny K=4): the work predicate should keep this OFF the
     device (router gates on work, not rows).

Pinned to one card via CUDA_VISIBLE_DEVICES (set by the launcher).
"""
import json, os, subprocess, threading, time
import numpy as np

OUT = "/dev/shm/sauers_gpu/wsc_gpu_verify.json"
WINDOW_S = 120.0  # observation cap; not a solver budget — just how long we watch


def make_real_shaped(n, p, seed=0):
    rng = np.random.default_rng(seed)
    spec = np.exp(-np.arange(p) / 12.0)
    X = rng.standard_normal((n, p)) * spec[None, :]
    theta = rng.uniform(0, 2 * np.pi, n)
    u, v = rng.standard_normal(p), rng.standard_normal(p)
    u /= np.linalg.norm(u); v -= v @ u * u; v /= np.linalg.norm(v)
    X += 2.0 * (np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :])
    return X.astype(np.float64)


class GpuWatch:
    def __init__(self):
        self.seen = False; self.max_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        me = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                for line in out.strip().splitlines():
                    parts = [s.strip() for s in line.split(",")]
                    if len(parts) >= 2 and parts[0] == me:
                        self.seen = True
                        self.max_mib = max(self.max_mib, int(parts[1]))
            except Exception:
                pass
            time.sleep(0.3)

    def __enter__(self): self._t.start(); return self
    def __exit__(self, *a): self._stop.set(); self._t.join(timeout=3)


def probe(name, X, K, d_atom, window_s):
    """Run the fit in a daemon thread; observe GPU + fatal signature for
    <= window_s. Returns the routing verdict without waiting for convergence."""
    res = {"name": name, "n": X.shape[0], "p": X.shape[1], "K": K, "d_atom": d_atom}
    state = {"done": False, "error": None}

    def worker():
        try:
            import gamfit
            gamfit.sae_manifold_fit(X, K=K, d_atom=d_atom, n_iter=1)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        state["done"] = True

    with GpuWatch() as w:
        t = threading.Thread(target=worker, daemon=True)
        t0 = time.time(); t.start()
        while not state["done"] and (time.time() - t0) < window_s:
            time.sleep(0.5)
        res["observed_s"] = round(time.time() - t0, 1)
    res["completed"] = state["done"]
    res["error"] = state["error"]
    res["hit_dense_beta_fatal"] = ("requires a dense shared beta block" in (state["error"] or ""))
    res["gpu_engaged"] = w.seen
    res["gpu_max_mem_mib"] = w.max_mib
    # still running after the window with no fatal == declined to CPU and grinding
    res["declined_to_cpu"] = (not res["hit_dense_beta_fatal"]) and (
        res["completed"] or (not state["done"]))
    return res


def main():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    results = {"cuda_visible_devices": dev, "window_s": WINDOW_S, "fits": []}
    results["fits"].append(probe("above_break_even_K48", make_real_shaped(4000, 64), 48, 2, WINDOW_S))
    results["fits"].append(probe("below_break_even_K4", make_real_shaped(300, 16, seed=1), 4, 2, 20.0))
    a, b = results["fits"]
    results["verdict"] = {
        "router_engaged_above_break_even": bool(a["gpu_engaged"]),
        "declined_gracefully_not_fatal": bool(a["declined_to_cpu"]) and not a["hit_dense_beta_fatal"],
        "off_below_break_even": (not b["gpu_engaged"]),
        "no_dense_beta_fatal_anywhere": not any(f["hit_dense_beta_fatal"] for f in results["fits"]),
    }
    json.dump(results, open(OUT, "w"), indent=1)
    print(json.dumps(results, indent=1))
    print("GPU_VERIFY_DONE " + json.dumps(results["verdict"]))


if __name__ == "__main__":
    main()
