"""CUDA-adapted scaling benchmark. See scaling_bench.py for design."""
from __future__ import annotations
import gc, json, os, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_sae.kernels.sparse_decode import (
    dense_curve_decode,
    sparse_curve_decode,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "scaling_bench"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[bench] device={DEVICE}", flush=True)


def _peak_mb() -> float:
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def _reset_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _make_sparse_gate(B, F, K_active, device):
    g = torch.zeros(B, F, device=device, dtype=torch.float32)
    scores = torch.rand(B, F, device=device)
    _, idx = scores.topk(K_active, dim=1)
    g.scatter_(1, idx, torch.rand(B, K_active, device=device) + 0.1)
    return g


def _time_step(fn, *args, n_warmup=2, n_iters=5):
    for _ in range(n_warmup):
        out = fn(*args)
        out.sum().backward()
        del out
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    gc.collect()
    _reset_peak()
    times = []
    peak = 0.0
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.sum().backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        peak = max(peak, _peak_mb())
        del out
    return float(np.median(times)), peak


def run_one(F, B, P, D, K_active, mode):
    try:
        _reset_peak()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gate = _make_sparse_gate(B, F, K_active, DEVICE)
        atoms = torch.randn(B, F, P, device=DEVICE, dtype=torch.float32, requires_grad=True)
        basis = (torch.randn(F, P, D, device=DEVICE, dtype=torch.float32) * (1.0 / (P ** 0.5))).requires_grad_(True)
        fn = sparse_curve_decode if mode == "sparse" else dense_curve_decode
        t, peak = _time_step(fn, gate, atoms, basis)
        param_mb = (F * P * D * 4 + B * F * P * 4 + B * F * 4) / 1e6
        return {"F": F, "mode": mode, "median_step_s": t, "peak_alloc_mb": peak,
                "param_input_mb_theoretical": param_mb}
    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        print(f"  [{mode:>6s} F={F:>6d}] OOM/Error: {msg[:120]}", flush=True)
        return {"F": F, "mode": mode, "skipped": True, "error": msg[:200]}
    finally:
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def main():
    B = 256; P = 7; D = 7168; K_active = 64
    F_LIST = [2**12, 2**14, 2**16, 2**17, 2**18]

    rows = []
    for F in F_LIST:
        print(f"\n[bench] F={F} (K_active={K_active}, B={B}, D={D}, P={P})", flush=True)
        for mode in ("dense", "sparse"):
            r = run_one(F, B, P, D, K_active, mode)
            if r is not None:
                if r.get("skipped"):
                    print(f"  [{mode:>6s}] SKIPPED")
                else:
                    print(f"  [{mode:>6s}] step={r['median_step_s']*1000:7.1f} ms  "
                          f"peak={r['peak_alloc_mb']:8.1f} MB", flush=True)
                rows.append(r)

    with open(OUT / "scaling_bench.json", "w") as f:
        json.dump({"rows": rows, "B": B, "P": P, "D": D, "K_active": K_active}, f, indent=2)

    md = ["# Sparse vs dense curve-decode scaling on V100\n",
          f"B={B}, K_active={K_active}, D={D}, P={P}\n",
          "| F | mode | step (ms) | peak (MB) | note |",
          "|---:|:---:|---:|---:|:---|"]
    for r in rows:
        if r.get("skipped"):
            md.append(f"| {r['F']} | {r['mode']} | — | — | OOM |")
        else:
            md.append(f"| {r['F']} | {r['mode']} | {r['median_step_s']*1000:.1f} | "
                      f"{r['peak_alloc_mb']:.1f} | |")
    (OUT / "scaling_bench.md").write_text("\n".join(md) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mode, color in [("dense", "C3"), ("sparse", "C0")]:
        xs = [r["F"] for r in rows if r["mode"] == mode and not r.get("skipped")]
        ts = [r["median_step_s"]*1000 for r in rows if r["mode"] == mode and not r.get("skipped")]
        ms = [r["peak_alloc_mb"] for r in rows if r["mode"] == mode and not r.get("skipped")]
        if xs:
            axes[0].plot(xs, ts, "o-", color=color, label=mode, lw=2, ms=8)
            axes[1].plot(xs, ms, "o-", color=color, label=mode, lw=2, ms=8)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("F (atoms)"); axes[0].set_ylabel("median step time (ms)")
    axes[0].set_title(f"Step time vs F  (K={K_active}, B={B}, D={D}) — V100")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("F (atoms)"); axes[1].set_ylabel("peak CUDA alloc (MB)")
    axes[1].set_title(f"Peak memory vs F")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "scaling_bench.png", dpi=120)
    print(f"\n[bench] saved {OUT/'scaling_bench.png'}", flush=True)


if __name__ == "__main__":
    main()
