"""Scaling benchmark: sparse vs dense curve-atom decode on MPS.

Sweep F ∈ {2^12, 2^14, 2^16, 2^18} with K_active=64, B=256, D=7168, P=7.
For each F, measure:
  - peak resident memory (process RSS via psutil + torch.mps.current_allocated_memory)
  - wall time per "training step" (forward + .sum().backward()) — median of 5

The dense path materializes a (B, F·P) weight tensor and a (F, P, D)
decoder, so peak is dominated by the (F, P, D) parameter and a
(B, F·P) intermediate. The sparse path keeps only the K_active fraction
of those.

Results are written to runs/scaling_bench.{png,json,md}.  F values that
OOM are recorded with a "skipped (OOM)" entry and the run continues.
"""
from __future__ import annotations
import gc
import json
import os
import time
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
OUT = ROOT / "runs"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[bench] device={DEVICE}", flush=True)


def _mps_alloc_mb() -> float:
    if DEVICE.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except Exception:
            return float("nan")
    return float("nan")


def _make_sparse_gate(B: int, F: int, K_active: int, device) -> torch.Tensor:
    """Build a (B, F) gate that is exactly K_active-sparse per row."""
    g = torch.zeros(B, F, device=device, dtype=torch.float32)
    # vectorized topk-of-random
    scores = torch.rand(B, F, device=device)
    _, idx = scores.topk(K_active, dim=1)
    g.scatter_(1, idx, torch.rand(B, K_active, device=device) + 0.1)
    return g


def _time_step(fn, *args, n_warmup=2, n_iters=5) -> tuple[float, float]:
    """Return (median_step_seconds, peak_alloc_mb)."""
    for _ in range(n_warmup):
        out = fn(*args)
        out.sum().backward()
        del out
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    gc.collect()
    peak = 0.0
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.sum().backward()
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        peak = max(peak, _mps_alloc_mb())
        del out
    return float(np.median(times)), peak


def run_one(F: int, B: int, P: int, D: int, K_active: int, mode: str) -> dict | None:
    """One benchmark cell. Returns None on OOM."""
    try:
        gate = _make_sparse_gate(B, F, K_active, DEVICE)
        atoms = torch.randn(B, F, P, device=DEVICE, dtype=torch.float32, requires_grad=True)
        basis = (
            torch.randn(F, P, D, device=DEVICE, dtype=torch.float32) * (1.0 / (P ** 0.5))
        ).requires_grad_(True)
        fn = sparse_curve_decode if mode == "sparse" else dense_curve_decode
        t, peak = _time_step(fn, gate, atoms, basis)
        # Param + biggest input tensor estimate
        param_mb = (F * P * D * 4 + B * F * P * 4 + B * F * 4) / 1e6
        return {
            "F": F, "mode": mode, "median_step_s": t, "peak_alloc_mb": peak,
            "param_input_mb_theoretical": param_mb,
        }
    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        print(f"  [{mode:>6s} F={F:>6d}] OOM/Error: {msg[:120]}", flush=True)
        return {"F": F, "mode": mode, "skipped": True, "error": msg[:200]}
    finally:
        for var in ("gate", "atoms", "basis"):
            if var in dir():
                pass
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()


def main():
    B = 256
    P = 7
    D = 7168
    K_active = 64
    F_LIST = [2**12, 2**14, 2**16, 2**18]

    rows: list[dict] = []
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

    # Save JSON
    with open(OUT / "scaling_bench.json", "w") as f:
        json.dump({"rows": rows, "B": B, "P": P, "D": D, "K_active": K_active}, f, indent=2)

    # Markdown summary
    md = ["# Sparse vs dense curve-decode scaling on MPS\n",
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

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mode, color in [("dense", "C3"), ("sparse", "C0")]:
        xs = [r["F"] for r in rows if r["mode"] == mode and not r.get("skipped")]
        ts = [r["median_step_s"]*1000 for r in rows if r["mode"] == mode and not r.get("skipped")]
        ms = [r["peak_alloc_mb"] for r in rows if r["mode"] == mode and not r.get("skipped")]
        axes[0].plot(xs, ts, "o-", color=color, label=mode, lw=2, ms=8)
        axes[1].plot(xs, ms, "o-", color=color, label=mode, lw=2, ms=8)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("F (atoms)"); axes[0].set_ylabel("median step time (ms)")
    axes[0].set_title(f"Step time vs F  (K={K_active}, B={B}, D={D})")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("F (atoms)"); axes[1].set_ylabel("peak MPS alloc (MB)")
    axes[1].set_title(f"Peak memory vs F")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "scaling_bench.png", dpi=120)
    print(f"\n[bench] saved {OUT/'scaling_bench.png'}", flush=True)


if __name__ == "__main__":
    main()
