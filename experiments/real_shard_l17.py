"""Real-shard ManifoldSAE training on Qwen3.6-35B-A3B L17 activations (MSI lane).

Trains the gamfit-native ``gamfit.torch.ManifoldSAE`` on the
``caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b`` per-token residual-stream cache
(layer 17, [tokens, 2048] fp16, ~1.4M tokens over 200 question-shards), against
a matched-L0 linear TopK SAE baseline trained on the same split with the same
step budget. Fills the SAC scoreboard "live real-shard EV" cells (parity #1).

Split hygiene: shards are one SuperGPQA question x 5 rollouts each, so the
held-out split is BY SHARD (question-level) — rollouts of one question never
straddle the split.

Resilience (harvest-gotchas): results.json is rewritten after every eval,
checkpoints saved every ``--ckpt-every`` steps, and the script resumes from the
newest checkpoint if present. sae.fit (closed-form REML refresh) is wrapped so
a solver refusal degrades to pure-backprop instead of killing the run.

Device: gamfit ManifoldSAE numerics are Rust/CPU-backed; we attempt the
requested device and fall back to CPU on a device-mismatch error. The linear
TopK baseline uses CUDA when available.

Usage (MSI):
  python real_shard_l17.py --data-dir $RUN/data --out $RUN/out --pilot   # smoke
  python real_shard_l17.py --data-dir $RUN/data --out $RUN/out \
      --n-atoms 1024 --target-k 8 --steps 20000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def discover_shards(data_dir: Path) -> list[Path]:
    shards = sorted((data_dir / "shards").glob("q*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no shards under {data_dir}/shards")
    return shards


def split_shards(shards: list[Path], heldout_frac: float, seed: int):
    rng = random.Random(seed)
    order = shards[:]
    rng.shuffle(order)
    n_held = max(1, int(round(len(order) * heldout_frac)))
    return sorted(order[n_held:]), sorted(order[:n_held])


def load_layer(shards: list[Path], layer: int, max_tokens: int | None = None) -> torch.Tensor:
    """Concatenate acts_L{layer} across shards. Kept fp16 in RAM; cast per batch."""
    key = f"acts_L{layer}"
    parts: list[torch.Tensor] = []
    total = 0
    for p in shards:
        with safe_open(str(p), framework="pt") as f:
            t = f.get_tensor(key)  # [tokens, D] fp16
        parts.append(t)
        total += t.shape[0]
        if max_tokens is not None and total >= max_tokens:
            break
    x = torch.cat(parts, 0)
    if max_tokens is not None:
        x = x[:max_tokens]
    return x


class Standardizer:
    """Train-mean centering + isotropic scale so mean ||x|| ~= sqrt(D)."""

    def __init__(self, x_train_fp16: torch.Tensor):
        xf = x_train_fp16.float()
        self.mu = xf.mean(0)
        c = xf - self.mu
        self.scale = (c.norm(dim=1).mean() / math.sqrt(xf.shape[1])).item()
        if not (self.scale > 0 and math.isfinite(self.scale)):
            raise ValueError(f"degenerate scale {self.scale}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu.to(x.device)) / self.scale

    def state(self) -> dict:
        return {"mu": self.mu, "scale": self.scale}


def batches(x_fp16: torch.Tensor, bs: int, std: Standardizer, device: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = x_fp16.shape[0]
    while True:
        idx = torch.randint(0, n, (bs,), generator=g)
        yield std(x_fp16[idx]).to(device)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    resid = (x - x_hat).pow(2).sum()
    tot = (x - x.mean(0)).pow(2).sum()
    return float(1.0 - (resid / tot).item())


@torch.no_grad()
def eval_manifold(sae, x_fp16: torch.Tensor, std: Standardizer, device: str,
                  bs: int = 8192, act_thresh: float = 1e-3) -> dict:
    """Held-out EV + sparsity stats. ``out.assignments`` (N, F) is the signed
    effective per-atom coefficient — the sparse code — so L0/firing/dead are
    measured on |assignments| > act_thresh (out.gate is a logit, NOT firing)."""
    sae.eval()
    n = x_fp16.shape[0]
    sse, stot, l0_sum, nb = 0.0, 0.0, 0.0, 0
    fire = None
    mu_ref = None
    for i in range(0, n, bs):
        xb = std(x_fp16[i:i + bs]).to(device)
        out = sae(xb)
        if mu_ref is None:
            mu_ref = xb.mean(0)
        sse += (xb - out.x_hat).pow(2).sum().item()
        stot += (xb - mu_ref).pow(2).sum().item()
        act = (out.assignments.abs() > act_thresh).float()
        l0_sum += act.sum().item()
        fire = act.sum(0) if fire is None else fire + act.sum(0)
        nb += xb.shape[0]
    sae.train()
    return {
        "ev": 1.0 - sse / stot,
        "l0": l0_sum / nb,
        "dead_atoms": int((fire == 0).sum().item()),
        "n_tokens": nb,
    }


# --------------------------------------------------------------------------
# Linear TopK baseline (matched L0)
# --------------------------------------------------------------------------

class TopKSAE(torch.nn.Module):
    def __init__(self, d: int, width: int, k: int):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, width)
        self.dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(width, d)))
        self.b_d = torch.nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.dec.copy_(torch.nn.functional.normalize(self.dec, dim=1))
            self.enc.weight.copy_(self.dec)

    def forward(self, x):
        z = torch.relu(self.enc(x - self.b_d))
        top = torch.topk(z, self.k, dim=-1)
        zs = torch.zeros_like(z).scatter_(-1, top.indices, top.values)
        dec = torch.nn.functional.normalize(self.dec, dim=1)
        return zs @ dec + self.b_d, zs


def train_topk_baseline(x_fp16, x_held_fp16, std, width, k, steps, bs, lr, seed, log):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = x_fp16.shape[1]
    model = TopKSAE(d, width, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it = batches(x_fp16, bs, std, device, seed + 1)
    t0 = time.time()
    for step in range(steps):
        xb = next(it)
        x_hat, _ = model(xb)
        loss = (x_hat - xb).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0:
            log(f"[topk {step:6d}] mse={loss.item():.4e}")
    model.eval()
    n = x_held_fp16.shape[0]
    sse, stot, l0_sum = 0.0, 0.0, 0.0
    mu_ref = None
    with torch.no_grad():
        for i in range(0, n, 8192):
            xb = std(x_held_fp16[i:i + 8192]).to(device)
            x_hat, zs = model(xb)
            if mu_ref is None:
                mu_ref = xb.mean(0)
            sse += (xb - x_hat).pow(2).sum().item()
            stot += (xb - mu_ref).pow(2).sum().item()
            l0_sum += (zs > 0).float().sum().item()
    return {
        "ev": 1.0 - sse / stot,
        "l0": l0_sum / n,
        "width": width,
        "k": k,
        "train_seconds": time.time() - t0,
        "device": device,
    }


# --------------------------------------------------------------------------
# Closed-form fit watchdog
# --------------------------------------------------------------------------

def _fit_worker(cfg, state_dict, xb_np, q):
    """Child-process body for :func:`fit_with_watchdog`: rebuild the SAE from
    ``cfg`` + ``state_dict``, run the closed-form ``fit`` on the batch, and hand
    the refreshed ``state_dict`` back. Runs in its own process so a hang in the
    (GIL-holding, signal-proof) Rust fit can be bounded by killing the process."""
    try:
        import torch as _torch
        from gamfit.torch import ManifoldSAE as _ManifoldSAE

        sae = _ManifoldSAE(cfg)
        sae.load_state_dict(state_dict)
        with _torch.no_grad():
            sae.fit(_torch.from_numpy(xb_np))
        q.put(("ok", {k: v.detach().cpu() for k, v in sae.state_dict().items()}))
    except Exception as e:  # noqa: BLE001 — reported to the parent as a refusal
        q.put(("err", f"{type(e).__name__}: {e}"))


def fit_with_watchdog(sae, cfg, xb, timeout_s):
    """Run ``sae.fit(xb)`` under a hard wall-clock timeout, applying the
    refreshed parameters in-place on success. Returns ``(applied, reason)``.

    The closed-form Rust fit can spin without ever returning to the Python eval
    loop (an in-process ``signal.SIGALRM`` never delivers, and ``SIGTERM`` does
    not interrupt the FFI call), so a hang is only bounded by running the fit in
    a killable subprocess and escalating to ``SIGKILL``. ``load_state_dict``
    copies in-place, so the caller's optimizer state stays valid across the
    refresh. On timeout/error the fit is skipped and training continues on pure
    backprop (the closed-form refresh is an accelerator, not a correctness
    requirement)."""
    import multiprocessing as mp
    import queue as _queue

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    xb_np = xb.detach().cpu().numpy()
    host_state = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
    p = ctx.Process(target=_fit_worker, args=(cfg, host_state, xb_np, q), daemon=True)
    p.start()
    try:
        status, payload = q.get(timeout=timeout_s)
    except _queue.Empty:
        status, payload = "timeout", f"timeout>{timeout_s:g}s"
    finally:
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():  # FFI ignored SIGTERM — SIGKILL is uncatchable
                p.kill()
                p.join()
        else:
            p.join()
    if status == "ok":
        with torch.no_grad():
            sae.load_state_dict(payload)
        return True, "ok"
    return False, payload


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--heldout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", action="store_true",
                    help="tiny run: 4 train shards, small width, few steps")
    # manifold SAE
    ap.add_argument("--n-atoms", type=int, default=1024)
    ap.add_argument("--target-k", type=int, default=8)
    ap.add_argument("--sparsity-kind", default="ibp_gumbel",
                    choices=["ibp_gumbel", "softmax_topk", "jumprelu"],
                    help="ibp_gumbel/jumprelu support the closed-form sae.fit "
                         "REML refresh; softmax_topk is gradient-only "
                         "(fit-every is forced to 0)")
    ap.add_argument("--atom-manifold", default="product",
                    choices=["circle", "cylinder", "sphere", "product"])
    ap.add_argument("--intrinsic-rank", type=int, default=2)
    ap.add_argument("--n-basis", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit-every", type=int, default=500,
                    help="closed-form REML refresh cadence; 0 disables")
    ap.add_argument("--fit-timeout", type=float, default=300.0,
                    help="hard wall-clock cap (s) on each closed-form sae.fit; on "
                         "timeout the refresh is skipped and counted as a "
                         "fit_refusal. The Rust fit can hang un-interruptibly, so "
                         "it runs in a killable subprocess — a signal watchdog "
                         "cannot bound it.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--sparsity-weight", type=float, default=1e-4)
    # baseline
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--baseline-width", type=int, default=0,
                    help="0 => parameter-matched: n_atoms * (n_basis+intrinsic_rank)")
    args = ap.parse_args()

    if args.pilot:
        args.n_atoms = min(args.n_atoms, 64)
        args.steps = min(args.steps, 300)
        args.eval_every = 100
        args.ckpt_every = 100
        args.fit_every = min(args.fit_every, 100) if args.fit_every else 0
        args.batch_size = min(args.batch_size, 1024)
        args.fit_timeout = min(args.fit_timeout, 120.0)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data ----
    shards = discover_shards(args.data_dir)
    train_shards, held_shards = split_shards(shards, args.heldout_frac, args.seed)
    if args.pilot:
        train_shards, held_shards = train_shards[:4], held_shards[:2]
    log(f"shards: {len(train_shards)} train / {len(held_shards)} held-out")

    x_train = load_layer(train_shards, args.layer)
    x_held = load_layer(held_shards, args.layer)
    log(f"tokens: train={tuple(x_train.shape)} held={tuple(x_held.shape)} dtype={x_train.dtype}")
    std = Standardizer(x_train)
    log(f"standardizer: scale={std.scale:.4f}")

    (args.out / "split.json").write_text(json.dumps({
        "train": [p.name for p in train_shards],
        "held": [p.name for p in held_shards],
        "seed": args.seed, "layer": args.layer,
    }, indent=1))

    results: dict = {"args": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
                     "evals": [], "fit_refusals": 0}

    def flush_results() -> None:
        (args.out / "results.json").write_text(json.dumps(results, indent=1))

    # ---- manifold SAE ----
    from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

    d_model = x_train.shape[1]
    if args.atom_manifold == "circle" and args.intrinsic_rank != 1:
        log("atom_manifold=circle forces intrinsic_rank=1")
        args.intrinsic_rank = 1
    if args.atom_manifold == "sphere" and args.intrinsic_rank != 2:
        log("atom_manifold=sphere forces intrinsic_rank=2")
        args.intrinsic_rank = 2
    if args.sparsity_kind == "softmax_topk" and args.fit_every:
        log("sparsity=softmax_topk has no closed-form fit lane; forcing fit_every=0")
        args.fit_every = 0

    cfg = ManifoldSAEConfig(
        input_dim=d_model,
        n_atoms=args.n_atoms,
        intrinsic_rank=args.intrinsic_rank,
        atom_manifold=args.atom_manifold,
        n_basis_per_atom=args.n_basis,
        sparsity={"kind": args.sparsity_kind, "target_k": args.target_k,
                  "tau_start": 4.0, "tau_min": 1.0, "tau_steps": max(1, args.steps // 2)},
        dtype=torch.float32,
    )
    sae = ManifoldSAE(cfg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = std(x_train[:8]).to(device)
    try:
        sae = sae.to(device)
        sae(probe)
    except (RuntimeError, TypeError) as e:  # Rust numerics are CPU-backed
        log(f"device {device} failed ({type(e).__name__}: {e}); falling back to cpu")
        device = "cpu"
        sae = sae.to(device)
        sae(probe)
    log(f"manifold SAE on {device}: n_atoms={args.n_atoms} k={args.target_k} "
        f"manifold={args.atom_manifold} rank={args.intrinsic_rank} basis={args.n_basis}")
    results["manifold_device"] = device

    # resume
    start_step = 0
    ckpts = sorted(args.out.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device, weights_only=False)
        sae.load_state_dict(state["sae"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"] + 1
        log(f"resumed from {ckpts[-1].name} at step {start_step}")

    it = batches(x_train, args.batch_size, std, device, args.seed)
    t0 = time.time()
    for step in range(start_step, args.steps):
        xb = next(it)
        out = sae(xb)
        loss = (out.x_hat - xb).pow(2).mean() \
            + args.sparsity_weight * sae.sparsity_penalty(out.gate)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.sparsity.advance_temperature()

        if args.fit_every and step > 0 and step % args.fit_every == 0:
            applied, reason = fit_with_watchdog(sae, cfg, xb, args.fit_timeout)
            if not applied:
                results["fit_refusals"] += 1
                log(f"[step {step}] sae.fit skipped ({reason}); continuing on backprop")

        if step % max(1, args.steps // 50) == 0:
            rate = (step - start_step + 1) / max(1e-9, time.time() - t0)
            log(f"[step {step:6d}] loss={loss.item():.4e} ({rate:.2f} steps/s)")

        if step % args.eval_every == 0 or step == args.steps - 1:
            m = eval_manifold(sae, x_held, std, device)
            m["step"] = step
            results["evals"].append(m)
            log(f"[eval {step:6d}] held EV={m['ev']:.4f} L0={m['l0']:.2f} "
                f"dead={m['dead_atoms']}/{args.n_atoms}")
            flush_results()

        if step % args.ckpt_every == 0 and step > 0:
            torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(),
                        "step": step, "std": std.state()},
                       args.out / f"ckpt_{step:07d}.pt")
            for old in sorted(args.out.glob("ckpt_*.pt"))[:-2]:
                old.unlink()

    final = eval_manifold(sae, x_held, std, device)
    results["manifold_final"] = final
    log(f"[final] manifold held EV={final['ev']:.4f} L0={final['l0']:.2f} "
        f"dead={final['dead_atoms']}/{args.n_atoms}")
    try:
        sae.lock_snapshot()
        log("snapshot locked")
    except Exception as e:  # noqa: BLE001
        log(f"lock_snapshot failed: {e}")
    torch.save({"sae": sae.state_dict(), "cfg": dataclasses.asdict(cfg),
                "std": std.state()}, args.out / "manifold_sae_final.pt")
    flush_results()

    # ---- matched baseline ----
    if not args.no_baseline:
        width = args.baseline_width or args.n_atoms * (args.n_basis + args.intrinsic_rank)
        log(f"training linear TopK baseline: width={width} k={args.target_k}")
        results["baseline_topk"] = train_topk_baseline(
            x_train, x_held, std, width, args.target_k,
            args.steps, args.batch_size, args.lr, args.seed, log)
        log(f"[final] topk baseline held EV={results['baseline_topk']['ev']:.4f} "
            f"L0={results['baseline_topk']['l0']:.2f}")
        flush_results()

    log("DONE")


if __name__ == "__main__":
    main()
