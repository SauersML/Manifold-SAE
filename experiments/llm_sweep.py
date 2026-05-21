"""Manifold-SAE vs vanilla SAE on real LM activations: F-sweep + visualizations.

Run from a Colab T4 cell as::

    !cd /content/Manifold-SAE && python -m experiments.llm_sweep

What this does:
    1. Harvest residual-stream activations from a HuggingFace causal LM
       (cached on /content/runs/LLM_SWEEP/ across re-runs).
    2. For each F in F_values: train vanilla TopK SAE and Manifold-SAE at
       matched (F, top_k), evaluate, lock-and-cache the curve SAE with a
       snapshot batch sized to gamfit's densification limit.
    3. Save plots:
         - explained variance vs F (Pareto curve)        → pareto.png
         - alive features vs F                            → alive.png
         - sample learned curve atoms in 2D PCA           → curves.png
         - position distribution per feature              → positions.png
    4. Save per-F results to results.json and one-shot summary.

This is the head-to-head benchmark on real LM residuals at the scale
where the architectural advantage becomes visible.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn


# ---------------------------------------------------------------------------
# Same gamfit dual-cuBLAS bridge as llm_real.py — see docs/known_issues.md.
def _bypass_gamfit_cuda_check() -> None:
    import gamfit._cuda as _gc

    def _no_conflicts():
        return {
            "platform": "linux", "mapped": {}, "conflicts": {},
            "packaged_nvidia_roots": [], "packaged_complete_stacks": [],
            "system_complete_stacks": [],
        }

    _gc.cuda_diagnostics = _no_conflicts
    _gc.assert_no_cuda_library_conflicts = lambda context: None
    for mod_name in ("gamfit._binding", "gamfit.torch._reml", "gamfit._api"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "assert_no_cuda_library_conflicts"):
                mod.assert_no_cuda_library_conflicts = lambda context: None
            if hasattr(mod, "cuda_diagnostics"):
                mod.cuda_diagnostics = _no_conflicts
        except ImportError:
            pass
    try:
        import gamfit._binding as _gb
        if hasattr(_gb.rust_module, "cache_clear"):
            _gb.rust_module.cache_clear()
    except ImportError:
        pass


_bypass_gamfit_cuda_check()
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    # LM + harvest
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer: int = 12
    n_tokens: int = 80_000
    seq_len: int = 256
    text_dataset: str = "wikitext"
    text_subset: str = "wikitext-2-raw-v1"
    text_split: str = "train"
    min_text_len: int = 200
    max_texts: int = 4000

    # F values to sweep — small to where curve advantage shows, up to where
    # vanilla saturates. Pick top_k as ~1.5% of F (matches standard SAE practice).
    F_values: tuple[int, ...] = (64, 128, 256, 512, 1024)
    top_k_ratio: float = 1.0 / 64.0
    top_k_min: int = 4

    # Architecture
    sae_n_basis: int = 10
    sae_intrinsic_rank: int = 2
    sae_sparsity_weight: float = 3e-4
    sae_ortho_weight: float = 1e-3

    # Training (curve is CPU-bound — keep small)
    n_steps_vanilla: int = 1500
    n_steps_curve: int = 800
    batch_size_vanilla: int = 1024
    snapshot_density_mib: int = 150          # densified-design ceiling for gamfit
    lr: float = 1e-3

    # Eval
    eval_n: int = 8192
    eval_chunk: int = 256                    # vanilla can take eval at once; curve chunks

    # Visualization
    plot_n_atoms: int = 9                    # how many curve atoms to render
    plot_F: int = 256                        # which F's atoms to plot
    plot_t_resolution: int = 96

    output_dir: str = "/content/runs/LLM_SWEEP"
    resume: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# Activation harvest (forward hook on target block)
# ---------------------------------------------------------------------------


def harvest_activations(cfg: SweepConfig, device: torch.device) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {cfg.model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    blocks = None
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    if blocks is None:
        raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[cfg.layer].register_forward_hook(hook)

    ds = load_dataset(cfg.text_dataset, cfg.text_subset, split=cfg.text_split)
    texts = [t for t in ds["text"] if len(t) > cfg.min_text_len][: cfg.max_texts]

    chunks: list[torch.Tensor] = []
    n_collected = 0
    torch.set_grad_enabled(False)
    print(f"[harvest] {cfg.n_tokens:,} tokens at layer {cfg.layer} (of {len(blocks)} blocks)", flush=True)
    try:
        for i in range(0, len(texts), 8):
            if n_collected >= cfg.n_tokens:
                break
            batch = tok(
                texts[i : i + 8], return_tensors="pt", padding=True,
                truncation=True, max_length=cfg.seq_len,
            ).to(device)
            model(**batch)
            h = captured["h"]
            mask = batch["attention_mask"].bool()
            chunks.append(h[mask].cpu().float())
            n_collected += int(mask.sum())
    finally:
        handle.remove()
    torch.set_grad_enabled(True)
    X = torch.cat(chunks, dim=0)[: cfg.n_tokens]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return X


# ---------------------------------------------------------------------------
# Vanilla TopK SAE baseline (matched encoder shape: LayerNorm → Linear(D, 4D) → GELU → Linear(4D, F))
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    def __init__(self, D: int, F: int, top_k: int) -> None:
        super().__init__()
        self.F = F
        self.top_k = top_k
        H = 4 * D
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H)
        self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(sae, X, n_steps: int, batch_size: int, lr: float, label: str,
              is_curve: bool = False, sae_cfg=None) -> float:
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = X.shape[0]
    t0 = time.time()
    log_every = max(n_steps // 5, 1)
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), device=X.device)
        batch = X[idx]
        opt.zero_grad(set_to_none=True)
        if is_curve:
            from manifold_sae.losses import total_loss
            out = sae(batch)
            losses = total_loss(out, batch, sae_cfg)
            loss = losses["total"]
            mse = losses["mse"]
        else:
            recon, z = sae(batch)
            mse = ((recon - batch) ** 2).mean()
            loss = mse + 3e-4 * z.abs().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"    [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Eval (chunked for curve to respect gamfit density limit)
# ---------------------------------------------------------------------------


def eval_vanilla(sae, X_eval, F: int) -> tuple[float, int]:
    sae.eval()
    with torch.no_grad():
        recon, z = sae(X_eval)
        mse = float(((recon - X_eval) ** 2).mean())
        alive = int((z > 0).any(0).sum())
    return mse, alive


def eval_curve(sae, X_eval, F: int, chunk: int) -> tuple[float, int]:
    sae.eval()
    sq_sum = 0.0
    n_tok = 0
    device = X_eval.device
    alive_mask = torch.zeros(F, dtype=torch.bool, device=device)
    with torch.no_grad():
        for start in range(0, X_eval.shape[0], chunk):
            x = X_eval[start : start + chunk]
            out = sae(x)
            sq_sum += float(((out.reconstruction - x) ** 2).sum())
            n_tok += x.numel()
            alive_mask |= (out.amplitudes > 1e-3).any(0)
    return sq_sum / max(n_tok, 1), int(alive_mask.sum())


def snapshot_batch_size(F: int, K: int, density_mib: int) -> int:
    """Largest batch (in tokens) that keeps gamfit's densified F·B·K design
    matrix under density_mib MiB. Used for lock-and-cache snapshot fitting.
    """
    max_bytes = density_mib * 1024 * 1024
    return max(32, min(8192, max_bytes // (F * K * 8)))


# ---------------------------------------------------------------------------
# Curve extraction for visualization
# ---------------------------------------------------------------------------


def extract_curves_2d(sae, t_grid: torch.Tensor, n_atoms: int) -> tuple[np.ndarray, list[int]]:
    """Sample each alive curve atom on a fine t grid, project to 2D PCA.

    Returns (curves_2d, atom_indices) where curves_2d has shape
    (n_chosen, T, 2) and atom_indices lists the SAE feature indices chosen.
    """
    import gamfit.torch as gt

    device = next(sae.parameters()).device
    t = t_grid.to(device=device, dtype=torch.float64)
    phi = gt.duchon_basis_1d(t, sae.centers, m=2, periodic=False)         # (T, K)
    g_intrinsic = torch.einsum("tk,fkr->ftr", phi, sae.B_locked)          # (F, T, R)
    g_ambient = torch.einsum("ftr,fdr->ftd", g_intrinsic, sae.directions.to(torch.float64))  # (F, T, D)

    # Choose features with the largest curve-Frobenius (i.e. "most alive on the snapshot").
    norms = g_ambient.reshape(g_ambient.shape[0], -1).norm(dim=1).cpu().numpy()
    chosen = list(np.argsort(-norms)[:n_atoms])
    out = np.zeros((len(chosen), t.shape[0], 2), dtype=np.float64)
    for i, k in enumerate(chosen):
        gk = g_ambient[k].cpu().numpy()
        gk_c = gk - gk.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(gk_c, full_matrices=False)
        pcs = vh[:2]
        out[i] = gk_c @ pcs.T
    return out, [int(k) for k in chosen]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pareto(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = [r["F"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(F, [r["vanilla_explained"] for r in results], "o-", label="vanilla TopK SAE")
    ax.plot(F, [r["curve_explained"] for r in results], "s-", label="Manifold-SAE (training)")
    ax.plot(F, [r["curve_locked_explained"] for r in results], "^--", label="Manifold-SAE (locked snapshot)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dictionary size F")
    ax.set_ylabel("explained variance")
    ax.set_title("Reconstruction Pareto curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=110)
    plt.close(fig)


def plot_alive(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = [r["F"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(F, [r["vanilla_alive"] / r["F"] for r in results], "o-", label="vanilla alive ratio")
    ax.plot(F, [r["curve_alive"] / r["F"] for r in results], "s-", label="curve alive ratio")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dictionary size F")
    ax.set_ylabel("alive feature fraction")
    ax.set_title("Alive features vs dictionary size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "alive.png", dpi=110)
    plt.close(fig)


def plot_curves(curves_2d: np.ndarray, atom_indices: list[int], F: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = curves_2d.shape[0]
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(n):
        ax = axes[i // cols, i % cols]
        c = curves_2d[i]
        ax.plot(c[:, 0], c[:, 1], "o-", color="C0", markersize=2)
        ax.scatter([c[0, 0]], [c[0, 1]], color="green", s=30, label="t=0", zorder=5)
        ax.scatter([c[-1, 0]], [c[-1, 1]], color="red", s=30, label="t=1", zorder=5)
        ax.set_title(f"atom #{atom_indices[i]}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Learned curve atoms at F={F} (2D PCA of g_k(t) over t∈[0,1])")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=110)
    plt.close(fig)


def plot_positions(positions: np.ndarray, alive_idx: np.ndarray, F: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = alive_idx[:9]
    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)
    for i, k in enumerate(chosen):
        ax = axes[i // cols, i % cols]
        ax.hist(positions[:, k], bins=30, range=(0.0, 1.0), color="C0")
        ax.set_title(f"atom #{int(k)}")
        ax.set_xlim(0, 1)
    for i in range(len(chosen), rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Firing-position distribution per atom at F={F}")
    fig.tight_layout()
    fig.savefig(out_dir / "positions.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-F training + eval
# ---------------------------------------------------------------------------


def run_one_F(cfg: SweepConfig, F: int, X_n: torch.Tensor, var: float, device: torch.device,
              out_dir: Path, do_visualize: bool) -> dict:
    print(f"\n========== F = {F} ==========", flush=True)
    D = X_n.shape[1]
    top_k = max(cfg.top_k_min, int(F * cfg.top_k_ratio))
    bsc = max(32, min(512, cfg.snapshot_density_mib * 1024 * 1024 // (F * cfg.sae_n_basis * 8)))
    print(f"  D={D}  F={F}  top_k={top_k}  batch_size_curve={bsc}", flush=True)

    # --- Vanilla
    vanilla = VanillaSAE(D, F, top_k).to(device)
    n_v = sum(p.numel() for p in vanilla.parameters())
    t_v = train_one(vanilla, X_n, cfg.n_steps_vanilla, cfg.batch_size_vanilla, cfg.lr,
                    f"van[F={F}]", is_curve=False)
    X_eval = X_n[: min(cfg.eval_n, X_n.shape[0])]
    mse_v, alive_v = eval_vanilla(vanilla, X_eval, F)

    # --- Curve
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=D, n_features=F, n_basis=cfg.sae_n_basis, top_k=top_k,
        intrinsic_rank=cfg.sae_intrinsic_rank,
        sparsity_weight=cfg.sae_sparsity_weight,
        ortho_weight=cfg.sae_ortho_weight,
        encoder_type="linear", continuous_amp=True,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    n_c = sum(p.numel() for p in curve.parameters())
    t_c = train_one(curve, X_n, cfg.n_steps_curve, bsc, cfg.lr,
                    f"crv[F={F}]", is_curve=True, sae_cfg=sae_cfg)
    mse_c, alive_c = eval_curve(curve, X_eval, F, cfg.eval_chunk)

    # --- Lock-and-cache with a properly-sized snapshot batch
    snap_n = snapshot_batch_size(F, cfg.sae_n_basis, cfg.snapshot_density_mib)
    snap = X_n[: snap_n]
    print(f"  [snapshot] fitting REML on {snap_n} tokens (~{F*snap_n*cfg.sae_n_basis*8/1024**2:.0f} MiB densified)", flush=True)
    curve.update_snapshot(snap)
    curve.inference_mode = True
    mse_locked, _ = eval_curve(curve, X_eval, F, cfg.eval_chunk)
    curve.inference_mode = False  # leave training-mode accessible for visualization extraction

    print(f"  [F={F}] vanilla MSE={mse_v:.4f} expl={1-mse_v/var:.3f} alive={alive_v}/{F} params={n_v/1e6:.1f}M", flush=True)
    print(f"  [F={F}] curve   MSE={mse_c:.4f} expl={1-mse_c/var:.3f} alive={alive_c}/{F} params={n_c/1e6:.1f}M", flush=True)
    print(f"  [F={F}] locked  MSE={mse_locked:.4f} expl={1-mse_locked/var:.3f}", flush=True)

    result = {
        "F": F, "top_k": top_k, "batch_size_curve": bsc,
        "vanilla_mse": mse_v, "vanilla_explained": 1 - mse_v / var, "vanilla_alive": alive_v,
        "vanilla_params_M": n_v / 1e6, "vanilla_train_s": t_v,
        "curve_mse": mse_c, "curve_explained": 1 - mse_c / var, "curve_alive": alive_c,
        "curve_params_M": n_c / 1e6, "curve_train_s": t_c,
        "curve_locked_mse": mse_locked,
        "curve_locked_explained": 1 - mse_locked / var,
    }

    if do_visualize:
        print(f"  [viz] extracting curve atoms + positions for F={F}", flush=True)
        curve.inference_mode = True  # use locked B for visualization
        t_grid = torch.linspace(0.01, 0.99, cfg.plot_t_resolution, dtype=torch.float64)
        curves_2d, atom_indices = extract_curves_2d(curve, t_grid, cfg.plot_n_atoms)
        plot_curves(curves_2d, atom_indices, F, out_dir)

        # Position distribution on the eval batch
        curve.inference_mode = False
        with torch.no_grad():
            out = curve(X_eval[: min(2048, X_eval.shape[0])])
            positions = out.positions.detach().cpu().numpy()
            alive = ((out.amplitudes > 1e-3).any(0).cpu().numpy()).nonzero()[0]
        plot_positions(positions, alive, F, out_dir)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: SweepConfig | None = None) -> int:
    if cfg is None:
        cfg = SweepConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} output_dir={out_dir}", flush=True)

    # Activation cache (shared across F)
    act_path = out_dir / "activations.pt"
    act_struct = {
        "model_name": cfg.model_name, "layer": cfg.layer,
        "text_dataset": cfg.text_dataset, "text_subset": cfg.text_subset, "seq_len": cfg.seq_len,
    }
    X: torch.Tensor | None = None
    if cfg.resume and act_path.exists():
        cached = torch.load(act_path, map_location="cpu", weights_only=False)
        cached_struct = {k: cached.get("sig", {}).get(k) for k in act_struct}
        cached_n = int(cached["X"].shape[0]) if "X" in cached else 0
        if cached_struct == act_struct and cached_n >= cfg.n_tokens:
            X = cached["X"][: cfg.n_tokens]
            print(f"[harvest] reusing cached: cached={cached_n} requested={cfg.n_tokens}", flush=True)
    if X is None:
        X = harvest_activations(cfg, device)
        torch.save({"X": X, "sig": {**act_struct, "n_tokens": int(X.shape[0])}}, act_path)
        print(f"[harvest] saved {act_path}", flush=True)

    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = ((X - mu) / max(sigma, 1e-6)).to(device)
    var = float(X_n.var().item())
    print(f"[setup] X_n shape={tuple(X_n.shape)} var={var:.3f}", flush=True)

    results: list[dict] = []
    for F in cfg.F_values:
        do_viz = (F == cfg.plot_F)
        result = run_one_F(cfg, F, X_n, var, device, out_dir, do_visualize=do_viz)
        results.append(result)
        # Save incrementally so partial sweeps still have a results file.
        (out_dir / "results.json").write_text(json.dumps({"config": asdict(cfg), "var": var, "results": results}, indent=2, default=float))

    plot_pareto(results, out_dir)
    plot_alive(results, out_dir)

    print("\n=========== Summary ===========", flush=True)
    print(f"{'F':>6} {'top_k':>6}  {'van expl':>9}  {'crv expl':>9}  {'lock expl':>10}  {'van alive':>10}  {'crv alive':>10}")
    for r in results:
        print(f"{r['F']:6d} {r['top_k']:6d}  {r['vanilla_explained']:9.3f}  "
              f"{r['curve_explained']:9.3f}  {r['curve_locked_explained']:10.3f}  "
              f"{r['vanilla_alive']:6d}/{r['F']:<3d}  {r['curve_alive']:6d}/{r['F']:<3d}", flush=True)
    print(f"\nplots: {out_dir}/pareto.png, alive.png, curves.png (F={cfg.plot_F}), positions.png (F={cfg.plot_F})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
