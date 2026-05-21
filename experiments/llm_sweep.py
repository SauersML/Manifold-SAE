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

    # F values to sweep — pushed small so per-atom geometry matters. At larger
    # F (and the default Qwen 0.5B layer-12 signal) both architectures saturate
    # at >99% explained variance; the architectural difference only shows up
    # where atoms are scarce and each one has to compress.
    F_values: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    # Aggressive sparsity. At F=16 with TopK=2, both SAEs are starved — the
    # comparison stops being saturated and becomes diagnostic.
    top_k_min: int = 2
    top_k_ratio: float = 1.0 / 128.0

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


def _ckpt_sig(F: int, top_k: int, label: str, sae_cfg_dict: dict | None = None) -> dict:
    sig = {"label": label, "F": F, "top_k": top_k}
    if sae_cfg_dict is not None:
        sig["n_basis"] = sae_cfg_dict.get("n_basis")
        sig["intrinsic_rank"] = sae_cfg_dict.get("intrinsic_rank")
    return sig


def _save_ckpt(sae, opt, step: int, sig: dict, path: Path) -> None:
    """Atomic write: save to tmp then rename, so a crash mid-save doesn't
    leave a half-written checkpoint that breaks the next resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "step": step, "sig": sig}, tmp)
    tmp.replace(path)


def train_one(sae, X, n_steps: int, batch_size: int, lr: float, label: str,
              ckpt_path: Path | None = None, resume: bool = True, sig: dict | None = None,
              ckpt_every: int = 200,
              is_curve: bool = False, sae_cfg=None) -> float:
    """Train one SAE. Per-step warm-start with periodic checkpoint save:
      - Resume from saved (step, weights, optimizer) if checkpoint matches sig.
      - Save snapshot every ckpt_every steps so a crash mid-training preserves
        progress (atomic write — partial saves never corrupt the file).
      - Skip entirely if already trained to n_steps."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    start_step = 0
    if resume and ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cached_sig = ckpt.get("sig", {})
        if all(cached_sig.get(k) == v for k, v in (sig or {}).items()):
            sae.load_state_dict(ckpt["sae"])
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError):
                print(f"    [{label}] optimizer state mismatched; reinitializing", flush=True)
            start_step = int(ckpt["step"])
            print(f"    [{label}] resumed from {ckpt_path} at step {start_step}", flush=True)
        else:
            print(f"    [{label}] checkpoint sig mismatch — starting fresh", flush=True)

    n = X.shape[0]
    t0 = time.time()
    log_every = max(n_steps // 5, 1)
    if start_step >= n_steps:
        print(f"    [{label}] already trained to step {start_step} (target {n_steps}); skipping", flush=True)
        return 0.0
    for step in range(start_step, n_steps):
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
        # Periodic checkpoint save (atomic via tmp+rename).
        if ckpt_path is not None and (step + 1) % ckpt_every == 0 and (step + 1) < n_steps:
            _save_ckpt(sae, opt, step + 1, sig or {}, ckpt_path)
    if ckpt_path is not None:
        _save_ckpt(sae, opt, n_steps, sig or {}, ckpt_path)
        print(f"    [{label}] saved checkpoint to {ckpt_path}", flush=True)
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


@torch.no_grad()
def extract_curves_2d(sae, t_grid: torch.Tensor, n_atoms: int) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Sample each alive curve atom on a fine t grid, project to 2D PCA.

    Returns (curves_2d, atom_indices, intrinsic_dim) where:
      - curves_2d has shape (n_chosen, T, 2)
      - atom_indices lists the SAE feature indices chosen
      - intrinsic_dim per chosen atom is `second_sv / first_sv` of g_k(t)
        across t — 0 means "flat line" (architecturally equivalent to vanilla),
        1 means "fully 2D curve" (architecturally distinct from vanilla)
    """
    import gamfit.torch as gt

    device = next(sae.parameters()).device
    t = t_grid.to(device=device, dtype=torch.float64)
    phi = gt.duchon_basis_1d(t, sae.centers, m=2, periodic=False)         # (T, K)
    g_intrinsic = torch.einsum("tk,fkr->ftr", phi, sae.B_locked)          # (F, T, R)
    g_ambient = torch.einsum("ftr,fdr->ftd", g_intrinsic, sae.directions.to(torch.float64))  # (F, T, D)

    norms = g_ambient.reshape(g_ambient.shape[0], -1).norm(dim=1).cpu().numpy()
    chosen = list(np.argsort(-norms)[:n_atoms])
    out_curves = np.zeros((len(chosen), t.shape[0], 2), dtype=np.float64)
    intrinsic = np.zeros(len(chosen), dtype=np.float64)
    for i, k in enumerate(chosen):
        gk = g_ambient[k].cpu().numpy()
        gk_c = gk - gk.mean(axis=0, keepdims=True)
        _, sv, vh = np.linalg.svd(gk_c, full_matrices=False)
        pcs = vh[:2]
        out_curves[i] = gk_c @ pcs.T
        if len(sv) >= 2 and sv[0] > 1e-12:
            intrinsic[i] = float(sv[1] / sv[0])
    return out_curves, [int(k) for k in chosen], intrinsic


@torch.no_grad()
def per_pc_explained(X_eval: torch.Tensor, recon: torch.Tensor, k: int = 64) -> np.ndarray:
    """Variance explained on the top-k principal components of X_eval.

    Saturated reconstruction quality (overall MSE ≈ 0) masks per-direction
    differences. The head of the PC spectrum is easy; the tail is where
    atom geometry actually matters. Returns (k,) array of explained
    fraction per PC, from largest to smallest variance.
    """
    X_c = X_eval - X_eval.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    pcs = Vt[:k]                                          # (k, D)
    proj_X = X_c @ pcs.t()                                # (B, k)
    proj_r = (recon - X_eval.mean(0, keepdim=True)) @ pcs.t()
    var_X = (proj_X ** 2).mean(0)
    err = ((proj_X - proj_r) ** 2).mean(0)
    return (1.0 - err / var_X.clamp(min=1e-12)).cpu().numpy()


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


def plot_curves(curves_2d: np.ndarray, atom_indices: list[int], F: int, out_dir: Path,
                intrinsic_dim: np.ndarray) -> None:
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
        ax.set_title(f"atom #{atom_indices[i]}  σ₂/σ₁={intrinsic_dim[i]:.2f}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(
        f"Learned curve atoms at F={F} (2D PCA of g_k(t) over t∈[0,1])\n"
        f"σ₂/σ₁ ≈ 0 means atom is essentially a line (equivalent to vanilla SAE direction); "
        f"σ₂/σ₁ ≈ 1 means genuinely 2D curve"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_intrinsic_dim_hist(sae, F: int, out_dir: Path) -> None:
    """Distribution of σ₂/σ₁ across ALL alive atoms (not just the rendered 9).

    Tells you whether the curve SAE is actually using its 2D intrinsic
    capacity or collapsing to vanilla-SAE-style direction atoms.
    """
    import gamfit.torch as gt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(sae.parameters()).device
    t = torch.linspace(0.02, 0.98, 64, dtype=torch.float64, device=device)
    phi = gt.duchon_basis_1d(t, sae.centers, m=2, periodic=False)
    g = torch.einsum("tk,fkr->ftr", phi, sae.B_locked)
    amb = torch.einsum("ftr,fdr->ftd", g, sae.directions.to(torch.float64)).cpu().numpy()

    ratios = []
    norms = []
    for k in range(amb.shape[0]):
        gk_c = amb[k] - amb[k].mean(0, keepdims=True)
        _, sv, _ = np.linalg.svd(gk_c, full_matrices=False)
        norms.append(float(np.linalg.norm(amb[k])))
        if len(sv) >= 2 and sv[0] > 1e-12:
            ratios.append(float(sv[1] / sv[0]))

    norms = np.array(norms)
    alive_thresh = max(1e-6, float(np.percentile(norms, 50)) * 0.1)
    ratios_alive = np.array(ratios)[norms[: len(ratios)] > alive_thresh]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(ratios_alive) > 0:
        ax.hist(ratios_alive, bins=30, range=(0, 1), color="C0")
    ax.set_xlabel("σ₂ / σ₁  (rank-2-ness of curve in ambient)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Curve-atom intrinsic dimensionality at F={F}\n"
        f"{len(ratios_alive)} alive atoms; far-left bin = essentially-vanilla atoms"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "intrinsic_dim.png", dpi=110)
    plt.close(fig)


def plot_per_pc(results: list[dict], out_dir: Path) -> None:
    """For each F, plot explained-variance per PC. The first ~10 PCs are
    trivial; PCs 20-50 are where the architectures diverge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        F = r["F"]
        ax.plot(r["per_pc_vanilla"], "-", alpha=0.7, label=f"vanilla F={F}")
        ax.plot(r["per_pc_curve"], "--", alpha=0.7, label=f"curve   F={F}")
    ax.set_xlabel("principal-component index (top = largest variance)")
    ax.set_ylabel("explained variance on this PC")
    ax.set_title("Per-PC reconstruction quality (saturation-aware comparison)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "per_pc.png", dpi=110)
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

    # Per-F eval cache. If a previous run completed this F with the same
    # structural config, just return the saved result and skip everything.
    eval_cache_path = out_dir / f"eval_F{F}.json"
    # `forward_semantics` bumps when the SAE forward changes in a way that
    # alters reported MSE numbers. v2: fixed the amp²·curve(t) bug in
    # _forward_training. Old eval cache entries reported wrong locked MSE.
    # Note: this invalidates ONLY the cached evaluation results, NOT the
    # SAE weights — the trained encoder + W are still useful and re-eval
    # under the corrected forward will produce honest numbers.
    eval_sig = {
        "forward_semantics": 2,
        "F": F, "top_k": top_k, "n_steps_vanilla": cfg.n_steps_vanilla,
        "n_steps_curve": cfg.n_steps_curve, "batch_size_curve": bsc,
        "n_basis": cfg.sae_n_basis, "intrinsic_rank": cfg.sae_intrinsic_rank,
        "eval_n": cfg.eval_n,
    }
    if cfg.resume and eval_cache_path.exists():
        cached = json.loads(eval_cache_path.read_text())
        if all(cached.get("eval_sig", {}).get(k) == v for k, v in eval_sig.items()):
            # Viz hasn't been redone for this re-run, but the metrics are saved.
            # Only re-run the run when do_visualize=True and the viz output is
            # missing — otherwise return the cache verbatim.
            need_viz = do_visualize and not (out_dir / "curves.png").exists()
            if not need_viz:
                print(f"  [F={F}] eval cache hit at {eval_cache_path}; skipping retrain+eval", flush=True)
                return cached["result"]
            else:
                print(f"  [F={F}] eval cache hit but viz missing; rebuilding curve SAE for viz only", flush=True)

    # --- Vanilla
    vanilla = VanillaSAE(D, F, top_k).to(device)
    n_v = sum(p.numel() for p in vanilla.parameters())
    t_v = train_one(
        vanilla, X_n, cfg.n_steps_vanilla, cfg.batch_size_vanilla, cfg.lr,
        f"van[F={F}]",
        ckpt_path=out_dir / f"vanilla_F{F}.pt",
        resume=cfg.resume,
        sig=_ckpt_sig(F, top_k, "vanilla"),
        is_curve=False,
    )
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
    t_c = train_one(
        curve, X_n, cfg.n_steps_curve, bsc, cfg.lr,
        f"crv[F={F}]",
        ckpt_path=out_dir / f"curve_F{F}.pt",
        resume=cfg.resume,
        sig=_ckpt_sig(F, top_k, "curve", {"n_basis": cfg.sae_n_basis, "intrinsic_rank": cfg.sae_intrinsic_rank}),
        is_curve=True, sae_cfg=sae_cfg,
    )
    mse_c, alive_c = eval_curve(curve, X_eval, F, cfg.eval_chunk)

    # --- Lock-and-cache with a properly-sized snapshot batch.
    snap_n = snapshot_batch_size(F, cfg.sae_n_basis, cfg.snapshot_density_mib)
    snap = X_n[: snap_n]
    print(f"  [snapshot] fitting REML on {snap_n} tokens (~{F*snap_n*cfg.sae_n_basis*8/1024**2:.0f} MiB densified)", flush=True)
    # Capture training-mode reconstruction on the snapshot batch BEFORE
    # update_snapshot so we can diagnose locked-vs-training MSE divergence
    # post hoc. If snapshot_training_mse is close to mse_c but locked_mse
    # is far, the issue is in the locked forward path or the rescale fix.
    with torch.no_grad():
        snapshot_training_recon = curve(snap)
        snapshot_training_mse = float(((snapshot_training_recon.reconstruction - snap) ** 2).mean())
    curve.update_snapshot(snap)
    # Re-save curve checkpoint *after* update_snapshot so the locked
    # buffers (B_locked, lam_locked, soft_min_locked, soft_max_locked,
    # has_snapshot) persist across re-runs — next invocation reloads
    # the snapshot-ready SAE without re-fitting.
    ckpt_path = out_dir / f"curve_F{F}.pt"
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["sae"] = curve.state_dict()
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(ckpt_path)
        print(f"  [snapshot] re-saved {ckpt_path} with snapshot buffers", flush=True)
    curve.inference_mode = True
    # Lock-and-cache health check: eval inference-mode on the SAME snapshot
    # batch. If this differs from snapshot_training_mse, the locked forward
    # path is broken; if it matches, the lock-and-cache architecturally
    # generalizes only to the snapshot data, not to chunks drawn from outside.
    with torch.no_grad():
        out_snap_inf = curve(snap)
        snapshot_locked_mse = float(((out_snap_inf.reconstruction - snap) ** 2).mean())
    print(f"  [diag] training-fwd on snapshot batch: mse={snapshot_training_mse:.4f}", flush=True)
    print(f"  [diag] locked-fwd   on snapshot batch: mse={snapshot_locked_mse:.4f}", flush=True)
    mse_locked, _ = eval_curve(curve, X_eval, F, cfg.eval_chunk)
    curve.inference_mode = False

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
        "snapshot_training_mse": snapshot_training_mse,
        "snapshot_locked_mse": snapshot_locked_mse,
    }

    # Per-PC reconstruction quality on the top-K PCs of X_eval — the
    # discriminating metric when overall MSE is saturated.
    pc_k = min(64, X_eval.shape[1])
    with torch.no_grad():
        recon_v, _ = vanilla(X_eval)
        pc_v = per_pc_explained(X_eval, recon_v, k=pc_k)
        # Curve: chunked recon for memory
        recon_c = torch.empty_like(X_eval)
        for start in range(0, X_eval.shape[0], cfg.eval_chunk):
            x = X_eval[start : start + cfg.eval_chunk]
            recon_c[start : start + cfg.eval_chunk] = curve(x).reconstruction
        pc_c = per_pc_explained(X_eval, recon_c, k=pc_k)
    result["per_pc_vanilla"] = pc_v.tolist()
    result["per_pc_curve"] = pc_c.tolist()

    if do_visualize:
        print(f"  [viz] extracting curve atoms + positions for F={F}", flush=True)
        curve.inference_mode = True
        t_grid = torch.linspace(0.01, 0.99, cfg.plot_t_resolution, dtype=torch.float64)
        curves_2d, atom_indices, intrinsic_dim = extract_curves_2d(curve, t_grid, cfg.plot_n_atoms)
        plot_curves(curves_2d, atom_indices, F, out_dir, intrinsic_dim)
        result["plotted_atom_intrinsic_dim"] = intrinsic_dim.tolist()

        curve.inference_mode = False
        with torch.no_grad():
            out = curve(X_eval[: min(2048, X_eval.shape[0])])
            positions = out.positions.detach().cpu().numpy()
            alive = ((out.amplitudes > 1e-3).any(0).cpu().numpy()).nonzero()[0]
        plot_positions(positions, alive, F, out_dir)

        # Curve dimensionality: histogram of (σ₂/σ₁) across ALL alive atoms.
        # 0 = essentially a line (equivalent to vanilla); near 1 = genuine 2D
        # curve. Tells you whether the curve SAE is actually using its
        # intrinsic rank or collapsing to vanilla behavior.
        plot_intrinsic_dim_hist(curve, F, out_dir)

    # Per-F eval cache: write the full result dict + structural signature.
    # On the next re-run, run_one_F sees this file and returns immediately
    # without retraining, snapshotting, or re-evaluating.
    eval_cache_path.write_text(json.dumps({"eval_sig": eval_sig, "result": result}, indent=2, default=float))
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
    plot_per_pc(results, out_dir)

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
