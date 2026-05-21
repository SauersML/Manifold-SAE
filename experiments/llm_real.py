"""Curve SAE vs vanilla TopK SAE on real LM residual-stream activations.

Single-process, both phases on GPU. The gamfit dual-cuBLAS conflict on
Colab is resolved upstream via LD_PRELOAD in colab_run.txt — once only
one cuBLAS is mapped in the process, gamfit's safety check passes and
its CUDA dispatch can run REML on the GPU.

Run from a Colab T4 cell as::

    LD_PRELOAD=<pip cublas> python -m experiments.llm_real

or import and call ``main(Config(...))`` to override defaults.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn


# ---------------------------------------------------------------------------
# Transitional bridge: neutralize gamfit's CUDA dual-stack safety check on
# environments (Colab, several cloud images) that have both /usr/local/cuda*
# and the torch-bundled nvidia/cublas-cu12 reachable. The proper fix is
# upstream in gam (commit downgrades the assert to a once-per-process
# warning); until that ships, we patch at the *source* — `cuda_diagnostics`
# — so every consumer of the diagnostic sees an empty conflict set. This
# works regardless of which module captured the assert function reference
# at import time.
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
class Config:
    # Default: Qwen2.5-0.5B (pure text causal LM, Apache 2.0, 24 layers,
    # hidden_size=896, Qwen2 tokenizer). Verified to work with AutoModel +
    # standard forward hooks. Qwen3.5 is multimodal and behaves differently.
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer: int = 12  # mid for 24-layer Qwen2.5-0.5B
    n_tokens: int = 80_000
    seq_len: int = 256
    text_dataset: str = "wikitext"
    text_subset: str = "wikitext-2-raw-v1"
    text_split: str = "train"
    min_text_len: int = 200
    max_texts: int = 4000
    n_features: int = 2048
    top_k: int = 24
    # vanilla is fast on GPU; curve does gamfit REML per batch (currently on
    # CPU due to the cuBLAS dual-stack issue) so fewer steps + smaller batch
    # keep total run time under ~15 min on T4 + 1 CPU core.
    n_steps: int = 3000
    n_steps_curve: int = 1500
    batch_size: int = 1024
    batch_size_curve: int = 128       # F·B = 2048·128 = 262K rows; ~20 MiB densified
    lr: float = 1e-3
    sae_n_basis: int = 10
    sae_intrinsic_rank: int = 2
    sae_sparsity_weight: float = 3e-4
    sae_ortho_weight: float = 1e-3
    eval_n: int = 8192
    # Use an absolute path outside the repo so checkpoints survive
    # `rm -rf /content/Manifold-SAE` in the Colab setup cell.
    # Env var override lets a job submitter point outputs at any working-dir
    # tree without code edits. See heimdall_jobs/submit.py.
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/LLM_REAL")
    seed: int = 0
    # Warm-start: if a checkpoint exists at <output_dir>/checkpoint.pt with
    # a matching config, resume (model + optimizer + step count). Set
    # resume=False to force a fresh run.
    resume: bool = True


# ---------------------------------------------------------------------------
# Activation harvest
# ---------------------------------------------------------------------------


def harvest_activations(cfg: Config, device: torch.device) -> torch.Tensor:
    """Collect residual-stream activations from one layer using a forward
    hook on that block's output. Avoids ``output_hidden_states=True`` which
    materializes all 25 layers' activations in memory per batch.
    """
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {cfg.model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    # Find the transformer-block module list and pick our target layer.
    # Supports gpt2 (.h[i]), Qwen2/Qwen3/Llama/Mistral (.layers[i]).
    blocks = None
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    if blocks is None:
        raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")
    if cfg.layer < 0 or cfg.layer >= len(blocks):
        raise ValueError(f"layer {cfg.layer} out of range for model with {len(blocks)} blocks")

    captured = {}

    def hook(_module, _inputs, output):
        # Block outputs are typically a tuple (hidden_state, ...) or a single tensor.
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[cfg.layer].register_forward_hook(hook)

    print(f"[harvest] loading text: {cfg.text_dataset}/{cfg.text_subset}", flush=True)
    ds = load_dataset(cfg.text_dataset, cfg.text_subset, split=cfg.text_split)
    texts = [t for t in ds["text"] if len(t) > cfg.min_text_len][: cfg.max_texts]

    chunks = []
    n_collected = 0
    torch.set_grad_enabled(False)
    print(f"[harvest] collecting {cfg.n_tokens:,} tokens at layer {cfg.layer} (block {cfg.layer} of {len(blocks)})", flush=True)
    try:
        for i in range(0, len(texts), 8):
            if n_collected >= cfg.n_tokens:
                break
            batch = tok(
                texts[i : i + 8],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.seq_len,
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
    print(f"[harvest] X shape: {tuple(X.shape)}  mean={X.mean():.3f}  std={X.std():.3f}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return X


# ---------------------------------------------------------------------------
# Vanilla TopK SAE baseline
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


def _ckpt_signature(cfg: Config, label: str) -> dict:
    """Structural-only signature: changing lr / batch_size / n_steps does NOT
    invalidate a checkpoint (the weights stay loadable; training simply
    continues). Only fields that change the model's shape are included.
    """
    sig: dict[str, object] = {
        "label": label,
        "model_name": cfg.model_name,
        "layer": cfg.layer,
        "n_features": cfg.n_features,
        "top_k": cfg.top_k,
    }
    if label == "curve":
        sig["sae_n_basis"] = cfg.sae_n_basis
        sig["sae_intrinsic_rank"] = cfg.sae_intrinsic_rank
    return sig


def train_sae(sae, X, cfg: Config, label: str, ckpt_path: Path, is_curve: bool = False, sae_cfg=None) -> float:
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    start_step = 0
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        new_sig = _ckpt_signature(cfg, label)
        cached_sig = ckpt.get("sig", {})
        # Subset match: cached signature may contain extra (legacy) fields
        # like lr / batch_size that we've since dropped from the signature.
        # As long as every field of the NEW signature matches the cached
        # value for that field, the weights are compatible.
        compatible = all(cached_sig.get(k) == v for k, v in new_sig.items())
        if compatible:
            sae.load_state_dict(ckpt["sae"])
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError):
                print(f"  [{label}] optimizer state incompatible; reinitializing optimizer (weights resumed)", flush=True)
            start_step = int(ckpt["step"])
            print(f"  [{label}] resumed from {ckpt_path} at step {start_step}", flush=True)
        else:
            diffs = [(k, cached_sig.get(k), v) for k, v in new_sig.items() if cached_sig.get(k) != v]
            print(f"  [{label}] checkpoint signature mismatch on {diffs} — starting fresh", flush=True)

    n = X.shape[0]
    batch_size = cfg.batch_size_curve if is_curve else cfg.batch_size
    n_steps = cfg.n_steps_curve if is_curve else cfg.n_steps
    t0 = time.time()
    log_every = max(n_steps // 10, 1)
    if start_step >= n_steps:
        print(f"  [{label}] already trained for {start_step} steps (target {n_steps}); skipping", flush=True)
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
            print(f"  [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
    torch.save(
        {"sae": sae.state_dict(), "opt": opt.state_dict(), "step": n_steps, "sig": _ckpt_signature(cfg, label)},
        ckpt_path,
    )
    print(f"  [{label}] saved checkpoint to {ckpt_path}", flush=True)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config | None = None) -> int:
    if cfg is None:
        cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}  output_dir={out_dir}", flush=True)

    # Cache harvested activations. Reuse rule: same (model, layer, dataset,
    # seq_len) AND cached has at least n_tokens rows. Asking for FEWER tokens
    # is satisfied by slicing the cache; only asking for MORE tokens (or
    # changing model/layer/dataset) triggers re-harvest.
    act_path = out_dir / "activations.pt"
    act_struct = {
        "model_name": cfg.model_name,
        "layer": cfg.layer,
        "text_dataset": cfg.text_dataset,
        "text_subset": cfg.text_subset,
        "seq_len": cfg.seq_len,
    }
    X = None
    if cfg.resume and act_path.exists():
        cached = torch.load(act_path, map_location="cpu", weights_only=False)
        cached_struct = {k: cached.get("sig", {}).get(k) for k in act_struct}
        cached_n = int(cached["X"].shape[0]) if "X" in cached else 0
        if cached_struct == act_struct and cached_n >= cfg.n_tokens:
            X = cached["X"][: cfg.n_tokens]
            print(f"[harvest] reusing cached activations from {act_path}: cached={cached_n} requested={cfg.n_tokens} shape={tuple(X.shape)}", flush=True)
        else:
            print(f"[harvest] cache mismatch (struct equal={cached_struct == act_struct}, cached_n={cached_n} < {cfg.n_tokens}); re-harvesting", flush=True)
    if X is None:
        X = harvest_activations(cfg, device)
        torch.save({"X": X, "sig": {**act_struct, "n_tokens": int(X.shape[0])}}, act_path)
    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = ((X - mu) / max(sigma, 1e-6)).to(device)
    var = float(X_n.var().item())
    D = X_n.shape[1]

    print("\n[vanilla] training", flush=True)
    vanilla = VanillaSAE(D, cfg.n_features, cfg.top_k).to(device)
    t_v = train_sae(vanilla, X_n, cfg, "vanilla", out_dir / "vanilla_ckpt.pt", is_curve=False)

    print("\n[curve] training (gamfit REML each batch, GPU)", flush=True)
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=D, n_features=cfg.n_features, n_basis=cfg.sae_n_basis,
        top_k=cfg.top_k, intrinsic_rank=cfg.sae_intrinsic_rank,
        sparsity_weight=cfg.sae_sparsity_weight, ortho_weight=cfg.sae_ortho_weight,
        encoder_type="linear", continuous_amp=True,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    t_c = train_sae(curve, X_n, cfg, "curve", out_dir / "curve_ckpt.pt", is_curve=True, sae_cfg=sae_cfg)

    print("\n[eval]", flush=True)
    eval_n = min(cfg.eval_n, X_n.shape[0])
    eval_batch = X_n[:eval_n]

    # Vanilla SAE: GPU, batch the whole eval at once.
    with torch.no_grad():
        recon_v, z_v = vanilla(eval_batch)
        mse_v = float(((recon_v - eval_batch) ** 2).mean())
        alive_v = int((z_v > 0).any(0).sum())

    # Curve SAE: gamfit refuses to densify F·B·K above ~300 MiB. Chunk the
    # eval forward into batches matching cfg.batch_size_curve and average MSE
    # token-wise. Alive features are tracked via OR across chunks.
    sq_sum = 0.0
    n_tok = 0
    alive_mask = torch.zeros(cfg.n_features, dtype=torch.bool, device=device)
    with torch.no_grad():
        for start in range(0, eval_n, cfg.batch_size_curve):
            chunk = eval_batch[start : start + cfg.batch_size_curve]
            out_c = curve(chunk)
            sq_sum += float(((out_c.reconstruction - chunk) ** 2).sum())
            n_tok += chunk.numel()
            alive_mask |= (out_c.amplitudes > 1e-3).any(0)
    mse_c = sq_sum / max(n_tok, 1)
    alive_c = int(alive_mask.sum())
    print(f"vanilla: MSE={mse_v:.4f}  expl={1-mse_v/var:.3f}  alive={alive_v}/{cfg.n_features}  train_s={t_v:.0f}", flush=True)
    print(f"curve  : MSE={mse_c:.4f}  expl={1-mse_c/var:.3f}  alive={alive_c}/{cfg.n_features}  train_s={t_c:.0f}", flush=True)

    # Lock-and-cache snapshot: also chunk-safe. Use the first batch_size_curve
    # tokens as the snapshot reference set (a held-out sample of representative
    # data is what gamfit's REML needs).
    print("\n[snapshot] lock-and-cache the curve SAE", flush=True)
    snapshot_batch = X_n[: cfg.batch_size_curve]
    curve.update_snapshot(snapshot_batch)
    curve.inference_mode = True

    sq_sum_inf = 0.0
    n_tok_inf = 0
    with torch.no_grad():
        for start in range(0, eval_n, cfg.batch_size_curve):
            chunk = eval_batch[start : start + cfg.batch_size_curve]
            out_inf = curve(chunk)
            sq_sum_inf += float(((out_inf.reconstruction - chunk) ** 2).sum())
            n_tok_inf += chunk.numel()
    mse_inf = sq_sum_inf / max(n_tok_inf, 1)
    print(f"locked snapshot recon MSE={mse_inf:.4f}", flush=True)

    report = {
        "config": asdict(cfg),
        "var": var,
        "vanilla": {"mse": mse_v, "explained": 1 - mse_v / var, "alive": alive_v, "train_seconds": t_v},
        "curve": {"mse": mse_c, "explained": 1 - mse_c / var, "alive": alive_c, "train_seconds": t_c, "locked_mse": mse_inf},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out_dir / 'results.json'}", flush=True)
    print(json.dumps(report, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
