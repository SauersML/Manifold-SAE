"""Train a Manifold-SAE on real LLM residual-stream activations.

Validates the LLM-applicability hypothesis: persistent-curve atoms,
feedforward encoder, scales to D=768 with thousands of features.

Pipeline
--------
1. Load gpt2-small via transformers; freeze.
2. Stream a small text corpus through it, capture residual-stream
   activations at a chosen layer.
3. Train ``ManifoldSAE`` (linear encoder) on the captured activations.
4. Report reconstruction MSE, sparsity, feature liveness; sketch a
   couple of learned curves by walking position [0, 1] for the most
   active features.

This is end-to-end on MPS / CPU — no need for cluster scale to verify
that the architecture trains and produces sensible curves.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    model_name: str = "gpt2"
    layer_index: int = 6
    n_tokens: int = 100_000
    seq_len: int = 128
    sae_features: int = 1024
    n_basis: int = 8
    top_k: int = 32
    intrinsic_rank: int = 4
    n_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    sparsity_weight: float = 1e-3
    ortho_weight: float = 1e-3
    reml_weight: float = 1.0
    seed: int = 0
    output_dir: str = "runs/LLM_CURVE_SAE"
    # corpus: built-in tiny dummy by default; override if you want real text
    corpus_lines: int = 4000


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_corpus(n_lines: int) -> list[str]:
    """Tiny built-in corpus so the experiment runs without HF Hub access.

    Procedurally generated sentences over a small structured vocabulary
    — enough lexical/semantic variety that gpt2 produces non-degenerate
    activations, while staying offline.
    """
    rng = np.random.default_rng(0)
    subjects = ["The cat", "A dog", "The teacher", "An engineer", "She", "He", "The child", "My friend", "The doctor", "A scientist"]
    verbs = ["sat on", "ran past", "wrote about", "looked at", "discovered", "questioned", "explained", "compared", "modified", "ignored"]
    objects = ["the bench", "a red apple", "the experiment", "two ideas", "the broken clock", "fresh bread", "a heavy book", "the empty room", "her notes", "the dark window"]
    adverbs = ["quickly", "carefully", "silently", "with great surprise", "in the rain", "yesterday", "for hours", "thoughtfully", "for the first time", "as if puzzled"]
    out: list[str] = []
    for _ in range(n_lines):
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(adverbs)}."
        out.append(s)
    return out


def _capture_activations(cfg: Config, device: torch.device) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    print(f"[capture] loading {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name, output_hidden_states=True)
    model.eval().to(device)

    text = _load_corpus(cfg.corpus_lines)
    print(f"[capture] corpus lines: {len(text)}")

    chunks: list[torch.Tensor] = []
    total = 0
    BATCH = 16
    with torch.no_grad():
        for start in range(0, len(text), BATCH):
            batch_text = text[start : start + BATCH]
            enc = tok(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=cfg.seq_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            # hidden_states is a tuple of length n_layers+1; pick layer_index
            hs = outputs.hidden_states[cfg.layer_index]  # (B, T, D)
            mask = enc["attention_mask"].bool()  # (B, T)
            flat = hs[mask]  # (n_real_tokens, D)
            chunks.append(flat.detach().cpu())
            total += flat.shape[0]
            if total >= cfg.n_tokens:
                break
    acts = torch.cat(chunks, dim=0)
    if acts.shape[0] > cfg.n_tokens:
        acts = acts[: cfg.n_tokens]
    print(f"[capture] activations: {tuple(acts.shape)}, dtype={acts.dtype}")
    return acts


def main(cfg: Config = Config()) -> int:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    print(f"[setup] device={device}")

    acts = _capture_activations(cfg, device).to(torch.float32)
    # Center & scale so SAE sees normalized residuals
    mu = acts.mean(dim=0, keepdim=True)
    acts_c = acts - mu
    scale = acts_c.std().item()
    acts_n = acts_c / max(scale, 1e-6)
    print(f"[setup] residual D={acts_n.shape[1]}, scale={scale:.4f}")

    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    from manifold_sae.losses import total_loss

    sae_cfg = ManifoldSAEConfig(
        input_dim=acts_n.shape[1],
        n_features=cfg.sae_features,
        n_basis=cfg.n_basis,
        top_k=cfg.top_k,
        intrinsic_rank=cfg.intrinsic_rank,
        sparsity_weight=cfg.sparsity_weight,
        ortho_weight=cfg.ortho_weight,
        reml_weight=cfg.reml_weight,
        encoder_type="linear",
    )
    print(f"[setup] SAE config: F={cfg.sae_features} R={cfg.intrinsic_rank} K={cfg.n_basis} topk={cfg.top_k}")
    sae = ManifoldSAE(sae_cfg).to(device)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"[setup] SAE params: {n_params/1e6:.2f}M")

    optim = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    ds = TensorDataset(acts_n)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    history = {"step": [], "mse": [], "sparsity": [], "frac_alive": []}
    t0 = time.time()
    step = 0
    it = iter(loader)
    while step < cfg.n_steps:
        try:
            (batch,) = next(it)
        except StopIteration:
            it = iter(loader)
            (batch,) = next(it)
        batch = batch.to(device)
        optim.zero_grad(set_to_none=True)
        out = sae(batch)
        losses = total_loss(out, batch, sae_cfg)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optim.step()
        if step % max(cfg.n_steps // 20, 1) == 0 or step == cfg.n_steps - 1:
            with torch.no_grad():
                alive = (out.amplitudes > 0.5).any(dim=0).float().mean().item()
            history["step"].append(step)
            history["mse"].append(float(losses["mse"].item()))
            history["sparsity"].append(float(losses["sparsity"].item()))
            history["frac_alive"].append(alive)
            print(f"[step {step:5d}] mse={losses['mse'].item():.4e}  spars={losses['sparsity'].item():.3f}  alive={alive:.3f}")
        step += 1
    train_seconds = time.time() - t0
    print(f"[train] {train_seconds:.1f}s")

    sae.eval()
    with torch.no_grad():
        eval_batch = acts_n[: min(4096, acts_n.shape[0])].to(device)
        out = sae(eval_batch)
        mse_eval = float(torch.mean((out.reconstruction - eval_batch) ** 2).item())
        var_eval = float(eval_batch.var().item())
        explained = 1.0 - mse_eval / max(var_eval, 1e-12)
        fire_freq = (out.amplitudes > 0.5).float().mean(dim=0).cpu().numpy()  # (F,)
        alive = (fire_freq > 1e-4).sum().item()
        print(f"[eval] MSE={mse_eval:.4e} var={var_eval:.4e} frac-explained={explained:.4f}")
        print(f"[eval] alive features: {alive}/{cfg.sae_features}")

        # Probe top-K most active features: walk position over [0,1] and
        # report the curve's ambient-space trajectory's variance (a proxy
        # for whether the feature genuinely encodes a continuous family).
        top_idx = np.argsort(-fire_freq)[:8]
        curve_extents = []
        for k in top_idx:
            import gamfit.torch as gt
            t = torch.linspace(0.05, 0.95, 32, dtype=torch.float64, device=device)
            phi = gt.duchon_basis_1d(t, sae.centers, m=2, periodic=False)
            B_k = sae.coeff[k].to(torch.float64)
            W_k = sae.directions[k].to(torch.float64)
            curve = phi @ B_k @ W_k.T  # (T, D)
            extent = float(curve.std(dim=0).norm().item())
            curve_extents.append({"feature": int(k), "fire_freq": float(fire_freq[k]), "curve_extent": extent})
        print(f"[eval] top-active feature curve extents:")
        for r in curve_extents:
            print(f"  feat {r['feature']:4d}  fire={r['fire_freq']:.3f}  extent={r['curve_extent']:.3f}")

    report = {
        "config": asdict(cfg),
        "train_seconds": train_seconds,
        "n_params_M": n_params / 1e6,
        "mse_eval": mse_eval,
        "var_eval": var_eval,
        "frac_explained": explained,
        "alive_features": alive,
        "n_features": cfg.sae_features,
        "top_curves": curve_extents,
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[done] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
