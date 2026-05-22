"""AxBench-style representation-steering scaffold.

Wu et al. 2025 (*AxBench: Benchmarking Representation Steering*) measure
steering methods on three axes per concept:

  CONCEPT score   — does the steered output express the target concept more?
  NATURAL score   — does the output stay coherent (low perplexity under
                     the unmodified LM)?
  OFF-TARGET      — does steering for concept C avoid disturbing tokens
                     unrelated to C?

This is a minimal scaffold that runs the same protocol on a custom
concept (magnitude). The structure is general — swap in any concept
set + classifier and it ports to other AxBench tasks.

Three steering methods compared:

  1. NO-OP                       baseline (output without steering)
  2. MANIFOLD-SAE atom-position  modify atom k's t_k from its baseline
                                  value to a sweep value
  3. LINEAR ATOM DIRECTION       add `α · W_k` to the residual (the
                                  standard direction-style steering
                                  baseline; what vanilla SAEs do)

Per (prompt, steering_method, magnitude): record output distribution +
perplexity + concept-bucket score.

If MANIFOLD-SAE atom-position steering wins on (CONCEPT score, NATURAL
score) trade-off relative to LINEAR steering at matched concept shift,
that's the architectural advantage shown by Wurgaft et al. 2025's
manifold-steering result, native to our architecture.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


# Concept catalogue. Each (concept, magnitude_label) pair has a prompt
# template and a "target token bucket" — a set of token-IDs we'd
# expect the steered output to land in. For magnitude we use the
# digit tokens themselves.
CONCEPT_TASKS = {
    "magnitude_increase": {
        "templates": [
            "There were {N} apples in the basket.",
            "She counted {N} items in total.",
            "It cost about {N} dollars.",
        ],
        "source_values": [1, 2, 5, 10],
        "target_values": [100, 200, 500, 1000],
    },
}


@dataclass
class Config:
    checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "/home/athuser/gnome_home/manifold_sae/runs/llm_sweep_q15b_L18/curve_F128.pt",
    )
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    concept_task: str = os.environ.get("MSAE_AXBENCH_TASK", "magnitude_increase")
    # Steering sweep — alpha for direction-style; t_new for atom-position
    direction_alphas: tuple[float, ...] = field(default_factory=lambda: (-2.0, -1.0, 0.0, 1.0, 2.0))
    atom_t_news: tuple[float, ...] = field(default_factory=lambda: (0.05, 0.25, 0.5, 0.75, 0.95))
    n_continuation_tokens: int = 8
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/AXBENCH_LITE",
    )
    seed: int = 0


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def load_curve_sae(path: Path, D: int, device: torch.device):
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    cfg = ManifoldSAEConfig(
        input_dim=D, n_features=sig["F"], n_basis=sig.get("n_basis", 10),
        top_k=sig["top_k"], intrinsic_rank=sig.get("intrinsic_rank", 2),
        encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(cfg).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.inference_mode = bool(sae.has_snapshot.item())
    return sae


def magnitude_bucket(token_str: str) -> int:
    """Bucket a generated token by magnitude tier. -1 = not a number."""
    try:
        n = int(token_str.strip().replace(",", ""))
    except ValueError:
        return -1
    return min(int(np.log10(max(n, 1)) * 2), 6)  # 0..6


def evaluate_steering_method(model, blocks, tok, sae, sae_layer, prompts,
                              method: str, magnitude_param, atom: int,
                              n_continuation: int, device) -> list[dict]:
    """Run a steering protocol on each prompt; return per-prompt outputs.

    method: 'noop' | 'atom_t' | 'direction'
    magnitude_param: t_new (atom_t) or alpha (direction); ignored for noop
    """
    patch = {"r_new": None}
    saved_block = blocks[sae_layer]
    orig_forward = saved_block.forward
    def patched(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if patch["r_new"] is not None:
            h.data[0, -1, :] = patch["r_new"].to(h.dtype).data
        return out
    saved_block.forward = patched
    out_records = []
    try:
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(device)
            # 1. Capture baseline residual + logits.
            patch["r_new"] = None
            cap = {}
            hh = saved_block.register_forward_hook(
                lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
            )
            with torch.no_grad():
                out_base = model(**inputs)
            hh.remove()
            r_L = cap["h"][0, -1, :]
            logits_base = out_base.logits[0, -1, :]
            probs_base = torch.softmax(logits_base, dim=-1)
            base_top = int(logits_base.argmax().item())
            base_token = tok.decode(base_top)

            # 2. Compute the steered residual under the chosen method.
            if method == "noop":
                r_L_new = r_L
            elif method == "atom_t":
                # Manifold-SAE atom-position steering: modify atom's t.
                import gamfit.torch as gt
                with torch.no_grad():
                    sae_out = sae(r_L.unsqueeze(0))
                positions = sae_out.positions[0]
                amp = sae_out.amplitudes[0]
                # If amp is zero, the atom is silent — skip; recon won't change.
                if amp[atom] < 1e-6:
                    r_L_new = r_L
                else:
                    t_orig = positions[atom].to(torch.float64)
                    t_new = torch.tensor([float(magnitude_param)], dtype=torch.float64, device=device)
                    centers = sae.centers.to(device)
                    phi_orig = gt.duchon_basis_1d(t_orig.unsqueeze(0), centers, m=2, periodic=False)
                    phi_new = gt.duchon_basis_1d(t_new, centers, m=2, periodic=False)
                    B_k = sae.B_locked[atom].to(device)
                    dir_k = sae.directions[atom].to(device)
                    g_orig = (phi_orig @ B_k).to(dir_k.dtype) @ dir_k.t()
                    g_new = (phi_new @ B_k).to(dir_k.dtype) @ dir_k.t()
                    delta = amp[atom] * (g_new - g_orig).squeeze(0)
                    r_L_new = r_L + delta.to(r_L.dtype)
            elif method == "direction":
                # Standard direction-style steering: add α · W_k to residual.
                dir_k = sae.directions[atom, :, 0].to(device).to(r_L.dtype)  # primary direction
                dir_k = dir_k / (dir_k.norm() + 1e-9)
                r_L_new = r_L + float(magnitude_param) * dir_k
            else:
                raise ValueError(f"unknown method {method}")

            # 3. Run forward with the steered residual.
            patch["r_new"] = r_L_new
            with torch.no_grad():
                out_steered = model(**inputs)
            patch["r_new"] = None
            logits_steered = out_steered.logits[0, -1, :]
            probs_steered = torch.softmax(logits_steered, dim=-1)
            kl = float(torch.nn.functional.kl_div(probs_steered.log(), probs_base, reduction="sum"))
            steered_top = int(logits_steered.argmax().item())
            steered_token = tok.decode(steered_top)

            # Naturalness: KL to unperturbed AT this token (lower = more natural)
            out_records.append({
                "prompt": prompt[:40],
                "method": method,
                "magnitude_param": float(magnitude_param),
                "atom": int(atom),
                "kl_divergence": kl,
                "base_top_token": base_token,
                "steered_top_token": steered_token,
                "base_top_bucket": magnitude_bucket(base_token),
                "steered_top_bucket": magnitude_bucket(steered_token),
                "bucket_shift": magnitude_bucket(steered_token) - magnitude_bucket(base_token),
            })
    finally:
        saved_block.forward = orig_forward
    return out_records


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[axbench] device={device} output_dir={out_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
    D = model.config.hidden_size

    sae = load_curve_sae(Path(cfg.checkpoint), D, device)
    print(f"[axbench] loaded SAE F={sae.config.n_features}", flush=True)

    task = CONCEPT_TASKS[cfg.concept_task]
    source_prompts = [t.format(N=N) for N in task["source_values"] for t in task["templates"]]

    # Pick the most-active atom on source prompts (no a-priori knowledge).
    cap = {}
    hh = blocks[cfg.layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    amp_sums = torch.zeros(sae.config.n_features, device=device)
    with torch.no_grad():
        for p in source_prompts:
            inputs = tok(p, return_tensors="pt").to(device)
            model(**inputs)
            sae_out = sae(cap["h"][0, -1, :].unsqueeze(0))
            amp_sums += sae_out.amplitudes[0]
    hh.remove()
    target_atom = int(amp_sums.argmax().item())
    print(f"[axbench] target atom = {target_atom} (mean amp on source prompts: {amp_sums[target_atom]/len(source_prompts):.3f})", flush=True)

    # Evaluate all three methods × magnitude sweep on the source prompts.
    all_records = []
    print("[axbench] no-op baseline …", flush=True)
    all_records.extend(evaluate_steering_method(
        model, blocks, tok, sae, cfg.layer, source_prompts, "noop", 0.0, target_atom,
        cfg.n_continuation_tokens, device,
    ))
    print("[axbench] manifold-SAE atom-t sweep …", flush=True)
    for t_new in cfg.atom_t_news:
        all_records.extend(evaluate_steering_method(
            model, blocks, tok, sae, cfg.layer, source_prompts, "atom_t", t_new, target_atom,
            cfg.n_continuation_tokens, device,
        ))
    print("[axbench] linear-direction α sweep …", flush=True)
    for alpha in cfg.direction_alphas:
        all_records.extend(evaluate_steering_method(
            model, blocks, tok, sae, cfg.layer, source_prompts, "direction", alpha, target_atom,
            cfg.n_continuation_tokens, device,
        ))

    # Summarize per method × magnitude.
    def aggregate(method, param):
        rec = [r for r in all_records if r["method"] == method and abs(r["magnitude_param"] - param) < 1e-6]
        if not rec: return None
        kls = [r["kl_divergence"] for r in rec]
        bs = [r["bucket_shift"] for r in rec]
        return {
            "method": method, "magnitude_param": param,
            "n_prompts": len(rec),
            "mean_kl": float(np.mean(kls)),
            "mean_bucket_shift": float(np.mean(bs)),
            "median_bucket_shift": float(np.median(bs)),
        }
    summary = []
    summary.append(aggregate("noop", 0.0))
    for t in cfg.atom_t_news: summary.append(aggregate("atom_t", t))
    for a in cfg.direction_alphas: summary.append(aggregate("direction", a))
    summary = [s for s in summary if s]
    print("\n=== Per (method, magnitude_param) summary ===", flush=True)
    print(f"{'method':<12} {'param':<8} {'mean KL':<10} {'mean bucket shift':<18}", flush=True)
    for s in summary:
        print(f"  {s['method']:<10} {s['magnitude_param']:<+8.2f} {s['mean_kl']:<10.4f} {s['mean_bucket_shift']:<+.3f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg),
        "target_atom": target_atom,
        "summary": summary,
        "all_records": all_records,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
