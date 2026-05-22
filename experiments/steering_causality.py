"""Causal steering test: do Manifold-SAE atoms ACTUALLY encode their concept?

A correlation between `t_k` and a planted concept (the `llm_probe`
metric) is suggestive but not causal. To test causality we **patch the
LM's residual stream** with a modified SAE reconstruction and measure
whether the LM's downstream output shifts in the predicted direction.

Procedure
=========
For a chosen concept (default: magnitude N), a trained Manifold-SAE
checkpoint, and a prompt that primes the concept:

  1. Forward Qwen up to layer L, harvest residual r_L for the
     answer-position token.
  2. Apply SAE encoder to r_L → (positions, amplitudes).
  3. For atom k whose `t_k` correlates with the concept (identified
     from `llm_probe` results), modify ONLY that atom's position:
     `t_k → t_k_new`. Leave amplitudes and other atoms untouched.
  4. Re-run SAE decoder with the modified position → reconstruction r'_L.
  5. Patch r_L_patched = r_L + (r'_L − r_L) — replaces the modified
     atom's contribution without touching other features.
  6. Continue Qwen forward from layer L+1 with r_L_patched.
  7. Compare top-k logits / probabilities at the answer position.

Reports per (prompt × t_k_new) pair:
  * KL divergence between original and patched output distributions
  * Top-5 tokens at the answer position
  * Predicted-magnitude shift if the concept is magnitude

A causally-effective atom should produce systematic, monotone output
shifts as `t_k` is swept. A purely correlative atom produces noisy
shifts that don't track the t-axis.

This is the qualitative test that the Goodfire et al. manifold-steering
paper does with cubic-spline-fit centroid manifolds — but here we use
the Manifold-SAE atom's native `g_k(t)` as the manifold parameterization.
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

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
from manifold_sae._cluster_bridge import require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class SteerConfig:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-0.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "12"))
    # Path to a trained Manifold-SAE checkpoint.
    sae_checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "/content/runs/LLM_SWEEP/curve_F256.pt",
    )
    # Which atom (and its concept axis) to steer. Read from a probe
    # results.json if available, else specified by hand.
    probe_results_path: str | None = os.environ.get("MSAE_PROBE_RESULTS")
    target_concept: str = os.environ.get("MSAE_CONCEPT", "magnitude")
    target_atom: int = int(os.environ.get("MSAE_ATOM", "-1"))   # -1 = pick from probe

    # Prompts to steer (one per line in the env var, default below)
    prompts: tuple[str, ...] = field(default_factory=lambda: (
        "There were N apples in the basket. N was equal to ",
        "She counted N items. N is the number ",
        "The total was N. Specifically, ",
    ))
    t_sweep: tuple[float, ...] = field(default_factory=lambda: (0.05, 0.2, 0.5, 0.8, 0.95))
    top_k_logits: int = 10

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/STEERING",
    )
    seed: int = 0


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def load_curve_sae(checkpoint_path: Path, D: int, device: torch.device):
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return sae, cfg


def pick_atom_from_probe(probe_path: Path, concept: str) -> int:
    """Pick the atom with the highest position-vs-concept Spearman from a
    completed llm_probe run. Falls back to atom 0 if not findable.
    """
    if not probe_path.exists():
        return 0
    d = json.loads(probe_path.read_text())
    phase2 = d.get("phase2", {}) or {}
    best_atom = 0
    best_score = -1.0
    for key, r in phase2.items():
        if not isinstance(r, dict): continue
        if not key.startswith(concept):
            continue
        cp = r.get("curve_position") or {}
        if cp.get("best") and cp["best"] > best_score:
            best_score = cp["best"]
            best_atom = cp.get("best_atom_idx", 0)
    return int(best_atom)


def encode_then_decode_modified(sae, r_L: torch.Tensor, atom: int, t_new: float, device):
    """Run SAE on r_L, then modify atom `atom`'s t to `t_new` and re-decode."""
    from manifold_sae.sae import _soft_rescale_positions
    import gamfit.torch as gt

    sae_in = r_L.unsqueeze(0).to(device)               # (1, D)
    x_centered = sae_in - sae.b_dec
    y_proj = torch.einsum("bd,fdr->bfr", x_centered, sae.directions)
    z_raw, mask_soft, mask_binary = sae.encoder(x_centered, y_proj)

    if sae.inference_mode and bool(sae.has_snapshot.item()):
        positions, _, _ = _soft_rescale_positions(
            z_raw,
            frozen_min=sae.soft_min_locked.to(z_raw.dtype),
            frozen_max=sae.soft_max_locked.to(z_raw.dtype),
        )
    else:
        positions, _, _ = _soft_rescale_positions(z_raw)

    # Reconstruction at the ORIGINAL positions (for delta computation)
    F = sae.config.n_features
    B = 1
    t_flat_orig = positions.t().contiguous().view(-1).to(torch.float64)
    phi_orig = gt.duchon_basis_1d(t_flat_orig, sae.centers, m=2,
                                   periodic=sae.config.periodic).view(F, B, -1)
    g_orig = torch.einsum("fbk,fkr->fbr", phi_orig, sae.B_locked).to(r_L.dtype)
    contrib_orig = torch.einsum("fbr,fdr->bfd", g_orig * mask_binary.t().unsqueeze(-1), sae.directions)
    recon_orig = contrib_orig.sum(dim=1) + sae.b_dec.unsqueeze(0)

    # MODIFIED positions: set atom's t to t_new
    positions_mod = positions.clone()
    positions_mod[0, atom] = float(t_new)
    t_flat_mod = positions_mod.t().contiguous().view(-1).to(torch.float64)
    phi_mod = gt.duchon_basis_1d(t_flat_mod, sae.centers, m=2,
                                  periodic=sae.config.periodic).view(F, B, -1)
    g_mod = torch.einsum("fbk,fkr->fbr", phi_mod, sae.B_locked).to(r_L.dtype)
    contrib_mod = torch.einsum("fbr,fdr->bfd", g_mod * mask_binary.t().unsqueeze(-1), sae.directions)
    recon_mod = contrib_mod.sum(dim=1) + sae.b_dec.unsqueeze(0)

    return recon_orig.squeeze(0), recon_mod.squeeze(0), float(positions[0, atom])


def run_steering(cfg: SteerConfig, device: torch.device) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
    D = model.config.hidden_size

    sae, _ = load_curve_sae(Path(cfg.sae_checkpoint), D, device)
    print(f"[steer] loaded SAE F={sae.config.n_features} top_k={sae.config.top_k}", flush=True)

    # Pick atom BY ACTIVITY on the test prompts (more robust than the old
    # path of loading from a stale probe file). The atom we want to steer
    # is the one with highest mean amplitude across all `cfg.prompts`.
    if cfg.target_atom < 0:
        amp_sums = torch.zeros(sae.config.n_features, device=device)
        cap = {}
        hh = blocks[cfg.layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
        )
        with torch.no_grad():
            for prompt in cfg.prompts:
                inputs = tok(prompt, return_tensors="pt").to(device)
                model(**inputs)
                r = cap["h"][0, -1, :]
                out = sae(r.unsqueeze(0))
                amp_sums += out.amplitudes[0]
        hh.remove()
        atom = int(amp_sums.argmax().item())
        print(f"[steer] picked atom={atom} by activity (mean amp on prompts: {amp_sums[atom].item()/len(cfg.prompts):.3f})", flush=True)
    else:
        atom = cfg.target_atom
        print(f"[steer] using explicit target_atom={atom}", flush=True)

    captured: dict = {}
    saved_block = blocks[cfg.layer]
    orig_forward = saved_block.forward
    patch_value = {"r_new": None}
    seen_count = {"n": 0}

    def patched_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if patch_value["r_new"] is not None:
            r_new = patch_value["r_new"]
            # In-place modification of the original tensor — return-value
            # replacement isn't reliable when this block was wrapped by an
            # outer container that captured `out` before our patched
            # forward returned. In-place edit guarantees downstream layers
            # see the patched residual.
            h.data[0, -1, :] = r_new.to(h.dtype).data
        captured["h"] = h.detach()
        return out  # let downstream see the (now-mutated) original tensor

    saved_block.forward = patched_forward

    results: list[dict] = []
    torch.set_grad_enabled(False)
    try:
        for prompt in cfg.prompts:
            inputs = tok(prompt, return_tensors="pt").to(device)
            # Baseline forward
            patch_value["r_new"] = None
            out_base = model(**inputs)
            r_L_base = captured["h"][0, -1, :].clone()
            logits_base = out_base.logits[0, -1, :]
            probs_base = torch.softmax(logits_base, dim=-1)

            # Diagnostic: amplitude of selected atom on this prompt. If 0,
            # the patch is a no-op and KL will be 0.
            sae_in = r_L_base.unsqueeze(0).to(device)
            x_centered = sae_in - sae.b_dec
            y_proj = torch.einsum("bd,fdr->bfr", x_centered, sae.directions)
            _z_raw, _ms, mb = sae.encoder(x_centered, y_proj)
            atom_amp = float(mb[0, atom].abs().item())
            print(f"  [diag] prompt='{prompt[:25]}...' atom={atom} amp={atom_amp:.4f}",
                  flush=True)

            for t_new in cfg.t_sweep:
                # Compute the modified reconstruction
                _, recon_mod, t_orig = encode_then_decode_modified(sae, r_L_base, atom, t_new, device)
                # The patched r is: original residual minus original SAE recon + modified SAE recon.
                # Equivalent: patch r_L_base + (recon_mod - recon_orig). But our encode_then_decode
                # already does the comparison; for safety we compute it explicitly here.
                # We patch with the modified reconstruction directly — effectively replacing the
                # SAE-explained part. But that's destructive. Better: patch by SHIFT.
                _, recon_orig, _ = encode_then_decode_modified(sae, r_L_base, atom, t_orig, device)
                shift = recon_mod - recon_orig
                shift_norm = float(shift.norm().item())
                r_L_patched = r_L_base + shift

                # Re-run forward with patch
                patch_value["r_new"] = r_L_patched
                out_patched = model(**inputs)
                logits_patched = out_patched.logits[0, -1, :]
                probs_patched = torch.softmax(logits_patched, dim=-1)

                kl = torch.nn.functional.kl_div(probs_patched.log(), probs_base, reduction="sum").item()
                # Top-k tokens by patched probability
                top = torch.topk(probs_patched, cfg.top_k_logits)
                top_tokens = [(tok.decode(idx), float(p)) for idx, p in zip(top.indices, top.values)]
                results.append({
                    "prompt": prompt,
                    "atom": atom,
                    "atom_amp_on_prompt": atom_amp,
                    "t_original": t_orig,
                    "t_new": t_new,
                    "patch_shift_norm": shift_norm,
                    "kl_divergence": kl,
                    "top_tokens": top_tokens,
                })
                print(f"  prompt='{prompt[:30]}...' t={t_orig:.3f}->{t_new:.3f} "
                      f"KL={kl:.4f} top1='{top_tokens[0][0]}' p={top_tokens[0][1]:.3f}",
                      flush=True)
                patch_value["r_new"] = None
    finally:
        saved_block.forward = orig_forward
    torch.set_grad_enabled(True)
    return {"atom": atom, "results": results, "config": asdict(cfg)}


def main() -> int:
    cfg = SteerConfig()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} sae_checkpoint={cfg.sae_checkpoint}", flush=True)

    if not Path(cfg.sae_checkpoint).exists():
        print(f"[error] SAE checkpoint not found: {cfg.sae_checkpoint}", flush=True)
        return 1

    report = run_steering(cfg, device)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[done] wrote {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
