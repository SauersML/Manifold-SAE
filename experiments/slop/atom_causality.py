"""Counterfactual atom ablation + cross-SAE alignment — two CREATIVE
benchmarks of the architecture's claims.

Benchmark A — Counterfactual atom ablation
==========================================
For each "concept-encoding" atom (identified by holdout test from
`llm_probe`), find prompts that strongly activate it. Zero out the
atom's contribution to the SAE reconstruction, patch the modified
residual back into the LM, measure the resulting shift in the output
distribution.

Stronger than correlational interpretability: tests whether the atom
is CAUSALLY LOAD-BEARING for its concept. If ablating a magnitude
atom on "There were 500 apples." causes the LM to predict a generic
number rather than the right magnitude, the atom is necessary, not
just correlated.

Reports per-atom: average KL between unablated and ablated outputs +
top-token shifts. Compares to ablating a RANDOM atom of similar
firing rate as a control.

Benchmark B — Cross-SAE alignment
=================================
Two SAEs trained from different random seeds on the same activations
should — IF the architecture is identifying universal features —
find the same concepts in their atom dictionaries. Hungarian-match
their atoms by direction cosine similarity in residual stream; report
the matched-pair correlation distribution and the top-firing-tokens
overlap.

If the matching is near-identity (high similarity, same top tokens),
Manifold-SAE atoms are universal. If random, they're optimization
artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "<repo_root>/runs/llm_sweep/curve_F256.pt",
    )
    # Optional second checkpoint for cross-SAE alignment. If empty, we
    # train one in-job from a fresh seed.
    checkpoint_2: str = os.environ.get("MSAE_CHECKPOINT_2", "")
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-0.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "12"))
    # Ablation: which prompts to test on
    ablation_n_prompts: int = 30
    # Cross-SAE: how many tokens to fit the second SAE from
    cross_sae_n_tokens: int = 30_000
    cross_sae_steps: int = 2000

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/ATOM_CAUSALITY",
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
    from manifold_sae.sae import load_sae
    return load_sae(path, input_dim=D, device=device)


# ---------------------------------------------------------------------------
# Benchmark A — Counterfactual atom ablation
# ---------------------------------------------------------------------------


def magnitude_prompts(n: int = 30) -> tuple[list[str], list[int]]:
    """Prompts that prime magnitude. Returns (prompts, magnitudes)."""
    templates = [
        "There were {N} apples in the basket.",
        "She counted {N} items.",
        "The total was {N} dollars.",
        "He had {N} marbles in his pocket.",
        "We saw {N} birds in the sky.",
    ]
    mags = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    prompts, labels = [], []
    for N in mags:
        for t in templates[: max(1, n // len(mags))]:
            prompts.append(t.format(N=N))
            labels.append(N)
    return prompts[:n], labels[:n]


def ablate_atom(sae, model, blocks, layer: int, tok, prompts: list[str],
                 ablate_atom_k: int | None, device) -> list[dict]:
    """For each prompt, run the LM with atom `ablate_atom_k` zeroed in the
    SAE reconstruction. Patch the modified residual at `layer`.
    Returns per-prompt outputs.
    """
    patch_value = {"r_new": None}
    saved_block = blocks[layer]
    orig_forward = saved_block.forward
    def patched(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if patch_value["r_new"] is not None:
            h.data[0, -1, :] = patch_value["r_new"].to(h.dtype).data
        return out
    saved_block.forward = patched
    results = []
    try:
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(device)
            # 1) Baseline: no patch.
            patch_value["r_new"] = None
            with torch.no_grad():
                out_base = model(**inputs)
            logits_base = out_base.logits[0, -1, :]
            probs_base = torch.softmax(logits_base, dim=-1)
            # 2) Get the SAE recon at this prompt's residual.
            with torch.no_grad():
                # Need a hook to grab residual; re-run with hook
                cap = {}
                hh = saved_block.register_forward_hook(
                    lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).clone())
                )
                model(**inputs)
                hh.remove()
                r_L = cap["h"][0, -1, :].clone()
                sae_out = sae(r_L.unsqueeze(0).to(dtype=sae.cfg.dtype))
                # Decompose contribution per atom.
                amp = sae_out.amplitudes[0]                          # (F,)
                # 3) Compute the original SAE reconstruction at this residual.
                recon_orig = sae_out.x_hat[0]
                # 4) Build the ABLATED reconstruction: subtract atom k's exact
                #    ambient contribution. Cutover: the decoder block lives in
                #    ambient R^D, so atom k's contribution at this token is
                #    z_k * (curves[k] @ decoder_blocks[k]) — read straight off
                #    the output bundle (no `directions`, no separate basis call).
                if ablate_atom_k is not None and amp[ablate_atom_k] > 1e-6:
                    z_k = sae_out.z[0, ablate_atom_k]
                    curve_k = sae_out.curves[0, ablate_atom_k]        # (K,)
                    block_k = sae.decoder_blocks[ablate_atom_k]       # (K, D)
                    contrib_k = z_k * (curve_k @ block_k)             # (D,)
                    delta = -contrib_k                                # subtract
                    r_L_ablated = r_L + delta.to(r_L.dtype)
                else:
                    r_L_ablated = r_L.clone()  # no-op
                # 5) Patch the modified residual.
                patch_value["r_new"] = r_L_ablated
                out_abl = model(**inputs)
                logits_abl = out_abl.logits[0, -1, :]
                probs_abl = torch.softmax(logits_abl, dim=-1)
                patch_value["r_new"] = None

                kl = float(torch.nn.functional.kl_div(probs_abl.log(), probs_base, reduction="sum").item())
                top_base = torch.topk(probs_base, 5)
                top_abl = torch.topk(probs_abl, 5)
                results.append({
                    "prompt": prompt[:50],
                    "atom_amp_on_prompt": float(amp[ablate_atom_k].item()) if ablate_atom_k is not None else 0.0,
                    "kl_divergence": kl,
                    "top_tokens_base": [(tok.decode(int(i)), float(p))
                                          for i, p in zip(top_base.indices, top_base.values)],
                    "top_tokens_ablated": [(tok.decode(int(i)), float(p))
                                              for i, p in zip(top_abl.indices, top_abl.values)],
                })
    finally:
        saved_block.forward = orig_forward
    return results


def benchmark_ablation(sae, model, blocks, layer, tok, device, cfg: Config) -> dict:
    """Pick the atom with highest holdout |ρ| for magnitude (if available
    from a prior probe), ablate it, compare to a random-atom control."""
    prompts, _ = magnitude_prompts(cfg.ablation_n_prompts)

    # Find the atom with highest amplitude across our magnitude prompts —
    # serves as our proxy for "magnitude-encoding atom" without needing
    # a probe JSON.
    amp_sums = np.zeros(sae.cfg.n_atoms)
    with torch.no_grad():
        cap = {}
        hh = blocks[layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
        )
        try:
            for p in prompts:
                inputs = tok(p, return_tensors="pt").to(device)
                model(**inputs)
                r = cap["h"][0, -1, :]
                out = sae(r.unsqueeze(0).to(dtype=sae.cfg.dtype))
                amp_sums += out.amplitudes[0].cpu().numpy()
        finally:
            hh.remove()
    target_atom = int(np.argmax(amp_sums))
    # Random control: an atom with similar firing rate but a random pick.
    candidates = [k for k in range(sae.cfg.n_atoms) if amp_sums[k] > 0.1 * amp_sums[target_atom]]
    rng = np.random.default_rng(cfg.seed)
    control_atom = int(rng.choice([k for k in candidates if k != target_atom]
                                    or [target_atom]))
    print(f"[ablation] target atom = {target_atom} (amp_sum {amp_sums[target_atom]:.2f})", flush=True)
    print(f"[ablation] control atom = {control_atom} (amp_sum {amp_sums[control_atom]:.2f})", flush=True)

    target_results = ablate_atom(sae, model, blocks, layer, tok, prompts, target_atom, device)
    control_results = ablate_atom(sae, model, blocks, layer, tok, prompts, control_atom, device)

    target_kls = [r["kl_divergence"] for r in target_results]
    control_kls = [r["kl_divergence"] for r in control_results]
    return {
        "target_atom": target_atom,
        "control_atom": control_atom,
        "target_mean_kl": float(np.mean(target_kls)),
        "target_median_kl": float(np.median(target_kls)),
        "control_mean_kl": float(np.mean(control_kls)),
        "control_median_kl": float(np.median(control_kls)),
        "target_results": target_results[:6],         # sample for inspection
        "control_results": control_results[:6],
    }


# ---------------------------------------------------------------------------
# Benchmark B — Cross-SAE alignment
# ---------------------------------------------------------------------------


def cross_sae_alignment(sae_a, sae_b) -> dict:
    """Hungarian-match atoms across two SAEs by direction cosine
    similarity. Report distribution of matched-pair similarities.
    """
    from scipy.optimize import linear_sum_assignment

    # Cutover: no `directions`. The atom's primary ambient direction is the top
    # right-singular vector of its decoder block (K x D).
    def _primary_dirs(sae) -> np.ndarray:
        blocks = sae.decoder_blocks.detach().cpu().numpy()  # (F, K, D)
        out = np.zeros((blocks.shape[0], blocks.shape[2]), dtype=np.float64)
        for k in range(blocks.shape[0]):
            _, _, vt = np.linalg.svd(blocks[k].astype(np.float64), full_matrices=False)
            out[k] = vt[0]
        return out

    Wa_p = _primary_dirs(sae_a)                              # (F, D)
    Wb_p = _primary_dirs(sae_b)
    F = Wa_p.shape[0]
    if Wb_p.shape[0] != F:
        return {"error": "F must match between the two SAEs"}
    # Normalize for cosine similarity.
    Wa_n = Wa_p / (np.linalg.norm(Wa_p, axis=1, keepdims=True) + 1e-12)
    Wb_n = Wb_p / (np.linalg.norm(Wb_p, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(Wa_n @ Wb_n.T)                              # (F, F), |cos|

    # Hungarian: maximize sum of |cos| → minimize negative.
    row, col = linear_sum_assignment(-sim)
    matched_sims = sim[row, col]
    return {
        "n_atoms": int(F),
        "mean_matched_similarity": float(matched_sims.mean()),
        "median_matched_similarity": float(np.median(matched_sims)),
        "n_matched_above_0.5": int((matched_sims > 0.5).sum()),
        "n_matched_above_0.7": int((matched_sims > 0.7).sum()),
        "n_matched_above_0.9": int((matched_sims > 0.9).sum()),
        "matched_similarities": matched_sims.tolist(),
        "permutation_a_to_b": col.tolist(),
    }


def train_seeded_sae(cfg: Config, X: torch.Tensor, ref_sae_config, device) -> "ManifoldSAE":
    """Train a fresh Manifold-SAE from a different random seed on the
    same data the reference SAE was trained on.
    """
    import dataclasses
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE
    torch.manual_seed(cfg.seed + 1)        # different seed from reference
    # The closed-form REML solve in sae.fit requires float64; rebuild the
    # reference config at double precision for this seeded retrain.
    seed_config = dataclasses.replace(ref_sae_config, dtype=torch.float64)
    sae = ManifoldSAE(seed_config).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    X = X.to(device=device, dtype=torch.float64)
    for step in range(cfg.cross_sae_steps):
        idx = torch.randint(0, X.shape[0], (256,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, sae)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  [seed-B step {step}] mse={torch.nn.functional.mse_loss(out.x_hat, batch).item():.4e}", flush=True)
    sae.eval()
    sae.fit(X[: min(2048, X.shape[0])])
    sae.lock_snapshot()
    return sae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} output_dir={out_dir}", flush=True)

    if not Path(cfg.checkpoint).exists():
        print(f"[error] checkpoint not found: {cfg.checkpoint}", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
    D = model.config.hidden_size

    sae = load_curve_sae(Path(cfg.checkpoint), D, device)
    print(f"[setup] loaded SAE_A F={sae.cfg.n_atoms} top_k={sae.cfg.sparsity.target_k}", flush=True)

    print("\n=== Benchmark A: counterfactual atom ablation ===", flush=True)
    ablation = benchmark_ablation(sae, model, blocks, cfg.layer, tok, device, cfg)
    print(f"  target atom {ablation['target_atom']}: mean_KL={ablation['target_mean_kl']:.4f}", flush=True)
    print(f"  control atom {ablation['control_atom']}: mean_KL={ablation['control_mean_kl']:.4f}", flush=True)
    ratio = ablation['target_mean_kl'] / max(ablation['control_mean_kl'], 1e-9)
    print(f"  target/control KL ratio: {ratio:.2f}× (>1 = target atom causally matters)", flush=True)

    print("\n=== Benchmark B: cross-SAE alignment ===", flush=True)
    # Harvest a batch of residuals for cross-SAE training.
    from datasets import load_dataset
    print(f"[xsae] harvesting {cfg.cross_sae_n_tokens} tokens for fresh SAE training", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    captured = {}
    hh = blocks[cfg.layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    Xs = []
    with torch.no_grad():
        for d in ds:
            if len(Xs) >= cfg.cross_sae_n_tokens: break
            text = d.get("text", "")
            if not isinstance(text, str) or len(text) < 100: continue
            inputs = tok(text[:1500], return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**inputs)
            for i in range(min(captured["h"].shape[1], 32)):
                Xs.append(captured["h"][0, i, :].cpu())
                if len(Xs) >= cfg.cross_sae_n_tokens: break
    hh.remove()
    X = torch.stack(Xs[:cfg.cross_sae_n_tokens], dim=0)
    mu = X.mean(0, keepdim=True); sigma = X.std(0).clamp(min=1e-6)  # per-dim std (was scalar — see _normalize.py)
    X_n = (X - mu) / sigma
    print(f"  harvested {X.shape}, training seed-B SAE…", flush=True)
    sae_b = train_seeded_sae(cfg, X_n, sae.cfg, device)
    align = cross_sae_alignment(sae, sae_b)
    print(f"  mean matched-pair |cos|: {align['mean_matched_similarity']:.3f}", flush=True)
    print(f"  median: {align['median_matched_similarity']:.3f}", flush=True)
    print(f"  pairs > 0.5 / 0.7 / 0.9: "
          f"{align['n_matched_above_0.5']} / {align['n_matched_above_0.7']} / "
          f"{align['n_matched_above_0.9']}  (of {align['n_atoms']})", flush=True)

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg),
        "ablation": ablation,
        "alignment": align,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
