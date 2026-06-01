"""Four novel atom-level benchmarks on a trained Manifold-SAE.

Run against any curve SAE checkpoint. Reports four measurements per
alive atom:

  1. polysemy_score          — number of distinct token-cluster modes
                                each atom fires on (K-means in residual
                                space restricted to firing tokens).
                                Monosemantic atom = 1; polysemantic = many.
  2. cross_layer_transfer    — for each atom's direction W_k, project
                                residuals at layers ≠ training layer onto
                                it; measure correlation with the same
                                tokens' firing rate at the training layer.
                                High = the atom's direction encodes the
                                same concept at multiple layers
                                (information propagation).
  3. adversarial_max         — gradient ascent on a learnable token
                                embedding to maximize atom k's amplitude.
                                Recover the nearest-neighbor real token to
                                the optimized embedding. Token-free atom
                                interpretation.
  4. probe_classification    — linear probe on SAE atom firings predicting
                                a planted concept (magnitude bucket).
                                Compare to linear probe on raw activations.
                                If atom-firings outperform raw, the
                                decomposition is genuinely more probeable.

Configurable via env:
  MSAE_CHECKPOINT, MSAE_MODEL, MSAE_LAYER, MSAE_DASH_N_TOKENS, MSAE_CONCEPT
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "<repo_root>/runs/llm_sweep/curve_F256.pt",
    )
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-0.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "12"))
    n_tokens: int = int(os.environ.get("MSAE_DASH_N_TOKENS", "20000"))
    # For cross-layer transfer
    other_layers: tuple[int, ...] = (4, 8, 16, 20)
    # For polysemy
    polysemy_max_atoms: int = 40              # only analyze top-N alive atoms
    polysemy_max_clusters: int = 8
    polysemy_min_fires: int = 30
    # For adversarial
    adv_n_atoms: int = 10                     # only optimize for the top-N alive atoms
    adv_steps: int = 200
    # For probe classification
    concept: str = os.environ.get("MSAE_CONCEPT", "magnitude")

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/ATOM_ANALYSIS",
    )
    seed: int = 0


# ---------------------------------------------------------------------------
# Harvest helpers
# ---------------------------------------------------------------------------


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def harvest_multi_layer(model_name: str, layers: list[int], n_tokens: int,
                         device: torch.device) -> tuple[dict[int, torch.Tensor], list[str], list[int]]:
    """Harvest residuals at multiple layers, paired by token. Returns:
       {layer: (N, D)}, tokens, prompt_ids
    """
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    def make_hook(L):
        def hook(_m, _i, output):
            captured[L] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
    for L in layers:
        handles.append(blocks[L].register_forward_hook(make_hook(L)))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [d["text"] for d in ds if isinstance(d.get("text"), str) and len(d["text"]) > 100]

    out: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    tokens_out: list[str] = []
    prompt_ids: list[int] = []
    collected = 0
    torch.set_grad_enabled(False)
    for prompt_idx, text in enumerate(texts):
        if collected >= n_tokens:
            break
        inputs = tok(text[:2000], return_tensors="pt", truncation=True, max_length=256).to(device)
        model(**inputs)
        ids = inputs["input_ids"][0]
        T = ids.shape[0]
        for i in range(T):
            if collected >= n_tokens:
                break
            tok_str = tok.decode(ids[i])
            if tok_str in ("<|endoftext|>", "<pad>") or not tok_str.strip():
                continue
            for L in layers:
                out[L].append(captured[L][0, i, :].cpu())
            tokens_out.append(tok_str)
            prompt_ids.append(prompt_idx)
            collected += 1
    torch.set_grad_enabled(True)
    for h in handles:
        h.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {L: torch.stack(out[L], dim=0) for L in layers}, tokens_out, prompt_ids


def load_curve_sae(path: Path, D: int, device: torch.device):
    from manifold_sae.sae import load_sae

    return load_sae(path, input_dim=D, device=device)


# ---------------------------------------------------------------------------
# Benchmark 1: polysemy quantification
# ---------------------------------------------------------------------------


def polysemy_per_atom(sae, X: torch.Tensor, tokens: list[str], cfg: Config, device) -> dict:
    """For each alive atom, cluster the residuals of firing tokens; report
    the count of "meaningful" clusters (k chosen via silhouette score).
    """
    from sklearn.cluster import KMeans
    with torch.no_grad():
        out = sae(X.to(device=device, dtype=sae.cfg.dtype))
    amp = out.amplitudes.cpu().numpy()                    # (N, F)
    fire_counts = (amp > 1e-6).sum(axis=0)
    alive = [k for k in range(amp.shape[1])
             if fire_counts[k] >= cfg.polysemy_min_fires]
    if len(alive) > cfg.polysemy_max_atoms:
        order = np.argsort(-fire_counts)
        alive = [k for k in order if k in alive][:cfg.polysemy_max_atoms]

    X_np = X.cpu().numpy()
    results = []
    for k in alive:
        firing = np.where(amp[:, k] > 1e-6)[0]
        if len(firing) < cfg.polysemy_min_fires:
            continue
        X_k = X_np[firing]
        # Try k=1..max_clusters, pick k with best silhouette.
        # Skip silhouette eval if too few points.
        best_k = 1
        if len(firing) >= 4 * cfg.polysemy_max_clusters:
            from sklearn.metrics import silhouette_score
            best_sil = -2.0
            for n_clust in range(2, min(cfg.polysemy_max_clusters + 1, len(firing) // 4)):
                km = KMeans(n_clusters=n_clust, n_init=5, random_state=cfg.seed).fit(X_k)
                if len(set(km.labels_)) < n_clust:
                    continue
                sil = silhouette_score(X_k, km.labels_)
                if sil > best_sil:
                    best_sil = sil; best_k = n_clust
            # If even the best multi-cluster fit is poor (sil < 0.05),
            # call it monosemantic.
            if best_sil < 0.05:
                best_k = 1
        results.append({
            "atom": int(k),
            "n_fires": int(fire_counts[k]),
            "polysemy_k": int(best_k),
        })
    poly_counts = np.array([r["polysemy_k"] for r in results])
    return {
        "per_atom": results,
        "mean_polysemy": float(poly_counts.mean()) if len(poly_counts) else 0.0,
        "n_monosemantic": int((poly_counts == 1).sum()),
        "n_polysemantic": int((poly_counts >= 2).sum()),
        "n_analyzed": len(results),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: cross-layer transfer
# ---------------------------------------------------------------------------


def cross_layer_transfer(sae, X_train_layer: torch.Tensor,
                          X_other_layers: dict[int, torch.Tensor],
                          cfg: Config, device) -> dict:
    """For each atom k, take W_k (direction in residual stream at the
    training layer). Project residuals from OTHER layers onto W_k. Does
    the projection correlate with the atom's amplitude on the same tokens?
    High correlation = direction carries the concept across layers.
    """
    with torch.no_grad():
        out_train = sae(X_train_layer.to(device=device, dtype=sae.cfg.dtype))
    amp_train = out_train.amplitudes.cpu().numpy()        # (N, F)
    # Cutover: there is no `directions` (W_k) anymore. The atom's primary
    # ambient direction is the top right-singular vector of its decoder block
    # decoder_blocks[k] (K x D) — the dominant axis its curve sweeps in R^D.
    blocks = sae.decoder_blocks.detach().cpu().numpy()    # (F, K, D)
    F_total = blocks.shape[0]
    W_primary = np.zeros((F_total, blocks.shape[2]), dtype=np.float64)
    for k in range(F_total):
        _, _, vt = np.linalg.svd(blocks[k].astype(np.float64), full_matrices=False)
        W_primary[k] = vt[0]                              # (D,)

    results: dict[int, dict] = {}
    for L, X_L in X_other_layers.items():
        X_np = X_L.cpu().numpy().astype(np.float64)
        # Project: (N, F) = X @ W_primary.T
        proj = X_np @ W_primary.T
        # For each atom, correlate proj[:, k] with amp_train[:, k]
        per_atom = []
        for k in range(F_total):
            if (amp_train[:, k] > 1e-6).sum() < 30:
                per_atom.append(None); continue
            x = proj[:, k]
            y = amp_train[:, k]
            mx = x - x.mean(); my = y - y.mean()
            denom = float(np.sqrt((mx**2).sum() * (my**2).sum()))
            r = float((mx * my).sum() / denom) if denom > 0 else 0.0
            per_atom.append(r)
        rho_arr = np.array([p for p in per_atom if p is not None])
        results[L] = {
            "mean_abs_corr": float(np.abs(rho_arr).mean()) if len(rho_arr) else 0.0,
            "median_abs_corr": float(np.median(np.abs(rho_arr))) if len(rho_arr) else 0.0,
            "n_atoms_above_0.3": int((np.abs(rho_arr) > 0.3).sum()),
            "n_atoms_evaluated": len(rho_arr),
        }
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: adversarial atom maximization
# ---------------------------------------------------------------------------


def adversarial_max(sae, model_name: str, layer: int, cfg: Config, device,
                     tok_module=None) -> dict:
    """Gradient-ascend a token embedding to maximize atom k's amplitude.
    Then find the nearest real token to the optimized embedding.
    """
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name) if tok_module is None else tok_module
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)

    # We need the token-embedding table for nearest-neighbor lookup.
    if hasattr(model, "embed_tokens"):
        embed_table = model.embed_tokens.weight.detach()  # (V, D_embed)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_table = model.model.embed_tokens.weight.detach()
    elif hasattr(model, "wte"):
        embed_table = model.wte.weight.detach()
    else:
        return {"error": "could not find token embedding table"}

    D_embed = embed_table.shape[1]
    V = embed_table.shape[0]

    # Pick which atoms to optimize for (highest-firing alive ones, sampled).
    # We need a small set of activations to know which atoms are alive.
    # Use the first 256 tokens of wikitext as a sniff test.
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sample_text = next(d["text"] for d in ds
                       if isinstance(d.get("text"), str) and len(d["text"]) > 500)
    sniff = tok(sample_text[:1000], return_tensors="pt", truncation=True, max_length=256).to(device)
    captured: dict[int, torch.Tensor] = {}
    hook_handle = blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    with torch.no_grad():
        model(**sniff)
    hook_handle.remove()
    X_sniff = captured["h"][0]                               # (T, D)
    with torch.no_grad():
        out = sae(X_sniff)
    amp = out.amplitudes.cpu().numpy()
    fire_counts = (amp > 1e-6).sum(axis=0)
    alive = np.argsort(-fire_counts)[:cfg.adv_n_atoms].tolist()

    results = []
    for k in alive:
        # Optimize a single token embedding e (D_embed,) to maximize atom k's
        # amplitude when injected at position 0 of a fixed context.
        e = nn.Parameter(embed_table.mean(0).clone() + 0.01 * torch.randn(D_embed, device=device))
        opt = torch.optim.Adam([e], lr=1e-2)
        context_ids = sniff["input_ids"][0, :32]
        ctx_embeds = embed_table[context_ids].detach()

        # Hook to grab layer-L residual at last position.
        last_h: dict = {}
        h_hook = blocks[layer].register_forward_hook(
            lambda m, i, o, _last_h=last_h: _last_h.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)
            )
        )
        try:
            for step in range(cfg.adv_steps):
                opt.zero_grad()
                inputs_embeds = torch.cat([ctx_embeds, e.unsqueeze(0)], dim=0).unsqueeze(0)
                _ = model(inputs_embeds=inputs_embeds)
                h_at_layer = last_h["h"][0, -1, :].unsqueeze(0)   # (1, D)
                sae_out = sae(h_at_layer)
                atom_amp = sae_out.amplitudes[0, k]
                loss = -atom_amp                                  # maximize
                loss.backward()
                opt.step()
        finally:
            h_hook.remove()

        # Nearest token in the embedding table to the optimized e.
        e_d = e.detach()
        dists = ((embed_table - e_d.unsqueeze(0)) ** 2).sum(dim=1)
        top5_idx = dists.topk(5, largest=False).indices.cpu().numpy()
        top5_tokens = [tok.decode(int(i)) for i in top5_idx]
        results.append({
            "atom": int(k),
            "final_amp": float(atom_amp.item()),
            "top5_nearest_tokens": top5_tokens,
        })
        print(f"  [adv] atom {k}: final amp={atom_amp.item():.3f}  "
              f"top tokens: {top5_tokens[:3]}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"per_atom": results}


# ---------------------------------------------------------------------------
# Benchmark 4: probe classification on planted concept
# ---------------------------------------------------------------------------


def probe_classification(sae, model_name: str, layer: int, cfg: Config, device) -> dict:
    """Train a linear probe on SAE atom firings to predict magnitude bucket.
    Compare to a linear probe on raw activations.

    Concept: number tokens 1..1000 in prompts. Probe target: log-bucket of
    the magnitude. Higher probe accuracy on SAE features = atom-decomposition
    carries more structured signal for this concept than raw activations.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)

    captured = {}
    h_hook = blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )

    # Generate magnitude prompts.
    magnitudes = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
    templates = [
        "There were {N} apples in the basket.",
        "She counted {N} items.",
        "The price was {N} dollars.",
    ]
    Xs = []; labels = []
    with torch.no_grad():
        for N in magnitudes:
            bucket = int(np.log10(N + 1) * 2)      # 0..6 buckets
            for t in templates:
                prompt = t.format(N=N)
                inputs = tok(prompt, return_tensors="pt").to(device)
                model(**inputs)
                Xs.append(captured["h"][0, -1, :].cpu())
                labels.append(bucket)
    h_hook.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    X = torch.stack(Xs, dim=0)
    y = np.array(labels)

    # Normalize same way SAE training did.
    mu = X.mean(0, keepdim=True); sigma = X.std(0).clamp(min=1e-6)  # per-dim std (was scalar — see _normalize.py)
    X_n = (X - mu) / sigma

    # Get SAE features.
    with torch.no_grad():
        sae_out = sae(X_n.to(device=device, dtype=sae.cfg.dtype))
    pos_flat = sae_out.positions.reshape(sae_out.positions.shape[0], -1)  # (N, F*d)
    sae_features = torch.cat([pos_flat, sae_out.amplitudes], dim=1).cpu().numpy()
    raw_features = X.cpu().numpy()

    # 80/20 train/test split (random shuffle, seeded).
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    tr, te = perm[:n_train], perm[n_train:]

    probe_raw = LogisticRegression(max_iter=500).fit(raw_features[tr], y[tr])
    probe_sae = LogisticRegression(max_iter=500).fit(sae_features[tr], y[tr])
    return {
        "concept": cfg.concept,
        "n_classes": len(set(y)),
        "n_train": int(n_train),
        "n_test": int(len(y) - n_train),
        "accuracy_raw_train": float(accuracy_score(y[tr], probe_raw.predict(raw_features[tr]))),
        "accuracy_raw_test": float(accuracy_score(y[te], probe_raw.predict(raw_features[te]))),
        "accuracy_sae_train": float(accuracy_score(y[tr], probe_sae.predict(sae_features[tr]))),
        "accuracy_sae_test": float(accuracy_score(y[te], probe_sae.predict(sae_features[te]))),
    }


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

    # Harvest at training layer + other layers (for cross-layer transfer).
    all_layers = [cfg.layer] + [L for L in cfg.other_layers if L != cfg.layer]
    print(f"[harvest] layers {all_layers}, {cfg.n_tokens} tokens", flush=True)
    X_layers, tokens, _ = harvest_multi_layer(cfg.model_name, all_layers, cfg.n_tokens, device)

    X_train_layer = X_layers[cfg.layer]
    mu = X_train_layer.mean(0, keepdim=True); sigma = float(X_train_layer.std().item())
    X_n = (X_train_layer - mu) / max(sigma, 1e-6)
    D = X_train_layer.shape[1]

    sae = load_curve_sae(Path(cfg.checkpoint), D, device)
    print(f"[setup] loaded SAE F={sae.cfg.n_atoms} top_k={sae.cfg.sparsity.target_k}", flush=True)

    # Save partial results after each benchmark so a later crash doesn't
    # lose the earlier work.
    accumulated = {"config": asdict(cfg)}
    def save():
        (out_dir / "results.json").write_text(json.dumps(accumulated, indent=2, default=float))

    print("\n=== Benchmark 1: polysemy ===", flush=True)
    poly = polysemy_per_atom(sae, X_n, tokens, cfg, device)
    print(f"  n_analyzed={poly['n_analyzed']}  monosemantic={poly['n_monosemantic']}  "
          f"polysemantic={poly['n_polysemantic']}  mean_polysemy_k={poly['mean_polysemy']:.2f}", flush=True)
    accumulated["polysemy"] = poly; save()

    print("\n=== Benchmark 2: cross-layer transfer ===", flush=True)
    other_layers_X_n = {L: (X_layers[L] - X_layers[L].mean(0, keepdim=True)) /
                         max(float(X_layers[L].std().item()), 1e-6)
                         for L in cfg.other_layers if L != cfg.layer}
    xl = cross_layer_transfer(sae, X_n, other_layers_X_n, cfg, device)
    for L, r in xl.items():
        print(f"  layer {L}: mean|ρ|={r['mean_abs_corr']:.3f}  "
              f"atoms above 0.3: {r['n_atoms_above_0.3']}/{r['n_atoms_evaluated']}", flush=True)
    accumulated["cross_layer_transfer"] = xl; save()

    print("\n=== Benchmark 3: adversarial atom maximization ===", flush=True)
    try:
        adv = adversarial_max(sae, cfg.model_name, cfg.layer, cfg, device)
        print(f"  optimized {len(adv.get('per_atom', []))} atoms", flush=True)
        accumulated["adversarial_max"] = adv; save()
    except Exception as e:
        print(f"  [skipped] {e}", flush=True)
        accumulated["adversarial_max"] = {"error": str(e)}; save()

    print("\n=== Benchmark 4: probe classification (magnitude) ===", flush=True)
    try:
        probe = probe_classification(sae, cfg.model_name, cfg.layer, cfg, device)
        print(f"  raw acc: train={probe['accuracy_raw_train']:.3f}  test={probe['accuracy_raw_test']:.3f}", flush=True)
        print(f"  SAE acc: train={probe['accuracy_sae_train']:.3f}  test={probe['accuracy_sae_test']:.3f}", flush=True)
        accumulated["probe_classification"] = probe; save()
    except Exception as e:
        print(f"  [skipped] {e}", flush=True)
        accumulated["probe_classification"] = {"error": str(e)}; save()
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
