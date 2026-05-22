"""DBSCAN clustering of vanilla SAE features + spline-manifold fit per cluster.

This is the Goodfire/Bhalla-style post-hoc pipeline written cleanly, so we
can A/B test it directly against Manifold-SAE's native compact-capture:

  1. Train (or load) a vanilla TopK SAE on LM activations.
  2. Compute per-feature firing patterns over a corpus.
  3. Cluster features with DBSCAN on (1 − cosine_sim of firing patterns).
  4. For each cluster of ≥2 atoms, fit a 1D smoothing-spline manifold
     through the corresponding decoder directions in residual space.
  5. Report per-cluster: member atoms, total firing count, top tokens
     sorted along the manifold's principal axis, manifold arc length.

A clean cluster + smooth recovered curve through its directions =
"compact capture" of a 1D manifold, post-hoc. Compare to Manifold-SAE
where each atom IS already a compact-capture unit by construction.

Reads via env: MSAE_CHECKPOINT (vanilla .pt), MSAE_MODEL, MSAE_LAYER.
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


@dataclass
class Config:
    checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "/home/athuser/gnome_home/manifold_sae/runs/llm_sweep_q15b_L18/vanilla_F128.pt",
    )
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    n_tokens: int = 20_000
    # DBSCAN
    dbscan_eps: float = float(os.environ.get("MSAE_DBSCAN_EPS", "0.4"))
    dbscan_min_samples: int = int(os.environ.get("MSAE_DBSCAN_MIN_SAMPLES", "2"))
    # Min firing count to consider an atom alive
    min_fires: int = 20
    # How many atoms per cluster to render in the report
    max_tokens_per_cluster: int = 40

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/DBSCAN_MANIFOLDS",
    )
    seed: int = 0


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


class VanillaSAE(nn.Module):
    """Mirror of llm_sweep.VanillaSAE for checkpoint loading."""
    def __init__(self, D, F, top_k):
        super().__init__()
        self.F = F; self.top_k = top_k
        H = max(4 * D, 2 * F)
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H); self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x):
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        if self.top_k < self.F:
            vals, idx = torch.topk(z, self.top_k, dim=1)
            gate = torch.zeros_like(z).scatter_(1, idx, vals)
            z = gate
        return z @ self.W_dec, z


def load_vanilla_sae(path: Path, D: int, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    sae = VanillaSAE(D, sig["F"], sig["top_k"]).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    return sae


def harvest(model_name: str, layer: int, n_tokens: int, device: torch.device):
    from datasets import load_dataset
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
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    Xs, toks = [], []
    with torch.no_grad():
        for d in ds:
            if len(Xs) >= n_tokens: break
            text = d.get("text", "")
            if not isinstance(text, str) or len(text) < 100: continue
            inputs = tok(text[:1500], return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**inputs)
            ids = inputs["input_ids"][0]
            for i in range(captured["h"].shape[1]):
                if len(Xs) >= n_tokens: break
                tok_str = tok.decode(ids[i])
                if tok_str in ("<|endoftext|>", "<pad>") or not tok_str.strip():
                    continue
                Xs.append(captured["h"][0, i, :].cpu())
                toks.append(tok_str)
    h_hook.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(Xs[:n_tokens], dim=0), toks[:n_tokens]


def dbscan_cluster_features(firing: np.ndarray, cfg: Config) -> dict:
    """Cluster atoms by 1 − cosine_sim of firing patterns. Atom k ∈ ℝ^N
    where N is number of tokens.
    """
    from sklearn.cluster import DBSCAN

    # cosine_sim of columns; (F, F) similarity matrix
    cols = firing / (np.linalg.norm(firing, axis=0, keepdims=True) + 1e-9)
    sim = cols.T @ cols                                            # (F, F)
    # Convert to distance: 1 - |sim| (treat negative correlations as similar too — same axis).
    # Clip to [0, ∞) — sklearn DBSCAN's precomputed-distance check rejects
    # negatives, and float32 noise on unit-norm cosines can give |sim| > 1.
    dist = np.clip(1.0 - np.abs(sim), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    # DBSCAN on the precomputed distance
    db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples,
                metric="precomputed").fit(dist)
    labels = db.labels_                                            # -1 = noise
    return {
        "labels": labels.tolist(),
        "n_clusters": int(labels.max() + 1) if (labels >= 0).any() else 0,
        "n_noise": int((labels == -1).sum()),
    }


def fit_spline_manifold(decoder_dirs: np.ndarray, firings: np.ndarray,
                         X: np.ndarray) -> dict:
    """Given a cluster's decoder directions (n_atoms, D) and firing
    patterns (N, n_atoms), fit a 1D smoothing spline through the
    cluster's contribution per token. Returns per-token coordinate
    along the manifold.

    Method: compute the cluster's contribution to recon per firing token,
    PCA to a 1D coordinate (the "atom-axis").
    """
    n_atoms, D = decoder_dirs.shape
    # Cluster contribution per firing token: (N, D) = firing @ decoder_dirs
    # restricted to firing tokens (any atom in cluster fires).
    fired = (firings > 1e-6).any(axis=1)
    if fired.sum() < 30:
        return {"valid": False, "reason": "too few firing tokens"}
    contrib = firings[fired] @ decoder_dirs                        # (N_fire, D)
    # PCA: get the principal direction
    c_centered = contrib - contrib.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(c_centered, full_matrices=False)
    pc1 = c_centered @ Vt[0]                                       # (N_fire,)
    # Arc length along manifold = sum of consecutive distances after sorting by PC1
    order = np.argsort(pc1)
    sorted_contrib = c_centered[order]
    diffs = np.diff(sorted_contrib, axis=0)
    arc_length = float(np.linalg.norm(diffs, axis=1).sum())
    return {
        "valid": True,
        "n_atoms": int(n_atoms),
        "n_firing_tokens": int(fired.sum()),
        "principal_var_ratio": float(S[0] / (S.sum() + 1e-9)),
        "arc_length": arc_length,
        "fired_indices": np.where(fired)[0].tolist(),
        "pc1_per_firing_token": pc1.tolist(),
    }


def render_report(catalog: list[dict], tokens: list[str], cfg: Config) -> str:
    md = [f"# DBSCAN-clustered vanilla SAE manifolds\n\n",
          f"Checkpoint: `{cfg.checkpoint}`\n",
          f"DBSCAN(eps={cfg.dbscan_eps}, min_samples={cfg.dbscan_min_samples})\n\n",
          f"## {len(catalog)} clusters\n\n"]
    for ci, entry in enumerate(catalog):
        if not entry.get("valid", False):
            md.append(f"### Cluster {ci}: skipped ({entry.get('reason', 'invalid')})\n\n")
            continue
        atoms = entry["atoms"]
        md.append(f"### Cluster {ci} — {len(atoms)} atoms, "
                  f"{entry['n_firing_tokens']} firing tokens, "
                  f"arc length {entry['arc_length']:.2f}, "
                  f"PC1 variance {entry['principal_var_ratio']:.2f}\n\n")
        md.append(f"Member atoms: {atoms}\n\n")
        md.append("Top tokens sorted by manifold PC1 axis:\n\n")
        md.append("| PC1 | token |\n| --- | --- |\n")
        fired = entry["fired_indices"]
        pc1 = entry["pc1_per_firing_token"]
        order = np.argsort(pc1)
        n_show = min(cfg.max_tokens_per_cluster, len(order))
        # Show evenly-spaced samples along PC1
        sample_idx = np.linspace(0, len(order) - 1, n_show, dtype=int)
        for si in sample_idx:
            tok_idx = fired[order[si]]
            md.append(f"| {pc1[order[si]]:.2f} | `{tokens[tok_idx]!r}` |\n")
        md.append("\n")
    return "".join(md)


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dbscan] device={device} output_dir={out_dir}", flush=True)

    if not Path(cfg.checkpoint).exists():
        print(f"[error] checkpoint not found: {cfg.checkpoint}", file=sys.stderr)
        return 1

    print(f"[harvest] {cfg.n_tokens} tokens at layer {cfg.layer}", flush=True)
    X, tokens = harvest(cfg.model_name, cfg.layer, cfg.n_tokens, device)
    print(f"  X={X.shape}  tokens={len(tokens)}", flush=True)

    D = X.shape[1]
    sae = load_vanilla_sae(Path(cfg.checkpoint), D, device)
    print(f"[setup] loaded vanilla SAE F={sae.F} top_k={sae.top_k}", flush=True)

    # Get firing pattern.
    mu = X.mean(0, keepdim=True); sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)
    with torch.no_grad():
        _, gate = sae(X_n.to(device))
    firing = gate.cpu().numpy()                                    # (N, F)
    fire_counts = (firing > 1e-6).sum(axis=0)
    alive = [k for k in range(sae.F) if fire_counts[k] >= cfg.min_fires]
    print(f"[cluster] {len(alive)} alive atoms (≥ {cfg.min_fires} fires)", flush=True)
    firing_alive = firing[:, alive]                                # (N, n_alive)

    db = dbscan_cluster_features(firing_alive, cfg)
    labels = np.array(db["labels"])
    print(f"  DBSCAN → {db['n_clusters']} clusters, {db['n_noise']} noise atoms", flush=True)

    # For each cluster: fit a manifold through the member atoms' decoder
    # directions and report top tokens sorted by manifold PC1.
    catalog = []
    W_dec = sae.W_dec.detach().cpu().numpy()                       # (F, D)
    for cluster_id in range(db["n_clusters"]):
        member_local = np.where(labels == cluster_id)[0]
        member_global = [alive[i] for i in member_local]
        if len(member_global) < 2: continue
        cluster_W = W_dec[member_global]                           # (n, D)
        cluster_firing = firing[:, member_global]                  # (N, n)
        manifold = fit_spline_manifold(cluster_W, cluster_firing, X.numpy())
        manifold["atoms"] = member_global
        manifold["cluster_id"] = int(cluster_id)
        catalog.append(manifold)
        if manifold.get("valid"):
            print(f"  cluster {cluster_id}: {len(member_global)} atoms, "
                  f"{manifold['n_firing_tokens']} firing tokens, "
                  f"arc length {manifold['arc_length']:.2f}", flush=True)

    md = render_report(catalog, tokens, cfg)
    (out_dir / "report.md").write_text(md)
    (out_dir / "catalog.json").write_text(json.dumps({
        "config": asdict(cfg),
        "dbscan": db,
        "catalog": catalog,
    }, indent=2, default=float))
    print(f"\n[done] wrote {out_dir / 'report.md'} ({len([c for c in catalog if c.get('valid')])} clusters)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
