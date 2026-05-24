"""Train skip-transcoder on paired cogito-L20/L40 residuals; compare to L1 SAE.

Spec
----
- 15 epochs, MPS, F=512, rank_skip=64.
- Loads paired (X_in, Y_out) from
  ``runs/COGITO_PAIRED_L20_L40_STANDIN/paired.pt`` (PCA-stand-in fallback —
  see ``scripts/harvest_paired_l20_l40.py``).
- Reports interp score (HSV coherence + xkcd compactness) for both the
  skip-transcoder and a vanilla L1 SAE trained on Y_out (= L40) only at
  matched F and sparsity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from manifold_sae.transcoder import (
    TranscoderConfig,
    TranscoderTrainer,
    interp_score_hsv_coherence,
)


def _train_l1_baseline(
    Y: torch.Tensor,
    F: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    lambda_l1: float,
    target_sparsity: float,
) -> torch.nn.Module:
    """Minimal autoencoder L1-SAE baseline matched at the same F.

    Trained directly on Y_out so it never sees X_in; this is the
    head-to-head comparison the Paulo et al. paper makes.
    """
    in_dim = Y.shape[1]
    enc = torch.nn.Linear(in_dim, F).to(device)
    dec = torch.nn.Linear(F, in_dim, bias=False).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    Y = Y.to(device, dtype=torch.float32)
    N = Y.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        running = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            yb = Y[idx]
            z = torch.nn.functional.relu(enc(yb))
            y_hat = dec(z)
            mse = (y_hat - yb).pow(2).mean()
            l1 = z.abs().mean()
            loss = mse + lambda_l1 * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        print(f"[l1 epoch {epoch:2d}] loss={running / max(1, n_batches):.5f}")

    class _Wrap(torch.nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            z = torch.nn.functional.relu(self.enc(x))
            return self.dec(z), z

    return _Wrap(enc, dec)


def _interp_for_l1(
    model: torch.nn.Module, X: torch.Tensor, hsv: torch.Tensor, top_k: int = 20
) -> dict[str, float]:
    """Same operational metric as the transcoder evaluator, on the L1 SAE."""
    with torch.no_grad():
        _, z = model(X.to(next(model.parameters()).device, dtype=torch.float32))
        z = z.cpu()
    F = z.shape[1]
    H = hsv[:, 0].cpu()
    hsv_cpu = hsv.cpu()
    hue_scores = []
    compact_scores = []
    for k in range(F):
        zk = z[:, k]
        if (zk.abs() > 1e-8).sum() < top_k:
            continue
        top_idx = zk.topk(top_k).indices
        angles = 2 * torch.pi * H[top_idx]
        sin_m = float(torch.sin(angles).mean().item())
        cos_m = float(torch.cos(angles).mean().item())
        R = (sin_m * sin_m + cos_m * cos_m) ** 0.5
        circ_std = (-2.0 * torch.log(torch.tensor(max(R, 1e-8)))) ** 0.5
        hue_scores.append(1.0 / (1.0 + float(circ_std.item())))
        sub = hsv_cpu[top_idx]
        sub = sub / sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
        mean_dir = sub.mean(dim=0, keepdim=True)
        mean_dir = mean_dir / mean_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        compact_scores.append(float((sub @ mean_dir.t()).mean().item()))
    return {
        "hue_coherence_mean": float(sum(hue_scores) / max(1, len(hue_scores))),
        "xkcd_compactness_mean": float(sum(compact_scores) / max(1, len(compact_scores))),
        "n_atoms_scored": len(hue_scores),
        "combined_interp_score": float(
            0.5 * sum(hue_scores) / max(1, len(hue_scores))
            + 0.5 * sum(compact_scores) / max(1, len(compact_scores))
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paired",
                   default="/Users/user/Manifold-SAE/runs/COGITO_PAIRED_L20_L40_STANDIN/paired.pt")
    p.add_argument("--out_dir",
                   default="/Users/user/Manifold-SAE/runs/COGITO_SKIP_TRANSCODER")
    p.add_argument("--F", type=int, default=512)
    p.add_argument("--rank_skip", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="mps")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_transcoder] loading paired residuals from {args.paired}")
    blob = torch.load(args.paired, map_location="cpu")
    X_in: torch.Tensor = blob["X_in"]
    Y_out: torch.Tensor = blob["Y_out"]
    hsv: torch.Tensor = blob["hsv"]
    print(f"[train_transcoder] X_in {X_in.shape} Y_out {Y_out.shape}")

    cfg = TranscoderConfig(
        in_dim=X_in.shape[1],
        out_dim=Y_out.shape[1],
        n_atoms=args.F,
        rank_skip=args.rank_skip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    trainer = TranscoderTrainer(cfg)
    tr = trainer.fit(X_in, Y_out)
    torch.save(
        {
            "state_dict": tr.smooth.state_dict(),
            "config": cfg.__dict__,
            "history": tr.history,
            "final_mse": tr.final_mse,
            "final_explained_variance": tr.final_explained_variance,
            "final_sparsity": tr.final_sparsity,
            "meta": blob.get("meta", {}),
        },
        out_dir / "transcoder.pt",
    )
    print(f"[train_transcoder] transcoder EV={tr.final_explained_variance:.4f} sparsity={tr.final_sparsity:.4f}")

    transcoder_interp = interp_score_hsv_coherence(tr.smooth, X_in, hsv)

    print("[train_transcoder] training L1-SAE baseline on Y_out only")
    device = torch.device(cfg.device if cfg.device != "mps" or torch.backends.mps.is_available() else "cpu")
    l1_model = _train_l1_baseline(
        Y_out, F=args.F, epochs=args.epochs, batch_size=args.batch_size,
        device=device, lambda_l1=1e-3, target_sparsity=tr.final_sparsity,
    )
    l1_interp = _interp_for_l1(l1_model, Y_out, hsv)

    report = {
        "transcoder": transcoder_interp,
        "l1_baseline": l1_interp,
        "transcoder_explained_variance": tr.final_explained_variance,
        "transcoder_final_sparsity": tr.final_sparsity,
        "verdict": (
            "transcoder_wins_interp"
            if transcoder_interp["combined_interp_score"] > l1_interp["combined_interp_score"]
            else "l1_wins_interp"
        ),
        "delta_interp": (
            transcoder_interp["combined_interp_score"] - l1_interp["combined_interp_score"]
        ),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
