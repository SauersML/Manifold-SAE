"""Train UniversalSAE on cogito-L40 + Qwen2.5-0.5B-L12 paired activations.

Headline question: does the HSV hue-ring (project_cogito_color_manifold...)
appear in BOTH models from the SAME shared atoms?

Pairing
-------
Each row r in cogito's X_L40 corresponds to (color, template) =
(r // 28, r % 28). The Qwen harvest uses a TEMPLATE SUBSET (default
[0, 7, 16, 24]); we therefore subsample the cogito rows to the matching
(color, template_idx) pairs before training.

Output
------
runs/UNIVERSAL_COGITO_QWEN/
  ckpt.pt                  trained UniversalSAE state_dict
  metrics.json             per-model R², universality stats, alive counts
  universal_hue_ring.png   2-panel: per-model PCA(decoder columns of
                           universal atoms), points colored by HSV hue
  affinity.png             histogram of per-atom universality scores
  config.json              run configuration
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch

from manifold_sae.cross_llm.universal_sae import UniversalSAE
from manifold_sae.cross_llm.harvest_local import load_xkcd_colors


COGITO_X_PATH = Path("runs/COLOR_COGITO_L40/X_L40.npy")
COGITO_N_TEMPLATES = 28
QWEN_DEFAULT_DIR = Path("runs/COLOR_QWEN_05B_L12")
OUT_DEFAULT = Path("runs/UNIVERSAL_COGITO_QWEN")


# ---------------------------------------------------------------------------
# Data loading + pairing
# ---------------------------------------------------------------------------
def load_paired(
    cogito_x_path: Path,
    qwen_dir: Path,
    *,
    n_templates_cogito: int = COGITO_N_TEMPLATES,
    n_colors_cap: int | None = None,
) -> tuple[dict[str, np.ndarray], list[int], np.ndarray]:
    """Load paired (cogito-L40, qwen-L12) rows in matching (color, template) order.

    Returns
    -------
    X_by_model : {"cogito_L40": (N, 7168), "qwen_L12": (N, 896)}
    template_indices : list[int]  the template indices from the qwen meta
    hue : (n_colors,)  HSV hue for each color (for plotting)
    """
    meta = json.loads((qwen_dir / "meta.json").read_text())
    t_idx = meta["template_indices"]
    n_colors = meta["n_colors"]
    n_t = len(t_idx)

    X_q = np.load(qwen_dir / "X.npy")  # (n_colors * n_t, 896)
    if X_q.shape[0] != n_colors * n_t:
        raise RuntimeError(
            f"qwen X.npy shape {X_q.shape} disagrees with meta "
            f"(n_colors={n_colors} * n_templates={n_t})"
        )

    # Cogito is row-major (color, template) with 28 templates.
    X_c_full = np.load(cogito_x_path, mmap_mode="r")
    n_colors_cogito = X_c_full.shape[0] // n_templates_cogito
    if n_colors > n_colors_cogito:
        raise RuntimeError(
            f"qwen has {n_colors} colors but cogito only has {n_colors_cogito}"
        )
    if n_colors_cap is not None:
        n_colors = min(n_colors, n_colors_cap)

    # Build cogito row indices in the matching order.
    rows = np.array([
        ci * n_templates_cogito + ti
        for ci in range(n_colors)
        for ti in t_idx
    ], dtype=np.int64)
    X_c = np.ascontiguousarray(X_c_full[rows], dtype=np.float32)
    X_q = X_q[: n_colors * n_t].astype(np.float32, copy=False)

    print(f"[load] paired N={X_c.shape[0]}  cogito D={X_c.shape[1]}  "
          f"qwen D={X_q.shape[1]}", flush=True)

    # Compute HSV hue per color, using xkcd colors restricted to qwen-meta order.
    color_pairs = [(name, tuple(rgb)) for name, rgb in meta["colors"][:n_colors]]
    hue = np.zeros(n_colors, dtype=np.float32)
    for i, (_name, (r, g, b)) in enumerate(color_pairs):
        h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        hue[i] = h

    X_by_model = {"cogito_L40": X_c, "qwen_L12": X_q}
    return X_by_model, t_idx, hue


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    X_by_model: dict[str, np.ndarray],
    *,
    F: int = 256,
    sparsity_weight: float = 1e-3,
    activation: str = "topk",
    top_k: int = 16,
    lr: float = 3e-3,
    epochs: int = 400,
    batch_size: int = 256,
    device: str = "cpu",
    log_every: int = 50,
) -> UniversalSAE:
    model_dims = {m: int(X.shape[1]) for m, X in X_by_model.items()}
    sae = UniversalSAE(
        F=F, model_dims=model_dims, sparsity_weight=sparsity_weight,
        activation=activation, top_k=top_k,
    ).to(device)

    X_t = {m: torch.from_numpy(X).to(device) for m, X in X_by_model.items()}
    sae.fit_centers(X_t)

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    N = next(iter(X_t.values())).shape[0]

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        total = 0.0
        nb = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            batch = {m: X_t[m][idx] for m in X_t}
            out = sae(batch)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            r2 = sae.per_model_r2(X_t)
            alive = int(sae.alive_mask(X_t).sum().item())
            uscore = sae.universality_score().mean().item()
            print(f"[train] epoch={epoch:4d}  loss={total/nb:.4f}  "
                  f"r2={ {m: round(v, 3) for m, v in r2.items()} }  "
                  f"alive={alive}/{F}  univ_mean={uscore:.3f}",
                  flush=True)
    return sae


# ---------------------------------------------------------------------------
# Analysis + plots
# ---------------------------------------------------------------------------
def analyze_and_plot(
    sae: UniversalSAE,
    X_by_model: dict[str, np.ndarray],
    *,
    hue_per_color: np.ndarray,
    n_templates: int,
    out_dir: Path,
    universality_threshold: float = 0.15,
) -> dict:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(sae.parameters()).device
    X_t = {m: torch.from_numpy(X).to(device) for m, X in X_by_model.items()}

    r2 = sae.per_model_r2(X_t)
    affinity = sae.atom_model_affinity().cpu().numpy()  # (F, M)
    universal_mask = sae.universal_atom_mask(threshold=universality_threshold).cpu().numpy()
    u_score = sae.universality_score(threshold=1e-3).cpu().numpy()
    alive_mask = sae.alive_mask(X_t).cpu().numpy()
    alive_and_universal = alive_mask & universal_mask

    n_alive = int(alive_mask.sum())
    n_univ = int(universal_mask.sum())
    n_alive_univ = int(alive_and_universal.sum())
    F_total = sae.F
    print(f"[analysis] alive={n_alive}/{F_total}  universal={n_univ}/{F_total}  "
          f"alive_and_universal={n_alive_univ}", flush=True)

    # ---- Affinity histogram ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(u_score, bins=11, range=(-0.05, 1.05), color="steelblue", alpha=0.8)
    ax[0].set_xlabel("universality score (frac models with mass)")
    ax[0].set_ylabel("atom count")
    ax[0].set_title(f"alive={n_alive}/{F_total}  universal≥{universality_threshold}={n_univ}")

    if affinity.shape[1] >= 2:
        sc = ax[1].scatter(affinity[alive_mask, 0], affinity[alive_mask, 1],
                           s=8, alpha=0.7, c=u_score[alive_mask], cmap="viridis")
        ax[1].plot([0, 1], [1, 0], ls="--", c="grey", lw=0.8)
        ax[1].set_xlabel(f"share[{sae.model_names[0]}]")
        ax[1].set_ylabel(f"share[{sae.model_names[1]}]")
        ax[1].set_title("per-atom decoder mass split (alive)")
        plt.colorbar(sc, ax=ax[1], label="univ score")
    fig.tight_layout()
    fig.savefig(out_dir / "affinity.png", dpi=120)
    plt.close(fig)

    # ---- Hue-ring panel: per-color centroid of activations, projected
    # via PCA of decoder columns restricted to universal+alive atoms.
    # We project the *centered* per-color centroid (averaged over templates)
    # in each model onto that model's decoder columns U_m = W_m[:, mask]^T
    # then PCA-2 those projections (one PCA per model). Colors come out the
    # same iff the same atoms are doing the work in both models.
    pick = alive_and_universal
    if pick.sum() < 2:
        print("[analysis] WARN — <2 alive+universal atoms; falling back to all alive",
              flush=True)
        pick = alive_mask
    pick_idx = np.where(pick)[0]

    # Per-color centroid over the n_templates rows.
    centroids: dict[str, np.ndarray] = {}
    for m in sae.model_names:
        X = X_by_model[m]
        n_colors = X.shape[0] // n_templates
        Xc = X[: n_colors * n_templates].reshape(n_colors, n_templates, -1).mean(axis=1)
        mu = getattr(sae, f"mu_{m}").detach().cpu().numpy()
        s_arr = getattr(sae, f"scale_{m}").detach().cpu().numpy().reshape(-1)
        s = float(s_arr[0])
        centroids[m] = (Xc - mu) * s  # (n_colors, D_m)

    proj_2d: dict[str, np.ndarray] = {}
    for m in sae.model_names:
        W_uni = sae.decoders[m].detach().cpu().numpy()[pick_idx, :]  # (k, D_m)
        # Activations on universal-atom subspace = centroid @ W_uni^T
        A = centroids[m] @ W_uni.T   # (n_colors, k)
        Ac = A - A.mean(0, keepdims=True)
        # PCA-2
        U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
        proj = Ac @ Vt[:2].T   # (n_colors, 2)
        proj_2d[m] = proj

    # Plot hue ring side-by-side.
    fig, axes = plt.subplots(1, len(sae.model_names),
                             figsize=(5 * len(sae.model_names), 5),
                             squeeze=False)
    n_colors = len(hue_per_color)
    rgb_colors = np.zeros((n_colors, 3))
    for i, h in enumerate(hue_per_color):
        # high-S, high-V version of the same hue for vivid plotting
        rgb_colors[i] = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)

    # Hue-ring quality metric: circular correlation between atan2(y, x) and
    # 2π·hue. Same metric used in auto_82.
    hue_rad = 2 * np.pi * hue_per_color
    circ_metric: dict[str, float] = {}
    for ax_i, m in enumerate(sae.model_names):
        p = proj_2d[m]
        ax = axes[0, ax_i]
        ax.scatter(p[:, 0], p[:, 1], c=rgb_colors, s=20, alpha=0.85,
                   edgecolors="black", linewidths=0.2)
        th = np.arctan2(p[:, 1], p[:, 0])
        cc = _circ_corr_js(th, hue_rad)
        circ_metric[m] = cc
        ax.set_title(f"{m}: hue-ring (circ-corr={cc:+.3f})")
        ax.set_xlabel("PC1 (universal-atom subspace)")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(
        f"Universal-atom hue-ring across {len(sae.model_names)} models "
        f"(alive+universal={n_alive_univ}/{F_total})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "universal_hue_ring.png", dpi=120)
    plt.close(fig)

    metrics = {
        "r2_per_model": {m: float(v) for m, v in r2.items()},
        "n_alive": n_alive,
        "n_universal": n_univ,
        "n_alive_and_universal": n_alive_univ,
        "universal_fraction_of_alive": float(n_alive_univ) / max(n_alive, 1),
        "universal_fraction_of_total": float(n_univ) / F_total,
        "universality_threshold": universality_threshold,
        "circular_correlation_hue_vs_proj": circ_metric,
        "model_names": sae.model_names,
        "model_dims": dict(zip(sae.model_names, sae.dims)),
        "F": F_total,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[analysis] metrics -> {out_dir/'metrics.json'}", flush=True)
    return metrics


def _circ_corr_js(a_rad: np.ndarray, b_rad: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation."""
    a_bar = np.angle(np.mean(np.exp(1j * a_rad)))
    b_bar = np.angle(np.mean(np.exp(1j * b_rad)))
    num = np.sum(np.sin(a_rad - a_bar) * np.sin(b_rad - b_bar))
    den = np.sqrt(np.sum(np.sin(a_rad - a_bar) ** 2)
                  * np.sum(np.sin(b_rad - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cogito-x", default=str(COGITO_X_PATH))
    ap.add_argument("--qwen-dir", default=str(QWEN_DEFAULT_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--activation", default="topk")
    ap.add_argument("--sparsity-weight", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-colors-cap", type=int, default=None)
    ap.add_argument("--universality-threshold", type=float, default=0.15)
    args = ap.parse_args()

    X_by_model, t_idx, hue = load_paired(
        Path(args.cogito_x), Path(args.qwen_dir),
        n_colors_cap=args.n_colors_cap,
    )
    n_templates = len(t_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    sae = train(
        X_by_model,
        F=args.F, sparsity_weight=args.sparsity_weight,
        activation=args.activation, top_k=args.top_k,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device,
    )
    torch.save({
        "state_dict": sae.state_dict(),
        "model_names": sae.model_names,
        "dims": sae.dims,
        "F": sae.F,
    }, out_dir / "ckpt.pt")

    metrics = analyze_and_plot(
        sae, X_by_model, hue_per_color=hue, n_templates=n_templates,
        out_dir=out_dir, universality_threshold=args.universality_threshold,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
