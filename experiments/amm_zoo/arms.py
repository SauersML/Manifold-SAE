"""Featurizer arms for the AMM zoo, at MATCHED decoder-scalar budget + matched L0.

All arms get the same total decoder scalars (= the true generative budget
``Σ_g b_g · d``) and the same active-code count per token (``L0``). Each arm turns
its fit into a list of :class:`metrics.RecoveredFactor` on the held-out split.

Arms
----
* ``topk_sae``      — TopK-SAE, ``b=1`` directions (vanilla BSF at block_size 1).
* ``bsf_vanilla``   — vanilla Block-Sparse Featurizer, ``b=2`` (free encoder).
* ``bsf_grassmann`` — Grassmannian BSF, ``b=2`` (tied γ, Stiefel decoder).
* ``ours``          — Grassmannian block T1 (== bsf_grassmann) THEN a per-block
                      K=1 circle chart: classify each block's code cloud
                      (ring vs blob), and on ring blocks DENOISE onto the fitted
                      circle (radius fixed, angle kept). This is the
                      direction ⊂ block ⊂ chart pipeline; it is the arm that both
                      IDs topology and denoises curved factors at high noise.

The BSF baselines answer "subspace" (topology ``linear``, dim ``b``) for every
block — they have no curvature model — which is the point of the topology-ID
table. ``ours`` reads topology off the chart.

bsf.py is imported (review-verified), never reimplemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "bsf_baseline"))

import torch  # noqa: E402
from bsf import BSF, BSFConfig, TrainConfig, train_bsf  # noqa: E402

from metrics import RecoveredFactor  # noqa: E402


def matched_budget(dataset, block_size: int) -> tuple[int, int, int]:
    """(``n_blocks``, ``k_blocks``, ``L0``) matching the true generative budget:
    total decoder scalars ``Σ_g b_g · d`` and active code count ``L0 ≈ k · mean(b_g)``."""
    n_latent = sum(f.block_dim for f in dataset.factors)  # = Σ b_g (× d = scalars)
    n_blocks = max(1, n_latent // block_size)
    mean_b = float(np.mean([f.block_dim for f in dataset.factors]))
    l0 = int(round(dataset.k * mean_b))
    k_blocks = max(1, round(l0 / block_size))
    return n_blocks, k_blocks, k_blocks * block_size


def train_arm(
    dataset,
    *,
    mode: str,
    block_size: int,
    steps: int = 3000,
    lr: float = 3e-3,
    batch_size: int = 4096,
    aux_k_blocks: int = 4,
    seed: int = 0,
) -> BSF:
    """Fit a BSF (or TopK-SAE at ``block_size=1``) on the TRAIN split."""
    n_blocks, k_blocks, _l0 = matched_budget(dataset, block_size)
    cfg = BSFConfig(
        d_model=dataset.d,
        n_blocks=n_blocks,
        block_size=block_size,
        k_blocks=k_blocks,
        mode=mode,
        aux_k_blocks=min(aux_k_blocks, n_blocks),
        seed=seed,
    )
    model = BSF(cfg)
    xtr = torch.tensor(dataset.train.x, dtype=torch.float64)
    train_bsf(model, xtr, TrainConfig(steps=steps, batch_size=batch_size, lr=lr, seed=seed))
    return model


def train_sasa(dataset, *, block_size: int = 2, steps: int = 3000, lr: float = 3e-3,
               batch_size: int = 4096, aux_k_blocks: int = 4, seed: int = 0) -> BSF:
    """SASA-style arm (Subspace-Aware SAE, arXiv 2606.06333): a FREE encoder (like
    vanilla) but with **learned decoder SUBSPACES** — each block's decoder is
    re-orthonormalised (Stiefel) every step — plus block-level sparsity (block-
    TopK). It is the LLM-side cousin of BSF: subspace-aware but with no tied
    encoder and no curvature model. Reuses BSF's module (forward/decode/reproject);
    only the training loop differs from ``train_bsf`` (reproject in vanilla mode)."""
    n_blocks, k_blocks, _l0 = matched_budget(dataset, block_size)
    cfg = BSFConfig(
        d_model=dataset.d, n_blocks=n_blocks, block_size=block_size, k_blocks=k_blocks,
        mode="vanilla", aux_k_blocks=min(aux_k_blocks, n_blocks), seed=seed,
    )
    model = BSF(cfg)
    xtr = torch.tensor(dataset.train.x, dtype=torch.float64)
    n = xtr.shape[0]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    reproj_every = 20
    for step in range(steps):
        xb = xtr if batch_size >= n else xtr[torch.randint(0, n, (batch_size,), generator=gen)]
        out = model(xb)
        loss = ((out.x_hat - xb) ** 2).mean() + out.aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % reproj_every == 0:
            model.reproject_stiefel()  # subspace-aware: orthonormalise every block
    model.reproject_stiefel()
    model.eval()
    return model


@torch.no_grad()
def _extract_blocks(model: BSF, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(``z_sparse[N,G,b]``, ``active[N,G] bool``, ``decoder[G,b,d]``) on rows ``x``."""
    xt = torch.tensor(x, dtype=torch.float64)
    out = model(xt, update_util=False)
    zsp = out.z_sparse.cpu().numpy()
    active = out.mask.cpu().numpy() > 0
    dec = model.decoder.detach().cpu().numpy()
    return zsp, active, dec


def bsf_recovered(model: BSF, dataset, split: str = "test") -> list[RecoveredFactor]:
    """Every block is a recovered factor; BSF answers topology='linear' (subspace)
    for all of them (no curvature model) — the chance-level topology-ID baseline."""
    x = (dataset.test if split == "test" else dataset.train).x
    zsp, active, dec = _extract_blocks(model, x)
    n, g, b = zsp.shape
    recs: list[RecoveredFactor] = []
    for gi in range(g):
        contrib = zsp[:, gi, :] @ dec[gi]  # (n, d)
        act = active[:, gi]
        coord = zsp[:, gi, :].astype(np.float64).copy()
        coord[~act] = np.nan
        recs.append(
            RecoveredFactor(
                contribution=contrib,
                coord=coord,
                active=act,
                topology="linear",  # subspace — BSF has no topology model
                intrinsic_dim=b,
                embedding_dim=b,
                n_params=b * dataset.d,
                name=f"blk{gi}",
            )
        )
    return recs


# --------------------------------------------------------------------------- #
# OUR arm: per-block circle chart (ring classification + denoise).
# --------------------------------------------------------------------------- #
def _classify_block(codes: np.ndarray, *, ring_rel_std: float = 0.35, arc_gap: float = 0.9) -> dict:
    """Classify a b=2 block's active code cloud as a ring (circle/arc) or a blob
    (linear), from the RELATIVE radial spread. A clean circle sits at ~constant
    radius (small ``std(r)/mean(r)``); an isotropic Gaussian blob has Rayleigh
    radii (rel-std ≈ 0.52). Returns topology, intrinsic dim, and the ring radius."""
    if codes.shape[0] < 10 or codes.shape[1] != 2:
        return {"topology": "linear", "dim": codes.shape[1], "radius": None}
    r = np.linalg.norm(codes, axis=1)
    mean_r = float(r.mean())
    if mean_r < 1e-8:
        return {"topology": "linear", "dim": 2, "radius": None}
    rel_std = float(r.std() / mean_r)
    if rel_std >= ring_rel_std:
        return {"topology": "linear", "dim": 2, "radius": None}
    # Ring: circle vs open arc by the largest angular gap.
    ang = np.sort(np.arctan2(codes[:, 1], codes[:, 0]))
    gaps = np.diff(ang)
    wrap = (ang[0] + 2 * np.pi) - ang[-1]
    max_gap = float(max(gaps.max() if gaps.size else 0.0, wrap))
    topo = "arc" if max_gap > arc_gap * np.pi else "circle"
    return {"topology": topo, "dim": 1, "radius": float(np.median(r))}


def ours_recovered(model: BSF, dataset, split: str = "test") -> list[RecoveredFactor]:
    """Grassmannian block T1 (the trained ``model``) + a per-block K=1 circle chart.

    Chart params (topology + ring radius) are FIT on TRAIN active codes and applied
    on ``split``. On ring blocks the chart DENOISES: it projects each token's code
    onto the fitted circle (radius fixed to the learned R, angle kept), which
    removes the radial noise the block would otherwise reconstruct — the mechanism
    behind the R²-vs-σ crossing. On blob (linear) blocks the chart is the identity,
    so it never hurts the control factors.
    """
    ztr, atr, dec = _extract_blocks(model, dataset.train.x)
    x = (dataset.test if split == "test" else dataset.train).x
    zsp, active, _ = _extract_blocks(model, x)
    n, g, b = zsp.shape

    # Held-in guard subsample: whether ring-denoising a block REDUCES train
    # reconstruction MSE (so the chart is applied only when it helps — it can never
    # hurt the linear control factors, which is the honest claim).
    b_dec = model.b_dec.detach().cpu().numpy()
    m_tr = ztr.shape[0]
    rng = np.random.default_rng(model.cfg.seed + 7)
    sub = np.arange(m_tr) if m_tr <= 8000 else rng.choice(m_tr, 8000, replace=False)
    xtr_sub = dataset.train.x[sub]
    xhat_tr = np.einsum("ngb,gbd->nd", ztr[sub], dec) + b_dec  # (m, d)
    resid_tr = xtr_sub - xhat_tr

    recs: list[RecoveredFactor] = []
    for gi in range(g):
        cls = _classify_block(ztr[atr[:, gi], gi, :])
        code = zsp[:, gi, :]  # (n, b)
        act = active[:, gi]
        use_chart = False
        if cls["topology"] in ("circle", "arc") and cls["radius"] is not None:
            # Would projecting block gi's TRAIN codes onto the ring lower train MSE?
            ztr_g = ztr[sub, gi, :]
            th_tr = np.arctan2(ztr_g[:, 1], ztr_g[:, 0])
            ring_tr = cls["radius"] * np.stack([np.cos(th_tr), np.sin(th_tr)], axis=1)
            delta = (ztr_g - ring_tr) @ dec[gi]  # ambient change if we swap in the ring
            new_resid = resid_tr + delta
            use_chart = float((new_resid ** 2).sum()) <= float((resid_tr ** 2).sum())
        if use_chart:
            theta = np.arctan2(code[:, 1], code[:, 0])
            ring = cls["radius"] * np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n,2)
            contrib = ring @ dec[gi]  # denoised onto the curve
            rcoord = theta[:, None].astype(np.float64).copy()  # (n,1)
            topo, dim = cls["topology"], 1
        else:
            contrib = code @ dec[gi]  # linear / not-helpful: block unchanged
            rcoord = code.astype(np.float64).copy()  # (n,b)
            topo, dim = "linear", b
        rcoord[~act] = np.nan
        recs.append(
            RecoveredFactor(
                contribution=contrib,
                coord=rcoord,
                active=act,
                topology=topo,
                intrinsic_dim=dim,
                embedding_dim=b,  # the block's ambient span is still b even when
                #                  the intrinsic (ring) dim is 1
                # A ring chart codes 1 intrinsic coord (cheaper) but shares the b×d
                # block decoder; a blob keeps the full b-wide code.
                n_params=b * dataset.d,
                name=f"chart{gi}",
                meta={"ring_radius": cls["radius"], "charted": bool(use_chart)},
            )
        )
    return recs


ARMS = {
    "topk_sae": {"mode": "vanilla", "block_size": 1, "chart": False},
    "bsf_vanilla": {"mode": "vanilla", "block_size": 2, "chart": False},
    "bsf_grassmann": {"mode": "grassmann", "block_size": 2, "chart": False},
    "sasa": {"mode": "sasa", "block_size": 2, "chart": False},
    "ours": {"mode": "grassmann", "block_size": 2, "chart": True},
}


def run_arm(dataset, arm: str, *, steps: int = 3000, seed: int = 0) -> list[RecoveredFactor]:
    """Train one named arm and return its recovered factors on the TEST split."""
    spec = ARMS[arm]
    if spec["mode"] == "sasa":
        model = train_sasa(dataset, block_size=spec["block_size"], steps=steps, seed=seed)
    else:
        model = train_arm(
            dataset, mode=spec["mode"], block_size=spec["block_size"], steps=steps, seed=seed
        )
    if spec["chart"]:
        return ours_recovered(model, dataset, "test")
    return bsf_recovered(model, dataset, "test")
