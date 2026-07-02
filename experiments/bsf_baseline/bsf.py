"""Block-Sparse Featurizers (BSF) — a faithful torch reimplementation.

This is our from-scratch reimplementation of Goodfire's "Block-Sparse
Featurizers" as the head-to-head *baseline* against the gam curved-manifold
dictionary, and as a fast laptop-scale Tier-1 prototype. Pure python/torch, no
gam/gamfit dependency.

The one idea BSF adds to a TopK SAE: the latent code is carved into ``G`` blocks
of width ``b``, and sparsity is enforced *per block* rather than per scalar
latent. We keep the ``k`` blocks with the largest L2 norm ``‖z_g‖₂`` and zero
the rest (block-TopK). Because the surviving code within a block is a full
signed ``b``-vector (NO ReLU anywhere), each block represents a whole
``b``-dimensional *subspace* (a linear chart), not a one-sided cone the way a
ReLU latent does. A curved feature such as the weekday circle — intrinsically
1-D but extrinsically ≥2-D — can therefore live inside a single block.

Three models, one code path:

* **Vanilla BSF** (:class:`BSF` with ``mode="vanilla"``) — free encoder
  ``W_enc`` and free block decoder ``D``; per-latent unit-norm decoder rows.
* **Grassmannian BSF** (``mode="grassmann"``) — *tied* encoder
  ``z_g = γ · x Dᵀ`` with a single learned scalar ``γ``, and every block
  decoder ``D_g`` held **column-orthonormal** (a point on the Stiefel manifold
  ``St(b, d)``), re-projected by QR every ``reproj_every`` steps. Then
  ``z_g D_g = γ·Pₘ x`` is exactly ``γ`` times the orthogonal projection of ``x``
  onto block ``g``'s subspace — a Grassmannian featurizer.
* **TopK-SAE baseline** — literally ``mode="vanilla", b=1``: block-TopK with
  block width 1 is signed TopK on the scalar latents. This is the comparison
  column, trained on an identical budget.

Both models optionally carry an **AuxK dead-block loss**: the residual is
re-reconstructed from the ``k_aux`` blocks with the lowest recent utilization,
resurrecting dead blocks (Gao et al. AuxK, adapted from scalar latents to
blocks).

See ``train.py`` for the synthetic planted-subspace recovery test, the real
activation EV sweep, and the weekday/month cyclic-feature block finding.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from torch import nn


# ==========================================================================
# Block-TopK
# ==========================================================================
def block_topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean ``(N, G)`` mask keeping the ``k`` blocks with largest ‖z_g‖₂.

    ``z`` is ``(N, G, b)``. Ties broken by ``torch.topk``'s ordering. The mask
    is returned as a float gate to be broadcast over the block width; it carries
    no gradient (block *selection* is treated as constant, exactly as a scalar
    TopK SAE treats its top-k indexing — gradients flow through the surviving
    code values, not the choice of which blocks survive).
    """
    n, g, _ = z.shape
    if k >= g:
        return z.new_ones((n, g))
    norms = torch.linalg.vector_norm(z, dim=2)  # (N, G)
    idx = torch.topk(norms, k, dim=1).indices  # (N, k)
    mask = torch.zeros_like(norms)
    mask.scatter_(1, idx, 1.0)
    return mask


# ==========================================================================
# Config + model
# ==========================================================================
@dataclass
class BSFConfig:
    d_model: int
    n_blocks: int  # G
    block_size: int  # b
    k_blocks: int  # active blocks per token (block-TopK)
    mode: str = "vanilla"  # "vanilla" | "grassmann"
    # AuxK dead-block auxiliary reconstruction
    aux_k_blocks: int = 0  # 0 disables AuxK
    aux_coef: float = 1.0 / 32.0
    dead_ema: float = 0.999  # utilization EMA decay
    # Grassmannian Stiefel re-projection cadence (optimizer steps)
    reproj_every: int = 20
    seed: int = 0

    @property
    def n_latent(self) -> int:
        return self.n_blocks * self.block_size

    @property
    def n_decoder_params(self) -> int:
        return self.n_latent * self.d_model


class BSF(nn.Module):
    """One Block-Sparse Featurizer covering vanilla, Grassmannian and TopK-SAE.

    Decoder ``D`` is stored as ``(G, b, d)``; block ``g``'s decoder ``D_g`` is a
    ``(b, d)`` slice whose ``b`` rows span its subspace. A learned pre-decoder
    bias ``b_dec`` is subtracted before encoding and added after decoding
    (standard SAE centering — materially lifts EV on real activations).
    """

    def __init__(self, cfg: BSFConfig):
        super().__init__()
        self.cfg = cfg
        g, b, d = cfg.n_blocks, cfg.block_size, cfg.d_model
        gen = torch.Generator().manual_seed(cfg.seed)

        # Decoder blocks (G, b, d). Init column-orthonormal in both modes so the
        # Grassmannian model starts on the Stiefel manifold and vanilla starts
        # from a well-conditioned block.
        dec = torch.empty(g, b, d, dtype=torch.float64)
        for gi in range(g):
            m = torch.randn(d, b, generator=gen, dtype=torch.float64)
            q, _ = torch.linalg.qr(m)  # (d, b) orthonormal columns
            dec[gi] = q.T
        self.decoder = nn.Parameter(dec)
        self.b_dec = nn.Parameter(torch.zeros(d, dtype=torch.float64))

        if cfg.mode == "vanilla":
            # Free encoder (d, G*b); init as decoder transpose (tied-ish start).
            self.encoder = nn.Parameter(dec.reshape(g * b, d).T.clone())
            self.enc_bias = nn.Parameter(torch.zeros(g * b, dtype=torch.float64))
            self.log_gamma = None
        elif cfg.mode == "grassmann":
            # Tied encoder: z = γ·(x-b_dec) Dᵀ. Single learned scalar γ.
            self.encoder = None
            self.enc_bias = None
            self.log_gamma = nn.Parameter(torch.zeros((), dtype=torch.float64))
        else:  # pragma: no cover - guarded config
            raise ValueError(f"unknown mode {cfg.mode!r}")

        # Utilization EMA over blocks (fraction of recent tokens each block was
        # active). Buffer, not a parameter — drives AuxK dead-block selection.
        self.register_buffer("util_ema", torch.zeros(g, dtype=torch.float64))
        self._step = 0

    # -- encode ------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-TopK block codes ``z`` of shape ``(N, G, b)``."""
        g, b = self.cfg.n_blocks, self.cfg.block_size
        xc = x - self.b_dec
        if self.cfg.mode == "vanilla":
            z = xc @ self.encoder + self.enc_bias  # (N, G*b)
        else:
            dflat = self.decoder.reshape(g * b, self.cfg.d_model)  # (G*b, d)
            z = torch.exp(self.log_gamma) * (xc @ dflat.T)  # (N, G*b)
        return z.reshape(-1, g, b)

    # -- decode ------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Additive block decode ``x_hat = Σ_g z_g D_g + b_dec``."""
        g, b, d = self.cfg.n_blocks, self.cfg.block_size, self.cfg.d_model
        recon = z.reshape(-1, g * b) @ self.decoder.reshape(g * b, d)
        return recon + self.b_dec

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, update_util: bool = True) -> "BSFOutput":
        z = self.encode(x)
        mask = block_topk_mask(z, self.cfg.k_blocks)  # (N, G)
        z_sparse = z * mask.unsqueeze(-1)
        x_hat = self.decode(z_sparse)

        aux_loss = x.new_zeros(())
        if self.cfg.aux_k_blocks > 0 and self.training:
            aux_loss = self._auxk_loss(x, x_hat.detach(), z)

        if update_util and self.training:
            with torch.no_grad():
                self.util_ema.mul_(self.cfg.dead_ema).add_(
                    mask.mean(0).double() * (1.0 - self.cfg.dead_ema)
                )
        return BSFOutput(x_hat=x_hat, z=z, z_sparse=z_sparse, mask=mask, aux_loss=aux_loss)

    def _auxk_loss(self, x: torch.Tensor, x_hat_detached: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """AuxK: reconstruct the residual using the ``k_aux`` least-used blocks.

        Adapted from Gao et al. — instead of the ``k_aux`` lowest-utilization
        *scalar* latents, we resurrect the ``k_aux`` lowest-utilization *blocks*
        with their full ``b``-wide code, fitting the current residual
        ``x - x_hat`` (x_hat detached so AuxK only trains the dead blocks, not
        the main path).
        """
        k_aux = min(self.cfg.aux_k_blocks, self.cfg.n_blocks)
        dead_idx = torch.topk(self.util_ema, k_aux, largest=False).indices  # (k_aux,)
        dead_mask = torch.zeros_like(self.util_ema)
        dead_mask[dead_idx] = 1.0
        z_dead = z * dead_mask.reshape(1, -1, 1)
        # decode dead blocks only, WITHOUT the bias (fitting a residual)
        g, b, d = self.cfg.n_blocks, self.cfg.block_size, self.cfg.d_model
        aux_recon = z_dead.reshape(-1, g * b) @ self.decoder.reshape(g * b, d)
        residual = x - x_hat_detached
        return self.cfg.aux_coef * ((residual - aux_recon) ** 2).mean()

    # -- Grassmannian Stiefel retraction -----------------------------------
    @torch.no_grad()
    def reproject_stiefel(self) -> None:
        """Re-orthonormalize every block decoder onto the Stiefel manifold.

        For block ``g`` we want the ``b`` rows of ``D_g`` orthonormal. QR of
        ``D_gᵀ`` gives ``(d,b)`` orthonormal columns ``Q``; set ``D_g = Qᵀ``.
        The sign convention (positive ``R`` diagonal) keeps the retraction close
        to the current point rather than flipping frames.
        """
        if self.cfg.mode != "grassmann":
            return
        d = self.decoder.data  # (G, b, d)
        for gi in range(self.cfg.n_blocks):
            q, r = torch.linalg.qr(d[gi].T)  # (d,b),(b,b)
            sign = torch.sign(torch.diagonal(r))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            d[gi] = (q * sign).T

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Vanilla: renormalize decoder rows (per-latent) to unit L2 norm.

        The standard SAE gauge fix — the encoder can rescale freely, so we pin
        the decoder scale. (Grassmannian pins scale via orthonormality instead.)
        """
        if self.cfg.mode != "vanilla":
            return
        norms = torch.linalg.vector_norm(self.decoder.data, dim=2, keepdim=True)
        self.decoder.data /= norms.clamp_min(1e-8)

    def maybe_retract(self) -> None:
        """Call once per optimizer step; applies the mode's gauge fix."""
        self._step += 1
        if self.cfg.mode == "grassmann":
            if self._step % self.cfg.reproj_every == 0:
                self.reproject_stiefel()
        else:
            self.normalize_decoder()


@dataclass
class BSFOutput:
    x_hat: torch.Tensor
    z: torch.Tensor  # pre-topk codes (N, G, b)
    z_sparse: torch.Tensor  # post-topk codes (N, G, b)
    mask: torch.Tensor  # block gate (N, G)
    aux_loss: torch.Tensor


# ==========================================================================
# Training
# ==========================================================================
@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 512
    lr: float = 3e-3
    reproj_every: int = 20
    log_every: int = 500
    seed: int = 0


def train_bsf(
    model: BSF,
    X: torch.Tensor,
    tcfg: TrainConfig,
    X_val: torch.Tensor | None = None,
    verbose: bool = False,
) -> dict:
    """Train a BSF by minimizing MSE reconstruction (+ AuxK) with Adam.

    Full-batch when ``batch_size >= N`` (the tiny weekday/month / synthetic
    cases); minibatched otherwise. Returns a small history dict.
    """
    model.cfg.reproj_every = tcfg.reproj_every
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    n = X.shape[0]
    gen = torch.Generator().manual_seed(tcfg.seed)
    hist = {"loss": [], "recon": [], "aux": [], "step": []}
    for step in range(tcfg.steps):
        if tcfg.batch_size >= n:
            xb = X
        else:
            idx = torch.randint(0, n, (tcfg.batch_size,), generator=gen)
            xb = X[idx]
        out = model(xb)
        recon = ((out.x_hat - xb) ** 2).mean()
        loss = recon + out.aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.maybe_retract()
        if verbose and (step % tcfg.log_every == 0 or step == tcfg.steps - 1):
            hist["step"].append(step)
            hist["loss"].append(float(loss))
            hist["recon"].append(float(recon))
            hist["aux"].append(float(out.aux_loss))
            v = ""
            if X_val is not None:
                v = f" val_ev={ev(model, X_val):.4f}"
            print(f"  step {step:5d} recon={float(recon):.5f} aux={float(out.aux_loss):.5f}{v}", flush=True)
    model.eval()
    return hist


# ==========================================================================
# Metrics
# ==========================================================================
@torch.no_grad()
def reconstruct(model: BSF, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(X, update_util=False).x_hat


def _ev_np(x: np.ndarray, xhat: np.ndarray) -> float:
    """Explained variance == R² for a reconstruction (1 - SSE/SST, SST about the mean)."""
    sst = float(((x - x.mean(0)) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - ((x - xhat) ** 2).sum() / sst)


@torch.no_grad()
def ev(model: BSF, X: torch.Tensor) -> float:
    xhat = reconstruct(model, X)
    return _ev_np(X.cpu().numpy(), xhat.cpu().numpy())


def stable_rank(a: np.ndarray) -> float:
    """Stable (numerical) rank ‖A‖_F² / ‖A‖₂² — an effective dimensionality that,
    unlike the exact rank, is robust to a heavy singular-value tail."""
    if a.size == 0:
        return 0.0
    s = np.linalg.svd(a, compute_uv=False)
    top = float(s[0])
    if top <= 0:
        return 0.0
    return float((s ** 2).sum() / (top ** 2))


@torch.no_grad()
def block_diagnostics(model: BSF, X: torch.Tensor) -> dict:
    """Per-block stable rank, utilization, and activation frequency.

    * ``stable_rank`` — of block ``g``'s *contribution* matrix ``A_g``
      (``(N, d)``: each active token's decoded contribution ``z_g D_g``). This
      is the paper's per-block stable-rank statistic (they report ≈3).
    * ``utilization`` — participation ratio of the within-block code covariance,
      divided by ``b``: the fraction of the block's ``b`` dims that carry
      variance across the tokens that activate it.
    * ``active_freq`` — fraction of tokens for which the block is in the top-k.
    """
    model.eval()
    out = model(X, update_util=False)
    z_sparse = out.z_sparse.cpu().numpy()  # (N, G, b)
    mask = out.mask.cpu().numpy()  # (N, G)
    dec = model.decoder.detach().cpu().numpy()  # (G, b, d)
    g, b = model.cfg.n_blocks, model.cfg.block_size

    per_block = []
    for gi in range(g):
        active = mask[:, gi] > 0
        n_active = int(active.sum())
        codes = z_sparse[active, gi, :]  # (n_active, b)
        contrib = codes @ dec[gi]  # (n_active, d)
        sr = stable_rank(contrib) if n_active > 1 else 0.0
        # within-block utilization: participation ratio of code covariance / b
        if n_active > 1 and b > 1:
            cov = np.cov(codes, rowvar=False)
            lam = np.linalg.eigvalsh(cov).clip(min=0)
            pr = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-30)
            util = float(pr / b)
        elif b == 1:
            util = 1.0 if n_active > 1 else 0.0
        else:
            util = 0.0
        per_block.append(
            {
                "block": gi,
                "active_freq": n_active / X.shape[0],
                "stable_rank": sr,
                "utilization": util,
                "energy": float((contrib ** 2).sum()),
            }
        )
    active_blocks = [pb for pb in per_block if pb["active_freq"] > 0]
    srs = [pb["stable_rank"] for pb in active_blocks if pb["stable_rank"] > 0]
    utils = [pb["utilization"] for pb in active_blocks if pb["utilization"] > 0]
    return {
        "per_block": per_block,
        "mean_stable_rank": float(np.mean(srs)) if srs else 0.0,
        "mean_utilization": float(np.mean(utils)) if utils else 0.0,
        "n_active_blocks": len(active_blocks),
        "n_blocks": g,
    }


# ==========================================================================
# Shard-format loader (the on-disk contract from gam residual_shard_io.py)
# ==========================================================================
_SHARD_DTYPE = np.dtype("<u2")


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_shard_harvest(out_dir: str, max_rows: int | None = None) -> np.ndarray:
    """Read a gam ``residual_shard_bf16`` harvest directory into float32 ``(N, d)``.

    Accepts the exact on-disk format written by gam's ``ShardWriter`` (a
    ``manifest.json`` plus little-endian uint16 bf16 shard files). Kept
    dependency-free so this baseline never imports gam. ``max_rows`` caps the
    number of rows materialized (laptop memory).
    """
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    d_model = int(manifest["d_model"])
    chunks: list[np.ndarray] = []
    have = 0
    for shard in manifest["shards"]:
        rows = int(shard["rows"])
        mm = np.memmap(
            os.path.join(out_dir, shard["file"]), dtype=_SHARD_DTYPE, mode="r"
        ).reshape(rows, d_model)
        take = rows if max_rows is None else min(rows, max_rows - have)
        chunks.append(_bf16_bits_to_f32(np.asarray(mm[:take])))
        have += take
        if max_rows is not None and have >= max_rows:
            break
    if not chunks:
        return np.empty((0, d_model), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ==========================================================================
# PCA reduction (train-only) — shared by every eval that reduces d for speed
# ==========================================================================
def pca_reduce(train: np.ndarray, test: np.ndarray, r: int):
    """Train-only centering + PCA to ``r`` comps. Returns (tr, te, mu, Vt)."""
    mu = train.mean(0)
    tc = train - mu
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    vt = vt[: min(r, vt.shape[0])]
    return tc @ vt.T, (test - mu) @ vt.T, mu, vt


def iterate_batches(X: np.ndarray, n: int, seed: int = 0) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    for i in range(0, len(order), n):
        yield X[order[i : i + n]]
