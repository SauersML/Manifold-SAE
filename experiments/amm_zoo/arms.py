"""Featurizer arms for the AMM zoo.

The four comparison arms use the review-verified BSF implementation. ``ours``
is the production :func:`gamfit.sae_manifold_fit` model: the topology family is
discovered from the training activations, routing is Top-k at the planted AMM
sparsity, and per-atom held-out contributions come from the native frozen-
decoder OOS solve.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "bsf_baseline"))

from bsf import BSF, BSFConfig  # noqa: E402
from metrics import RecoveredFactor  # noqa: E402


def matched_budget(dataset, block_size: int) -> tuple[int, int, int]:
    """Return ``(n_blocks, active_blocks, active_scalar_coordinates)``.

    Decoder rows match the planted frame budget ``sum(b_g)`` and active scalar
    coordinates match ``k * mean(b_g)``. The AMM zoo's total width is divisible
    by every block size used by the benchmark.
    """
    n_latent = sum(f.block_dim for f in dataset.factors)
    if n_latent % block_size:
        raise ValueError(
            f"planted decoder width {n_latent} is not divisible by block_size={block_size}"
        )
    n_blocks = n_latent // block_size
    l0 = int(round(dataset.k * np.mean([f.block_dim for f in dataset.factors])))
    k_blocks = max(1, round(l0 / block_size))
    return n_blocks, k_blocks, k_blocks * block_size


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _reproject_blocks(model: BSF) -> None:
    """Batched Stiefel projection for both Grassmann and SASA decoders."""
    q, r = torch.linalg.qr(model.decoder.transpose(1, 2))
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    model.decoder.copy_((q * sign.unsqueeze(1)).transpose(1, 2))


def train_bsf_arm(
    dataset,
    *,
    mode: str,
    block_size: int,
    steps: int,
    batch_size: int,
    lr: float = 3e-3,
    aux_k_blocks: int = 4,
    seed: int = 0,
) -> BSF:
    """Train one BSF-family arm in FP32 on CUDA when available.

    ``mode='sasa'`` keeps the free encoder but projects every decoder block to
    Stiefel. This projection is implemented here because ``BSF.reproject_stiefel``
    intentionally no-ops for a vanilla/free-encoder model.
    """
    n_blocks, k_blocks, _ = matched_budget(dataset, block_size)
    bsf_mode = "vanilla" if mode == "sasa" else mode
    cfg = BSFConfig(
        d_model=dataset.d,
        n_blocks=n_blocks,
        block_size=block_size,
        k_blocks=k_blocks,
        mode=bsf_mode,
        aux_k_blocks=min(aux_k_blocks, n_blocks),
        seed=seed,
    )
    device = _device()
    model = BSF(cfg).to(device=device, dtype=torch.float32)
    xtr = torch.as_tensor(dataset.train.x, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device.type).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = xtr.shape[0]

    model.train()
    for step in range(steps):
        if batch_size >= n:
            xb = xtr
        else:
            rows = torch.randint(n, (batch_size,), generator=generator, device=device)
            xb = xtr[rows]
        out = model(xb)
        loss = ((out.x_hat - xb) ** 2).mean() + out.aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if mode in ("grassmann", "sasa"):
            if (step + 1) % cfg.reproj_every == 0:
                _reproject_blocks(model)
        else:
            with torch.no_grad():
                norm = torch.linalg.vector_norm(model.decoder, dim=2, keepdim=True)
                model.decoder.div_(norm.clamp_min(1e-8))

    if mode in ("grassmann", "sasa"):
        _reproject_blocks(model)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.eval()
    return model


@torch.inference_mode()
def _extract_blocks(
    model: BSF, x: np.ndarray, *, chunk_size: int = 65_536
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse block codes, activity, and decoder without a full GPU copy."""
    n = x.shape[0]
    g, b = model.cfg.n_blocks, model.cfg.block_size
    codes = np.empty((n, g, b), dtype=np.float32)
    active = np.empty((n, g), dtype=bool)
    device = model.decoder.device
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        xb = torch.as_tensor(x[lo:hi], dtype=torch.float32, device=device)
        out = model(xb, update_util=False)
        codes[lo:hi] = out.z_sparse.cpu().numpy()
        active[lo:hi] = out.mask.cpu().numpy() > 0
    decoder = model.decoder.detach().cpu().numpy().astype(np.float32, copy=False)
    return codes, active, decoder


def bsf_recovered(model: BSF, dataset, split: str = "test") -> list[RecoveredFactor]:
    """Convert every learned BSF block into one held-out recovered factor."""
    x = (dataset.test if split == "test" else dataset.train).x
    codes, active, decoder = _extract_blocks(model, x)
    _, g, b = codes.shape
    recovered: list[RecoveredFactor] = []
    for block in range(g):
        contribution = codes[:, block] @ decoder[block]
        contribution[~active[:, block]] = 0.0
        coord = codes[:, block].copy()
        coord[~active[:, block]] = np.nan
        recovered.append(
            RecoveredFactor(
                contribution=contribution,
                coord=coord,
                active=active[:, block],
                topology="linear",
                intrinsic_dim=b,
                embedding_dim=b,
                n_params=b * dataset.d,
                name=f"blk{block}",
            )
        )
    return recovered


def _hybrid_linear_images(model: Any) -> dict[int, dict[str, Any]]:
    """Read the model's persisted load-bearing curved/linear verdicts."""
    report = model.hybrid_split
    if not report:
        return {}
    images: dict[int, dict[str, Any]] = {}
    for entry in report.get("atoms", []):
        image = entry.get("linear_image")
        if image:
            images[int(image["atom_idx"])] = image
    return images


def _effective_embedding_dim(contribution: np.ndarray, active: np.ndarray) -> int:
    rows = contribution[active]
    if rows.shape[0] < 3:
        return 0
    if rows.shape[0] > 4096:
        take = np.linspace(0, rows.shape[0] - 1, 4096, dtype=int)
        rows = rows[take]
    centered = rows.astype(np.float64) - rows.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        return 0
    return max(1, int(np.count_nonzero(singular >= 0.08 * singular[0])))


def _topology_from_atom(
    basis: str, intrinsic_dim: int, embedding_dim: int, hybrid_linear: bool
) -> str:
    if hybrid_linear or basis in {"linear", "linear_block"}:
        return "linear"
    if basis == "periodic":
        return "circle"
    if basis in {"torus", "sphere", "mobius", "cylinder"}:
        return basis
    if basis in {"duchon", "euclidean"}:
        if intrinsic_dim == 1:
            return "helix" if embedding_dim >= 3 else "arc"
        return "linear" if embedding_dim <= intrinsic_dim else "euclidean"
    return basis


def _coord_periods(basis: str, intrinsic_dim: int) -> list[float | None]:
    if basis in {"periodic", "torus"}:
        return [1.0] * intrinsic_dim
    if basis == "sphere" and intrinsic_dim == 2:
        return [None, float(2.0 * np.pi)]
    if basis == "mobius" and intrinsic_dim == 2:
        return [2.0, None]
    if basis == "cylinder" and intrinsic_dim == 2:
        return [1.0, None]
    return [None] * intrinsic_dim


def train_manifold_sae(dataset, *, n_iter: int, seed: int) -> Any:
    """Fit the production all-zoo Manifold SAE on the training split."""
    import gamfit

    return gamfit.sae_manifold_fit(
        np.ascontiguousarray(dataset.train.x, dtype=np.float64),
        K=dataset.G,
        d_atom=2,
        atom_topology="auto",
        assignment="topk",
        top_k=dataset.k,
        n_iter=n_iter,
        random_state=seed,
    )


def manifold_sae_recovered(
    model: Any, dataset, *, split: str = "test", oos_batch_size: int = 10_000
) -> list[RecoveredFactor]:
    """Read exact effective OOS images and coordinates from the public model state."""
    x = (dataset.test if split == "test" else dataset.train).x
    n = x.shape[0]
    k = int(model.chosen_k)
    basis = list(model.basis_kinds)
    dims = [int(v) for v in model.atom_dims]
    decoder_params = [int(np.asarray(v).size) for v in model.decoder_blocks]
    if not (len(basis) == len(dims) == len(decoder_params) == k):
        raise RuntimeError("ManifoldSAE atom metadata lengths disagree with chosen_k")

    contributions = [np.zeros((n, dataset.d), dtype=np.float32) for _ in range(k)]
    coords = [np.full((n, dims[i]), np.nan, dtype=np.float64) for i in range(k)]
    active = [np.zeros(n, dtype=bool) for _ in range(k)]
    hybrid = _hybrid_linear_images(model)

    for lo in range(0, n, oos_batch_size):
        hi = min(lo + oos_batch_size, n)
        payload = dict(
            model.converged_latents(np.ascontiguousarray(x[lo:hi], dtype=np.float64))
        )
        assignments = np.asarray(payload["assignments"], dtype=np.float64)
        coord_blocks = list(payload["coords"])
        atom_images = list(payload["atom_images"])
        if assignments.shape != (hi - lo, k):
            raise RuntimeError(
                f"OOS assignments have shape {assignments.shape}; expected {(hi - lo, k)}"
            )
        if len(coord_blocks) != k or len(atom_images) != k:
            raise RuntimeError(
                "OOS public payload atom lengths disagree with chosen_k: "
                f"coords={len(coord_blocks)} images={len(atom_images)} chosen_k={k}"
            )

        for atom_index in range(k):
            assignment = assignments[:, atom_index]
            atom_coord = np.asarray(coord_blocks[atom_index], dtype=np.float64)
            atom_image = np.asarray(atom_images[atom_index], dtype=np.float64)
            if atom_coord.shape != (hi - lo, dims[atom_index]):
                raise RuntimeError(
                    f"OOS atom {atom_index} coords have shape {atom_coord.shape}; "
                    f"expected {(hi - lo, dims[atom_index])}"
                )
            if atom_image.shape != (hi - lo, dataset.d):
                raise RuntimeError(
                    f"OOS atom {atom_index} image has shape {atom_image.shape}; "
                    f"expected {(hi - lo, dataset.d)}"
                )
            contribution = assignment[:, None] * atom_image
            fired = np.abs(assignment) > 1e-8
            contribution[~fired] = 0.0
            contributions[atom_index][lo:hi] = contribution.astype(np.float32)
            active[atom_index][lo:hi] = fired
            coords[atom_index][lo:hi][fired] = atom_coord[fired, : dims[atom_index]]

    recovered: list[RecoveredFactor] = []
    for atom_index in range(k):
        embedding_dim = _effective_embedding_dim(contributions[atom_index], active[atom_index])
        topology = _topology_from_atom(
            basis[atom_index], dims[atom_index], embedding_dim, atom_index in hybrid
        )
        recovered.append(
            RecoveredFactor(
                contribution=contributions[atom_index],
                coord=coords[atom_index],
                active=active[atom_index],
                topology=topology,
                intrinsic_dim=dims[atom_index],
                embedding_dim=embedding_dim,
                n_params=decoder_params[atom_index],
                name=f"msae{atom_index}",
                meta={
                    "basis_kind": basis[atom_index],
                    "coord_periods": _coord_periods(basis[atom_index], dims[atom_index]),
                    "hybrid_linear": atom_index in hybrid,
                },
            )
        )
    return recovered


ARMS = {
    "topk_sae": {"mode": "vanilla", "block_size": 1},
    "bsf_vanilla": {"mode": "vanilla", "block_size": 2},
    "bsf_grassmann": {"mode": "grassmann", "block_size": 2},
    "sasa": {"mode": "sasa", "block_size": 2},
    "ours": {"mode": "manifold_sae"},
}


def run_arm(
    dataset,
    arm: str,
    *,
    steps: int,
    batch_size: int,
    manifold_iters: int,
    manifold_oos_batch: int,
    seed: int,
    timing: dict[str, Any] | None = None,
) -> list[RecoveredFactor]:
    """Train one arm and return its held-out recovered factors."""
    timing = {} if timing is None else timing
    spec = ARMS[arm]
    t0 = time.perf_counter()
    if spec["mode"] == "manifold_sae":
        model = train_manifold_sae(dataset, n_iter=manifold_iters, seed=seed)
        timing["fit_s"] = round(time.perf_counter() - t0, 6)
        t1 = time.perf_counter()
        recovered = manifold_sae_recovered(
            model, dataset, split="test", oos_batch_size=manifold_oos_batch
        )
        timing["extract_oos_s"] = round(time.perf_counter() - t1, 6)
        timing["chosen_k"] = int(model.chosen_k)
        timing["device"] = "native_cpu"
        return recovered

    model = train_bsf_arm(
        dataset,
        mode=spec["mode"],
        block_size=spec["block_size"],
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    timing["fit_s"] = round(time.perf_counter() - t0, 6)
    timing["device"] = str(model.decoder.device)
    t1 = time.perf_counter()
    recovered = bsf_recovered(model, dataset, "test")
    timing["extract_oos_s"] = round(time.perf_counter() - t1, 6)
    return recovered
