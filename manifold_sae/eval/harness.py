"""Reusable SAE evaluation harness for the Manifold-SAE project.

Goal
----
Score any SAE-like model on a uniform battery of metrics:

  1. Reconstruction R^2 on a validation set.
  2. Sparsity: mean active fraction, L0, L1, gini coefficient.
  3. Dead-atom fraction.
  4. Feature-absorption score (Bussmann 2024 style): for each labeled concept
     c, find the atom a* that most predicts c; then for inputs containing c,
     measure the fraction where a* fails to fire while *some* other atom
     does fire and partially predicts c. This is the "absorbed by a sibling"
     rate; higher = more absorption.
  5. HSV-axis coherence of the top-20 activating colors per atom (low
     circular std of hue + low SV spread = coherent).
  6. Manifold-dim of top-20 colors per atom, estimated as PCA effective rank
     (Shannon entropy of normalized eigenvalues). ~1 = curve atom,
     ~0 = point atom, larger = scattered.
  7. Causal ablation Delta-R^2: zero each atom's contribution one at a time,
     measure the resulting drop in validation R^2.
  8. Probe accuracy: linear probe from atom activations to HSV / modifier
     count / monoword labels; report R^2 / accuracy per target.
  9. Steering quality: pick the k atoms most correlated with hue, push their
     activations by +1 sigma, re-decode, and measure the correlation between
     the intended hue push and the observed hue delta of the decoded vector.
 10. Compute-normalized scores: R^2 per FLOP and R^2 per active atom.

Design
------
Models speak via an ``SAEWrapper`` adapter so the harness is decoupled from
each architecture. Adding a new architecture = write a loader in
``registry.py`` that returns an ``SAEWrapper``. The harness never imports
model code directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class SAEWrapper(ABC):
    """Uniform interface over every SAE variant we evaluate.

    ``encode(x)`` returns per-atom activations of shape (B, F) with a
    consistent firing-magnitude convention (>=0 for firing, 0 for off). The
    harness uses ``> firing_threshold`` to decide "firing".

    ``decode_from_activations(z)`` is needed for ablation + steering. It must
    accept the same ``z`` shape that ``encode`` returns and produce a
    reconstruction (B, D).

    ``flops_per_token`` is an approximate count for compute-normalized
    scoring (rough order of magnitude, not exact).
    """

    name: str
    n_features: int
    input_dim: int
    firing_threshold: float = 1e-3

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode_from_activations(self, z: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def flops_per_token(self) -> float:
        # Default: 2*D*F encode + 2*F*D decode = 4*D*F.
        return 4.0 * float(self.input_dim) * float(self.n_features)


# ---------------------------------------------------------------------------
# Metrics (pure-numpy where possible so the harness is light)
# ---------------------------------------------------------------------------


def _to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x.to(device)


def reconstruction_r2(
    model: SAEWrapper,
    X: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    """1 - residual var / total var, averaged per-dim then per-row."""
    device = X.device
    total_sse = 0.0
    total_sst = 0.0
    mean = X.mean(0, keepdim=True)
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        with torch.no_grad():
            recon = model.reconstruct(xb)
        total_sse += float(((xb - recon) ** 2).sum().item())
        total_sst += float(((xb - mean) ** 2).sum().item())
    if total_sst <= 0:
        return float("nan")
    return 1.0 - total_sse / total_sst


def collect_activations(
    model: SAEWrapper,
    X: torch.Tensor,
    batch_size: int = 1024,
) -> np.ndarray:
    out = []
    for i in range(0, X.shape[0], batch_size):
        with torch.no_grad():
            z = model.encode(X[i:i + batch_size])
        out.append(z.detach().cpu().numpy())
    return np.concatenate(out, 0)


def sparsity_stats(acts: np.ndarray, threshold: float) -> dict[str, float]:
    fires = acts > threshold
    mean_active = float(fires.mean())
    l0_per_row = fires.sum(1).mean()
    l1_per_row = np.abs(acts).sum(1).mean()
    # gini coefficient over per-atom average firing magnitude.
    per_atom = np.abs(acts).mean(0)
    g = _gini(per_atom)
    return {
        "mean_active_fraction": mean_active,
        "L0": float(l0_per_row),
        "L1": float(l1_per_row),
        "gini": float(g),
    }


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        x = x - x.min()
    s = x.sum()
    if s <= 1e-12:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    # standard formula
    return float((n + 1 - 2 * (cum.sum() / s)) / n)


def dead_atom_fraction(acts: np.ndarray, threshold: float) -> float:
    fires_per_atom = (acts > threshold).mean(0)
    return float((fires_per_atom < 1e-5).mean())


# ---------------------------------------------------------------------------
# Concept-level metrics (need labels)
# ---------------------------------------------------------------------------


def hsv_coherence_top20(
    acts: np.ndarray,
    row_color_idx: np.ndarray,
    color_hsv: np.ndarray,
    top_k_atoms: int = 20,
    top_k_colors: int = 20,
) -> dict[str, Any]:
    """For top-20 most-active atoms, measure HSV coherence of the
    ``top_k_colors`` colors that most activate them.

    Returns mean coherence (higher = more coherent), per-atom details.
    """
    F = acts.shape[1]
    by_color = _aggregate_by_color(acts, row_color_idx, n_colors=color_hsv.shape[0])
    active_per_atom = (acts > 0).mean(0)
    top_atoms = np.argsort(-active_per_atom)[:top_k_atoms]
    per_atom = []
    coherences = []
    for k in top_atoms:
        scores = by_color[:, k]
        top_ci = np.argsort(-scores)[:top_k_colors]
        hsv = color_hsv[top_ci]
        coh = _hsv_coherence(hsv)
        coherences.append(coh)
        per_atom.append({"atom": int(k), "coherence": coh, "top_colors_idx": top_ci.tolist()})
    return {
        "mean_top20_coherence": float(np.mean(coherences)) if coherences else float("nan"),
        "per_atom": per_atom,
    }


def _aggregate_by_color(acts: np.ndarray, row_color_idx: np.ndarray, n_colors: int) -> np.ndarray:
    F = acts.shape[1]
    out = np.zeros((n_colors, F), dtype=np.float64)
    counts = np.zeros(n_colors, dtype=np.int64)
    np.add.at(out, row_color_idx, acts.astype(np.float64))
    np.add.at(counts, row_color_idx, 1)
    out /= np.maximum(counts[:, None], 1)
    return out


def _hsv_coherence(hsv: np.ndarray) -> float:
    """1 - (0.5 * circular variance of hue + 0.5 * mean SV std).

    Range roughly [0, 1]; higher = more coherent.
    """
    hue_rad = hsv[:, 0] * 2 * np.pi
    R = np.sqrt(np.cos(hue_rad).mean() ** 2 + np.sin(hue_rad).mean() ** 2)
    circ_var = 1.0 - R
    sv_std = hsv[:, 1:].std(0).mean()
    return float(1.0 - 0.5 * circ_var - 0.5 * sv_std)


def manifold_dim_top20(
    acts: np.ndarray,
    row_color_idx: np.ndarray,
    color_rgb: np.ndarray,
    top_k_atoms: int = 20,
    top_k_colors: int = 20,
) -> dict[str, Any]:
    """Effective rank of the top-20 color RGB points per atom.

    Effective rank = exp(Shannon entropy of normalized eigenvalues).
    A pure 1D-curve atom should give effective rank near 1; a single-point
    atom near 0 (we clamp at 1.0 via convention since 1 sample has rank 1).
    Scattered atoms give effective rank closer to 3 (full RGB).
    """
    by_color = _aggregate_by_color(acts, row_color_idx, n_colors=color_rgb.shape[0])
    active_per_atom = (acts > 0).mean(0)
    top_atoms = np.argsort(-active_per_atom)[:top_k_atoms]
    per_atom = []
    dims = []
    for k in top_atoms:
        scores = by_color[:, k]
        top_ci = np.argsort(-scores)[:top_k_colors]
        pts = color_rgb[top_ci]
        d = _effective_rank(pts)
        dims.append(d)
        per_atom.append({"atom": int(k), "effective_rank": d})
    return {
        "mean_effective_rank": float(np.mean(dims)) if dims else float("nan"),
        "per_atom": per_atom,
    }


def _effective_rank(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return 1.0
    pts = pts - pts.mean(0, keepdims=True)
    # SVD on points (N, d) -> singular values length min(N, d).
    s = np.linalg.svd(pts, full_matrices=False, compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    if total <= 1e-12:
        return 0.0
    p = s2 / total
    p = p[p > 1e-12]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


# ---------------------------------------------------------------------------
# Causal ablation
# ---------------------------------------------------------------------------


def causal_ablation_delta_r2(
    model: SAEWrapper,
    X: torch.Tensor,
    base_r2: float | None = None,
    atom_subset: list[int] | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    """For each atom (or each in ``atom_subset``), zero it and measure R^2 drop.

    Decodes from the modified activation vector. Reports per-atom drops and
    summary stats.
    """
    if base_r2 is None:
        base_r2 = reconstruction_r2(model, X, batch_size=batch_size)
    # Pre-collect activations once.
    Z = collect_activations(model, X, batch_size=batch_size)  # (N, F)
    F = Z.shape[1]
    if atom_subset is None:
        atom_subset = list(range(F))
    device = X.device
    X_mean = X.mean(0, keepdim=True)
    sst = float(((X - X_mean) ** 2).sum().item())
    drops = {}
    for k in atom_subset:
        Zk = Z.copy()
        Zk[:, k] = 0.0
        sse = 0.0
        for i in range(0, Zk.shape[0], batch_size):
            zb = torch.from_numpy(Zk[i:i + batch_size]).to(device)
            with torch.no_grad():
                recon = model.decode_from_activations(zb)
            sse += float(((X[i:i + batch_size] - recon) ** 2).sum().item())
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        drops[int(k)] = base_r2 - r2
    arr = np.array(list(drops.values()))
    return {
        "base_r2": base_r2,
        "per_atom_delta_r2": drops,
        "mean_delta_r2": float(arr.mean()) if arr.size else 0.0,
        "max_delta_r2": float(arr.max()) if arr.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def linear_probe(
    Z: np.ndarray,
    y: np.ndarray,
    ridge: float = 1.0,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Ridge regression / classification probe Z -> y. Returns held-out R^2
    (regression) or accuracy (classification, when y is integer dtype with
    few classes).
    """
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx = order[:n_test]
    tr_idx = order[n_test:]
    Ztr, Zte = Z[tr_idx], Z[test_idx]
    ytr, yte = y[tr_idx], y[test_idx]
    is_classification = (y.dtype.kind in "iu") and (np.unique(y).size <= 20)
    if is_classification:
        return _logistic_probe(Ztr, ytr, Zte, yte, ridge=ridge)
    if y.ndim == 1:
        y_was_1d = True
        ytr = ytr[:, None]
        yte = yte[:, None]
    else:
        y_was_1d = False
    # Solve (Z^T Z + ridge I) W = Z^T y.
    F = Ztr.shape[1]
    A = Ztr.T @ Ztr + ridge * np.eye(F)
    B = Ztr.T @ ytr
    W = np.linalg.solve(A, B)
    pred = Zte @ W
    res = ((yte - pred) ** 2).sum(0)
    tot = ((yte - yte.mean(0)) ** 2).sum(0).clip(min=1e-12)
    r2 = 1.0 - (res / tot)
    if y_was_1d:
        return {"r2": float(r2[0])}
    return {"r2_per_dim": r2.tolist(), "r2_mean": float(r2.mean())}


def _logistic_probe(Ztr, ytr, Zte, yte, ridge: float) -> dict[str, float]:
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None, :]).astype(np.float64)
    F = Ztr.shape[1]
    A = Ztr.T @ Ztr + ridge * np.eye(F)
    W = np.linalg.solve(A, Ztr.T @ Y)
    pred = Zte @ W
    pred_class = classes[np.argmax(pred, axis=1)]
    return {"accuracy": float((pred_class == yte).mean())}


# ---------------------------------------------------------------------------
# Feature absorption (Bussmann 2024 inspired)
# ---------------------------------------------------------------------------


def feature_absorption(
    acts: np.ndarray,
    labels: np.ndarray,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    """Per-concept absorption rate.

    For each concept c (each column of ``labels``, a binary indicator):
      * Find the atom a* that has the highest precision*recall product
        as a single-atom predictor for c.
      * On rows where c=1, the absorption gap is the fraction of rows where
        a* does NOT fire but at least one other atom fires AND that other
        atom is also predictive of c (precision >= 0.5 on c).

    Returns mean absorption rate over concepts.
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    fires = acts > threshold
    n_concepts = labels.shape[1]
    per_concept = []
    for c in range(n_concepts):
        y = labels[:, c].astype(bool)
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        # Precision and recall per atom for c=1.
        atom_pos = fires & y[:, None]
        atom_neg = fires & (~y[:, None])
        recall = atom_pos.sum(0) / max(int(y.sum()), 1)
        precision = atom_pos.sum(0) / np.maximum(fires.sum(0), 1)
        f_like = recall * precision
        a_star = int(np.argmax(f_like))
        precise_atoms = np.where(precision >= 0.5)[0]
        precise_atoms = [a for a in precise_atoms if a != a_star]
        # Rows where c holds but a* didn't fire:
        rows = np.where(y & ~fires[:, a_star])[0]
        if rows.size == 0:
            absorption = 0.0
        else:
            other_fires = fires[rows][:, precise_atoms].any(axis=1) if precise_atoms else np.zeros(rows.size, dtype=bool)
            absorption = float(other_fires.mean())
        per_concept.append({"concept": int(c), "a_star": a_star,
                            "precision_star": float(precision[a_star]),
                            "recall_star": float(recall[a_star]),
                            "absorption_rate": absorption})
    if not per_concept:
        return {"mean_absorption": float("nan"), "per_concept": []}
    return {
        "mean_absorption": float(np.mean([p["absorption_rate"] for p in per_concept])),
        "per_concept": per_concept,
    }


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


def steering_quality_hue(
    model: SAEWrapper,
    X: torch.Tensor,
    Z: np.ndarray,
    row_hue: np.ndarray,
    k_steer: int = 8,
    push_sigma: float = 1.0,
    batch_size: int = 512,
) -> dict[str, float]:
    """Pick the ``k_steer`` atoms most correlated with hue, push +1 sigma,
    re-decode, and measure correlation between intended hue push and
    observed decoded hue delta.

    "Observed decoded hue" is approximated by projecting the decoded vector
    delta onto the hue-direction in residual-stream space, which is itself
    estimated as a linear regressor from X to row_hue.
    """
    # 1. Hue direction in input space, by ridge regression X -> row_hue.
    Xn = X.detach().cpu().numpy()
    h = row_hue
    # circular: use cos(2*pi*h), sin(2*pi*h).
    H = np.stack([np.cos(2 * np.pi * h), np.sin(2 * np.pi * h)], axis=1)
    F = Xn.shape[1]
    A = Xn.T @ Xn + 1.0 * np.eye(F)
    W = np.linalg.solve(A, Xn.T @ H)  # (F, 2)
    # 2. Atom -> hue correlation in the same {cos, sin} basis.
    Z_cent = Z - Z.mean(0, keepdims=True)
    H_cent = H - H.mean(0, keepdims=True)
    corr = Z_cent.T @ H_cent / max(Z.shape[0] - 1, 1)
    corr_norm = np.linalg.norm(corr, axis=1)
    top_atoms = np.argsort(-corr_norm)[:k_steer]
    # 3. For each row, push the top atoms by +sigma along their hue gradient.
    sigma = Z[:, top_atoms].std(0).clip(min=1e-6)
    intended = corr[top_atoms]  # (k, 2)
    # Decode base, decode pushed, take difference, project onto W's columns.
    device = X.device
    deltas_obs = []
    intended_full = (sigma[:, None] * intended).sum(0)  # net (cos, sin) direction
    # Per-row intended is the same direction; per-row observed delta projects onto W.
    for i in range(0, X.shape[0], batch_size):
        zb = torch.from_numpy(Z[i:i + batch_size].copy()).to(device)
        with torch.no_grad():
            base = model.decode_from_activations(zb)
        zb_pushed = zb.clone()
        zb_pushed[:, top_atoms] = zb_pushed[:, top_atoms] + torch.from_numpy(sigma * push_sigma).to(device)
        with torch.no_grad():
            pushed = model.decode_from_activations(zb_pushed)
        d = (pushed - base).detach().cpu().numpy()
        deltas_obs.append(d @ W)  # (b, 2)
    deltas = np.concatenate(deltas_obs, 0)  # (N, 2)
    # cosine similarity between mean observed delta and intended direction.
    mean_delta = deltas.mean(0)
    denom = (np.linalg.norm(mean_delta) * np.linalg.norm(intended_full) + 1e-12)
    cos = float((mean_delta @ intended_full) / denom)
    return {
        "k_steer": int(k_steer),
        "push_sigma": float(push_sigma),
        "steering_cosine": cos,
        "mean_obs_delta_norm": float(np.linalg.norm(mean_delta)),
        "intended_norm": float(np.linalg.norm(intended_full)),
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class HarnessResult:
    model_name: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model_name, **self.metrics}


@dataclass
class HarnessLabels:
    """Optional concept-level labels for a row-aligned dataset.

    Fields the harness will use if provided:
      row_color_idx : (N,) int, color id per row
      color_hsv     : (n_colors, 3) HSV in [0,1]
      color_rgb     : (n_colors, 3) RGB in [0,1]
      row_hue       : (N,) float, per-row hue label (typically derived).
      row_modifier_count : (N,) int
      row_monoword  : (N,) int (0/1)
      concept_labels: (N, C) bool, used for feature-absorption
    """
    row_color_idx: np.ndarray | None = None
    color_hsv: np.ndarray | None = None
    color_rgb: np.ndarray | None = None
    row_hue: np.ndarray | None = None
    row_modifier_count: np.ndarray | None = None
    row_monoword: np.ndarray | None = None
    concept_labels: np.ndarray | None = None


class Harness:
    """Run the full metric battery on a wrapped SAE."""

    def __init__(
        self,
        model: SAEWrapper,
        X_val: torch.Tensor,
        labels: HarnessLabels | None = None,
        ablation_subset: int | None = 32,
        seed: int = 0,
    ):
        self.model = model
        self.X = X_val
        self.labels = labels or HarnessLabels()
        self.ablation_subset = ablation_subset
        self.seed = seed

    def run(self) -> HarnessResult:
        m: dict[str, Any] = {}
        # 1. R^2
        m["val_r2"] = reconstruction_r2(self.model, self.X)
        # 2-3. activations + sparsity + dead
        Z = collect_activations(self.model, self.X)
        m["sparsity"] = sparsity_stats(Z, self.model.firing_threshold)
        m["dead_atom_fraction"] = dead_atom_fraction(Z, self.model.firing_threshold)
        active_atoms = max(int((Z > self.model.firing_threshold).any(0).sum()), 1)
        # 10. compute-normalized
        m["r2_per_flop"] = m["val_r2"] / self.model.flops_per_token
        m["r2_per_active_atom"] = m["val_r2"] / active_atoms
        # 5-6. HSV coherence + manifold dim (need color labels)
        if self.labels.row_color_idx is not None and self.labels.color_hsv is not None:
            m["hsv_coherence"] = hsv_coherence_top20(
                Z, self.labels.row_color_idx, self.labels.color_hsv
            )
        if self.labels.row_color_idx is not None and self.labels.color_rgb is not None:
            m["manifold_dim"] = manifold_dim_top20(
                Z, self.labels.row_color_idx, self.labels.color_rgb
            )
        # 4. Absorption
        if self.labels.concept_labels is not None:
            m["feature_absorption"] = feature_absorption(Z, self.labels.concept_labels,
                                                        threshold=self.model.firing_threshold)
        # 7. Causal ablation (subset for speed)
        subset = None
        if self.ablation_subset is not None and self.ablation_subset < Z.shape[1]:
            # Take top-N most-active atoms.
            order = np.argsort(-(Z > self.model.firing_threshold).mean(0))
            subset = order[:self.ablation_subset].tolist()
        m["ablation"] = causal_ablation_delta_r2(
            self.model, self.X, base_r2=m["val_r2"], atom_subset=subset
        )
        # 8. Probes
        probes: dict[str, Any] = {}
        if self.labels.row_color_idx is not None and self.labels.color_hsv is not None:
            row_hsv = self.labels.color_hsv[self.labels.row_color_idx]
            probes["hsv"] = linear_probe(Z, row_hsv, seed=self.seed)
        if self.labels.row_modifier_count is not None:
            probes["modifier_count"] = linear_probe(
                Z, self.labels.row_modifier_count.astype(np.float64), seed=self.seed
            )
        if self.labels.row_monoword is not None:
            probes["monoword"] = linear_probe(
                Z, self.labels.row_monoword.astype(np.int64), seed=self.seed
            )
        m["probes"] = probes
        # 9. Steering
        if self.labels.row_hue is not None:
            m["steering"] = steering_quality_hue(self.model, self.X, Z, self.labels.row_hue)
        m["n_active_atoms"] = active_atoms
        return HarnessResult(model_name=self.model.name, metrics=m)
