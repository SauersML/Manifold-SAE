"""Gauge-fixed concept charts for harvested language-model activations.

This module implements the validated Manifold-SAE HSV-style gauge-fix
recipe used in ``auto_exp_38``, ``auto_exp_53``, and ``auto_exp_54``:
center activations, build a PCA working basis, regress concept labels from
PC scores, rotate the chart by the regression SVD, and select the latent
dimension with BIC unless the caller pins ``d``.

The implementation is intentionally NumPy-first. ``gamfit`` owns the
general GAM/REML machinery in this repository, but this flat affine
gauge-fix is faster and more transparent as direct linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class GaugeFit:
    """Fitted gauge chart and anchor registry.

    Attributes:
        targets: Ordered target names used to fit the supervised chart.
        d: Selected latent dimension.
        mu: Activation mean with shape ``(width,)``.
        pca_basis: PCA basis with shape ``(width, k)``.
        axes: Gauge axes in activation space with shape ``(width, d)``.
        free_axes: Orthogonal complement inside the PCA basis.
        regression: OLS map from PC scores to standardized targets.
        singular_values: Singular values of the target regression map.
        bic_by_d: BIC score for each candidate dimension.
        r2: In-sample target R-squared values.
        target_mean: Target-column mean.
        target_std: Target-column standard deviation.
        sigma: Per-gauge-axis activation standard deviation.
        anchors: Named concept anchors in activation space.
    """

    targets: tuple[str, ...]
    d: int
    mu: FloatArray
    pca_basis: FloatArray
    axes: FloatArray
    free_axes: FloatArray
    regression: FloatArray
    singular_values: FloatArray
    bic_by_d: dict[int, float]
    r2: dict[str, float]
    target_mean: FloatArray
    target_std: FloatArray
    sigma: FloatArray
    anchors: dict[str, FloatArray] = field(default_factory=dict)

    def transform(self, activations: NDArray[np.floating]) -> FloatArray:
        """Project activations into the fitted gauge coordinates."""
        x = np.asarray(activations, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return (x - self.mu) @ self.axes

    def inverse_offset(self, latent_offset: Sequence[float]) -> FloatArray:
        """Map a latent-space offset back into activation space."""
        dz = np.asarray(latent_offset, dtype=np.float64)
        if dz.shape != (self.d,):
            raise ValueError(f"latent offset shape {dz.shape} != ({self.d},)")
        return self.axes @ dz

    def register_anchor(self, name: str, vector: NDArray[np.floating]) -> None:
        """Register a named concept anchor in activation space."""
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape != self.mu.shape:
            raise ValueError(f"anchor dim {v.shape[0]} != activation width {self.mu.shape[0]}")
        self.anchors[name] = v

    def anchor(self, name: str) -> FloatArray:
        """Return a registered anchor."""
        if name not in self.anchors:
            raise KeyError(f"unknown anchor {name!r}; known anchors: {sorted(self.anchors)}")
        return self.anchors[name]

    def save(self, path: str | Path) -> None:
        """Persist the fit as a compressed NumPy archive."""
        p = Path(path)
        np.savez_compressed(
            p,
            targets=np.asarray(self.targets, dtype=object),
            d=np.asarray(self.d),
            mu=self.mu,
            pca_basis=self.pca_basis,
            axes=self.axes,
            free_axes=self.free_axes,
            regression=self.regression,
            singular_values=self.singular_values,
            bic_keys=np.asarray(list(self.bic_by_d), dtype=np.int64),
            bic_vals=np.asarray(list(self.bic_by_d.values()), dtype=np.float64),
            r2_keys=np.asarray(list(self.r2), dtype=object),
            r2_vals=np.asarray(list(self.r2.values()), dtype=np.float64),
            target_mean=self.target_mean,
            target_std=self.target_std,
            sigma=self.sigma,
            anchor_keys=np.asarray(list(self.anchors), dtype=object),
            anchor_vals=np.stack(list(self.anchors.values()), axis=0)
            if self.anchors
            else np.zeros((0, self.mu.shape[0]), dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaugeFit":
        """Load a fit saved by :meth:`save`."""
        data = np.load(Path(path), allow_pickle=True)
        anchors = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in zip(data["anchor_keys"].tolist(), data["anchor_vals"])
        }
        return cls(
            targets=tuple(str(x) for x in data["targets"].tolist()),
            d=int(data["d"]),
            mu=np.asarray(data["mu"], dtype=np.float64),
            pca_basis=np.asarray(data["pca_basis"], dtype=np.float64),
            axes=np.asarray(data["axes"], dtype=np.float64),
            free_axes=np.asarray(data["free_axes"], dtype=np.float64),
            regression=np.asarray(data["regression"], dtype=np.float64),
            singular_values=np.asarray(data["singular_values"], dtype=np.float64),
            bic_by_d={int(k): float(v) for k, v in zip(data["bic_keys"], data["bic_vals"])},
            r2={str(k): float(v) for k, v in zip(data["r2_keys"].tolist(), data["r2_vals"])},
            target_mean=np.asarray(data["target_mean"], dtype=np.float64),
            target_std=np.asarray(data["target_std"], dtype=np.float64),
            sigma=np.asarray(data["sigma"], dtype=np.float64),
            anchors=anchors,
        )


def fit_gauge(
    activations: NDArray[np.floating],
    labels: Mapping[str, Sequence[float] | NDArray[np.floating]],
    *,
    targets: Sequence[str] | None = None,
    d: int | None = None,
    k: int | None = None,
    ridge: float = 1e-4,
    anchor_rows: Mapping[str, Sequence[int]] | None = None,
) -> GaugeFit:
    """Fit a gauge-fixed concept chart.

    Args:
        activations: Matrix ``(n_prompts, width)`` of harvested activations.
        labels: Numeric label columns keyed by concept name.
        targets: Label keys to supervise. Defaults to all label keys.
        d: Optional fixed chart dimension. ``None`` selects by BIC.
        k: PCA working dimension. Defaults to ``min(n - 1, width, 128)``.
        ridge: Relative ridge strength for stable OLS.
        anchor_rows: Optional named row-index groups used for steering anchors.

    Returns:
        A fitted :class:`GaugeFit`.
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("activations must be a 2D matrix")
    n, width = x.shape
    if n < 3:
        raise ValueError("at least three activation rows are required")
    target_names = tuple(targets or labels.keys())
    y_raw, y_names = _target_matrix(labels, target_names, n)
    y_mean = y_raw.mean(axis=0)
    y_std = y_raw.std(axis=0) + 1e-12
    y = (y_raw - y_mean) / y_std

    mu = x.mean(axis=0)
    xc = x - mu
    pca_basis = _pca_basis(xc, k or min(n - 1, width, 128))
    z = xc @ pca_basis
    reg = _ridge_solve(z, y, ridge)
    u, singular_values, _vt = np.linalg.svd(reg, full_matrices=False)

    max_d = max(1, min(u.shape[1], pca_basis.shape[1], y.shape[1]))
    selected_d, bic_by_d = _select_d_bic(z, y, u, max_d) if d is None else (int(d), {})
    selected_d = max(1, min(selected_d, max_d))

    axes_raw = pca_basis @ u[:, :selected_d]
    axes, _ = np.linalg.qr(axes_raw)
    axes = axes[:, :selected_d]
    if pca_basis.shape[1] > selected_d and u.shape[1] > selected_d:
        free_raw = pca_basis @ u[:, selected_d:]
        free_axes, _ = np.linalg.qr(free_raw)
    else:
        free_axes = np.zeros((width, 0), dtype=np.float64)

    latent = xc @ axes
    sigma = latent.std(axis=0) + 1e-12
    y_hat = z @ reg
    ss_res = ((y - y_hat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = {name: float(1.0 - ssr / sst) for name, ssr, sst in zip(y_names, ss_res, ss_tot)}
    anchors = {
        name: x[np.asarray(rows, dtype=np.int64)].mean(axis=0)
        for name, rows in (anchor_rows or {}).items()
    }
    return GaugeFit(
        targets=target_names,
        d=selected_d,
        mu=mu,
        pca_basis=pca_basis,
        axes=axes,
        free_axes=free_axes,
        regression=reg,
        singular_values=singular_values,
        bic_by_d=bic_by_d,
        r2=r2,
        target_mean=y_mean,
        target_std=y_std,
        sigma=sigma,
        anchors=anchors,
    )


def _target_matrix(
    labels: Mapping[str, Sequence[float] | NDArray[np.floating]],
    targets: Sequence[str],
    n: int,
) -> tuple[FloatArray, tuple[str, ...]]:
    cols: list[FloatArray] = []
    names: list[str] = []
    for key in targets:
        if key not in labels:
            raise KeyError(f"missing target label {key!r}")
        arr = np.asarray(labels[key])
        if arr.shape[0] != n:
            raise ValueError(f"label {key!r} has length {arr.shape[0]} != {n}")
        if arr.ndim == 1:
            cols.append(arr.astype(np.float64))
            names.append(key)
        elif arr.ndim == 2:
            for j in range(arr.shape[1]):
                cols.append(arr[:, j].astype(np.float64))
                names.append(f"{key}_{j}")
        else:
            raise ValueError(f"label {key!r} must be 1D or 2D")
    return np.column_stack(cols), tuple(names)


def _pca_basis(xc: FloatArray, k: int) -> FloatArray:
    k = max(1, min(int(k), min(xc.shape)))
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:k].T.copy()


def _ridge_solve(z: FloatArray, y: FloatArray, ridge: float) -> FloatArray:
    scale = np.trace(z.T @ z) / max(z.shape[1], 1)
    a = z.T @ z + float(ridge) * scale * np.eye(z.shape[1])
    return np.linalg.solve(a, z.T @ y)


def _select_d_bic(z: FloatArray, y: FloatArray, u: FloatArray, max_d: int) -> tuple[int, dict[int, float]]:
    bic: dict[int, float] = {}
    n, t = y.shape
    for cand in range(1, max_d + 1):
        q = u[:, :cand]
        zq = z @ q
        beta = np.linalg.pinv(zq) @ y
        resid = y - zq @ beta
        rss = float((resid**2).sum())
        params = cand * t + cand
        bic[cand] = n * t * np.log(rss / max(n * t, 1) + 1e-12) + params * np.log(n)
    return min(bic, key=bic.__getitem__), bic
