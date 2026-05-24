"""HSV-style gauge-fix recipe.

Validated in Manifold-SAE auto_exp_38 (free axes align with semantic
non-target features once a gauge-fix companion is supplied) and
auto_exp_53 (BIC-/CV-optimal latent dimension ``d == rank(targets)``),
and confirmed transferable in auto_exp_54.

The recipe:

  1. Mean-center activations  ``Xc = X - mu``.
  2. PCA to a working basis of size ``K`` (default ``min(N-1, p, 128)``).
  3. Fit a linear map ``W : R^K -> R^T`` from PC scores to the stacked
     target table by ordinary least squares  ``W = (Z^T Z)^-1 Z^T Y``.
  4. Take the SVD of ``W = U S V^T``; the **gauge-fixed subspace** is
     spanned by the first ``d = rank(Y)`` left-singular axes (in PC
     space).  These rows are by construction the directions in
     activation space whose projection onto the target table has the
     largest signal-to-noise ratio, and they are mutually orthogonal --
     this is the "gauge fix".
  5. The latent coordinates of each prompt are the projection of its
     centered activation onto those ``d`` axes, in the **original
     activation basis**.

The result is a flat-affine chart of the manifold whose first ``d``
axes are *interpretable* (each one regresses a known target) and whose
*remaining* axes (returned by :meth:`GaugeFix.free_axes`, optional)
form an orthogonal complement in PC-space — auto_exp_38 showed these
free axes pick up semantic structure (e.g. monoword, modifier-count,
template-σ) automatically when the supervised axes provide a stable
reference frame.

This module deliberately does not depend on ``gamfit`` so it installs
cleanly via pip; we re-implement the small slice of the recipe needed
end-to-end in NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass
class GaugeFix:
    """Fit + transform handle for the HSV-style gauge-fix recipe.

    Parameters
    ----------
    targets
        Ordered list of label-table keys to use as the supervised
        gauge-fix targets.  At fit time, ``labels[k]`` must be a 1-D
        array of length ``N`` for every ``k in targets``.
    d
        Latent dimension.  ``None`` (default) follows auto_exp_53 and
        sets ``d = rank(target_matrix)``.  Override only if you know
        you want a richer chart (and read the locality warnings).
    K
        PCA basis size used internally.  ``None`` -> ``min(N-1, p, 128)``
        which is the production cogito setting.
    standardize_targets
        Z-score each target column before regression.  Recommended
        (auto_exp_53 BIC scoring assumes this).
    """

    targets: Sequence[str]
    d: int | None = None
    K: int | None = None
    standardize_targets: bool = True

    # Fitted state ----------------------------------------------------------
    mu_: np.ndarray | None = field(default=None, init=False, repr=False)
    pca_basis_: np.ndarray | None = field(default=None, init=False, repr=False)  # (p, K)
    W_: np.ndarray | None = field(default=None, init=False, repr=False)         # (K, T)
    axes_: np.ndarray | None = field(default=None, init=False, repr=False)      # (p, d)
    free_axes_: np.ndarray | None = field(default=None, init=False, repr=False) # (p, K-d)
    target_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    target_std_:  np.ndarray | None = field(default=None, init=False, repr=False)
    r2_: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    sigma_: np.ndarray | None = field(default=None, init=False, repr=False)
    anchors_: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    n_: int = field(default=0, init=False, repr=False)
    p_: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        labels: Mapping[str, np.ndarray],
        *,
        anchor_labels: Mapping[str, Sequence[int | str]] | None = None,
    ) -> "GaugeFix":
        """Fit the gauge-fix chart.

        Parameters
        ----------
        X : (N, p) activation matrix.
        labels : dict mapping every name in ``self.targets`` -> length-N array.
                 Extra keys are ignored; categorical (str) labels are 1-hot
                 encoded automatically per key.
        anchor_labels : optional mapping ``{concept_name: row_indices_or_str_label}``
                 used by :class:`ManifoldSteerer`.  E.g.
                 ``{"red": [3, 17, 42], "blue": [5, 9, 88]}``.  Each anchor
                 is stored as the *mean* activation of its constituent rows.

        Returns
        -------
        self
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        N, p = X.shape
        self.n_, self.p_ = N, p

        # 1. mean center
        mu = X.mean(0)
        Xc = X - mu

        # 2. PCA basis (truncated SVD on the *covariance* shortcut)
        K = self.K or min(N - 1, p, 128)
        K = max(1, min(K, N - 1, p))
        # Use SVD on Xc directly: Xc = U S Vt, columns of V are PC axes.
        # For large p we use the gram-trick if N < p.
        if N <= p:
            G = Xc @ Xc.T / max(N - 1, 1)             # (N, N)
            evals, evecs = np.linalg.eigh(G)
            order = np.argsort(evals)[::-1][:K]
            evals = np.maximum(evals[order], 0.0)
            U = evecs[:, order]                        # (N, K)
            # PC axes in p-space:  V = Xc^T U / sqrt(eval * (N-1))
            s = np.sqrt(evals * max(N - 1, 1)) + 1e-12
            V = (Xc.T @ U) / s                         # (p, K)
        else:
            # p < N: direct SVD on Xc^T Xc.
            C = Xc.T @ Xc / max(N - 1, 1)
            evals, evecs = np.linalg.eigh(C)
            order = np.argsort(evals)[::-1][:K]
            V = evecs[:, order]
        self.pca_basis_ = V.astype(np.float32)         # (p, K)
        Z = Xc @ V                                     # (N, K) PC scores

        # 3. Build target matrix Y (handle categorical via 1-hot)
        Y_cols: list[np.ndarray] = []
        Y_names: list[str] = []
        for k in self.targets:
            if k not in labels:
                raise KeyError(f"target {k!r} missing from labels")
            col = np.asarray(labels[k])
            if col.dtype.kind in ("U", "S", "O"):
                uniq = sorted(set(col.tolist()))
                if len(uniq) < 2:
                    raise ValueError(f"categorical target {k!r} has <2 levels")
                # k-1 dummies (drop reference) to keep rank
                for lvl in uniq[1:]:
                    Y_cols.append((col == lvl).astype(np.float32))
                    Y_names.append(f"{k}={lvl}")
            else:
                Y_cols.append(col.astype(np.float32))
                Y_names.append(k)
        Y = np.stack(Y_cols, axis=1)                    # (N, T)

        if self.standardize_targets:
            self.target_mean_ = Y.mean(0)
            self.target_std_ = Y.std(0) + 1e-8
            Yn = (Y - self.target_mean_) / self.target_std_
        else:
            self.target_mean_ = np.zeros(Y.shape[1], np.float32)
            self.target_std_ = np.ones(Y.shape[1], np.float32)
            Yn = Y

        # 4. OLS in PC space; ridge for numerical stability.
        ridge = 1e-4 * np.trace(Z.T @ Z) / max(K, 1)
        A = Z.T @ Z + ridge * np.eye(K, dtype=Z.dtype)
        W = np.linalg.solve(A, Z.T @ Yn).astype(np.float32)  # (K, T)
        self.W_ = W

        # 5. SVD -> gauge-fixed axes in PC space
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # rank(Y) for default d
        rank = int(np.sum(S > S.max() * 1e-6)) if S.size else 1
        d = self.d if self.d is not None else max(rank, 1)
        d = min(d, K, U.shape[1])
        self.d = d

        U_d = U[:, :d]                                   # (K, d)
        axes_p = V @ U_d                                 # (p, d) in original activation space
        # Re-orthonormalise in p-space to be safe.
        Q, _ = np.linalg.qr(axes_p)
        self.axes_ = Q.astype(np.float32)
        self.mu_ = mu.astype(np.float32)

        # Free axes: orthogonal complement of U_d inside the PC subspace
        if K > d:
            U_free = U[:, d:]                            # (K, K-d)
            free_p = V @ U_free
            Qf, _ = np.linalg.qr(free_p)
            self.free_axes_ = Qf.astype(np.float32)
        else:
            self.free_axes_ = np.zeros((p, 0), dtype=np.float32)

        # Per-axis std on training data (for alpha=1 -> 1-sigma convention)
        latent = Xc @ self.axes_                         # (N, d)
        self.sigma_ = latent.std(0).astype(np.float32) + 1e-8

        # Per-target R^2 of the *gauge-fixed* projection (predictiveness check)
        Y_hat = Z @ W                                    # (N, T)  in standardised space
        ss_res = ((Yn - Y_hat) ** 2).sum(0)
        ss_tot = ((Yn - Yn.mean(0)) ** 2).sum(0) + 1e-12
        for name, ssr, sst in zip(Y_names, ss_res, ss_tot):
            self.r2_[name] = float(1.0 - ssr / sst)

        # Anchors (mean activation per concept)
        if anchor_labels:
            for cname, sel in anchor_labels.items():
                sel_arr = np.asarray(sel)
                if sel_arr.dtype.kind in ("U", "S", "O"):
                    # Need a string-valued column to match against; use the
                    # first categorical target.
                    raise ValueError(
                        "string anchor selectors not yet supported; pass row indices"
                    )
                rows = sel_arr.astype(int)
                self.anchors_[cname] = X[rows].mean(0).astype(np.float32)

        # Locality / variance warnings (auto_exp_49)
        Xc_var = float((Xc ** 2).sum() / N)
        ax_var = float((latent ** 2).sum() / N)
        ratio = ax_var / max(Xc_var, 1e-12)
        if ratio < 0.05:
            warnings.warn(
                f"gauge-fixed subspace captures only {ratio*100:.1f}% of "
                f"activation variance; expect noisier steering per auto_exp_49.",
                stacklevel=2,
            )
        for nm, r2 in self.r2_.items():
            if r2 < 0.3:
                warnings.warn(
                    f"target {nm!r} has weak in-sample R^2={r2:.2f}; "
                    f"the gauge-fix may be ill-posed for this label.",
                    stacklevel=2,
                )
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new activations onto the d gauge-fixed axes."""
        if self.axes_ is None or self.mu_ is None:
            raise RuntimeError("GaugeFix is not fitted")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mu_) @ self.axes_

    def fit_transform(self, X, labels, **kw) -> np.ndarray:
        return self.fit(X, labels, **kw).transform(X)

    # --------------------------------------------------------------- helpers
    def axes(self) -> np.ndarray:
        """Return the (p, d) gauge-fixed axes in activation space."""
        if self.axes_ is None:
            raise RuntimeError("not fitted")
        return self.axes_

    def free_axes(self) -> np.ndarray:
        """Return the (p, K-d) orthogonal-complement axes; per auto_exp_38
        these tend to align with *unsupervised* semantic features."""
        if self.free_axes_ is None:
            raise RuntimeError("not fitted")
        return self.free_axes_

    def sigma(self) -> np.ndarray:
        """Per-axis training-data std (the alpha=1 unit)."""
        if self.sigma_ is None:
            raise RuntimeError("not fitted")
        return self.sigma_

    def r2(self) -> dict[str, float]:
        """In-sample R^2 of each target in standardised space."""
        return dict(self.r2_)

    def anchor(self, name: str) -> np.ndarray:
        if name not in self.anchors_:
            raise KeyError(f"unknown anchor {name!r}; known={list(self.anchors_)}")
        return self.anchors_[name]

    def register_anchor(self, name: str, vector: np.ndarray) -> None:
        """Register a concept anchor (mean activation) post-hoc."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.p_:
            raise ValueError(f"anchor dim {v.shape[0]} != p={self.p_}")
        self.anchors_[name] = v
