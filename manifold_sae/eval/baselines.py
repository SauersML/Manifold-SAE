"""Trivial baselines for context. Each returns an SAEWrapper.

  - PCA-N: encode = X @ V[:N]; decode = z @ V[:N]^T. Activations are dense.
  - RandomProjection-N: same shape with a fixed random orthonormal V.
  - Identity: encode = pad/truncate to F; decode reverses.
"""
from __future__ import annotations

import numpy as np
import torch

from .harness import SAEWrapper


class _LinearProjBaseline(SAEWrapper):
    def __init__(self, V: torch.Tensor, mean: torch.Tensor, name: str, threshold: float = 1e-3):
        # V: (D, F), columns are encoding directions.
        self.V = V
        self.mean = mean
        self.name = name
        self.input_dim = V.shape[0]
        self.n_features = V.shape[1]
        self.firing_threshold = threshold

    def encode(self, x):
        return (x - self.mean) @ self.V

    def decode_from_activations(self, z):
        return z @ self.V.T + self.mean

    def reconstruct(self, x):
        return self.decode_from_activations(self.encode(x))


def pca_baseline(X_train: np.ndarray, n_features: int = 512, name: str = "PCA-512", device: str = "cpu") -> SAEWrapper:
    mean = X_train.mean(0)
    Xc = X_train - mean
    # SVD on centered data; columns of Vt^T are PCs.
    # Use randomized SVD-equivalent via numpy linalg on subsample if huge.
    if Xc.shape[0] > 8000:
        rng = np.random.default_rng(0)
        idx = rng.choice(Xc.shape[0], 8000, replace=False)
        Xs = Xc[idx]
    else:
        Xs = Xc
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt[:n_features].T  # (D, n_features)
    return _LinearProjBaseline(
        torch.from_numpy(V.astype(np.float32)).to(device),
        torch.from_numpy(mean.astype(np.float32)).to(device),
        name=name,
    )


def random_projection_baseline(d_in: int, n_features: int = 512, seed: int = 0,
                               name: str = "RandomProj-512", device: str = "cpu") -> SAEWrapper:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_in, n_features)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    V = Q[:, :n_features] if n_features <= d_in else A / np.sqrt(d_in)
    return _LinearProjBaseline(
        torch.from_numpy(V).to(device),
        torch.zeros(d_in, dtype=torch.float32, device=device),
        name=name,
    )


def identity_baseline(d_in: int, name: str = "Identity", device: str = "cpu") -> SAEWrapper:
    V = torch.eye(d_in, dtype=torch.float32, device=device)
    return _LinearProjBaseline(V, torch.zeros(d_in, device=device), name=name)
