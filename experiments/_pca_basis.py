"""Canonical PCA basis for the cogito color manifold.

One source of truth for "PCA on standardized per-color centroids" used
across color_manifold_gam.py, plot_color_geometry.py, plot_hypersheet_sweep.py,
build_color_manifold_probe.py, build_manifold_tangent_probe.py,
plot_per_template_alignment.py, plot_latent_discovery_comparison.py, etc.

Convention (the one used by color_manifold_gam.py main path):
  1. Load the (N_prompts, D) layer-40 residual cache.
  2. Per color, average ONLY the TOP_TEMPLATES=[8, 13, 16, 17, 18, 5]
     rows (the color-focused templates that score highest on per-template
     R² — averaging all 28 templates dilutes the signal).
  3. Per-feature standardize: (centroid - mu) / sigma  (sigma clipped at 1e-6).
  4. sklearn.decomposition.PCA(n_components=K, svd_solver="full").fit(...)
     on the standardized matrix. (sklearn centers internally, but we've
     already centered via the per-feature mu — double-centering is a no-op
     up to fp roundoff.)
  5. Vt = pca.components_  (K, D);  evr = pca.explained_variance_ratio_.

Cache at runs/COLOR_COGITO_L40/pca_basis_K{K}.npz so downstream scripts
(probe builders, plots) all use IDENTICAL principal directions.

Use load_pc_basis(K) and project(centroids, basis) — do not re-roll SVD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


N_TEMPLATES = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]      # color-focused templates
DEFAULT_CACHE_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40")
DEFAULT_HARVEST = DEFAULT_CACHE_DIR / "X_L40.npy"


def _load_harvest(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)
        return np.asarray(d["X"])
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"need .npy or .npz harvest; got {path.suffix}")


def _per_color_centroids(
    harvest: np.ndarray,
    n_templates: int = N_TEMPLATES,
    template_subset: list[int] | None = TOP_TEMPLATES,
) -> np.ndarray:
    """Mean per color across a subset of templates. Truncates to whole colors."""
    n_colors = harvest.shape[0] // n_templates
    harvest = harvest[: n_colors * n_templates]
    if template_subset is None:
        template_subset = list(range(n_templates))
    out = np.zeros((n_colors, harvest.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        base = ci * n_templates
        rows = [base + ti for ti in template_subset]
        out[ci] = harvest[rows].mean(axis=0)
    return out


def _standardize(X: np.ndarray, standardize: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature center (and optionally scale). Returns (Xn, mu, sigma).
    If standardize=False, sigma is all-ones (center-only mode)."""
    mu = X.mean(axis=0, keepdims=True)
    if standardize:
        sigma = X.std(axis=0, keepdims=True).clip(min=1e-6)
    else:
        sigma = np.ones_like(mu)
    Xn = (X - mu) / sigma
    return Xn, mu.squeeze(0), sigma.squeeze(0)


def load_pc_basis(
    K: int = 64,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    harvest_path: Path | None = None,
    template_subset: list[int] | None = TOP_TEMPLATES,
    standardize: bool = True,
    force_recompute: bool = False,
) -> dict:
    """Canonical per-feature-standardized PCA basis of cogito's color centroids.

    Returns dict with keys:
      mu          : (D,) per-feature mean of the centroid matrix
      sigma       : (D,) per-feature std (or ones if standardize=False)
      Vt          : (K, D) principal directions (rows are PCs)
      evr         : (K,) explained variance ratios
      n_components: int K
      cached_path : Path to the .npz cache file
      standardized: bool
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if standardize else "_centeronly"
    cached_path = cache_dir / f"pca_basis_K{K}{tag}.npz"

    if cached_path.exists() and not force_recompute:
        d = np.load(cached_path, allow_pickle=False)
        return {
            "mu": d["mu"], "sigma": d["sigma"], "Vt": d["Vt"],
            "evr": d["evr"], "n_components": int(d["n_components"]),
            "cached_path": cached_path,
            "standardized": bool(d["standardized"]) if "standardized" in d.files else standardize,
        }

    harvest_path = Path(harvest_path) if harvest_path else DEFAULT_HARVEST
    harvest = _load_harvest(harvest_path)
    centroids = _per_color_centroids(harvest, template_subset=template_subset)
    Xn, mu, sigma = _standardize(centroids, standardize=standardize)

    K_use = min(K, Xn.shape[0], Xn.shape[1])
    pca = PCA(n_components=K_use, svd_solver="full")
    pca.fit(Xn)
    Vt = pca.components_.astype(np.float64)
    evr = pca.explained_variance_ratio_.astype(np.float64)

    np.savez(
        cached_path,
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64),
        Vt=Vt,
        evr=evr,
        n_components=np.array(K_use, dtype=np.int64),
        standardized=np.array(bool(standardize)),
    )
    return {
        "mu": mu, "sigma": sigma, "Vt": Vt, "evr": evr,
        "n_components": K_use, "cached_path": cached_path,
        "standardized": standardize,
    }


def project(centroids: np.ndarray, basis: dict) -> np.ndarray:
    """Project (N, D) centroids into the K-D PC latent of the canonical basis.

    Returns (N, K) array. Applies the SAME (mu, sigma, Vt) the basis was
    fit with — use this whenever you want results comparable to a saved
    basis (e.g. for projecting a held-out fold, or for the probe builders).
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    Xn = (centroids - basis["mu"]) / basis["sigma"]
    return Xn @ basis["Vt"].T


def top_pcs(
    X: np.ndarray, d: int, center: bool = True, standardize: bool = False,
) -> np.ndarray:
    """Generic helper: top-d PCs of an arbitrary (N, D) matrix via sklearn.

    Drop-in replacement for ``Xc @ Vt.T[:, :d]`` after centering. Use this
    instead of np.linalg.svd in one-off PCA calls where the input is not
    the canonical centroid matrix (e.g. per-template Z, alternating-fit
    PCA init, latent-discovery comparison fold-splits).
    """
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        sigma = X.std(axis=0, keepdims=True).clip(min=1e-6)
        X = X / sigma
    d_use = min(d, X.shape[0], X.shape[1])
    pca = PCA(n_components=d_use, svd_solver="full")
    if not center:
        # sklearn always centers; emulate no-center by re-adding the mean
        # shift to the components' span. For our callers center=True always.
        raise NotImplementedError("center=False not supported; sklearn PCA always centers")
    return pca.fit_transform(X)


def fit_top_pcs(
    X: np.ndarray, d: int, standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, Vt) where T = top-d PC scores of X and Vt is (d, D)."""
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        sigma = X.std(axis=0, keepdims=True).clip(min=1e-6)
        X = X / sigma
    d_use = min(d, X.shape[0], X.shape[1])
    pca = PCA(n_components=d_use, svd_solver="full")
    T = pca.fit_transform(X)
    return T, pca.components_.astype(np.float64)
