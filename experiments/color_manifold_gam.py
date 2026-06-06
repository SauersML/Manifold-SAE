"""Color manifold as a single GAM object.

A single 3D radial Duchon smooth  f : (R, G, B) → ℝ^D_residual  fit by
REML to Qwen3.6-27B's residuals on the full xkcd 954-color list. The
fitted f IS the LM's color manifold as a continuous function over the
unit RGB cube — interpolate, differentiate, slice, plot anything.

Pipeline
--------
1. Harvest last-token residuals for 954 colors × 28 templates.
2. Average per color → 954 × D.
3. Project to top-K PCs (default K=64) → Z ∈ ℝ^{954×K}.
4. For each layer, fit the GAM zoo (all on the same Z, same 5-fold CV
   by color):

   LINEAR baselines
     L_lin_rgb           : Z[c, :] = W [R,G,B,1]                — linear in RGB
     L_lin_hsv           : Z[c, :] = W [cos2πh, sin2πh, s, v, 1] — linear in HSV (period-aware)

   1D Duchon smooths (multi-smooth additive single GAM)
     L_add_rgb           : f_R(R) + f_G(G) + f_B(B)
     L_add_hsv           : g_hue(cos,sin) + g_s(s) + g_v(v)

   Joint multi-dim Duchon  (one radial kernel over multi-D coords)
     L_joint_rgb         : f(R, G, B)                            ← THE headline object
     L_joint_hsv         : f(cos2πh, sin2πh, s, v)               ← 4D periodic-hue analog
     L_joint_rgb_with_hue: f(R,G,B) + g(cos2πh, sin2πh)
                                                                  ← multi-smooth single GAM
                                                                    asks: does hue add info
                                                                    beyond RGB?

5. Held-out R² (5-fold CV grouped by color, 191 unseen per fold).
6. For the winning spec at each layer, save:
   - fitted coefficients
   - axis-aligned slices of f
   - the hue loop (sweep h at fixed sat=val=1)
   - variance decomposition (smooth f vs residual)
   - anisotropy log-scales (planned for v2)

Plots: bar chart of held-out R² across specs, slice curves on the
held-out scatter, the hue loop in top-3 PCs, per-PC R² heatmap.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


def _pca_Vt(Zc: np.ndarray) -> np.ndarray:
    """Return Vt = (min(N,D), D) for ALREADY-CENTERED Zc, via sklearn PCA.

    Drop-in for ``Vt = _pca_Vt(Zc); Vt``.
    Uses _pca_basis.fit_top_pcs (sklearn.decomposition.PCA, svd_solver=full)
    so all PCA in this repo uses one code path.
    """
    from _pca_basis import fit_top_pcs
    k = min(Zc.shape[0], Zc.shape[1])
    _, Vt = fit_top_pcs(Zc, d=k, standardize=False)
    return Vt


# =============================================================================
# Templates — 28 diverse, beautiful, varied syntactic roles
# =============================================================================
TEMPLATES = [
    "She slipped into a {x} silk dress and floated down the staircase.",
    "His {x} velvet jacket caught every eye in the room.",
    "A long, {x} scarf trailed behind her in the wind.",
    "The dawn sky deepened from grey to {x} before the storm broke.",
    "Across the meadow stretched a sea of {x} wildflowers.",
    "From the cliff we watched the ocean turn a strange {x}.",
    "The painter mixed his pigments until the canvas glowed a perfect {x}.",
    "She dipped her brush in the {x} pool of paint on the palette.",
    "It was the kind of {x} that you only see in renaissance frescoes.",
    "The cathedral's stained-glass rose window burned a luminous {x} at sunset.",
    "He polished the {x} car until the chrome shone like a mirror.",
    "A single {x} candle lit the small, dusty chapel.",
    "The hummingbird's throat flashed an iridescent {x} as it darted past.",
    "Her tabby cat had eyes the unmistakable {x} of an autumn leaf.",
    "A great {x} stallion thundered across the open plain.",
    "The chef plated a glistening, almost-{x} reduction beside the duck.",
    "She bit into the macaron, finding a soft {x} filling within.",
    "Her hair fell across her shoulders in waves of soft {x}.",
    "His skin turned a sickly {x} after three days at sea.",
    "She had freckles and {x} eyes that seemed to change with the weather.",
    "The jeweler held up a flawless {x} stone, catching the lamplight.",
    "Centuries of oxidation had stained the bronze a deep {x}.",
    "An eerie {x} fog rolled in from the harbor at midnight.",
    "Her bedroom walls were a calm, washed-out {x}, like an old photograph.",
    "I bought a {x} fountain pen at the antique market.",
    "The neon sign above the diner flickered {x} against the night.",
    "Grief, in her writing, was always a kind of {x}.",
    "He saw the world through {x} glasses and refused to take them off.",
]


XKCD_URL = "https://xkcd.com/color/rgb.txt"


def load_xkcd_colors() -> list[tuple[str, int, int, int]]:
    """Load the xkcd color survey. Cache locally after first fetch."""
    cache = Path(__file__).parent / "xkcd_colors.txt"
    if cache.exists():
        text = cache.read_text()
    else:
        print(f"[xkcd] fetching {XKCD_URL}", flush=True)
        with urllib.request.urlopen(XKCD_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        try:
            cache.write_text(text)
        except OSError:
            pass
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("License") or s.startswith("Copyright"):
            continue
        m = re.match(r"^(.+?)\s+#?([0-9a-fA-F]{6})$", s)
        if not m:
            continue
        name, hex_ = m.group(1).strip(), m.group(2)
        r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
        out.append((name, r, g, b))
    return out


def rgb_to_hsv_arr(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float64)
    for i in range(rgb.shape[0]):
        out[i] = colorsys.rgb_to_hsv(rgb[i, 0] / 255.0, rgb[i, 1] / 255.0, rgb[i, 2] / 255.0)
    return out


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen3.6-27B")
    # Layer indices to probe. Set via MSAE_LAYERS="i,j,k". The defaults are
    # safe for 28+ layer models; the script validates against
    # `model.config.num_hidden_layers` at startup and complains if any index
    # is out of range, so the env-var path can probe deeper for a 48-layer
    # model without script changes.
    layers: tuple[int, ...] = field(default_factory=lambda: tuple(
        int(x) for x in os.environ.get("MSAE_LAYERS", "8,20,36,44").split(",")
    ))
    n_pcs: int = 64
    batch_size: int = int(os.environ.get("MSAE_BATCH_SIZE", "16"))  # 27B is heavy
    use_bf16: bool = True
    n_folds: int = 5
    # 3D Duchon center lattice: regular 5x5x5 over [0,1]^3. Adjustable via env.
    lattice_per_side: int = int(os.environ.get("MSAE_LATTICE", "5"))
    init_log_lambda: float = 0.0
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/COLOR_MANIFOLD_GAM")
    save_residuals: bool = True
    # When set, skip local model loading + forward pass entirely and load the
    # (N_prompts, D) residual matrix from disk. Accepted formats:
    #   .npy — bare array of shape (N, D)
    #   .npz — dict with key 'X' of shape (N, D) (this is what
    #          color_geometry.py's incremental cache writes)
    # Used to point at /v1/encode harvests from cogito-probed (or any other
    # external model). With this set, ``layers`` must be a single layer
    # matching the harvest's layer; analysis runs on however many full colors
    # the cache covers (we floor to floor(N / n_templates)).
    harvest_from: str = os.environ.get("MSAE_HARVEST_FROM", "")
    # Comma-separated template indices (0..27) to keep when averaging per
    # color. Empty/unset = use all 28 templates. Setting this to e.g.
    # "8,13,16,17,18,5" restricts the per-color centroid to a hand-picked
    # subset (e.g. the color-focused templates that score highest in the
    # per-template alignment analysis).
    template_subset: str = os.environ.get("MSAE_TEMPLATE_SUBSET", "")


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("no blocks")


# =============================================================================
# Harvest
# =============================================================================
def harvest(cfg: Config, model, tok, blocks, prompts: list[str], device) -> dict[int, torch.Tensor]:
    """Return dict[layer_idx] = (N, D) of last-token residuals (cpu, fp32)."""
    caps: dict[int, list] = {L: [] for L in cfg.layers}
    handles = []
    for L in cfg.layers:
        def make(L=L):
            return lambda m, i, o: caps[L].append(
                (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().cpu().float()
            )
        handles.append(blocks[L].register_forward_hook(make()))
    if tok.padding_side != "left":
        tok.padding_side = "left"
    with torch.no_grad():
        for s in range(0, len(prompts), cfg.batch_size):
            batch = prompts[s:s + cfg.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=64).to(device)
            model(**enc)
    for h in handles:
        h.remove()
    return {L: torch.cat(caps[L], dim=0) for L in cfg.layers}


# =============================================================================
# 3D Duchon basis (pure radial, m=2, s=1 → kernel = r, null space {1,R,G,B})
# =============================================================================
def lattice_centers(per_side: int) -> np.ndarray:
    """Regular `per_side^3` lattice in [0,1]^3."""
    ax = np.linspace(0.0, 1.0, per_side)
    R, G, B = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)  # (K, 3)


# gamfit ≥ 0.1.109 exposes the multi-d Duchon basis + matched function-norm
# penalty for all d we need. No hand-rolled kernel; see duchon_basis_radial.


def duchon_basis_radial(X: np.ndarray, centers: np.ndarray,
                          periodic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Multi-d Duchon design + matched function-norm penalty.

    All paths delegate to gamfit. gamfit requires `2(p+s) > d` for the
    Duchon spline to be well-defined, which means m=2 only handles
    d ∈ {1, 2, 3}. For higher d we just bump m until gamfit accepts the
    dimensionality (higher m = smoother kernel; same Sobolev-space
    polyharmonic-spline theory under the hood).

    Returns (Phi, P) — a matched pair in the same coefficient space.
    """
    import gamfit
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    d = X.shape[1]
    per_axis = (bool(periodic),) * d
    last_err: Exception | None = None
    for m in range(2, 12):
        try:
            Phi = np.asarray(gamfit.duchon_basis(
                X, centers, m=m, periodic_per_axis=per_axis,
            ))
            P = np.asarray(gamfit.duchon_function_norm_penalty(
                centers, m=m, periodic_per_axis=per_axis,
            ))
            return Phi, P
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"duchon_basis_radial: no m in [2, 12] accepted by gamfit "
                       f"for d={d}, periodic={periodic}: {last_err}")


def bspline_1d_basis(t: np.ndarray, n_basis: int = 10, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """1D penalized B-spline basis + 2nd-order difference penalty —
    every choice delegated to gamfit:

      * knot vector: built by gamfit's `_resolve_knots_tensor` (clamped
        with quantile-spaced interior knots derived from t)
      * basis: `gt.bspline_basis`
      * penalty: `gt.smoothness_penalty`

    `n_basis` only controls the interior-knot count (gamfit clamps the
    boundaries). No hand-rolled knots, no hand-rolled penalty.
    """
    import gamfit.torch as gt
    from gamfit.torch._basis import _resolve_knots_tensor
    t_t = torch.from_numpy(np.ascontiguousarray(t, dtype=np.float64))
    # Pass `n_basis` (an int) -> gamfit picks quantile-spaced interior knots
    # of that count and clamps the ends; same knots feed basis + penalty.
    knots_t = _resolve_knots_tensor(t_t, n_basis, degree=degree)
    with torch.no_grad():
        B_t = gt.bspline_basis(t_t, knots_t, degree=degree, periodic=False)
        P_t, _null = gt.smoothness_penalty(knots_t, degree=degree, order=2)
    return B_t.detach().cpu().numpy(), P_t.detach().cpu().numpy()


# =============================================================================
# Smoothing-parameter selection via gamfit's REML primitive
# =============================================================================
_PSD_JITTER_REL = 1e-8


def _psd_clean(P: np.ndarray) -> np.ndarray:
    """Symmetrize + add relative-jitter ridge so gamfit's strict PSD check
    accepts the penalty even when float-precision noise leaves the matrix
    slightly indefinite."""
    P = 0.5 * (P + P.T)
    diag_max = float(np.max(np.abs(np.diag(P)))) if P.shape[0] > 0 else 1.0
    return P + _PSD_JITTER_REL * max(diag_max, 1.0) * np.eye(P.shape[0])


def _additive_fit_predict(designs_tr: list, penalties: list, train_Z: np.ndarray,
                            designs_te: list) -> tuple[np.ndarray, np.ndarray]:
    """Multi-smooth single-GAM fit via `gt.gaussian_reml_fit_additive`.
    Returns (train_pred, test_pred) for joint multi-output Z.

    gamfit picks one shared λ across the smooths; per-smooth coefficients
    come back as a list of blocks, which we use with the per-smooth test
    designs to assemble a prediction.
    """
    import gamfit.torch as gt
    designs_tr_t = [torch.from_numpy(np.ascontiguousarray(d, dtype=np.float64))
                     for d in designs_tr]
    pen_t = [torch.from_numpy(np.ascontiguousarray(_psd_clean(P), dtype=np.float64))
              for P in penalties]
    y_t = torch.from_numpy(np.ascontiguousarray(train_Z, dtype=np.float64))
    with torch.no_grad():
        out = gt.gaussian_reml_fit_additive(designs_tr_t, y_t, pen_t)
    train_pred = out.fitted.detach().cpu().numpy()
    coef_blocks = [c.detach().cpu().numpy() for c in out.coefficients]
    test_pred = sum(d @ c for d, c in zip(designs_te, coef_blocks))
    return train_pred, test_pred


def reml_fit(Phi: np.ndarray, Z: np.ndarray, P: np.ndarray,
              init_log_lambda: float = 0.0) -> tuple[np.ndarray, float]:
    """Fit coefficients B (K, R) for the multi-output smooth Z = Phi · B + ε
    with penalty λ · β' P β. The smoothing parameter λ is chosen by
    **gamfit's closed-form REML** (Rust core).

    Inputs are numpy; conversion to torch + back is local. The penalty is
    symmetrized + ridge-jittered to survive gamfit's strict PSD check
    when the structural penalty has float-precision indefiniteness on the
    null-space block.
    """
    import gamfit.torch as gt
    P_clean = _psd_clean(P)
    init_lam = float(math.exp(init_log_lambda)) if init_log_lambda is not None else None
    x_t = torch.from_numpy(np.ascontiguousarray(Phi, dtype=np.float64))
    y_t = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64))
    p_t = torch.from_numpy(np.ascontiguousarray(P_clean, dtype=np.float64))
    with torch.no_grad():
        out = gt.gaussian_reml_fit(x_t, y_t, p_t, init_lambda=init_lam)
    B = out.coefficients.detach().cpu().numpy()
    lam = float(out.lam.item())
    log_lambda = math.log(max(lam, 1e-30))
    return B, log_lambda


def ridge_fit(Phi: np.ndarray, Z: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Plain ridge via gamfit's `gaussian_weighted_ridge` (closed-form,
    Rust core). The "penalty" here is identity-on-coefficients to give
    standard ridge (X'X + αI)⁻¹X'Y, with unit row weights."""
    import gamfit.torch as gt
    N, M = Phi.shape
    X_t = torch.from_numpy(np.ascontiguousarray(Phi, dtype=np.float64))
    Y_t = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64))
    P_t = torch.eye(M, dtype=torch.float64)
    w_t = torch.ones(N, dtype=torch.float64)
    with torch.no_grad():
        coef_t, _ = gt.gaussian_weighted_ridge(X_t, Y_t, P_t, w_t, ridge_lambda=float(alpha))
    return coef_t.detach().cpu().numpy()


# =============================================================================
# Unsupervised manifold fitting — principal-curve / principal-manifold of
# the activation cloud. The intrinsic coords are LATENT, discovered by
# alternating between smooth-fit and per-point projection. No GT axes given.
# =============================================================================
def lattice_centers_nd(per_side: int, d: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Regular `per_side^d` lattice in [lo, hi]^d."""
    axes = [np.linspace(lo, hi, per_side) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.flatten() for m in mesh], axis=1)


def _rescale_T_to_unit(T: np.ndarray) -> np.ndarray:
    """Per-axis affine rescale of T so each column spans [0, 1]. Stable when
    the column has near-zero spread (then leave at 0.5)."""
    lo = T.min(axis=0, keepdims=True)
    hi = T.max(axis=0, keepdims=True)
    span = (hi - lo)
    out = np.where(span > 1e-8, (T - lo) / np.maximum(span, 1e-8), 0.5 * np.ones_like(T))
    return out


def _initialize_T_from_pca(Z: np.ndarray, d: int) -> np.ndarray:
    """Initial latent coords = top-d PCs of Z, rescaled to [0, 1]^d."""
    Zc = Z - Z.mean(0, keepdims=True)
    Vt = _pca_Vt(Zc)
    T = Zc @ Vt.T[:, :d]
    return _rescale_T_to_unit(T)


def _project_points_batched(Z_target: np.ndarray, B: np.ndarray, centers: np.ndarray,
                              d: int, grid_per_axis: int) -> np.ndarray:
    """Vectorized projection of an (N, K_pcs) batch onto the manifold."""
    grid = lattice_centers_nd(grid_per_axis, d, 0.0, 1.0)
    Phi_grid, _ = duchon_basis_radial(grid, centers)
    Z_grid = Phi_grid @ B                                          # (M, K_pcs)
    # ‖z_n − Z_grid_m‖² = ‖z_n‖² − 2 z_n·Z_grid_m + ‖Z_grid_m‖²
    Zn_sq = (Z_target ** 2).sum(axis=1, keepdims=True)             # (N, 1)
    Zg_sq = (Z_grid ** 2).sum(axis=1, keepdims=True).T             # (1, M)
    cross = Z_target @ Z_grid.T                                    # (N, M)
    sqd = Zn_sq - 2 * cross + Zg_sq
    best = np.argmin(sqd, axis=1)                                  # (N,)
    return grid[best]                                              # (N, d)


def fit_unsupervised_manifold(Z: np.ndarray, d: int, cfg: Config,
                                n_iters: int = 12,
                                centers_per_axis: int | None = None,
                                grid_per_axis: int | None = None,
                                verbose: bool = False,
                                init_T: np.ndarray | None = None,
                                ) -> dict:
    """Alternating fit of a d-dim principal manifold through Z (N, K_pcs).

    Returns:
      T_final  (N, d)  — discovered latent coords in [0, 1]^d
      B        (K_basis, K_pcs) — smooth coefficients
      centers  (n_centers, d)
      log_lambda — REML-selected log-λ
      history  list of {iter, dT, train_mse, log_lambda}
    """
    if centers_per_axis is None:
        # Auto-shrink center count so the basis isn't overcomplete vs the
        # training rows.  Floor for U_4d is 3 → 81 centers (which still
        # needs ≥ 86 training rows after the nullspace; for smaller folds
        # the caller's CV-loop will catch it and skip).
        defaults = {1: 16, 2: 8, 3: 5, 4: 3, 5: 3, 6: 2, 8: 2}
        centers_per_axis = defaults[d]
    if grid_per_axis is None:
        grid_per_axis = {1: 200, 2: 40, 3: 20, 4: 9, 5: 5, 6: 4, 8: 3}[d]

    centers = lattice_centers_nd(centers_per_axis, d, 0.0, 1.0)
    T = init_T.copy() if init_T is not None else _initialize_T_from_pca(Z, d)
    history = []
    last_log_lambda = cfg.init_log_lambda

    for it in range(n_iters):
        Phi, P = duchon_basis_radial(T, centers)
        B, last_log_lambda = reml_fit(Phi, Z, P, init_log_lambda=last_log_lambda)

        # Projection step (re-fit T given B)
        T_new = _project_points_batched(Z, B, centers, d, grid_per_axis)
        # Stability: rescale to [0,1]^d so centers stay in support
        T_new = _rescale_T_to_unit(T_new)
        dT = float(np.linalg.norm(T_new - T) / max(1.0, np.linalg.norm(T)))
        Z_hat = Phi @ B
        train_mse = float(((Z - Z_hat) ** 2).mean())
        history.append({"iter": it, "dT": dT, "train_mse": train_mse,
                          "log_lambda": last_log_lambda})
        if verbose:
            print(f"      [unsup d={d} iter {it:2d}] dT={dT:.3e} "
                  f"mse={train_mse:.3e} log_lam={last_log_lambda:+.2f}", flush=True)
        T = T_new
        if dT < 1e-4:
            break

    # Final smooth fit at converged T
    Phi, P = duchon_basis_radial(T, centers)
    B, last_log_lambda = reml_fit(Phi, Z, P, init_log_lambda=last_log_lambda)

    return {"T": T, "B": B, "centers": centers, "log_lambda": last_log_lambda,
            "history": history, "centers_per_axis": centers_per_axis,
            "grid_per_axis": grid_per_axis}


def predict_unsupervised(Z_target: np.ndarray, fit: dict, d: int,
                          grid_per_axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project unseen Z_target onto a fitted unsupervised manifold.

    Returns (T_test (N, d), Z_pred (N, K_pcs)).
    """
    g: int = int(grid_per_axis if grid_per_axis is not None else fit["grid_per_axis"])
    T_test = _project_points_batched(Z_target, fit["B"], fit["centers"], d, g)
    Phi_test, _ = duchon_basis_radial(T_test, fit["centers"])
    Z_pred = Phi_test @ fit["B"]
    return T_test, Z_pred


# =============================================================================
# Spec definitions
# =============================================================================
def coord_rgb(R, G, B, hsv) -> np.ndarray:
    return np.stack([R, G, B], axis=1)


def coord_hsv_periodic(R, G, B, hsv) -> np.ndarray:
    h = hsv[:, 0]
    return np.stack([np.cos(2 * np.pi * h), np.sin(2 * np.pi * h), hsv[:, 1], hsv[:, 2]], axis=1)


SPECS = {
    # SUPERVISED: GT axes drive the parameterization
    "L_lin_rgb": ("ridge_rgb",),
    "L_lin_hsv": ("ridge_hsv",),
    "L_lin_lab": ("ridge_lab",),                       # CIE-Lab, perceptually uniform
    "L_lin_luminance": ("ridge_luminance",),           # single-axis brightness only
    "L_poly_rgb": ("poly_rgb",),                       # degree-2 RGB polynomial (linear+cross)
    "L_poly_hsv": ("poly_hsv",),                       # degree-2 HSV-periodic polynomial
    "L_add_rgb": ("additive_rgb",),
    "L_add_hsv": ("additive_hsv",),
    "L_joint_rgb": ("smooth_rgb",),
    "L_joint_hsv": ("smooth_hsv",),
    "L_joint_lab": ("smooth_lab",),                    # 3D Duchon in CIE-Lab
    "L_joint_rgb_with_hue": ("smooth_rgb_plus_hue",),
    "L_tensor_bspline_rgb": ("tensor_bspline_rgb",),   # tensor-product B-spline in RGB
    # PERCEPTUAL color spaces (Oklab, Lch) — different inductive biases for "color distance"
    "L_lin_oklab":           ("ridge_oklab",),           # linear in Oklab (2020 state of the art)
    "L_joint_oklab":         ("smooth_oklab",),          # 3D Duchon in Oklab
    "L_lin_lch":             ("ridge_lch",),             # linear in CIE-Lch (cylindrical Lab)
    "L_lch_with_cyclic_h":   ("lch_with_cyclic_hue",),   # 2D Duchon (L, C) + cyclic h
    "L_lab_with_cyclic_hue": ("lab_with_cyclic_hue",),   # 3D Lab + cyclic h additive
    "L_perceptual_add":      ("perceptual_additive",),   # 1D B-splines on (L, C) + cyclic h
    # HIGHER-DEGREE polynomials
    "L_poly3_rgb": ("poly3_rgb",),                       # degree-3 RGB polynomial
    "L_poly3_hsv": ("poly3_hsv",),                       # degree-3 HSV polynomial
    "L_poly3_lab": ("poly3_lab",),                       # degree-3 polynomial in CIE-Lab
    "L_poly4_hsv": ("poly4_hsv",),                       # degree-4 HSV polynomial — overfit?
    "L_poly_lab":  ("poly_lab",),                        # degree-2 polynomial in CIE-Lab
    "L_poly_oklab":("poly_oklab",),                      # degree-2 polynomial in Oklab
    "L_poly_lch":  ("poly_lch",),                        # degree-2 polynomial in Lch (h lifted to cos/sin)
    "L_const_mean": ("const_mean_baseline",),            # predicts training Z mean — R² should be ≈ 0
    # MORE SURFACE FITS — alternative parametrizations
    "M_hsv_bicone":          ("manifold_hsv_bicone",),       # double-cone (sat = 0 at both V=0 and V=1)
    "M_chroma_disk":         ("manifold_chroma_disk",),      # 2D color wheel disk (chroma*cos h, chroma*sin h)
    "M_chroma_disk_plus_L":  ("manifold_disk_plus_L",),      # disk + 1D lightness
    "M_rgb_finer_grid":      ("manifold_rgb_7x7x7",),        # 7³=343 centers vs default 5³=125
    "L_kernel_rbf_rgb":      ("kernel_rbf_rgb",),            # explicit Gaussian RBF kernel ridge
    "L_rgb_lab_combo":       ("rgb_lab_multi_smooth",),      # 3D RGB + 3D Lab additive
    "L_joint_oklab_with_h": ("joint_oklab_with_h",),     # 3D Oklab Duchon + cyclic hue
    "L_chroma_lum_2d":       ("chroma_lum_2d",),         # 2D Duchon on (chroma, lightness) — no hue
    "L_hue_polyharmonic":    ("hue_polyharmonic",),      # cyclic B-spline on hue + B-splines on C, L
    # k-NN broader sweep
    "N_knn_rgb_k30":  ("knn_rgb_30",),
    "N_knn_lab_k30":  ("knn_lab_30",),
    # k-NN SWEEP (the kNN family is competitive — see what k is best)
    "N_knn_rgb_k5":   ("knn_rgb_5",),
    "N_knn_rgb_k20":  ("knn_rgb_20",),
    "N_knn_lab_k5":   ("knn_lab_5",),
    "N_knn_lab_k20":  ("knn_lab_20",),
    "N_knn_oklab_k10":("knn_oklab_10",),
    # CYCLIC B-spline (non-Duchon periodic) and multi-smooth combinations
    "L_cyclic_hue":                ("cyclic_hue",),               # 1D cyclic B-spline on hue
    "L_cyclic_hue_plus_lin_v":     ("cyclic_hue_plus_lin_v",),    # + linear value
    "L_cyclic_hue_plus_bspline_v": ("cyclic_hue_plus_bspline_v",),# + 1D B-spline value
    "L_cyclic_hue_plus_bspline_s_v": ("cyclic_hue_plus_bspline_s_v",), # + B-spline s + B-spline v
    "L_cyclic_hue_plus_lin_rgb":   ("cyclic_hue_plus_lin_rgb",),  # + linear RGB
    "L_cyclic_hue_plus_joint_rgb": ("cyclic_hue_plus_joint_rgb",),# + 3D Duchon RGB
    # MANIFOLD smooths — different topology assumptions
    "M_cyl_hue_val":     ("manifold_cyl_hue_val",),       # S¹ × ℝ — hue periodic, value linear
    "M_torus_hue_sat":   ("manifold_torus_hue_sat",),     # S¹ × S¹ — both periodic (silly but tests it)
    "M_torus_hue_val":   ("manifold_torus_hue_val",),     # S¹ × S¹ — periodic value as a sanity check
    "M_sphere_hueval":   ("manifold_sphere_hueval",),     # S² — Runge color sphere (lat=value, lon=hue)
    "M_sphere_plus_chroma": ("manifold_sphere_plus_chroma",),  # S² + 1D chroma — multi-smooth
    "M_hsv_cone":        ("manifold_hsv_cone",),          # cone embed (sv·cos h, sv·sin h, v)
    # NONPARAMETRIC baselines (no smooth prior)
    "N_knn_rgb_k10":   ("knn_rgb_10",),                # k=10 nearest in RGB-Euclidean
    "N_knn_hsv_k10":   ("knn_hsv_10",),                # k=10 nearest in HSV-periodic-Euclidean
    "N_knn_lab_k10":   ("knn_lab_10",),                # k=10 nearest in Lab-Euclidean
    # UNSUPERVISED: latent parameterization, discovered by alternation
    "U_1d": ("unsup_1d",),
    "U_2d": ("unsup_2d",),
    "U_3d": ("unsup_3d",),
    "U_4d": ("unsup_4d",),
    "U_5d": ("unsup_5d",),
    "U_6d": ("unsup_6d",),
    "U_8d": ("unsup_8d",),
    # NEW unsupervised approaches (different inductive biases)
    "U_pca_2d": ("unsup_pca_2d",),                   # linear top-2 PCA reconstruction
    "U_pca_3d": ("unsup_pca_3d",),                   # linear top-3 — baseline for U_3d
    "U_pca_4d": ("unsup_pca_4d",),
    "U_pca_8d": ("unsup_pca_8d",),
    "U_pca_16d": ("unsup_pca_16d",),
    "U_pca_24d": ("unsup_pca_24d",),
    "U_pca_32d": ("unsup_pca_32d",),
    "U_pca_48d": ("unsup_pca_48d",),
    "U_pca_64d": ("unsup_pca_64d",),
    "U_pca_96d": ("unsup_pca_96d",),
    "U_pca_128d": ("unsup_pca_128d",),
    # Hybrid: PCA latent + nonlinear Duchon smooth back to residuals
    "U_pca8_smooth": ("unsup_pca_then_smooth_8d",),
    "U_pca16_smooth": ("unsup_pca_then_smooth_16d",),
    # Additive 1D smooths on PCA latent
    "U_pca_add_3d":  ("unsup_pca_additive_3d",),
    "U_pca_add_8d":  ("unsup_pca_additive_8d",),
    "U_pca_add_16d": ("unsup_pca_additive_16d",),
    # 2D-pair Duchon smooths on PCA latent (additive over consecutive pairs)
    "U_pca_pairs_4d": ("unsup_pca_pairs_4d",),
    "U_pca_pairs_8d": ("unsup_pca_pairs_8d",),
    # PCA-init U_3d — start alternation at PCA-3 instead of random
    "U_3d_pca_init": ("unsup_3d_pca_init",),
    # PCA latent + tensor B-spline (different smooth family)
    "U_pca3_tensor": ("unsup_pca_tensor_3d",),
    # Non-negative matrix factorization (parts-based decomposition)
    "U_nmf_8d":   ("unsup_nmf_8d",),
    "U_nmf_16d":  ("unsup_nmf_16d",),
    # Robust PCA via L1 (less sensitive to outlier colors)
    "U_centroid_kde_smooth_3d": ("unsup_kde_3d",),
    # Best-PCA-init followed by ridge-on-top
    "U_pca_centered_8d_smooth": ("unsup_pca8_with_smooth",),
    # ALL-DUCHON unsupervised GAM zoo on PCA latents (no B-splines)
    "U_pca3_duchon_joint":      ("unsup_pca3_duchon_joint",),
    "U_pca4_duchon_joint":      ("unsup_pca4_duchon_joint",),
    "U_pca6_duchon_joint":      ("unsup_pca6_duchon_joint",),
    "U_pca8_duchon_add1d":      ("unsup_pca8_duchon_additive",),     # 1D Duchon per PC, additive
    "U_pca16_duchon_add1d":     ("unsup_pca16_duchon_additive",),
    "U_pca24_duchon_add1d":     ("unsup_pca24_duchon_additive",),    # NEW — push higher
    "U_pca32_duchon_add1d":     ("unsup_pca32_duchon_additive",),
    "U_pca48_duchon_add1d":     ("unsup_pca48_duchon_additive",),
    "U_pca16_duchon_pairs":     ("unsup_pca16_duchon_pairs",),
    "U_pca6_duchon_triples":    ("unsup_pca6_duchon_triples",),       # 3D Duchon on PC triples (additive)
    "U_pca12_duchon_triples":   ("unsup_pca12_duchon_triples",),
    "U_pca8_duchon_m3":         ("unsup_pca8_duchon_m3",),            # higher m = smoother kernel
    "U_pca16_duchon_m3":        ("unsup_pca16_duchon_m3",),
    "U_pca8_duchon_finer_centers":  ("unsup_pca8_duchon_finer",),     # 200 centers vs 60 default
    "U_kmeans_10": ("unsup_kmeans_10",),             # cluster-and-mean prediction
    "U_kmeans_30": ("unsup_kmeans_30",),
    "U_kmeans_50": ("unsup_kmeans_50",),
    "U_loop_1d": ("unsup_loop_1d",),                 # 1D periodic latent (S¹)
    "U_3d_multistart": ("unsup_3d_multistart",),     # U_3d with 5 random inits, pick best
}


# =============================================================================
# Color space conversions and feature constructors used by new specs
# =============================================================================

def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB ∈ [0,1] → CIE-Lab, D65. Standard formula (no clipping)."""
    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    rgb_lin = linearize(rgb01)
    xyz = rgb_lin @ M.T
    # D65 white reference
    xyz_n = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    def f(t):
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)
    fxyz = f(xyz_n)
    L = 116.0 * fxyz[:, 1] - 16.0
    a = 500.0 * (fxyz[:, 0] - fxyz[:, 1])
    b = 200.0 * (fxyz[:, 1] - fxyz[:, 2])
    return np.stack([L, a, b], axis=1)


def _poly_features_degree2(X: np.ndarray) -> np.ndarray:
    """Build [1, x_i, x_i x_j] degree-2 features. X is (N, d) → (N, 1 + d + d(d+1)/2)."""
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def _poly_features_degree3(X: np.ndarray) -> np.ndarray:
    """Degree-3 polynomial features. X (N, d) → (N, 1 + d + d(d+1)/2 + ...)."""
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append((X[:, i] * X[:, j] * X[:, k])[:, None])
    return np.concatenate(cols, axis=1)


def rgb_to_oklab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB ∈ [0,1] → Oklab (Björn Ottosson, 2020) — current state of the
    art perceptually uniform color space. Standard formula, no clipping."""
    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    rgb_lin = linearize(rgb01)
    M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ])
    lms = rgb_lin @ M1.T
    lms_cbrt = np.cbrt(lms)
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    return lms_cbrt @ M2.T          # (L, a, b)


def rgb_to_lch(rgb01: np.ndarray) -> np.ndarray:
    """RGB → CIE-Lch (cylindrical Lab): L (lightness, 0-100), C (chroma, ≥0),
    h (hue angle in [0,1])."""
    lab = rgb_to_lab(rgb01)
    L = lab[:, 0]
    C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    h = (np.arctan2(lab[:, 2], lab[:, 1]) / (2 * np.pi)) % 1.0
    return np.stack([L, C, h], axis=1)


def bspline_1d_cyclic_basis(t: np.ndarray, n_basis: int = 12, degree: int = 3,
                            knots=None, return_knots: bool = False):
    """Cyclic (periodic) 1D B-spline. Inputs in [0, 1]; the basis wraps so
    the fit doesn't have a seam at 0/1.

    Penalty: cyclic second-difference operator on coefficients. This is the
    standard matched penalty for a periodic B-spline (Eilers-Marx / mgcv
    bs="cc" style). gamfit's smoothness_penalty currently returns the
    non-periodic matrix sized to the open-knot basis count, which doesn't
    match the wrapped-basis column count — so we build the cyclic penalty
    manually here.

    If `knots` (a resolved knot tensor) is given, the basis is evaluated on
    THOSE fixed knots (so out-of-sample / single-point evaluation yields the
    SAME basis columns as the fit). Otherwise gamfit resolves quantile-spaced
    interior knots from `t` for the integer `n_basis`. With return_knots=True
    the resolved knot tensor and effective degree are also returned, so a
    caller can reuse them for the derivative / prediction basis.
    """
    import gamfit.torch as gt
    from gamfit.torch._basis import _resolve_knots_tensor
    t_t = torch.from_numpy(np.ascontiguousarray(t, dtype=np.float64))
    if knots is None:
        knots, degree = _resolve_knots_tensor(t_t, n_basis, degree=degree)
    with torch.no_grad():
        B_t = gt.bspline_basis(t_t, knots, degree=degree, periodic=True)
    B = B_t.detach().cpu().numpy()
    k = B.shape[1]
    # Cyclic 2nd-difference operator D (k × k): row i = e_i − 2 e_{i+1} + e_{i+2}, mod k
    D = np.zeros((k, k))
    for i in range(k):
        D[i, i] = 1.0
        D[i, (i + 1) % k] = -2.0
        D[i, (i + 2) % k] = 1.0
    P = D.T @ D
    if return_knots:
        return B, P, knots, int(degree)
    return B, P


def _knn_predict(train_X: np.ndarray, train_Z: np.ndarray, test_X: np.ndarray,
                   k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-NN regression in the feature space `train_X`. Returns
    (train_pred, test_pred) where train_pred is leave-one-out for fairness."""
    def knn_pred(query, ref_X, ref_Z, exclude_self: bool):
        sq = np.sum((query[:, None, :] - ref_X[None, :, :]) ** 2, axis=2)
        if exclude_self:
            np.fill_diagonal(sq, np.inf)
        idx = np.argpartition(sq, k, axis=1)[:, :k]
        return np.array([ref_Z[idx[i]].mean(axis=0) for i in range(query.shape[0])])
    return (
        knn_pred(train_X, train_X, train_Z, exclude_self=True),
        knn_pred(test_X, train_X, train_Z, exclude_self=False),
    )


# =============================================================================
# Fit + CV
# =============================================================================
def kfold_color_indices(n_colors: int, n_folds: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_colors)
    fold_of = np.empty(n_colors, dtype=int)
    fold_of[perm] = np.arange(n_colors) % n_folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_folds)]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between 1D arrays x and y."""
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if len(x) != len(y) or len(x) < 2: return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def fit_and_predict(spec_name: str, train_X_rgb, train_X_hsv, train_Z,
                     test_X_rgb, test_X_hsv, test_Z, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Fit one spec on train, predict for test. Return (Z_train_pred, Z_test_pred).
    Z_train_pred is returned for the variance-decomposition use."""
    centers_rgb = lattice_centers(cfg.lattice_per_side)
    centers_hsv = None  # built ad hoc below

    # Inputs: train_X_rgb has shape (N, 3) in [0,1]; train_X_hsv has 4 cols
    # (cos 2πh, sin 2πh, sat, val). Lab features built on demand.
    train_X_lab = rgb_to_lab(train_X_rgb)
    test_X_lab = rgb_to_lab(test_X_rgb)

    if spec_name == "L_lin_rgb":
        Phi_tr = np.concatenate([train_X_rgb, np.ones((train_X_rgb.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_rgb, np.ones((test_X_rgb.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_lin_lab":
        # CIE-Lab is perceptually-uniform: equal Euclidean distance ≈ equal
        # perceived color difference. If cogito encodes colors perceptually
        # (rather than chromatically), this should beat RGB linear.
        Phi_tr = np.concatenate([train_X_lab, np.ones((train_X_lab.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_lab, np.ones((test_X_lab.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_lin_luminance":
        # Single-axis baseline: just the perceptual brightness.
        # If THIS gives ~the same R² as anything else, color information
        # at this layer is essentially 1D (= just lightness).
        lum_tr = (0.299 * train_X_rgb[:, 0] + 0.587 * train_X_rgb[:, 1] +
                  0.114 * train_X_rgb[:, 2])[:, None]
        lum_te = (0.299 * test_X_rgb[:, 0] + 0.587 * test_X_rgb[:, 1] +
                  0.114 * test_X_rgb[:, 2])[:, None]
        Phi_tr = np.concatenate([lum_tr, np.ones((lum_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([lum_te, np.ones((lum_te.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly_rgb":
        # Degree-2 polynomial in RGB — bridges linear and nonlinear.
        # Tests whether the gap between linear-RGB and joint-Duchon-RGB
        # comes from cross-axis interactions (captured here) or from
        # higher-order smoothness (only captured by Duchon).
        Phi_tr = _poly_features_degree2(train_X_rgb)
        Phi_te = _poly_features_degree2(test_X_rgb)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly_hsv":
        Phi_tr = _poly_features_degree2(train_X_hsv)
        Phi_te = _poly_features_degree2(test_X_hsv)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_joint_lab":
        # 3D Duchon in CIE-Lab space — perceptual-cube analogue of joint_rgb.
        # Centers placed on the convex-hull-shifted unit cube of the data.
        lab_min = train_X_lab.min(0); lab_max = train_X_lab.max(0)
        lab_ranges = lab_max - lab_min + 1e-9
        per_side = cfg.lattice_per_side
        ax = [np.linspace(lab_min[d], lab_max[d], per_side) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_tr, P = duchon_basis_radial(train_X_lab, centers)
        Phi_te, _ = duchon_basis_radial(test_X_lab, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_tensor_bspline_rgb":
        # Tensor-product B-spline: independent 1D B-spline basis per axis,
        # Kronecker'd together. Tests "axes encoded independently with
        # nonlinear shape" — a strong contender against the radial joint.
        # We use 10 basis fns per axis → 1000 product features, then ridge.
        def tprod_features(X, n_per_axis: int = 8):
            cols = []
            for ax in range(3):
                Bk, _ = bspline_1d_basis(X[:, ax], n_basis=n_per_axis)
                cols.append(Bk)
            # Tensor product across the 3 marginals
            out = cols[0][:, :, None, None] * cols[1][:, None, :, None] * cols[2][:, None, None, :]
            return out.reshape(X.shape[0], -1)
        Phi_tr = tprod_features(train_X_rgb)
        Phi_te = tprod_features(test_X_rgb)
        W = ridge_fit(Phi_tr, train_Z, alpha=10.0)        # mildly regularize
        return Phi_tr @ W, Phi_te @ W

    # =====================================================================
    # CYCLIC B-spline (non-Duchon periodic) + multi-smooth combinations
    # =====================================================================
    train_hue = _hue_from_X_hsv(train_X_hsv) if False else None     # avoid double-defining

    # We need hue in [0, 1] for cyclic. Reconstruct from X_hsv's (cos, sin).
    h_tr = (np.arctan2(train_X_hsv[:, 1], train_X_hsv[:, 0]) / (2 * np.pi)) % 1.0
    h_te = (np.arctan2(test_X_hsv[:, 1], test_X_hsv[:, 0]) / (2 * np.pi)) % 1.0

    if spec_name == "L_cyclic_hue":
        Phi_tr, P = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_cyclic_hue_plus_lin_v":
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        # Linear value is just (v, 1) — give it a tiny ridge penalty.
        v_tr = train_X_hsv[:, 3:4]; v_te = test_X_hsv[:, 3:4]
        Phi_v_tr = np.concatenate([v_tr, np.ones_like(v_tr)], axis=1)
        Phi_v_te = np.concatenate([v_te, np.ones_like(v_te)], axis=1)
        P_v = 1e-3 * np.eye(2)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_v_tr], [P_h, P_v], train_Z, [Phi_h_te, Phi_v_te],
        )

    if spec_name == "L_cyclic_hue_plus_bspline_v":
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        Phi_v_tr, P_v = bspline_1d_basis(train_X_hsv[:, 3], n_basis=10)
        Phi_v_te, _ = bspline_1d_basis(test_X_hsv[:, 3], n_basis=10)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_v_tr], [P_h, P_v], train_Z, [Phi_h_te, Phi_v_te],
        )

    if spec_name == "L_cyclic_hue_plus_bspline_s_v":
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        Phi_s_tr, P_s = bspline_1d_basis(train_X_hsv[:, 2], n_basis=10)
        Phi_s_te, _ = bspline_1d_basis(test_X_hsv[:, 2], n_basis=10)
        Phi_v_tr, P_v = bspline_1d_basis(train_X_hsv[:, 3], n_basis=10)
        Phi_v_te, _ = bspline_1d_basis(test_X_hsv[:, 3], n_basis=10)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_s_tr, Phi_v_tr], [P_h, P_s, P_v], train_Z,
            [Phi_h_te, Phi_s_te, Phi_v_te],
        )

    if spec_name == "L_cyclic_hue_plus_lin_rgb":
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        Phi_rgb_tr = np.concatenate([train_X_rgb, np.ones((train_X_rgb.shape[0], 1))], axis=1)
        Phi_rgb_te = np.concatenate([test_X_rgb, np.ones((test_X_rgb.shape[0], 1))], axis=1)
        P_rgb = 1e-3 * np.eye(4)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_rgb_tr], [P_h, P_rgb], train_Z,
            [Phi_h_te, Phi_rgb_te],
        )

    if spec_name == "L_cyclic_hue_plus_joint_rgb":
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(h_tr, n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(h_te, n_basis=12)
        Phi_rgb_tr, P_rgb = duchon_basis_radial(train_X_rgb, centers_rgb)
        Phi_rgb_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_rgb_tr], [P_h, P_rgb], train_Z,
            [Phi_h_te, Phi_rgb_te],
        )

    # =====================================================================
    # PERCEPTUAL color-space specs (Oklab, Lch)
    # =====================================================================
    train_X_oklab = rgb_to_oklab(train_X_rgb)
    test_X_oklab = rgb_to_oklab(test_X_rgb)
    train_X_lch = rgb_to_lch(train_X_rgb)
    test_X_lch = rgb_to_lch(test_X_rgb)

    if spec_name == "L_lin_oklab":
        Phi_tr = np.concatenate([train_X_oklab, np.ones((train_X_oklab.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_oklab, np.ones((test_X_oklab.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_joint_oklab":
        # 3D Duchon in Oklab. Centers placed on a 5×5×5 lattice bracketing
        # the training data's Oklab bounding box.
        lo, hi = train_X_oklab.min(0), train_X_oklab.max(0)
        ax = [np.linspace(lo[d], hi[d], 5) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_tr, P = duchon_basis_radial(train_X_oklab, centers)
        Phi_te, _ = duchon_basis_radial(test_X_oklab, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_lin_lch":
        # Lch is (L, C, h). Lift h to (cos, sin) so linear can see it.
        h_tr = train_X_lch[:, 2]; h_te = test_X_lch[:, 2]
        feat_tr = np.stack([
            train_X_lch[:, 0], train_X_lch[:, 1],
            np.cos(2*np.pi*h_tr), np.sin(2*np.pi*h_tr),
        ], axis=1)
        feat_te = np.stack([
            test_X_lch[:, 0], test_X_lch[:, 1],
            np.cos(2*np.pi*h_te), np.sin(2*np.pi*h_te),
        ], axis=1)
        Phi_tr = np.concatenate([feat_tr, np.ones((feat_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([feat_te, np.ones((feat_te.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_lch_with_cyclic_h":
        # 2D Duchon on (L, C) + cyclic B-spline on h.
        feat_tr = train_X_lch[:, :2]; feat_te = test_X_lch[:, :2]
        lo, hi = feat_tr.min(0), feat_tr.max(0)
        ax = [np.linspace(lo[d], hi[d], 6) for d in range(2)]
        L_g, C_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), C_g.flatten()], axis=1)
        Phi_LC_tr, P_LC = duchon_basis_radial(feat_tr, centers)
        Phi_LC_te, _ = duchon_basis_radial(feat_te, centers)
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(train_X_lch[:, 2], n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(test_X_lch[:, 2], n_basis=12)
        return _additive_fit_predict(
            [Phi_LC_tr, Phi_h_tr], [P_LC, P_h], train_Z,
            [Phi_LC_te, Phi_h_te],
        )

    if spec_name == "L_lab_with_cyclic_hue":
        # 3D Lab Duchon + cyclic hue (the perceptual analogue of
        # L_joint_rgb_with_hue).
        lab_tr = rgb_to_lab(train_X_rgb); lab_te = rgb_to_lab(test_X_rgb)
        lo, hi = lab_tr.min(0), lab_tr.max(0)
        ax = [np.linspace(lo[d], hi[d], 5) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_lab_tr, P_lab = duchon_basis_radial(lab_tr, centers)
        Phi_lab_te, _ = duchon_basis_radial(lab_te, centers)
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(train_X_lch[:, 2], n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(test_X_lch[:, 2], n_basis=12)
        return _additive_fit_predict(
            [Phi_lab_tr, Phi_h_tr], [P_lab, P_h], train_Z,
            [Phi_lab_te, Phi_h_te],
        )

    if spec_name == "L_perceptual_add":
        # Pure perceptual-additive: 1D B-spline on L (lightness), 1D B-spline
        # on C (chroma), cyclic B-spline on hue. Tests the "color is just
        # three perceptual axes processed independently" hypothesis.
        Phi_L_tr, P_L = bspline_1d_basis(train_X_lch[:, 0] / 100.0, n_basis=10)
        Phi_L_te, _ = bspline_1d_basis(test_X_lch[:, 0] / 100.0, n_basis=10)
        C_max = max(float(train_X_lch[:, 1].max()), 1.0)
        Phi_C_tr, P_C = bspline_1d_basis(train_X_lch[:, 1] / C_max, n_basis=10)
        Phi_C_te, _ = bspline_1d_basis(test_X_lch[:, 1] / C_max, n_basis=10)
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(train_X_lch[:, 2], n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(test_X_lch[:, 2], n_basis=12)
        return _additive_fit_predict(
            [Phi_L_tr, Phi_C_tr, Phi_h_tr],
            [P_L, P_C, P_h], train_Z,
            [Phi_L_te, Phi_C_te, Phi_h_te],
        )

    if spec_name == "L_poly3_rgb":
        Phi_tr = _poly_features_degree3(train_X_rgb)
        Phi_te = _poly_features_degree3(test_X_rgb)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly3_hsv":
        Phi_tr = _poly_features_degree3(train_X_hsv)
        Phi_te = _poly_features_degree3(test_X_hsv)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly_lab":
        Phi_tr = _poly_features_degree2(train_X_lab)
        Phi_te = _poly_features_degree2(test_X_lab)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly3_lab":
        Phi_tr = _poly_features_degree3(train_X_lab)
        Phi_te = _poly_features_degree3(test_X_lab)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly4_hsv":
        def _poly4(X):
            N, d = X.shape
            cols = [np.ones((N, 1)), X]
            for i in range(d):
                for j in range(i, d):
                    cols.append((X[:, i] * X[:, j])[:, None])
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        cols.append((X[:, i] * X[:, j] * X[:, k])[:, None])
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        for l_ in range(k, d):
                            cols.append((X[:, i] * X[:, j] * X[:, k] * X[:, l_])[:, None])
            return np.concatenate(cols, axis=1)
        Phi_tr = _poly4(train_X_hsv)
        Phi_te = _poly4(test_X_hsv)
        W = ridge_fit(Phi_tr, train_Z, alpha=10.0)        # stronger ridge for d4
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "M_hsv_bicone":
        # HSV bicone: saturation vanishes at BOTH lightness extremes (white
        # AND black), not just black. Closer to actual color appearance:
        # very light or very dark colors have washed-out chroma. Coords:
        #   x = s · v · sin(π·v) · cos(2π·h)
        #   y = s · v · sin(π·v) · sin(2π·h)
        #   z = v
        # (The sin(πv) factor pinches the cone at both v=0 and v=1.)
        def to_bicone(X_hsv4):
            hue = (np.arctan2(X_hsv4[:, 1], X_hsv4[:, 0]) / (2*np.pi))
            s, v = X_hsv4[:, 2], X_hsv4[:, 3]
            r = s * v * np.sin(np.pi * v)
            return np.stack([r * np.cos(2*np.pi*hue),
                             r * np.sin(2*np.pi*hue), v], axis=1)
        train_bi = to_bicone(train_X_hsv); test_bi = to_bicone(test_X_hsv)
        # Centers along the bicone surface + axis
        ang = np.linspace(0, 2*np.pi, 8, endpoint=False)
        vs = np.linspace(0.1, 0.9, 4)
        centers = []
        for v in vs:
            r = v * np.sin(np.pi * v)
            for a in ang:
                centers.append([r * np.cos(a), r * np.sin(a), v])
            centers.append([0.0, 0.0, v])
        centers = np.array(centers)
        Phi_tr, P = duchon_basis_radial(train_bi, centers)
        Phi_te, _ = duchon_basis_radial(test_bi, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "M_chroma_disk":
        # 2D chromaticity disk — colors plotted by their (chroma, hue) only,
        # ignoring lightness. (chroma * cos h, chroma * sin h)
        def to_disk(X_lch_3):
            h = X_lch_3[:, 2]
            r = X_lch_3[:, 1] / 100.0     # normalize
            return np.stack([r * np.cos(2*np.pi*h), r * np.sin(2*np.pi*h)], axis=1)
        train_d = to_disk(train_X_lch); test_d = to_disk(test_X_lch)
        ang = np.linspace(0, 2*np.pi, 12, endpoint=False)
        rad = np.linspace(0.0, 1.2, 4)
        centers = []
        for r in rad:
            for a in ang:
                centers.append([r * np.cos(a), r * np.sin(a)])
        centers = np.array(centers)
        Phi_tr, P = duchon_basis_radial(train_d, centers)
        Phi_te, _ = duchon_basis_radial(test_d, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "M_chroma_disk_plus_L":
        # Disk + 1D lightness, additive
        def to_disk(X_lch_3):
            h = X_lch_3[:, 2]; r = X_lch_3[:, 1] / 100.0
            return np.stack([r * np.cos(2*np.pi*h), r * np.sin(2*np.pi*h)], axis=1)
        train_d = to_disk(train_X_lch); test_d = to_disk(test_X_lch)
        ang = np.linspace(0, 2*np.pi, 12, endpoint=False)
        rad = np.linspace(0.0, 1.2, 4)
        centers = np.array([[r*np.cos(a), r*np.sin(a)] for r in rad for a in ang])
        Phi_d_tr, P_d = duchon_basis_radial(train_d, centers)
        Phi_d_te, _ = duchon_basis_radial(test_d, centers)
        Phi_L_tr, P_L = bspline_1d_basis(train_X_lch[:, 0] / 100.0, n_basis=10)
        Phi_L_te, _ = bspline_1d_basis(test_X_lch[:, 0] / 100.0, n_basis=10)
        return _additive_fit_predict(
            [Phi_d_tr, Phi_L_tr], [P_d, P_L], train_Z,
            [Phi_d_te, Phi_L_te],
        )

    if spec_name == "M_rgb_finer_grid":
        # Same 3D RGB Duchon as L_joint_rgb but with 7³=343 centers
        # instead of 5³=125. Tests if more capacity helps.
        ax = np.linspace(0.0, 1.0, 7)
        R, G, B = np.meshgrid(ax, ax, ax, indexing="ij")
        centers = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)
        Phi_tr, P = duchon_basis_radial(train_X_rgb, centers)
        Phi_te, _ = duchon_basis_radial(test_X_rgb, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_kernel_rbf_rgb":
        # Explicit Gaussian RBF kernel ridge — different kernel family than
        # Duchon's polyharmonic. Same {centers, kernel function} structure
        # but a Gaussian decays exponentially while Duchon's r grows linearly.
        from scipy.spatial.distance import cdist
        sigma = 0.25
        D_tr = cdist(train_X_rgb, train_X_rgb)
        D_te = cdist(test_X_rgb, train_X_rgb)
        K_tr = np.exp(-(D_tr / sigma) ** 2 / 2)
        K_te = np.exp(-(D_te / sigma) ** 2 / 2)
        # Kernel ridge: B = (K + α I)^-1 y
        alpha = 1.0
        W = np.linalg.solve(K_tr + alpha * np.eye(K_tr.shape[0]), train_Z)
        return K_tr @ W, K_te @ W

    if spec_name == "L_rgb_lab_combo":
        # Multi-smooth: 3D RGB Duchon + 3D Lab Duchon, additive. Tests
        # whether RGB and Lab capture complementary structure.
        from color_manifold_gam import lattice_centers as _lat
        centers_rgb_local = _lat(cfg.lattice_per_side)
        lab_tr = rgb_to_lab(train_X_rgb); lab_te = rgb_to_lab(test_X_rgb)
        lo, hi = lab_tr.min(0), lab_tr.max(0)
        ax_lab = [np.linspace(lo[d], hi[d], 5) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax_lab, indexing="ij")
        centers_lab = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_rgb_tr, P_rgb = duchon_basis_radial(train_X_rgb, centers_rgb_local)
        Phi_rgb_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb_local)
        Phi_lab_tr, P_lab = duchon_basis_radial(lab_tr, centers_lab)
        Phi_lab_te, _ = duchon_basis_radial(lab_te, centers_lab)
        return _additive_fit_predict(
            [Phi_rgb_tr, Phi_lab_tr], [P_rgb, P_lab], train_Z,
            [Phi_rgb_te, Phi_lab_te],
        )

    if spec_name == "L_const_mean":
        # Predict training mean — sanity baseline. R² on held-out should be
        # very near 0, slightly negative on average (mean prediction has zero
        # explanatory power on truly held-out data).
        mean_pred = train_Z.mean(axis=0, keepdims=True)
        train_pred = np.repeat(mean_pred, train_Z.shape[0], axis=0)
        test_pred = np.repeat(mean_pred, test_X_rgb.shape[0], axis=0)
        return train_pred, test_pred

    if spec_name == "L_poly_oklab":
        Phi_tr = _poly_features_degree2(train_X_oklab)
        Phi_te = _poly_features_degree2(test_X_oklab)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_poly_lch":
        # Lch: lift h to (cos, sin) so polynomial features see angular structure.
        h_tr = train_X_lch[:, 2]; h_te = test_X_lch[:, 2]
        feat_tr = np.stack([
            train_X_lch[:, 0] / 100.0, train_X_lch[:, 1] / 100.0,
            np.cos(2*np.pi*h_tr), np.sin(2*np.pi*h_tr),
        ], axis=1)
        feat_te = np.stack([
            test_X_lch[:, 0] / 100.0, test_X_lch[:, 1] / 100.0,
            np.cos(2*np.pi*h_te), np.sin(2*np.pi*h_te),
        ], axis=1)
        Phi_tr = _poly_features_degree2(feat_tr)
        Phi_te = _poly_features_degree2(feat_te)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_joint_oklab_with_h":
        # 3D Oklab Duchon + cyclic hue (perceptual analogue of joint_rgb_with_hue)
        lo, hi = train_X_oklab.min(0), train_X_oklab.max(0)
        ax = [np.linspace(lo[d], hi[d], 5) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_ok_tr, P_ok = duchon_basis_radial(train_X_oklab, centers)
        Phi_ok_te, _ = duchon_basis_radial(test_X_oklab, centers)
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(train_X_lch[:, 2], n_basis=12)
        Phi_h_te, _ = bspline_1d_cyclic_basis(test_X_lch[:, 2], n_basis=12)
        return _additive_fit_predict(
            [Phi_ok_tr, Phi_h_tr], [P_ok, P_h], train_Z,
            [Phi_ok_te, Phi_h_te],
        )

    if spec_name == "L_chroma_lum_2d":
        # 2D Duchon on (L, C) only — no hue. Tests the "achromatic axis +
        # how saturated the color is" prediction. If this scores nearly as
        # well as the full hue-aware specs, color identity at L40 is mostly
        # encoded by lightness and chroma, not hue.
        feat_tr = np.stack([train_X_lch[:, 0] / 100.0,
                              train_X_lch[:, 1] / 100.0], axis=1)
        feat_te = np.stack([test_X_lch[:, 0] / 100.0,
                              test_X_lch[:, 1] / 100.0], axis=1)
        ax = [np.linspace(0, 1, 6), np.linspace(0, 1.5, 6)]
        L_g, C_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), C_g.flatten()], axis=1)
        Phi_tr, P = duchon_basis_radial(feat_tr, centers)
        Phi_te, _ = duchon_basis_radial(feat_te, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_hue_polyharmonic":
        # Cyclic hue B-spline + B-splines on L and C — gives hue full smooth
        # freedom while keeping L, C as separate 1D smooths.
        Phi_h_tr, P_h = bspline_1d_cyclic_basis(train_X_lch[:, 2], n_basis=16)
        Phi_h_te, _ = bspline_1d_cyclic_basis(test_X_lch[:, 2], n_basis=16)
        Phi_L_tr, P_L = bspline_1d_basis(train_X_lch[:, 0] / 100.0, n_basis=10)
        Phi_L_te, _ = bspline_1d_basis(test_X_lch[:, 0] / 100.0, n_basis=10)
        Cmx = max(float(train_X_lch[:, 1].max()), 1.0)
        Phi_C_tr, P_C = bspline_1d_basis(train_X_lch[:, 1] / Cmx, n_basis=10)
        Phi_C_te, _ = bspline_1d_basis(test_X_lch[:, 1] / Cmx, n_basis=10)
        return _additive_fit_predict(
            [Phi_h_tr, Phi_L_tr, Phi_C_tr],
            [P_h, P_L, P_C], train_Z,
            [Phi_h_te, Phi_L_te, Phi_C_te],
        )

    if spec_name == "N_knn_rgb_k30":
        return _knn_predict(train_X_rgb, train_Z, test_X_rgb, k=30)

    if spec_name == "N_knn_lab_k30":
        return _knn_predict(train_X_lab, train_Z, test_X_lab, k=30)

    if spec_name == "N_knn_rgb_k5":
        return _knn_predict(train_X_rgb, train_Z, test_X_rgb, k=5)

    if spec_name == "N_knn_rgb_k20":
        return _knn_predict(train_X_rgb, train_Z, test_X_rgb, k=20)

    if spec_name == "N_knn_lab_k5":
        return _knn_predict(train_X_lab, train_Z, test_X_lab, k=5)

    if spec_name == "N_knn_lab_k20":
        return _knn_predict(train_X_lab, train_Z, test_X_lab, k=20)

    if spec_name == "N_knn_oklab_k10":
        return _knn_predict(train_X_oklab, train_Z, test_X_oklab, k=10)

    if spec_name == "N_knn_rgb_k10":
        return _knn_predict(train_X_rgb, train_Z, test_X_rgb, k=10)

    if spec_name == "N_knn_hsv_k10":
        return _knn_predict(train_X_hsv, train_Z, test_X_hsv, k=10)

    if spec_name == "N_knn_lab_k10":
        return _knn_predict(train_X_lab, train_Z, test_X_lab, k=10)

    if spec_name == "L_lin_hsv":
        Phi_tr = np.concatenate([train_X_hsv, np.ones((train_X_hsv.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_hsv, np.ones((test_X_hsv.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_add_rgb":
        # f_R(R) + f_G(G) + f_B(B) via gamfit's purpose-built additive REML.
        designs_tr, designs_te, penalties = [], [], []
        for ax in range(3):
            B_tr, P_ax = bspline_1d_basis(train_X_rgb[:, ax], n_basis=10)
            B_te, _ = bspline_1d_basis(test_X_rgb[:, ax], n_basis=10)
            designs_tr.append(B_tr); designs_te.append(B_te); penalties.append(P_ax)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name == "L_add_hsv":
        # g_hue(cos h, sin h) + g_s(s) + g_v(v) via gamfit additive REML.
        # 2D radial Duchon over the (cos h, sin h) plane with 12 centers ON
        # the unit circle (the natural support).
        angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        centers_hue = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        Bh_tr, Ph = duchon_basis_radial(train_X_hsv[:, :2], centers_hue)
        Bh_te, _ = duchon_basis_radial(test_X_hsv[:, :2], centers_hue)
        Bs_tr, Ps = bspline_1d_basis(train_X_hsv[:, 2], n_basis=10)
        Bs_te, _ = bspline_1d_basis(test_X_hsv[:, 2], n_basis=10)
        Bv_tr, Pv = bspline_1d_basis(train_X_hsv[:, 3], n_basis=10)
        Bv_te, _ = bspline_1d_basis(test_X_hsv[:, 3], n_basis=10)
        return _additive_fit_predict(
            [Bh_tr, Bs_tr, Bv_tr], [Ph, Ps, Pv], train_Z,
            [Bh_te, Bs_te, Bv_te],
        )

    if spec_name == "L_joint_rgb":
        Phi_tr, P = duchon_basis_radial(train_X_rgb, centers_rgb)
        Phi_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_hsv":
        # 4D coords (cos h, sin h, sat, val). Centers placed near the natural
        # support: (cos 2πθ_i, sin 2πθ_i, s_j, v_k) for a grid of angles
        # crossed with sat + val. We add small per-center jitter so the
        # cos²+sin²=1 manifold constraint doesn't make the polynomial null
        # space rank-deficient — gamfit's `_thin_plate_penalty` requires
        # centers to span the polynomial block (rank d+1 = 5 in 4D).
        n_hue, n_sv = 8, 4                                    # 8 × 4 × 4 = 128 centers
        thetas = np.linspace(0.0, 2 * np.pi, n_hue, endpoint=False)
        sats = np.linspace(0.0, 1.0, n_sv)
        vals = np.linspace(0.0, 1.0, n_sv)
        rng_c = np.random.default_rng(42)
        centers_hsv = []
        for th in thetas:
            for s in sats:
                for v in vals:
                    centers_hsv.append([np.cos(th), np.sin(th), s, v])
        centers_hsv = np.array(centers_hsv) + 0.02 * rng_c.standard_normal((len(thetas) * n_sv * n_sv, 4))
        Phi_tr, P = duchon_basis_radial(train_X_hsv, centers_hsv)
        Phi_te, _ = duchon_basis_radial(test_X_hsv, centers_hsv)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_rgb_with_hue":
        # Multi-smooth single GAM: f(R,G,B) + g(cos h, sin h)
        # Routed through gamfit's additive REML (single shared λ across smooths).
        Phi_rgb_tr, P_rgb = duchon_basis_radial(train_X_rgb, centers_rgb)
        Phi_rgb_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb)
        angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        centers_hue = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        Phi_hue_tr, P_hue = duchon_basis_radial(train_X_hsv[:, :2], centers_hue)
        Phi_hue_te, _ = duchon_basis_radial(test_X_hsv[:, :2], centers_hue)
        return _additive_fit_predict(
            [Phi_rgb_tr, Phi_hue_tr], [P_rgb, P_hue], train_Z,
            [Phi_rgb_te, Phi_hue_te],
        )

    # =====================================================================
    # MANIFOLD smooths — these use the public gamfit Duchon (with proper
    # periodic_per_axis support) and gamfit.sphere_basis, NOT the local
    # `duchon_basis_radial` helper (which is non-periodic for d ≥ 2).
    # =====================================================================

    # Periodic manifolds via EUCLIDEAN EMBEDDING: each topological manifold
    # is embedded into ℝᵏ as a constrained submanifold; a regular Euclidean
    # Duchon over the embedding IS a smooth function on the manifold.
    #
    #   cylinder S¹×ℝ → ℝ³  :  (cos 2πh, sin 2πh, v)
    #   torus S¹×S¹  → ℝ⁴  :  (cos 2πh, sin 2πh, cos 2πq, sin 2πq)
    #   sphere S²    → ℝ³  :  (sin lat cos lon, sin lat sin lon, cos lat)
    #
    # Centers are placed ON the manifold (then live in the embedding too),
    # so the kernel only "sees" distances along the manifold's intrinsic
    # geometry to a good approximation.

    def _hue_from_X_hsv(X_hsv4: np.ndarray) -> np.ndarray:
        return (np.arctan2(X_hsv4[:, 1], X_hsv4[:, 0]) / (2 * np.pi)) % 1.0

    # NOTE: gamfit 0.1.109 exposes mixed-periodicity Duchon but the periodic
    # path is locked at p=1, s=0 — so 2(p+s)=2 fails the d≥2 constraint
    # 2(p+s) > d. We fall back to the cos/sin-embedded Euclidean Duchon
    # for cylinder/torus until gamfit's periodic path supports higher m.

    if spec_name == "M_cyl_hue_val":
        h_tr = _hue_from_X_hsv(train_X_hsv); v_tr = train_X_hsv[:, 3]
        h_te = _hue_from_X_hsv(test_X_hsv);  v_te = test_X_hsv[:, 3]
        emb_tr = np.stack([np.cos(2*np.pi*h_tr), np.sin(2*np.pi*h_tr), v_tr], axis=1)
        emb_te = np.stack([np.cos(2*np.pi*h_te), np.sin(2*np.pi*h_te), v_te], axis=1)
        hs = np.linspace(0, 1, 8, endpoint=False); vs = np.linspace(0, 1, 4)
        centers = np.array([
            [np.cos(2*np.pi*h), np.sin(2*np.pi*h), v] for h in hs for v in vs
        ])
        Phi_tr, P = duchon_basis_radial(emb_tr, centers)
        Phi_te, _ = duchon_basis_radial(emb_te, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    def _periodic_2d_centers(n_a: int = 12, n_b: int = 6):
        a = np.linspace(0, 1, n_a, endpoint=False)
        b = np.linspace(0, 1, n_b, endpoint=False)
        return np.array([[ai, bi] for ai in a for bi in b])

    def _torus_embed(h, q):
        return np.stack([np.cos(2*np.pi*h), np.sin(2*np.pi*h),
                          np.cos(2*np.pi*q), np.sin(2*np.pi*q)], axis=1)

    def _torus_centers(n_h: int = 8, n_q: int = 4, jitter: float = 0.03, seed: int = 42):
        rng = np.random.default_rng(seed)
        c = []
        for h in np.linspace(0, 1, n_h, endpoint=False):
            for q in np.linspace(0, 1, n_q, endpoint=False):
                c.append([np.cos(2*np.pi*h), np.sin(2*np.pi*h),
                          np.cos(2*np.pi*q), np.sin(2*np.pi*q)])
        return np.array(c) + jitter * rng.standard_normal((n_h * n_q, 4))

    if spec_name == "M_torus_hue_sat":
        emb_tr = _torus_embed(_hue_from_X_hsv(train_X_hsv), train_X_hsv[:, 2])
        emb_te = _torus_embed(_hue_from_X_hsv(test_X_hsv), test_X_hsv[:, 2])
        centers = _torus_centers()
        Phi_tr, P = duchon_basis_radial(emb_tr, centers)
        Phi_te, _ = duchon_basis_radial(emb_te, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "M_torus_hue_val":
        emb_tr = _torus_embed(_hue_from_X_hsv(train_X_hsv), train_X_hsv[:, 3])
        emb_te = _torus_embed(_hue_from_X_hsv(test_X_hsv), test_X_hsv[:, 3])
        centers = _torus_centers()
        Phi_tr, P = duchon_basis_radial(emb_tr, centers)
        Phi_te, _ = duchon_basis_radial(emb_te, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    def _to_sphere_latlon(X_hsv4):
        """Runge sphere: latitude=value (deg, in [-90, 90]), longitude=hue
        (deg, in [0, 360]). Returns shape (N, 2) in DEGREES, as
        gamfit.sphere_basis expects when radians=False."""
        hue = _hue_from_X_hsv(X_hsv4)
        v = X_hsv4[:, 3]
        lat = 90.0 * (2.0 * v - 1.0)
        lon = 360.0 * hue
        return np.stack([lat, lon], axis=1)

    if spec_name == "M_sphere_hueval":
        # Proper Sobolev S² smooth via gamfit's top-level sphere_basis API
        # (exposed in 0.1.109+). Lon = hue, lat = value.
        import gamfit
        sp_tr = _to_sphere_latlon(train_X_hsv)
        sp_te = _to_sphere_latlon(test_X_hsv)
        Phi_tr_t, P_t = gamfit.sphere_basis(
            sp_tr, n_centers=48, penalty_order=2, kernel="sobolev", radians=False,
        )
        Phi_te_t, _ = gamfit.sphere_basis(
            sp_te, n_centers=48, penalty_order=2, kernel="sobolev", radians=False,
        )
        Phi_tr = np.asarray(Phi_tr_t)
        Phi_te = np.asarray(Phi_te_t)
        P = np.asarray(P_t)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "M_sphere_plus_chroma":
        # S² Sobolev sphere + 1D B-spline on chroma (= sat · val)
        import gamfit
        sp_tr = _to_sphere_latlon(train_X_hsv)
        sp_te = _to_sphere_latlon(test_X_hsv)
        Phi_sp_tr_t, P_sp_t = gamfit.sphere_basis(
            sp_tr, n_centers=48, penalty_order=2, kernel="sobolev", radians=False,
        )
        Phi_sp_te_t, _ = gamfit.sphere_basis(
            sp_te, n_centers=48, penalty_order=2, kernel="sobolev", radians=False,
        )
        train_chroma = train_X_hsv[:, 2] * train_X_hsv[:, 3]
        test_chroma = test_X_hsv[:, 2] * test_X_hsv[:, 3]
        Phi_ch_tr, P_ch = bspline_1d_basis(train_chroma, n_basis=10)
        Phi_ch_te, _ = bspline_1d_basis(test_chroma, n_basis=10)
        return _additive_fit_predict(
            [np.asarray(Phi_sp_tr_t), Phi_ch_tr],
            [np.asarray(P_sp_t), P_ch], train_Z,
            [np.asarray(Phi_sp_te_t), Phi_ch_te],
        )

    if spec_name == "M_hsv_cone":
        # HSV-cone embedding: pure black/white collapse to the axis; sat
        # gives radius. Coordinates: (s·v·cos h, s·v·sin h, v). 3D Euclidean
        # Duchon — natural for the HSV color cone.
        def to_cone(X_hsv4):
            hue = (np.arctan2(X_hsv4[:, 1], X_hsv4[:, 0]) / (2 * np.pi))   # [-π, π]
            s, v = X_hsv4[:, 2], X_hsv4[:, 3]
            r = s * v
            return np.stack([r * np.cos(2 * np.pi * hue),
                             r * np.sin(2 * np.pi * hue), v], axis=1)
        train_co = to_cone(train_X_hsv)
        test_co = to_cone(test_X_hsv)
        # Place centers on a small cone-aligned set: 4 hue angles × 3 radii × 3 values
        ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        rad = np.linspace(0.1, 1.0, 3)
        val = np.linspace(0.1, 1.0, 3)
        centers = []
        for v in val:
            for r in rad:
                for a in ang:
                    centers.append([r * np.cos(a), r * np.sin(a), v])
        # plus a few centers on the axis (the achromatic line)
        for v in val: centers.append([0.0, 0.0, v])
        centers = np.array(centers)
        Phi_tr, P = duchon_basis_radial(train_co, centers)
        Phi_te, _ = duchon_basis_radial(test_co, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name in ("U_1d", "U_2d", "U_3d", "U_4d", "U_5d", "U_6d", "U_8d"):
        d = int(spec_name[2])
        # No GT axes used. Fit joint (T_train, B) by alternation on train Z.
        fit = fit_unsupervised_manifold(train_Z, d, cfg, n_iters=12, verbose=False)
        Phi_tr, _ = duchon_basis_radial(fit["T"], fit["centers"])
        Z_tr_pred = Phi_tr @ fit["B"]
        _, Z_te_pred = predict_unsupervised(test_Z, fit, d)
        return Z_tr_pred, Z_te_pred

    # ----- NEW unsupervised approaches -----

    if spec_name.startswith("U_pca_add_"):
        # Additive 1D smooths on each of the top-k PCs.
        # Tests: are the PCs independently informative, or do they interact?
        k = int(spec_name[len("U_pca_add_"):].rstrip("d"))
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:k]
        T_tr = Zc @ proj.T
        test_Zc = test_Z - train_Z.mean(0, keepdims=True)
        T_te = test_Zc @ proj.T
        # Normalize each PC to [0,1] for the B-spline basis
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_n = (T_tr - t_min) / (t_max - t_min + 1e-9)
        T_te_n = (T_te - t_min) / (t_max - t_min + 1e-9)
        T_te_n = np.clip(T_te_n, 0.0, 1.0)
        # Build list of (Phi_tr, Phi_te, P) per PC
        designs_tr, designs_te, penalties = [], [], []
        for i in range(k):
            B_tr, P_i = bspline_1d_basis(T_tr_n[:, i], n_basis=10)
            B_te, _ = bspline_1d_basis(T_te_n[:, i], n_basis=10)
            designs_tr.append(B_tr); designs_te.append(B_te); penalties.append(P_i)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name.startswith("U_pca_pairs_"):
        # 2D Duchon smooths on consecutive PCA pairs (PC1-PC2, PC3-PC4, …)
        # Additively combined. Tests pairwise interaction structure.
        k = int(spec_name[len("U_pca_pairs_"):].rstrip("d"))
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:k]
        T_tr = Zc @ proj.T
        test_Zc = test_Z - train_Z.mean(0, keepdims=True)
        T_te = test_Zc @ proj.T
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_n = (T_tr - t_min) / (t_max - t_min + 1e-9)
        T_te_n = (T_te - t_min) / (t_max - t_min + 1e-9)
        T_te_n = np.clip(T_te_n, 0.0, 1.0)
        designs_tr, designs_te, penalties = [], [], []
        for pair_i in range(0, k, 2):
            j_end = min(pair_i + 2, k)
            pair_tr = T_tr_n[:, pair_i:j_end]
            pair_te = T_te_n[:, pair_i:j_end]
            if pair_tr.shape[1] == 1:
                B_tr, P_p = bspline_1d_basis(pair_tr[:, 0], n_basis=10)
                B_te, _ = bspline_1d_basis(pair_te[:, 0], n_basis=10)
            else:
                ax_pair = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]
                xg, yg = np.meshgrid(*ax_pair, indexing="ij")
                centers_pair = np.stack([xg.flatten(), yg.flatten()], axis=1)
                B_tr, P_p = duchon_basis_radial(pair_tr, centers_pair)
                B_te, _ = duchon_basis_radial(pair_te, centers_pair)
            designs_tr.append(B_tr); designs_te.append(B_te); penalties.append(P_p)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name == "U_3d_pca_init":
        # Init U_3d's alternation with PCA-3d coords. Should converge to a
        # higher-R² local optimum than the default deterministic init.
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        T_pca = Zc @ Vt.T[:, :3]
        t_min = T_pca.min(0); t_max = T_pca.max(0)
        T0 = np.clip((T_pca - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
        fit = fit_unsupervised_manifold(train_Z, 3, cfg, n_iters=15,
                                         verbose=False, init_T=T0)
        Phi_tr, _ = duchon_basis_radial(fit["T"], fit["centers"])
        Z_tr_pred = Phi_tr @ fit["B"]
        _, Z_te_pred = predict_unsupervised(test_Z, fit, 3)
        return Z_tr_pred, Z_te_pred

    if spec_name.startswith("U_nmf_"):
        # NMF on shifted-positive train_Z. Test: project via least-squares
        # then reconstruct via the fitted W. NMF gives "parts-based"
        # decomposition (additive components, all positive). Useful when
        # the structure is genuinely additive non-negative.
        k = int(spec_name[len("U_nmf_"):].rstrip("d"))
        # Shift train data to be non-negative
        shift = train_Z.min(axis=0, keepdims=True) - 1e-3
        train_pos = train_Z - shift
        test_pos = test_Z - shift
        from sklearn.decomposition import NMF
        try:
            nmf = NMF(n_components=k, init="random", random_state=0,
                       max_iter=300, tol=1e-4)
            H_tr = nmf.fit_transform(train_pos)            # (N, k)
            W = nmf.components_                            # (k, D)
            train_pred = H_tr @ W + shift
            # Project test via least squares: argmin ||test_pos - H W||²
            # H_te = test_pos @ W^T @ (W W^T)^-1, clipped to nonneg
            WWt = W @ W.T
            H_te = test_pos @ W.T @ np.linalg.pinv(WWt)
            H_te = np.maximum(H_te, 0)
            test_pred = H_te @ W + shift
            return train_pred, test_pred
        except Exception:
            # NMF can fail to converge; fall back to PCA-k
            Zc = train_Z - train_Z.mean(0, keepdims=True)
            Vt = _pca_Vt(Zc)
            proj = Vt[:k]
            train_pred = (Zc @ proj.T) @ proj + train_Z.mean(0, keepdims=True)
            test_Zc = test_Z - train_Z.mean(0, keepdims=True)
            test_pred = (test_Zc @ proj.T) @ proj + train_Z.mean(0, keepdims=True)
            return train_pred, test_pred

    if spec_name == "U_centroid_kde_smooth_3d":
        # Predict each test residual as a weighted average of training
        # residuals, with weights from a Gaussian kernel in residual space
        # (kernel density smoothing). Distance-decayed analogue of k-NN.
        sigma_sq = float(np.mean(np.var(train_Z, axis=0)))    # ~ trace(cov)/D
        d2_tr = np.sum((train_Z[:, None, :] - train_Z[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2_tr, np.inf)
        # Use per-row exponential weights with scale ~ median of nearest 20
        med = np.partition(d2_tr, 20, axis=1)[:, :20].mean(axis=1).mean()
        w_tr = np.exp(-d2_tr / (2 * med))
        w_tr = w_tr / (w_tr.sum(axis=1, keepdims=True) + 1e-9)
        train_pred = w_tr @ train_Z
        d2_te = np.sum((test_Z[:, None, :] - train_Z[None, :, :]) ** 2, axis=2)
        w_te = np.exp(-d2_te / (2 * med))
        w_te = w_te / (w_te.sum(axis=1, keepdims=True) + 1e-9)
        test_pred = w_te @ train_Z
        return train_pred, test_pred

    # ALL-DUCHON unsupervised GAM zoo on PCA latents (no B-splines anywhere)

    def _pca_latent(train_Z, test_Z, k):
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:k]
        T_tr = Zc @ proj.T
        T_te = (test_Z - train_Z.mean(0, keepdims=True)) @ proj.T
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_n = np.clip((T_tr - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
        T_te_n = np.clip((T_te - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
        return T_tr_n, T_te_n

    def _joint_duchon_on_latent(T_tr_n, T_te_n, n_per_axis: int = 4):
        """Joint Duchon on the k-D PCA latent. Centers on a uniform lattice."""
        k = T_tr_n.shape[1]
        ax = [np.linspace(0, 1, n_per_axis) for _ in range(k)]
        grids = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([g.flatten() for g in grids], axis=1)
        Phi_tr, P = duchon_basis_radial(T_tr_n, centers)
        Phi_te, _ = duchon_basis_radial(T_te_n, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name in ("U_pca3_duchon_joint", "U_pca4_duchon_joint", "U_pca6_duchon_joint"):
        k = int(spec_name[len("U_pca"):].split("_")[0])
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, k)
        # Adapt n_per_axis so total centers stay reasonable
        n_per = {3: 5, 4: 4, 6: 3}[k]
        return _joint_duchon_on_latent(T_tr_n, T_te_n, n_per_axis=n_per)

    if spec_name.startswith("U_pca") and spec_name.endswith("_duchon_add1d"):
        k = int(spec_name[len("U_pca"):].split("_")[0])
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, k)
        # 1D Duchon per PC additively
        designs_tr, designs_te, penalties = [], [], []
        centers_1d = np.linspace(0, 1, 12).reshape(-1, 1)
        for i in range(k):
            pts_tr = T_tr_n[:, i:i+1]
            pts_te = T_te_n[:, i:i+1]
            Phi_tr, P = duchon_basis_radial(pts_tr, centers_1d)
            Phi_te, _ = duchon_basis_radial(pts_te, centers_1d)
            designs_tr.append(Phi_tr); designs_te.append(Phi_te); penalties.append(P)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name == "U_pca16_duchon_pairs":
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, 16)
        designs_tr, designs_te, penalties = [], [], []
        ax_pair = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]
        xg, yg = np.meshgrid(*ax_pair, indexing="ij")
        centers_pair = np.stack([xg.flatten(), yg.flatten()], axis=1)
        for i in range(0, 16, 2):
            pair_tr = T_tr_n[:, i:i+2]; pair_te = T_te_n[:, i:i+2]
            Phi_tr, P = duchon_basis_radial(pair_tr, centers_pair)
            Phi_te, _ = duchon_basis_radial(pair_te, centers_pair)
            designs_tr.append(Phi_tr); designs_te.append(Phi_te); penalties.append(P)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name in ("U_pca6_duchon_triples", "U_pca12_duchon_triples"):
        k = int(spec_name[len("U_pca"):].split("_")[0])
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, k)
        designs_tr, designs_te, penalties = [], [], []
        ax = [np.linspace(0, 1, 4) for _ in range(3)]
        xg, yg, zg = np.meshgrid(*ax, indexing="ij")
        centers_tri = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=1)
        for i in range(0, k, 3):
            tri_tr = T_tr_n[:, i:i+3]; tri_te = T_te_n[:, i:i+3]
            Phi_tr, P = duchon_basis_radial(tri_tr, centers_tri)
            Phi_te, _ = duchon_basis_radial(tri_te, centers_tri)
            designs_tr.append(Phi_tr); designs_te.append(Phi_te); penalties.append(P)
        return _additive_fit_predict(designs_tr, penalties, train_Z, designs_te)

    if spec_name in ("U_pca8_duchon_m3", "U_pca16_duchon_m3"):
        # Force gamfit to use m=3 (smoother polyharmonic kernel). Centers
        # placed via k-means on the latent (bounded count, not full lattice
        # which would OOM at d=16 with 2^16 = 65536 grid points).
        k = 8 if spec_name == "U_pca8_duchon_m3" else 16
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, k)
        from numpy.random import default_rng
        K = 80
        rng = default_rng(0)
        idx = rng.choice(T_tr_n.shape[0], min(K, T_tr_n.shape[0]), replace=False)
        centers = T_tr_n[idx].copy()
        for _ in range(10):
            d2 = np.sum((T_tr_n[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            a = d2.argmin(axis=1)
            for ki in range(len(centers)):
                m = (a == ki)
                if m.any():
                    centers[ki] = T_tr_n[m].mean(0)
        import gamfit
        try:
            Phi_tr = np.asarray(gamfit.duchon_basis(T_tr_n, centers, m=3, periodic_per_axis=(False,)*k))
            Phi_te = np.asarray(gamfit.duchon_basis(T_te_n, centers, m=3, periodic_per_axis=(False,)*k))
            P = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=3, periodic_per_axis=(False,)*k))
            B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
            return Phi_tr @ B, Phi_te @ B
        except Exception:
            Phi_tr, P = duchon_basis_radial(T_tr_n, centers)
            Phi_te, _ = duchon_basis_radial(T_te_n, centers)
            B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
            return Phi_tr @ B, Phi_te @ B

    if spec_name == "U_pca8_duchon_finer_centers":
        T_tr_n, T_te_n = _pca_latent(train_Z, test_Z, 8)
        # Place 200 centers via k-means on the latent for finer adaptation
        from numpy.random import default_rng
        rng = default_rng(0)
        K = 200
        idx = rng.choice(T_tr_n.shape[0], min(K, T_tr_n.shape[0]), replace=False)
        centers = T_tr_n[idx].copy()
        for _ in range(10):
            d2 = np.sum((T_tr_n[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            a = d2.argmin(axis=1)
            for ki in range(len(centers)):
                m = (a == ki)
                if m.any():
                    centers[ki] = T_tr_n[m].mean(0)
        Phi_tr, P = duchon_basis_radial(T_tr_n, centers)
        Phi_te, _ = duchon_basis_radial(T_te_n, centers)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "U_pca_centered_8d_smooth":
        # PCA-8d latent + smooth fit in PC-latent space. Same as U_pca8_smooth
        # but using centers placed via k-means on the PC-latent for finer
        # adaptation to data density.
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:8]
        T_tr = Zc @ proj.T
        test_Zc = test_Z - train_Z.mean(0, keepdims=True)
        T_te = test_Zc @ proj.T
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_n = (T_tr - t_min) / (t_max - t_min + 1e-9)
        T_te_n = np.clip((T_te - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
        # k-means centers in the normalized latent
        rng = np.random.default_rng(0)
        K = 32
        idx = rng.choice(T_tr_n.shape[0], K, replace=False)
        centers = T_tr_n[idx].copy()
        for _ in range(15):
            d2 = np.sum((T_tr_n[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            a = d2.argmin(axis=1)
            for ki in range(K):
                m = (a == ki)
                if m.any():
                    centers[ki] = T_tr_n[m].mean(0)
        try:
            Phi_tr, P = duchon_basis_radial(T_tr_n, centers)
            B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
            Phi_te, _ = duchon_basis_radial(T_te_n, centers)
            return Phi_tr @ B, Phi_te @ B
        except Exception:
            train_pred = T_tr @ proj + train_Z.mean(0, keepdims=True)
            test_pred = T_te @ proj + train_Z.mean(0, keepdims=True)
            return train_pred, test_pred

    if spec_name == "U_pca3_tensor":
        # PCA-3 latent + tensor-product B-spline (different smooth family
        # than Duchon's radial kernel).
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:3]
        T_tr = Zc @ proj.T
        test_Zc = test_Z - train_Z.mean(0, keepdims=True)
        T_te = test_Zc @ proj.T
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_n = (T_tr - t_min) / (t_max - t_min + 1e-9)
        T_te_n = np.clip((T_te - t_min) / (t_max - t_min + 1e-9), 0.0, 1.0)
        def tprod(X, n_per_axis: int = 6):
            cols = []
            for ax in range(3):
                Bk, _ = bspline_1d_basis(X[:, ax], n_basis=n_per_axis)
                cols.append(Bk)
            out = cols[0][:, :, None, None] * cols[1][:, None, :, None] * cols[2][:, None, None, :]
            return out.reshape(X.shape[0], -1)
        Phi_tr = tprod(T_tr_n)
        Phi_te = tprod(T_te_n)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name in ("U_pca8_smooth", "U_pca16_smooth"):
        k = 8 if spec_name == "U_pca8_smooth" else 16
        # Project to top-k PCs to get a k-d "latent" T, then fit a
        # nonlinear Duchon smooth from T → residual. Held-out: project test
        # residual to k PCs, predict via smooth. This separates the
        # subspace discovery (linear, PCA) from the nonlinear surface fit.
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:k]
        T_tr = Zc @ proj.T
        # Normalize to [0,1]^k for Duchon centers
        t_min = T_tr.min(0); t_max = T_tr.max(0)
        T_tr_norm = (T_tr - t_min) / (t_max - t_min + 1e-9)
        T_tr_norm = np.clip(T_tr_norm, 0.001, 0.999)
        # Centers: random subset of training T (10% capped at 60)
        n_centers = min(60, max(20, T_tr_norm.shape[0] // 10))
        rng = np.random.default_rng(0)
        center_idx = rng.choice(T_tr_norm.shape[0], n_centers, replace=False)
        centers = T_tr_norm[center_idx]
        try:
            Phi_tr, P = duchon_basis_radial(T_tr_norm, centers)
            B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
            test_Zc = test_Z - train_Z.mean(0, keepdims=True)
            T_te = test_Zc @ proj.T
            T_te_norm = (T_te - t_min) / (t_max - t_min + 1e-9)
            T_te_norm = np.clip(T_te_norm, 0.001, 0.999)
            Phi_te, _ = duchon_basis_radial(T_te_norm, centers)
            return Phi_tr @ B, Phi_te @ B
        except Exception as exc:
            # If high-d Duchon penalty rejects this, return linear PCA fallback
            train_pred = T_tr @ proj + train_Z.mean(0, keepdims=True)
            test_Zc = test_Z - train_Z.mean(0, keepdims=True)
            test_pred = (test_Zc @ proj.T) @ proj + train_Z.mean(0, keepdims=True)
            return train_pred, test_pred

    if spec_name.startswith("U_pca_"):
        k = int(spec_name[len("U_pca_"):].rstrip("d"))
        # Linear unsupervised: top-k PCs of train_Z; predict via reconstruction.
        # Held-out R² = how much of test_Z lies in the top-k-PC subspace
        # learned from training. This is the linear baseline U_kd alternation
        # should beat if it's actually finding nonlinear manifold structure.
        Zc = train_Z - train_Z.mean(0, keepdims=True)
        Vt = _pca_Vt(Zc)
        proj = Vt[:k]
        train_pred = (Zc @ proj.T) @ proj + train_Z.mean(0, keepdims=True)
        test_Zc = test_Z - train_Z.mean(0, keepdims=True)
        test_pred = (test_Zc @ proj.T) @ proj + train_Z.mean(0, keepdims=True)
        return train_pred, test_pred

    if spec_name.startswith("U_kmeans_"):
        K = int(spec_name[len("U_kmeans_"):])
        # k-means cluster-and-mean: train centroids, predict held-out via
        # nearest training cluster. If colors cluster by category (e.g.
        # "all the blues" together), this works well; if they spread
        # continuously, this is a stair-step approximation.
        # Simple Lloyd's: init by random training rows, 20 iterations.
        rng = np.random.default_rng(0)
        N_tr, D = train_Z.shape
        idx = rng.choice(N_tr, size=K, replace=False)
        C = train_Z[idx].copy()
        for _ in range(20):
            d2 = np.sum((train_Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
            assignments = d2.argmin(axis=1)
            for ki in range(K):
                m = (assignments == ki)
                if m.any():
                    C[ki] = train_Z[m].mean(0)
        # Assign training to nearest cluster
        d2_tr = np.sum((train_Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
        a_tr = d2_tr.argmin(axis=1)
        train_pred = C[a_tr]
        d2_te = np.sum((test_Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
        a_te = d2_te.argmin(axis=1)
        test_pred = C[a_te]
        return train_pred, test_pred

    if spec_name == "U_loop_1d":
        # 1D PERIODIC latent — discover a closed loop. Constrains the
        # color manifold to be a circle (color wheel hypothesis).
        # Init t uniformly on [0,1), refine with cyclic B-spline smooth.
        N_tr = train_Z.shape[0]
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 1, N_tr)
        for it in range(15):
            Phi_tr, P = bspline_1d_cyclic_basis(t, n_basis=12)
            B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
            # Re-project: grid search over [0, 1) periodic
            grid = np.linspace(0, 1, 400, endpoint=False)
            Phi_grid, _ = bspline_1d_cyclic_basis(grid, n_basis=12)
            preds = Phi_grid @ B
            # For each training point, find best grid t
            d_to_grid = np.sum((train_Z[:, None, :] - preds[None, :, :]) ** 2, axis=2)
            t_new = grid[d_to_grid.argmin(axis=1)]
            if np.allclose(t, t_new, atol=1e-4):
                break
            t = t_new
        # Final fit
        Phi_tr, P = bspline_1d_cyclic_basis(t, n_basis=12)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        train_pred = Phi_tr @ B
        # Predict test by projection onto the same loop
        grid = np.linspace(0, 1, 400, endpoint=False)
        Phi_grid, _ = bspline_1d_cyclic_basis(grid, n_basis=12)
        preds = Phi_grid @ B
        d_to_grid = np.sum((test_Z[:, None, :] - preds[None, :, :]) ** 2, axis=2)
        t_test = grid[d_to_grid.argmin(axis=1)]
        Phi_te, _ = bspline_1d_cyclic_basis(t_test, n_basis=12)
        test_pred = Phi_te @ B
        return train_pred, test_pred

    if spec_name == "U_3d_multistart":
        # U_3d with 5 random inits via different seeds — pick the one with
        # the lowest TRAINING MSE (proxy for best local optimum).
        best_fit = None
        best_train_mse = float("inf")
        for seed in range(5):
            # The existing fit_unsupervised_manifold uses a deterministic init;
            # we perturb by passing init_T via small random noise on a PCA init.
            rng = np.random.default_rng(seed)
            Zc = train_Z - train_Z.mean(0, keepdims=True)
            Vt = _pca_Vt(Zc)
            T_pca = (Zc @ Vt.T[:, :3])
            # Normalize to [0, 1] cube
            t_min = T_pca.min(0); t_max = T_pca.max(0)
            T0 = (T_pca - t_min) / (t_max - t_min + 1e-9)
            T0 = T0 + 0.02 * rng.standard_normal(T0.shape)
            T0 = np.clip(T0, 0.001, 0.999)
            fit = fit_unsupervised_manifold(
                train_Z, 3, cfg, n_iters=12, verbose=False, init_T=T0,
            )
            mse = fit["history"][-1]["train_mse"]
            if mse < best_train_mse:
                best_train_mse = mse
                best_fit = fit
        Phi_tr, _ = duchon_basis_radial(best_fit["T"], best_fit["centers"])
        Z_tr_pred = Phi_tr @ best_fit["B"]
        _, Z_te_pred = predict_unsupervised(test_Z, best_fit, 3)
        return Z_tr_pred, Z_te_pred

    raise ValueError(f"unknown spec: {spec_name}")


def cv_fit_one_spec(spec_name: str, X_rgb, X_hsv, Z, cfg: Config) -> dict:
    n_colors = X_rgb.shape[0]
    folds = kfold_color_indices(n_colors, cfg.n_folds)
    fold_r2_macro, fold_r2_per_pc = [], []
    for f_idx, (tr, te) in enumerate(folds):
        Z_tr_pred, Z_te_pred = fit_and_predict(
            spec_name, X_rgb[tr], X_hsv[tr], Z[tr],
            X_rgb[te], X_hsv[te], Z[te], cfg,
        )
        macro = r2_score(Z[te], Z_te_pred)
        per_pc = np.array([
            r2_score(Z[te][:, k:k+1], Z_te_pred[:, k:k+1])
            for k in range(Z.shape[1])
        ])
        fold_r2_macro.append(macro)
        fold_r2_per_pc.append(per_pc)
    return {
        "r2_macro_mean": float(np.mean(fold_r2_macro)),
        "r2_macro_std": float(np.std(fold_r2_macro)),
        "r2_per_pc_mean": [float(x) for x in np.mean(fold_r2_per_pc, axis=0)],
        "per_fold_r2_macro": [float(x) for x in fold_r2_macro],
    }


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    cfg = Config()
    use_external_harvest = bool(cfg.harvest_from)
    if not use_external_harvest:
        require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() and not use_external_harvest else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name}  layers={cfg.layers}  device={device}"
          + (f"  harvest_from={cfg.harvest_from}" if use_external_harvest else ""),
          flush=True)

    colors = load_xkcd_colors()
    n_c, n_t = len(colors), len(TEMPLATES)
    print(f"[data] {n_c} colors × {n_t} templates = {n_c * n_t} prompts", flush=True)

    rgb_per_color = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    hsv_per_color = rgb_to_hsv_arr(rgb_per_color * 255.0)            # h, s, v in [0,1]
    # 4D HSV-periodic coords: (cos 2πh, sin 2πh, sat, val)
    X_hsv = np.stack([
        np.cos(2 * np.pi * hsv_per_color[:, 0]),
        np.sin(2 * np.pi * hsv_per_color[:, 0]),
        hsv_per_color[:, 1],
        hsv_per_color[:, 2],
    ], axis=1)
    X_rgb = rgb_per_color
    color_axes = {
        "R": rgb_per_color[:, 0], "G": rgb_per_color[:, 1], "B": rgb_per_color[:, 2],
        "hue": hsv_per_color[:, 0], "sat": hsv_per_color[:, 1], "value": hsv_per_color[:, 2],
        "luminance": 0.299 * rgb_per_color[:, 0] + 0.587 * rgb_per_color[:, 1] + 0.114 * rgb_per_color[:, 2],
    }

    prompts, c_idx = [], []
    for ci, (name, _, _, _) in enumerate(colors):
        for tpl in TEMPLATES:
            prompts.append(tpl.format(x=name))
            c_idx.append(ci)
    c_idx = np.array(c_idx)

    if use_external_harvest:
        # Skip model loading entirely; load the (N, D) residual matrix that
        # color_geometry.py (or another harvester) already produced. We
        # require ``cfg.layers`` to be a single layer matching the harvest.
        if len(cfg.layers) != 1:
            raise ValueError(
                f"MSAE_HARVEST_FROM mode requires exactly one entry in "
                f"MSAE_LAYERS (the layer the harvest was produced for); "
                f"got layers={cfg.layers}"
            )
        L_target = cfg.layers[0]
        cache_path = Path(cfg.harvest_from)
        if cache_path.suffix == ".npz":
            ck = np.load(cache_path, allow_pickle=False)
            X_full_np = ck["X"]
            if "layer" in ck.files:
                cached_layer = int(ck["layer"])
                if cached_layer != L_target:
                    raise ValueError(
                        f"harvest cache layer {cached_layer} != MSAE_LAYERS={L_target}"
                    )
            n_cached_prompts = X_full_np.shape[0]
        elif cache_path.suffix == ".npy":
            X_full_np = np.load(cache_path)
            n_cached_prompts = X_full_np.shape[0]
        else:
            raise ValueError(
                f"MSAE_HARVEST_FROM must be .npy or .npz, got {cache_path.suffix}"
            )

        # Truncate to whole colors only.
        n_full_colors = n_cached_prompts // n_t
        if n_full_colors < cfg.n_folds + 1:
            raise ValueError(
                f"harvest only covers {n_full_colors} complete colors; "
                f"need at least n_folds+1={cfg.n_folds + 1} for color-grouped CV"
            )
        n_complete_rows = n_full_colors * n_t
        X_full_t = torch.from_numpy(X_full_np[:n_complete_rows]).float()
        print(f"[harvest_from] loaded {cache_path.name}: "
              f"{n_cached_prompts} rows -> using {n_full_colors} complete "
              f"colors × {n_t} templates = {n_complete_rows} rows  "
              f"D={X_full_t.shape[1]}", flush=True)

        # Truncate downstream arrays so they match the rows we kept.
        rgb_per_color = rgb_per_color[:n_full_colors]
        hsv_per_color = hsv_per_color[:n_full_colors]
        X_hsv = X_hsv[:n_full_colors]
        X_rgb = X_rgb[:n_full_colors]
        for k in list(color_axes.keys()):
            color_axes[k] = color_axes[k][:n_full_colors]
        colors = colors[:n_full_colors]
        c_idx = c_idx[:n_complete_rows]
        prompts = prompts[:n_complete_rows]
        n_c = n_full_colors
        layer_resids = {L_target: X_full_t}
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32
        print(f"[load] {cfg.model_name} dtype={dtype}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device).eval()
        blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
        n_layers = len(blocks)
        print(f"[load] hidden_layers={n_layers}", flush=True)
        bad = [L for L in cfg.layers if L < 0 or L >= n_layers]
        if bad:
            raise ValueError(f"layer indices {bad} out of range for model with {n_layers} layers")

        layer_resids = harvest(cfg, model, tok, blocks, prompts, device)
        del model; torch.cuda.empty_cache()

    if cfg.save_residuals and not use_external_harvest:
        torch.save({
            "layer_resids": layer_resids,
            "colors": colors, "c_idx": c_idx,
            "templates": TEMPLATES, "config": asdict(cfg),
        }, out_dir / "residuals.pt")
        print(f"[save] residuals → {out_dir/'residuals.pt'}", flush=True)

    per_layer_results = {}
    # Optional template subset (per-template analysis showed that ~6 templates
    # produce 3-4× stronger color signal than the rest; averaging across all
    # 28 dilutes the signal). MSAE_TEMPLATE_SUBSET="8,13,16,17,18,5" keeps
    # only those template-indexed rows for the per-color centroid.
    template_subset_mask: np.ndarray | None = None
    if cfg.template_subset.strip():
        keep_idx = sorted({int(x) for x in cfg.template_subset.split(",")})
        n_t_total = len(TEMPLATES)
        # Build a mask of length N=n_c*n_t selecting only the kept templates
        template_subset_mask = np.zeros(len(c_idx), dtype=bool)
        for ci in range(n_c):
            base = ci * n_t_total
            for ti in keep_idx:
                if 0 <= ti < n_t_total:
                    template_subset_mask[base + ti] = True
        print(f"[template_subset] keeping templates {keep_idx} ({len(keep_idx)}/"
              f"{n_t_total}) → {int(template_subset_mask.sum())} prompts",
              flush=True)
    for L in cfg.layers:
        X_full = layer_resids[L]                          # (N, D) fp32
        # Per-color average across (subset of) templates
        per_color = torch.zeros(n_c, X_full.shape[1])
        for ci in range(n_c):
            m = (c_idx == ci)
            if template_subset_mask is not None:
                m = m & template_subset_mask
            per_color[ci] = X_full[m].mean(0)
        # Per-dim normalize
        mu = per_color.mean(0, keepdim=True)
        sigma = per_color.std(0, keepdim=True).clamp(min=1e-6)
        Xn = ((per_color - mu) / sigma).numpy().astype(np.float64)
        # Top-K PCA — sklearn.decomposition.PCA via _pca_basis (one source
        # of truth for PCA in this repo). sklearn centers internally; Xn is
        # already per-feature-standardized so this is a single centering.
        from sklearn.decomposition import PCA as _SkPCA
        _pca = _SkPCA(n_components=cfg.n_pcs, svd_solver="full").fit(Xn)
        Vt = _pca.components_                                         # (K_pcs, D)
        Z = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T                   # (954, K_pcs)
        explained_var_ratio = _pca.explained_variance_ratio_
        print(f"\n=== L={L} ===  D={X_full.shape[1]}  top-{cfg.n_pcs}-PCs explain "
              f"{explained_var_ratio.sum():.2%}", flush=True)

        spec_results = {}
        print(f"  {'spec':24} {'R²_macro':>10} {'± std':>6}", flush=True)
        for spec_name in SPECS:
            try:
                r = cv_fit_one_spec(spec_name, X_rgb, X_hsv, Z, cfg)
                spec_results[spec_name] = r
                print(f"  {spec_name:24} {r['r2_macro_mean']:+10.3f} "
                      f"{r['r2_macro_std']:6.3f}", flush=True)
            except Exception as e:
                print(f"  {spec_name:24} FAILED: {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
                spec_results[spec_name] = {"error": str(e)}

        # Full-data unsupervised fits (no CV) — discover the latent t per
        # color for each intrinsic d. We use these to answer "what known
        # axis does the discovered manifold align with?" via Spearman.
        unsup_full = {}
        for d in (1, 2, 3, 4):
            fit = fit_unsupervised_manifold(Z, d, cfg, n_iters=15, verbose=False)
            T = fit["T"]                                          # (954, d)
            # Spearman of each latent axis vs each known axis
            spearmans = {}
            for ax_name, ax_vals in color_axes.items():
                per_latent_rhos = [float(_spearman_np(T[:, k], ax_vals)) for k in range(d)]
                spearmans[ax_name] = {
                    "per_latent_rho": per_latent_rhos,
                    "best_latent": int(np.argmax(np.abs(per_latent_rhos))),
                    "best_abs_rho": max(abs(r) for r in per_latent_rhos),
                }
            # For each latent axis, also report the BEST-aligned known axis
            best_per_latent = []
            for k in range(d):
                axis_rhos: dict[str, float] = {
                    ax: float(_spearman_np(T[:, k], color_axes[ax]))
                    for ax in color_axes
                }
                best_ax: str = max(axis_rhos.keys(), key=lambda a: abs(axis_rhos[a]))
                best_per_latent.append({"latent_idx": k,
                                          "best_axis": best_ax,
                                          "rho": axis_rhos[best_ax],
                                          "all_rhos": axis_rhos})
            unsup_full[f"d={d}"] = {
                "T": T.tolist(),
                "log_lambda": fit["log_lambda"],
                "n_iters": len(fit["history"]),
                "final_train_mse": fit["history"][-1]["train_mse"],
                "axis_to_latent_spearman": spearmans,
                "best_axis_per_latent": best_per_latent,
            }
            # Print summary
            print(f"  [unsup d={d}] log_lam={fit['log_lambda']:+.2f}  "
                  f"iters={len(fit['history'])}  best per latent: " +
                  ", ".join(f"{x['best_axis']}={x['rho']:+.2f}" for x in best_per_latent),
                  flush=True)

        per_layer_results[f"L{L}"] = {
            "explained_variance_ratio_topK": [float(x) for x in explained_var_ratio],
            "specs": spec_results,
            "unsupervised_full_data": unsup_full,
            "Vt_topK": Vt[:cfg.n_pcs].tolist(),     # principal directions
            "mu": mu.flatten().tolist(),
            "sigma": sigma.flatten().tolist(),
        }

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg),
        "templates": list(TEMPLATES),
        "color_axes_per_color_index": {k: v.tolist() for k, v in color_axes.items()},
        "per_layer": per_layer_results,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
