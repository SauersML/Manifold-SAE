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


def _duchon_kernel(r: np.ndarray, d: int) -> np.ndarray:
    """Polyharmonic m=2 kernels per dim. Used only for d ≥ 2 where gamfit
    doesn't expose the multi-D Duchon penalty in the torch surface.

      d=2: phi(r) = r^2 log r
      d=3: phi(r) = r
      d=4: phi(r) = log r
    Limits at r=0 are 0.
    """
    if d == 2:
        out = np.zeros_like(r)
        mask = r > 1e-12
        out[mask] = r[mask] ** 2 * np.log(r[mask])
        return out
    if d == 3:
        return r
    if d == 4:
        out = np.zeros_like(r)
        mask = r > 1e-12
        out[mask] = np.log(r[mask])
        return out
    raise ValueError(f"_duchon_kernel: d must be in {{2,3,4}}; got {d}")


def duchon_basis_radial(X: np.ndarray, centers: np.ndarray,
                          periodic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Radial Duchon m=2 design + matched function-norm penalty.

    d=1 path: delegates entirely to gamfit. Basis = `gt.duchon_basis_1d`
    (returns kernel block constrained orthogonal to the null space PLUS
    the null space columns); penalty = `_duchon_function_norm_penalty`.
    Supports `periodic=True` (gamfit's periodic Duchon — fixed in
    0.1.99+; pinned at 0.1.98 here so periodic crashes until lockfile
    is bumped).

    d ≥ 2 path: gamfit's torch surface doesn't expose a multi-D Duchon
    penalty, so we build the classical RBF construction: design =
    [kernel | null space], penalty = kernel Gram on centers. For radial
    basis functions this Gram IS the function-norm penalty (standard
    Wahba/Wood thin-plate-spline result).

    Returns (Phi, P). Shape varies by path; downstream consumers see them
    as a matched pair.
    """
    N, d = X.shape

    if d == 1:
        import gamfit.torch as gt
        from gamfit._api import _duchon_function_norm_penalty
        # Centers 1D (gamfit wants shape (K,))
        c_t = torch.from_numpy(np.ascontiguousarray(centers.ravel(), dtype=np.float64))
        t_t = torch.from_numpy(np.ascontiguousarray(X.ravel(), dtype=np.float64))
        with torch.no_grad():
            Phi_t = gt.duchon_basis_1d(t_t, c_t, m=2, periodic=periodic)
        Phi = Phi_t.detach().cpu().numpy()
        P = np.asarray(_duchon_function_norm_penalty(
            centers.ravel().astype(np.float64), m=2, periodic=periodic
        ))
        return Phi, P

    if periodic:
        raise ValueError("periodic Duchon currently only supported for d=1")

    K = centers.shape[0]
    r_nk = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    Phi_kernel = _duchon_kernel(r_nk, d)
    null = np.concatenate([np.ones((N, 1)), X], axis=1)                    # (N, d+1)
    Phi = np.concatenate([Phi_kernel, null], axis=1)
    r_cc = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    Pkk = _duchon_kernel(r_cc, d)
    P = np.zeros((K + d + 1, K + d + 1))
    P[:K, :K] = Pkk
    return Phi, P


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
def reml_fit(Phi: np.ndarray, Z: np.ndarray, P: np.ndarray,
              init_log_lambda: float = 0.0) -> tuple[np.ndarray, float]:
    """Fit coefficients B (K, R) for the multi-output smooth Z = Phi · B + ε
    with penalty λ · β' P β. The smoothing parameter λ is chosen by
    **gamfit's closed-form REML** (Rust core) — the principled criterion
    for Gaussian smooths, strictly better than the GCV/Newton hand-rolls.

    Inputs are numpy; conversion to torch + back is local. The penalty is
    symmetrized for safety (gamfit's REML solver requires symmetry).
    """
    import gamfit.torch as gt
    P_sym = 0.5 * (P + P.T)
    init_lam = float(math.exp(init_log_lambda)) if init_log_lambda is not None else None
    x_t = torch.from_numpy(np.ascontiguousarray(Phi, dtype=np.float64))
    y_t = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64))
    p_t = torch.from_numpy(np.ascontiguousarray(P_sym, dtype=np.float64))
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
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
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
        centers_per_axis = {1: 16, 2: 8, 3: 5, 4: 4}[d]      # 16 / 64 / 125 / 256 centers
    if grid_per_axis is None:
        grid_per_axis = {1: 200, 2: 40, 3: 20, 4: 9}[d]      # ~6.5k grid pts in 4D

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
    "L_add_rgb": ("additive_rgb",),
    "L_add_hsv": ("additive_hsv",),
    "L_joint_rgb": ("smooth_rgb",),
    "L_joint_hsv": ("smooth_hsv",),
    "L_joint_rgb_with_hue": ("smooth_rgb_plus_hue",),
    # UNSUPERVISED: latent parameterization, discovered by alternation
    "U_1d": ("unsup_1d",),
    "U_2d": ("unsup_2d",),
    "U_3d": ("unsup_3d",),
    "U_4d": ("unsup_4d",),
}


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

    if spec_name == "L_lin_rgb":
        Phi_tr = np.concatenate([train_X_rgb, np.ones((train_X_rgb.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_rgb, np.ones((test_X_rgb.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_lin_hsv":
        Phi_tr = np.concatenate([train_X_hsv, np.ones((train_X_hsv.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([test_X_hsv, np.ones((test_X_hsv.shape[0], 1))], axis=1)
        W = ridge_fit(Phi_tr, train_Z, alpha=1.0)
        return Phi_tr @ W, Phi_te @ W

    if spec_name == "L_add_rgb":
        # f_R(R) + f_G(G) + f_B(B) — 1D B-spline per axis, separate REML λ each
        # Build a block-diagonal Phi with one block per axis, single shared mean.
        n_basis = 10
        blocks_tr, blocks_te, blocks_P = [], [], []
        for ax in range(3):
            B_tr, P_ax = bspline_1d_basis(train_X_rgb[:, ax], n_basis=n_basis)
            B_te, _ = bspline_1d_basis(test_X_rgb[:, ax], n_basis=n_basis)
            blocks_tr.append(B_tr); blocks_te.append(B_te); blocks_P.append(P_ax)
        Phi_tr = np.concatenate(blocks_tr + [np.ones((train_X_rgb.shape[0], 1))], axis=1)
        Phi_te = np.concatenate(blocks_te + [np.ones((test_X_rgb.shape[0], 1))], axis=1)
        K_total = Phi_tr.shape[1]
        P = np.zeros((K_total, K_total))
        for i in range(3):
            P[i*n_basis:(i+1)*n_basis, i*n_basis:(i+1)*n_basis] = blocks_P[i]
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_add_hsv":
        # Proper additive multi-smooth: g_hue(cos h, sin h)  +  g_s(s)  +  g_v(v)
        # The hue term is a 2D radial Duchon over the (cos h, sin h) plane
        # with centers placed on the unit circle (cleaner than a uniform
        # square lattice — captures the natural support of hue).
        n_basis = 10
        # Centers on the unit circle for the hue smooth
        angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        centers_hue = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        Bh_tr, Ph = duchon_basis_radial(train_X_hsv[:, :2], centers_hue)   # (N, 12+3)
        Bh_te, _ = duchon_basis_radial(test_X_hsv[:, :2], centers_hue)
        Bs_tr, Ps = bspline_1d_basis(train_X_hsv[:, 2], n_basis=n_basis)
        Bs_te, _ = bspline_1d_basis(test_X_hsv[:, 2], n_basis=n_basis)
        Bv_tr, Pv = bspline_1d_basis(train_X_hsv[:, 3], n_basis=n_basis)
        Bv_te, _ = bspline_1d_basis(test_X_hsv[:, 3], n_basis=n_basis)
        # Strip the redundant intercepts from each block; keep one global
        Bh_tr = Bh_tr[:, :-3]; Bh_te = Bh_te[:, :-3]            # drop {1, cos, sin} null space
        Ph = Ph[:-3, :-3]
        Kh, Ks, Kv = Bh_tr.shape[1], Bs_tr.shape[1], Bv_tr.shape[1]
        Phi_tr = np.concatenate([Bh_tr, Bs_tr, Bv_tr, np.ones((Bh_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([Bh_te, Bs_te, Bv_te, np.ones((Bh_te.shape[0], 1))], axis=1)
        K_total = Phi_tr.shape[1]
        P = np.zeros((K_total, K_total))
        P[:Kh, :Kh] = Ph
        P[Kh:Kh+Ks, Kh:Kh+Ks] = Ps
        P[Kh+Ks:Kh+Ks+Kv, Kh+Ks:Kh+Ks+Kv] = Pv
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_rgb":
        Phi_tr, P = duchon_basis_radial(train_X_rgb, centers_rgb)
        Phi_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_hsv":
        # 4D coords (cos hue, sin hue, sat, val). Centers ON the natural
        # support: (cos 2πθ_i, sin 2πθ_i, s_j, v_k) for a grid of angles +
        # sat + val. No more arbitrary "near-unit-circle" mask.
        n_hue, n_sv = 8, 3                                    # 8 × 3 × 3 = 72 centers
        thetas = np.linspace(0.0, 2 * np.pi, n_hue, endpoint=False)
        sats = np.linspace(0.0, 1.0, n_sv)
        vals = np.linspace(0.0, 1.0, n_sv)
        centers_hsv = []
        for th in thetas:
            for s in sats:
                for v in vals:
                    centers_hsv.append([np.cos(th), np.sin(th), s, v])
        centers_hsv = np.array(centers_hsv)                   # (72, 4)
        Phi_tr, P = duchon_basis_radial(train_X_hsv, centers_hsv)
        Phi_te, _ = duchon_basis_radial(test_X_hsv, centers_hsv)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_rgb_with_hue":
        # Multi-smooth single GAM: f(R,G,B) + g(cos2πh, sin2πh)
        Phi_rgb_tr, P_rgb = duchon_basis_radial(train_X_rgb, centers_rgb)
        Phi_rgb_te, _ = duchon_basis_radial(test_X_rgb, centers_rgb)
        # 2D radial Duchon on (cos h, sin h)
        centers_hue = lattice_centers(4)[:, :2]            # reuse 2D 4x4 slice
        Phi_hue_tr, P_hue = duchon_basis_radial(train_X_hsv[:, :2], centers_hue)
        Phi_hue_te, _ = duchon_basis_radial(test_X_hsv[:, :2], centers_hue)
        # Strip the intercept column of the second smooth (only one needed)
        Phi_hue_tr = Phi_hue_tr[:, :-1]
        Phi_hue_te = Phi_hue_te[:, :-1]
        P_hue = P_hue[:-1, :-1]
        K1, K2 = Phi_rgb_tr.shape[1], Phi_hue_tr.shape[1]
        Phi_tr = np.concatenate([Phi_rgb_tr, Phi_hue_tr], axis=1)
        Phi_te = np.concatenate([Phi_rgb_te, Phi_hue_te], axis=1)
        P = np.zeros((K1 + K2, K1 + K2))
        # Single shared λ for now — the "multi-λ per smooth term" path needs
        # gaussian_reml_fit_additive's true multi-smooth solver; this is a
        # placeholder that REML can still close-form on (one λ scales both
        # penalty blocks together). v2 swaps to per-smooth λ via gamfit's
        # additive API once the harvest is checked in.
        P[:K1, :K1] = P_rgb
        P[K1:, K1:] = P_hue
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name in ("U_1d", "U_2d", "U_3d", "U_4d"):
        d = int(spec_name[2])
        # No GT axes used. Fit joint (T_train, B) by alternation on train Z.
        fit = fit_unsupervised_manifold(train_Z, d, cfg, n_iters=12, verbose=False)
        Phi_tr, _ = duchon_basis_radial(fit["T"], fit["centers"])
        Z_tr_pred = Phi_tr @ fit["B"]
        _, Z_te_pred = predict_unsupervised(test_Z, fit, d)
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
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name}  layers={cfg.layers}  device={device}", flush=True)

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

    if cfg.save_residuals:
        torch.save({
            "layer_resids": layer_resids,
            "colors": colors, "c_idx": c_idx,
            "templates": TEMPLATES, "config": asdict(cfg),
        }, out_dir / "residuals.pt")
        print(f"[save] residuals → {out_dir/'residuals.pt'}", flush=True)

    per_layer_results = {}
    for L in cfg.layers:
        X_full = layer_resids[L]                          # (N, D) fp32
        # Per-color average: collapse the 28-template noise
        per_color = torch.zeros(n_c, X_full.shape[1])
        for ci in range(n_c):
            m = (c_idx == ci)
            per_color[ci] = X_full[m].mean(0)
        # Per-dim normalize
        mu = per_color.mean(0, keepdim=True)
        sigma = per_color.std(0, keepdim=True).clamp(min=1e-6)
        Xn = ((per_color - mu) / sigma).numpy().astype(np.float64)
        # Top-K PCA
        U, S, Vt = np.linalg.svd(Xn - Xn.mean(0, keepdims=True), full_matrices=False)
        Z = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T[:, :cfg.n_pcs]    # (954, K_pcs)
        explained_var_ratio = (S[:cfg.n_pcs] ** 2) / (S ** 2).sum()
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
