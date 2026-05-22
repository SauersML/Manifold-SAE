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


def duchon3d_design_and_penalty(
    X: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Phi (N, K+4), Penalty (K+4, K+4)).

    Phi columns: K radial kernels phi_k(x) = ||x - c_k||  +  null-space
    [1, R, G, B]. Penalty is the kernel Gram matrix (K x K) in the kernel
    block, zeros on the null-space block.
    """
    N = X.shape[0]
    K = centers.shape[0]
    # pairwise distances
    diff = X[:, None, :] - centers[None, :, :]        # (N, K, 3)
    Dnk = np.linalg.norm(diff, axis=2)                # (N, K) = ||x-c_k||
    null = np.concatenate([np.ones((N, 1)), X], axis=1)  # (N, 4): [1, R, G, B]
    Phi = np.concatenate([Dnk, null], axis=1)         # (N, K+4)
    # penalty: Gram of kernels on centers, then expand
    diff_cc = centers[:, None, :] - centers[None, :, :]
    Pkk = np.linalg.norm(diff_cc, axis=2)             # (K, K)
    P = np.zeros((K + 4, K + 4))
    P[:K, :K] = Pkk
    return Phi, P


def duchon_basis_general(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generic radial Duchon m=2 in d dims with kernel r. Null space {1, x_1, ..., x_d}."""
    N, d = X.shape
    K = centers.shape[0]
    Dnk = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    null = np.concatenate([np.ones((N, 1)), X], axis=1)              # (N, d+1)
    Phi = np.concatenate([Dnk, null], axis=1)
    Pkk = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    P = np.zeros((K + d + 1, K + d + 1))
    P[:K, :K] = Pkk
    return Phi, P


def bspline_1d_basis(t: np.ndarray, n_basis: int = 10, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """1D penalized B-spline basis + 2nd-derivative penalty matrix.

    Equally spaced knots over [0, 1]. Returns (B (N, n_basis), P (n_basis, n_basis)).
    The 2nd-derivative penalty is the matrix D2^T D2 where D2 is the 2nd-difference
    operator on coefficients — standard P-spline construction.
    """
    from scipy.interpolate import BSpline
    knots = np.concatenate([
        np.repeat(0.0, degree),
        np.linspace(0.0, 1.0, n_basis - degree + 1),
        np.repeat(1.0, degree),
    ])
    B = np.zeros((len(t), n_basis))
    for k in range(n_basis):
        coef = np.zeros(n_basis); coef[k] = 1.0
        spl = BSpline(knots, coef, degree)
        B[:, k] = spl(np.clip(t, 0.0, 1.0))
    # 2nd-difference penalty on coefficients
    D2 = np.zeros((n_basis - 2, n_basis))
    for i in range(n_basis - 2):
        D2[i, i] = 1.0; D2[i, i + 1] = -2.0; D2[i, i + 2] = 1.0
    P = D2.T @ D2
    return B, P


# =============================================================================
# REML fit
# =============================================================================
def reml_fit(Phi: np.ndarray, Z: np.ndarray, P: np.ndarray, init_log_lambda: float = 0.0,
              max_iter: int = 30, tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """Closed-form Gaussian REML for a single design Phi (N, K) and multi-output
    target Z (N, R). Selects a single log-lambda by Newton iteration on the
    REML score. Returns (B (K, R), log_lambda).

    Robust to mild near-singularity via a tiny ridge on the kernel block.
    """
    N, K = Phi.shape
    PhitPhi = Phi.T @ Phi
    PhitZ = Phi.T @ Z                               # (K, R)
    ZtZ = (Z * Z).sum(axis=0)                       # (R,)

    log_lambda = float(init_log_lambda)
    eye_jitter = 1e-10 * np.eye(K)
    for it in range(max_iter):
        lam = math.exp(log_lambda)
        A = PhitPhi + lam * P + eye_jitter
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            eye_jitter *= 10
            continue
        # B = A^{-1} Phi'Z
        B = np.linalg.solve(A, PhitZ)
        # REML score gradient w.r.t. log_lambda
        # d(logdet A)/d log_lambda = lam * tr(A^{-1} P)
        # d(B'PhitPhi B - 2 B'PhitZ + ZtZ + lam B'P B)/d log_lambda ≈ lam B'P B
        # Use second-derivative diagonal approximation; small problem.
        Ainv_P = np.linalg.solve(A, P)
        tr_AinvP = float(np.trace(Ainv_P))
        # Effective degrees-of-freedom-ish gradient:
        # ∂/∂log_lambda [ logdet(A) + (1/sig^2)(... ) ] but we don't track sig^2 separately.
        # Pragmatic: Newton on
        #   g(log_lambda) = lam * sum_r (B[:,r]'P B[:,r])/ZtZ[r] - tr_AinvP
        bPb_per_r = ((B.T @ P) * B.T).sum(axis=1)
        # weighted avg by Z scale
        wts = 1.0 / np.clip(ZtZ, 1e-12, None)
        score = float(lam * (bPb_per_r * wts).sum() - tr_AinvP * len(ZtZ))
        # Crude line update: half-step proportional to score
        step = -0.5 * np.tanh(score / max(1.0, abs(score)))
        log_lambda += float(step)
        if abs(step) < tol:
            break
    lam = math.exp(log_lambda)
    A = PhitPhi + lam * P + eye_jitter
    B = np.linalg.solve(A, PhitZ)
    return B, log_lambda


def ridge_fit(Phi: np.ndarray, Z: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Plain ridge (used when there's no penalty matrix — linear baselines)."""
    PhitPhi = Phi.T @ Phi
    A = PhitPhi + alpha * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ Z)


# =============================================================================
# Spec definitions
# =============================================================================
def coord_rgb(R, G, B, hsv) -> np.ndarray:
    return np.stack([R, G, B], axis=1)


def coord_hsv_periodic(R, G, B, hsv) -> np.ndarray:
    h = hsv[:, 0]
    return np.stack([np.cos(2 * np.pi * h), np.sin(2 * np.pi * h), hsv[:, 1], hsv[:, 2]], axis=1)


SPECS = {
    # name -> (coord_fn, basis_kind, basis_kwargs)
    "L_lin_rgb": ("ridge_rgb",),
    "L_lin_hsv": ("ridge_hsv",),
    "L_add_rgb": ("additive_rgb",),
    "L_add_hsv": ("additive_hsv",),
    "L_joint_rgb": ("smooth_rgb",),
    "L_joint_hsv": ("smooth_hsv",),
    "L_joint_rgb_with_hue": ("smooth_rgb_plus_hue",),
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
        n_basis = 10
        # hue handled as cos/sin pair (2 columns, treated as one block with shared λ)
        Bh_tr = train_X_hsv[:, :2]
        Bh_te = test_X_hsv[:, :2]
        Bs_tr, Ps = bspline_1d_basis(train_X_hsv[:, 2], n_basis=n_basis)
        Bs_te, _ = bspline_1d_basis(test_X_hsv[:, 2], n_basis=n_basis)
        Bv_tr, Pv = bspline_1d_basis(train_X_hsv[:, 3], n_basis=n_basis)
        Bv_te, _ = bspline_1d_basis(test_X_hsv[:, 3], n_basis=n_basis)
        Phi_tr = np.concatenate([Bh_tr, Bs_tr, Bv_tr, np.ones((Bh_tr.shape[0], 1))], axis=1)
        Phi_te = np.concatenate([Bh_te, Bs_te, Bv_te, np.ones((Bh_te.shape[0], 1))], axis=1)
        K_total = Phi_tr.shape[1]
        P = np.zeros((K_total, K_total))
        # hue cos/sin: small L2 penalty
        P[0, 0] = P[1, 1] = 1.0
        P[2:2+n_basis, 2:2+n_basis] = Ps
        P[2+n_basis:2+2*n_basis, 2+n_basis:2+2*n_basis] = Pv
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_rgb":
        Phi_tr, P = duchon3d_design_and_penalty(train_X_rgb, centers_rgb)
        Phi_te, _ = duchon3d_design_and_penalty(test_X_rgb, centers_rgb)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_hsv":
        # 4D coords: (cos hue, sin hue, sat, val). Lattice in [-1,1]^2 x [0,1]^2.
        # Use the same lattice density, just in 4D.
        per_side = max(3, cfg.lattice_per_side - 1)        # keep K small in 4D
        ax_hc = np.linspace(-1.0, 1.0, per_side)
        ax_hs = np.linspace(-1.0, 1.0, per_side)
        ax_s = np.linspace(0.0, 1.0, per_side)
        ax_v = np.linspace(0.0, 1.0, per_side)
        Hc, Hs, S, V = np.meshgrid(ax_hc, ax_hs, ax_s, ax_v, indexing="ij")
        centers_hsv = np.stack([Hc.flatten(), Hs.flatten(), S.flatten(), V.flatten()], axis=1)
        # subsample: keep only centers with cos^2 + sin^2 ≈ 1 (on the hue circle slice)
        mask = (np.abs(centers_hsv[:, 0] ** 2 + centers_hsv[:, 1] ** 2 - 1.0) < 0.3)
        centers_hsv = centers_hsv[mask]
        Phi_tr, P = duchon_basis_general(train_X_hsv, centers_hsv)
        Phi_te, _ = duchon_basis_general(test_X_hsv, centers_hsv)
        B, _ = reml_fit(Phi_tr, train_Z, P, cfg.init_log_lambda)
        return Phi_tr @ B, Phi_te @ B

    if spec_name == "L_joint_rgb_with_hue":
        # Multi-smooth single GAM: f(R,G,B) + g(cos2πh, sin2πh)
        Phi_rgb_tr, P_rgb = duchon3d_design_and_penalty(train_X_rgb, centers_rgb)
        Phi_rgb_te, _ = duchon3d_design_and_penalty(test_X_rgb, centers_rgb)
        # 2D radial Duchon on (cos h, sin h)
        centers_hue = lattice_centers(4)[:, :2]            # reuse 2D 4x4 slice
        Phi_hue_tr, P_hue = duchon_basis_general(train_X_hsv[:, :2], centers_hue)
        Phi_hue_te, _ = duchon_basis_general(test_X_hsv[:, :2], centers_hue)
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

        per_layer_results[f"L{L}"] = {
            "explained_variance_ratio_topK": [float(x) for x in explained_var_ratio],
            "specs": spec_results,
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
