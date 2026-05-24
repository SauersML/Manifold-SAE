"""SAE Feature Steering Benchmark.

Quantitative causal-control evaluation across 4 steering protocols, designed
to compare SAE variants on something more demanding than reconstruction R^2
or sparsity: *can you actually push a feature and get the intended effect?*

The 4 protocols
---------------
1. **Linear push** — pick the top-k atoms most correlated with hue. Add
   +alpha * sigma_atom to z, decode, measure correlation between the
   intended push direction (in hue-projection space) and the observed
   delta. Reports R^2 of intended vs observed across rows.

2. **Anchor swap** (motivated by `auto_exp_44`) — for each pair (source row,
   anchor color), encode source -> z. Replace the "hue block" of z (atoms
   most correlated with hue cos/sin) with the encoding of the anchor's
   prototypical row in those same atoms. Decode. Measure whether the new
   decoded activation is closer in hue to the anchor than the original.

3. **Magnitude scaling** — scale the top hue atom (per row) by a grid
   ``[0, 0.5, 1, 2, 5]``. Measure how the projected "hue intensity" of
   the decode responds (monotonicity / Spearman).

4. **Compositional** — pick a hue-axis atom A_h and a lightness-axis atom
   A_v. Push both simultaneously by alpha and beta. Measure how
   independent the two effects are (cross-effect ratio).

Each protocol returns ``(steering_R^2, side_effect_norm, monotonicity)``.

Design follows ``manifold_sae/eval/harness.py``: the bench accepts any
``SAEWrapper`` so adding a new SAE variant only requires writing a loader
in ``registry.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch

from .harness import SAEWrapper, collect_activations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hue_cs(hue: np.ndarray) -> np.ndarray:
    """Circular hue -> (cos, sin) on the unit circle (2 * pi * hue)."""
    return np.stack(
        [np.cos(2.0 * np.pi * hue), np.sin(2.0 * np.pi * hue)], axis=1
    ).astype(np.float64)


def _hue_direction(X: np.ndarray, hue: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Linear regressor from input space -> (cos hue, sin hue). Returns W
    of shape (D, 2). This gives us a way to project arbitrary decoded
    deltas back onto a continuous hue coordinate.
    """
    H = _hue_cs(hue)
    D = X.shape[1]
    A = X.T @ X + ridge * np.eye(D, dtype=X.dtype)
    b = X.T @ H.astype(X.dtype)
    return np.linalg.solve(A, b)


def _value_direction(X: np.ndarray, value: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Linear regressor X -> value (lightness V). Returns W of shape (D,)."""
    D = X.shape[1]
    A = X.T @ X + ridge * np.eye(D, dtype=X.dtype)
    b = X.T @ value.astype(X.dtype)
    return np.linalg.solve(A, b)


def _atom_hue_corr(Z: np.ndarray, hue: np.ndarray) -> np.ndarray:
    """Per-atom correlation magnitude with hue (cos,sin). Shape (F,)."""
    H = _hue_cs(hue)
    Z = Z.astype(np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    Hc = H - H.mean(0, keepdims=True)
    z_std = Zc.std(0).clip(min=1e-6)
    h_std = Hc.std(0).clip(min=1e-6)
    corr = (Zc.T @ Hc) / (Zc.shape[0] - 1)
    corr = corr / (z_std[:, None] * h_std[None, :])
    return np.linalg.norm(corr, axis=1)


def _atom_scalar_corr(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-atom correlation with a scalar target. Shape (F,)."""
    Z = Z.astype(np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    yc = y.astype(np.float64) - float(y.mean())
    z_std = Zc.std(0).clip(min=1e-6)
    y_std = float(yc.std()) or 1e-6
    return (Zc.T @ yc) / (Zc.shape[0] - 1) / (z_std * y_std)


def _decode_batched(
    model: SAEWrapper,
    X: torch.Tensor,
    Z_modified: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Decode each row of Z_modified, re-encoding X first to populate any
    wrapper-internal caches (e.g. ManifoldFourier needs the theta basis).

    Returns numpy array of decoded activations (N, D).
    """
    device = X.device
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        with torch.no_grad():
            # Force the wrapper to set up any per-row caches.
            _ = model.encode(xb)
            zb = torch.from_numpy(Z_modified[i : i + batch_size]).to(device).to(xb.dtype)
            recon = model.decode_from_activations(zb)
        out.append(recon.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _base_decode(model: SAEWrapper, X: torch.Tensor, Z: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return _decode_batched(model, X, Z, batch_size=batch_size)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = (np.linalg.norm(ra) * np.linalg.norm(rb)) or 1e-12
    return float((ra @ rb) / den)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProtocolScore:
    name: str
    steering_r2: float = float("nan")
    side_effect_norm: float = float("nan")
    monotonicity: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class BenchResult:
    model_name: str
    protocols: dict[str, ProtocolScore] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "protocols": {k: v.to_dict() for k, v in self.protocols.items()},
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, p in self.protocols.items():
            out[f"{k}_r2"] = p.steering_r2
            out[f"{k}_side"] = p.side_effect_norm
            out[f"{k}_mono"] = p.monotonicity
        # composite score: mean R^2 - mean side-effect (both clipped to [0,1])
        r2s = [np.nan_to_num(p.steering_r2, nan=0.0) for p in self.protocols.values()]
        sides = [np.nan_to_num(p.side_effect_norm, nan=0.0) for p in self.protocols.values()]
        out["composite"] = float(np.mean(r2s) - 0.25 * np.mean(sides))
        return out


# ---------------------------------------------------------------------------
# Main bench
# ---------------------------------------------------------------------------


class SteeringBench:
    """Quantitative steering benchmark.

    Parameters
    ----------
    model       : SAEWrapper to evaluate.
    X_val       : (N, D) torch.Tensor validation activations (already
                  preprocessed / mean-subtracted the same way the model expects).
    hsv_labels  : (N, 3) numpy array of per-row HSV in [0, 1].
    name_labels : optional (N,) int array of "color name id" or similar.
                  Used by protocol 2 (anchor swap) to find prototypical rows.
    """

    def __init__(
        self,
        model: SAEWrapper,
        X_val: torch.Tensor,
        hsv_labels: np.ndarray,
        name_labels: np.ndarray | None = None,
        k_hue_atoms: int = 8,
        k_value_atoms: int = 8,
        seed: int = 0,
    ):
        self.model = model
        self.X = X_val
        self.hsv = np.asarray(hsv_labels, dtype=np.float64)
        assert self.hsv.shape == (X_val.shape[0], 3), \
            f"hsv_labels must be (N, 3), got {self.hsv.shape}"
        self.names = (
            np.asarray(name_labels) if name_labels is not None else None
        )
        self.k_hue = k_hue_atoms
        self.k_val = k_value_atoms
        self.seed = seed

        # Cache base encoding once.
        self._Z = collect_activations(self.model, self.X)
        self._Z = np.ascontiguousarray(self._Z)

        # Cache linear hue / value directions in input space.
        Xn = self.X.detach().cpu().numpy().astype(np.float64)
        self._W_hue = _hue_direction(Xn, self.hsv[:, 0])
        self._W_val = _value_direction(Xn, self.hsv[:, 2])

        # Atom rankings.
        self._atom_hue_score = _atom_hue_corr(self._Z, self.hsv[:, 0])
        self._atom_val_score = np.abs(_atom_scalar_corr(self._Z, self.hsv[:, 2]))
        self._top_hue_atoms = np.argsort(-self._atom_hue_score)[: self.k_hue]
        self._top_val_atoms = np.argsort(-self._atom_val_score)[: self.k_val]

        # Pre-decode base.
        self._base_decoded = _base_decode(self.model, self.X, self._Z)

    # ------------------------------------------------------------------
    # Protocol 1: Linear push
    # ------------------------------------------------------------------

    def protocol_linear_push(self, alpha: float = 1.0) -> ProtocolScore:
        """Push top hue-atoms by +alpha * sigma_atom in the *positive hue corr
        direction*. Compare projected hue-delta to expected.
        """
        Z = self._Z
        top = self._top_hue_atoms
        sigma = Z[:, top].std(0).clip(min=1e-6)
        # Direction in (cos,sin) space contributed by each atom (sign).
        H = _hue_cs(self.hsv[:, 0])
        Zc = Z.astype(np.float64) - Z.mean(0, keepdims=True)
        Hc = H - H.mean(0, keepdims=True)
        atom_dir = (Zc.T @ Hc) / max(Z.shape[0] - 1, 1)  # (F, 2)
        signs = np.sign(np.einsum("fk,fk->f", atom_dir[top], atom_dir[top].sum(0, keepdims=True).repeat(top.size, axis=0)))
        signs = np.where(signs == 0, 1.0, signs)

        Z_pushed = Z.copy()
        Z_pushed[:, top] = Z_pushed[:, top] + (alpha * sigma * signs)[None, :]
        decoded = _decode_batched(self.model, self.X, Z_pushed)
        delta = decoded - self._base_decoded
        # Project per-row delta onto (cos,sin) hue direction.
        proj = delta @ self._W_hue  # (N, 2)
        # Intended per-row direction (uniform across rows for this protocol):
        intended = (alpha * sigma * signs)[:, None] * atom_dir[top]  # (k, 2)
        intended_sum = intended.sum(0)  # (2,)
        # Per-row R^2 between observed and intended (cos,sin).
        # Cosine similarity squared is a reasonable scalar steering R^2.
        denom = np.linalg.norm(proj, axis=1) * (np.linalg.norm(intended_sum) + 1e-12)
        cos = (proj @ intended_sum) / (denom + 1e-12)
        steering_r2 = float(np.mean(cos ** 2))
        # Side-effect: orthogonal-to-intended decoded delta norm / total delta norm.
        u = intended_sum / (np.linalg.norm(intended_sum) + 1e-12)
        proj_par = proj @ u
        proj_perp = proj - np.outer(proj_par, u)
        denom_total = np.linalg.norm(delta, axis=1).mean() + 1e-12
        side = float(np.linalg.norm(proj_perp, axis=1).mean() / denom_total)
        # Monotonicity vs alpha: run a small sweep.
        mono = self._linear_push_monotonicity(top, sigma, signs, atom_dir)
        return ProtocolScore(
            name="linear_push",
            steering_r2=steering_r2,
            side_effect_norm=side,
            monotonicity=mono,
            extra={"alpha": alpha, "k_hue": int(self.k_hue),
                   "top_atoms": top.tolist()},
        )

    def _linear_push_monotonicity(self, top, sigma, signs, atom_dir) -> float:
        alphas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        proj_means = []
        Z = self._Z
        u = (sigma * signs * np.linalg.norm(atom_dir[top], axis=1)).sum()
        for a in alphas:
            Zp = Z.copy()
            Zp[:, top] = Zp[:, top] + (a * sigma * signs)[None, :]
            decoded = _decode_batched(self.model, self.X, Zp)
            delta = decoded - self._base_decoded
            proj = delta @ self._W_hue
            intended = (a * sigma * signs)[:, None] * atom_dir[top]
            intended_sum = intended.sum(0)
            if np.linalg.norm(intended_sum) < 1e-9:
                proj_means.append(0.0)
            else:
                u_ = intended_sum / np.linalg.norm(intended_sum)
                proj_means.append(float((proj @ u_).mean()))
        return _spearman(alphas, np.array(proj_means))

    # ------------------------------------------------------------------
    # Protocol 2: Anchor swap
    # ------------------------------------------------------------------

    def _anchor_swap_decode(
        self,
        Z_mod: np.ndarray,
        src: np.ndarray,
        a_row: int,
        atom_mask: np.ndarray,
        supports_theta_swap: bool,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Decode the amp-swapped Z_mod, transplanting donor theta on the
        masked atoms when the wrapper supports it.
        """
        if not supports_theta_swap:
            return _decode_batched(self.model, self.X[src], Z_mod, batch_size=batch_size)

        device = self.X.device
        out = []
        for i in range(0, Z_mod.shape[0], batch_size):
            xb = self.X[src[i : i + batch_size]]
            with torch.no_grad():
                _ = self.model.encode(xb)  # populate target theta cache
                zb = torch.from_numpy(Z_mod[i : i + batch_size]).to(device).to(xb.dtype)
                # Donor row index is into self.X (a_row is an absolute
                # row id), so pass self.X as x_source.
                recon = self.model.swap_theta_from(
                    zb, self.X, a_row, atom_mask
                )
            out.append(recon.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    def protocol_anchor_swap(self, n_anchors: int = 4, n_sources: int | None = None) -> ProtocolScore:
        """For each (source, anchor) pair, replace the values of the hue-atom
        block in z_source with those of z_anchor. Decode and measure whether
        the decoded hue moves toward the anchor's hue.
        """
        Z = self._Z
        N = Z.shape[0]
        top = self._top_hue_atoms
        rng = np.random.default_rng(self.seed)
        # Pick `n_anchors` representative rows spread around the hue circle.
        hue = self.hsv[:, 0]
        targets = np.linspace(0.0, 1.0, n_anchors, endpoint=False)
        anchor_rows = np.array(
            [int(np.argmin(np.abs(hue - t))) for t in targets]
        )
        if n_sources is None:
            n_sources = min(N, 256)
        src = rng.choice(N, size=n_sources, replace=False)

        hue_after_all = []
        hue_source_all = []
        hue_anchor_all = []
        side_norms = []
        # Wrappers like ManifoldFourierWrapper cache a per-row theta
        # basis; a raw amp-swap leaves the TARGET's theta in place and
        # the hue transplant silently no-ops. If the wrapper exposes a
        # swap_theta_from API, use it so the donor theta travels with
        # the donor amp block. TopK / L1 wrappers don't need this.
        supports_theta_swap = hasattr(self.model, "swap_theta_from")
        atom_mask = np.zeros(Z.shape[1], dtype=bool)
        atom_mask[top] = True
        for a_row in anchor_rows:
            Z_mod = Z[src].copy()
            Z_mod[:, top] = Z[a_row, top][None, :]
            decoded = self._anchor_swap_decode(
                Z_mod, src, int(a_row), atom_mask, supports_theta_swap
            )
            # Project decoded to hue.
            cs = decoded @ self._W_hue  # (n_src, 2)
            theta = np.arctan2(cs[:, 1], cs[:, 0]) / (2.0 * np.pi)
            theta = np.mod(theta, 1.0)
            hue_after_all.append(theta)
            hue_source_all.append(hue[src])
            hue_anchor_all.append(np.full(src.size, hue[a_row]))
            # Side effect: lightness shift (we only meant to change hue).
            v_decoded = decoded @ self._W_val
            v_base = self._base_decoded[src] @ self._W_val
            side_norms.append(float(np.mean(np.abs(v_decoded - v_base))))

        h_after = np.concatenate(hue_after_all)
        h_src = np.concatenate(hue_source_all)
        h_anch = np.concatenate(hue_anchor_all)

        # circular distance helper
        def cdist(a, b):
            d = np.abs(a - b)
            return np.minimum(d, 1.0 - d)

        d_before = cdist(h_src, h_anch)
        d_after = cdist(h_after, h_anch)
        # Steering R^2: fraction of distance closed.
        # 1 - (mean d_after / mean d_before)
        closed = 1.0 - (d_after.mean() / max(d_before.mean(), 1e-6))
        steering_r2 = float(max(min(closed, 1.0), -1.0))
        side = float(np.mean(side_norms))
        # Monotonicity: across the 4 anchors, does the order of decoded hues
        # match the order of anchor hues?
        per_anchor_mean = np.array([h.mean() for h in hue_after_all])
        anchor_hues = np.array([hue[a] for a in anchor_rows])
        mono = _spearman(anchor_hues, per_anchor_mean)
        return ProtocolScore(
            name="anchor_swap",
            steering_r2=steering_r2,
            side_effect_norm=side,
            monotonicity=mono,
            extra={"n_anchors": n_anchors, "n_sources": int(n_sources),
                   "anchor_rows": anchor_rows.tolist(),
                   "mean_d_before": float(d_before.mean()),
                   "mean_d_after": float(d_after.mean())},
        )

    # ------------------------------------------------------------------
    # Protocol 3: Magnitude scaling
    # ------------------------------------------------------------------

    def protocol_magnitude_scaling(self) -> ProtocolScore:
        """Scale the single top hue atom by [0, 0.5, 1, 2, 5]. Measure
        whether projected hue intensity responds monotonically.
        """
        Z = self._Z
        a_star = int(self._top_hue_atoms[0])
        scales = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        proj_intensities = []
        side_norms = []
        deltas_par = []
        for s in scales:
            Zp = Z.copy()
            Zp[:, a_star] = Zp[:, a_star] * s
            decoded = _decode_batched(self.model, self.X, Zp)
            cs = decoded @ self._W_hue
            intensity = float(np.linalg.norm(cs, axis=1).mean())
            proj_intensities.append(intensity)
            delta = decoded - self._base_decoded
            d_perp = delta - (delta @ self._W_hue) @ self._W_hue.T
            # side: norm not explained by hue projection.
            side_norms.append(float(np.linalg.norm(d_perp, axis=1).mean()))
            deltas_par.append(float(np.linalg.norm(decoded @ self._W_hue, axis=1).mean()))
        proj_intensities = np.array(proj_intensities)
        mono = _spearman(scales, proj_intensities)
        # R^2 of linear fit intensity ~ scale
        s_c = scales - scales.mean(); p_c = proj_intensities - proj_intensities.mean()
        r2 = 0.0
        if s_c.std() > 0 and p_c.std() > 0:
            r2 = float((s_c @ p_c) ** 2 / ((s_c @ s_c) * (p_c @ p_c)))
        side = float(np.mean(side_norms) / (np.mean(deltas_par) + 1e-9))
        return ProtocolScore(
            name="magnitude_scaling",
            steering_r2=r2,
            side_effect_norm=side,
            monotonicity=mono,
            extra={"scales": scales.tolist(),
                   "intensities": proj_intensities.tolist(),
                   "top_atom": a_star},
        )

    # ------------------------------------------------------------------
    # Protocol 4: Compositional (hue + lightness)
    # ------------------------------------------------------------------

    def protocol_compositional(self, alpha: float = 1.0, beta: float = 1.0) -> ProtocolScore:
        """Push hue-atom by alpha AND value-atom by beta simultaneously.
        Compare to (alpha-only, beta-only) decoded deltas: are the effects
        approximately additive and orthogonal?
        """
        Z = self._Z
        a_h = self._top_hue_atoms[: max(self.k_hue // 2, 1)]
        a_v = self._top_val_atoms[: max(self.k_val // 2, 1)]
        # Avoid double-counting if any overlap.
        a_v = np.array([i for i in a_v if i not in set(a_h.tolist())])
        if a_v.size == 0:
            a_v = np.array([int(self._top_val_atoms[-1])])

        sig_h = Z[:, a_h].std(0).clip(min=1e-6)
        sig_v = Z[:, a_v].std(0).clip(min=1e-6)

        Z_h = Z.copy(); Z_h[:, a_h] = Z_h[:, a_h] + alpha * sig_h
        Z_v = Z.copy(); Z_v[:, a_v] = Z_v[:, a_v] + beta * sig_v
        Z_hv = Z.copy()
        Z_hv[:, a_h] = Z_hv[:, a_h] + alpha * sig_h
        Z_hv[:, a_v] = Z_hv[:, a_v] + beta * sig_v

        d_h = _decode_batched(self.model, self.X, Z_h) - self._base_decoded
        d_v = _decode_batched(self.model, self.X, Z_v) - self._base_decoded
        d_hv = _decode_batched(self.model, self.X, Z_hv) - self._base_decoded

        # Additivity: how well d_hv ≈ d_h + d_v ?
        residual = d_hv - (d_h + d_v)
        total = d_hv
        ss_res = float((residual ** 2).sum())
        ss_tot = float((total ** 2).sum()) + 1e-9
        additivity_r2 = float(1.0 - ss_res / ss_tot)

        # Cross-axis leakage:
        # Hue push should ideally not move value, value push should ideally
        # not move hue. Compute normalized cross-talk.
        h_cs_from_h = d_h @ self._W_hue
        h_cs_from_v = d_v @ self._W_hue
        v_from_h = d_h @ self._W_val
        v_from_v = d_v @ self._W_val

        h_intended = float(np.linalg.norm(h_cs_from_h, axis=1).mean()) + 1e-9
        v_intended = float(np.abs(v_from_v).mean()) + 1e-9
        cross_h2v = float(np.abs(v_from_h).mean() / v_intended)
        cross_v2h = float(np.linalg.norm(h_cs_from_v, axis=1).mean() / h_intended)
        side = 0.5 * (cross_h2v + cross_v2h)

        # Monotonicity sweep across alpha (keep beta=alpha).
        sweep = np.array([0.0, 0.5, 1.0, 2.0])
        intensities = []
        for s in sweep:
            Zs = Z.copy()
            Zs[:, a_h] = Zs[:, a_h] + s * sig_h
            Zs[:, a_v] = Zs[:, a_v] + s * sig_v
            dec = _decode_batched(self.model, self.X, Zs)
            ds = dec - self._base_decoded
            cs = ds @ self._W_hue
            inten = float(np.linalg.norm(cs, axis=1).mean()
                          + np.abs(ds @ self._W_val).mean())
            intensities.append(inten)
        mono = _spearman(sweep, np.array(intensities))
        return ProtocolScore(
            name="compositional",
            steering_r2=additivity_r2,
            side_effect_norm=side,
            monotonicity=mono,
            extra={"alpha": alpha, "beta": beta,
                   "hue_atoms": a_h.tolist(),
                   "value_atoms": a_v.tolist(),
                   "cross_h2v": cross_h2v,
                   "cross_v2h": cross_v2h},
        )

    # ------------------------------------------------------------------

    def run(self) -> BenchResult:
        out = BenchResult(model_name=self.model.name)
        out.protocols["linear_push"] = self.protocol_linear_push()
        out.protocols["anchor_swap"] = self.protocol_anchor_swap()
        out.protocols["magnitude_scaling"] = self.protocol_magnitude_scaling()
        out.protocols["compositional"] = self.protocol_compositional()
        return out


__all__ = [
    "SteeringBench",
    "BenchResult",
    "ProtocolScore",
]
