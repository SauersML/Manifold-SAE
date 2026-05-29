"""Pair atoms across modalities (cogito-L40 LLM ↔ SD-UNet image) and plot.

Strategy
--------
For each atom k in modality M (with z[:, k] ∈ ℝ^n_prompts), compute the HSV-H
centroid of its top-20 activating prompts. Atoms are then matched between
modalities by 2D cosine similarity in (cos(H), sin(H)) circular embedding —
this respects the wrap-around at red.

Universality fraction = #atoms in cogito with a matched SD atom within
matched_threshold (default cosine ≥ 0.85) divided by # alive cogito atoms.

Side-by-side polar scatter shows the two hue rings together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _alive_atoms(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Return boolean mask of atoms with non-trivial range across prompts."""
    return (z.max(axis=0) - z.min(axis=0)) > eps


def atom_hue_centroids(
    z: np.ndarray,
    hues: np.ndarray,
    top_n: int = 20,
) -> np.ndarray:
    """For each atom, mean (cos(H), sin(H)) of its top-N activating prompts.

    Returns
    -------
    centroids : (F, 2) ndarray.  Dead atoms get (0, 0).
    """
    n, F = z.shape
    out = np.zeros((F, 2), dtype=np.float64)
    for k in range(F):
        col = z[:, k]
        if col.max() - col.min() < 1e-6:
            continue
        top = np.argsort(-col)[:top_n]
        h = hues[top]
        out[k, 0] = float(np.cos(h).mean())
        out[k, 1] = float(np.sin(h).mean())
    return out


def universality_fraction(
    z_cog: np.ndarray,
    hues_cog: np.ndarray,
    z_sd: np.ndarray,
    hues_sd: np.ndarray,
    matched_threshold: float = 0.85,
    top_n: int = 20,
) -> dict:
    """Pair cogito atoms ↔ SD atoms by hue-centroid cosine."""
    Cc = atom_hue_centroids(z_cog, hues_cog, top_n=top_n)
    Cs = atom_hue_centroids(z_sd, hues_sd, top_n=top_n)
    alive_c = _alive_atoms(z_cog)
    alive_s = _alive_atoms(z_sd)

    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n > 1e-12, n, 1.0)

    Ccn = _norm(Cc); Csn = _norm(Cs)
    sim = Ccn @ Csn.T                                  # (F_c, F_s)
    # mask: only consider alive atoms on each side
    sim[~alive_c, :] = -1.0
    sim[:, ~alive_s] = -1.0

    best_sd = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)
    matched = (best_sim >= matched_threshold) & alive_c
    n_alive_c = int(alive_c.sum())
    n_match = int(matched.sum())
    return dict(
        alive_cogito=n_alive_c,
        alive_sd=int(alive_s.sum()),
        matched=n_match,
        universality_fraction=(n_match / n_alive_c) if n_alive_c else 0.0,
        matched_threshold=matched_threshold,
        pairs=[(int(i), int(best_sd[i]), float(best_sim[i]))
               for i in range(len(best_sd)) if matched[i]],
    )


def _polar_panel(ax, theta, rgb, title: str):
    ax.scatter(theta, np.ones_like(theta), c=rgb, s=22,
               edgecolor="black", linewidth=0.2, alpha=0.85)
    ax.set_title(title, fontsize=10)


def side_by_side_hue_ring(
    z_sd: np.ndarray,
    hues_sd: np.ndarray,
    rgb_sd: np.ndarray,
    z_cog: np.ndarray,
    cogito_meta_path: Path,
    out_path: Path,
    sd_atom: int | None = None,
    cog_atom: int | None = None,
) -> Path:
    """2-panel polar plot: cogito best-hue atom + SD best-hue atom."""
    import matplotlib.pyplot as plt
    from experiments.slop.auto_exp_77_diffusion_sae import (   # local import to avoid cycle at import time
        best_atom_hue_corr,
        rgb_to_hue_radians,
    )

    if cog_atom is None:
        # Need cogito hues + rgb to score.
        cmeta = json.loads(Path(cogito_meta_path).read_text())
        cog_colors = cmeta["colors"]
        cog_rgbs = np.array([c[1] for c in cog_colors], dtype=np.float64)
        cog_n_tmpl = cmeta.get("n_templates", 28)
        cog_hues = np.repeat(rgb_to_hue_radians(cog_rgbs), cog_n_tmpl)
        m = min(z_cog.shape[0], cog_hues.shape[0])
        z_cog, cog_hues = z_cog[:m], cog_hues[:m]
        cog_rgb_perp = np.repeat(np.clip(cog_rgbs/255.0, 0, 1), cog_n_tmpl, axis=0)[:m]
        cog_res = best_atom_hue_corr(z_cog, cog_hues)
        cog_atom = cog_res["best_atom"]
        cog_corr = cog_res["best_abs"]
    else:
        cog_corr = float("nan")
        cog_rgb_perp = np.ones((z_cog.shape[0], 3)) * 0.5

    if sd_atom is None:
        sd_res = best_atom_hue_corr(z_sd, hues_sd)
        sd_atom = sd_res["best_atom"]
        sd_corr = sd_res["best_abs"]
    else:
        sd_corr = float("nan")

    col_c = z_cog[:, cog_atom]
    theta_c = (col_c - col_c.min()) / max(col_c.max() - col_c.min(), 1e-12) * 2 * np.pi
    col_s = z_sd[:, sd_atom]
    theta_s = (col_s - col_s.min()) / max(col_s.max() - col_s.min(), 1e-12) * 2 * np.pi

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             subplot_kw={"projection": "polar"})
    _polar_panel(axes[0], theta_c, cog_rgb_perp,
                 f"cogito-L40 (atom {cog_atom})  |circ_corr|={cog_corr:.3f}")
    _polar_panel(axes[1], theta_s, rgb_sd,
                 f"SD UNet (atom {sd_atom})  |circ_corr|={sd_corr:.3f}")
    fig.suptitle("Cross-modality hue rings: LLM (cogito-L40) vs Image (SD UNet)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "atom_hue_centroids",
    "universality_fraction",
    "side_by_side_hue_ring",
]
