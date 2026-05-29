"""auto_exp_77 — SAE-on-Diffusion-UNet hue-ring contrarian probe.

Tests modality-universality of the cogito-L40 hue-ring atom by training a
Manifold-SAE on Stable-Diffusion UNet residuals on the same xkcd-color
prompts and measuring whether the learned atom positions θ_i correlate
circularly with HSV hue.

Pipeline:

    1. Load runs/COLOR_SD_UNET_MID/{X.npy, meta.json}.
       (Run scripts/harvest_sd_unet.py first if missing.)
    2. Train ManifoldSAE F=128 (manifold_sae.diffusion.train_sd_manifold_sae).
    3. Hue-ring metric: max-over-atoms circular correlation between θ_i and
       HSV_H of the prompt color.
    4. Falsifiable verdict:
            |circ_corr| ≥ 0.30  →  hue-ring is MODALITY-UNIVERSAL (major)
            |circ_corr| <  0.30 →  hue-ring is TEXT-LLM-SPECIFIC.
       cogito-L40 baseline is ~0.6+ (see project_phate_atlas_sae.md and the
       universal-SAE memory entry).
    5. Optional cross-modality atlas (manifold_sae.diffusion.cross_modality_atlas)
       — produces side-by-side hue-ring plots cogito vs SD.

Outputs:
    runs/COLOR_SD_UNET_MID/auto_exp_77/
        verdict.json
        hue_ring_sd.png
        cross_modality.png      (only if cogito z_locked is on disk)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SD_DIR = Path("runs/COLOR_SD_UNET_MID")
OUT_DIR = SD_DIR / "auto_exp_77"

COGITO_LATENT_CANDIDATES = [
    Path("runs/COLOR_COGITO_L40/manifold_sae/z_locked.npy"),
    Path("runs/COGITO_L40_MANIFOLD_F512/z_locked.npy"),
    Path("runs/sae_comparison/manifold/z_locked.npy"),
]


# -----------------------------------------------------------------------------
# Circular hue helpers
# -----------------------------------------------------------------------------
def rgb_to_hue_radians(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0, 255] or [0, 1] → hue in radians [0, 2π)."""
    import colorsys
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    out = np.empty(rgb.shape[0], dtype=np.float64)
    for i, (r, g, b) in enumerate(rgb):
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        out[i] = h * 2.0 * np.pi
    return out


def circular_correlation(theta: np.ndarray, phi: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation between two angle vectors.

    Returns a value in [-1, +1].
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    tbar = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    pbar = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())
    num = np.sum(np.sin(theta - tbar) * np.sin(phi - pbar))
    den = np.sqrt(np.sum(np.sin(theta - tbar) ** 2) *
                  np.sum(np.sin(phi - pbar) ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def best_atom_hue_corr(positions: np.ndarray, hues: np.ndarray) -> dict:
    """positions: (n_prompts, F) in [0, 1] (locked-mode θ).  hues: (n_prompts,) rad.

    For each atom, treat its position as a 2π-scaled angle and compute the
    circular correlation with hue. Return the best (max |corr|) atom + the
    distribution.
    """
    n, F = positions.shape
    corrs = np.zeros(F, dtype=np.float64)
    for k in range(F):
        # Skip dead atoms (constant column).
        col = positions[:, k]
        if (col.max() - col.min()) < 1e-4:
            corrs[k] = 0.0
            continue
        theta = (col - col.min()) / max(col.max() - col.min(), 1e-12)
        theta = theta * 2.0 * np.pi
        corrs[k] = circular_correlation(theta, hues)
    abs_c = np.abs(corrs)
    k_star = int(np.argmax(abs_c))
    return dict(
        per_atom=corrs.tolist(),
        best_abs=float(abs_c[k_star]),
        best_signed=float(corrs[k_star]),
        best_atom=k_star,
        top5=[int(i) for i in np.argsort(-abs_c)[:5]],
    )


# -----------------------------------------------------------------------------
# Train wrapper (uses the trainer module)
# -----------------------------------------------------------------------------
def ensure_trained(retrain: bool) -> dict:
    out = SD_DIR / "manifold_sae" / "train_log.json"
    if out.exists() and not retrain:
        print(f"[auto_exp_77] reusing trained SAE at {out}", flush=True)
        return json.loads(out.read_text())
    from manifold_sae.diffusion.train_sd_manifold_sae import TrainConfig, train
    cfg = TrainConfig()
    return train(cfg)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip_train", action="store_true",
                    help="Assume runs/COLOR_SD_UNET_MID/manifold_sae/ exists; "
                         "just run the hue-ring metric on z_locked.npy.")
    args = ap.parse_args()

    if not (SD_DIR / "X.npy").exists():
        print(f"[auto_exp_77] MISSING {SD_DIR/'X.npy'} — run "
              "scripts/harvest_sd_unet.py first.", flush=True)
        return 2

    meta = json.loads((SD_DIR / "meta.json").read_text())
    colors = meta["colors"]  # list of [name, [r, g, b]]
    n_colors = meta["n_colors"]
    n_templates = meta["n_templates"]

    # Row layout: row = color_idx * n_templates + template_idx.
    rgbs = np.array([rgb for (_, rgb) in colors], dtype=np.float64)   # (n_colors, 3)
    hues_per_color = rgb_to_hue_radians(rgbs)                          # (n_colors,)
    hues = np.repeat(hues_per_color, n_templates)                      # (n_prompts,)

    if not args.skip_train:
        summary = ensure_trained(args.retrain)
        sd_r2 = summary.get("full_val_r2", summary.get("last_val_r2"))
    else:
        sd_r2 = None

    z_path = SD_DIR / "manifold_sae" / "z_locked.npy"
    if not z_path.exists():
        print(f"[auto_exp_77] no z_locked at {z_path}; cannot score.", flush=True)
        return 3
    z_sd = np.load(z_path)
    if z_sd.shape[0] != hues.shape[0]:
        # If harvest used a smaller n_colors than meta suggests, trim:
        m = min(z_sd.shape[0], hues.shape[0])
        z_sd, hues = z_sd[:m], hues[:m]

    sd_result = best_atom_hue_corr(z_sd, hues)
    sd_corr = sd_result["best_abs"]
    verdict = "MODALITY-UNIVERSAL" if sd_corr >= 0.30 else "TEXT-LLM-SPECIFIC"
    print(f"\n[auto_exp_77] SD hue-ring |circ_corr|={sd_corr:.3f}  "
          f"(best atom {sd_result['best_atom']}, signed={sd_result['best_signed']:+.3f})",
          flush=True)
    print(f"[auto_exp_77] VERDICT: {verdict}  "
          f"(threshold 0.30; cogito-L40 baseline ~0.6+)", flush=True)

    # ---- Optional cogito comparison from disk.
    cogito_corr = None
    cogito_z_path = None
    for cand in COGITO_LATENT_CANDIDATES:
        if cand.exists():
            cogito_z_path = cand
            break
    if cogito_z_path is not None:
        try:
            z_cog = np.load(cogito_z_path)
            # Try cogito harvest meta to get its color row count + template count.
            cogito_meta_path = cogito_z_path.parent.parent / "meta.json"
            if cogito_meta_path.exists():
                cmeta = json.loads(cogito_meta_path.read_text())
                cog_colors = cmeta.get("colors")
                cog_n_tmpl = cmeta.get("n_templates", 28)
                if cog_colors is not None:
                    cog_rgbs = np.array([c[1] for c in cog_colors], dtype=np.float64)
                    cog_hues_per = rgb_to_hue_radians(cog_rgbs)
                    cog_hues = np.repeat(cog_hues_per, cog_n_tmpl)
                    m = min(z_cog.shape[0], cog_hues.shape[0])
                    cog_res = best_atom_hue_corr(z_cog[:m], cog_hues[:m])
                    cogito_corr = cog_res["best_abs"]
        except Exception as e:
            print(f"[auto_exp_77] cogito comparison failed: {e}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(
        sd_model=meta.get("model_name"),
        hook_block=meta.get("hook_block"),
        n_prompts=int(z_sd.shape[0]),
        n_colors=int(n_colors),
        n_templates=int(n_templates),
        sd_full_val_r2=sd_r2,
        sd_hue_circ_corr_abs=sd_corr,
        sd_hue_circ_corr_signed=sd_result["best_signed"],
        sd_best_atom=sd_result["best_atom"],
        sd_top5_atoms=sd_result["top5"],
        cogito_hue_circ_corr_abs=cogito_corr,
        cogito_z_path=str(cogito_z_path) if cogito_z_path else None,
        verdict=verdict,
        threshold=0.30,
    )
    (OUT_DIR / "verdict.json").write_text(json.dumps(out, indent=2))
    print(f"[auto_exp_77] wrote {OUT_DIR/'verdict.json'}", flush=True)

    # ---- Plots.
    try:
        import matplotlib.pyplot as plt
        k = sd_result["best_atom"]
        col = z_sd[:, k]
        theta = (col - col.min()) / max(col.max() - col.min(), 1e-12) * 2 * np.pi
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
        rgb_clip = np.clip(rgbs / 255.0, 0, 1)
        rgb_per_prompt = np.repeat(rgb_clip, n_templates, axis=0)[:theta.size]
        ax.scatter(theta, np.ones_like(theta), c=rgb_per_prompt, s=18,
                   edgecolor="black", linewidth=0.2, alpha=0.85)
        ax.set_title(f"SD UNet hue ring (atom {k})\n"
                     f"|circ_corr|={sd_corr:.3f}   verdict: {verdict}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "hue_ring_sd.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[auto_exp_77] wrote {OUT_DIR/'hue_ring_sd.png'}", flush=True)
    except Exception as e:
        print(f"[auto_exp_77] plot failed: {e}", flush=True)

    # ---- Cross-modality atlas (if cogito z available).
    if cogito_corr is not None:
        try:
            from manifold_sae.diffusion.cross_modality_atlas import side_by_side_hue_ring
            side_by_side_hue_ring(
                z_sd=z_sd, hues_sd=hues, rgb_sd=np.repeat(rgbs/255.0, n_templates, axis=0)[:z_sd.shape[0]],
                z_cog=z_cog, cogito_meta_path=cogito_meta_path,
                out_path=OUT_DIR / "cross_modality.png",
            )
        except Exception as e:
            print(f"[auto_exp_77] cross_modality plot failed: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
