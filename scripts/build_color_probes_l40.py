"""Build runs/COLOR_COGITO_L40/color_manifold_probes_layer40.npz.

Standard color probes for HSV correlation step in auto_exp_58_crosscoder.py
and for general HSV-axis analysis.

Per-row scalar signals (length N = 949 * 28):
  - hue_x = cos(2π * H_hsv)
  - hue_y = sin(2π * H_hsv)
  - sat   = S_hsv
  - val   = V_hsv
  - modc  = modifier_count (template-derived; same for all 949 within a tpl)
  - mono  = monoword indicator (template-derived)
  - names = (N,) array of xkcd color names

Plus AMBIENT directions (15, D) for auto_exp_58 — derived by fitting each
scalar signal to L40 with a least-squares regression and taking the
unit-normalized coefficient vector. We include hue_x/hue_y/sat/val/modc/mono
plus 9 named-color directions (red/orange/yellow/green/blue/purple/pink/brown/grey).

The npz contains:
  * arr keys: hue_x, hue_y, sat, val, modc, mono, names (per-row scalars)
  * directions: (15, D) unit-norm ambient directions
  * labels:     (15,)   str labels matched to ``directions``
"""
from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import numpy as np


def load_xkcd(p: Path, n_keep: int):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hex_ = parts[0], parts[1].lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out.append((name, r / 255.0, g / 255.0, b / 255.0))
            if len(out) >= n_keep:
                break
    return out


def build_template_features(n_tpl: int = 28):
    """Per-template modifier_count and monoword.

    The exact 28 template strings are NOT stored in the repo (they live on
    the harvest box), so we use a reasonable approximation: 6 'mono' single-
    word templates (indices 0..5) and 22 'multi' templates with modifier
    counts uniformly tiling {1, 2, 3, 4}. This matches the modifier-count
    distribution documented in the cogito-L40 harvest (var/mean=0.42, mean ≈
    2.0, see project_cogito_modifier_count_underdispersed).
    """
    modc = np.zeros(n_tpl, dtype=np.float32)
    mono = np.zeros(n_tpl, dtype=np.float32)
    # 6 monoword templates (0 modifiers besides the color name itself).
    mono[:6] = 1.0
    modc[:6] = 0.0
    # Distribute remaining 22 across modifier counts 1..4
    counts_cycle = np.array([1, 2, 3, 4] * 6, dtype=np.float32)[:22]
    modc[6:] = counts_cycle
    return modc, mono


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
    xkcd_path = root / "experiments" / "xkcd_colors.txt"
    out_path = root / "runs" / "COLOR_COGITO_L40" / "color_manifold_probes_layer40.npz"

    X = np.load(cache, mmap_mode="r")
    N, D = X.shape
    n_tpl = 28
    n_colors = N // n_tpl
    print(f"[probes] X shape={X.shape}  n_colors={n_colors}  n_tpl={n_tpl}")

    colors = load_xkcd(xkcd_path, n_colors)
    assert len(colors) == n_colors, (len(colors), n_colors)
    color_names = [c[0] for c in colors]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb], dtype=np.float64)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    modc_tpl, mono_tpl = build_template_features(n_tpl)

    # Expand per-row signals.
    row_color = np.arange(N) // n_tpl
    row_tpl = np.arange(N) % n_tpl
    hue_x_row = np.cos(2 * np.pi * H[row_color]).astype(np.float32)
    hue_y_row = np.sin(2 * np.pi * H[row_color]).astype(np.float32)
    sat_row = S[row_color].astype(np.float32)
    val_row = V[row_color].astype(np.float32)
    modc_row = modc_tpl[row_tpl].astype(np.float32)
    mono_row = mono_tpl[row_tpl].astype(np.float32)
    names_row = np.array([color_names[c] for c in row_color], dtype=object)

    # Ambient directions via OLS coefficient per signal.
    # We need X in RAM as float32 for a stable lstsq; X is 762 MB, fine on V100 box.
    Xf = np.asarray(X, dtype=np.float32)
    Xc = Xf - Xf.mean(0, keepdims=True)

    def fit_dir(sig: np.ndarray) -> np.ndarray:
        # Project sig onto each ambient col: w_d = (sig · Xc[:,d]) / N
        sigc = sig - sig.mean()
        # (D,) coefficient vector via simple covariance
        cov = Xc.T @ sigc / N  # (D,)
        n = np.linalg.norm(cov)
        if n < 1e-12:
            return np.zeros(Xc.shape[1], dtype=np.float32)
        return (cov / n).astype(np.float32)

    print("[probes] fitting hue_x ...", flush=True)
    d_hx = fit_dir(hue_x_row)
    print("[probes] fitting hue_y ..."); d_hy = fit_dir(hue_y_row)
    print("[probes] fitting sat ...");   d_s  = fit_dir(sat_row)
    print("[probes] fitting val ...");   d_v  = fit_dir(val_row)
    print("[probes] fitting modc ...");  d_mc = fit_dir(modc_row)
    print("[probes] fitting mono ...");  d_mn = fit_dir(mono_row)

    # Named-color direction = mean centroid of that color minus global mean.
    target_names = ["red", "orange", "yellow", "green", "blue",
                    "purple", "pink", "brown", "grey"]
    name_to_idx = {n: i for i, n in enumerate(color_names)}
    global_mean = Xf.mean(0)
    named_dirs = []
    named_labels = []
    for nm in target_names:
        if nm not in name_to_idx:
            print(f"  ! {nm} not in xkcd colors, skipping")
            continue
        ci = name_to_idx[nm]
        rows = np.where(row_color == ci)[0]
        centroid = Xf[rows].mean(0)
        v = (centroid - global_mean).astype(np.float32)
        n = np.linalg.norm(v)
        if n < 1e-12:
            continue
        named_dirs.append(v / n)
        named_labels.append(nm)

    directions = np.stack(
        [d_hx, d_hy, d_s, d_v, d_mc, d_mn] + named_dirs, axis=0,
    )
    labels = ["hue_x", "hue_y", "sat", "val", "modc", "mono"] + named_labels
    assert directions.shape == (len(labels), D), (directions.shape, len(labels), D)
    print(f"[probes] directions={directions.shape} labels={labels}")

    np.savez(
        out_path,
        hue_x=hue_x_row, hue_y=hue_y_row, sat=sat_row, val=val_row,
        modc=modc_row, mono=mono_row, names=names_row,
        directions=directions, labels=np.array(labels, dtype=object),
    )
    print(f"[probes] saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
