"""Smoke test: fit GaugeFix on cached cogito-L40 + xkcd-HSV labels and
assert that hue is recovered.

If the cached harvest is missing, the test falls back to a synthetic
manifold so the package always installs cleanly.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

import numpy as np
import pytest

from concept_manifold_steering import GaugeFix, ManifoldSteerer
from concept_manifold_steering.diagnostics import (
    per_anchor_curvature,
    null_topology_control,
)


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
XKCD = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
N_TEMPLATES = 28  # 26572 / 949


def _load_xkcd() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    pat = re.compile(r"^(.+?)\t(#[0-9a-fA-F]{6})")
    for line in XKCD.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = pat.match(line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def _hsv_from_hex(hx: str) -> tuple[float, float, float]:
    r = int(hx[1:3], 16) / 255.0
    g = int(hx[3:5], 16) / 255.0
    b = int(hx[5:7], 16) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h, s, v


def _build_real_labels(N: int):
    """Build per-row HSV labels by replicating each color across the 28
    templates -- matches the cogito harvest layout (auto memory)."""
    colors = _load_xkcd()
    n_colors = N // N_TEMPLATES
    colors = colors[:n_colors]
    hues, sats, vals, names = [], [], [], []
    for name, hx in colors:
        h, s, v = _hsv_from_hex(hx)
        for _ in range(N_TEMPLATES):
            hues.append(h)
            sats.append(s)
            vals.append(v)
            names.append(name)
    extra = N - len(hues)
    if extra > 0:
        # pad with last color
        hues  += [hues[-1]] * extra
        sats  += [sats[-1]] * extra
        vals  += [vals[-1]] * extra
        names += [names[-1]] * extra
    return (np.asarray(hues, np.float32),
            np.asarray(sats, np.float32),
            np.asarray(vals, np.float32),
            np.asarray(names))


def _synthetic_fallback(seed: int = 0):
    """Tiny synthetic 3-axis color manifold so the suite passes without
    the cogito harvest."""
    rng = np.random.default_rng(seed)
    N, p, d = 300, 64, 3
    Z = rng.uniform(0, 1, size=(N, d)).astype(np.float32)  # h, s, v
    W = rng.standard_normal((d, p)).astype(np.float32)
    X = Z @ W + 0.05 * rng.standard_normal((N, p)).astype(np.float32)
    labels = {"hue": Z[:, 0], "saturation": Z[:, 1], "value": Z[:, 2]}
    return X, labels


def test_gauge_recovers_hue():
    if HARVEST.exists() and XKCD.exists():
        X = np.load(HARVEST, mmap_mode="r")
        # Subsample for the smoke test: every 7th row -> ~3800 rows, keeps
        # both color and template coverage.
        idx = np.arange(0, X.shape[0], 7)
        X = np.asarray(X[idx], dtype=np.float32)
        h, s, v, _ = _build_real_labels(X.shape[0])
        # We subsampled, so trim labels to match (built for full N then sliced)
        # Easier: just rebuild for N.
        # Replicate logic: align by colors.
        # _build_real_labels assumes contiguous template-replication, but our
        # subsample broke that — instead rebuild labels for the subsampled rows.
        full_h, full_s, full_v, full_n = _build_real_labels(26572)
        # Clamp idx to label length.
        idx = idx[idx < len(full_h)]
        X = X[: len(idx)]
        h, s, v = full_h[idx], full_s[idx], full_v[idx]
        labels = {"hue": h, "saturation": s, "value": v}
        src = "cogito-L40-cached"
    else:
        X, labels = _synthetic_fallback()
        src = "synthetic-fallback"

    gauge = GaugeFix(targets=["hue", "saturation", "value"], K=64).fit(X, labels)
    r2 = gauge.r2()
    print(f"[smoke:{src}] R² = {r2}")
    assert "hue" in r2
    assert r2["hue"] > 0.5, f"hue R²={r2['hue']:.3f} < 0.5 ({src})"
    # Transform smoke
    Z = gauge.transform(X[:10])
    assert Z.shape == (10, gauge.d)
    # Free axes available
    free = gauge.free_axes()
    assert free.shape[0] == X.shape[1]


def test_steer_dry_run_payload():
    X, labels = _synthetic_fallback()
    gauge = GaugeFix(targets=["hue", "saturation", "value"], K=16).fit(
        X, labels,
        anchor_labels={"red": [0, 1, 2], "blue": [50, 51, 52]},
    )
    steerer = ManifoldSteerer(gauge, server_url="", layer=40, model="dummy")
    res = steerer.steer("Sky is", concept="red", alpha=1.5, dry_run=True)
    assert res.intervened is False
    assert "interventions" in res.request["extra_body"]
    iv = res.request["extra_body"]["interventions"][0]
    assert iv["layer"] == 40
    assert iv["scale"] != 0


def test_diagnostics_smoke():
    X, labels = _synthetic_fallback()
    gauge = GaugeFix(targets=["hue", "saturation"], K=16).fit(
        X, labels,
        anchor_labels={"A": [0, 1, 2, 3], "B": [100, 101, 102, 103]},
    )
    curv = per_anchor_curvature(gauge, X, {"A": [0, 1, 2, 3], "B": [100, 101, 102, 103]})
    assert set(curv) == {"A", "B"}
    null = null_topology_control(gauge, X, labels, n_perm=20)
    assert "hue" in null
    assert "null_p" in null["hue"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
