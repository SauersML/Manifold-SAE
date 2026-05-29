"""
auto_51 — Centroid trajectories in U_3d for "purple", "blue", "violet" families.

Idea (kkkkkk): Plot U_3d's predicted centroid trajectory for the three semantically
adjacent color families {"purple", "blue", "violet"}. Do these similar colors
share a latent neighborhood in the GAM's unsupervised d=3 embedding?

For each family we collect all xkcd color names that contain the family token
(case-insensitive whole-word match), look up their U_3d coordinates from the
GAM's unsupervised d=3 fit (stored as 'T' under per_layer.L40.unsupervised_full_data),
sort them along an intrinsic 1-D order (lightness V from HSV) to define a
*trajectory*, and plot:

  - all 949 colors as a faint grey background scatter in PCA(U_3d -> 2D),
  - per family: a smoothed trajectory of the family members ordered by V,
    coloured at each node by the member's actual RGB,
  - per family: a star marker at the centroid of the family,
  - per family: a hollow ring at the *canonical* single-word color
    (xkcd "purple", "blue", "violet"),
  - inter-family centroid distances printed in the title and saved to JSON.

Constraints respected: no Gaussian RBF; no length_scale on Duchon (we don't
fit any Duchon here — we read the pre-fit d=3 manifold T from results.json
and use PCA for a 2-D viewport).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RUN_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS  = RUN_DIR / "results.json"
XKCD     = Path("/Users/user/Manifold-SAE/experiments/xkcd_colors.txt")
OUT_PNG  = RUN_DIR / "auto_51.png"
OUT_JSON = RUN_DIR / "auto_51.json"

FAMILIES = ["purple", "blue", "violet"]
FAM_PLOT_COLOR = {"purple": "#7e1e9c", "blue": "#0343df", "violet": "#9a0eea"}


def load_xkcd_names() -> list[str]:
    names = []
    for line in XKCD.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # tab-separated: "name\t#rrggbb\t"
        name = line.split("\t", 1)[0].strip()
        names.append(name)
    return names


def hsv_v(R: float, G: float, B: float) -> float:
    return max(R, G, B)


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    T = np.asarray(pl["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)
    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    n_c = Rax.size
    assert T.shape == (n_c, 3), (T.shape, n_c)

    names = load_xkcd_names()
    assert len(names) == n_c, (len(names), n_c)

    # PCA projection (T already low-dim, but we want a 2-D viewport).
    pca = PCA(n_components=2, random_state=0)
    T2 = pca.fit_transform(T)
    print(f"[pca] explained_variance_ratio_2d = {pca.explained_variance_ratio_}", flush=True)

    # Identify family membership: token match (word-boundary) so that
    # "blue" matches "blue", "navy blue", "blue/green" but not "blueberry".
    fam_idx: dict[str, list[int]] = {f: [] for f in FAMILIES}
    fam_re = {f: re.compile(rf"(?<![a-z]){re.escape(f)}(?![a-z])", re.I) for f in FAMILIES}
    canon_idx: dict[str, int | None] = {f: None for f in FAMILIES}
    for i, nm in enumerate(names):
        nm_l = nm.lower()
        for f in FAMILIES:
            if fam_re[f].search(nm_l):
                fam_idx[f].append(i)
            if nm_l == f:
                canon_idx[f] = i
    for f in FAMILIES:
        print(f"[family] {f}: {len(fam_idx[f])} members, canonical idx={canon_idx[f]}", flush=True)

    # Per-family centroid in 3D and pairwise distances.
    centroids_3d = {f: T[fam_idx[f]].mean(axis=0) for f in FAMILIES}
    pair_d: dict[str, float] = {}
    for i, fa in enumerate(FAMILIES):
        for fb in FAMILIES[i + 1:]:
            d = float(np.linalg.norm(centroids_3d[fa] - centroids_3d[fb]))
            pair_d[f"{fa}-{fb}"] = d
    # Compare to mean inter-color distance for scale.
    rng = np.random.default_rng(0)
    samp = rng.choice(n_c, size=4000, replace=True)
    diffs = T[samp[:2000]] - T[samp[2000:]]
    typical_d = float(np.linalg.norm(diffs, axis=1).mean())
    print(f"[geom] pair distances (3D)={pair_d}; typical random-pair d={typical_d:.3f}", flush=True)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    ax_main, ax_zoom = axes

    # Background: all 949 colors at their U_3d->PC2 positions, painted by RGB.
    bg_rgb = np.clip(np.column_stack([Rax, Gax, Bax]), 0, 1)
    for ax in (ax_main, ax_zoom):
        ax.scatter(T2[:, 0], T2[:, 1], s=10, c=bg_rgb, alpha=0.35,
                   edgecolors="none")

    # Family trajectories: order each family by HSV.V and connect.
    traj_summary: dict[str, list[dict]] = {}
    for f in FAMILIES:
        idxs = np.asarray(fam_idx[f])
        vs = np.array([hsv_v(Rax[i], Gax[i], Bax[i]) for i in idxs])
        order = np.argsort(vs)  # dark -> light
        idxs_o = idxs[order]
        xy = T2[idxs_o]
        node_rgb = bg_rgb[idxs_o]
        col = FAM_PLOT_COLOR[f]
        # Line in family colour, nodes painted by their own RGB.
        for ax in (ax_main, ax_zoom):
            ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=1.6, alpha=0.85,
                    label=f"{f} trajectory (n={len(idxs)})")
            ax.scatter(xy[:, 0], xy[:, 1], s=55, c=node_rgb,
                       edgecolors=col, linewidths=1.2, zorder=3)
        # Centroid (in 2D-projected space).
        c2 = T2[idxs].mean(axis=0)
        for ax in (ax_main, ax_zoom):
            ax.scatter([c2[0]], [c2[1]], marker="*", s=420,
                       facecolors=col, edgecolors="black", linewidths=1.2,
                       zorder=5)
        # Canonical single-word color: hollow ring.
        ci = canon_idx[f]
        if ci is not None:
            xy_c = T2[ci]
            for ax in (ax_main, ax_zoom):
                ax.scatter([xy_c[0]], [xy_c[1]], marker="o", s=260,
                           facecolors="none", edgecolors=col, linewidths=2.6,
                           zorder=6)
                ax.annotate(f, xy=xy_c, xytext=(8, 8),
                            textcoords="offset points", fontsize=11,
                            color=col, fontweight="bold")
        traj_summary[f] = [
            {"name": names[int(i)], "v": float(v), "u2d": [float(x) for x in T2[int(i)]]}
            for i, v in zip(idxs_o, vs[order])
        ]

    # Zoom plot: tight bounds around union of family points.
    union = np.concatenate([np.asarray(fam_idx[f]) for f in FAMILIES])
    xy_u = T2[union]
    pad = 0.15 * (xy_u.max(0) - xy_u.min(0) + 1e-9)
    ax_zoom.set_xlim(xy_u[:, 0].min() - pad[0], xy_u[:, 0].max() + pad[0])
    ax_zoom.set_ylim(xy_u[:, 1].min() - pad[1], xy_u[:, 1].max() + pad[1])

    evr = pca.explained_variance_ratio_
    ax_main.set_title(
        "U_3d centroid trajectories: purple / blue / violet  (full view)\n"
        f"PC1 EVR={evr[0]:.2f}  PC2 EVR={evr[1]:.2f}  "
        f"centroid pair-dist: " + ", ".join(f"{k}={v:.2f}" for k, v in pair_d.items())
        + f"  (typical random-pair={typical_d:.2f})"
    )
    ax_zoom.set_title("zoom: family neighbourhood (nodes ordered dark to light)")
    for ax in (ax_main, ax_zoom):
        ax.set_xlabel("U_3d  PC1")
        ax.set_ylabel("U_3d  PC2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "families": FAMILIES,
        "n_members": {f: len(fam_idx[f]) for f in FAMILIES},
        "canonical_index": {f: canon_idx[f] for f in FAMILIES},
        "centroids_3d": {f: centroids_3d[f].tolist() for f in FAMILIES},
        "pair_distances_3d": pair_d,
        "typical_random_pair_distance_3d": typical_d,
        "pca_evr_2d": evr.tolist(),
        "trajectories": traj_summary,
    }, indent=2))
    print(f"[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
