"""Real gamfit Duchon color curves across OLMo training checkpoints.

This intentionally separates two different claims:

1. ``hue_periodic`` is a real CLOSED LOOP, but its circular coordinate is the
   external hue coordinate. It is not presented as gamfit discovering the order.

2. ``learned_latent`` lets gamfit choose the 1-D latent coordinate with
   ``gaussian_reml_optimize_latent(..., manifold="circle")``. On gamfit 0.1.180
   this returns a reproducible non-periodic Duchon decoder but reports
   ``converged=False``; see SauersML/gam#879. It is therefore a diagnostic
   no-forced-order curve, not a trustworthy closed loop.

Every displayed state is an actual chronological checkpoint from the extracted
``/tmp/colall/*.npz`` files. The video is stepwise over checkpoints: no geometry
interpolation, no UMAP, no nearest-color tour, no nearest-rep tour.
"""
from __future__ import annotations

import argparse
import colorsys
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import gamfit
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np


CORE_RGB = {
    "red": (229, 0, 0),
    "orange": (249, 115, 6),
    "yellow": (255, 255, 20),
    "green": (21, 176, 26),
    "cyan": (0, 255, 255),
    "blue": (3, 67, 223),
    "indigo": (56, 2, 130),
    "violet": (154, 14, 234),
    "purple": (126, 30, 156),
    "magenta": (194, 0, 120),
    "pink": (255, 129, 192),
    "crimson": (140, 0, 15),
    "maroon": (101, 0, 33),
    "navy": (1, 21, 62),
    "olive": (110, 117, 14),
    "teal": (2, 147, 134),
    "turquoise": (6, 194, 172),
    "lavender": (199, 159, 239),
    "gold": (219, 180, 12),
    "coral": (252, 90, 80),
    "mint": (159, 254, 176),
}


@dataclass
class Checkpoint:
    path: str
    label: str
    stage: str
    order: tuple[int, int, str]
    points: np.ndarray
    rgb: np.ndarray
    hue_curve: np.ndarray
    learned_curve: np.ndarray | None
    hue_r2: float
    hue_edf: float
    hue_lambda: float
    learned_r2: float | None
    learned_edf: float | None
    learned_lambda: float | None
    learned_converged: bool | None
    learned_grad_t_norm: float | None


def rgb_key(rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(round(x)) for x in rgb)


def hue01(rgb: np.ndarray) -> float:
    return float(colorsys.rgb_to_hsv(*(rgb / 255.0))[0])


def sort_key(path: str) -> tuple[int, int, str]:
    stem = Path(path).stem
    clean = re.sub(r"^\d+__", "", stem)
    if "stage1-step" in clean:
        return (0, int(re.search(r"stage1-step(\d+)", clean).group(1)), clean)
    if "stage2-ingredient1-step" in clean:
        return (1, int(re.search(r"stage2-ingredient1-step(\d+)", clean).group(1)), clean)
    if "stage3-step" in clean:
        return (2, int(re.search(r"stage3-step(\d+)", clean).group(1)), clean)
    if "TRAJ_SFT" in clean:
        return (3, int(re.search(r"step(\d+)", clean).group(1)), clean)
    if "TRAJ_DPO" in clean:
        return (4, 0, clean)
    if "TRAJ_RL__step_" in clean:
        return (5, int(re.search(r"step_(\d+)", clean).group(1)), clean)
    if "TRAJ_RL31__step_" in clean:
        return (6, int(re.search(r"step_(\d+)", clean).group(1)), clean)
    return (99, 0, clean)


def stage_from_order(order: tuple[int, int, str]) -> str:
    return ["pretrain-1", "pretrain-2", "pretrain-3", "SFT", "DPO", "RL3.0", "RL3.1"][min(order[0], 6)]


def label_from_order(order: tuple[int, int, str]) -> str:
    stage = stage_from_order(order)
    return f"{stage} step {order[1]}" if stage != "DPO" else "DPO"


def load_color_means(path: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    z = np.load(path)
    vectors = np.asarray(z["V"], dtype=np.float64)
    rgbs = np.asarray(z["rgb"], dtype=np.float64)
    lookup = {value: name for name, value in CORE_RGB.items()}
    keep = []
    names = []
    for i, rgb in enumerate(rgbs):
        name = lookup.get(rgb_key(rgb))
        if name is not None:
            keep.append(i)
            names.append(name)
    if len(keep) != len(CORE_RGB):
        raise RuntimeError(f"{path}: expected {len(CORE_RGB)} core colors, got {len(keep)}")
    by_name = {name: idx for name, idx in zip(names, keep, strict=True)}
    ordered_idx = [by_name[name] for name in CORE_RGB]
    return vectors[ordered_idx], list(CORE_RGB), rgbs[ordered_idx]


def periodic_duchon_basis(t: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.asarray(
        gamfit.duchon_basis(
            np.asarray(t, dtype=np.float64).reshape(-1, 1),
            np.asarray(centers, dtype=np.float64).reshape(-1, 1),
            m=2,
            periodic_per_axis=[1.0],
        )
    )


def plain_duchon_basis(t: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.asarray(
        gamfit.duchon_basis(
            np.asarray(t, dtype=np.float64).reshape(-1, 1),
            np.asarray(centers, dtype=np.float64).reshape(-1, 1),
            m=2,
        )
    )


def fit_block_basis(basis: np.ndarray, penalty: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coefs = []
    edfs = []
    lambdas = []
    for dim in range(y.shape[1]):
        result = gamfit.gaussian_reml_fit_blocks_forward([basis], [penalty], y=y[:, dim])
        coef = np.asarray(result["coefficients"], dtype=np.float64).ravel()[: basis.shape[1]]
        coefs.append(coef)
        edfs.append(float(np.asarray(result["edf"]).ravel()[0]))
        lambdas.append(float(np.asarray(result["lambdas"]).ravel()[0]))
    return np.stack(coefs, axis=1), float(np.mean(edfs)), float(np.median(lambdas))


def r2(y: np.ndarray, fitted: np.ndarray) -> float:
    den = float(((y - y.mean(axis=0)) ** 2).sum())
    return float(1 - ((y - fitted) ** 2).sum() / den) if den > 0 else float("nan")


def fit_hue_periodic_loop(points: np.ndarray, rgbs: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    t = np.array([hue01(rgb) for rgb in rgbs])
    centers = np.sort(t)
    basis = periodic_duchon_basis(t, centers)
    coef, edf, lam = fit_block_basis(basis, np.eye(basis.shape[1]), points)
    grid = np.linspace(0, 1, 360)
    curve = periodic_duchon_basis(grid, centers) @ coef
    return curve, r2(points, basis @ coef), edf, lam


def fit_learned_latent_curve(points: np.ndarray) -> tuple[np.ndarray | None, float | None, float | None, float | None, bool | None, float | None]:
    # Initial angle is only an initialization. gamfit then optimizes t; convergence
    # status is reported and preserved in diagnostics.
    t0 = ((np.arctan2(points[:, 1], points[:, 0]) / (2 * np.pi)) % 1).astype(np.float64)
    centers = np.linspace(0, 1, 14, dtype=np.float64).reshape(-1, 1)
    try:
        result = gamfit.gaussian_reml_optimize_latent(
            y=points.astype(np.float64),
            n_obs=len(points),
            latent_dim=1,
            centers=centers,
            penalty=np.eye(len(centers)),
            m=2,
            manifold="circle",
            basis_kind="duchon",
            init="caller",
            t=t0,
            max_iter=120,
            n_restarts=1,
            seed=0,
        )
    except Exception:
        return None, None, None, None, None, None
    t = np.asarray(result["t"], dtype=np.float64).reshape(-1, 1)
    coef = np.asarray(result["coefficients"], dtype=np.float64)
    fitted = np.asarray(result["fitted"], dtype=np.float64)
    order = np.argsort(t[:, 0])
    lo, hi = float(t[:, 0].min()), float(t[:, 0].max())
    grid = np.linspace(lo, hi, 360).reshape(-1, 1)
    curve = plain_duchon_basis(grid, centers) @ coef
    # Verify the dense decoder is the same basis family as the fitted values.
    repro = float(np.max(np.abs(plain_duchon_basis(t, centers) @ coef - fitted)))
    if not np.isfinite(repro) or repro > 1e-5 or len(order) < 3:
        return None, None, None, None, bool(result.get("converged", False)), float(result.get("grad_t_norm", math.nan))
    return (
        curve,
        r2(points, fitted),
        float(result.get("edf", math.nan)),
        float(result.get("lambda", math.nan)),
        bool(result.get("converged", False)),
        float(result.get("grad_t_norm", math.nan)),
    )


def build_states(files: list[str]) -> list[Checkpoint]:
    raw = [(path, *load_color_means(path)) for path in files]
    final_vectors = raw[-1][1]
    final_centered = final_vectors - final_vectors.mean(axis=0)
    display_basis = np.linalg.svd(final_centered, full_matrices=False)[2][:3]

    states = []
    for path, vectors, _names, rgbs in raw:
        order = sort_key(path)
        points = (vectors - vectors.mean(axis=0)) @ display_basis.T
        hue_curve, hue_score, hue_edf, hue_lam = fit_hue_periodic_loop(points, rgbs)
        learned_curve, learned_score, learned_edf, learned_lam, learned_conv, learned_grad = fit_learned_latent_curve(points)
        states.append(
            Checkpoint(
                path=path,
                label=label_from_order(order),
                stage=stage_from_order(order),
                order=order,
                points=points,
                rgb=rgbs / 255.0,
                hue_curve=hue_curve,
                learned_curve=learned_curve,
                hue_r2=hue_score,
                hue_edf=hue_edf,
                hue_lambda=hue_lam,
                learned_r2=learned_score,
                learned_edf=learned_edf,
                learned_lambda=learned_lam,
                learned_converged=learned_conv,
                learned_grad_t_norm=learned_grad,
            )
        )
    return states


def shared_bounds(states: list[Checkpoint]) -> tuple[np.ndarray, float]:
    arrays = []
    for state in states:
        arrays.append(state.points)
        arrays.append(state.hue_curve)
        if state.learned_curve is not None:
            arrays.append(state.learned_curve)
    all_points = np.concatenate(arrays, axis=0)
    center = all_points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_points - center, axis=1)) * 1.15)
    return center, radius


def stage_color(stage: str) -> str:
    return {
        "pretrain-1": "#7dd3fc",
        "pretrain-2": "#38bdf8",
        "pretrain-3": "#22d3ee",
        "SFT": "#a7f3d0",
        "DPO": "#fde68a",
        "RL3.0": "#fca5a5",
        "RL3.1": "#f0abfc",
    }[stage]


def render_video(states: list[Checkpoint], outdir: Path, kind: str, fps: int, seconds: float, dpi: int) -> Path:
    center, radius = shared_bounds(states)
    total_frames = int(fps * seconds)
    fig = plt.figure(figsize=(10.8, 8.8), dpi=dpi)
    fig.patch.set_facecolor("#080a0f")
    ax = fig.add_subplot(111, projection="3d")
    movie_path = outdir / f"{kind}.mp4"
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=30000,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow"],
        metadata={"title": f"OLMo color gamfit Duchon {kind}", "artist": "Manifold-SAE"},
    )
    with writer.saving(fig, str(movie_path), dpi=dpi):
        for frame_i in range(total_frames):
            progress = frame_i / max(total_frames - 1, 1)
            state_i = min(len(states) - 1, int(progress * len(states)))
            state = states[state_i]
            curve = state.hue_curve if kind == "hue_periodic_duchon_loop" else state.learned_curve
            if curve is None:
                curve = state.hue_curve * np.nan
            title = "Closed hue-periodic Duchon loop" if kind == "hue_periodic_duchon_loop" else "Gamfit-learned latent Duchon curve"
            score = state.hue_r2 if kind == "hue_periodic_duchon_loop" else state.learned_r2
            edf = state.hue_edf if kind == "hue_periodic_duchon_loop" else state.learned_edf
            status = "actual checkpoint fit"
            if kind != "hue_periodic_duchon_loop":
                status = f"actual checkpoint fit · converged={state.learned_converged}"

            ax.clear()
            ax.set_facecolor("#080a0f")
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("final color PC1", color="#aab3c5", labelpad=12)
            ax.set_ylabel("final color PC2", color="#aab3c5", labelpad=12)
            ax.set_zlabel("final color PC3", color="#aab3c5", labelpad=12)
            ax.tick_params(colors="#465166", labelsize=7, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((0.04, 0.05, 0.08, 0.18))
                axis.pane.set_edgecolor((0.60, 0.67, 0.80, 0.22))
            ax.grid(True, color="#1f2937")
            ax.view_init(elev=22 + 4 * np.sin(frame_i / 80), azim=36 + frame_i * 0.32)

            for size, alpha in [(170, 0.07), (78, 0.24), (34, 1.0)]:
                rgba = [(r, g, b, alpha) for r, g, b in state.rgb]
                ax.scatter(state.points[:, 0], state.points[:, 1], state.points[:, 2], s=size, c=rgba, depthshade=False)
            if np.isfinite(curve).all():
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=(1, 1, 1, 0.14), lw=11)
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=(1, 1, 1, 0.96), lw=3.2)

            fig.text(0.055, 0.945, title, color="#f8fafc", fontsize=24, weight="semibold")
            fig.text(
                0.055,
                0.909,
                f"{state.label}   ·   {status}   ·   R² {score if score is not None else float('nan'):.2f}   ·   EDF {edf if edf is not None else float('nan'):.1f}",
                color="#aeb7c8",
                fontsize=12,
            )
            fig.text(0.055, 0.878, f"{Path(state.path).name}", color="#64748b", fontsize=9)
            fig.patches.clear()
            fig.patches.extend(
                [
                    Rectangle((0.055, 0.055), 0.89, 0.010, transform=fig.transFigure, color=(1, 1, 1, 0.11), lw=0),
                    Rectangle((0.055, 0.055), 0.89 * progress, 0.010, transform=fig.transFigure, color=stage_color(state.stage), lw=0),
                ]
            )
            writer.grab_frame()
    plt.close(fig)
    return movie_path


def write_site(outdir: Path, states: list[Checkpoint], videos: list[Path]) -> None:
    diag = []
    for state in states:
        diag.append(
            {
                "file": state.path,
                "label": state.label,
                "stage": state.stage,
                "sort_key": list(state.order),
                "hue_periodic": {"r2": state.hue_r2, "edf": state.hue_edf, "lambda": state.hue_lambda},
                "learned_latent": {
                    "r2": state.learned_r2,
                    "edf": state.learned_edf,
                    "lambda": state.learned_lambda,
                    "converged": state.learned_converged,
                    "grad_t_norm": state.learned_grad_t_norm,
                },
            }
        )
    (outdir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{d['label']}</td><td><code>{Path(d['file']).name}</code></td>"
        f"<td>{d['hue_periodic']['r2']:.3f}</td><td>{d['learned_latent']['r2'] if d['learned_latent']['r2'] is not None else 'failed'}</td>"
        f"<td>{d['learned_latent']['converged']}</td></tr>"
        for i, d in enumerate(diag)
    )
    cards = "\n".join(
        f"""
        <section>
          <h2>{video.stem.replace('_', ' ')}</h2>
          <video controls preload="metadata" src="{video.name}"></video>
        </section>
        """
        for video in videos
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Real gamfit Duchon color fits across training</title>
<style>
body {{ margin:0; background:#080a0f; color:#e5edf8; font:15px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; }}
main {{ max-width:1160px; margin:0 auto; padding:32px 28px 60px; }}
h1 {{ font-size:30px; font-weight:650; margin:0 0 8px; }}
h2 {{ font-size:18px; font-weight:600; margin:26px 0 10px; color:#f8fafc; }}
p {{ color:#aeb7c8; max-width:920px; }}
video {{ width:100%; border:1px solid #1e293b; background:#020617; display:block; }}
code {{ color:#c4b5fd; }}
.note {{ border-left:3px solid #f59e0b; padding:10px 14px; background:#111827; color:#d7dfed; }}
table {{ width:100%; border-collapse:collapse; margin-top:22px; font-size:12px; }}
td, th {{ border-bottom:1px solid #1e293b; padding:7px 6px; text-align:left; }}
th {{ color:#93a4bd; font-weight:500; }}
</style>
<main>
  <h1>Real gamfit Duchon color fits across training</h1>
  <p>57 actual checkpoints, explicit chronological sort from pretrain stage 1 through final RL3.1. The videos are stepwise over checkpoints: no interpolated manifolds, no UMAP, no nearest-color or nearest-rep tours.</p>
  <p class="note">The closed loop uses external hue as the periodic coordinate. The no-forced-order gamfit latent fit is shown separately as a diagnostic curve because gamfit currently reports non-convergence for the latent circle optimizer; see SauersML/gam#879.</p>
  {cards}
  <h2>Checkpoint audit</h2>
  <table><thead><tr><th>#</th><th>label</th><th>source file</th><th>hue-loop R²</th><th>learned R²</th><th>learned converged</th></tr></thead><tbody>{rows}</tbody></table>
</main>
"""
    (outdir / "index.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="/tmp/colall/*.npz")
    parser.add_argument("--outdir", default="results/color_duchon_training_real")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.input_glob), key=sort_key)
    if not files:
        raise SystemExit(f"no files matched {args.input_glob}")
    states = build_states(files)
    videos = [
        render_video(states, outdir, "hue_periodic_duchon_loop", args.fps, args.seconds, args.dpi),
        render_video(states, outdir, "learned_latent_duchon_curve", args.fps, args.seconds, args.dpi),
    ]
    write_site(outdir, states, videos)
    print(f"wrote {outdir / 'index.html'}")
    print(f"chronology first={Path(states[0].path).name} last={Path(states[-1].path).name}")
    print(f"videos: {', '.join(v.name for v in videos)}")


if __name__ == "__main__":
    main()
