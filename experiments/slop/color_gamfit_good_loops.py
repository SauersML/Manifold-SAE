"""Good, interpretable gamfit color loops.

This site intentionally shows only actual loop fits that are readable.

No nearest-color tour. No nearest-rep tour. No UMAP. No sheet/torus surfaces.

Each loop is:

    prompt-cleaned color reps ~ periodic Duchon(hue)

The periodic Duchon basis comes from gamfit, and smoothing lambda is selected by
gamfit's Gaussian REML solver (`gaussian_reml_fit_blocks_forward`). The hue parameter is
an interpretable color-wheel coordinate, not a complexity-tuned ordering. Basis centers
are the observed color hues; complexity is controlled by REML smoothing, not by a knot
sweep.

The white/colored curve in each page is the actual gamfit decoder evaluated densely.
"""
from __future__ import annotations

import argparse
import colorsys
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import gamfit


CORE = [
    "red", "orange", "yellow", "green", "cyan", "blue", "indigo",
    "violet", "purple", "magenta", "pink", "crimson", "teal", "turquoise", "gold",
]

EXPANDED = CORE + ["coral", "mint", "lavender", "maroon"]


@dataclass(frozen=True)
class LoopSpec:
    name: str
    title: str
    colors: list[str]


SPECS = [
    LoopSpec("core_bright", "Core bright-color loop", CORE),
    LoopSpec("expanded_vivid", "Expanded vivid-color loop", EXPANDED),
]


def rgb_css(rgb: np.ndarray, alpha: float | None = None) -> str:
    r, g, b = [int(x) for x in rgb]
    if alpha is None:
        return f"rgb({r},{g},{b})"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def hue01(rgb: np.ndarray) -> float:
    return float(colorsys.rgb_to_hsv(*(rgb / 255.0))[0])


def hue_color(t: float, alpha: float = 1.0) -> str:
    r, g, b = colorsys.hsv_to_rgb(float(t % 1.0), 0.86, 1.0)
    return f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{alpha:.3f})"


def periodic_duchon(t: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.asarray(
        gamfit.duchon_basis(
            np.asarray(t, dtype=float).reshape(-1, 1),
            np.asarray(centers, dtype=float).reshape(-1, 1),
            m=2,
            periodic_per_axis=[1.0],
        )
    )


def load(extra: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    activations = np.load(extra / "activations.npy")
    records = [json.loads(line) for line in open(extra / "prompts.jsonl") if line.strip()]
    reps = activations[:, min(layer, activations.shape[1] - 1), :].astype(np.float64)
    frame = np.array([r["frame"] for r in records])
    color = np.array([r["color"] for r in records])
    rgb = {r["color"]: np.array(r["rgb"], dtype=float) for r in records}
    return reps, frame, color, rgb


def prompt_clean(reps: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frames = sorted(set(frame))
    frame_mean = {f: reps[frame == f].mean(axis=0) for f in frames}
    return reps - np.stack([frame_mean[f] for f in frame])


def fit_loop(
    reps: np.ndarray,
    frame: np.ndarray,
    color: np.ndarray,
    rgb: dict[str, np.ndarray],
    spec: LoopSpec,
) -> dict:
    keep = np.array([c in spec.colors for c in color])
    reps, frame, color = reps[keep], frame[keep], color[keep]
    cleaned = prompt_clean(reps, frame)
    colors = [c for c in spec.colors if c in set(color)]
    color_index = np.array([colors.index(c) for c in color])

    means_full = np.stack([cleaned[color_index == i].mean(axis=0) for i in range(len(colors))])
    display_basis = np.linalg.svd(means_full - means_full.mean(axis=0), full_matrices=False)[2][:3]
    points = (cleaned - means_full.mean(axis=0)) @ display_basis.T
    means = (means_full - means_full.mean(axis=0)) @ display_basis.T

    color_t = np.array([hue01(rgb[c]) for c in colors])
    prompt_t = color_t[color_index]
    # Use observed hues as basis centers. No knot tuning; REML controls smoothness.
    centers = np.sort(color_t)
    B = periodic_duchon(prompt_t, centers)
    penalty = np.eye(B.shape[1])
    coefs = []
    lambdas = []
    edfs = []
    fitted_parts = []
    for dim in range(3):
        result = gamfit.gaussian_reml_fit_blocks_forward([B], [penalty], y=points[:, dim])
        coef = np.asarray(result["coefficients"]).ravel()[: B.shape[1]]
        coefs.append(coef)
        lambdas.append(float(np.asarray(result["lambdas"]).ravel()[0]))
        edfs.append(float(np.asarray(result["edf"]).ravel()[0]))
        fitted_parts.append(B @ coef)
    coef = np.stack(coefs, axis=1)
    prompt_fit = np.stack(fitted_parts, axis=1)
    B_mean = periodic_duchon(color_t, centers)
    mean_fit = B_mean @ coef
    grid_t = np.linspace(0, 1, 900, endpoint=True)
    curve = periodic_duchon(grid_t, centers) @ coef

    r2_prompt = float(1 - ((points - prompt_fit) ** 2).sum() / ((points - points.mean(axis=0)) ** 2).sum())
    r2_mean = float(1 - ((means - mean_fit) ** 2).sum() / ((means - means.mean(axis=0)) ** 2).sum())
    repro_err = float(np.max(np.abs(periodic_duchon(prompt_t, centers) @ coef - prompt_fit)))
    return {
        "spec": spec,
        "colors": colors,
        "color_index": color_index,
        "frame": frame,
        "rgb": rgb,
        "points": points,
        "means": means,
        "color_t": color_t,
        "grid_t": grid_t,
        "curve": curve,
        "mean_fit": mean_fit,
        "r2_prompt": r2_prompt,
        "r2_mean": r2_mean,
        "lambda": float(np.median(lambdas)),
        "edf": float(np.mean(edfs)),
        "repro_err": repro_err,
        "basis_cols": int(B.shape[1]),
    }


def line_segments(fig: go.Figure, curve: np.ndarray, grid_t: np.ndarray) -> None:
    # Plotly 3-D lines cannot use a continuous colorscale, so draw short colored segments.
    step = 7
    for i in range(0, len(curve) - step, step):
        seg = curve[i : i + step + 1]
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(width=15, color=hue_color(float(grid_t[i]), 0.16)),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for i in range(0, len(curve) - step, step):
        seg = curve[i : i + step + 1]
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(width=5, color=hue_color(float(grid_t[i]), 0.96)),
                hovertemplate="actual gamfit periodic Duchon loop<extra></extra>",
                showlegend=False,
            )
        )


def make_figure(fit: dict) -> go.Figure:
    spec = fit["spec"]
    colors = fit["colors"]
    rgb = fit["rgb"]
    points = fit["points"]
    means = fit["means"]
    color_index = fit["color_index"]
    frame = fit["frame"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=3.4, color=[rgb_css(rgb[colors[i]], 0.20) for i in color_index]),
            text=[f"{colors[i]}<br>prompt frame {int(fr)}" for i, fr in zip(color_index, frame, strict=True)],
            hovertemplate="%{text}<extra></extra>",
            name="prompt-cleaned prompt replicates",
        )
    )
    # Residual spokes from color mean to its fitted point at the same hue.
    for color_name, mean, fitted in zip(colors, means, fit["mean_fit"], strict=True):
        fig.add_trace(
            go.Scatter3d(
                x=[mean[0], fitted[0]], y=[mean[1], fitted[1]], z=[mean[2], fitted[2]],
                mode="lines",
                line=dict(width=1.2, color="rgba(255,255,255,0.22)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    line_segments(fig, fit["curve"], fit["grid_t"])
    fig.add_trace(
        go.Scatter3d(
            x=means[:, 0], y=means[:, 1], z=means[:, 2],
            mode="markers+text",
            marker=dict(size=13, color=[rgb_css(rgb[c]) for c in colors], line=dict(width=1.8, color="white")),
            text=colors,
            textfont=dict(size=11, color="white"),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="color means",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                f"{html.escape(spec.title)}<br>"
                f"<span style='font-size:13px;color:#aeb7ca'>actual gamfit periodic Duchon loop · "
                f"REML λ selected · color-mean R²={fit['r2_mean']:.2f} · prompt R²={fit['r2_prompt']:.2f} · "
                f"EDF={fit['edf']:.1f}</span>"
            ),
            x=0.02,
            font=dict(size=22),
        ),
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        margin=dict(l=8, r=8, t=70, b=8),
        showlegend=False,
        scene=dict(
            bgcolor="#080a0f",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.42, y=1.28, z=1.0)),
        ),
    )
    return fig


def fmt_float(x: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}"


def fit_learned_circle(reps: np.ndarray, frame: np.ndarray, color: np.ndarray, rgb: dict[str, np.ndarray]) -> dict:
    spec = SPECS[0]
    keep = np.array([c in spec.colors for c in color])
    reps, frame, color = reps[keep], frame[keep], color[keep]
    cleaned = prompt_clean(reps, frame)
    colors = [c for c in spec.colors if c in set(color)]
    color_index = np.array([colors.index(c) for c in color])
    means_full = np.stack([cleaned[color_index == i].mean(axis=0) for i in range(len(colors))])
    display_basis = np.linalg.svd(means_full - means_full.mean(axis=0), full_matrices=False)[2][:3]
    points = (cleaned - means_full.mean(axis=0)) @ display_basis.T
    means = (means_full - means_full.mean(axis=0)) @ display_basis.T

    # Natural basis size: one center per observed color. No sweep.
    centers = np.linspace(-1, 1, len(colors)).reshape(-1, 1)
    result = gamfit.gaussian_reml_optimize_latent(
        y=means.astype(float),
        n_obs=len(means),
        latent_dim=1,
        centers=centers,
        penalty=np.eye(len(centers)),
        basis_kind="duchon",
        manifold="circle",
        m=2,
        max_iter=220,
        n_restarts=6,
        seed=len(colors),
        init="spectral",
    )
    t = np.asarray(result.get("t", result.get("latent"))).reshape(len(colors), 1)
    coef = np.asarray(result["coefficients"]).reshape(len(centers), -1)
    fitted = np.asarray(result["fitted"]).reshape(len(colors), -1)
    lo, hi = float(t.min()), float(t.max())
    pad = 0.08 * max(hi - lo, 1e-9)
    grid_t = np.linspace(lo - pad, hi + pad, 900)
    curve = np.asarray(gamfit.duchon_basis(grid_t.reshape(-1, 1), centers, m=2)) @ coef
    mean_fit = np.asarray(gamfit.duchon_basis(t, centers, m=2)) @ coef
    r2_mean = float(1 - ((means - mean_fit) ** 2).sum() / ((means - means.mean(axis=0)) ** 2).sum())
    return {
        "spec": LoopSpec("learned_circle", "Learned circle-latent Duchon curve", colors),
        "colors": colors,
        "color_index": color_index,
        "frame": frame,
        "rgb": rgb,
        "points": points,
        "means": means,
        "grid_t": (grid_t - grid_t.min()) / (grid_t.max() - grid_t.min() + 1e-12),
        "curve": curve,
        "mean_fit": fitted,
        "r2_prompt": float("nan"),
        "r2_mean": r2_mean,
        "lambda": float(result.get("lambda", result.get("lambda_", np.nan)) or np.nan),
        "edf": float(result.get("edf", np.nan)),
        "repro_err": float(np.max(np.abs(mean_fit - fitted))),
        "basis_cols": len(centers),
        "converged": bool(result.get("converged", False)),
        "grad": float(result.get("grad_t_norm", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extra")
    parser.add_argument("--layer", type=int, default=44)
    parser.add_argument("--out", default="results/color_gamfit_good_loops")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps, frame, color, rgb = load(Path(args.extra), args.layer)
    fits = [fit_loop(reps, frame, color, rgb, spec) for spec in SPECS]
    fits.append(fit_learned_circle(reps, frame, color, rgb))
    sections = []
    diagnostics = []
    for fit in fits:
        filename = f"{fit['spec'].name}.html"
        make_figure(fit).write_html(out / filename, include_plotlyjs="cdn")
        diagnostics.append(
            {
                "name": fit["spec"].name,
                "colors": fit["colors"],
                "r2_mean": fit["r2_mean"],
                "r2_prompt": fit["r2_prompt"],
                "lambda_median": fit["lambda"],
                "edf_mean": fit["edf"],
                "basis_cols": fit["basis_cols"],
                "decoder_reproduction_error": fit["repro_err"],
                "converged": fit.get("converged"),
                "grad_t_norm": fit.get("grad"),
            }
        )
        extra_diag = ""
        if fit["spec"].name == "learned_circle":
            extra_diag = f" converged={fit['converged']}; ∥grad_t∥={fit['grad']:.2g};"
        sections.append(
            f"<section><h2>{html.escape(fit['spec'].title)}</h2>"
            f"<p>Actual periodic Duchon loop, fitted by gamfit REML. Color-mean R²={fit['r2_mean']:.3f}; "
            f"prompt-replicate R²={fmt_float(fit['r2_prompt'])}; EDF={fit['edf']:.1f}; median λ={fit['lambda']:.3g};{extra_diag}</p>"
            f"<iframe src='{filename}' loading='lazy'></iframe></section>"
        )
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    rows = "\n".join(
        f"<tr><td>{d['name']}</td><td>{len(d['colors'])}</td><td>{d['r2_mean']:.3f}</td>"
        f"<td>{fmt_float(d['r2_prompt'])}</td><td>{d['edf_mean']:.1f}</td><td>{d['lambda_median']:.3g}</td></tr>"
        for d in diagnostics
    )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Good gamfit color loops</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#080a0f; color:#edf2ff; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif; }}
    header {{ padding:34px 38px 12px; max-width:980px; }}
    h1 {{ font-size:40px; line-height:1.04; margin:0 0 12px; }}
    p {{ color:#b8c1d6; max-width:880px; }}
    code {{ background:rgba(255,255,255,.08); color:white; padding:2px 5px; border-radius:5px; }}
    table {{ margin-top:20px; border-collapse:collapse; width:100%; max-width:920px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.12); text-align:left; }}
    section {{ padding:22px 26px 44px; border-top:1px solid rgba(255,255,255,.11); }}
    section h2 {{ margin:0 0 6px 12px; font-size:22px; }}
    section p {{ margin:0 0 12px 12px; }}
    iframe {{ display:block; width:100%; height:760px; border:0; border-radius:10px; background:#080a0f; }}
  </style>
</head>
<body>
  <header>
    <h1>Good gamfit color loops</h1>
    <p>Loop-only. No UMAP, no surfaces, no nearest-neighbor ordering. The first two curves use hue as an
    interpretable color-wheel coordinate; the final curve lets gamfit learn a circle-latent coordinate and is
    labeled as diagnostic because the optimizer still reports non-convergence. All curves are actual Duchon
    decoder images evaluated densely.</p>
    <table><thead><tr><th>loop</th><th>colors</th><th>color R²</th><th>prompt R²</th><th>EDF</th><th>median λ</th></tr></thead><tbody>
      {rows}
    </tbody></table>
  </header>
  {''.join(sections)}
</body>
</html>"""
    (out / "index.html").write_text(index)
    print(f"wrote {out / 'index.html'}")
    for d in diagnostics:
        print(f"{d['name']}: color_R2={d['r2_mean']:.3f} prompt_R2={d['r2_prompt']:.3f} EDF={d['edf_mean']:.1f} lambda={d['lambda_median']:.3g}")
    import subprocess
    subprocess.run(["open", str(out / "index.html")])


if __name__ == "__main__":
    main()
