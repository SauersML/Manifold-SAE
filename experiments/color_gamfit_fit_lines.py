"""Actual gamfit fit lines/surfaces for the color representation.

No forced ordering. No nearest-color tour. No nearest-rep tour. No UMAP.

Pipeline:
1. Use the balanced color prompt design to remove the shared prompt-frame term.
2. Collapse each color to its prompt-cleaned color mean.
3. Project those means to their top-3 display PCs.
4. Fit order-free gamfit latent models with Duchon decoders:
      gaussian_reml_optimize_latent(..., basis_kind="duchon")
   Gamfit chooses the latent positions.
5. Evaluate the learned decoder densely and plot the actual decoder image.

The plot is therefore the real gamfit fit, not connected data points and not latent
coordinates. The 3-D axes are only a display projection of the fitted decoder image.
"""
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import gamfit


CORE_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan",
    "magenta", "teal", "turquoise", "violet", "indigo", "crimson", "gold",
}


@dataclass(frozen=True)
class FitSpec:
    name: str
    title: str
    manifold: str
    latent_dim: int
    centers_kind: str
    n_centers: int
    max_iter: int = 240
    n_restarts: int = 6
    seed: int = 2


SPECS = [
    FitSpec("line_k8", "Learned 1-D Duchon curve", "euclidean", 1, "line", 8, seed=8),
    FitSpec("circle_k12", "Learned circle-latent Duchon curve", "circle", 1, "line", 12, seed=12),
    FitSpec("torus_k16", "Learned torus-latent Duchon surface", "torus", 2, "torus", 16, seed=16),
    FitSpec("sheet_k24", "Learned 2-D Euclidean Duchon sheet", "euclidean", 2, "grid2", 24, seed=24),
]


def rgb_css(rgb: np.ndarray, alpha: float | None = None) -> str:
    r, g, b = [int(x) for x in rgb]
    if alpha is None:
        return f"rgb({r},{g},{b})"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def load_color_data(extra: Path, layer: int) -> dict[str, Any]:
    activations = np.load(extra / "activations.npy")
    records = [json.loads(line) for line in open(extra / "prompts.jsonl") if line.strip()]
    reps = activations[:, min(layer, activations.shape[1] - 1), :].astype(np.float64)
    frame = np.array([r["frame"] for r in records])
    color = np.array([r["color"] for r in records])
    rgb = {r["color"]: np.array(r["rgb"], dtype=float) for r in records}

    keep = np.array([c in CORE_COLORS for c in color])
    reps, frame, color = reps[keep], frame[keep], color[keep]
    colors = sorted(set(color))
    color_index = np.array([colors.index(c) for c in color])
    n_frames = len(set(frame))

    frame_mean = np.stack([reps[frame == f].mean(axis=0) for f in range(n_frames)])
    prompt_clean = reps - frame_mean[frame]
    color_means = np.stack([prompt_clean[color_index == i].mean(axis=0) for i in range(len(colors))])

    centered = color_means - color_means.mean(axis=0)
    display_basis = np.linalg.svd(centered, full_matrices=False)[2][:3]
    means3 = centered @ display_basis.T
    prompt3 = (prompt_clean - color_means.mean(axis=0)) @ display_basis.T

    return {
        "colors": colors,
        "rgb": rgb,
        "frame": frame,
        "color_index": color_index,
        "y": means3 - means3.mean(axis=0),
        "means3": means3,
        "prompt3": prompt3,
    }


def centers_for(spec: FitSpec) -> tuple[np.ndarray, np.ndarray]:
    if spec.centers_kind == "line":
        centers = np.linspace(-1, 1, spec.n_centers).reshape(-1, 1)
    elif spec.centers_kind == "grid2":
        side = max(3, int(round(spec.n_centers ** 0.5)))
        vals = np.linspace(-1, 1, side)
        centers = np.array([[a, b] for a in vals for b in vals], dtype=float)
    elif spec.centers_kind == "torus":
        side = max(4, int(round(spec.n_centers ** 0.5)))
        vals = np.linspace(0, 1, side, endpoint=False)
        centers = np.array([[a, b] for a in vals for b in vals], dtype=float)
    else:
        raise ValueError(spec.centers_kind)
    return centers, np.eye(len(centers))


def fit_spec(y: np.ndarray, spec: FitSpec) -> dict[str, Any]:
    centers, penalty = centers_for(spec)
    result = gamfit.gaussian_reml_optimize_latent(
        y=y.astype(float),
        n_obs=len(y),
        latent_dim=spec.latent_dim,
        centers=centers,
        penalty=penalty,
        basis_kind="duchon",
        manifold=spec.manifold,
        m=2,
        max_iter=spec.max_iter,
        n_restarts=spec.n_restarts,
        seed=spec.seed,
        init="spectral",
    )
    t = np.asarray(result.get("t", result.get("latent"))).reshape(len(y), -1)
    fitted = np.asarray(result["fitted"]).reshape(len(y), -1)
    coef = np.asarray(result["coefficients"]).reshape(len(centers), -1)
    repro = np.asarray(gamfit.duchon_basis(t, centers, m=2)) @ coef
    repro_err = float(np.max(np.abs(repro - fitted)))
    r2 = float(1 - ((y - fitted) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())
    spread = float(np.sqrt(((t - t.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return {
        "spec": spec,
        "centers": centers,
        "coef": coef,
        "t": t,
        "fitted": fitted,
        "r2": r2,
        "repro_err": repro_err,
        "converged": bool(result.get("converged", False)),
        "grad": float(result.get("grad_t_norm", np.nan)),
        "spread": spread,
        "reml": float(result.get("reml_score", np.nan)),
    }


def eval_curve(fit: dict[str, Any], n: int = 700) -> np.ndarray:
    spec: FitSpec = fit["spec"]
    centers = fit["centers"]
    coef = fit["coef"]
    t = fit["t"]
    if spec.latent_dim == 1:
        lo, hi = float(t[:, 0].min()), float(t[:, 0].max())
        pad = 0.08 * max(hi - lo, 1e-9)
        grid = np.linspace(lo - pad, hi + pad, n).reshape(-1, 1)
        return np.asarray(gamfit.duchon_basis(grid, centers, m=2)) @ coef

    mins = t.min(axis=0)
    maxs = t.max(axis=0)
    pad = 0.08 * np.maximum(maxs - mins, 1e-9)
    a = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], 44)
    b = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], 44)
    aa, bb = np.meshgrid(a, b)
    grid = np.c_[aa.ravel(), bb.ravel()]
    z = np.asarray(gamfit.duchon_basis(grid, centers, m=2)) @ coef
    return z.reshape(len(b), len(a), -1)


def fit_figure(data: dict[str, Any], fit: dict[str, Any]) -> go.Figure:
    spec: FitSpec = fit["spec"]
    colors = data["colors"]
    rgb = data["rgb"]
    means3 = data["means3"]
    prompt3 = data["prompt3"]
    color_index = data["color_index"]
    frame = data["frame"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=prompt3[:, 0],
            y=prompt3[:, 1],
            z=prompt3[:, 2],
            mode="markers",
            marker=dict(size=3.4, color=[rgb_css(rgb[colors[i]], 0.24) for i in color_index]),
            text=[f"{colors[i]}<br>prompt frame {int(f)}" for i, f in zip(color_index, frame, strict=True)],
            hovertemplate="%{text}<extra></extra>",
            name="prompt-cleaned prompt replicates",
        )
    )
    obj = eval_curve(fit)
    if spec.latent_dim == 1:
        curve = obj
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(width=16, color="rgba(255,255,255,0.16)"),
                hoverinfo="skip",
                name="fit glow",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(width=5, color="white"),
                hovertemplate="actual gamfit decoder curve<extra></extra>",
                name="actual gamfit fit",
            )
        )
    else:
        surf = obj
        fig.add_trace(
            go.Surface(
                x=surf[:, :, 0],
                y=surf[:, :, 1],
                z=surf[:, :, 2],
                colorscale=[[0, "rgba(255,255,255,0.10)"], [1, "rgba(255,255,255,0.50)"]],
                opacity=0.33,
                showscale=False,
                hoverinfo="skip",
                name="actual gamfit surface",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=means3[:, 0],
            y=means3[:, 1],
            z=means3[:, 2],
            mode="markers+text",
            marker=dict(size=13, color=[rgb_css(rgb[c]) for c in colors], line=dict(width=1.7, color="white")),
            text=colors,
            textfont=dict(size=11, color="white"),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="color means",
        )
    )
    status = "converged" if fit["converged"] else "not converged"
    title = (
        f"{html.escape(spec.title)}<br>"
        f"<span style='font-size:13px;color:#aeb7ca'>actual Duchon decoder image · "
        f"R²={fit['r2']:.3f} · {status} · ∥grad_t∥={fit['grad']:.2g} · repro err={fit['repro_err']:.1e}</span>"
    )
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, font=dict(size=22)),
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
            camera=dict(eye=dict(x=1.5, y=1.35, z=1.06)),
        ),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extra")
    parser.add_argument("--layer", type=int, default=44)
    parser.add_argument("--out", default="results/color_gamfit_fits")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_color_data(Path(args.extra), args.layer)
    y = data["y"]
    fits = [fit_spec(y, spec) for spec in SPECS]

    sections: list[str] = []
    for i, fit in enumerate(fits):
        filename = f"{fit['spec'].name}.html"
        fig = fit_figure(data, fit)
        fig.write_html(out / filename, include_plotlyjs="cdn")
        iframe = f"<iframe src='{filename}' loading='lazy'></iframe>"
        sections.append(
            f"<section><h2>{html.escape(fit['spec'].title)}</h2>"
            f"<p>R²={fit['r2']:.3f}; converged={fit['converged']}; ∥grad_t∥={fit['grad']:.2g}; "
            f"decoder reproduction error={fit['repro_err']:.1e}. The white object is the actual gamfit decoder evaluated densely.</p>"
            f"{iframe}</section>"
        )

    rows = "\n".join(
        f"<tr><td>{html.escape(f['spec'].name)}</td><td>{html.escape(f['spec'].manifold)}</td>"
        f"<td>{f['r2']:.3f}</td><td>{f['converged']}</td><td>{f['grad']:.2g}</td><td>{f['spread']:.2g}</td></tr>"
        for f in fits
    )
    diagnostics = [
        {
            "name": f["spec"].name,
            "title": f["spec"].title,
            "manifold": f["spec"].manifold,
            "latent_dim": f["spec"].latent_dim,
            "r2": f["r2"],
            "converged": f["converged"],
            "grad_t_norm": f["grad"],
            "spread": f["spread"],
            "reml": f["reml"],
            "decoder_reproduction_error": f["repro_err"],
        }
        for f in fits
    ]
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Actual gamfit color fits</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin:0;
      background:#080a0f;
      color:#edf2ff;
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;
    }}
    header {{ padding:34px 38px 10px; max-width:980px; }}
    h1 {{ font-size:40px; line-height:1.04; margin:0 0 12px; }}
    p {{ color:#b7c0d5; max-width:850px; }}
    code {{ background:rgba(255,255,255,.08); padding:2px 5px; border-radius:5px; color:white; }}
    table {{ margin-top:20px; border-collapse:collapse; width:100%; max-width:920px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.12); text-align:left; }}
    section {{ padding:22px 26px 42px; border-top:1px solid rgba(255,255,255,.11); }}
    section h2 {{ margin:0 0 6px 12px; font-size:22px; }}
    section p {{ margin:0 0 12px 12px; }}
    iframe {{ display:block; width:100%; height:720px; border:0; border-radius:10px; background:#080a0f; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual gamfit fit lines/surfaces</h1>
    <p>No forced ordering. No UMAP. No connected data points. Each white curve/surface is the actual
    <code>gamfit.gaussian_reml_optimize_latent</code> Duchon decoder image evaluated densely, plotted over
    prompt-frame-cleaned color points. The 3-D view is only a display projection.</p>
    <table><thead><tr><th>fit</th><th>manifold</th><th>R²</th><th>converged</th><th>grad</th><th>spread</th></tr></thead><tbody>
    {rows}
    </tbody></table>
  </header>
  {''.join(sections)}
</body>
</html>"""
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'}")
    for f in fits:
        print(f"{f['spec'].name}: R2={f['r2']:.3f} converged={f['converged']} grad={f['grad']:.2g} repro={f['repro_err']:.1e}")
    import subprocess
    subprocess.run(["open", str(out / "index.html")])


if __name__ == "__main__":
    main()
