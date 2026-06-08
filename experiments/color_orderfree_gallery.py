"""Order-free color geometry gallery.

This is intentionally different from the earlier loop-order experiments:

* No nearest-color order.
* No nearest-representation order.
* No supplied 1-D parameterization of the colors.

The only fitted manifolds here are order-free gamfit latent fits where gamfit chooses
the latent coordinate itself:

    gaussian_reml_optimize_latent(..., basis_kind="duchon")

The visualizations are honest about failures. If circle/torus/sphere collapse or fail to
converge, the page says so instead of drawing a fake loop. UMAP pages are just alternate
3-D views of the prompt-cleaned point cloud, not loop fits.

Outputs a production gallery under results/color_orderfree_gallery by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import gamfit


CORE_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan",
    "magenta", "teal", "turquoise", "violet", "indigo", "crimson", "gold",
}

EXPANDED_COLORS = CORE_COLORS | {"coral", "mint", "lavender", "maroon"}

EXCLUDED_COLORS = {
    "black", "white", "grey", "gray", "beige", "brown", "tan", "navy",
    "olive", "salmon", "peach", "lime",
}


def rgb_css(rgb: np.ndarray, alpha: float | None = None) -> str:
    r, g, b = [int(x) for x in rgb]
    if alpha is None:
        return f"rgb({r},{g},{b})"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def center_scale(x: np.ndarray) -> np.ndarray:
    y = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt((y**2).sum() / max(len(y), 1))
    return y / (scale + 1e-12)


def variance_partition(reps: np.ndarray, color_index: np.ndarray, frame: np.ndarray, n_colors: int, n_frames: int) -> dict[str, float]:
    grand = reps.mean(axis=0)

    def ss(a: np.ndarray) -> float:
        return float((a**2).sum())

    total = ss(reps - grand)
    color_ss = sum(
        (color_index == i).sum() * ss(reps[color_index == i].mean(axis=0) - grand)
        for i in range(n_colors)
    )
    frame_mean = np.stack([reps[frame == f].mean(axis=0) for f in range(n_frames)])
    frame_ss = sum((frame == f).sum() * ss(frame_mean[f] - grand) for f in range(n_frames))
    return {
        "color": color_ss / total,
        "frame": frame_ss / total,
        "residual": 1.0 - (color_ss + frame_ss) / total,
    }


def remove_frame_term(reps: np.ndarray, frame: np.ndarray, n_frames: int) -> np.ndarray:
    frame_mean = np.stack([reps[frame == f].mean(axis=0) for f in range(n_frames)])
    return reps - frame_mean[frame]


def make_hover(colors: list[str], color_index: np.ndarray, frame: np.ndarray | None = None) -> list[str]:
    out = []
    for row, ci in enumerate(color_index):
        text = colors[int(ci)]
        if frame is not None:
            text += f"<br>prompt frame {int(frame[row])}"
        out.append(text)
    return out


def plot_cloud_page(
    out: Path,
    name: str,
    title: str,
    points: np.ndarray,
    means: np.ndarray,
    colors: list[str],
    color_index: np.ndarray,
    frame: np.ndarray,
    rgb: dict[str, np.ndarray],
    subtitle: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=4, color=[rgb_css(rgb[colors[i]], 0.36) for i in color_index]),
            text=make_hover(colors, color_index, frame),
            hovertemplate="%{text}<extra></extra>",
            name="prompt-cleaned prompt replicates",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=means[:, 0],
            y=means[:, 1],
            z=means[:, 2],
            mode="markers+text",
            marker=dict(size=13, color=[rgb_css(rgb[c]) for c in colors], line=dict(width=1.6, color="white")),
            text=colors,
            textfont=dict(size=11, color="white"),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="color means",
        )
    )
    apply_layout(fig, f"{title}<br><span style='font-size:13px;color:#a8b0bf'>{subtitle}</span>")
    fig.write_html(out / name, include_plotlyjs="cdn")


def apply_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, font=dict(size=22)),
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        margin=dict(l=8, r=8, t=58, b=8),
        showlegend=False,
        scene=dict(
            bgcolor="#080a0f",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.05)),
        ),
    )


def centers_for(manifold: str, dim: int, k: int = 18) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(0)
    if manifold in {"euclidean", "circle"}:
        if dim == 1:
            centers = np.linspace(-1, 1, k).reshape(-1, 1)
        else:
            side = max(3, int(round(k ** (1 / dim))))
            grids = np.meshgrid(*([np.linspace(-1, 1, side)] * dim), indexing="ij")
            centers = np.stack([g.ravel() for g in grids], axis=1)
        latent_dim = dim
    elif manifold == "sphere":
        centers = rng.normal(size=(k, 3))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        latent_dim = 3
    elif manifold == "torus":
        side = max(4, int(round(k**0.5)))
        vals = np.linspace(0, 1, side, endpoint=False)
        centers = np.array([[a, b] for a in vals for b in vals], dtype=float)
        latent_dim = 2
    else:
        raise ValueError(manifold)
    return centers.astype(float), np.eye(len(centers)), latent_dim


def score_reconstruction(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(1 - ((y - fitted) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())


def fit_latent(y: np.ndarray, manifold: str, dim: int) -> dict[str, Any]:
    centers, penalty, latent_dim = centers_for(manifold, dim)
    kwargs: dict[str, Any] = dict(
        y=y.astype(float),
        n_obs=len(y),
        latent_dim=latent_dim,
        centers=centers,
        penalty=penalty,
        basis_kind="duchon",
        manifold=manifold,
        m=2,
        max_iter=160,
        n_restarts=3,
        seed=4,
        init="spectral",
    )
    if manifold == "sphere":
        rng = np.random.default_rng(4)
        t = rng.normal(size=(len(y), 3))
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        kwargs["t"] = t.reshape(-1)
        kwargs["init"] = "caller"
    try:
        result = gamfit.gaussian_reml_optimize_latent(**kwargs)
        t = np.asarray(result.get("t", result.get("latent"))).reshape(len(y), -1)
        fitted = np.asarray(result.get("fitted"))
        if fitted.ndim == 1:
            fitted = fitted.reshape(len(y), -1)
        spread = float(np.sqrt(((t - t.mean(axis=0)) ** 2).sum(axis=1).mean()))
        return {
            "ok": True,
            "manifold": manifold,
            "dim": dim,
            "latent_dim": latent_dim,
            "t": t,
            "fitted": fitted,
            "r2": score_reconstruction(y, fitted),
            "reml": float(result.get("reml_score", np.nan)),
            "converged": bool(result.get("converged", False)),
            "grad_t_norm": float(result.get("grad_t_norm", np.nan)),
            "spread": spread,
        }
    except Exception as exc:
        return {"ok": False, "manifold": manifold, "dim": dim, "error": str(exc)[:220]}


def plot_latent_page(out: Path, fit: dict[str, Any], colors: list[str], rgb: dict[str, np.ndarray]) -> str:
    manifold = fit["manifold"]
    dim = fit["dim"]
    filename = f"gamfit_latent_{manifold}_d{dim}.html"
    if not fit["ok"]:
        (out / filename).write_text(f"<html><body><pre>{json.dumps(fit, indent=2)}</pre></body></html>")
        return filename

    t = fit["t"]
    if t.shape[1] == 1:
        x = t[:, 0]
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        title_shape = "learned 1-D latent positions"
    elif t.shape[1] == 2:
        x, y = t[:, 0], t[:, 1]
        z = np.zeros_like(x)
        title_shape = "learned 2-D latent positions"
    else:
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        title_shape = "learned 3-D latent positions"

    marker_colors = [rgb_css(rgb[c]) for c in colors]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            marker=dict(size=13, color=marker_colors, line=dict(width=1.6, color="white")),
            text=colors,
            textfont=dict(size=11, color="white"),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    status = "converged" if fit["converged"] else "not converged"
    warning = ""
    if fit["spread"] < 1e-3 or (not np.isfinite(fit["grad_t_norm"])) or fit["grad_t_norm"] > 1e3:
        warning = " · likely collapsed/unstable"
    apply_layout(
        fig,
        f"gamfit order-free {manifold} d={dim}: {title_shape}<br>"
        f"<span style='font-size:13px;color:#a8b0bf'>Duchon decoder · R²={fit['r2']:.2f} · {status} · "
        f"∥grad_t∥={fit['grad_t_norm']:.2g} · spread={fit['spread']:.2g}{warning}</span>",
    )
    fig.write_html(out / filename, include_plotlyjs="cdn")
    return filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extra", help="Path to color extra/ directory with activations.npy and prompts.jsonl")
    parser.add_argument("--layer", type=int, default=44)
    parser.add_argument("--palette", choices=["core", "expanded"], default="core")
    parser.add_argument("--out", default="results/color_orderfree_gallery")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    palette = CORE_COLORS if args.palette == "core" else EXPANDED_COLORS
    extra = Path(args.extra)
    activations = np.load(extra / "activations.npy")
    records = [json.loads(line) for line in open(extra / "prompts.jsonl") if line.strip()]
    reps = activations[:, min(args.layer, activations.shape[1] - 1), :].astype(np.float64)
    frame = np.array([r["frame"] for r in records])
    color = np.array([r["color"] for r in records])
    rgb = {r["color"]: np.array(r["rgb"], dtype=float) for r in records}

    keep = np.array([(c in palette) and (c not in EXCLUDED_COLORS) for c in color])
    reps = reps[keep]
    frame = frame[keep]
    color = color[keep]
    colors = sorted(set(color))
    color_index = np.array([colors.index(c) for c in color])
    n_colors = len(colors)
    n_frames = len(set(frame))

    partition = variance_partition(reps, color_index, frame, n_colors, n_frames)
    prompt_free = remove_frame_term(reps, frame, n_frames)
    color_means = np.stack([prompt_free[color_index == i].mean(axis=0) for i in range(n_colors)])

    pc_basis = np.linalg.svd(color_means - color_means.mean(axis=0), full_matrices=False)[2][:3]
    pc_points = (prompt_free - color_means.mean(axis=0)) @ pc_basis.T
    pc_means = (color_means - color_means.mean(axis=0)) @ pc_basis.T
    plot_cloud_page(
        out,
        "cloud_color_pc3.html",
        "Prompt-cleaned color cloud",
        pc_points,
        pc_means,
        colors,
        color_index,
        frame,
        rgb,
        "top-3 PCs of color means · no ordering · no fitted loop",
    )

    umap_pages: list[str] = []
    try:
        import umap
        configs = [
            ("umap_cosine_local.html", "UMAP cosine · local", dict(metric="cosine", n_neighbors=5, min_dist=0.12, random_state=11)),
            ("umap_cosine_balanced.html", "UMAP cosine · balanced", dict(metric="cosine", n_neighbors=10, min_dist=0.25, random_state=12)),
            ("umap_cosine_global.html", "UMAP cosine · global", dict(metric="cosine", n_neighbors=18, min_dist=0.45, random_state=13)),
            ("umap_euclidean.html", "UMAP euclidean", dict(metric="euclidean", n_neighbors=10, min_dist=0.25, random_state=14)),
        ]
        for filename, title, config in configs:
            reducer = umap.UMAP(n_components=3, **config)
            emb = reducer.fit_transform(center_scale(prompt_free))
            emb_means = np.stack([emb[color_index == i].mean(axis=0) for i in range(n_colors)])
            plot_cloud_page(
                out,
                filename,
                title,
                emb,
                emb_means,
                colors,
                color_index,
                frame,
                rgb,
                "order-free view of prompt-cleaned prompt replicates",
            )
            umap_pages.append(filename)
    except Exception as exc:
        (out / "umap_error.txt").write_text(str(exc))

    # Order-free gamfit latent fits. Fit only color means to avoid learning the prompt-frame structure.
    y = center_scale(pc_means)
    latent_specs = [
        ("euclidean", 1),
        ("euclidean", 2),
        ("euclidean", 3),
        ("circle", 1),
        ("torus", 2),
        ("sphere", 2),
    ]
    latent_results = [fit_latent(y, manifold, dim) for manifold, dim in latent_specs]
    latent_pages = [plot_latent_page(out, result, colors, rgb) for result in latent_results]
    (out / "gamfit_latent_results.json").write_text(json.dumps(latent_results, indent=2, default=lambda x: "<array>"))

    rows = []
    for result in latent_results:
        if result["ok"]:
            rows.append(
                f"<tr><td>{result['manifold']} d={result['dim']}</td><td>{result['r2']:.3f}</td>"
                f"<td>{result['converged']}</td><td>{result['grad_t_norm']:.2g}</td><td>{result['spread']:.2g}</td>"
                f"<td><a href='gamfit_latent_{result['manifold']}_d{result['dim']}.html'>open</a></td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{result['manifold']} d={result['dim']}</td><td colspan='4'>error: {result['error']}</td>"
                f"<td><a href='gamfit_latent_{result['manifold']}_d{result['dim']}.html'>open</a></td></tr>"
            )

    umap_links = "".join(f"<a href='{p}'>{p.replace('.html','').replace('_',' ')}</a>" for p in umap_pages)
    latent_links = "".join(f"<a href='{p}'>{p.replace('.html','').replace('_',' ')}</a>" for p in latent_pages)
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Order-free color geometry</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 18% 8%, rgba(72,112,255,.18), transparent 34rem),
        radial-gradient(circle at 82% 14%, rgba(255,72,120,.13), transparent 30rem),
        #080a0f;
      color: #edf2ff;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1020px; padding: 42px 38px 64px; }}
    h1 {{ font-size: 40px; line-height: 1.02; margin: 0 0 12px; }}
    h2 {{ margin-top: 34px; font-size: 20px; }}
    p {{ color: #b7c0d5; max-width: 800px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 26px 0; }}
    .metric {{ border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.045); border-radius: 8px; padding: 14px 15px; }}
    .metric b {{ display: block; font-size: 24px; color: white; }}
    a {{ display: block; color: #f8fbff; text-decoration: none; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.07); border-radius: 8px; padding: 13px 15px; margin: 9px 0; }}
    a:hover {{ background: rgba(255,255,255,.12); }}
    table {{ border-collapse: collapse; margin-top: 12px; width: 100%; }}
    td, th {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.11); }}
    code {{ background: rgba(255,255,255,.08); border-radius: 5px; padding: 2px 5px; color: white; }}
  </style>
</head>
<body>
<main>
  <h1>Order-free color geometry</h1>
  <p>RL3.1 final, layer {args.layer}. No nearest-color order, no nearest-representation order, no supplied loop parameter.
  UMAP pages are order-free views. Gamfit pages use <code>gaussian_reml_optimize_latent</code> with <code>basis_kind="duchon"</code>, so gamfit chooses the latent coordinates.</p>
  <div class="metrics">
    <div class="metric"><b>{n_colors}</b><span>core colors</span></div>
    <div class="metric"><b>{len(color_index)}</b><span>prompt replicates</span></div>
    <div class="metric"><b>{partition['frame']:.0%}</b><span>prompt-frame variance</span></div>
    <div class="metric"><b>{partition['color']:.0%}</b><span>color variance</span></div>
  </div>
  <h2>Point-cloud views</h2>
  <a href="cloud_color_pc3.html">prompt-cleaned color cloud · PC3</a>
  {umap_links}
  <h2>Gamfit order-free latent fits</h2>
  <p>These are not forced loops. They are learned latent positions. Known gamfit issue #876 may cause periodic latent manifolds to collapse; the table exposes convergence, gradient, and spread.</p>
  <table><thead><tr><th>fit</th><th>R²</th><th>converged</th><th>grad</th><th>spread</th><th>page</th></tr></thead><tbody>
  {''.join(rows)}
  </tbody></table>
  <h2>Direct links</h2>
  {latent_links}
</main>
</body>
</html>"""
    (out / "index.html").write_text(index)
    print(f"wrote order-free gallery -> {out / 'index.html'}")
    print("latent summary:")
    for result in latent_results:
        if result["ok"]:
            print(
                f"  {result['manifold']} d={result['dim']}: R2={result['r2']:.3f} "
                f"conv={result['converged']} grad={result['grad_t_norm']:.2g} spread={result['spread']:.2g}"
            )
        else:
            print(f"  {result['manifold']} d={result['dim']}: ERROR {result['error']}")

    import subprocess

    subprocess.run(["open", str(out / "index.html")])


if __name__ == "__main__":
    main()
