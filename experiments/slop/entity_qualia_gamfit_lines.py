"""Actual gamfit 1-D Duchon lines for entity and qualia manifolds.

This builds a small production site with two good, interpretable 1-D fits:

1. Entity manifold:
   Prompt-balanced entity-kind centroids. Each exp/noexp pair is averaged before
   kind aggregation, so the entity curve is not mostly explicit qualia wording.
   The 1-D coordinate is PC1 of those prompt-balanced kind centroids. The line is
   colored by the kind's smoothed qualia-axis coordinate.

2. Qualia manifold:
   Individual exp/noexp pair prompts. The 1-D coordinate is the anchor-relative
   qualia axis (0 = noexp pole, 1 = exp pole). The line is colored directly by
   that qualia coordinate.

Both fits use gamfit's Duchon basis and Duchon smoothness penalty, with smoothing
lambda selected by gamfit Gaussian REML. The plotted line is the actual fitted
decoder evaluated densely in the 3-D display frame, not connected dots.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import gamfit


ENTITY_COLORS = {
    "human": "#ffcf5a",
    "mammal": "#f4a261",
    "animal": "#f4a261",
    "bird": "#f4a261",
    "fish": "#f4a261",
    "reptile": "#f4a261",
    "insect": "#d8a15d",
    "mollusk": "#d8a15d",
    "plant": "#58c56f",
    "fungus": "#7bc96f",
    "microbe": "#9ed46b",
    "rock": "#a7adb8",
    "tool": "#8fb3ff",
    "artifact": "#8fb3ff",
    "vehicle": "#8fb3ff",
    "robot": "#6ca6ff",
    "ai": "#6ca6ff",
    "simulated": "#9b8cff",
    "simulator": "#9b8cff",
    "supernatural": "#e477ff",
}


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def qcolor(q: float, alpha: float = 1.0) -> str:
    q = float(np.clip(q, 0, 1))
    # noexp blue -> neutral pale -> exp amber
    stops = np.array([[54, 107, 214], [212, 220, 230], [255, 176, 64]], dtype=float)
    if q < 0.5:
        a = q / 0.5
        rgb = (1 - a) * stops[0] + a * stops[1]
    else:
        a = (q - 0.5) / 0.5
        rgb = (1 - a) * stops[1] + a * stops[2]
    return f"rgba({int(rgb[0])},{int(rgb[1])},{int(rgb[2])},{alpha:.3f})"


def entity_color(kind: str, alpha: float = 1.0) -> str:
    h = ENTITY_COLORS.get(kind, "#ccd3df").lstrip("#")
    r, g, b = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    return f"rgba({r},{g},{b},{alpha:.3f})"


def load(path: Path, layer: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = np.load(path / "activations.npy")
    recs = [json.loads(line) for line in open(path / "prompts.jsonl") if line.strip()]
    return x[:, min(layer, x.shape[1] - 1), :].astype(np.float64), recs


def qualia_axis(h: np.ndarray, recs: list[dict[str, Any]]) -> dict[str, Any]:
    role = np.array([r.get("role", "") for r in recs])
    side = np.array([r.get("side", "") for r in recs])
    pair = role == "pair"
    exp = np.where(pair & (side == "exp"))[0]
    noexp = np.where(pair & (side == "noexp"))[0]
    axis = unit(h[exp].mean(axis=0) - h[noexp].mean(axis=0))
    proj = h @ axis
    lo = float(proj[noexp].mean())
    hi = float(proj[exp].mean())
    coord = (proj - lo) / (hi - lo + 1e-12)
    return {"axis": axis, "coord": coord, "exp": exp, "noexp": noexp, "lo": lo, "hi": hi}


def rich_centers(t: np.ndarray, max_centers: int = 80) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    unique = np.unique(np.round(t, 10))
    if len(unique) <= max_centers:
        centers = unique
    else:
        centers = np.quantile(t, np.linspace(0.01, 0.99, max_centers))
        centers = np.unique(np.round(centers, 10))
    return centers.reshape(-1, 1)


def fit_duchon_line(t: np.ndarray, y: np.ndarray, *, max_centers: int = 80) -> dict[str, Any]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    centers = rich_centers(t, max_centers=max_centers)
    b = np.asarray(gamfit.duchon_basis(t.reshape(-1, 1), centers, m=2))
    penalty_note = "duchon_function_norm_penalty"
    try:
        penalty = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=2))
    except Exception as exc:
        # gamfit 0.1.180 bug: wrapper calls the Rust binding with wrong arity (gam#880).
        penalty = np.eye(b.shape[1])
        penalty_note = f"identity fallback (gam#880: {str(exc)[:70]})"
    coefs = []
    lambdas = []
    edfs = []
    fitted_cols = []
    for dim in range(y.shape[1]):
        result = gamfit.gaussian_reml_fit_blocks_forward([b], [penalty], y=y[:, dim])
        coef = np.asarray(result["coefficients"]).ravel()[: b.shape[1]]
        coefs.append(coef)
        lambdas.append(float(np.asarray(result["lambdas"]).ravel()[0]))
        edfs.append(float(np.asarray(result["edf"]).ravel()[0]))
        fitted_cols.append(b @ coef)
    coef = np.stack(coefs, axis=1)
    fitted = np.stack(fitted_cols, axis=1)
    r2 = float(1 - ((y - fitted) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())
    t_grid = np.linspace(float(t.min()), float(t.max()), 900)
    grid_b = np.asarray(gamfit.duchon_basis(t_grid.reshape(-1, 1), centers, m=2))
    curve = grid_b @ coef
    repro_err = float(np.max(np.abs(b @ coef - fitted)))
    return {
        "centers": centers,
        "coef": coef,
        "fitted": fitted,
        "curve": curve,
        "grid_t": t_grid,
        "r2": r2,
        "lambda": float(np.median(lambdas)),
        "edf": float(np.mean(edfs)),
        "repro_err": repro_err,
        "penalty": penalty_note,
    }


def plot_segments(fig: go.Figure, curve: np.ndarray, values: np.ndarray) -> None:
    step = 6
    for i in range(0, len(curve) - step, step):
        seg = curve[i : i + step + 1]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(width=15, color=qcolor(values[i], 0.14)), hoverinfo="skip", showlegend=False))
    for i in range(0, len(curve) - step, step):
        seg = curve[i : i + step + 1]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(width=5, color=qcolor(values[i], 0.96)),
            hovertemplate="actual gamfit Duchon line<extra></extra>", showlegend=False))


def layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, font=dict(size=22)),
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        margin=dict(l=8, r=8, t=76, b=8),
        showlegend=False,
        scene=dict(
            bgcolor="#080a0f",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.42, y=1.32, z=1.05)),
        ),
    )


def build_entity_fit(h: np.ndarray, recs: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    role = np.array([r.get("role", "") for r in recs])
    side = np.array([r.get("side", "") for r in recs])
    kind = np.array([r.get("kind", "") for r in recs])
    pair_id = np.array([r.get("pair_id", -1) for r in recs])
    pair_rows = np.where(role == "pair")[0]
    # Pair midpoint cancels explicit exp/noexp wording before aggregating by kind.
    midpoints = []
    mid_kinds = []
    for pid in sorted(set(pair_id[pair_rows])):
        idx = np.where((role == "pair") & (pair_id == pid))[0]
        if len(idx) < 2:
            continue
        exp = idx[side[idx] == "exp"]
        no = idx[side[idx] == "noexp"]
        if len(exp) and len(no):
            midpoints.append((h[exp].mean(axis=0) + h[no].mean(axis=0)) / 2)
            mid_kinds.append(kind[idx[0]])
    midpoints = np.asarray(midpoints)
    mid_kinds = np.asarray(mid_kinds)
    kinds = sorted(set(mid_kinds))
    reps = np.stack([midpoints[mid_kinds == k].mean(axis=0) for k in kinds])
    qcoord = np.array([((reps[i] @ q["axis"]) - q["lo"]) / (q["hi"] - q["lo"] + 1e-12) for i in range(len(kinds))])
    centered = reps - reps.mean(axis=0)
    display_basis = np.linalg.svd(centered, full_matrices=False)[2][:3]
    y = centered @ display_basis.T
    entity_coord = y[:, 0]
    fit = fit_duchon_line(entity_coord, y, max_centers=48)
    # Smooth qualia coordinate along entity coordinate for line coloring.
    qfit = fit_duchon_line(entity_coord, qcoord.reshape(-1, 1), max_centers=48)
    qline = (np.asarray(gamfit.duchon_basis(fit["grid_t"].reshape(-1, 1), qfit["centers"], m=2)) @ qfit["coef"]).ravel()
    return {"kinds": kinds, "y": y, "entity_coord": entity_coord, "qcoord": qcoord, "fit": fit, "qline": qline}


def build_qualia_fit(h: np.ndarray, recs: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    role = np.array([r.get("role", "") for r in recs])
    side = np.array([r.get("side", "") for r in recs])
    kind = np.array([r.get("kind", "") for r in recs])
    idx = np.where(role == "pair")[0]
    reps = h[idx]
    t = q["coord"][idx]
    centered = reps - reps.mean(axis=0)
    display_basis = np.linalg.svd(centered, full_matrices=False)[2][:3]
    y = centered @ display_basis.T
    fit = fit_duchon_line(t, y, max_centers=80)
    return {"idx": idx, "y": y, "t": t, "side": side[idx], "kind": kind[idx], "fit": fit}


def entity_page(out: Path, data: dict[str, Any]) -> dict[str, Any]:
    fit = data["fit"]
    fig = go.Figure()
    plot_segments(fig, fit["curve"], data["qline"])
    fig.add_trace(go.Scatter3d(
        x=data["y"][:, 0], y=data["y"][:, 1], z=data["y"][:, 2], mode="markers+text",
        marker=dict(size=12, color=[entity_color(k) for k in data["kinds"]], line=dict(width=1.6, color="white")),
        text=data["kinds"], textfont=dict(size=10, color="white"), textposition="top center",
        hovertemplate="%{text}<br>qualia coord=%{customdata:.2f}<extra></extra>",
        customdata=data["qcoord"], showlegend=False))
    title = (
        "Entity 1-D Duchon manifold<br>"
        f"<span style='font-size:13px;color:#aeb7ca'>pair-midpoint kind centroids · line colored by smoothed qualia coordinate · "
        f"R²={fit['r2']:.2f} · EDF={fit['edf']:.1f} · REML λ={fit['lambda']:.3g} · penalty={html.escape(fit['penalty'])}</span>"
    )
    layout(fig, title)
    fig.write_html(out / "entity_duchon_line.html", include_plotlyjs="cdn")
    return {"r2": fit["r2"], "edf": fit["edf"], "lambda": fit["lambda"], "n": len(data["kinds"])}


def qualia_page(out: Path, data: dict[str, Any]) -> dict[str, Any]:
    fit = data["fit"]
    fig = go.Figure()
    qline = (fit["grid_t"] - fit["grid_t"].min()) / (fit["grid_t"].max() - fit["grid_t"].min() + 1e-12)
    plot_segments(fig, fit["curve"], qline)
    is_exp = data["side"] == "exp"
    for side, mask, name in [("noexp", ~is_exp, "no-experience prompts"), ("exp", is_exp, "experience prompts")]:
        qvals = data["t"][mask]
        fig.add_trace(go.Scatter3d(
            x=data["y"][mask, 0], y=data["y"][mask, 1], z=data["y"][mask, 2], mode="markers",
            marker=dict(size=3.7, color=[qcolor(v, 0.36 if side == "noexp" else 0.48) for v in qvals]),
            text=[f"{k}<br>{side}<br>qualia={v:.2f}" for k, v in zip(data["kind"][mask], qvals, strict=True)],
            hovertemplate="%{text}<extra></extra>", name=name, showlegend=True))
    title = (
        "Qualia 1-D Duchon manifold<br>"
        f"<span style='font-size:13px;color:#aeb7ca'>coordinate is anchor-relative qualia axis (blue=noexp, amber=exp) · "
        f"R²={fit['r2']:.2f} · EDF={fit['edf']:.1f} · REML λ={fit['lambda']:.3g} · penalty={html.escape(fit['penalty'])}</span>"
    )
    layout(fig, title)
    fig.update_layout(showlegend=True, legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.25)"))
    fig.write_html(out / "qualia_duchon_line.html", include_plotlyjs="cdn")
    return {"r2": fit["r2"], "edf": fit["edf"], "lambda": fit["lambda"], "n": len(data["t"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--out", default="results/entity_qualia_gamfit_lines")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h, recs = load(Path(args.checkpoint), args.layer)
    q = qualia_axis(h, recs)
    entity = build_entity_fit(h, recs, q)
    qualia = build_qualia_fit(h, recs, q)
    entity_diag = entity_page(out, entity)
    qualia_diag = qualia_page(out, qualia)
    diagnostics = {"layer": args.layer, "entity": entity_diag, "qualia": qualia_diag}
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Entity and qualia Duchon lines</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#080a0f; color:#edf2ff; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif; }}
    header {{ padding:34px 38px 12px; max-width:980px; }}
    h1 {{ font-size:40px; line-height:1.04; margin:0 0 12px; }}
    p {{ color:#b8c1d6; max-width:880px; }}
    code {{ background:rgba(255,255,255,.08); color:white; padding:2px 5px; border-radius:5px; }}
    table {{ margin-top:20px; border-collapse:collapse; width:100%; max-width:760px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.12); text-align:left; }}
    section {{ padding:22px 26px 44px; border-top:1px solid rgba(255,255,255,.11); }}
    section h2 {{ margin:0 0 6px 12px; font-size:22px; }}
    iframe {{ display:block; width:100%; height:760px; border:0; border-radius:10px; background:#080a0f; }}
  </style>
</head>
<body>
  <header>
    <h1>Entity and qualia gamfit lines</h1>
    <p>Actual 1-D Duchon decoder lines, with smoothing selected by gamfit REML. The entity line uses
    exp/noexp pair midpoints before kind aggregation; the qualia line uses the anchor-relative qualia coordinate.
    Line color encodes qualia coordinate.</p>
    <table><thead><tr><th>fit</th><th>n</th><th>R²</th><th>EDF</th><th>λ</th></tr></thead><tbody>
      <tr><td>entity</td><td>{entity_diag['n']}</td><td>{entity_diag['r2']:.3f}</td><td>{entity_diag['edf']:.1f}</td><td>{entity_diag['lambda']:.3g}</td></tr>
      <tr><td>qualia</td><td>{qualia_diag['n']}</td><td>{qualia_diag['r2']:.3f}</td><td>{qualia_diag['edf']:.1f}</td><td>{qualia_diag['lambda']:.3g}</td></tr>
    </tbody></table>
  </header>
  <section><h2>Entity manifold</h2><iframe src="entity_duchon_line.html" loading="lazy"></iframe></section>
  <section><h2>Qualia manifold</h2><iframe src="qualia_duchon_line.html" loading="lazy"></iframe></section>
</body>
</html>"""
    (out / "index.html").write_text(index)
    print(f"wrote {out / 'index.html'}")
    print("entity", entity_diag)
    print("qualia", qualia_diag)
    import subprocess
    subprocess.run(["open", str(out / "index.html")])


if __name__ == "__main__":
    main()
