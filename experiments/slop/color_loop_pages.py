"""One clean interactive 3D page PER loop fit, with PROMPT modeled OUT.

The color reps are a balanced crossed design: 6 shared prompt templates ('frames') x 30
colors, 1 obs/cell. The prompt template is ~69% of the variance; color is only ~10%. So a
naive fit on all points mostly fits PROMPT. We solve it the standard additive-model way:

    rep ~ loop(t_color) + frame          (frame = nuisance; crossed+balanced => subtracts cleanly)

Every color's 6 prompts share ONE latent position t_color, so the penalized loop predicts one
point per color and is structurally unable to fit prompt-level variation; the frame term soaks
up the shared template effect. We display the prompt-removed points (within-color scatter is now
just interaction+noise ~20%) + the 24 color means + ONE fitted loop. One html page per fit, plus
an index. gamfit for all curve fits. No forced unit sphere. Read-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import gamfit
import plotly.graph_objects as go

VIVID = {"red","orange","yellow","green","blue","purple","pink","cyan","magenta","teal",
         "turquoise","violet","indigo","crimson","gold","maroon","lime","coral","salmon",
         "peach","mint","lavender","navy","olive"}


def srgb2lab(c255):
    c = c255 / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    xyz = (c @ M.T) / np.array([.95047, 1, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def nn_cycle(D):
    m = len(D); t = [0]; rem = set(range(1, m))
    while rem:
        t.append(min(rem, key=lambda k: D[t[-1], k])); rem.discard(t[-1])
    tl = lambda u: sum(D[u[i], u[(i + 1) % m]] for i in range(m)); imp = True
    while imp:
        imp = False
        for i in range(m - 1):
            for k in range(i + 1, m):
                nt = t[:i] + t[i:k + 1][::-1] + t[k + 1:]
                if tl(nt) < tl(t) - 1e-9:
                    t = nt; imp = True
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--out", default="/tmp/colorloops")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    X = np.load(Path(a.extra) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(a.extra) / "prompts.jsonl") if l.strip()]
    H = X[:, min(a.layer, X.shape[1] - 1), :].astype(np.float64)
    frame = np.array([r["frame"] for r in recs]); color = np.array([r["color"] for r in recs])
    rgb = {r["color"]: np.array(r["rgb"], float) for r in recs}
    keep = np.array([c in VIVID for c in color]); H, frame, color = H[keep], frame[keep], color[keep]
    cols = sorted(set(color)); n = len(cols); ci = np.array([cols.index(c) for c in color])
    nf = len(set(frame))

    # ---- model out the prompt: rep ~ ... + frame (subtract per-frame mean; exact for crossed+balanced) ----
    frmean = np.stack([H[frame == f].mean(0) for f in range(nf)])
    Hadj = H - frmean[frame]                                   # prompt-removed reps
    cadj = np.stack([Hadj[ci == i].mean(0) for i in range(n)])  # color means (prompt-free)
    # variance partition (full-d)
    grand = H.mean(0); SS = lambda A: (A ** 2).sum()
    tot = SS(H - grand)
    ss_c = sum((ci == i).sum() * SS(np.stack([H[ci == i].mean(0)])[0] - grand) for i in range(n))
    ss_f = sum((frame == f).sum() * SS(frmean[f] - grand) for f in range(nf))
    part = (ss_c / tot, ss_f / tot, 1 - (ss_c + ss_f) / tot)

    Mc = cadj - cadj.mean(0); _, _, Vt = np.linalg.svd(Mc, full_matrices=False); P3 = Vt[:3]
    Ycol = Mc @ P3.T
    Yadj = (Hadj - cadj.mean(0)) @ P3.T                        # prompt-removed points, display frame
    Lab = np.stack([srgb2lab(rgb[c]) for c in cols])
    from scipy.spatial.distance import cdist
    order_color = nn_cycle(cdist(Lab, Lab))
    order_rep = nn_cycle(cdist(Ycol, Ycol))

    def tpos(order, Yd):
        seg = np.array([np.linalg.norm(Yd[order[i]] - Yd[order[(i + 1) % n]]) for i in range(n)])
        tp = np.zeros(n); tp[order] = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()
        return tp

    def gcv(B, P, Y):
        nn = len(Y); best = None
        for lam in np.logspace(-4, 4, 50):
            A = B @ np.linalg.solve(B.T @ B + lam * P, B.T)
            g = nn * ((Y - A @ Y) ** 2).sum() / Y.shape[1] / (nn - np.trace(A)) ** 2
            if best is None or g < best[0]:
                best = (g, lam)
        return np.linalg.solve(B.T @ B + best[1] * P, B.T @ Y), best[1]

    def basis(kind, t, K=12):
        if kind == "bspline":
            return np.asarray(gamfit.periodic_spline_curve_basis(t, K)[0]), np.eye(K)
        if kind == "duchon":
            C = np.linspace(0, 1, K, endpoint=False).reshape(-1, 1)
            return np.asarray(gamfit.duchon_basis(t.reshape(-1, 1), C, m=2, periodic_per_axis=[1.0])), None
        if kind == "fourier":
            cc = [np.ones_like(t)]
            for k in range(1, 6):
                cc += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
            return np.stack(cc, 1), None

    def fit(kind, order, Ypts, Ymeans, pidx):
        tp = tpos(order, Ymeans); B, P = basis(kind, tp[pidx])
        P = np.eye(B.shape[1]) if P is None else P
        coef, lam = gcv(B, P, Ypts)
        loop = basis(kind, np.linspace(0, 1, 400))[0] @ coef
        r2m = 1 - ((Ymeans - basis(kind, tp)[0] @ coef) ** 2).sum() / ((Ymeans - Ymeans.mean(0)) ** 2).sum()
        return loop, r2m

    rgbm = [f"rgb({int(rgb[c][0])},{int(rgb[c][1])},{int(rgb[c][2])})" for c in cols]
    rgbpt = [f"rgba({int(rgb[cols[i]][0])},{int(rgb[cols[i]][1])},{int(rgb[cols[i]][2])},0.45)" for i in ci]

    def page(fname, title, pts, means, loop, axlab=("PC1", "PC2", "PC3"), ptcols=None):
        fig = go.Figure()
        if pts is not None:
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                          marker=dict(size=4, color=ptcols if ptcols is not None else rgbpt),
                          name="prompt-removed prompts", hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode="lines",
                      line=dict(width=7, color="white"), name="loop"))
        fig.add_trace(go.Scatter3d(x=means[:, 0], y=means[:, 1], z=means[:, 2], mode="markers+text",
                      marker=dict(size=12, color=rgbm, line=dict(width=1.5, color="white")),
                      text=cols, textfont=dict(color="white", size=10), name="colors"))
        fig.update_layout(template="plotly_dark", title=title, showlegend=False,
                          scene=dict(xaxis_title=axlab[0], yaxis_title=axlab[1], zaxis_title=axlab[2]))
        fig.write_html(out / fname); return title

    pages = []
    for kind, lbl in [("bspline", "periodic B-spline"), ("duchon", "periodic Duchon"), ("fourier", "Fourier (5 harm.)")]:
        loop, r2m = fit(kind, order_rep, Yadj, Ycol, ci)
        pages.append((f"pc_{kind}.html", page(f"pc_{kind}.html",
                      f"{lbl} · rep-optimal order · PC axes · R²(color)={r2m:.2f}", Yadj, Ycol, loop)))
    loop, r2m = fit("bspline", order_color, Yadj, Ycol, ci)
    pages.append(("pc_nearestcolor.html", page("pc_nearestcolor.html",
                  f"nearest-COLOR order · B-spline · PC axes · R²(color)={r2m:.2f}", Yadj, Ycol, loop)))
    try:
        import umap
        emb = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.25, random_state=0, metric="cosine").fit_transform(Hadj)
        embm = np.stack([emb[ci == i].mean(0) for i in range(n)])
        for order, oname in [(order_color, "nearest-COLOR"), (nn_cycle(cdist(embm, embm)), "rep-optimal")]:
            tp = tpos(order, embm); B = np.asarray(gamfit.periodic_spline_curve_basis(tp[ci], 12)[0])
            coef, _ = gcv(B, np.eye(12), emb)
            loop = np.asarray(gamfit.periodic_spline_curve_basis(np.linspace(0, 1, 400), 12)[0]) @ coef
            r2m = 1 - ((embm - np.asarray(gamfit.periodic_spline_curve_basis(tp, 12)[0]) @ coef) ** 2).sum() / ((embm - embm.mean(0)) ** 2).sum()
            fn = f"umap_{oname.split('-')[0].lower()}.html"
            ptc = [f"rgba({int(rgb[cols[i]][0])},{int(rgb[cols[i]][1])},{int(rgb[cols[i]][2])},0.45)" for i in ci]
            pages.append((fn, page(fn, f"{oname} order · B-spline · UMAP-3D axes · R²(color)={r2m:.2f}",
                          emb, embm, loop, axlab=("UMAP1", "UMAP2", "UMAP3"), ptcols=ptc)))
    except Exception as e:
        print("UMAP skipped:", str(e)[:80])

    idx = ["<html><head><style>body{background:#111;color:#eee;font:16px/1.5 sans-serif;padding:34px;max-width:760px}"
           "a{color:#7cf;display:block;margin:9px 0;text-decoration:none}a:hover{color:#fff}h1{font-weight:300}"
           "table{margin:10px 0 22px;border-collapse:collapse}td{padding:3px 14px 3px 0}b{color:#fd8}</style>"
           "<title>color loops</title></head><body>",
           "<h1>color loops — one page per fit (RL3.1 L44, 24 vivid colors)</h1>",
           "<p>Prompt template modeled out (<code>rep ~ loop(t_color) + frame</code>); each color's 6 prompts "
           "share one position, so the loop fits <b>color</b>, not prompt. Variance partition:</p><table>"
           f"<tr><td>COLOR (signal)</td><td><b>{part[0]:.1%}</b></td></tr>"
           f"<tr><td>PROMPT / frame (modeled out)</td><td><b>{part[1]:.1%}</b></td></tr>"
           f"<tr><td>residual (interaction+noise)</td><td><b>{part[2]:.1%}</b></td></tr></table>"]
    for fn, title in pages:
        idx.append(f'<a href="{fn}">▸ {title}</a>')
    idx.append("</body></html>")
    (out / "index.html").write_text("\n".join(idx))
    print(f"wrote {len(pages)} pages + index -> {out}/index.html")
    import subprocess
    subprocess.run(["open", str(out / "index.html")])


if __name__ == "__main__":
    main()
