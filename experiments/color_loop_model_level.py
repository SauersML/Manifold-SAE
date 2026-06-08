"""Model-level prompt/color separation: ONE joint gamfit REML, no pre-subtraction.

Earlier we removed the prompt template by hand (subtract per-frame mean) before fitting the
loop. Here we instead let gamfit do it AT THE MODEL LEVEL: a single multi-block Gaussian REML
fit on all noisy prompt replicates with two terms

    rep ~ periodic-spline loop(t_color)      [smoothness penalty, lambda_s]
        + frame                              [ridge / random effect, lambda_f]

gamfit.gaussian_reml_fit_blocks_forward jointly estimates lambda_s (loop smoothness) and
lambda_f (prompt shrinkage) by REML. Because every color's 6 prompts share one position
t_color, the loop term predicts one point per color and is structurally unable to fit prompt;
the frame random effect soaks up the shared-template variance. The most useful response space
is --response colorpc: top-3 PCs of the prompt-free color means, while the model itself is fit
on the raw per-prompt observations projected into that color subspace. That lets gamfit see the
noise without letting the loop chase prompt templates. gamfit-only. Read-only.
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
    ap.add_argument("--K", type=int, default=30, help="raw-PC working dimension (holds prompt variance)")
    ap.add_argument("--order", choices=["color", "rep"], default="color")
    ap.add_argument("--response", choices=["colorpc", "rawpc"], default="colorpc",
                    help="colorpc fits the joint model in the top-3 prompt-free color PCs; rawpc uses raw top-K PCs")
    ap.add_argument("--html", default="/tmp/loop_model_level.html")
    a = ap.parse_args()
    X = np.load(Path(a.extra) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(a.extra) / "prompts.jsonl") if l.strip()]
    H = X[:, min(a.layer, X.shape[1] - 1), :].astype(np.float64)
    frame = np.array([r["frame"] for r in recs]); color = np.array([r["color"] for r in recs])
    rgb = {r["color"]: np.array(r["rgb"], float) for r in recs}
    keep = np.array([c in VIVID for c in color]); H, frame, color = H[keep], frame[keep], color[keep]
    cols = sorted(set(color)); n = len(cols); ci = np.array([cols.index(c) for c in color]); nf = len(set(frame))

    gm = H.mean(0)
    # loop order from prompt-free color means (only used to place t; not the fit)
    frmean = np.stack([H[frame == f].mean(0) for f in range(nf)])
    cadj = np.stack([(H - frmean[frame])[ci == i].mean(0) for i in range(n)])
    if a.response == "colorpc":
        Q = np.linalg.svd(cadj - cadj.mean(0), full_matrices=False)[2][:3]
    else:
        Q = np.linalg.svd(H - gm, full_matrices=False)[2][:a.K]
    Yraw = (H - gm) @ Q.T                                                       # n_obs x K, still includes prompt variance
    Cm = (cadj - cadj.mean(0)) @ Q.T
    Lab = np.stack([srgb2lab(rgb[c]) for c in cols])
    from scipy.spatial.distance import cdist
    order = nn_cycle(cdist(Lab, Lab)) if a.order == "color" else nn_cycle(cdist(Cm, Cm))
    seg = np.array([np.linalg.norm(Cm[order[i]] - Cm[order[(i + 1) % n]]) for i in range(n)])
    tp = np.zeros(n); tp[order] = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()

    # ---- design blocks ----
    Bc, Pc = gamfit.periodic_spline_curve_basis(tp[ci], 12); Bc = np.asarray(Bc); Pc = np.asarray(Pc)
    F = np.zeros((len(ci), nf)); F[np.arange(len(ci)), frame] = 1.0; Pf = np.eye(nf)
    nb = Bc.shape[1]

    # ---- joint REML per working-PC column ----
    K = Yraw.shape[1]
    coef_loop = np.zeros((nb, K)); coef_frm = np.zeros((nf, K))
    edf_loop = np.zeros(K); edf_frm = np.zeros(K); lam = np.zeros((K, 2))
    for k in range(K):
        r = gamfit.gaussian_reml_fit_blocks_forward(designs=[Bc, F], penalties=[Pc, Pf], y=Yraw[:, k])
        c = np.asarray(r["coefficients"]).ravel()
        coef_loop[:, k] = c[:nb]; coef_frm[:, k] = c[nb:nb + nf]
        ed = np.asarray(r["edf"]).ravel(); edf_loop[k], edf_frm[k] = ed[0], ed[1]
        lam[k] = np.asarray(r["lambdas"]).ravel()
    loop_fit = Bc @ coef_loop; frm_fit = F @ coef_frm                          # n_obs x K
    tot = (Yraw ** 2).sum()
    ss_loop = (loop_fit ** 2).sum(); ss_frm = (frm_fit ** 2).sum()
    resid = Yraw - loop_fit - frm_fit; ss_res = (resid ** 2).sum()
    Bm = np.asarray(gamfit.periodic_spline_curve_basis(tp, 12)[0])
    loop_means = Bm @ coef_loop
    r2_color = 1 - ((Cm - loop_means) ** 2).sum() / ((Cm - Cm.mean(0)) ** 2).sum()
    print(f"# model-level joint REML  ({a.response}, {K} response dims, {len(ci)} obs, order={a.order})")
    print(f"  LOOP  term : {ss_loop/tot:6.1%} variance   mean EDF={edf_loop.mean():.1f}   median lambda_s={np.median(lam[:,0]):.2g}")
    print(f"  FRAME term : {ss_frm/tot:6.1%} variance   mean EDF={edf_frm.mean():.1f}   median lambda_f={np.median(lam[:,1]):.2g}")
    print(f"  residual   : {ss_res/tot:6.1%} (color×frame interaction + noise)")
    print(f"  loop R² on prompt-free color means: {r2_color:.3f}")
    # cross-check vs hand pre-subtraction loop
    Badj, _ = gamfit.periodic_spline_curve_basis(tp[ci], 12)
    cf2 = np.linalg.solve(Badj.T @ Badj + np.median(lam[:, 0]) * Pc, Badj.T @ ((H - frmean[frame] - cadj.mean(0)) @ Q.T))
    agree = 1 - np.linalg.norm(Bc @ coef_loop - Badj @ cf2) / np.linalg.norm(Badj @ cf2)
    print(f"  model-level loop vs hand pre-subtraction loop: {agree:.1%} agreement")

    # ---- project model loop into a color-display frame (top-3 PC of prompt-free color means) ----
    Cm3basis = np.linalg.svd(Cm - Cm.mean(0), full_matrices=False)[2][:3]       # 3 x K
    grid = np.linspace(0, 1, 400)
    loop_curve = (np.asarray(gamfit.periodic_spline_curve_basis(grid, 12)[0]) @ coef_loop) @ Cm3basis.T
    pts = (Yraw - frm_fit) @ Cm3basis.T                                         # model-deprompted points
    means3 = (Cm - Cm.mean(0)) @ Cm3basis.T
    rgbm = [f"rgb({int(rgb[c][0])},{int(rgb[c][1])},{int(rgb[c][2])})" for c in cols]
    rgbpt = [f"rgba({int(rgb[cols[i]][0])},{int(rgb[cols[i]][1])},{int(rgb[cols[i]][2])},0.45)" for i in ci]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                  marker=dict(size=4, color=rgbpt), name="prompts (frame term removed by model)", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=loop_curve[:, 0], y=loop_curve[:, 1], z=loop_curve[:, 2], mode="lines",
                  line=dict(width=7, color="white"), name="model-fitted loop"))
    fig.add_trace(go.Scatter3d(x=means3[:, 0], y=means3[:, 1], z=means3[:, 2], mode="markers+text",
                  marker=dict(size=12, color=rgbm, line=dict(width=1.5, color="white")),
                  text=cols, textfont=dict(color="white", size=10), name="colors"))
    fig.update_layout(template="plotly_dark", showlegend=False,
                      title=f"model-level loop: rep ~ loop(t_color) + frame · joint REML · "
                            f"loop {ss_loop/tot:.0%} / frame {ss_frm/tot:.0%} / resid {ss_res/tot:.0%} · "
                            f"color R² {r2_color:.2f}",
                      scene=dict(xaxis_title="color-PC1", yaxis_title="color-PC2", zaxis_title="color-PC3"))
    fig.write_html(a.html); print("wrote", a.html)
    import subprocess; subprocess.run(["open", a.html])


if __name__ == "__main__":
    main()
