"""auto_33: (jjjj) Cross-template centroid correlation matrix (28x28).

Probes information leakage / template-style structure in the L40 cogito
residuals: for each of the 28 templates, build a centroid by averaging
the per-prompt residual across the 949 colors. Then compute the 28x28
Pearson correlation matrix between template centroids (in raw 7168-d
residual space, and again after subtracting the grand mean across all
prompts to remove the dominant "always-on" direction). High off-diagonal
correlations would mean templates share a lot of common style direction
on top of any color-conditional signal -- evidence of leakage / redundancy
in the per-color centroid (which averages over templates).

Also plot the eigenvalue spectrum of each correlation matrix to visualise
"effective number of template directions".

Hard-constraint compliant: no Gaussian RBF, no length_scale on Duchon,
no Duchon at all here -- just numpy means + Pearson correlation + linear
PCA-style eigendecomposition on the 28x28 matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN_DIR / "auto_33.png"
X_PATH = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"


def short_template(t: str, n: int = 38) -> str:
    s = t.replace("{x}", "X")
    return s if len(s) <= n else s[: n - 1] + "..."


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)  # 28
    X = np.load(X_PATH, mmap_mode="r")  # (n_c*n_t, D)
    n_rows, D = X.shape
    n_c = n_rows // n_t
    print(f"[data] X={X.shape}, n_colors={n_c}, n_templates={n_t}")
    assert n_c * n_t == n_rows, "row count must be n_colors*n_templates (color-major)"

    # Per-template centroid: mean over colors. Iterate to keep RAM modest.
    cent = np.zeros((n_t, D), dtype=np.float64)
    # Row i has color = i // n_t, template = i % n_t (color-major order
    # confirmed in color_manifold_gam.py L2041-2044).
    block = 4096
    counts = np.zeros(n_t, dtype=np.int64)
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X[s:e], dtype=np.float64)
        idx = np.arange(s, e) % n_t
        for t in range(n_t):
            m = idx == t
            if m.any():
                cent[t] += chunk[m].sum(axis=0)
                counts[t] += int(m.sum())
    cent /= counts[:, None]
    print(f"[centroids] per-template counts: min={counts.min()} max={counts.max()}")

    grand = cent.mean(axis=0, keepdims=True)
    cent_dm = cent - grand  # subtract grand mean over templates

    def corr(M: np.ndarray) -> np.ndarray:
        Mz = M - M.mean(axis=1, keepdims=True)
        n = np.linalg.norm(Mz, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Mn = Mz / n
        return Mn @ Mn.T

    C_raw = corr(cent)
    C_dm = corr(cent_dm)

    # Eigenvalues of correlation matrices (positive semi-def, trace = n_t)
    eig_raw = np.sort(np.linalg.eigvalsh(C_raw))[::-1]
    eig_dm = np.sort(np.linalg.eigvalsh(C_dm))[::-1]

    # Off-diag stats
    iu = np.triu_indices(n_t, k=1)
    off_raw = C_raw[iu]
    off_dm = C_dm[iu]

    # Hierarchical-ish ordering: sort templates by 1st eigenvector loading on
    # the grand-mean-subtracted correlation matrix
    w, V = np.linalg.eigh(C_dm)
    order = np.argsort(V[:, -1])
    C_raw_o = C_raw[np.ix_(order, order)]
    C_dm_o = C_dm[np.ix_(order, order)]
    labels_o = [f"{i:2d} {short_template(templates[i])}" for i in order]

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 0.85],
                          height_ratios=[1, 0.45], hspace=0.35, wspace=0.35)

    # Heatmap 1: raw centroid correlations
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(C_raw_o, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(f"Raw centroid corr (mean |off-diag|={np.mean(np.abs(off_raw)):.3f},\n"
                 f"mean off-diag={off_raw.mean():.3f})")
    ax.set_xticks(range(n_t)); ax.set_yticks(range(n_t))
    ax.set_xticklabels([str(i) for i in order], fontsize=6)
    ax.set_yticklabels(labels_o, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Heatmap 2: grand-mean-subtracted
    ax = fig.add_subplot(gs[0, 1])
    vmax = max(0.1, float(np.max(np.abs(C_dm))))
    im = ax.imshow(C_dm_o, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(f"After subtracting grand template-mean\n"
                 f"(mean |off-diag|={np.mean(np.abs(off_dm)):.3f}, "
                 f"mean off-diag={off_dm.mean():.3f})")
    ax.set_xticks(range(n_t)); ax.set_yticks(range(n_t))
    ax.set_xticklabels([str(i) for i in order], fontsize=6)
    ax.set_yticklabels(labels_o, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Eigenvalue spectra
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.arange(1, n_t + 1), eig_raw, "o-", color="#1f77b4",
            lw=1.4, ms=4, label=f"raw  (eig1={eig_raw[0]:.2f})")
    ax.plot(np.arange(1, n_t + 1), eig_dm, "s-", color="#d62728",
            lw=1.4, ms=4, label=f"grand-mean-subtracted  (eig1={eig_dm[0]:.2f})")
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue of 28x28 corr matrix")
    ax.set_title("Spectrum (sum = 28)\nflat = templates fully decorrelated, spiked = leakage")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Histogram of off-diag
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(off_raw, bins=40, color="#1f77b4", alpha=0.7, label="raw")
    ax.hist(off_dm, bins=40, color="#d62728", alpha=0.7, label="grand-mean-subtracted")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("off-diagonal correlation")
    ax.set_ylabel("count")
    ax.set_title("Off-diagonal correlation distribution (378 pairs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Effective dimensionality
    ax = fig.add_subplot(gs[1, 1])
    cum_raw = np.cumsum(eig_raw) / eig_raw.sum()
    cum_dm = np.cumsum(eig_dm) / eig_dm.sum()

    def part_dim(eig: np.ndarray) -> float:
        p = eig / eig.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    pdim_raw = part_dim(eig_raw)
    pdim_dm = part_dim(eig_dm)
    ax.plot(np.arange(1, n_t + 1), cum_raw, "o-", color="#1f77b4",
            lw=1.4, ms=4, label=f"raw   (eff-dim={pdim_raw:.2f})")
    ax.plot(np.arange(1, n_t + 1), cum_dm, "s-", color="#d62728",
            lw=1.4, ms=4, label=f"dm  (eff-dim={pdim_dm:.2f})")
    ax.axhline(0.9, color="k", lw=0.5, ls=":")
    ax.set_xlabel("# components")
    ax.set_ylabel("cumulative variance share")
    ax.set_title("Cumulative spectrum + entropy-effective-dim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)

    # Template ranking by mean off-diag corr (after dm)
    ax = fig.add_subplot(gs[1, 2])
    mean_off_per_tpl = (C_dm.sum(axis=1) - np.diag(C_dm)) / (n_t - 1)
    order_b = np.argsort(mean_off_per_tpl)[::-1]
    y = np.arange(n_t)
    ax.barh(y, mean_off_per_tpl[order_b],
            color=["#d62728" if v > 0 else "#1f77b4" for v in mean_off_per_tpl[order_b]])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i:2d} {short_template(templates[i], 30)}" for i in order_b],
                       fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("mean off-diag corr (dm) with other 27 templates")
    ax.set_title("Per-template leakage score")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("(jjjj) Cross-template centroid correlation (28x28) on L40 cogito residuals\n"
                 "raw vs grand-mean-subtracted | high off-diag = templates share a common direction (leakage)",
                 fontsize=12, y=0.995)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}")

    # Summary
    print(f"\nRaw:  mean off-diag = {off_raw.mean():.4f}, "
          f"mean |off| = {np.mean(np.abs(off_raw)):.4f}, "
          f"min={off_raw.min():.3f}, max={off_raw.max():.3f}")
    print(f"dm :  mean off-diag = {off_dm.mean():.4f}, "
          f"mean |off| = {np.mean(np.abs(off_dm)):.4f}, "
          f"min={off_dm.min():.3f}, max={off_dm.max():.3f}")
    print(f"Top-1 eigenvalue: raw={eig_raw[0]:.3f}/{n_t}  dm={eig_dm[0]:.3f}/{n_t}")
    print(f"Effective dim (entropy): raw={pdim_raw:.2f}  dm={pdim_dm:.2f}")
    print(f"Top-5 leakiest templates (dm):")
    for i in order_b[:5]:
        print(f"  [{i:2d}] {mean_off_per_tpl[i]:+.3f}  {templates[i]}")
    print(f"Least-leaky templates (dm):")
    for i in order_b[-5:]:
        print(f"  [{i:2d}] {mean_off_per_tpl[i]:+.3f}  {templates[i]}")


if __name__ == "__main__":
    main()
