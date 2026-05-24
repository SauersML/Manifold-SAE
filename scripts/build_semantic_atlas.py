"""Build the cogito-L40 Manifold-SAE semantic atlas.

Outputs to runs/manifold_atlas/:
  atoms.json        — machine-readable list of AtomCards for the manifold SAE
  atoms_topk.json   — same for TopK baseline
  atoms_l1.json     — same for L1 baseline
  atoms.html        — browseable per-atom card grid (PHATE 2-D layout)
  index.html        — landing page with overall stats
  lineage.json      — cross-architecture concept lineage

CLI: `python scripts/build_semantic_atlas.py [--no-baselines] [--no-causal]`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.atlas.semantic_atlas import (
    build_semantic_atlas,
    cards_to_json,
    category_counts,
    AtomCard,
)
from manifold_sae.atlas.atom_lineage import build_lineage
from manifold_sae.atlas.phate_atlas import atom_atlas

# Re-use the model classes from the training script (avoids duplication).
# We import-by-exec because train_sae_comparison.py is not a clean module
# (it executes data loading at import time). Instead, replicate just the
# class bodies here for stand-alone use.

import torch.nn as nn
import torch.nn.functional as F


F_ATOMS = 512
N_COLORS = 949
N_TPL = 28


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_feat, top_k):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_feat))
        self.W_d = nn.Parameter(torch.randn(n_feat, d_in) * (1.0/np.sqrt(n_feat)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.top_k = top_k
    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        topv, topi = z.topk(self.top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topi, F.relu(topv))
        return z_sparse
    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_d + self.b_d, z


class L1SAE(nn.Module):
    def __init__(self, d_in, n_feat):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_e = nn.Parameter(torch.zeros(n_feat))
        self.W_d = nn.Parameter(torch.randn(n_feat, d_in) * (1.0/np.sqrt(n_feat)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        return F.relu(z)
    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_d + self.b_d, z


class ManifoldSAE(nn.Module):
    def __init__(self, d_in, n_feat, M_F=3):
        super().__init__()
        self.n_feat = n_feat; self.M_F = M_F
        self.W_gate = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        self.b_gate = nn.Parameter(torch.full((n_feat,), -2.0))
        self.W_theta = nn.Parameter(torch.randn(d_in, n_feat * 2) * (1.0/np.sqrt(d_in)))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_feat) * (1.0/np.sqrt(d_in)))
        basis_dim = 2 * M_F + 1
        self.D_k = nn.Parameter(torch.randn(n_feat, basis_dim, d_in) * (0.1/np.sqrt(basis_dim)))
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.log_ard = nn.Parameter(torch.zeros(n_feat))
    def theta(self, x):
        xc = x - self.b_d
        tp = xc @ self.W_theta
        B = x.shape[0]
        tp = tp.view(B, self.n_feat, 2)
        tp = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tp
    def fourier_basis(self, cs):
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c)]
        ck, sk = c.clone(), s.clone()
        feats += [ck, sk]
        for m in range(2, self.M_F + 1):
            ck_new = ck * c - sk * s
            sk_new = sk * c + ck * s
            ck, sk = ck_new, sk_new
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)
    def forward(self, x, tau=1.0, hard=False):
        xc = x - self.b_d
        gate_logit = xc @ self.W_gate + self.b_gate
        gate = torch.sigmoid(gate_logit)
        amp_raw = xc @ self.W_amp
        amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
        cs = self.theta(x)
        phi = self.fourier_basis(cs)
        w = gate * amp
        w_phi = (w.unsqueeze(-1) * phi).reshape(x.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        return w_phi @ D_flat + self.b_d, gate, amp
    def encode_for_eval(self, x):
        with torch.no_grad():
            xc = x - self.b_d
            gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
            amp_raw = xc @ self.W_amp
            amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
            return gate * amp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_xkcd_colors():
    p = ROOT / "experiments" / "xkcd_colors.txt"
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hex_ = parts[0], parts[1]
            hex_ = hex_.lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out.append((name, r, g, b))
    return out


def load_data():
    X_path = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
    X = np.load(X_path, mmap_mode="r")
    N, D = X.shape
    assert N == N_COLORS * N_TPL, (N, N_COLORS, N_TPL)
    rng = np.random.default_rng(0)
    color_perm = rng.permutation(N_COLORS)
    n_val_colors = int(0.2 * N_COLORS)
    val_colors = set(color_perm[:n_val_colors].tolist())
    train_colors = set(color_perm[n_val_colors:].tolist())
    row_color_full = np.arange(N) // N_TPL
    row_template_full = np.arange(N) % N_TPL
    train_idx = np.where(np.isin(row_color_full, list(train_colors)))[0]
    val_idx = np.where(np.isin(row_color_full, list(val_colors)))[0]
    X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
    X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)
    mu = X_train_np.mean(0)
    X_val_np -= mu
    val_var = float((X_val_np ** 2).sum() / X_val_np.size)
    return {
        "X_val": X_val_np,
        "val_var": val_var,
        "row_color": row_color_full[val_idx],
        "row_template": row_template_full[val_idx],
        "D": D,
    }


def load_model(path: Path, kind: str, d_in: int):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "manifold":
        m = ManifoldSAE(d_in, F_ATOMS, M_F=3)
    elif kind == "topk":
        m = TopKSAE(d_in, F_ATOMS, top_k=32)
    elif kind == "l1":
        m = L1SAE(d_in, F_ATOMS)
    else:
        raise ValueError(kind)
    m.load_state_dict(sd, strict=False)
    return m.eval()


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _hsv_to_hex(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, max(0, min(1, s)), max(0, min(1, v)))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2])
    )


def _category_color(cat: str) -> str:
    return {
        "hue-arc":           "#ff6b6b",
        "lightness-band":    "#4ecdc4",
        "name-token":        "#f9c74f",
        "modifier-count":    "#9d4edd",
        "template-specific": "#90be6d",
        "dead":              "#888888",
        "polysemantic":      "#577590",
    }.get(cat, "#cccccc")


def render_atoms_html(
    cards: list[AtomCard],
    embedding: np.ndarray,
    color_rgb: np.ndarray,
    out_path: Path,
) -> None:
    """Single-file HTML: 2-D scatter (PHATE/spectral) + clickable cards."""
    # normalise embedding to viewport
    if embedding.size:
        emb = embedding.copy()
        emb -= emb.min(0, keepdims=True)
        rng = emb.max(0, keepdims=True)
        rng[rng < 1e-9] = 1.0
        emb /= rng
    else:
        emb = np.zeros((len(cards), 2))

    W, H = 1100, 720
    PAD = 40

    dots = []
    for i, c in enumerate(cards):
        if c.n_active == 0:
            x, y = PAD + 4, PAD + 4 + (i % 60) * 8
            color = "#222"
        else:
            x = PAD + emb[i, 0] * (W - 2 * PAD) if emb.shape[0] > i else PAD
            y = PAD + emb[i, 1] * (H - 2 * PAD) if emb.shape[0] > i else PAD
            color = _hsv_to_hex(*c.hsv_centroid)
        r = 3 + 18 * float(np.clip(c.causal_delta_r2 * 50.0, 0, 1))
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="{color}" stroke="{_category_color(c.category)}" '
            f'stroke-width="1.5" data-atom="{c.atom_id}" '
            f'onclick="showAtom({c.atom_id})" '
            f'style="cursor:pointer; opacity:0.85"><title>'
            f'atom {c.atom_id} | {c.category} | Δr²={c.causal_delta_r2:.4f}'
            f'</title></circle>'
        )

    # per-atom hidden JSON
    atom_data = {}
    for c in cards:
        swatches = []
        for e in c.top_examples[:5]:
            ci = e["color_idx"]
            swatches.append({
                "color": e["color"],
                "rgb": _rgb_to_hex(color_rgb[ci]),
                "template": e["template_id"],
                "act": e["act"],
            })
        atom_data[c.atom_id] = {
            "atom_id": c.atom_id,
            "category": c.category,
            "explanation": c.explanation,
            "explanation_source": c.explanation_source,
            "n_active": c.n_active,
            "hsv_centroid": list(c.hsv_centroid),
            "hue_arc_span": c.hue_arc_span,
            "lightness_span": c.lightness_span,
            "saturation_span": c.saturation_span,
            "causal_delta_r2": c.causal_delta_r2,
            "swatches": swatches,
            "name_top_tokens": c.name_top_tokens,
            "template_concentration": c.template_concentration,
            "top_template_id": c.top_template_id,
        }

    cats = sorted(set(c.category for c in cards))
    legend = "".join(
        f'<span class="legend-item"><span class="dot" style="background:{_category_color(c)}"></span>{c}</span>'
        for c in cats
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Manifold-SAE Semantic Atlas — cogito-L40</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
       background:#0e0e12; color:#ddd; margin:0; padding:24px; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
.sub {{ color:#888; font-size:13px; margin-bottom:14px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:10px; font-size:12px; }}
.legend .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
                margin-right:5px; vertical-align:middle; }}
.layout {{ display:grid; grid-template-columns: {W+20}px 360px; gap:20px; }}
svg {{ background:#16161d; border-radius:8px; }}
#card {{ background:#16161d; border-radius:8px; padding:18px; min-height:{H}px;
        font-size:13px; line-height:1.45; }}
#card h2 {{ margin-top:0; font-size:15px; }}
.swatch-row {{ display:flex; gap:6px; margin:10px 0; }}
.swatch {{ width:48px; height:48px; border-radius:6px; border:1px solid #333; }}
.k {{ color:#888; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
       color:#fff; }}
.bigmetric {{ font-size:22px; margin:2px 0 14px; }}
a {{ color:#9cf; }}
</style></head><body>
<h1>Manifold-SAE Semantic Atlas — cogito-L40</h1>
<div class="sub">F={len(cards)} atoms. Each dot = one atom; color = HSV centroid of its top-20 colors;
ring color = category; radius ∝ causal Δ-R². Click a dot.</div>
<div class="legend">{legend}</div>
<div class="layout">
<svg width="{W}" height="{H}" id="map" viewBox="0 0 {W} {H}">
{''.join(dots)}
</svg>
<div id="card"><i>Click an atom dot.</i></div>
</div>
<p><a href="index.html">← back to atlas index</a></p>
<script>
const ATOMS = {json.dumps(atom_data)};
function showAtom(id) {{
  const a = ATOMS[id];
  if (!a) return;
  const sw = a.swatches.map(s =>
    `<div class="swatch" style="background:${{s.rgb}}" title="${{s.color}} (template ${{s.template}}) act=${{s.act.toFixed(2)}}"></div>`
  ).join("");
  const tags = `<span class="tag" style="background:${{catColor(a.category)}}">${{a.category}}</span>`;
  document.getElementById("card").innerHTML = `
    <h2>atom ${{a.atom_id}} ${{tags}}</h2>
    <div class="bigmetric">Δ R² = ${{a.causal_delta_r2.toFixed(5)}}</div>
    <div><b>Explanation</b> <span class="k">(${{a.explanation_source}})</span></div>
    <div>${{a.explanation}}</div>
    <div style="margin-top:10px"><b>Top-5 colors</b></div>
    <div class="swatch-row">${{sw}}</div>
    <div><span class="k">HSV centroid</span> H=${{(a.hsv_centroid[0]).toFixed(3)}} S=${{a.hsv_centroid[1].toFixed(3)}} V=${{a.hsv_centroid[2].toFixed(3)}}</div>
    <div><span class="k">Hue arc span</span> ${{a.hue_arc_span.toFixed(3)}}</div>
    <div><span class="k">Lightness span</span> ${{a.lightness_span.toFixed(3)}}</div>
    <div><span class="k">Saturation span</span> ${{a.saturation_span.toFixed(3)}}</div>
    <div><span class="k">N active (val)</span> ${{a.n_active}}</div>
    <div><span class="k">Template concentration</span> ${{a.template_concentration.toFixed(3)}} (top template = ${{a.top_template_id}})</div>
    <div><span class="k">Name tokens</span> ${{a.name_top_tokens.join(", ") || "—"}}</div>
  `;
}}
function catColor(c) {{
  return ({json.dumps({k: _category_color(k) for k in ["hue-arc","lightness-band","name-token","modifier-count","template-specific","dead","polysemantic"]})})[c] || "#ccc";
}}
</script>
</body></html>
"""
    out_path.write_text(html)


def render_index_html(
    cards: list[AtomCard],
    out_dir: Path,
) -> None:
    counts = category_counts(cards)
    n_total = len(cards)
    n_dead = counts.get("dead", 0)
    n_poly = counts.get("polysemantic", 0)
    n_hue = counts.get("hue-arc", 0)
    n_light = counts.get("lightness-band", 0)
    n_alive = n_total - n_dead

    # top causal atoms
    causal_sorted = sorted(cards, key=lambda c: -c.causal_delta_r2)[:10]
    causal_rows = "".join(
        f"<tr><td>{c.atom_id}</td><td><span class='tag' style='background:{_category_color(c.category)}'>{c.category}</span></td>"
        f"<td>{c.causal_delta_r2:.5f}</td><td style='background:{_rgb_to_hex(c.top_color_rgb)};width:24px'></td>"
        f"<td>{c.top_color}</td><td>{c.explanation}</td></tr>"
        for c in causal_sorted
    )

    hue_atoms = sorted(
        (c for c in cards if c.category == "hue-arc"),
        key=lambda c: c.hsv_centroid[0],
    )
    hue_ring = "".join(
        f'<div title="atom {c.atom_id} H={c.hsv_centroid[0]:.2f}" '
        f'style="width:18px;height:18px;border-radius:50%;background:{_hsv_to_hex(*c.hsv_centroid)};'
        f'border:1px solid #333"></div>'
        for c in hue_atoms
    )

    light_atoms = sorted(
        (c for c in cards if c.category == "lightness-band"),
        key=lambda c: c.hsv_centroid[2],
    )
    light_band = "".join(
        f'<div title="atom {c.atom_id} V={c.hsv_centroid[2]:.2f}" '
        f'style="width:18px;height:18px;border-radius:4px;background:{_hsv_to_hex(*c.hsv_centroid)};'
        f'border:1px solid #333"></div>'
        for c in light_atoms
    )

    cat_rows = "".join(
        f"<tr><td><span class='tag' style='background:{_category_color(k)}'>{k}</span></td>"
        f"<td>{counts.get(k,0)}</td><td>{100*counts.get(k,0)/max(1,n_total):.1f}%</td></tr>"
        for k in ["hue-arc","lightness-band","name-token","modifier-count",
                  "template-specific","polysemantic","dead"]
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Manifold-SAE Semantic Atlas — Index</title>
<style>
body {{ font-family:-apple-system,Helvetica,Arial,sans-serif;
       background:#0e0e12; color:#ddd; margin:0; padding:32px; max-width:1100px; }}
h1 {{ font-size:24px; }}
h2 {{ font-size:18px; margin-top:32px; border-bottom:1px solid #333; padding-bottom:6px; }}
table {{ border-collapse:collapse; font-size:13px; }}
td, th {{ padding:6px 10px; border-bottom:1px solid #222; }}
th {{ text-align:left; color:#aaa; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; color:#fff; }}
.bignum {{ font-size:28px; font-weight:600; }}
.kpi {{ display:inline-block; min-width:160px; margin-right:24px; vertical-align:top; }}
.kpi .l {{ color:#888; font-size:12px; }}
.group {{ display:flex; flex-wrap:wrap; gap:3px; }}
a {{ color:#9cf; }}
</style></head><body>
<h1>Manifold-SAE Semantic Atlas — cogito-L40</h1>
<p>Per-atom explanations + causal scores + cross-architecture lineage for the
F=512 R²=0.913 Manifold-SAE on the cogito-L40 color manifold.
<a href="atoms.html">Open the browseable 2-D atlas →</a></p>

<h2>Top-line stats</h2>
<div>
<div class="kpi"><div class="l">total atoms</div><div class="bignum">{n_total}</div></div>
<div class="kpi"><div class="l">alive atoms</div><div class="bignum">{n_alive}</div></div>
<div class="kpi"><div class="l">dead atoms</div><div class="bignum">{n_dead}</div></div>
<div class="kpi"><div class="l">polysemantic</div><div class="bignum">{n_poly}</div></div>
<div class="kpi"><div class="l">hue-arc atoms</div><div class="bignum">{n_hue}</div></div>
<div class="kpi"><div class="l">lightness-band atoms</div><div class="bignum">{n_light}</div></div>
</div>

<h2>Categorization</h2>
<table><tr><th>category</th><th>count</th><th>fraction</th></tr>{cat_rows}</table>

<h2>Top-10 most-causal atoms (by Δ R² when zeroed)</h2>
<table><tr><th>atom</th><th>category</th><th>Δ R²</th><th></th><th>top color</th><th>explanation</th></tr>{causal_rows}</table>

<h2>Hue-ring atoms (ordered by hue)</h2>
<div class="group">{hue_ring}</div>

<h2>Lightness-band atoms (ordered by V)</h2>
<div class="group">{light_band}</div>

<h2>Files</h2>
<ul>
<li><a href="atoms.html">atoms.html</a> — interactive 2-D atlas</li>
<li><code>atoms.json</code> — machine-readable cards (manifold)</li>
<li><code>atoms_topk.json</code>, <code>atoms_l1.json</code> — baseline cards</li>
<li><code>lineage.json</code> — cross-architecture concept lineage</li>
</ul>
</body></html>
"""
    (out_dir / "index.html").write_text(html)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--causal-batch", type=int, default=1024)
    args = ap.parse_args()

    out_dir = ROOT / "runs" / "manifold_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[build] loading data ...", flush=True)
    data = load_data()
    D = data["D"]
    print(f"[build] D={D} val_rows={data['X_val'].shape[0]} val_var={data['val_var']:.4f}",
          flush=True)

    xkcd = load_xkcd_colors()[:N_COLORS]
    color_names = [c[0] for c in xkcd]
    color_rgb = np.array([(r/255.0, g/255.0, b/255.0) for _, r, g, b in xkcd],
                         dtype=np.float32)

    sae_dir = ROOT / "runs" / "sae_comparison"
    device = "cpu"  # build on CPU for stability; F=512 is small

    llm_path = ROOT / "runs" / "autointerp_llm" / "explanations_manifold.jsonl"

    # ---- manifold ----
    print(f"[build] loading manifold SAE ...", flush=True)
    m_man = load_model(sae_dir / "model_manifold.pt", "manifold", D).to(device)
    print(f"[build] building manifold cards ...", flush=True)
    man_cards = build_semantic_atlas(
        sae_module=m_man,
        model_kind="manifold",
        X_val_np=data["X_val"],
        val_var=data["val_var"],
        row_color=data["row_color"],
        row_template=data["row_template"],
        color_names=color_names,
        color_rgb=color_rgb,
        device=device,
        llm_explanations_path=llm_path,
        compute_causal=not args.no_causal,
        causal_batch=args.causal_batch,
    )
    with open(out_dir / "atoms.json", "w") as f:
        json.dump(cards_to_json(man_cards), f)
    print(f"[build] manifold cards saved ({time.time()-t0:.1f}s)", flush=True)

    counts = category_counts(man_cards)
    print(f"[build] category counts: {counts}", flush=True)

    # ---- PHATE/spectral embedding for layout ----
    print(f"[build] embedding atoms ...", flush=True)
    try:
        atlas = atom_atlas(sae_dir / "model_manifold.pt", n_components=2, knn=15)
        emb_alive = atlas["embedding"]
        alive_mask = atlas["alive_mask"]
        # scatter alive points into a (F, 2) array
        emb = np.zeros((len(man_cards), 2))
        emb[alive_mask] = emb_alive
        emb[~alive_mask] = np.nan
    except Exception as e:
        print(f"[build] embedding failed: {e}; using random layout", flush=True)
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(len(man_cards), 2))

    # replace NaN (dead) with corner positions
    if np.isnan(emb).any():
        for i in range(emb.shape[0]):
            if np.isnan(emb[i]).any():
                emb[i] = [-2.0, -2.0 + 0.01 * (i % 60)]

    render_atoms_html(man_cards, emb, color_rgb, out_dir / "atoms.html")
    render_index_html(man_cards, out_dir)
    print(f"[build] HTML rendered", flush=True)

    # ---- baselines for lineage ----
    if not args.no_baselines:
        print(f"[build] loading TopK ...", flush=True)
        m_topk = load_model(sae_dir / "model_topk.pt", "topk", D).to(device)
        topk_cards = build_semantic_atlas(
            sae_module=m_topk, model_kind="topk",
            X_val_np=data["X_val"], val_var=data["val_var"],
            row_color=data["row_color"], row_template=data["row_template"],
            color_names=color_names, color_rgb=color_rgb,
            device=device, compute_causal=False,
        )
        with open(out_dir / "atoms_topk.json", "w") as f:
            json.dump(cards_to_json(topk_cards), f)

        print(f"[build] loading L1 ...", flush=True)
        m_l1 = load_model(sae_dir / "model_l1.pt", "l1", D).to(device)
        l1_cards = build_semantic_atlas(
            sae_module=m_l1, model_kind="l1",
            X_val_np=data["X_val"], val_var=data["val_var"],
            row_color=data["row_color"], row_template=data["row_template"],
            color_names=color_names, color_rgb=color_rgb,
            device=device, compute_causal=False,
        )
        with open(out_dir / "atoms_l1.json", "w") as f:
            json.dump(cards_to_json(l1_cards), f)

        print(f"[build] building lineage ...", flush=True)
        lineage = build_lineage(
            manifold_cards=man_cards, topk_cards=topk_cards, l1_cards=l1_cards, k=3,
        )
        with open(out_dir / "lineage.json", "w") as f:
            json.dump(lineage, f)

    # ---- summary ----
    causal_sorted = sorted(man_cards, key=lambda c: -c.causal_delta_r2)[:3]
    summary = {
        "n_atoms": len(man_cards),
        "category_counts": counts,
        "top3_causal": [
            {
                "atom_id": c.atom_id,
                "category": c.category,
                "delta_r2": c.causal_delta_r2,
                "top_color": c.top_color,
                "explanation": c.explanation,
            }
            for c in causal_sorted
        ],
        "wall_seconds": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[build] DONE in {time.time()-t0:.1f}s", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
