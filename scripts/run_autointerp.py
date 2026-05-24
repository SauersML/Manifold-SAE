"""Run the autointerp pipeline on the three trained SAEs.

Inputs:
  runs/sae_comparison/model_{topk,l1,manifold}.pt
  runs/COLOR_COGITO_L40/X_L40.npy
  experiments/xkcd_colors.txt

Outputs:
  runs/autointerp/report.md
  runs/autointerp/hypotheses.jsonl
  runs/autointerp/summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Pull in the model definitions from train_sae_comparison.
# Importing it will execute training-side code, so we instead read the file
# and re-define the classes inline (cheap; ~60 lines).
import torch.nn as nn
import torch.nn.functional as F


F_ATOMS = 512


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
        recon = z @ self.W_d + self.b_d
        return recon, z


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
        recon = z @ self.W_d + self.b_d
        return recon, z


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
        w = (gate * amp).unsqueeze(-1)
        w_phi = (w * phi).reshape(x.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        recon = w_phi @ D_flat + self.b_d
        return recon, gate, amp
    def encode_for_eval(self, x):
        with torch.no_grad():
            xc = x - self.b_d
            gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
            amp_raw = xc @ self.W_amp
            amp = F.softplus(amp_raw) * torch.exp(self.log_ard)
            return gate * amp


# ----------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------

from manifold_sae.autointerp.explain import (
    rgb_to_hsv,
    load_sae_activations,
    collect_top_activating,
    hypothesize_atom,
    causal_score_atom,
    hypothesis_to_dict,
)
from manifold_sae.autointerp.score import (
    score_hypothesis,
    aggregate_model_scores,
    bootstrap_ci,
)


def load_xkcd_colors(path: Path, n: int) -> tuple[list[str], np.ndarray]:
    out_names = []
    out_rgb = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hex_ = parts[0], parts[1].lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out_names.append(name)
            out_rgb.append((r/255.0, g/255.0, b/255.0))
            if len(out_names) >= n:
                break
    return out_names, np.array(out_rgb, dtype=np.float32)


def make_split(n_colors: int = 949, n_templates: int = 28, seed: int = 0):
    """Reproduces train/val split from scripts/train_sae_comparison.py."""
    N = n_colors * n_templates
    rng = np.random.default_rng(seed)
    color_perm = rng.permutation(n_colors)
    n_val_colors = int(0.2 * n_colors)
    val_colors = set(color_perm[:n_val_colors].tolist())
    train_colors = set(color_perm[n_val_colors:].tolist())
    row_color = np.arange(N) // n_templates
    row_template = np.arange(N) % n_templates
    train_idx = np.where(np.isin(row_color, list(train_colors)))[0]
    val_idx = np.where(np.isin(row_color, list(val_colors)))[0]
    return train_idx, val_idx, row_color, row_template


def main():
    out_dir = ROOT / "runs" / "autointerp"
    out_dir.mkdir(parents=True, exist_ok=True)
    sae_dir = ROOT / "runs" / "sae_comparison"

    device = "cpu"  # MPS fine but CPU avoids extra GPU loading; no cluster rule

    N_ATOMS_PER_MODEL = int(os.environ.get("N_ATOMS", "60"))
    N_TOP = int(os.environ.get("N_TOP", "20"))
    N_CAUSAL = int(os.environ.get("N_CAUSAL", "20"))

    print(f"[setup] N_ATOMS={N_ATOMS_PER_MODEL} N_TOP={N_TOP} N_CAUSAL={N_CAUSAL}")

    # ------- data -------
    X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
    N, D = X.shape
    N_COLORS, N_TPL = 949, 28
    train_idx, val_idx, row_color_all, row_template_all = make_split(N_COLORS, N_TPL)

    X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
    X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)
    mu = X_train_np.mean(0)
    X_train_np -= mu
    X_val_np -= mu
    val_var = float((X_val_np ** 2).mean())

    row_color = row_color_all[val_idx]
    row_template = row_template_all[val_idx]

    color_names, color_rgb = load_xkcd_colors(
        ROOT / "experiments" / "xkcd_colors.txt", n=N_COLORS,
    )
    color_hsv = rgb_to_hsv(color_rgb)
    print(f"[data] X shape={X.shape}  val rows={len(val_idx)}  colors={len(color_names)}")

    # ------- per-model loop -------
    model_specs = [
        ("TopK", "topk", sae_dir / "model_topk.pt",
         lambda: TopKSAE(D, F_ATOMS, top_k=32)),
        ("L1", "l1", sae_dir / "model_l1.pt",
         lambda: L1SAE(D, F_ATOMS)),
        ("Manifold", "manifold", sae_dir / "model_manifold.pt",
         lambda: ManifoldSAE(D, F_ATOMS, M_F=3)),
    ]

    all_hypotheses: list[dict] = []
    per_model_summary: dict[str, dict] = {}
    per_model_r2s: dict[str, list[float]] = {}
    per_model_atoms: dict[str, list[dict]] = {}

    for model_name, kind, ckpt_path, ctor in model_specs:
        t0 = time.time()
        print(f"\n=== {model_name} ===")
        sae = ctor().to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        sae.load_state_dict(state)
        sae.eval()

        # full-val activations
        acts_val = load_sae_activations(sae, X_val_np, kind, device=device, batch_size=1024)

        # pick atoms: top by activeness (firing rate)
        firing_rate = (acts_val > 1e-3).mean(0)
        atom_order = np.argsort(-firing_rate)
        chosen_atoms = atom_order[:N_ATOMS_PER_MODEL]
        # subset for causal (most expensive)
        causal_atoms = set(chosen_atoms[:N_CAUSAL].tolist())

        per_atom_scores: list[dict] = []
        per_atom_records: list[dict] = []
        for j, atom_id in enumerate(chosen_atoms):
            atom_id = int(atom_id)
            top_ex = collect_top_activating(
                acts_val, atom_id, row_color, row_template, color_names, n_top=N_TOP,
            )
            h = hypothesize_atom(
                atom_id, model_name, top_ex, color_hsv, color_names, n_templates=N_TPL,
            )
            # simulation R²
            sc = score_hypothesis(h, acts_val, color_hsv, color_names, row_color, row_template)
            # causal score on top subset
            if atom_id in causal_atoms:
                d_r2, d_cos = causal_score_atom(
                    sae, X_val_np, val_var, kind, atom_id, device=device, batch_size=1024,
                )
                h.causal_delta_r2 = d_r2
                h.causal_delta_cosine = d_cos
            per_atom_scores.append({**sc, "atom_id": atom_id})
            record = hypothesis_to_dict(h)
            record["simulation_r2"] = sc["r2"]
            record["firing_rate"] = float(firing_rate[atom_id])
            per_atom_records.append(record)
            all_hypotheses.append(record)
            if j % 10 == 0:
                print(f"  atom {atom_id:3d}  n_active={sc['n_active']:4d}  sim_R²={sc['r2']:+.3f}  "
                      f"ΔR²={h.causal_delta_r2:+.4f}  expl={h.explanation[:60]}")

        agg = aggregate_model_scores(per_atom_scores, min_active=5)
        # bootstrap CI on mean simulation R² among qualifying atoms
        r2s = agg.get("r2_array", [])
        pt, lo, hi = bootstrap_ci(r2s, n_boot=2000, statistic="mean", seed=0)
        agg["bootstrap_mean_r2"] = pt
        agg["ci95_low"] = lo
        agg["ci95_high"] = hi
        per_model_summary[model_name] = agg
        per_model_r2s[model_name] = r2s
        per_model_atoms[model_name] = per_atom_records
        print(f"  median_R²={agg['median_r2']:+.3f}  mean_R²={agg['mean_r2']:+.3f}  "
              f"95% CI=({lo:+.3f},{hi:+.3f})  evaluated={agg['n_atoms_evaluated']}  "
              f"t={time.time()-t0:.1f}s")

    # ------- pairwise bootstrap: is Manifold > L1/TopK? -------
    pairwise: dict[str, dict] = {}
    rng = np.random.default_rng(42)
    for a, b in [("Manifold", "TopK"), ("Manifold", "L1"), ("TopK", "L1")]:
        va = np.asarray(per_model_r2s.get(a, []), dtype=np.float64)
        vb = np.asarray(per_model_r2s.get(b, []), dtype=np.float64)
        if len(va) == 0 or len(vb) == 0:
            continue
        diffs = np.empty(2000)
        for i in range(2000):
            sa = va[rng.integers(0, len(va), size=len(va))]
            sb = vb[rng.integers(0, len(vb), size=len(vb))]
            diffs[i] = sa.mean() - sb.mean()
        pt = float(va.mean() - vb.mean())
        lo = float(np.quantile(diffs, 0.025))
        hi = float(np.quantile(diffs, 0.975))
        p_gt = float((diffs <= 0).mean())  # one-sided: prob diff ≤ 0
        pairwise[f"{a}_minus_{b}"] = {"diff": pt, "ci95": [lo, hi], "p_diff_le_0": p_gt}

    # ------- write outputs -------
    with open(out_dir / "hypotheses.jsonl", "w") as f:
        for h in all_hypotheses:
            f.write(json.dumps(h) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "per_model": per_model_summary,
            "pairwise_bootstrap": pairwise,
            "config": {
                "N_ATOMS_PER_MODEL": N_ATOMS_PER_MODEL,
                "N_TOP_EXAMPLES": N_TOP,
                "N_CAUSAL_ATOMS": N_CAUSAL,
                "min_active_filter": 5,
                "bootstrap_n": 2000,
            },
        }, f, indent=2)

    # ------- markdown report -------
    write_report(out_dir / "report.md", per_model_summary, per_model_atoms, pairwise,
                 N_ATOMS_PER_MODEL, N_TOP, N_CAUSAL)
    print(f"\n[done] wrote {out_dir / 'report.md'}")


def write_report(path: Path, per_model_summary, per_model_atoms, pairwise,
                 N_ATOMS, N_TOP, N_CAUSAL):
    lines = []
    lines.append("# Autointerp Report — SAE atoms on cogito-L40 (xkcd colors × templates)\n")
    lines.append(
        "Three F=512 SAEs (TopK, L1, Manifold) trained on cogito-L40 activations of 949 "
        "xkcd colors × 28 templates. This report applies a structured-hypothesis autointerp "
        "pipeline to all three.\n"
    )
    lines.append("## Design note: local rule-based hypothesizer (no LLM API)\n")
    lines.append(
        "Per project rules (no cluster, no external API), the hypothesizer is a "
        "deterministic Python function that bins each atom's top-20 val activations by "
        "HSV octants, runs name-token TF-IDF against the full xkcd vocabulary, and "
        "extracts a template-id pattern. The output is a structured hypothesis "
        "(`hue_range`, `saturation_range`, `lightness_range`, `name_pattern_regex`, "
        "`template_pattern`) + a 1-sentence NL gloss. This makes the inductive bias "
        "*auditable* and *reproducible*, which is the bottleneck Anthropic-style "
        "autointerp papers actually measure — the verbal fluency of GPT-4 captions is "
        "downstream of whether the hypothesis space can encode the atom at all. The "
        "simulation-accuracy R² in `score.py` is meaningful regardless of the "
        "hypothesizer's identity.\n"
    )

    # head-to-head
    lines.append("## Head-to-head comparison\n")
    lines.append(f"| model | atoms scored | median sim-R² | mean sim-R² | 95% CI (mean) | frac ≥ 0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ["TopK", "L1", "Manifold"]:
        s = per_model_summary[name]
        lines.append(
            f"| {name} | {s['n_atoms_evaluated']} | {s['median_r2']:+.3f} | "
            f"{s['mean_r2']:+.3f} | ({s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}) | "
            f"{s['frac_above_0.5']:.2f} |"
        )
    lines.append("")

    # pairwise
    lines.append("## Pairwise bootstrap (Δ-mean simulation-R²)\n")
    lines.append("| contrast | Δ mean R² | 95% CI | P(Δ ≤ 0) |")
    lines.append("|---|---:|---:|---:|")
    for k, v in pairwise.items():
        a, b = k.replace("_minus_", " − ").split(" − ")
        lines.append(
            f"| {a} − {b} | {v['diff']:+.3f} | "
            f"({v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}) | {v['p_diff_le_0']:.3f} |"
        )
    lines.append("")
    lines.append(
        "_Interpretation:_ a 95% CI strictly above 0 in a `Manifold − X` row means "
        "Manifold-SAE produces atoms that are genuinely more predictable from structured "
        "hypotheses than the comparison model — i.e. more interpretable independent of "
        "raw reconstruction R².\n"
    )

    # per-model details
    for name in ["TopK", "L1", "Manifold"]:
        atoms = per_model_atoms[name]
        atoms_sorted = sorted(atoms, key=lambda a: -a["simulation_r2"])

        lines.append(f"## {name} SAE — top-10 most-interpretable atoms\n")
        lines.append("| atom | sim R² | n_active | ΔR² (causal) | Δcos (causal) | hypothesis |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for a in atoms_sorted[:10]:
            ex = a["explanation"].replace("|", "/")
            lines.append(
                f"| {a['atom_id']} | {a['simulation_r2']:+.3f} | {a['n_active']} | "
                f"{a['causal_delta_r2']:+.4f} | {a['causal_delta_cosine']:.4f} | {ex} |"
            )
        lines.append("")

        lines.append(f"### {name} SAE — bottom-5 LEAST-interpretable atoms (debug)\n")
        for a in atoms_sorted[-5:]:
            tops = ", ".join(e["color"] for e in a["top_examples"][:8])
            lines.append(f"- **atom {a['atom_id']}** sim-R²={a['simulation_r2']:+.3f} "
                         f"compact={a['hsv_compactness']:.3f} n_active={a['n_active']}")
            lines.append(f"  - top colors: {tops}")
            lines.append(f"  - hypothesis: {a['explanation']}")
            lines.append(f"  - hue={a['hue_range']} sat={a['saturation_range']} "
                         f"val={a['lightness_range']} regex=`{a['name_pattern_regex']}` "
                         f"templates={a['template_pattern'][:6]}")
        lines.append("")

    lines.append("## Methodology\n")
    lines.append(
        f"- For each model, ranked atoms by val firing-rate and took the top {N_ATOMS}.\n"
        f"- Per atom: collected top {N_TOP} val examples, built structured hypothesis, "
        f"fit linear regression `features(h) → activation` over val, reported R².\n"
        f"- Top {N_CAUSAL} per model also received the causal ablation: zero the atom in "
        "the decoder and measure (i) Δ val R² and (ii) mean (1 − cos) of reconstruction shift.\n"
        "- Bootstrap (n=2000) over atoms gives 95% CIs on model-mean simulation R² and on "
        "pairwise differences.\n"
        "- Atoms with < 5 val firings excluded from R² aggregation (R² is noise there).\n"
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
