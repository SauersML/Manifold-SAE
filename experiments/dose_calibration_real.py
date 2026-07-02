"""Real-model dose calibration — predicted path-integrated nats vs MEASURED output KL.

This is the real-model counterpart to ``dose_calibration.py`` (the synthetic-teacher
version). Everything the teacher validated is now run on a real language model: we
harvest layer-``L`` residual-stream activations at real calendar-token sites, install
the **downstream output-Fisher** metric with the identical gamfit call the teacher's
``real_model_notes`` documents, fit a K=1 curved ``circle`` chart per calendar feature,
``steer`` it to obtain ``predicted_nats``, and then — the only step that needs a GPU —
patch the steer's ``delta`` back into the layer-``L`` residual, run the rest of the
forward pass, and read the shifted logits to get the **measured** output KL.

Design choices that make the figure unambiguous
-----------------------------------------------
* **Feature token is the LAST token of every prompt.** Then the forward-looking
  downstream Fisher (``harvest_downstream_output_fisher_factors``) has a single future
  position (the token itself), so ``G_n`` reduces bit-for-bit to the same-position
  output Fisher and the *measured* quantity is the clean next-token-distribution KL at
  that one position. No ambiguity about which positions the dose is being scored over.
* **Per-template (per-prompt) demeaning before geometry** — the W7 recipe: the raw
  residual is dominated by sentence context; subtract each prompt's mean over its own
  tokens so the calendar feature (a small component) drives the chart. A per-prompt
  constant shift is a translation, so it leaves every displacement ``delta`` — and
  therefore ``predicted_nats`` and the patch — invariant; it only cleans the fit.
* **The measured KL patches the RAW residual** (what the model actually computes),
  adding the frame-invariant ``delta``.

Methods on the plot (identical semantics to the teacher run):
  manifold      : ``steer`` along the fitted chart -> ``predicted_nats`` (path-integrated
                  downstream-output-Fisher dose) vs measured next-token KL of the move.
  linear_norm   : task baseline — a linear-SAE latent scaled by MATCHED ‖delta‖, whose
                  only honest dose is isotropic ``1/2 c_bar ‖delta‖^2`` (no metric).
  linear_fisher : fairness ref — same linear move but handed the exact base-point Fisher
                  ``1/2 delta^T G0 delta`` (still no path integral).

Run on node2 (needs a GPU + the model weights). Config via env (all optional):
  DOSE_MODEL     HF model dir (default /models/llama-3.1-8b-instruct)
  DOSE_LAYER     decoder layer index to hook (default 16)
  DOSE_RANK      output-Fisher factor rank (default 8)
  DOSE_FEATURES  comma list from {weekday,month} (default weekday,month)
  DOSE_NITER     manifold REML iters (default 40)
  DOSE_NBASES    base activations per atom for the sweep (default 10)
  DOSE_OUT       output dir (default experiments/dose_real_out)
  DOSE_DEVICE    cuda device string (default cuda:0; caller sets CUDA_VISIBLE_DEVICES)
  DOSE_DTYPE     float32|float64 for the forward/harvest (default float32)
"""

from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "16")

import numpy as np

# Reuse the teacher module's calibration + figure code verbatim (same statistics,
# same plot) so the real and synthetic figures are directly comparable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from dose_calibration import (  # noqa: E402
    calibration_stats, _calib, make_figure, local_fisher,
)


# --------------------------------------------------------------------------- #
# Prompt bank — calendar tokens, feature word LAST                            #
# --------------------------------------------------------------------------- #
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Templates end on the calendar word (so its last sub-token is the final position,
# i.e. the next-token prediction site). {w} is filled with the calendar word.
WEEKDAY_TEMPLATES = [
    "Today is {w}",
    "The meeting is scheduled for {w}",
    "I will see you on {w}",
    "Her flight leaves this coming {w}",
    "The store is closed every {w}",
    "We always play football on {w}",
    "The report is due next {w}",
    "He was born on a {w}",
    "The concert takes place on {w}",
    "Let us reconvene on {w}",
]
MONTH_TEMPLATES = [
    "The event is in {w}",
    "She was born in {w}",
    "Our vacation begins in {w}",
    "The fiscal year starts in {w}",
    "The festival happens every {w}",
    "We moved to the city in {w}",
    "The deadline is the end of {w}",
    "His term of office ended in {w}",
    "The harvest is gathered in {w}",
    "The new law took effect in {w}",
]

FEATURE_BANK = {
    "weekday": (WEEKDAYS, WEEKDAY_TEMPLATES, True),   # (words, templates, periodic)
    "month": (MONTHS, MONTH_TEMPLATES, True),
}


def build_prompts(words, templates):
    """Return (prompts, word_index) with the calendar word last in each prompt."""
    prompts, widx = [], []
    for ti, tmpl in enumerate(templates):
        for wi, w in enumerate(words):
            prompts.append(tmpl.format(w=w))
            widx.append(wi)
    return prompts, np.asarray(widx), None


# --------------------------------------------------------------------------- #
# Model plumbing                                                              #
# --------------------------------------------------------------------------- #
class LogitsLM:
    """Thin callable wrapper: ``module(input_ids) -> logits tensor`` (n_pos, C).

    gamfit's harvest treats ``model(inputs)`` as returning a logits tensor whose
    leading axes flatten to tokens, and hooks a submodule whose *output* is the
    residual-stream activation. We wrap a HF causal-LM so a single positional
    ``input_ids`` (1, T) call returns ``logits`` (1, T, C); the harvest flattens the
    batch axis away.
    """

    def __init__(self, hf_model):
        import torch.nn as nn

        class _W(nn.Module):
            def __init__(self, lm):
                super().__init__()
                self.lm = lm

            def forward(self, input_ids):
                return self.lm(input_ids=input_ids).logits

        self.module = _W(hf_model)

    def __call__(self, input_ids):
        return self.module(input_ids)


def resolve_hook_module(hf_model, layer_idx):
    """Return the decoder layer whose forward output is the layer-``L`` residual."""
    layers = hf_model.model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer {layer_idx} out of range 0..{len(layers) - 1}")
    return layers[layer_idx]


def assert_tensor_output(module_wrapper, hook_module, input_ids):
    """Verify the hook module outputs a bare tensor (harvest requires it)."""
    import torch

    captured = {}

    def _grab(_m, _i, out):
        captured["out"] = out

    h = hook_module.register_forward_hook(_grab)
    try:
        with torch.no_grad():
            module_wrapper(input_ids)
    finally:
        h.remove()
    out = captured["out"]
    if isinstance(out, tuple):
        raise RuntimeError(
            "hook module returns a tuple, not a tensor; gamfit harvest needs a "
            f"tensor output. tuple len={len(out)}, elem0 type={type(out[0])}. "
            "Pick a submodule that returns the residual tensor, or unwrap.")
    return tuple(out.shape)


# --------------------------------------------------------------------------- #
# Harvest: per-prompt downstream output-Fisher at the LAST position           #
# --------------------------------------------------------------------------- #
def harvest_calendar(lm, hook_module, tokenizer, prompts, rank, device, dtype,
                     oversample=4, hiter=2, trace_probes=8):
    """Harvest (X_last, U_last, template_mean) per prompt.

    For each prompt we run ``harvest_downstream_output_fisher_factors`` (the exact
    real-model call) and keep the LAST-position row: its activation ``X_last``, its
    downstream output-Fisher factor ``U_last`` (p, r), and the prompt's mean
    activation over its own tokens (for per-template demeaning). The last position
    has one future position (itself), so the downstream metric equals the
    same-position output Fisher there.
    """
    import torch
    from gamfit.torch.harvest import harvest_downstream_output_fisher_factors

    X_last, U_last, tmpl_mean, kept = [], [], [], []
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        try:
            shard = harvest_downstream_output_fisher_factors(
                lm.module, hook_module, ids, rank=rank,
                oversample=oversample, n_iter=hiter, trace_probes=trace_probes)
        except Exception as exc:  # noqa: BLE001
            print(f"[harvest] prompt {pi} '{prompt}' FAILED: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}", flush=True)
            continue
        X = np.asarray(shard.X, dtype=np.float64)         # (T, p)
        U = np.asarray(shard.U, dtype=np.float64)         # (T, p, r)
        X_last.append(X[-1])
        U_last.append(U[-1])
        tmpl_mean.append(X.mean(0))
        kept.append(pi)
        if (pi + 1) % 10 == 0:
            print(f"[harvest] {pi + 1}/{len(prompts)} prompts", flush=True)
    if not kept:
        raise RuntimeError("no prompt harvested")
    return (np.asarray(X_last), np.asarray(U_last),
            np.asarray(tmpl_mean), np.asarray(kept))


# --------------------------------------------------------------------------- #
# Measured KL: patch delta at layer-L last position, forward, read logits      #
# --------------------------------------------------------------------------- #
class MeasuredKL:
    """Patch a delta into the layer-``L`` residual at a prompt's last position and
    return the symmetrized next-token KL vs the unpatched logits.

    Uses the SAME splice mechanism gamfit's harvest uses (a forward hook that
    replaces the last row of the hook module's output), so the patched forward pass
    is a faithful ``h_L += delta`` intervention run through layers ``L..end``.
    """

    def __init__(self, lm, hook_module, tokenizer, device):
        self.lm = lm
        self.hook_module = hook_module
        self.tok = tokenizer
        self.device = device
        self._cache = {}

    def _ids(self, prompt):
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

    def base_logprobs(self, prompt):
        import torch
        if prompt in self._cache:
            return self._cache[prompt]
        ids = self._ids(prompt)
        with torch.no_grad():
            logits = self.lm.module(ids)               # (1, T, C)
        lp = torch.log_softmax(logits[0, -1].double(), -1)
        self._cache[prompt] = lp
        return lp

    def kl(self, prompt, delta):
        import torch
        ids = self._ids(prompt)
        delta_t = torch.tensor(np.asarray(delta), dtype=torch.float64, device=self.device)

        def _splice(_m, _i, out):
            flat = out.reshape(-1, out.shape[-1])
            rows = [flat[i] for i in range(flat.shape[0])]
            rows[-1] = rows[-1] + delta_t.to(out.dtype)
            return torch.stack(rows, 0).reshape(out.shape)

        h = self.hook_module.register_forward_hook(_splice)
        try:
            with torch.no_grad():
                logits = self.lm.module(ids)
        finally:
            h.remove()
        lp1 = torch.log_softmax(logits[0, -1].double(), -1)
        lp0 = self.base_logprobs(prompt)
        p0, p1 = lp0.exp(), lp1.exp()
        kl01 = float((p0 * (lp0 - lp1)).sum())
        kl10 = float((p1 * (lp1 - lp0)).sum())
        return 0.5 * (kl01 + kl10)


# --------------------------------------------------------------------------- #
# Fit one K=1 circle atom on the demeaned calendar activations                 #
# --------------------------------------------------------------------------- #
def fit_atom(H, n_iter, seed):
    """Fit a single ``circle`` atom (K=1) to the demeaned calendar activations."""
    import gamfit

    for kw in (dict(n_iter=n_iter, random_state=seed),
               dict(n_iter=n_iter + 20, random_state=seed + 101),
               dict(n_iter=n_iter + 40, random_state=seed + 202)):
        try:
            t0 = time.time()
            sae = gamfit.sae_manifold_fit(H, K=1, d_atom=1,
                                          atom_topology="circle", **kw)
            return sae, time.time() - t0, kw
        except Exception as exc:  # noqa: BLE001
            print(f"[fit] K=1 attempt {kw} failed: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}", flush=True)
    return None


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
def run_sweep(measurer, atoms, lin, doses, n_bases, shard_U_all, c_bar, seed):
    """One row per (method, atom, base, dose, sign) — real measured KL."""
    rng = np.random.default_rng(seed)
    rows = []
    lin_atoms = np.asarray(lin.atoms, dtype=np.float64)   # (Klin, p) in full space
    for rec in atoms:
        k = rec["atom"]
        sae = rec["sae"]
        H_red = rec["H_red"]               # demeaned activations, reduced fit frame
        H_full = rec["H_full"]             # demeaned activations, full p-dim space
        Vt = rec["Vt"]                     # (rdim, p) orthonormal PCA basis
        prompts = rec["prompts"]           # prompt text per row
        U = rec["U_full"]                  # (n, p, r) full-space fisher field
        idx = np.arange(len(H_red))
        bases = rng.choice(idx, size=min(n_bases, len(idx)), replace=False)
        for bi in bases:
            xb = H_red[bi:bi + 1]
            xb_full = H_full[bi]
            t0 = np.asarray(sae.project(xb, 0), dtype=np.float64).ravel()
            G0 = U[bi] @ U[bi].T
            proj = lin_atoms @ xb_full
            j = int(np.argmax(np.abs(proj)))
            d_unit = lin_atoms[j] / (np.linalg.norm(lin_atoms[j]) + 1e-12)
            prompt = prompts[bi]
            for dose in doses:
                for sign in (+1.0, -1.0):
                    plan = sae.steer(0, t0, t0 + sign * dose)
                    pred = plan.get("predicted_nats")
                    if pred is None or not np.isfinite(pred) or pred <= 0:
                        continue
                    delta_red = np.asarray(plan["delta"], dtype=np.float64)
                    delta = Vt.T @ delta_red                  # map reduced move to full space
                    dn = float(np.linalg.norm(delta))
                    vr = plan.get("validity_radius")
                    meas = measurer.kl(prompt, delta)
                    rows.append(dict(
                        method="manifold", atom=int(k), base=int(bi), dose=float(sign * dose),
                        delta_norm=dn, off_manifold=float(plan.get("off_manifold_norm", 0.0)),
                        validity_radius=(None if vr is None else float(vr)),
                        within_validity=(None if vr is None else bool(abs(dose) <= float(vr))),
                        predicted_nats=float(pred), measured_kl=float(meas)))
                    delta_lin = sign * dn * d_unit
                    meas_lin = measurer.kl(prompt, delta_lin)
                    rows.append(dict(
                        method="linear_norm", atom=int(k), base=int(bi), dose=float(sign * dose),
                        delta_norm=dn, off_manifold=None, validity_radius=None, within_validity=None,
                        predicted_nats=float(0.5 * c_bar * dn * dn), measured_kl=float(meas_lin)))
                    rows.append(dict(
                        method="linear_fisher", atom=int(k), base=int(bi), dose=float(sign * dose),
                        delta_norm=dn, off_manifold=None, validity_radius=None, within_validity=None,
                        predicted_nats=float(0.5 * delta_lin @ G0 @ delta_lin), measured_kl=float(meas_lin)))
        print(f"[sweep] atom {k}: {len(bases)} bases done", flush=True)
    return rows


def main() -> int:
    import torch
    import gamfit

    model_dir = os.environ.get("DOSE_MODEL", "/models/llama-3.1-8b-instruct")
    layer_idx = int(os.environ.get("DOSE_LAYER", "16"))
    rank = int(os.environ.get("DOSE_RANK", "8"))
    features = os.environ.get("DOSE_FEATURES", "weekday,month").split(",")
    n_iter = int(os.environ.get("DOSE_NITER", "40"))
    n_bases = int(os.environ.get("DOSE_NBASES", "10"))
    oversample = int(os.environ.get("DOSE_OVERSAMPLE", "4"))
    hiter = int(os.environ.get("DOSE_HITER", "2"))
    trace_probes = int(os.environ.get("DOSE_TRACE", "8"))
    max_tpl = int(os.environ.get("DOSE_MAXTPL", "0"))  # 0 = all templates
    seed = int(os.environ.get("DOSE_SEED", "0"))
    out = os.environ.get("DOSE_OUT", os.path.join(_HERE, "dose_real_out"))
    device = os.environ.get("DOSE_DEVICE", "cuda:0")
    dtype = torch.float64 if os.environ.get("DOSE_DTYPE", "float32") == "float64" else torch.float32
    os.makedirs(out, exist_ok=True)

    print(f"[cfg] model={model_dir} layer={layer_idx} rank={rank} features={features} "
          f"n_iter={n_iter} n_bases={n_bases} device={device} dtype={dtype}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    lm = LogitsLM(hf)
    hook_module = resolve_hook_module(hf, layer_idx)
    print(f"[model] loaded in {time.time() - t0:.1f}s; hidden={hf.config.hidden_size} "
          f"layers={hf.config.num_hidden_layers}", flush=True)

    probe_ids = tok("Today is Monday", return_tensors="pt").input_ids.to(device)
    shape = assert_tensor_output(lm.module, hook_module, probe_ids)
    print(f"[hook] layer {layer_idx} output shape {shape} (tensor OK)", flush=True)

    atoms = []
    all_U = []
    for feat in features:
        words, templates, periodic = FEATURE_BANK[feat]
        if max_tpl > 0:
            templates = templates[:max_tpl]
        prompts, widx, _ = build_prompts(words, templates)
        print(f"[{feat}] harvesting {len(prompts)} prompts", flush=True)
        th = time.time()
        X_last, U_last, tmpl_mean, kept = harvest_calendar(
            lm, hook_module, tok, prompts, rank, device, dtype,
            oversample=oversample, hiter=hiter, trace_probes=trace_probes)
        print(f"[{feat}] harvested {len(kept)} rows in {time.time() - th:.1f}s "
              f"p={X_last.shape[1]}", flush=True)
        # Per-template demeaning: subtract each prompt's own token-mean.
        H = X_last - tmpl_mean
        kept_prompts = [prompts[i] for i in kept]

        # Fit + steer in a PCA-reduced subspace. The p=d_model REML circle fit is
        # prohibitively slow and the calendar chart lives in a ~2-D subspace, so we
        # reduce to the top `rdim` PCA directions of the demeaned activations. With an
        # ORTHONORMAL basis Vt (rdim, p) this is exact for the calibration: a reduced
        # move delta_red maps to the full-space move delta_full = Vt^T delta_red with
        # identical norm, and predicted_nats = 1/2 delta_red^T (Vt G Vt^T) delta_red =
        # 1/2 delta_full^T G delta_full, so the reduced-space output-Fisher metric
        # G_red = Vt G Vt^T yields the SAME prediction while the measured KL still
        # patches the true full-space delta_full. (Same reduce-before-geometry recipe
        # as the W7 curved-feature probes.)
        rdim = min(int(os.environ.get("DOSE_RDIM", "48")), len(H) - 1)
        Hc = H - H.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        Vt = np.ascontiguousarray(Vt[:rdim])                     # (rdim, p) orthonormal
        H_red = H @ Vt.T                                          # (n, rdim)
        U_red = np.ascontiguousarray(np.einsum("rp,nps->nrs", Vt, U_last))  # (n, rdim, r)
        got = fit_atom(H_red, n_iter, seed + hash(feat) % 1000)
        if got is None:
            print(f"[{feat}] SKIPPED (fit failed)", flush=True)
            continue
        sae, fit_s, kw = got
        print(f"[{feat}] fit {fit_s:.1f}s (rdim={rdim}) r2={float(sae.reconstruction_r2):.4f} "
              f"topo={sae.atom_topologies} kw={kw}", flush=True)
        sae.fisher_factors = np.ascontiguousarray(U_red)
        sae.fisher_provenance = "output_fisher_downstream"
        atoms.append(dict(atom=feat, sae=sae, H_red=H_red, H_full=H, Vt=Vt,
                          raw_mean=tmpl_mean, prompts=kept_prompts,
                          U_full=U_last, fit_seconds=fit_s,
                          reconstruction_r2=float(sae.reconstruction_r2)))
        all_U.append(U_last)

    if not atoms:
        raise RuntimeError("no calendar atom fit succeeded")

    shard_U_all = np.concatenate(all_U, 0)
    p = shard_U_all.shape[1]
    c_bar = float((shard_U_all ** 2).reshape(len(shard_U_all), -1).sum(1).mean() / p)

    # Linear dictionary on the concatenated demeaned activations (a matched linear SAE).
    H_all = np.concatenate([a["H_full"] for a in atoms], 0)
    lin = gamfit.linear_dictionary_fit(H_all, max(len(atoms), 2))

    measurer = MeasuredKL(lm, hook_module, tok, device)
    doses = [0.01, 0.02, 0.04, 0.07, 0.11, 0.17, 0.25, 0.34]
    # remap atom key to int for the teacher figure/stats code
    for i, a in enumerate(atoms):
        a["atom_name"] = a["atom"]
        a["atom"] = i
    rows = run_sweep(measurer, atoms, lin, doses, n_bases, shard_U_all, c_bar, seed)
    stats = {m: calibration_stats(rows, m)
             for m in ("manifold", "linear_norm", "linear_fisher")}
    within = [r for r in rows if r["method"] == "manifold" and r.get("within_validity")]
    if len(within) >= 3:
        stats["manifold_within_validity"] = _calib(within)
    print(f"[sweep] {len(rows)} rows", flush=True)
    for m in ("manifold", "manifold_within_validity", "linear_norm", "linear_fisher"):
        if m in stats:
            print(f"[stats] {m:26s}={json.dumps(stats[m])}", flush=True)

    fig_path = os.path.join(out, "dose_calibration_real.png")
    make_figure(rows, stats, fig_path)

    payload = dict(
        config=dict(model=model_dir, layer=layer_idx, rank=rank, features=features,
                    n_iter=n_iter, n_bases=n_bases, seed=seed, dtype=str(dtype)),
        model=f"REAL model {os.path.basename(model_dir)} (layer {layer_idx}); measured "
              f"output KL = patched forward pass, exact next-token distribution",
        fit=dict(
            n_atoms_fit=len(atoms),
            per_atom=[dict(atom=a["atom_name"],
                           reconstruction_r2=a["reconstruction_r2"],
                           atom_topologies=list(a["sae"].atom_topologies),
                           n_rows=len(a["H_full"]), fit_seconds=a["fit_seconds"]) for a in atoms],
            mean_reconstruction_r2=float(np.mean([a["reconstruction_r2"] for a in atoms])),
            metric_provenance="OutputFisher downstream (harvest_downstream_output_fisher_factors)"),
        doses=doses, stats=stats, rows=rows)
    json_path = os.path.join(out, "dose_calibration_real.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    md = _report_md(stats, payload, fig_path, json_path)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(md)
    print("\n" + md, flush=True)
    print(f"[out] {json_path}\n[out] {fig_path}", flush=True)
    return 0


def _report_md(stats, payload, fig_path, json_path) -> str:
    def g(d, k):
        return d.get(k, float("nan"))

    def row(name, key):
        st = stats.get(key, {})
        return (f"| {name} | {g(st, 'n')} | {g(st, 'log_slope'):.3f} | "
                f"{g(st, 'log_r2'):.3f} | {g(st, 'ratio_median'):.3f} | "
                f"{g(st, 'mean_abs_log_ratio'):.3f} |")

    cfg = payload["config"]
    lines = [
        "# Real-model dose calibration — predicting an intervention's effect in nats\n",
        f"**Model:** `{payload['model']}`\n",
        "**Claim tested:** a curved manifold-SAE atom is an explicit parametric chart "
        "`g(t)` carrying a downstream output-Fisher metric, so `steer` reports "
        "`predicted_nats` — how far the model's output token distribution will move — "
        "*before* the edit. We plot that prediction against the **measured** output KL "
        "from actually patching the edit into the forward pass and re-reading the logits.\n",
        f"**Setup:** layer-{cfg['layer']} residual-stream activations at calendar-token "
        f"sites ({', '.join(cfg['features'])}); one K=1 `circle` chart per feature with the "
        "downstream output-Fisher metric attached "
        "(`harvest_downstream_output_fisher_factors`, the exact real-model call). Feature "
        "token is the last position, so the measured KL is the clean next-token-distribution "
        "shift. Per-template demeaning before geometry (W7 recipe).\n",
        f"- mean chart reconstruction R² = {payload['fit']['mean_reconstruction_r2']:.4f} "
        f"over {payload['fit']['n_atoms_fit']} atoms.\n",
        "\n## Headline (ideal = slope 1.0, R² 1.0, ratio 1.0)\n",
        "| method | n | slope (log-log) | R² | median meas/pred | mean|log ratio| |",
        "|---|---:|---:|---:|---:|---:|",
        row("**manifold chart — `predicted_nats`**", "manifold"),
        row("linear latent, norm dose (no metric) — *task baseline*", "linear_norm"),
        row("linear latent + base-point Fisher (fairness ref)", "linear_fisher"),
        "",
        f"![dose calibration real]({os.path.basename(fig_path)})\n",
        "\nLeft: predicted nats (x) vs measured output KL (y), one point per (atom, base, "
        "dose, sign), with y=x. Right: calibration ratio vs move magnitude.\n",
        f"\nData: `{os.path.basename(json_path)}`\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
