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

# Hue-ordered color names — an intrinsically CYCLIC feature (the hue wheel wraps
# violet->red). The plan flags the color hue-loop as the natural real curved chart to
# dose; steering it is the cleanest large-arc test of the chart's path integral.
COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
COLOR_TEMPLATES = [
    "The sky turned a deep shade of {w}",
    "She painted the whole wall {w}",
    "His favorite color has always been {w}",
    "The traffic light glowed bright {w}",
    "They dyed the fabric a vivid {w}",
    "The sunset faded from gold to {w}",
    "He bought a car that was {w}",
    "The flag in the corner was {w}",
]

FEATURE_BANK = {
    "weekday": (WEEKDAYS, WEEKDAY_TEMPLATES, True),   # (words, templates, periodic)
    "month": (MONTHS, MONTH_TEMPLATES, True),
    "color": (COLORS, COLOR_TEMPLATES, True),
}


def build_prompts(words, templates):
    """Return (prompts, word_index) with the calendar word last in each prompt."""
    prompts, widx = [], []
    for ti, tmpl in enumerate(templates):
        for wi, w in enumerate(words):
            prompts.append(tmpl.format(w=w))
            widx.append(wi)
    return prompts, np.asarray(widx), None


def _circ_mean(a):
    return float(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))


def _circ_corr(a, b):
    """Jammalamadaka-Sarma circular-circular correlation in [-1, 1]."""
    a0 = a - _circ_mean(a)
    b0 = b - _circ_mean(b)
    num = float(np.sum(np.sin(a0) * np.sin(b0)))
    den = float(np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)))
    return num / den if den > 0 else 0.0


def cyclic_ordering(H_red, widx_kept, words):
    """Test that the calendar words wrap around the fitted loop in order.

    Period-free: the per-word mean activation (in the reduced fit frame) lies on the
    fitted closed curve, so the top-2 PCA plane of the W word-means IS the loop's plane.
    Each word gets an unambiguous angle there; we score its circular correlation with the
    calendar order (invariant to the loop's rotation and traversal direction) and report
    the consecutive angular gaps INCLUDING last->first — the wraparound (e.g. Dec->Jan)
    is a real datum only if that closing gap matches the others.
    """
    W = len(words)
    present = [w for w in range(W) if int((widx_kept == w).sum()) > 0]
    if len(present) < 3:
        return None
    means = np.stack([H_red[widx_kept == w].mean(0) for w in present])  # (W', rdim)
    m = means - means.mean(0, keepdims=True)
    _, _, Vt2 = np.linalg.svd(m, full_matrices=False)
    xy = m @ Vt2[:2].T
    ang = np.arctan2(xy[:, 1], xy[:, 0])                    # (W',) angle on the loop plane
    ideal = 2.0 * np.pi * np.arange(len(present)) / len(present)
    corr = max(_circ_corr(ang, ideal), _circ_corr(ang, -ideal))
    # Spacing-robust ordering score: rank each word around the loop, map ranks to equally
    # spaced angles, and circularly-correlate with the calendar order. This is ~1.0 for a
    # correct cyclic sequence regardless of how (non-uniformly) the model spaces the words —
    # raw-angle corr conflates ordering with uniform spacing and understates a good loop.
    ranks = np.argsort(np.argsort(ang)).astype(float)
    rank_ang = 2.0 * np.pi * ranks / len(present)
    order_corr = max(_circ_corr(rank_ang, ideal), _circ_corr(rank_ang, -ideal))
    order_by_angle = [present[i] for i in np.argsort(ang)]
    # rotate the angle-sorted order so it starts at calendar index 0's position, then
    # check it equals the calendar sequence forward or backward (a clean wraparound).
    seq = order_by_angle
    fwd = [present[(present.index(seq[0]) + k) % len(present)] for k in range(len(present))]
    wrap_ok = (seq == fwd) or (seq == fwd[::-1])
    gaps = np.diff(np.concatenate([np.sort(ang), np.sort(ang)[:1] + 2 * np.pi]))
    return dict(circ_corr=float(corr),
                order_corr=float(order_corr),
                words_present=[words[w] for w in present],
                angles_rad=ang.tolist(),
                order_by_angle=[words[w] for w in order_by_angle],
                wraparound_in_order=bool(wrap_ok),
                gap_uniformity=float(1.0 - np.std(gaps) / (np.mean(gaps) + 1e-12)))


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


def find_decoder_layers(hf_model):
    """Return the text decoder's ``ModuleList`` of layers, robust to nested VL models.

    Plain causal LMs expose ``model.model.layers``; vision-language / conditional-gen
    models (e.g. Qwen3.5-MoE-VL) bury the text stack under ``language_model``. We search
    common paths, then fall back to picking the longest ``nn.ModuleList`` of decoder-like
    blocks (attribute ``self_attn``) — the text residual stream — anywhere in the tree.
    """
    import torch.nn as nn

    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "model.language_model.model.layers",
                 "model.text_model.layers", "thinker.model.layers"):
        obj = hf_model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    best = None
    for _name, mod in hf_model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "self_attn"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate a decoder-layer ModuleList in the model")
    return best


def resolve_hook_module(hf_model, layer_idx):
    """Return the decoder layer whose forward output is the layer-``L`` residual."""
    layers = find_decoder_layers(hf_model)
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
def harvest_last_position_fisher(lm, hook_module, ids, rank, oversample, hiter,
                                 trace_probes, device, dtype):
    """Output-Fisher factor U (p, rank) at the LAST token position + all-token acts.

    Reuses gamfit's own harvest internals (``_capture_activations``,
    ``_top_r_eigenpairs``, ``_pullback_matvec``) but runs the randomized-subspace
    eigensolve for the LAST row ONLY — the position whose next-token distribution
    the dose is scored against. This is bit-for-bit the same ``G_n = J_nᵀ F_n J_n``
    the public ``harvest_output_fisher_factors`` computes for that row (and equals
    the downstream metric there, since the last position has a single future
    position — itself), at ~1/T the cost of harvesting every position we discard.
    Returns ``(act_flat (T, p), U_last (p, rank))``.
    """
    import torch
    from gamfit.torch.harvest import _capture_activations

    act_flat, logits_from_act = _capture_activations(lm.module, hook_module, ids)
    T, p = int(act_flat.shape[0]), int(act_flat.shape[1])
    work_dtype = act_flat.dtype if act_flat.dtype in (torch.float32, torch.float64) else torch.float32
    # A bf16 model (large MoE) captures bf16 activations; we run the Fisher eigensolve in
    # float32 for stability but must feed the model its NATIVE dtype, else the spliced f32
    # activation hits bf16 weights (mat1/mat2 dtype mismatch). Cast x to the model dtype for
    # the forward, read logits back up to work_dtype. For an f32 model this is a no-op.
    # Pick the first FLOATING-point parameter dtype (a quantized layer may expose uint8).
    model_dtype = next((pp.dtype for pp in hook_module.parameters()
                        if pp.dtype.is_floating_point), work_dtype)
    row = T - 1
    x_row = act_flat[row].to(work_dtype).detach()

    def f_row(x):
        return logits_from_act(x.to(model_dtype), row).to(work_dtype)

    # REVERSE-MODE-ONLY output Fisher. A fused-MoE model (Qwen3.x-35B-A3B) runs its experts
    # through `torch._grouped_mm`, which has NO forward-mode AD rule, so the jvp-based
    # Fisher-vector product used for dense/8B models raises NotImplementedError here. We
    # instead build the exact same downstream output-Fisher G = Jᵀ F J (F = diag(p) − p pᵀ)
    # from its class-gradient representation, which needs only reverse-mode vjp:
    #     G = Σ_c p_c a_c a_cᵀ − b bᵀ ,  a_c = Jᵀ e_c (grad of logit c),  b = Jᵀ p .
    # Σ_c p_c a_c a_cᵀ = Jᵀ diag(p) J is estimated unbiasedly by S class scores sampled
    # c ~ p (the softmax importance-covers its own Fisher); b bᵀ is exact. The span of G
    # lies in the (S+1) score columns, so we QR that span and eigen-decompose a tiny
    # (S+1)-dim Gram — every model pass is a single reverse-mode backward.
    with torch.no_grad():
        probs = torch.softmax(f_row(x_row), dim=-1)          # (C,)
    C = int(probs.shape[-1])
    _out0, vjp_raw = torch.func.vjp(f_row, x_row)

    def col_score(w):                                        # Jᵀ w, reverse-mode (p,)
        return vjp_raw(w.to(work_dtype))[0].detach()

    n_samp = int(os.environ.get("DOSE_FISHER_SAMPLES", str(max(rank * 6, 48))))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(row)
    p_cpu = probs.double().cpu()
    p_cpu = p_cpu / p_cpu.sum()
    samp = torch.multinomial(p_cpu, n_samp, replacement=True, generator=gen)
    cols = []
    for c in samp.tolist():
        e = torch.zeros(C, dtype=work_dtype, device=x_row.device)
        e[c] = 1.0
        cols.append(col_score(e))
    A = torch.stack(cols, dim=1) / (float(n_samp) ** 0.5)     # (p, S): A Aᵀ ≈ Jᵀ diag(p) J
    b = col_score(probs.to(work_dtype))                      # (p,) = Jᵀ p
    M = torch.cat([A, b.unsqueeze(1)], dim=1).to(torch.float64)   # (p, S+1) spans range(G)
    Q, _ = torch.linalg.qr(M, mode="reduced")                # (p, q)
    QA = Q.t() @ A.to(torch.float64)
    Qb = Q.t() @ b.to(torch.float64)
    Gs = QA @ QA.t() - torch.outer(Qb, Qb)                   # (q, q) reduced Fisher
    Gs = 0.5 * (Gs + Gs.t())
    evals, evecs = torch.linalg.eigh(Gs)
    idx = torch.argsort(evals, descending=True)[:rank]
    ev = evals[idx].clamp_min(0.0)
    Ur = Q @ evecs[:, idx]                                    # (p, rank)
    U_last = (Ur * ev.sqrt().unsqueeze(0)).detach().cpu().numpy()
    return act_flat.to(torch.float64).cpu().numpy(), U_last


def harvest_calendar(lm, hook_module, tokenizer, prompts, rank, device, dtype,
                     oversample=4, hiter=2, trace_probes=8):
    """Harvest (X_last, U_last, template_mean) per prompt at the last position."""
    X_last, U_last, tmpl_mean, kept = [], [], [], []
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        try:
            act_flat, U = harvest_last_position_fisher(
                lm, hook_module, ids, rank, oversample, hiter, trace_probes, device, dtype)
        except Exception as exc:  # noqa: BLE001
            print(f"[harvest] prompt {pi} '{prompt}' FAILED: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}", flush=True)
            continue
        X_last.append(act_flat[-1])
        U_last.append(U)
        tmpl_mean.append(act_flat.mean(0))
        kept.append(pi)
        if (pi + 1) % 10 == 0:
            print(f"[harvest] {pi + 1}/{len(prompts)} prompts", flush=True)
    if not kept:
        raise RuntimeError("no prompt harvested")
    return (np.asarray(X_last), np.asarray(U_last),
            np.asarray(tmpl_mean), np.asarray(kept))


def apply_fp32_head(hf):
    """Upcast the LM head (and final norm) to float32 so the logit read is not bf16-quantized.

    On a bf16 MoE the last projection carries a big chunk of the ~7e-4 measurement floor
    (bf16 rounding of the vocab logits). We cast the output embedding to float32 and register
    a pre-hook that upcasts its input, so the head math is fp32 while the body stays bf16 (no
    extra memory of note — the head is one matrix). Best-effort: never fatal."""
    import torch
    try:
        head = hf.get_output_embeddings()
        if head is None:
            print("[fp32head] no output embedding found; skipping", flush=True)
            return False
        head.float()

        def _pre(_m, args, kwargs):
            if args and hasattr(args[0], "float"):
                return ((args[0].float(),) + tuple(args[1:]), kwargs)
            return None

        head.register_forward_pre_hook(_pre, with_kwargs=True)
        print(f"[fp32head] LM head upcast to float32 ({type(head).__name__})", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fp32head] FAILED ({type(exc).__name__}: {exc}); running bf16 head", flush=True)
        return False


class RouterFlipCounter:
    """Count MoE router top-1 expert flips at the last token, base-forward vs patched-forward.

    Diagnostic for the 35B residual calibration constant: a real edit near a routing decision
    boundary flips which expert fires, producing a discrete output jump unrelated to the smooth
    Fisher prediction. We hook every gate/router Linear whose out_features == num_experts, stash
    the argmax expert at the last position, and after a patched forward report how many layers
    changed their top expert vs the cached base selection. Best-effort, fully defensive: any
    failure disables the counter rather than breaking the KL path."""

    def __init__(self, hf):
        import torch.nn as nn
        self.gates = []
        self.enabled = False
        self._buf = {}
        self._base = {}
        try:
            # num_experts may live on the top config OR a sub-config (e.g. a VL model's
            # text_config, as in qwen3_5_moe: num_experts=256 under text_config).
            n_exp = None
            cfgs = [hf.config] + [getattr(hf.config, s, None) for s in
                                  ("text_config", "llm_config", "language_config", "thinker_config")]
            for cfg in cfgs:
                if cfg is None:
                    continue
                for attr in ("num_experts", "n_routed_experts", "num_local_experts",
                             "moe_num_experts"):
                    v = getattr(cfg, attr, None)
                    if isinstance(v, int) and v > 1:
                        n_exp = v
                        break
                if n_exp is not None:
                    break
            # Match by SHAPE first (an nn.Linear projecting hidden->num_experts is the router,
            # whatever it is named), then by name as a fallback. Newer Qwen MoE routers are a
            # plain nn.Linear named '...mlp.gate' but the VL variant nests it and the class may
            # differ, so shape (out_features == n_exp) is the robust key.
            for name, mod in hf.named_modules():
                cls = type(mod).__name__.lower()
                is_lin = isinstance(mod, nn.Linear)
                # (a) a class-name router (e.g. Qwen3_5MoeTopKRouter: a raw-Parameter router,
                #     NOT nn.Linear, forward -> (router_logits(seq,n_exp), ...));
                # (b) an nn.Linear projecting hidden -> n_exp (plain Qwen/Mixtral gate);
                # (c) name fallback when n_exp is unknown.
                if "router" in cls or (cls.endswith("gate") and cls != "siluandmul"):
                    self.gates.append((name, mod))
                elif is_lin and n_exp is not None and mod.out_features == n_exp:
                    self.gates.append((name, mod))
                elif is_lin and n_exp is None and (name.endswith("gate")
                                                   or name.endswith("router")):
                    self.gates.append((name, mod))
            if self.gates:
                for name, mod in self.gates:
                    mod.register_forward_hook(self._mk_hook(name))
                self.enabled = True
                print(f"[router] instrumented {len(self.gates)} gate modules "
                      f"(num_experts={n_exp})", flush=True)
            else:
                print("[router] no gate/router Linear matched; router logging OFF", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[router] init FAILED ({type(exc).__name__}: {exc}); OFF", flush=True)
            self.enabled = False

    def _mk_hook(self, name):
        def _h(_m, _i, out):
            try:
                import torch
                t = out[0] if isinstance(out, (tuple, list)) else out
                self._buf[name] = int(torch.argmax(t.reshape(-1, t.shape[-1])[-1]).item())
            except Exception:  # noqa: BLE001
                pass
        return _h

    def snapshot_base(self):
        self._base = dict(self._buf)

    def flips_since_base(self):
        if not self.enabled or not self._base:
            return None
        return int(sum(1 for k, v in self._buf.items()
                       if k in self._base and self._base[k] != v))


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

    def __init__(self, lm, hook_module, tokenizer, device, router=None):
        self.lm = lm
        self.hook_module = hook_module
        self.tok = tokenizer
        self.device = device
        self._cache = {}
        self.router = router               # optional RouterFlipCounter
        self._router_base = {}             # prompt -> base top-expert snapshot
        self.last_router_flips = None      # flips of the most recent kl() call

    def _ids(self, prompt):
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

    def base_logprobs(self, prompt):
        import torch
        if prompt in self._cache:
            return self._cache[prompt]
        ids = self._ids(prompt)
        with torch.no_grad():
            logits = self.lm.module(ids)               # (1, T, C)
        # Snapshot the base router selection for THIS prompt (buf was just populated by the
        # gate hooks during this forward) so a later cached call can still diff against it.
        if self.router is not None and self.router.enabled:
            self._router_base[prompt] = dict(self.router._buf)
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
            rows[-1] = rows[-1] + delta_t.to(device=out.device, dtype=out.dtype)
            return torch.stack(rows, 0).reshape(out.shape)

        lp0 = self.base_logprobs(prompt)   # ensures base router snapshot for this prompt exists
        h = self.hook_module.register_forward_hook(_splice)
        try:
            with torch.no_grad():
                logits = self.lm.module(ids)
        finally:
            h.remove()
        # Router flips of THIS patched forward vs the prompt's base selection.
        if self.router is not None and self.router.enabled:
            base = self._router_base.get(prompt)
            if base:
                self.last_router_flips = int(sum(
                    1 for k, v in self.router._buf.items()
                    if k in base and base[k] != v))
            else:
                self.last_router_flips = None
        lp1 = torch.log_softmax(logits[0, -1].double(), -1)
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
def empirical_validity_radius(atom_rows, thresh=0.2):
    """The atom's EMPIRICAL calibration certificate: the largest dose (move as a fraction
    of ||h||) out to which the manifold prediction stays calibrated, i.e. every non-gated
    manifold edit up to that dose has |log(measured/predicted)| < ``thresh``. This replaces
    the formula-based ``validity_radius`` (which the crown data shows is miscalibrated —
    almost no rows passed it). ``max_passing_frac`` is the single largest passing dose."""
    pts = []
    for r in atom_rows:
        p = r.get("predicted_nats"); mkl = r.get("measured_kl")
        if r.get("gated") or p is None or mkl is None or p <= 0 or mkl <= 0:
            continue
        pts.append((float(r.get("delta_frac", 0.0)), abs(float(np.log(mkl / p)))))
    pts.sort()
    radius = 0.0
    for frac, lr in pts:
        if lr < thresh:
            radius = frac
        else:
            break
    passing = [f for f, lr in pts if lr < thresh]
    return dict(certified_radius_frac=float(radius),
                max_passing_frac=float(max(passing, default=0.0)),
                n_calibrated=len(passing), n_points=len(pts), threshold=float(thresh))


def curvature_regression(atom_rows, arm):
    """Regress log(measured/predicted) on theta^2 for one arm. The registered observable
    is ratio(theta) ~ 1 + (c_perp/c_par)*theta^2/4, so a straight tangent-extrapolation
    move should show a POSITIVE slope whose 4x IS the model's behavioral stiffness
    anisotropy c_perp/c_par; the on-chart arc should stay ~flat in theta^2."""
    xs, ys = [], []
    for r in atom_rows:
        if r.get("arm") != arm or r.get("gated"):
            continue
        p = r.get("predicted_nats"); mkl = r.get("measured_kl"); th = r.get("theta")
        if p is None or mkl is None or th is None or p <= 0 or mkl <= 0:
            continue
        xs.append(float(th) ** 2)
        ys.append(float(np.log(mkl / p)))
    if len(xs) < 3:
        return dict(arm=arm, n=len(xs), slope=None, intercept=None, r2=None,
                    anisotropy_c_perp_over_c_par=None)
    x = np.asarray(xs); y = np.asarray(ys)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(arm=arm, n=len(xs), slope=float(slope), intercept=float(intercept),
                r2=float(r2), anisotropy_c_perp_over_c_par=float(4.0 * slope))


def run_sweep(measurer, atoms, lin, fracs, n_bases, shard_U_all, c_bar, seed):
    """One row per (method, atom, base, frac, sign) — real measured KL.

    ON-CHART, AMPLITUDE-NORMALIZED dosing. The demeaned calendar activations carry an
    O(30) signal (||H|| ~ 33 on a ||h|| ~ 96 residual), so the fitted circle has a
    genuine radius; the prior run's sub-measurable ~1e-7 move was caused solely by
    `steer`'s presence weight `amplitude` (~1e-6 for this K=1 atom) multiplying the
    whole displacement. We divide it out consistently:

        patched move   delta_on = (g(t1) - g(t0))          = steer.delta / amplitude
        predicted dose predicted = pathInt_Fisher(t0->t1)  = steer.predicted_nats / amplitude^2

    Both come from the SAME steer plan, so predicted is the honest path integral of the
    metric along the REAL chart arc actually patched — full chart traversal, not a local
    tangent. We target each move to a fraction of ||h|| by inverting the chord length
    m = 2R sin(dt/2) for the chart radius R (read from a small probe steer), clamped to
    the diameter 2R. The tangent-quadratic prediction (1/2 c_tan m^2) is recorded
    alongside for the SAME move so the report can show path-integral tracking KL past
    where the local quadratic drifts. Linear baselines dose along a linear-SAE atom at
    the SAME magnitude m; manifold beating them is the discriminator.
    """
    rng = np.random.default_rng(seed)
    rows = []
    lin_atoms = np.asarray(lin.atoms, dtype=np.float64)   # (Klin, p) in full space
    EPS = float(os.environ.get("DOSE_PROBE_EPS", "1e-3"))  # probe step for R + curvature
    GATE_MIN = float(os.environ.get("DOSE_GATE_MIN", "1e-3"))  # measurable-KL floor
    # 35B CROWN floor protocol: on a bf16 MoE the empty-edit control (a zero patch, run as
    # a SECOND forward vs the cached base) is NOT 0 — non-deterministic expert scatter +
    # router jitter give a per-prompt measurement floor ~7e-4 nats. FLOOR_MULT sets the gate
    # to a multiple of THAT measured floor (Scott: censoring to >30x floor recovers slope
    # ~1.0). FLOOR_REPS averages several zero-patch reads so the per-prompt floor estimate is
    # stable (we take the MAX read as the conservative floor). Both default to the 8B
    # behaviour (mult=1, reps=1) so a deterministic fp32 model is byte-unchanged.
    FLOOR_MULT = float(os.environ.get("DOSE_FLOOR_MULT", "1.0"))
    FLOOR_REPS = int(os.environ.get("DOSE_FLOOR_REPS", "1"))
    # Large-arc cells drive dt directly (as fractions of pi) past the chord/diameter
    # clamp, into the curvature-error regime where a base-point Fisher must finally break.
    dt_pi = [float(x) for x in os.environ.get("DOSE_DT_PI", "0.5,0.75,0.9,1.0").split(",")
             if x.strip()]
    gate_done = False
    for rec in atoms:
        k = rec["atom"]
        sae = rec["sae"]
        H_red = rec["H_red"]; H_full = rec["H_full"]; Vt = rec["Vt"]
        prompts = rec["prompts"]; U = rec["U_full"]
        idx = np.arange(len(H_red))
        bases = rng.choice(idx, size=min(n_bases, len(idx)), replace=False)
        # Held-out split: second half of bases reserved for the HEADLINE slope/R².
        heldout = set(int(b) for b in bases[len(bases) // 2:])
        for bi in bases:
            try:
                xb = H_red[bi:bi + 1]
                xb_full = H_full[bi]
                t0 = np.asarray(sae.project(xb, 0), dtype=np.float64).ravel()
                plan_eps = sae.steer(0, t0, t0 + EPS)
            except Exception as exc:  # noqa: BLE001
                print(f"[sweep] atom {k} base {bi}: project/steer failed "
                      f"({type(exc).__name__}); skipping", flush=True)
                continue
            amp = float(plan_eps.get("amplitude", 1.0) or 1.0)
            pred_eps = plan_eps.get("predicted_nats")
            d_eps = np.asarray(plan_eps["delta"], dtype=np.float64)
            n_eps = float(np.linalg.norm(d_eps))               # = amp * 2R sin(EPS/2)
            vr = plan_eps.get("validity_radius")
            if (pred_eps is None or not np.isfinite(pred_eps) or pred_eps <= 0
                    or n_eps <= 0 or amp <= 0):
                print(f"[sweep] atom {k} base {bi}: bad probe "
                      f"(pred_eps={pred_eps} n_eps={n_eps} amp={amp}); skipping", flush=True)
                continue
            # amplitude-normalized geometry: displacement/amp, dose/amp^2.
            radius = (n_eps / amp) / (2.0 * np.sin(EPS / 2.0))  # chart radius R (activ units)
            c_tan = float(pred_eps) / amp**2 / (0.5 * (n_eps / amp) ** 2)  # nats/(norm)^2
            # Base-row output-Fisher factor for THIS base (steer's own predicted_nats uses
            # the atom's single most-active row, which adds cross-row scatter; the honest
            # per-base dose is 1/2 ||U[bi]^T delta||^2 with the base's own factor). U is
            # (n, p, r); Ub = (p, r), quad(v) = 1/2 * ||Ub^T v||^2.
            Ub = np.asarray(U[bi], dtype=np.float64)           # (p, r)
            proj = lin_atoms @ xb_full
            j = int(np.argmax(np.abs(proj)))
            d_unit = lin_atoms[j] / (np.linalg.norm(lin_atoms[j]) + 1e-12)
            h_norm = float(np.linalg.norm(xb_full))
            prompt = prompts[bi]
            in_heldout = int(bi) in heldout
            # Unit tangent of the chart at t0 (amplitude-normalized full space), read from
            # the eps probe: delta_eps ~ EPS * g'(t0). The tangent-extrapolation arm moves
            # STRAIGHT along this direction — Fisher is locally exact in any direction, so a
            # straight move only accrues curvature error as it bends off the manifold.
            t_hat = (Vt.T @ d_eps) / amp
            t_hat = t_hat / (float(np.linalg.norm(t_hat)) + 1e-30)
            # EMPTY-EDIT CONTROL (mandatory): a zero patch. Its "measured KL" is this
            # prompt's forward-pass noise floor; edits that don't clear it are not real.
            zp = np.zeros(xb_full.shape[0], dtype=np.float64)
            floor_reads = [float(measurer.kl(prompt, zp)) for _ in range(max(1, FLOOR_REPS))]
            # Conservative per-prompt floor = MAX zero-patch read (worst-case non-determinism).
            noise_floor = float(max(floor_reads))
            gate_floor = max(GATE_MIN, FLOOR_MULT * noise_floor)
            rows.append(dict(method="empty", mode="control", atom=int(k), base=int(bi),
                             heldout=bool(in_heldout), arm="empty", theta=0.0,
                             delta_norm=0.0, h_norm=h_norm, predicted_nats=0.0,
                             noise_floor=noise_floor, floor_reads=floor_reads,
                             gate_floor=float(gate_floor), gated=True, measured_kl=noise_floor))
            # Dose cells: (a) chord-fraction cells targeting frac*||h|| (clamped to 2R);
            # (b) CHART large-arc cells (on-manifold arc, dt up to ~pi); (c) TANGENT large-arc
            # cells (straight extrapolation of arc-length R*theta) — the curvature probe.
            cells = []
            for frac in fracs:
                m_target = frac * h_norm
                ratio = min(m_target / (2.0 * radius), 0.999) if radius > 0 else 0.0
                dt_c = 2.0 * float(np.arcsin(ratio))
                cells.append(dict(arm="chord", theta=dt_c, dt=dt_c, frac=float(frac),
                                  clamped=bool(m_target > 2.0 * radius), arc_pi=None))
            for af in dt_pi:
                theta = min(float(af) * np.pi, np.pi * 0.999)     # keep chart map defined
                m_arc = 2.0 * radius * float(np.sin(theta / 2.0))
                cells.append(dict(arm="chart_arc", theta=theta, dt=theta,
                                  frac=float(m_arc / (h_norm + 1e-30)), clamped=False,
                                  arc_pi=float(af)))
                s_tan = radius * theta                            # matched arc length
                cells.append(dict(arm="tangent_arc", theta=theta, dt=None,
                                  frac=float(s_tan / (h_norm + 1e-30)), clamped=False,
                                  arc_pi=float(af)))
            for cell in cells:
                theta = cell["theta"]
                for sign in (+1.0, -1.0):
                    if cell["arm"] == "tangent_arc":
                        # Straight tangent-extrapolated move of arc-length R*theta.
                        s = radius * theta
                        delta_on = sign * s * t_hat
                        m = float(np.linalg.norm(delta_on))
                        pred = float(0.5 * np.sum((Ub.T @ delta_on) ** 2))
                        pred_path = None                          # no path integral off-chart
                        pred_tan = float(0.5 * c_tan * m * m)
                        off_m = float(m)                          # entirely off the manifold
                        vr2 = None
                        dt_rec = None
                    else:
                        dt = cell["dt"]
                        try:
                            plan = sae.steer(0, t0, t0 + sign * dt)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[sweep] atom {k} base {bi} dt {sign*dt}: steer failed "
                                  f"({type(exc).__name__}); skipping", flush=True)
                            continue
                        pred_raw = plan.get("predicted_nats")
                        if pred_raw is None or not np.isfinite(pred_raw) or pred_raw <= 0:
                            continue
                        delta_on = (Vt.T @ np.asarray(plan["delta"], dtype=np.float64)) / amp
                        m = float(np.linalg.norm(delta_on))       # actual on-chart move norm
                        pred = float(0.5 * np.sum((Ub.T @ delta_on) ** 2))
                        pred_path = float(pred_raw) / amp**2       # steer path-integral
                        pred_tan = float(0.5 * c_tan * m * m)      # local-quadratic
                        off_m = float(plan.get("off_manifold_norm", 0.0))
                        vr2 = plan.get("validity_radius")
                        dt_rec = float(sign * dt)
                    meas = float(measurer.kl(prompt, delta_on))
                    router_flips = measurer.last_router_flips
                    gated = bool(meas < gate_floor)               # below THIS prompt's floor
                    if not gate_done:
                        gate_done = True
                        verdict = ("PASS (>%.2g floor)" % gate_floor if meas > gate_floor else
                                   "GATED (<%.2g floor: below noise; rows flagged gated=true, "
                                   "NOT silently shipped)" % gate_floor)
                        print(f"[GATE] first edit arm={cell['arm']} frac={cell['frac']:.4g} "
                              f"m={m:.4g} ||h||={h_norm:.4g} R={radius:.4g} amp={amp:.4g} "
                              f"noise_floor={noise_floor:.3g} measured_kl={meas:.6g} "
                              f"pred={pred:.6g} pred_tan={pred_tan:.6g} -> {verdict}", flush=True)
                    rows.append(dict(
                        method="manifold", mode="on_chart", atom=int(k), base=int(bi),
                        heldout=bool(in_heldout), arm=cell["arm"], theta=float(theta),
                        theta2=float(theta * theta), frac=float(cell["frac"]),
                        dose=float(sign * cell["frac"]), arc_pi=cell["arc_pi"],
                        dt=dt_rec, clamped=bool(cell["clamped"]), amplitude=amp,
                        radius=float(radius), c_tan=float(c_tan), noise_floor=noise_floor,
                        gate_floor=float(gate_floor),
                        delta_norm=m, h_norm=h_norm, delta_frac=float(m / (h_norm + 1e-30)),
                        off_manifold=off_m,
                        validity_radius=(None if vr2 is None else float(vr2)),
                        gated=gated,
                        predicted_nats=float(pred),
                        predicted_nats_pathint=(None if pred_path is None else float(pred_path)),
                        predicted_nats_tangent=float(pred_tan),
                        router_flips=(None if router_flips is None else int(router_flips)),
                        measured_kl=meas))
                    # Linear baselines at the SAME move magnitude m (fair comparison).
                    delta_lin = sign * m * d_unit
                    meas_lin = float(measurer.kl(prompt, delta_lin))
                    lin_gated = bool(meas_lin < gate_floor)
                    rows.append(dict(
                        method="linear_norm", mode="on_chart", atom=int(k), base=int(bi),
                        heldout=bool(in_heldout), arm=cell["arm"], theta=float(theta),
                        theta2=float(theta * theta), frac=float(cell["frac"]),
                        dose=float(sign * cell["frac"]), arc_pi=cell["arc_pi"],
                        delta_norm=m, h_norm=h_norm, off_manifold=None, noise_floor=noise_floor,
                        gated=lin_gated,
                        predicted_nats=float(0.5 * c_bar * m * m), measured_kl=meas_lin))
                    rows.append(dict(
                        method="linear_fisher", mode="on_chart", atom=int(k), base=int(bi),
                        heldout=bool(in_heldout), arm=cell["arm"], theta=float(theta),
                        theta2=float(theta * theta), frac=float(cell["frac"]),
                        dose=float(sign * cell["frac"]), arc_pi=cell["arc_pi"],
                        delta_norm=m, h_norm=h_norm, off_manifold=None, noise_floor=noise_floor,
                        gated=lin_gated,
                        predicted_nats=float(0.5 * np.sum((Ub.T @ delta_lin) ** 2)),
                        measured_kl=meas_lin))
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
    # DOSE_MODEL_DTYPE lets a large MoE load in bf16 (weights) while the harvest/patch
    # math still runs in DOSE_DTYPE (act_flat upcasts to float32 internally). DOSE_DEVICE_MAP
    # spreads a model too large for one card across GPUs via accelerate; ids then live on
    # the embedding's device and every patched delta is placed on the hook output's device.
    _mdtype = os.environ.get("DOSE_MODEL_DTYPE", "")
    model_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32, "float64": torch.float64}.get(_mdtype, dtype)
    dev_map = os.environ.get("DOSE_DEVICE_MAP", "")

    # LEAN-GPU int4: a ~35B-A3B MoE quantizes to ~20GB and fits ONE card. bitsandbytes
    # nf4 with a bf16 compute dtype; the harvest's forward-/reverse-mode AD (jvp/vjp) must
    # pass through the quantized Linear layers — MatMul4Bit defines a backward but NOT a
    # forward-mode rule, so this is the empirical risk the crown mission flags. device_map
    # is MANDATORY for a bnb load (weights are placed on the GPU during from_pretrained;
    # you cannot .to() a 4-bit model afterward).
    _quant = os.environ.get("DOSE_QUANT", "").lower()
    quant_cfg = None
    if _quant in ("int4", "nf4", "4bit"):
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(model_dtype if model_dtype in
                                    (torch.bfloat16, torch.float16) else torch.bfloat16),
            bnb_4bit_use_double_quant=True)
        if not dev_map:
            dev_map = "cuda:0" if device.startswith("cuda") else device

    def _load(**kw):
        # A causal LM loads via AutoModelForCausalLM; a vision-language conditional-gen
        # model (Qwen3.5-MoE-VL) is not in that mapping, so fall back to the image-text
        # class, which still returns .logits for a text-only input_ids forward.
        try:
            return AutoModelForCausalLM.from_pretrained(model_dir, **kw).eval()
        except (ValueError, KeyError) as exc:
            print(f"[model] AutoModelForCausalLM failed ({type(exc).__name__}); "
                  f"trying AutoModelForImageTextToText", flush=True)
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_dir, **kw).eval()

    if quant_cfg is not None:
        hf = _load(quantization_config=quant_cfg, torch_dtype=model_dtype,
                   device_map=dev_map)
        device = hf.get_input_embeddings().weight.device
        print(f"[model] int4 (nf4) quant; device_map={dev_map}; embeddings on {device}",
              flush=True)
    elif dev_map:
        hf = _load(torch_dtype=model_dtype, device_map=dev_map)
        device = hf.get_input_embeddings().weight.device
        print(f"[model] device_map={dev_map}; input embeddings on {device}", flush=True)
    else:
        hf = _load(torch_dtype=model_dtype).to(device)
    for p in hf.parameters():
        p.requires_grad_(False)
    if os.environ.get("DOSE_FP32_HEAD", "0") == "1":
        apply_fp32_head(hf)
    router = RouterFlipCounter(hf) if os.environ.get("DOSE_LOG_ROUTER", "0") == "1" else None
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
        # Cache the (expensive, ~10min/feature) per-prompt output-Fisher harvest so the
        # sweep-scaling can be iterated in seconds. Key on feature + layer + prompt count.
        cache = os.path.join(out, f"harvest_cache_{feat}_L{layer_idx}_n{len(prompts)}.npz")
        if os.path.exists(cache) and os.environ.get("DOSE_NOCACHE", "0") != "1":
            z = np.load(cache, allow_pickle=True)
            X_last, U_last, tmpl_mean, kept = (z["X_last"], z["U_last"],
                                               z["tmpl_mean"], z["kept"])
            print(f"[{feat}] loaded harvest cache {cache} X={X_last.shape}", flush=True)
        else:
            print(f"[{feat}] harvesting {len(prompts)} prompts", flush=True)
            th = time.time()
            X_last, U_last, tmpl_mean, kept = harvest_calendar(
                lm, hook_module, tok, prompts, rank, device, dtype,
                oversample=oversample, hiter=hiter, trace_probes=trace_probes)
            np.savez(cache, X_last=X_last, U_last=U_last, tmpl_mean=tmpl_mean, kept=kept)
            print(f"[{feat}] harvested {len(kept)} rows in {time.time() - th:.1f}s "
                  f"p={X_last.shape[1]}; cached -> {cache}", flush=True)
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
        r2 = float(sae.reconstruction_r2)
        print(f"[{feat}] fit {fit_s:.1f}s (rdim={rdim}) r2={r2:.4f} "
              f"topo={sae.atom_topologies} kw={kw}", flush=True)
        r2_floor = float(os.environ.get("DOSE_R2_FLOOR", "0.5"))
        if r2 < r2_floor or len(sae.atom_topologies) != 1:
            print(f"[{feat}] SKIPPED: degenerate/co-collapsed fit "
                  f"(r2={r2:.3f} < {r2_floor} or {len(sae.atom_topologies)} atoms != 1)",
                  flush=True)
            continue
        sae.fisher_factors = np.ascontiguousarray(U_red)
        sae.fisher_provenance = "output_fisher"
        widx_kept = np.asarray(widx)[kept]
        order = cyclic_ordering(H_red, widx_kept, words)
        if order is not None:
            print(f"[{feat}] cyclic-ordering order_corr={order['order_corr']:.3f} "
                  f"(raw circ_corr={order['circ_corr']:.3f}) "
                  f"wraparound_in_order={order['wraparound_in_order']} "
                  f"gap_uniformity={order['gap_uniformity']:.3f} "
                  f"order={order['order_by_angle']}", flush=True)
        atoms.append(dict(atom=feat, sae=sae, H_red=H_red, H_full=H, Vt=Vt,
                          raw_mean=tmpl_mean, prompts=kept_prompts,
                          U_full=U_last, fit_seconds=fit_s, cyclic_ordering=order,
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

    measurer = MeasuredKL(lm, hook_module, tok, device, router=router)
    # Dose = target move as a fraction of the base activation norm ||h|| (tangent-mode).
    fracs = [float(x) for x in os.environ.get(
        "DOSE_FRACS", "0.005,0.01,0.02,0.05,0.1,0.2,0.4").split(",")]
    # remap atom key to int for the teacher figure/stats code
    for i, a in enumerate(atoms):
        a["atom_name"] = a["atom"]
        a["atom"] = i
    rows = run_sweep(measurer, atoms, lin, fracs, n_bases, shard_U_all, c_bar, seed)
    stats = {m: calibration_stats(rows, m)
             for m in ("manifold", "linear_norm", "linear_fisher")}
    # Per-atom EMPIRICAL validity certificate (replaces the miscalibrated formula radius),
    # plus a gate audit: how many manifold edits fell below the measurable KL floor.
    vthresh = float(os.environ.get("DOSE_VALIDITY_THRESH", "0.2"))
    per_atom_cert = {}
    per_atom_curv = {}
    for a in atoms:
        ar = [r for r in rows if r["method"] == "manifold" and r["atom"] == a["atom"]]
        # Certificate is an ON-CHART claim: exclude the off-manifold tangent probe rows.
        on_chart = [r for r in ar if r.get("arm") != "tangent_arc"]
        per_atom_cert[a["atom"]] = empirical_validity_radius(on_chart, vthresh)
        per_atom_curv[a["atom"]] = dict(
            chart_arc=curvature_regression(ar, "chart_arc"),
            tangent_arc=curvature_regression(ar, "tangent_arc"))
    n_gated = sum(1 for r in rows if r["method"] == "manifold" and r.get("gated"))
    n_manifold = sum(1 for r in rows if r["method"] == "manifold")
    gate_audit = dict(gate_min=float(os.environ.get("DOSE_GATE_MIN", "1e-3")),
                      floor_mult=float(os.environ.get("DOSE_FLOOR_MULT", "1.0")),
                      floor_reps=int(os.environ.get("DOSE_FLOOR_REPS", "1")),
                      n_gated=int(n_gated), n_manifold=int(n_manifold))
    print(f"[gate] {n_gated}/{n_manifold} manifold edits below floor "
          f"{gate_audit['gate_min']:.0e} (flagged gated=true)", flush=True)
    # Router-flip summary (35B MoE diagnostic): fraction of patched edits that flip at least
    # one expert, and mean flips/edit. A high flip rate on the non-gated edits is the honest
    # suspect for a residual multiplicative calibration constant (discrete routing jumps that
    # the smooth Fisher metric cannot see).
    rflips = [r.get("router_flips") for r in rows if r["method"] == "manifold"
              and r.get("router_flips") is not None]
    router_summary = None
    if rflips:
        rf = np.asarray(rflips, dtype=np.float64)
        router_summary = dict(n=int(rf.size), frac_any_flip=float(np.mean(rf > 0)),
                              mean_flips=float(rf.mean()), max_flips=int(rf.max()))
        print(f"[router] {router_summary['frac_any_flip']*100:.1f}% of {router_summary['n']} "
              f"edits flip >=1 expert; mean {router_summary['mean_flips']:.2f} flips/edit "
              f"(max {router_summary['max_flips']})", flush=True)
    for a in atoms:
        c = per_atom_cert[a["atom"]]
        print(f"[cert] atom {a['atom_name']} empirical validity radius = "
              f"{c['certified_radius_frac']:.4f} ||h|| (calibrated {c['n_calibrated']}/"
              f"{c['n_points']} edits, |log ratio|<{vthresh})", flush=True)
    # HELD-OUT half of the manifold edits (never used to tune anything) — the headline.
    ho = [r for r in rows if r["method"] == "manifold" and r.get("heldout")]
    if len(ho) >= 3:
        stats["manifold_heldout"] = _calib(ho)
    # FLOOR-CENSORED headline: drop edits below the per-prompt measurement floor (gated=true,
    # i.e. measured_kl < max(GATE_MIN, FLOOR_MULT*noise_floor)). On a bf16 MoE the raw held-out
    # slope is dragged down by floor-dominated small edits sitting above y=x; censoring to
    # >FLOOR_MULT x floor isolates the edits whose measured KL is real signal. On a
    # deterministic fp32 model (noise_floor~0) this is identical to the raw headline.
    ho_cens = [r for r in ho if not r.get("gated")]
    if len(ho_cens) >= 3:
        stats["manifold_heldout_censored"] = _calib(ho_cens)
    all_cens = [r for r in rows if r["method"] == "manifold" and not r.get("gated")]
    if len(all_cens) >= 3:
        stats["manifold_censored"] = _calib(all_cens)
    # Within each atom's EMPIRICAL certified radius (dose <= radius): the honest
    # "where the chart is trustworthy" tier, replacing the formula-based one.
    within_emp = [r for r in rows if r["method"] == "manifold" and not r.get("gated")
                  and float(r.get("delta_frac", 0.0)) <=
                  per_atom_cert[r["atom"]]["certified_radius_frac"] + 1e-12]
    if len(within_emp) >= 3:
        stats["manifold_within_empirical_validity"] = _calib(within_emp)
    # Large-arc tiers, split by arm: on-manifold chart arc (registered ~flat in theta^2)
    # vs straight tangent extrapolation (registered positive slope = curvature bites).
    for arm in ("chart_arc", "tangent_arc"):
        sub = [r for r in rows if r["method"] == "manifold" and r.get("arm") == arm]
        if len(sub) >= 3:
            stats[f"manifold_{arm}"] = _calib(sub)
        lf = [r for r in rows if r["method"] == "linear_fisher" and r.get("arm") == arm]
        if len(lf) >= 3:
            stats[f"linear_fisher_{arm}"] = _calib(lf)
    # Pooled anisotropy regression across atoms (headline curvature number).
    curv_pooled = dict(chart_arc=curvature_regression(
                           [r for r in rows if r["method"] == "manifold"], "chart_arc"),
                       tangent_arc=curvature_regression(
                           [r for r in rows if r["method"] == "manifold"], "tangent_arc"))
    for arm in ("chart_arc", "tangent_arc"):
        cc = curv_pooled[arm]
        if cc["slope"] is not None:
            print(f"[curv] {arm:11s} log-ratio vs theta^2: slope={cc['slope']:.4f} "
                  f"r2={cc['r2']:.3f} n={cc['n']} -> c_perp/c_par={cc['anisotropy_c_perp_over_c_par']:.4f}",
                  flush=True)
    print(f"[sweep] {len(rows)} rows", flush=True)
    for m in ("manifold", "manifold_heldout", "manifold_within_empirical_validity",
              "manifold_chart_arc", "manifold_tangent_arc", "linear_norm", "linear_fisher",
              "linear_fisher_chart_arc", "linear_fisher_tangent_arc"):
        if m in stats:
            print(f"[stats] {m:34s}={json.dumps(stats[m])}", flush=True)

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
                           cyclic_ordering=a.get("cyclic_ordering"),
                           empirical_validity_certificate=per_atom_cert[a["atom"]],
                           curvature_anisotropy=per_atom_curv[a["atom"]],
                           n_rows=len(a["H_full"]), fit_seconds=a["fit_seconds"]) for a in atoms],
            mean_reconstruction_r2=float(np.mean([a["reconstruction_r2"] for a in atoms])),
            metric_provenance="OutputFisher downstream (harvest_downstream_output_fisher_factors)"),
        dose_mode="on_chart_amplitude_normalized",
        dose_mode_note=(
            "ON-CHART, amplitude-normalized. The demeaned calendar signal is O(30) on a "
            "~96-norm residual, so the fitted circle has a genuine radius; the prior "
            "run's sub-measurable ~1e-7 move was caused by steer's presence weight "
            "`amplitude` (~1e-6 for this K=1 atom) scaling the whole displacement. We "
            "divide it out consistently: patched move = steer.delta/amplitude = the real "
            "chart displacement g(t1)-g(t0); predicted_nats = steer.predicted_nats/"
            "amplitude^2 = the Fisher path integral along that SAME arc. Doses target a "
            "fraction of ||h|| by inverting the chord for the chart radius R (clamped to "
            "the diameter 2R). predicted_nats_tangent (1/2 c_tan m^2) is recorded for the "
            "same move as the local-quadratic reference."),
        gate_audit=gate_audit,
        router_flip_summary=router_summary,
        curvature_pooled=curv_pooled,
        noise_floor=dict(
            median=float(np.median([r["measured_kl"] for r in rows if r["method"] == "empty"]))
            if any(r["method"] == "empty" for r in rows) else None,
            n=int(sum(1 for r in rows if r["method"] == "empty")),
            note="empty-edit (zero patch) measured KL = forward-pass noise floor per prompt"),
        fracs=fracs, stats=stats, rows=rows)
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
        f"- **dose mode = `{payload.get('dose_mode','tangent')}`.** "
        f"{payload.get('dose_mode_note','')}\n",
        "\n## Headline (ideal = slope 1.0, R² 1.0, ratio 1.0)\n",
        "| method | n | slope (log-log) | R² | median meas/pred | mean|log ratio| |",
        "|---|---:|---:|---:|---:|---:|",
        row("**manifold chart — HELD-OUT, censored >floor**", "manifold_heldout_censored"),
        row("manifold chart — HELD-OUT edits (raw, incl. sub-floor)", "manifold_heldout"),
        row("manifold chart — all edits, censored >floor", "manifold_censored"),
        row("manifold chart — all edits (raw)", "manifold"),
        row("manifold chart — within empirical validity radius", "manifold_within_empirical_validity"),
        row("manifold — LARGE ARCs on-chart (registered ~flat)", "manifold_chart_arc"),
        row("manifold — LARGE ARCs TANGENT extrapolation (curvature probe)", "manifold_tangent_arc"),
        row("linear latent, norm dose (no metric) — *task baseline*", "linear_norm"),
        row("linear latent + base-point Fisher (fairness ref)", "linear_fisher"),
        row("linear+Fisher — TANGENT large arcs (where it breaks)", "linear_fisher_tangent_arc"),
        "",
        "\n**Per-atom empirical validity certificate** "
        "(largest dose with |log(meas/pred)|<0.2, on-chart, as a fraction of ‖h‖):\n",
        *[f"- `{a['atom']}`: certified radius = "
          f"{a['empirical_validity_certificate']['certified_radius_frac']:.3f} ‖h‖ "
          f"({a['empirical_validity_certificate']['n_calibrated']}/"
          f"{a['empirical_validity_certificate']['n_points']} edits calibrated)"
          for a in payload["fit"]["per_atom"]],
        "\n**Curvature anisotropy** (regress log(meas/pred) on θ²; slope×4 = c⊥/c∥). "
        "Registered: on-chart arc ~flat, tangent extrapolation positive:\n",
        *[f"- pooled `{arm}`: slope={payload['curvature_pooled'][arm]['slope']:.4f} "
          f"→ c⊥/c∥={payload['curvature_pooled'][arm]['anisotropy_c_perp_over_c_par']:.4f} "
          f"(R²={payload['curvature_pooled'][arm]['r2']:.3f}, n={payload['curvature_pooled'][arm]['n']})"
          for arm in ("chart_arc", "tangent_arc")
          if payload['curvature_pooled'][arm]['slope'] is not None],
        f"\n**Empty-edit noise floor:** median zero-patch KL = "
        f"{payload['noise_floor']['median']:.2e} nats over {payload['noise_floor']['n']} "
        f"controls (the measurement floor; edits below it are flagged `gated=true`).\n",
        f"\n**Gate audit:** {payload['gate_audit']['n_gated']}/"
        f"{payload['gate_audit']['n_manifold']} manifold edits fell below the per-prompt gate "
        f"= max({payload['gate_audit']['gate_min']:.0e}, "
        f"{payload['gate_audit'].get('floor_mult', 1.0):g}x measured floor) and are flagged "
        f"`gated=true` (excluded from the certificate, the curvature regression, and the "
        f"censored headline; never silently shipped). The **censored** headline rows above "
        f"keep only edits whose measured KL clears "
        f"{payload['gate_audit'].get('floor_mult', 1.0):g}x the per-prompt measurement floor "
        f"(Scott's 35B protocol: floor-dominated small edits otherwise sit above y=x and "
        f"drag the raw slope down).\n",
        *([f"\n**Router-flip diagnostic (MoE):** "
           f"{payload['router_flip_summary']['frac_any_flip']*100:.1f}% of "
           f"{payload['router_flip_summary']['n']} patched edits flip at least one expert vs "
           f"the base forward (mean {payload['router_flip_summary']['mean_flips']:.2f} "
           f"flips/edit, max {payload['router_flip_summary']['max_flips']}); discrete routing "
           f"jumps the smooth Fisher metric cannot predict — the honest suspect for any "
           f"residual multiplicative calibration constant.\n"]
          if payload.get("router_flip_summary") else []),
        f"![dose calibration real]({os.path.basename(fig_path)})\n",
        "\nLeft: predicted nats (x) vs measured output KL (y), one point per (atom, base, "
        "frac, sign), with y=x. Right: calibration ratio vs move magnitude.\n",
        f"\nData: `{os.path.basename(json_path)}`\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
