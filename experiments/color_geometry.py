"""Color geometry — how does cogito (frontier-scale LLM) represent the
xkcd color space?

Harvests residual-stream activations from the cogito-probed inference
server (node1:8000) via /v1/encode at layer 40 — the same layer all
loaded interpretability probes target. ~7168-dim residuals per prompt.

Local mode (no Cogito access): set MSAE_USE_LOCAL=1 plus MSAE_MODEL=<hf-id>
to fall back to a transformers forward pass at layers MSAE_LAYERS.

A focused color-only experiment with the full xkcd 954-color list as
external ground truth. Asks three concrete questions:

  Q1 — INTRINSIC DIM (with template quotient).
      Per-template dim (across 954 colors at fixed template) is the
      cleanest measurement of the concept-axis dim with template
      variation removed. Compare with global dim (all 954×12).

  Q2 — IS THE LM LINEARLY ENCODING EACH AXIS?
      For each of {R, G, B, hue, sat, value, luminance}, fit a ridge
      regression  axis = w · residual + b  with 5-fold split ACROSS
      COLORS (not prompts). Held-out R² is the generalization-correct
      measure of "is this axis encoded linearly?". PC↔axis Spearman
      (the v1 measurement) is a weaker version that only checks the
      top PCs.

  Q3 — ARE THE AXES INDEPENDENTLY ENCODED?
      Train a ridge regression for axis A, project residuals onto its
      direction, subtract. Re-run probe for axis B on the residual. If
      R²_B is still high, the LM has SEPARATE directions for the two
      axes (vs. one direction with multiple axes piggybacked).

  Q4 — MULTI-OUTPUT JOINT PROBE.
      Single ridge regression  (R,G,B) = W · residual + b. Compare its
      R² to the sum of per-axis univariate R²s. If close, RGB is a
      linear function of residual; if much lower, RGB is non-linearly
      mixed.

Implementation
--------------
Mean-pooled residual harvest from cogito layer 40 via /v1/encode HTTP
batches. ~26.7k prompts (954 colors × 28 templates). Per-dim normalization.

The original script used last-token residuals on a local Qwen model.
On cogito-probed we use mean-pool over the prompt: last-token-only
isn't directly exposed by /v1/encode, and streaming per-token activations
for 26.7k prompts at 7168-d would be ~24 GB — mean-pool drops that to
~750 MB and is at least as informative for "is color X in this prompt"
since the color word's position varies across templates.

5-fold CV is split BY COLOR — all 28 templates of a color are in the
same fold — so we measure cross-COLOR generalization, not just
cross-prompt. Templates serve as repeats per color (improves SNR but
doesn't leak the answer).
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


# -----------------------------------------------------------------------------
# 28 diverse templates: varied domains (fashion, nature, art, food, animals,
# architecture, weather, emotion, materials), varied syntactic roles (subject,
# object, predicate, prepositional), varied lengths (5–18 words), varied
# positions (color word at start, middle, end).
# -----------------------------------------------------------------------------
TEMPLATES = [
    # Fashion / clothing
    "She slipped into a {x} silk dress and floated down the staircase.",
    "His {x} velvet jacket caught every eye in the room.",
    "A long, {x} scarf trailed behind her in the wind.",
    # Nature / landscape
    "The dawn sky deepened from grey to {x} before the storm broke.",
    "Across the meadow stretched a sea of {x} wildflowers.",
    "From the cliff we watched the ocean turn a strange {x}.",
    # Art / paint
    "The painter mixed his pigments until the canvas glowed a perfect {x}.",
    "She dipped her brush in the {x} pool of paint on the palette.",
    "It was the kind of {x} that you only see in renaissance frescoes.",
    # Architecture / objects
    "The cathedral's stained-glass rose window burned a luminous {x} at sunset.",
    "He polished the {x} car until the chrome shone like a mirror.",
    "A single {x} candle lit the small, dusty chapel.",
    # Animals
    "The hummingbird's throat flashed an iridescent {x} as it darted past.",
    "Her tabby cat had eyes the unmistakable {x} of an autumn leaf.",
    "A great {x} stallion thundered across the open plain.",
    # Food
    "The chef plated a glistening, almost-{x} reduction beside the duck.",
    "She bit into the macaron, finding a soft {x} filling within.",
    # Body / skin / hair
    "Her hair fell across her shoulders in waves of soft {x}.",
    "His skin turned a sickly {x} after three days at sea.",
    "She had freckles and {x} eyes that seemed to change with the weather.",
    # Materials / minerals / gems
    "The jeweler held up a flawless {x} stone, catching the lamplight.",
    "Centuries of oxidation had stained the bronze a deep {x}.",
    # Atmospheric / mood
    "An eerie {x} fog rolled in from the harbor at midnight.",
    "Her bedroom walls were a calm, washed-out {x}, like an old photograph.",
    # Manufactured / mundane
    "I bought a {x} fountain pen at the antique market.",
    "The neon sign above the diner flickered {x} against the night.",
    # Emotional / metaphorical (color word still describes a concrete noun)
    "Grief, in her writing, was always a kind of {x}.",
    "He saw the world through {x} glasses and refused to take them off.",
]


XKCD_URL = "https://xkcd.com/color/rgb.txt"


def load_xkcd_colors() -> list[tuple[str, int, int, int]]:
    cache = Path(__file__).parent / "xkcd_colors.txt"
    if cache.exists():
        text = cache.read_text()
    else:
        print(f"[xkcd] fetching {XKCD_URL}", flush=True)
        with urllib.request.urlopen(XKCD_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        try:
            cache.write_text(text)
        except OSError:
            pass
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("License") or s.startswith("Copyright"):
            continue
        m = re.match(r"^(.+?)\s+#?([0-9a-fA-F]{6})$", s)
        if not m:
            continue
        name, hex_ = m.group(1).strip(), m.group(2)
        r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
        out.append((name, r, g, b))
    return out


def rgb_to_hsv_vec(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float64)
    for i in range(rgb.shape[0]):
        out[i] = colorsys.rgb_to_hsv(rgb[i, 0]/255.0, rgb[i, 1]/255.0, rgb[i, 2]/255.0)
    return out


@dataclass
class Config:
    # Default backend: cogito-probed via HTTP /v1/encode at layer 40.
    # To fall back to a local HF model, set MSAE_USE_LOCAL=1 and configure
    # MSAE_MODEL / MSAE_LAYERS.
    use_local: bool = bool(int(os.environ.get("MSAE_USE_LOCAL", "0")))
    # Local-mode HF model (only used when use_local=True).
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-7B")
    # Layers to harvest. Cogito-probed currently only hooks layer 40
    # (the layer all loaded probes target). Local-mode default sweeps a
    # safe set of layers on any Qwen2.5-{1.5,3,7}B.
    layers: tuple[int, ...] = field(default_factory=lambda: tuple(
        int(x) for x in os.environ.get(
            "MSAE_LAYERS",
            "40" if not bool(int(os.environ.get("MSAE_USE_LOCAL", "0"))) else "4,12,20,26",
        ).split(",")
    ))
    # Cogito-probed server endpoint (only used when use_local=False).
    cogito_url: str = os.environ.get(
        "COGITO_API_BASE", os.environ.get("COGITO_URL", "http://localhost:8000")
    )
    # Aggregation mode for /v1/encode. "mean" pools over the prompt;
    # "tokens" returns per-token (we keep the last) — bandwidth-heavy.
    aggregate: str = os.environ.get("COGITO_AGGREGATE", "mean")
    # HTTP batch size — number of prompts per /v1/encode request.
    http_batch_size: int = int(os.environ.get("COGITO_BATCH", "64"))
    # HTTP request timeout (seconds). Cogito's forward pass at TP=8 is fast
    # but the server is shared; allow generous headroom.
    http_timeout: float = float(os.environ.get("COGITO_TIMEOUT", "180"))
    # HTTP retries on transient failures (network blip, queue overflow).
    http_retries: int = int(os.environ.get("COGITO_RETRIES", "4"))
    # Sleep this many seconds between consecutive /v1/encode batches. Keeps
    # us courteous to the shared server.
    sleep_between_batches: float = float(os.environ.get("COGITO_SLEEP", "0.25"))
    # Number of parallel HTTP workers. Default 1 = strictly sequential
    # submission. Bump (e.g. to 2-4) to overlap server compute with network
    # transit, but stay well below the burst pattern that wedged the server
    # before (12 concurrent encode requests in 7s).
    n_workers: int = int(os.environ.get("COGITO_WORKERS", "1"))
    # If set, cap total prompts at this number (truncates the prompt list).
    # Lets us do a smaller pilot run before committing to all 26.7k.
    max_prompts: int | None = (
        int(os.environ["COGITO_MAX_PROMPTS"])
        if os.environ.get("COGITO_MAX_PROMPTS") else None
    )
    # If set, only use the first N templates of TEMPLATES. Reduces total prompts
    # while keeping per-template-quotient analysis valid. Default uses all 28.
    n_templates: int | None = (
        int(os.environ["COGITO_N_TEMPLATES"])
        if os.environ.get("COGITO_N_TEMPLATES") else None
    )
    # Per-prompt truncation (only used in local-mode tokenizer call;
    # cogito-probed enforces max_model_len server-side).
    max_length: int = int(os.environ.get("COGITO_MAX_LENGTH", "64"))
    n_folds: int = 5
    ridge_alpha: float = 10.0          # ridge strength (the residuals are
                                        # whitened to unit-σ per dim so this
                                        # corresponds to mild regularization)
    n_pcs: int = 8                     # top-k PCs to test for PC↔axis
    batch_size: int = 32               # local-mode forward-pass batch
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/COLOR_GEOMETRY",
    )


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("no blocks")


def harvest_batched(model, tok, blocks, layer: int, prompts: list[str],
                     device, batch_size: int) -> torch.Tensor:
    """Local-mode harvest: forward pass + last-token residual via hook."""
    cap = {}
    h = blocks[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    if tok.padding_side != "left":
        tok.padding_side = "left"
    feats = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=64).to(device)
            model(**enc)
            # left-padded → last token at index -1 for every row.
            feats.append(cap["h"][:, -1, :].cpu())
    h.remove()
    return torch.cat(feats, dim=0).float()


def _post_encode(cfg: Config, payload: dict) -> dict:
    """POST to cogito-probed /v1/encode with retries. Returns parsed JSON."""
    body = json.dumps(payload).encode()
    url = cfg.cogito_url.rstrip("/") + "/v1/encode"
    last_err: Exception | None = None
    for attempt in range(cfg.http_retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=cfg.http_timeout) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"cogito encode error: {data['error']}")
                return data
        except (urllib.error.URLError, TimeoutError, RuntimeError, ConnectionError) as exc:
            last_err = exc
            if attempt < cfg.http_retries:
                backoff = 2.0 * (attempt + 1)
                print(f"  [encode] attempt {attempt + 1} failed ({exc}); "
                      f"retrying in {backoff:.1f}s", flush=True)
                time.sleep(backoff)
    raise RuntimeError(f"cogito encode failed after {cfg.http_retries + 1} attempts: {last_err}")


def _hung_request_count(cfg: Config, threshold_s: float = 60.0) -> int:
    """Count /v1/encode requests that have been 'active' longer than threshold_s.

    Used as a watchdog: a growing count means the server is wedging and we
    should back off rather than pile on more load.
    """
    try:
        with urllib.request.urlopen(
            cfg.cogito_url.rstrip("/") + "/v1/status", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return -1  # status check failed; treat as "unknown" not "many"
    active = (data.get("queue") or {}).get("active") or []
    return sum(
        1 for r in active
        if r.get("endpoint") == "encode" and (r.get("elapsed_s") or 0) >= threshold_s
    )


def harvest_via_cogito(
    cfg: Config, layer: int, prompts: list[str],
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Stream layer-`layer` residuals from the cogito-probed inference server.

    Submits prompts in batches of cfg.http_batch_size to /v1/encode and
    aggregates one D-dim vector per prompt according to cfg.aggregate
    ("mean" pools over the prompt; "tokens" takes the last token).

    Incremental caching: when ``cache_path`` is given, the partial harvest
    is persisted as a ``.npz`` after every checkpoint (and on graceful
    interruption). On restart, prompts already covered by the cache are
    skipped — bandwidth and server time aren't wasted re-fetching them.

    Watchdog: every ``hung_check_interval_batches`` batches we poll
    ``/v1/status``; if too many encode requests are stuck (> threshold for
    > 60 s), we raise ``RuntimeError`` so the caller can re-try later
    rather than add to a wedged server's queue.
    """
    if cfg.aggregate not in ("mean", "tokens"):
        raise ValueError(f"aggregate must be 'mean' or 'tokens', got {cfg.aggregate!r}")
    n = len(prompts)

    # Resume from incremental cache if available.
    feats: list[np.ndarray] = []
    n_resume = 0
    if cache_path is not None and cache_path.exists():
        ck = np.load(cache_path, allow_pickle=False)
        cached = ck["X"]
        cached_layer = int(ck["layer"])
        cached_n_target = int(ck["n_target"])
        cached_agg = str(ck["aggregate"])
        if (cached_layer == layer and cached_n_target == n
                and cached_agg == cfg.aggregate):
            feats.extend(cached[i] for i in range(cached.shape[0]))
            n_resume = cached.shape[0]
            print(f"  [encode] resume: loaded {n_resume}/{n} from "
                  f"{cache_path.name}", flush=True)
        else:
            print(f"  [encode] cache shape mismatch; ignoring partial "
                  f"({cached_layer=} vs {layer=}, {cached_n_target=} vs {n=}, "
                  f"{cached_agg=!r} vs {cfg.aggregate!r})", flush=True)

    t_start = time.time()
    n_done = n_resume
    # How often to snapshot to disk and how often to poll the watchdog.
    # Checkpoint frequently: a single dropped connection shouldn't waste more
    # than a few minutes of forward-pass work.
    checkpoint_every_batches = max(1, 1000 // max(1, cfg.http_batch_size))
    hung_check_interval_batches = max(1, 1000 // max(1, cfg.http_batch_size))
    initial_hung = max(_hung_request_count(cfg), 0)
    batch_idx = 0
    key = f"layer_{layer}"

    def _do_batch(batch: list[str]) -> list[np.ndarray]:
        """Single HTTP request → list of D-dim vectors per prompt."""
        payload = {
            "texts": batch,
            "layers": [layer],
            "aggregate": cfg.aggregate,
            "max_length": cfg.max_length,
        }
        data = _post_encode(cfg, payload)
        out = []
        for r in data["results"]:
            arr = np.asarray(r[key], dtype=np.float32)
            if cfg.aggregate == "tokens":
                if arr.ndim != 2:
                    raise RuntimeError(
                        f"unexpected token-mode shape {arr.shape}; expected (T, D)"
                    )
                arr = arr[-1]                       # last token per prompt
            elif arr.ndim != 1:
                raise RuntimeError(
                    f"unexpected mean-mode shape {arr.shape}; expected (D,)"
                )
            out.append(arr)
        return out

    # Build remaining batches (post-resume).
    pending_batches = [prompts[s:s + cfg.http_batch_size]
                       for s in range(n_resume, n, cfg.http_batch_size)]

    if cfg.n_workers <= 1:
        result_iter = iter(_do_batch(b) for b in pending_batches)
    else:
        from concurrent.futures import ThreadPoolExecutor
        # ``map`` yields results in submission order — so the saved cache stays
        # aligned with the prompt order regardless of which worker finished
        # first. ``chunksize=1`` keeps responses streaming as they finish
        # rather than batching them per-worker.
        _pool = ThreadPoolExecutor(max_workers=cfg.n_workers)
        result_iter = _pool.map(_do_batch, pending_batches, chunksize=1)

    try:
        for batch_results in result_iter:
            feats.extend(batch_results)
            n_done += len(batch_results)
            batch_idx += 1
            elapsed = time.time() - t_start
            rate = (n_done - n_resume) / max(elapsed, 1e-6)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"  [encode] {n_done}/{n}  ({rate:.1f}/s, ETA {eta:5.1f}s)",
                  flush=True)

            # Periodic checkpoint
            if cache_path is not None and (batch_idx % checkpoint_every_batches == 0):
                X_so_far = np.stack(feats, axis=0)
                np.savez(cache_path,
                         X=X_so_far, layer=np.int64(layer),
                         n_target=np.int64(n),
                         aggregate=np.array(cfg.aggregate))
                print(f"  [encode] checkpoint {n_done}/{n} -> {cache_path.name}",
                      flush=True)

            # Watchdog: bail out if hung-request count grows materially. Our
            # smoke-test path completes in ~0.3 s, so any encode request
            # active for > 60 s is by definition stuck.
            if batch_idx % hung_check_interval_batches == 0:
                cur_hung = _hung_request_count(cfg)
                if cur_hung > initial_hung + 3:
                    raise RuntimeError(
                        f"server appears to be wedging "
                        f"(hung encode count {initial_hung} -> {cur_hung}); "
                        f"aborting. Resume later from {cache_path}."
                    )

            # Be polite to the shared inference server: small pause between
            # batches so we don't queue-flood. With N parallel workers, the
            # natural network round-trip already rate-limits per worker, so
            # we don't need a long sleep at the gather side — but a small
            # pause still smooths out bursts.
            if n_done < n and cfg.sleep_between_batches > 0:
                time.sleep(cfg.sleep_between_batches)
    except BaseException as exc:
        # Persist whatever we have on ANY failure (Ctrl-C, network drop, server
        # error, watchdog abort) — losing N minutes of forward-pass work to a
        # VPN blip is the kind of thing that happens repeatedly otherwise.
        if cache_path is not None and feats:
            X_so_far = np.stack(feats, axis=0)
            np.savez(cache_path,
                     X=X_so_far, layer=np.int64(layer),
                     n_target=np.int64(n),
                     aggregate=np.array(cfg.aggregate))
            print(f"  [encode] aborted ({type(exc).__name__}); "
                  f"saved partial {len(feats)}/{n} to {cache_path.name}",
                  flush=True)
        raise

    X = np.stack(feats, axis=0)              # (N, D)
    return torch.from_numpy(X).float()


def per_dim_normalize(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = X.mean(0, keepdim=True)
    sigma = X.std(0, keepdim=True).clamp(min=1e-6)
    return (X - mu) / sigma, mu, sigma


# -----------------------------------------------------------------------------
# Intrinsic-dim metrics
# -----------------------------------------------------------------------------
def pca_dims(X: torch.Tensor) -> tuple[int, int, int]:
    Xc = X - X.mean(0, keepdim=True)
    _, s, _ = torch.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum().clamp(min=1e-12)
    def fge(t): return int((cum >= t).float().argmax().item()) + 1
    return fge(0.5), fge(0.9), fge(0.99)


def correlation_dim(X: torch.Tensor) -> float:
    Xn = X.numpy().astype(np.float64)
    N = Xn.shape[0]
    if N < 8: return float("nan")
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    d = np.sqrt(sq[np.triu_indices(N, k=1)])
    d_med = float(np.median(d))
    if not np.isfinite(d_med) or d_med == 0: return float("nan")
    rs = np.geomspace(0.05 * d_med, 2 * d_med, 14)
    cs = np.array([(d < r).mean() for r in rs])
    mask = (cs > 0.05) & (cs < 0.5) & (cs > 0)
    if mask.sum() < 3: mask = cs > 0
    if mask.sum() < 2: return float("nan")
    return float(np.polyfit(np.log(rs[mask]), np.log(cs[mask]), 1)[0])


def measure_dim(X: torch.Tensor) -> dict:
    k50, k90, k99 = pca_dims(X)
    return {"n": X.shape[0], "k50": k50, "k90": k90, "k99": k99,
            "corr_dim": correlation_dim(X)}


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = float(np.sqrt((rx*rx).sum() * (ry*ry).sum()))
    return float((rx*ry).sum() / denom) if denom > 0 else 0.0


# -----------------------------------------------------------------------------
# Ridge probe with color-grouped K-fold CV
# -----------------------------------------------------------------------------
def ridge_probe_cv(
    X: torch.Tensor,                   # (N, D) standardized
    y: np.ndarray,                     # (N,) regression target
    color_idx: np.ndarray,             # (N,) which color each row belongs to
    n_folds: int = 5,
    alpha: float = 10.0,
    seed: int = 0,
) -> dict:
    """Ridge regression with leave-color-group-out CV.

    Returns held-out R² (averaged across folds), held-out RMSE,
    and the fitted full-data weight vector (for downstream orthogonal
    decomposition).
    """
    rng = np.random.default_rng(seed)
    n_colors = int(color_idx.max()) + 1
    perm = rng.permutation(n_colors)
    fold_assignments = perm % n_folds                          # color -> fold
    color_to_fold = np.empty(n_colors, dtype=np.int64)
    color_to_fold[perm] = np.arange(n_colors) % n_folds
    fold_for_row = color_to_fold[color_idx]                    # row -> fold

    X_np = X.numpy().astype(np.float64)
    N, D = X_np.shape
    A = alpha * np.eye(D + 1)
    A[-1, -1] = 0.0                                            # don't penalize bias

    r2s = []
    rmses = []
    for k in range(n_folds):
        train_mask = fold_for_row != k
        test_mask = ~train_mask
        if train_mask.sum() < 10 or test_mask.sum() < 5: continue
        Xtr = X_np[train_mask]; ytr = y[train_mask]
        Xte = X_np[test_mask];  yte = y[test_mask]
        # augment with bias
        Xtr_b = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
        Xte_b = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
        # solve (X'X + αI) β = X'y
        try:
            beta = np.linalg.solve(Xtr_b.T @ Xtr_b + A, Xtr_b.T @ ytr)
        except np.linalg.LinAlgError:
            r2s.append(float("nan")); rmses.append(float("nan")); continue
        pred = Xte_b @ beta
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = math.sqrt(ss_res / max(1, len(yte)))
        r2s.append(r2); rmses.append(rmse)

    # Full-data weight for downstream use (orthogonal projection)
    X_b_full = np.concatenate([X_np, np.ones((N, 1))], axis=1)
    try:
        beta_full = np.linalg.solve(X_b_full.T @ X_b_full + A, X_b_full.T @ y)
        w_full = beta_full[:-1]
    except np.linalg.LinAlgError:
        w_full = None

    return {
        "held_out_r2_mean": float(np.nanmean(r2s)) if r2s else float("nan"),
        "held_out_r2_std": float(np.nanstd(r2s)) if r2s else float("nan"),
        "held_out_rmse_mean": float(np.nanmean(rmses)) if rmses else float("nan"),
        "per_fold_r2": [float(v) for v in r2s],
        "weights_l2": float(np.linalg.norm(w_full)) if w_full is not None else None,
        "weights": w_full,                                     # not serialized
    }


def multi_output_ridge_cv(
    X: torch.Tensor, Y: np.ndarray, color_idx: np.ndarray,
    n_folds: int = 5, alpha: float = 10.0, seed: int = 0,
) -> dict:
    """Joint ridge for multi-output target Y (N, K). Returns per-output and
    macro R²."""
    rng = np.random.default_rng(seed)
    n_colors = int(color_idx.max()) + 1
    perm = rng.permutation(n_colors)
    color_to_fold = np.empty(n_colors, dtype=np.int64)
    color_to_fold[perm] = np.arange(n_colors) % n_folds
    fold_for_row = color_to_fold[color_idx]

    X_np = X.numpy().astype(np.float64)
    N, D = X_np.shape
    K = Y.shape[1]
    A = alpha * np.eye(D + 1)
    A[-1, -1] = 0.0

    per_out_r2 = [[] for _ in range(K)]
    macro_r2 = []
    for fold in range(n_folds):
        train_mask = fold_for_row != fold
        test_mask = ~train_mask
        if train_mask.sum() < 10 or test_mask.sum() < 5: continue
        Xtr = np.concatenate([X_np[train_mask], np.ones((train_mask.sum(), 1))], axis=1)
        Xte = np.concatenate([X_np[test_mask], np.ones((test_mask.sum(), 1))], axis=1)
        Ytr = Y[train_mask]; Yte = Y[test_mask]
        try:
            B = np.linalg.solve(Xtr.T @ Xtr + A, Xtr.T @ Ytr)
        except np.linalg.LinAlgError:
            continue
        pred = Xte @ B                                         # (n_test, K)
        for kx in range(K):
            ss_res = ((Yte[:, kx] - pred[:, kx]) ** 2).sum()
            ss_tot = ((Yte[:, kx] - Yte[:, kx].mean()) ** 2).sum()
            per_out_r2[kx].append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        # macro: 1 - sum(ss_res over outputs) / sum(ss_tot over outputs)
        ssr = ((Yte - pred) ** 2).sum()
        sst = ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum()
        macro_r2.append(1 - ssr / sst if sst > 0 else float("nan"))
    return {
        "macro_r2": float(np.nanmean(macro_r2)) if macro_r2 else float("nan"),
        "per_output_r2": [float(np.nanmean(v)) if v else float("nan") for v in per_out_r2],
    }


def pc_axis_spearmans(X: torch.Tensor, axes: dict, n_pcs: int) -> dict:
    # Center-only PCA via sklearn (consistent with _pca_basis.top_pcs); no
    # standardization because the caller's per_dim_normalize has already
    # whitened X when appropriate.
    from _pca_basis import top_pcs
    pcs = top_pcs(X.numpy().astype(np.float64), d=n_pcs, standardize=False)
    out = {}
    for ax, vals in axes.items():
        rhos = [spearman(pcs[:, k], vals) for k in range(n_pcs)]
        best_k = int(np.argmax(np.abs(rhos)))
        out[ax] = {"per_pc": rhos, "best_pc": best_k,
                    "best_rho": float(rhos[best_k])}
    return out


def project_out(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove the direction w from X. X (N, D); w (D,)."""
    w = w / (np.linalg.norm(w) + 1e-12)
    proj = (X @ w)[:, None] * w[None, :]
    return X - proj


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    cfg = Config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = load_xkcd_colors()
    templates = TEMPLATES if cfg.n_templates is None else TEMPLATES[: cfg.n_templates]
    n_c, n_t = len(colors), len(templates)
    if cfg.use_local:
        require_cuda_if_env()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[setup] local {cfg.model_name} layers={cfg.layers} device={device}",
              flush=True)
    else:
        device = torch.device("cpu")
        print(f"[setup] cogito-probed via {cfg.cogito_url}  layers={cfg.layers}  "
              f"aggregate={cfg.aggregate}  http_batch={cfg.http_batch_size}", flush=True)
    print(f"[colors] {n_c}  templates={n_t}  -> {n_c * n_t} prompts", flush=True)

    # GT axes
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64)
    hsv = rgb_to_hsv_vec(rgb)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    axis_vals_per_color = {
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "hue": hsv[:, 0], "sat": hsv[:, 1], "value": hsv[:, 2],
        "luminance": lum,
    }

    # Prompts + indices
    prompts, c_idx, t_idx = [], [], []
    for ci, (name, _, _, _) in enumerate(colors):
        for ti, tpl in enumerate(templates):
            prompts.append(tpl.format(x=name))
            c_idx.append(ci); t_idx.append(ti)
    c_idx = np.array(c_idx); t_idx = np.array(t_idx)
    if cfg.max_prompts is not None and cfg.max_prompts < len(prompts):
        prompts = prompts[: cfg.max_prompts]
        c_idx = c_idx[: cfg.max_prompts]
        t_idx = t_idx[: cfg.max_prompts]
        print(f"[prompts] capped to first {len(prompts)} via COGITO_MAX_PROMPTS",
              flush=True)
    axis_vals_per_prompt = {k: v[c_idx] for k, v in axis_vals_per_color.items()}

    # In local-mode we need a HF model + tokenizer + block list. In
    # cogito-probed mode the server handles tokenization and forward passes
    # for us, so we skip the whole torch load.
    model = tok = blocks = None
    if cfg.use_local:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = (AutoModelForCausalLM
                 .from_pretrained(cfg.model_name, torch_dtype=torch.float32)
                 .to(device).eval())
        root = model.model if hasattr(model, "model") else model.transformer
        blocks = _find_blocks(root)

    results = {}
    for layer in cfg.layers:
        print(f"\n=== L={layer} ===", flush=True)
        # Final cache: complete (N, D) matrix; incremental cache: partial
        # harvest with metadata. Incremental supports resumption of an
        # interrupted run; final means we don't re-harvest at all.
        final_path = out_dir / f"X_L{layer}.npy"
        partial_path = out_dir / f"X_L{layer}.partial.npz"
        if final_path.exists():
            X = torch.from_numpy(np.load(final_path)).float()
            print(f"  X={tuple(X.shape)} (loaded from {final_path.name})",
                  flush=True)
        else:
            if cfg.use_local:
                X = harvest_batched(model, tok, blocks, layer, prompts,
                                    device, cfg.batch_size)
            else:
                X = harvest_via_cogito(cfg, layer, prompts, cache_path=partial_path)
            np.save(final_path, X.numpy().astype(np.float32))
            # Successful full harvest: clear the incremental cache.
            if partial_path.exists():
                partial_path.unlink()
            print(f"  X={tuple(X.shape)}  -> cached at {final_path.name}",
                  flush=True)
        Xn, _, _ = per_dim_normalize(X)

        # Q1 — intrinsic dim
        global_m = measure_dim(Xn)
        per_t_ms = [measure_dim(Xn[t_idx == ti]) for ti in range(n_t)]
        def aavg(d_list, k):
            v = [d[k] for d in d_list if not (isinstance(d[k], float) and math.isnan(d[k]))]
            return float(np.mean(v)) if v else float("nan")
        per_t_avg = {k: aavg(per_t_ms, k) for k in ["k50", "k90", "k99", "corr_dim"]}
        print(f"  Q1 intrinsic dim:")
        print(f"    global       k50={global_m['k50']:3d} k90={global_m['k90']:3d} "
              f"k99={global_m['k99']:3d} corr={global_m['corr_dim']:.2f}", flush=True)
        print(f"    per-template k50={per_t_avg['k50']:5.1f} k90={per_t_avg['k90']:5.1f} "
              f"k99={per_t_avg['k99']:5.1f} corr={per_t_avg['corr_dim']:.2f}", flush=True)

        # Q2 — per-axis ridge probe with color-grouped 5-fold CV (pooled over templates)
        ridge_per_axis = {}
        for ax, vals in axis_vals_per_prompt.items():
            res = ridge_probe_cv(Xn, vals, c_idx, cfg.n_folds, cfg.ridge_alpha)
            res_serial = {k: v for k, v in res.items() if k != "weights"}
            ridge_per_axis[ax] = res_serial
            ridge_per_axis[ax]["_weights"] = res["weights"]    # keep in-memory only
        print(f"  Q2 held-out ridge R² per axis (5-fold by color):", flush=True)
        for ax, r in ridge_per_axis.items():
            print(f"    {ax:9}: R²={r['held_out_r2_mean']:+.3f} ± {r['held_out_r2_std']:.3f}  "
                  f"per-fold={['%+.2f' % v for v in r['per_fold_r2']]}", flush=True)

        # Q4 — multi-output (R,G,B) joint probe
        Y_rgb = np.stack([axis_vals_per_prompt[k] for k in ["R", "G", "B"]], axis=1)
        joint = multi_output_ridge_cv(Xn, Y_rgb, c_idx, cfg.n_folds, cfg.ridge_alpha)
        print(f"  Q4 joint (R,G,B) ridge: macro R²={joint['macro_r2']:+.3f}  "
              f"per-output={['%+.2f' % v for v in joint['per_output_r2']]}", flush=True)

        # Q3 — orthogonal decomposition: project out best axis (highest R²),
        # re-probe the rest
        best_axis = max(ridge_per_axis.items(), key=lambda kv: kv[1]["held_out_r2_mean"])[0]
        w_best = ridge_per_axis[best_axis]["_weights"]
        if w_best is not None:
            X_orth = project_out(Xn.numpy().astype(np.float64), w_best)
            X_orth_t = torch.from_numpy(X_orth).float()
            print(f"  Q3 projecting out '{best_axis}' direction; re-probing others:", flush=True)
            ridge_after = {}
            for ax, vals in axis_vals_per_prompt.items():
                if ax == best_axis: continue
                res = ridge_probe_cv(X_orth_t, vals, c_idx, cfg.n_folds, cfg.ridge_alpha)
                res_serial = {k: v for k, v in res.items() if k != "weights"}
                ridge_after[ax] = res_serial
                pre = ridge_per_axis[ax]["held_out_r2_mean"]
                post = res_serial["held_out_r2_mean"]
                print(f"    {ax:9}: pre R²={pre:+.3f}  post R²={post:+.3f}  "
                      f"Δ={post - pre:+.3f}", flush=True)
        else:
            ridge_after = {}

        # PC↔axis (the v1-style cheap baseline)
        pc_axes = pc_axis_spearmans(Xn, axis_vals_per_prompt, cfg.n_pcs)

        # Strip in-memory weights before serializing
        for ax in ridge_per_axis:
            ridge_per_axis[ax].pop("_weights", None)

        results[f"L{layer}"] = {
            "global_dim": global_m,
            "per_template_dim_avg": per_t_avg,
            "ridge_per_axis": ridge_per_axis,
            "joint_rgb": joint,
            "best_axis_projected_out": best_axis,
            "ridge_after_projecting_out_best": ridge_after,
            "pc_vs_axes": pc_axes,
        }

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg),
        "n_colors": n_c, "n_templates": n_t,
        "templates": list(TEMPLATES),
        "results": results,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
