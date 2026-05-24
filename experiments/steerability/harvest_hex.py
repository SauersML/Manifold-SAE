"""Harvest cogito layer-40 residuals over ~300 hex color codes.

Builds an RGB-cube-spanning hex set (64-cell 4x4x4 uniform grid at
{0x33,0x66,0x99,0xCC} + 236 perceptually-spaced xkcd hex codes deduped
against the grid) and asks cogito to encode each one under 12 prompt
templates spanning multiple voices and contexts (3rd-person descriptive,
abstract/neutral, art/paint, digital/screen, design/fashion, two
first-person variants, 2nd-person, narrative, scientific measurement,
code/CSS, and nature/object). All 12 templates START with the hex code.
Aggregation is **post-hex mean**: for each prompt we request per-token
activations, then client-side mean over tokens after the hex prefix
(excluding the hex tokens themselves). Hex token counts are pre-computed
via a one-shot encode of each hex alone.

Result: ~300 hex * 12 templates = ~3600 rows of (D=7168,) residuals
saved as `X_L40_hex.npy` + per-row index in `hex_prompt_index.json`.

Cogito endpoint comes from env var `COGITO_API_BASE` (or `COGITO_URL`),
NEVER hardcoded. The default placeholder keeps the script safe to ship.

Usage:
  python harvest_hex.py --dry-run                  # print first 3 requests
  COGITO_API_BASE=http://<host>:8000 \
    python harvest_hex.py                          # live harvest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XKCD = HERE.parent / "xkcd_colors.txt"

DEFAULT_LAYER = int(os.environ.get("COGITO_LAYER", "40"))
DEFAULT_API = os.environ.get(
    "COGITO_API_BASE", os.environ.get("COGITO_URL", "http://localhost:8000")
)

# Six prompt templates. Each receives a {hex} field like "#33CC99".
# All six START with the hex code so client-side `post-hex` aggregation
# (mean over trailing tokens, excluding the hex prefix) yields a clean
# "what comes after this color" representation rather than mixing hex
# tokens into the pool.
TEMPLATES = [
    # 3rd-person descriptive / abstract
    "{hex} is a color.",
    "{hex} -- a hex code representing a color that is",
    # Art / paint context
    "{hex} — a swatch on the artist's palette.",
    # Digital / screen context
    "{hex} renders on the screen as a pixel of",
    # Design / fashion context
    "{hex} appears in the design palette as a featured shade.",
    # 1st-person (affective + associative)
    "{hex} — I look at this color and feel",
    "{hex} — when I see this color, I think of",
    # 2nd-person
    "{hex} — you stand before this color and find it",
    # Narrative
    "{hex} — the painter dipped the brush and the wall became",
    # Scientific / measurement
    "{hex} measured on the spectrophotometer corresponds to",
    # Code / CSS context
    "{hex} — in the CSS stylesheet, this color signals",
    # Nature / object context
    "{hex} — the petals of the flower were",
]


def grid_hex() -> list[str]:
    """4 x 4 x 4 uniform grid of 64 hex codes spanning the RGB cube."""
    vals = (0x33, 0x66, 0x99, 0xCC)
    out = []
    for r in vals:
        for g in vals:
            for b in vals:
                out.append(f"#{r:02X}{g:02X}{b:02X}")
    return out


def xkcd_perceptual_hex(n_keep: int, exclude: set[str]) -> list[str]:
    """136 perceptually-spaced xkcd hex codes (greedy farthest-point in
    RGB). Drops any duplicates of the grid via `exclude`."""
    if not XKCD.exists():
        # Defensive fallback: if xkcd cache is missing, just return a
        # static spread so dry-run still works.
        fallback = [
            "#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF",
            "#0000FF", "#7F00FF", "#FF00FF", "#FFFFFF", "#000000",
        ]
        return [h for h in fallback if h not in exclude][:n_keep]
    candidates: list[tuple[str, np.ndarray]] = []
    for line in XKCD.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("License"):
            continue
        parts = s.split("\t")
        if len(parts) < 2:
            continue
        hx = parts[1].strip().lstrip("#").upper()
        if len(hx) != 6:
            continue
        full = f"#{hx}"
        if full in exclude:
            continue
        rgb = np.array(
            [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)],
            dtype=np.float32,
        )
        candidates.append((full, rgb))
    if not candidates:
        return []
    # Greedy farthest-point selection in RGB.
    seed_idx = 0
    picked_idx = [seed_idx]
    arr = np.stack([rgb for _, rgb in candidates], axis=0)  # (N, 3)
    min_d = np.linalg.norm(arr - arr[seed_idx], axis=1)
    while len(picked_idx) < min(n_keep, len(candidates)):
        nxt = int(np.argmax(min_d))
        picked_idx.append(nxt)
        d_nxt = np.linalg.norm(arr - arr[nxt], axis=1)
        min_d = np.minimum(min_d, d_nxt)
    return [candidates[i][0] for i in picked_idx]


def build_hex_set() -> list[str]:
    grid = grid_hex()
    extra = xkcd_perceptual_hex(n_keep=236, exclude=set(grid))
    full = grid + extra
    # Dedup while preserving order.
    seen, out = set(), []
    for h in full:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def build_prompts(hexes: list[str]) -> list[dict]:
    rows = []
    for h in hexes:
        for ti, tpl in enumerate(TEMPLATES):
            rows.append({"hex": h, "template_idx": ti,
                          "prompt": tpl.format(hex=h)})
    return rows


def post_encode(api_base: str, texts: list[str], *, layer: int,
                aggregate: str = "tokens", timeout: float = 180.0,
                retries: int = 4) -> dict:
    body = json.dumps(
        {"texts": texts, "layers": [layer], "aggregate": aggregate,
         "max_length": 64}
    ).encode("utf-8")
    url = api_base.rstrip("/") + "/v1/encode"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"cogito error: {data['error']}")
                return data
        except (urllib.error.URLError, TimeoutError, RuntimeError,
                ConnectionError) as e:
            last = e
            if attempt < retries:
                backoff = 2.0 * (attempt + 1)
                print(f"[encode] attempt {attempt + 1} failed ({e}); "
                      f"retry in {backoff:.1f}s", file=sys.stderr, flush=True)
                time.sleep(backoff)
    raise RuntimeError(f"cogito encode failed after {retries + 1} tries: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=DEFAULT_API)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-x", default=str(HERE / "X_L40_hex.npy"))
    ap.add_argument("--out-idx", default=str(HERE / "hex_prompt_index.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hexes = build_hex_set()
    prompts = build_prompts(hexes)
    print(f"[hex] {len(hexes)} unique hex codes x {len(TEMPLATES)} templates "
          f"= {len(prompts)} prompts", flush=True)

    if args.dry_run:
        print("[dry-run] step 1: PREFIX-TOKEN-COUNT batch for hexes (sample):")
        prefix_payload = {"texts": hexes[:3], "layers": [args.layer],
                          "aggregate": "tokens", "max_length": 64}
        print(f"  POST {args.api_base.rstrip('/')}/v1/encode")
        print(f"    {json.dumps(prefix_payload)}")
        print("[dry-run] step 2: first 3 main prompts that WOULD be POSTed:")
        for p in prompts[:3]:
            payload = {"texts": [p["prompt"]], "layers": [args.layer],
                       "aggregate": "tokens", "max_length": 64}
            print(f"  POST {args.api_base.rstrip('/')}/v1/encode")
            print(f"    {json.dumps(payload)}")
        print("[dry-run] client-side post-processing: mean over tokens[n_hex:]")
        return 0

    if not args.api_base or "localhost" in args.api_base:
        print(f"[warn] api_base={args.api_base!r} -- set COGITO_API_BASE "
              f"to a real endpoint", file=sys.stderr)

    # PRECOMPUTE per-hex token counts via a cheap one-shot batch.
    # Each hex appears in every template at position 0, so its token-count
    # tells us how many leading tokens to skip on the client-side mean.
    print(f"[pretok] encoding {len(hexes)} hex strings alone to count "
          f"prefix tokens...", flush=True)
    key = f"layer_{args.layer}"
    hex_n_tokens: dict[str, int] = {}
    for s in range(0, len(hexes), args.batch_size):
        batch_hex = hexes[s:s + args.batch_size]
        data = post_encode(args.api_base, batch_hex, layer=args.layer,
                           aggregate="tokens")
        for h, r in zip(batch_hex, data["results"]):
            arr = np.asarray(r[key], dtype=np.float32)
            if arr.ndim != 2:
                raise RuntimeError(f"hex-alone unexpected shape {arr.shape}")
            hex_n_tokens[h] = arr.shape[0]
    print(f"[pretok] hex token counts: "
          f"min={min(hex_n_tokens.values())} "
          f"max={max(hex_n_tokens.values())} "
          f"median={int(np.median(list(hex_n_tokens.values())))}", flush=True)

    # MAIN harvest loop: aggregate="tokens" returns per-token activations,
    # then client-side we mean over the post-hex span (rows[n_hex:]).
    feats: list[np.ndarray] = []
    t0 = time.time()
    for s in range(0, len(prompts), args.batch_size):
        batch = prompts[s:s + args.batch_size]
        texts = [r["prompt"] for r in batch]
        data = post_encode(args.api_base, texts, layer=args.layer,
                           aggregate="tokens")
        for r_meta, r in zip(batch, data["results"]):
            arr = np.asarray(r[key], dtype=np.float32)
            if arr.ndim != 2:
                raise RuntimeError(f"unexpected shape {arr.shape}")
            n_hex = hex_n_tokens[r_meta["hex"]]
            if arr.shape[0] <= n_hex:
                raise RuntimeError(
                    f"prompt has {arr.shape[0]} tokens but hex prefix is "
                    f"{n_hex}; no post-hex tokens to mean")
            post = arr[n_hex:].mean(axis=0)
            feats.append(post)
        elapsed = time.time() - t0
        rate = len(feats) / max(elapsed, 1e-6)
        eta = (len(prompts) - len(feats)) / max(rate, 1e-6)
        print(f"  [encode] {len(feats)}/{len(prompts)} ({rate:.1f}/s, "
              f"ETA {eta:5.1f}s)", flush=True)

    X = np.stack(feats, axis=0)
    np.save(args.out_x, X)
    Path(args.out_idx).write_text(json.dumps(prompts, indent=2))
    print(f"[done] X={X.shape} -> {args.out_x}", flush=True)
    print(f"[done] index ({len(prompts)} rows) -> {args.out_idx}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
