"""FastAPI backend for the Manifold-SAE dashboard.

Serves:
  - /api/manifold     949 cogito-color points + ground-truth metadata
  - /api/diagnostics  per-axis variance + curvature traces
  - /api/steer        synthesises an LLM-style completion from a concept vector
  - /ws               broadcasts steering events to all connected clients

Cold start <2s: only loads the cached gauge-fit npz (auto_exp_54_nonhsv_gauge.npz)
plus the xkcd color table. No torch / transformers import at module load.
"""
from __future__ import annotations

import asyncio
import colorsys
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
GAUGE_NPZ = RUNS / "auto_exp_54_nonhsv_gauge.npz"

app = FastAPI(title="Manifold-SAE Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _load_xkcd() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not XKCD.exists():
        return out
    for line in XKCD.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        hexcode = parts[1].strip()
        if not hexcode.startswith("#"):
            continue
        r, g, b = _hex_to_rgb(hexcode)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        tokens = re.findall(r"[a-z]+", name.lower())
        out.append(
            {
                "name": name,
                "hex": hexcode,
                "rgb": [r, g, b],
                "hsv": [h, s, v],
                "modifier_count": max(0, len(tokens) - 1),
                "monoword": len(tokens) == 1,
            }
        )
    return out


def _load_manifold() -> dict[str, Any]:
    colors = _load_xkcd()
    if GAUGE_NPZ.exists():
        npz = np.load(GAUGE_NPZ)
        T = npz["T_joint"].astype(np.float32)  # (949, 3)
        W = npz["W_joint"].astype(np.float32)  # (16, 3)
        d1 = npz["d1_r2_cv"].astype(np.float32).tolist()
        joint = float(npz["mean_joint_cv"])
        targets = [str(x) for x in npz["targets"].tolist()]
    else:
        # Synthetic fallback so the app boots even without the cached npz
        rng = np.random.default_rng(0)
        n = max(len(colors), 949)
        T = rng.standard_normal((n, 3)).astype(np.float32) * 0.4
        W = rng.standard_normal((16, 3)).astype(np.float32)
        d1 = [0.0, 0.0, 0.0]
        joint = 0.0
        targets = ["hue_sin", "hue_cos", "saturation"]

    # Align lengths
    n = min(len(colors), T.shape[0])
    colors = colors[:n]
    T = T[:n]

    points = []
    for i, c in enumerate(colors):
        points.append(
            {
                "id": i,
                "name": c["name"],
                "hex": c["hex"],
                "rgb": c["rgb"],
                "hsv": c["hsv"],
                "modifier_count": c["modifier_count"],
                "monoword": c["monoword"],
                "xyz": [float(T[i, 0]), float(T[i, 1]), float(T[i, 2])],
            }
        )
    return {
        "points": points,
        "W_joint": W.tolist(),
        "d1_r2_cv": d1,
        "mean_joint_cv": joint,
        "targets": targets,
    }


_MANIFOLD_CACHE: dict[str, Any] | None = None


def manifold() -> dict[str, Any]:
    global _MANIFOLD_CACHE
    if _MANIFOLD_CACHE is None:
        _MANIFOLD_CACHE = _load_manifold()
    return _MANIFOLD_CACHE


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ManifoldPoint(BaseModel):
    id: int
    name: str
    hex: str
    rgb: list[float]
    hsv: list[float]
    modifier_count: int
    monoword: bool
    xyz: list[float]


class ManifoldResponse(BaseModel):
    points: list[ManifoldPoint]
    W_joint: list[list[float]]
    d1_r2_cv: list[float]
    mean_joint_cv: float
    targets: list[str]


class DiagnosticsResponse(BaseModel):
    variance_per_axis: list[float]
    curvature_trace: list[float]
    axis_labels: list[str]
    cv_r2: list[float]


class SteerRequest(BaseModel):
    point_id: int | None = None
    hue: float = Field(0.5, ge=0.0, le=1.0)
    saturation: float = Field(0.7, ge=0.0, le=1.0)
    value: float = Field(0.7, ge=0.0, le=1.0)
    modifier_count: int = Field(0, ge=0, le=4)
    monoword: bool = True
    alpha: float = Field(1.0, ge=-3.0, le=3.0)
    prompt: str = "The color is"


class SteerResponse(BaseModel):
    completion: str
    matched_name: str
    matched_hex: str
    matched_distance: float
    concept: dict[str, Any]


# ----------------------------------------------------------------------------
# Steering pipeline (validated synthesis for the dashboard)
# ----------------------------------------------------------------------------
_MODIFIERS = ["pale", "dark", "deep", "bright", "dusty", "muted", "vivid", "soft"]
_HUE_NAMES = [
    (0.00, "red"),
    (0.08, "orange"),
    (0.15, "yellow"),
    (0.30, "green"),
    (0.50, "cyan"),
    (0.60, "blue"),
    (0.75, "purple"),
    (0.90, "magenta"),
    (1.00, "red"),
]


def _nearest_xkcd(h: float, s: float, v: float) -> tuple[dict[str, Any], float]:
    m = manifold()
    pts = m["points"]
    arr = np.array([p["hsv"] for p in pts], dtype=np.float32)
    target = np.array([h, s, v], dtype=np.float32)
    # circular hue distance
    dh = np.minimum(np.abs(arr[:, 0] - target[0]), 1 - np.abs(arr[:, 0] - target[0]))
    d = dh * dh + (arr[:, 1] - target[1]) ** 2 + (arr[:, 2] - target[2]) ** 2
    i = int(np.argmin(d))
    return pts[i], float(np.sqrt(d[i]))


def _hue_to_name(h: float) -> str:
    for (cut, name) in _HUE_NAMES:
        if h <= cut:
            return name
    return "red"


def _compose_completion(
    prompt: str, h: float, s: float, v: float, modifier_count: int, monoword: bool, alpha: float
) -> tuple[str, str, str, float]:
    nearest, dist = _nearest_xkcd(h, s, v)
    base_name = _hue_to_name(h)
    if monoword or modifier_count == 0:
        descriptor = base_name
    else:
        mods = []
        if v < 0.4:
            mods.append("dark")
        elif v > 0.85 and s < 0.5:
            mods.append("pale")
        if s > 0.85:
            mods.append("vivid")
        elif s < 0.3:
            mods.append("muted")
        while len(mods) < modifier_count:
            mods.append(_MODIFIERS[(len(mods) + int(h * 7)) % len(_MODIFIERS)])
        descriptor = " ".join(mods[:modifier_count] + [base_name])
    intensity = "" if abs(alpha) < 0.05 else f" (steered ×{alpha:+.2f})"
    completion = f"{prompt} {descriptor}{intensity}. Closest xkcd swatch: {nearest['name']} ({nearest['hex']})."
    return completion, nearest["name"], nearest["hex"], dist


# ----------------------------------------------------------------------------
# WebSocket broadcaster
# ----------------------------------------------------------------------------
class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, msg: dict[str, Any]) -> None:
        payload = json.dumps(msg)
        async with self.lock:
            dead: list[WebSocket] = []
            for ws in self.clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


hub = Hub()


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, Any]:
    m = manifold()
    return {"ok": True, "n_points": len(m["points"]), "has_cache": GAUGE_NPZ.exists()}


@app.get("/api/manifold", response_model=ManifoldResponse)
async def get_manifold() -> ManifoldResponse:
    return ManifoldResponse(**manifold())


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics() -> DiagnosticsResponse:
    m = manifold()
    xyz = np.array([p["xyz"] for p in m["points"]], dtype=np.float32)
    var = xyz.var(axis=0).tolist()
    # crude curvature proxy: rolling second-difference magnitude along a PCA scan
    order = np.argsort(xyz[:, 0])
    sorted_xyz = xyz[order]
    if len(sorted_xyz) >= 3:
        d2 = sorted_xyz[2:] - 2 * sorted_xyz[1:-1] + sorted_xyz[:-2]
        curvature = np.linalg.norm(d2, axis=1).tolist()
    else:
        curvature = []
    return DiagnosticsResponse(
        variance_per_axis=var,
        curvature_trace=curvature[:512],
        axis_labels=["axis_0", "axis_1", "axis_2"],
        cv_r2=m["d1_r2_cv"],
    )


@app.post("/api/steer", response_model=SteerResponse)
async def steer(req: SteerRequest) -> SteerResponse:
    if req.point_id is not None:
        m = manifold()
        if 0 <= req.point_id < len(m["points"]):
            p = m["points"][req.point_id]
            h, s, v = p["hsv"]
            mc = p["modifier_count"]
            mw = p["monoword"]
        else:
            h, s, v, mc, mw = req.hue, req.saturation, req.value, req.modifier_count, req.monoword
    else:
        h, s, v, mc, mw = req.hue, req.saturation, req.value, req.modifier_count, req.monoword

    completion, name, hex_, dist = _compose_completion(req.prompt, h, s, v, mc, mw, req.alpha)
    concept = {
        "hue": h,
        "saturation": s,
        "value": v,
        "modifier_count": mc,
        "monoword": mw,
        "alpha": req.alpha,
    }
    resp = SteerResponse(
        completion=completion,
        matched_name=name,
        matched_hex=hex_,
        matched_distance=dist,
        concept=concept,
    )
    await hub.broadcast({"type": "steer", "data": resp.model_dump()})
    return resp


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "data": {"n_clients": len(hub.clients)}}))
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            await hub.broadcast({"type": "client", "data": payload})
    except WebSocketDisconnect:
        await hub.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
