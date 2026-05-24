"""pytest suite for the dashboard backend.

Run from repo root:
    pip install -r dashboard/backend/requirements.txt
    PYTHONPATH=dashboard pytest dashboard/tests/test_backend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `backend.app` importable when pytest is run from any cwd.
_DASH = Path(__file__).resolve().parents[1]
if str(_DASH) not in sys.path:
    sys.path.insert(0, str(_DASH))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402

client = TestClient(app)


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["n_points"] >= 1


def test_manifold_shape() -> None:
    r = client.get("/api/manifold")
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) >= 100
    p = body["points"][0]
    assert {"id", "name", "hex", "rgb", "hsv", "xyz", "modifier_count", "monoword"} <= set(p.keys())
    assert len(p["xyz"]) == 3
    assert len(p["rgb"]) == 3
    assert len(p["hsv"]) == 3


def test_diagnostics() -> None:
    r = client.get("/api/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert len(body["variance_per_axis"]) == 3
    assert len(body["axis_labels"]) == 3
    assert isinstance(body["curvature_trace"], list)


def test_steer_by_concept() -> None:
    payload = {
        "hue": 0.0,
        "saturation": 1.0,
        "value": 0.9,
        "modifier_count": 1,
        "monoword": False,
        "alpha": 1.0,
        "prompt": "The color is",
    }
    r = client.post("/api/steer", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "completion" in body
    assert body["matched_hex"].startswith("#")
    assert 0.0 <= body["matched_distance"] <= 2.0


def test_steer_by_point_id() -> None:
    manifold = client.get("/api/manifold").json()
    pid = manifold["points"][42]["id"]
    payload = {
        "point_id": pid,
        "hue": 0.0,
        "saturation": 0.0,
        "value": 0.0,
        "modifier_count": 0,
        "monoword": True,
        "alpha": 0.5,
        "prompt": "Picked color:",
    }
    r = client.post("/api/steer", json=payload)
    assert r.status_code == 200
    assert "Picked color:" in r.json()["completion"]


def test_websocket_hello() -> None:
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "hello"
        assert "n_clients" in msg["data"]
