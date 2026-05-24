"""FastAPI REST and WebSocket server for the steering pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gauge import GaugeFit
from .steer import ConceptSteerer


def create_app(
    *,
    gauge_path: str | Path | None = None,
    gauge: GaugeFit | None = None,
    layer: int = 1,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> Any:
    """Create a FastAPI app exposing steering requests and generation."""
    try:
        from fastapi import FastAPI, WebSocket
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise NotImplementedError("server requires fastapi and pydantic") from exc

    class SteerBody(BaseModel):
        prompt: str
        concept: str
        alpha: float = Field(default=1.0)
        max_new_tokens: int = Field(default=64, ge=1, le=4096)

    app = FastAPI(title="Cross-LLM Concept Steering Platform")
    state_gauge = gauge if gauge is not None else (GaugeFit.load(gauge_path) if gauge_path else None)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "has_gauge": state_gauge is not None,
            "has_model": model is not None and tokenizer is not None,
        }

    @app.get("/concepts")
    def concepts() -> dict[str, object]:
        if state_gauge is None:
            return {"anchors": []}
        return {"anchors": sorted(state_gauge.anchors), "targets": state_gauge.targets, "d": state_gauge.d}

    @app.post("/steering/request")
    def steering_request(body: SteerBody) -> dict[str, object]:
        if state_gauge is None:
            raise RuntimeError("no gauge loaded")
        req = ConceptSteerer(state_gauge, layer=layer).request(body.prompt, body.concept, body.alpha)
        return {
            "prompt": req.prompt,
            "concept": req.concept,
            "alpha": req.alpha,
            "layer": req.layer,
            "scale": req.scale,
            "direction": req.direction.tolist(),
        }

    @app.post("/steer")
    def steer(body: SteerBody) -> dict[str, object]:
        if state_gauge is None:
            raise RuntimeError("no gauge loaded")
        result = ConceptSteerer(state_gauge, layer=layer).steer_text(
            body.prompt,
            body.concept,
            body.alpha,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=body.max_new_tokens,
        )
        return {"text": result.text, "scale": result.request.scale, "layer": result.request.layer}

    @app.websocket("/ws/steer")
    async def ws_steer(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            body = SteerBody.model_validate(await websocket.receive_json())
            if state_gauge is None:
                await websocket.send_json({"error": "no gauge loaded"})
                return
            result = ConceptSteerer(state_gauge, layer=layer).steer_text(
                body.prompt,
                body.concept,
                body.alpha,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=body.max_new_tokens,
            )
            for ch in result.text:
                await websocket.send_text(ch)
            await websocket.send_json({"done": True, "scale": result.request.scale})
        finally:
            await websocket.close()

    return app


try:
    app = create_app()
except NotImplementedError:
    app = None
