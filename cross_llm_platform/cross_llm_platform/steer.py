"""Typed anchor-offset steering API.

The steering vector follows the ``auto_exp_44`` lesson: steer with
concept-anchor offsets in activation space, not tangents of the fitted
chart. ``alpha`` is measured in training-distribution sigma units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .gauge import GaugeFit


@dataclass(frozen=True, slots=True)
class SteeringRequest:
    """Resolved steering request ready for hook injection."""

    prompt: str
    concept: str
    alpha: float
    layer: int
    direction: NDArray[np.float64]
    scale: float


@dataclass(frozen=True, slots=True)
class SteeringResult:
    """Text generation result from a steered request."""

    prompt: str
    concept: str
    alpha: float
    text: str
    request: SteeringRequest


class ConceptSteerer:
    """Anchor-offset steerer backed by a fitted :class:`GaugeFit`.

    Args:
        gauge: Fitted gauge with registered anchors.
        layer: Decoder block index to edit. Hook indexing starts at one to
            match HuggingFace ``hidden_states``.
        base_concept: Optional source anchor. If omitted, offsets are from
            the harvest mean to the target anchor.
        position: ``last_token`` or ``all_tokens`` injection position.
    """

    def __init__(
        self,
        gauge: GaugeFit,
        *,
        layer: int,
        base_concept: str | None = None,
        position: str = "last_token",
    ) -> None:
        self.gauge = gauge
        self.layer = int(layer)
        self.base_concept = base_concept
        if position not in {"last_token", "all_tokens"}:
            raise ValueError("position must be 'last_token' or 'all_tokens'")
        self.position = position

    def direction(self, concept: str, *, base_concept: str | None = None) -> NDArray[np.float64]:
        """Return the unit anchor-offset direction for ``concept``."""
        target = self.gauge.anchor(concept)
        base_name = self.base_concept if base_concept is None else base_concept
        base = self.gauge.mu if base_name is None else self.gauge.anchor(base_name)
        raw = target - base
        norm = float(np.linalg.norm(raw))
        if norm <= 1e-12:
            raise ValueError(f"zero-norm steering direction for concept {concept!r}")
        return raw / norm

    def scale_for(self, direction: NDArray[np.float64], alpha: float) -> float:
        """Convert ``alpha`` sigma units into an activation-space scale."""
        proj = self.gauge.axes.T @ direction
        var = float(((proj * self.gauge.sigma) ** 2).sum())
        if self.gauge.free_axes.size:
            free_proj = self.gauge.free_axes.T @ direction
            var += float((free_proj**2).sum()) * float(np.median(self.gauge.sigma) ** 2)
        return float(alpha) * float(np.sqrt(max(var, 1e-12)))

    def request(self, prompt: str, concept: str, alpha: float = 1.0) -> SteeringRequest:
        """Resolve a prompt/concept/alpha triple into vector plus scale."""
        direction = self.direction(concept)
        return SteeringRequest(
            prompt=prompt,
            concept=concept,
            alpha=float(alpha),
            layer=self.layer,
            direction=direction,
            scale=self.scale_for(direction, alpha),
        )

    def steer_text(
        self,
        prompt: str,
        concept: str,
        alpha: float = 1.0,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_new_tokens: int = 64,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> SteeringResult:
        """Generate steered text using a local HuggingFace model.

        ``model`` and ``tokenizer`` are explicit to keep the steerer
        independent of any one serving stack. When omitted, this method
        raises a clear error rather than silently returning an echo.
        """
        req = self.request(prompt, concept, alpha)
        if model is None or tokenizer is None:
            raise NotImplementedError(
                "steer_text requires a loaded HuggingFace model and tokenizer. "
                "Use request() for a serializable vector or server.py for REST/WebSocket serving."
            )
        handles = self.forward_hook(model, req)
        try:
            import torch

            device = next(model.parameters()).device
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            kwargs.update(generation_kwargs or {})
            with torch.no_grad():
                output = model.generate(**encoded, **kwargs)
            text = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        finally:
            for handle in handles:
                handle.remove()
        return SteeringResult(prompt=prompt, concept=concept, alpha=float(alpha), text=text, request=req)

    def batch_steer(
        self,
        prompts: Sequence[str],
        concept: str,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> list[SteeringResult]:
        """Run :meth:`steer_text` for multiple prompts."""
        return [self.steer_text(prompt, concept, alpha, **kwargs) for prompt in prompts]

    def forward_hook(self, model: Any, request: SteeringRequest | None = None) -> list[Any]:
        """Install a HuggingFace forward hook and return removable handles."""
        req = request or SteeringRequest(
            prompt="",
            concept="",
            alpha=1.0,
            layer=self.layer,
            direction=np.zeros_like(self.gauge.mu),
            scale=0.0,
        )
        layers = _decoder_layers(model)
        if self.layer <= 0 or self.layer > len(layers):
            raise ValueError(f"layer {self.layer} outside model layer range 1..{len(layers)}")
        module = layers[self.layer - 1]

        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            delta = hidden.new_tensor(req.direction * req.scale).view(1, 1, -1)
            if delta.shape[-1] != hidden.shape[-1]:
                raise ValueError(
                    f"steering vector width {delta.shape[-1]} != hidden width {hidden.shape[-1]}"
                )
            edited = hidden.clone()
            if self.position == "last_token":
                edited[:, -1:, :] = edited[:, -1:, :] + delta
            else:
                edited = edited + delta
            if isinstance(output, tuple):
                return (edited, *output[1:])
            return edited

        return [module.register_forward_hook(_hook)]


def _decoder_layers(model: Any) -> Sequence[Any]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise ValueError("could not locate decoder layers on model")
