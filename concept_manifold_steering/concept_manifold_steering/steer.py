"""Anchor-offset activation steering.

Implements the steering recipe validated in Manifold-SAE auto_exp_44:
**use the difference between concept anchors, not manifold tangents**.
auto_exp_44 showed that small tangent vectors of the gauge-fixed
manifold do not carry semantic content reliably; the anchor offset
``v = mean(X | concept_to) - mean(X | concept_from)`` does.

Live steering goes through a remote vLLM server with an intervention
hook -- the same request shape used by ``cogito_intervene.py``::

    POST {server_url}/v1/chat/completions
    {
        "model": ..., "messages": [...], "stream": true,
        "extra_body": {
            "interventions": [{
                "layer": L, "vector": [...D...],
                "scale": alpha_sigma, "position": "last"
            }]
        }
    }
"""

from __future__ import annotations

import json
import os
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .gauge import GaugeFix


@dataclass
class SteerResult:
    prompt: str
    concept: str | None
    alpha: float
    completion: str
    intervened: bool
    request: dict


class ManifoldSteerer:
    """Wrap a fitted :class:`GaugeFix` with a live steering interface.

    Parameters
    ----------
    gauge
        Fitted :class:`GaugeFix` with at least one registered anchor.
    server_url
        Base URL of the OpenAI-compatible vLLM server with intervention
        hooks (e.g. ``"http://localhost:8000"``).  Read from
        ``COGITO_API_BASE`` env var if not given.
    layer
        Layer to inject the steering vector into.  Defaults to the
        ``MSAE_STEER_LAYER`` env var or 40.
    model
        Model name to send in the request.  Defaults to
        ``COGITO_MODEL`` env var or ``"cogito"``.
    base_concept
        Default ``concept_from`` for difference vectors; if ``None``,
        steering uses the *grand mean* (i.e. the bare anchor relative to
        the manifold mean) which auto_exp_44 found roughly equivalent
        but with worse off-target preservation.
    """

    def __init__(
        self,
        gauge: GaugeFix,
        server_url: str | None = None,
        *,
        layer: int | None = None,
        model: str | None = None,
        base_concept: str | None = None,
        timeout: float = 120.0,
    ):
        if gauge.axes_ is None:
            raise RuntimeError("GaugeFix is not fitted")
        self.gauge = gauge
        self.server_url = (server_url or os.environ.get("COGITO_API_BASE", "")).rstrip("/")
        self.layer = int(layer if layer is not None
                         else os.environ.get("MSAE_STEER_LAYER", "40"))
        self.model = model or os.environ.get("COGITO_MODEL", "cogito")
        self.base_concept = base_concept
        self.timeout = timeout

    # ------------------------------------------------------------ direction
    def direction(self, concept_to: str, concept_from: str | None = None) -> np.ndarray:
        """Compute the unit-norm anchor-offset steering direction in
        activation space.  If ``concept_from`` is omitted, uses the
        constructor default; if that is also ``None``, uses the manifold
        mean (``gauge.mu_``)."""
        if concept_to not in self.gauge.anchors_:
            raise KeyError(
                f"unknown concept {concept_to!r}; register via "
                f"gauge.register_anchor() or pass anchor_labels= to fit()."
            )
        to_vec = self.gauge.anchor(concept_to)
        cf = concept_from if concept_from is not None else self.base_concept
        if cf is None:
            from_vec = self.gauge.mu_
        else:
            from_vec = self.gauge.anchor(cf)
        v = (to_vec - from_vec).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-8:
            raise ValueError("zero-norm steering direction")
        return v / nrm

    def alpha_sigma(self, direction: np.ndarray) -> float:
        """1-sigma scale of ``direction`` on the harvested distribution.

        We approximate it from the fitted PCA basis: the variance of
        ``Xc @ direction`` equals ``sum( (V^T direction)^2 * lambda_k )``
        but we already have ``axes_`` orthonormal so a conservative
        estimate is the median of ``gauge.sigma()`` -- matched by
        cogito_intervene.axis_scale_unit when the axis lives in the
        gauge-fixed subspace.
        """
        # Projection onto gauge-fixed sub-basis (which is orthonormal in p).
        proj = self.gauge.axes_.T @ direction          # (d,)
        # Variance estimate from per-axis sigma:
        var = float((proj ** 2 * self.gauge.sigma_ ** 2).sum())
        # Add the orthogonal complement variance estimated from free axes
        if self.gauge.free_axes_ is not None and self.gauge.free_axes_.size:
            proj_free = self.gauge.free_axes_.T @ direction
            # Approximate per-free-axis sigma as median of fitted sigma.
            free_sigma = float(np.median(self.gauge.sigma_))
            var += float((proj_free ** 2).sum()) * free_sigma ** 2
        return float(np.sqrt(max(var, 1e-12)))

    # --------------------------------------------------------------- check
    def _locality_warning(self, concept_to: str) -> None:
        """auto_exp_85-style: warn when the concept anchor projects mostly
        outside the gauge-fixed subspace -- those steers usually destabilise
        unrelated features."""
        if concept_to not in self.gauge.anchors_:
            return
        v = self.gauge.anchor(concept_to) - self.gauge.mu_
        proj = self.gauge.axes_.T @ v
        explained = float((proj ** 2).sum() / max((v ** 2).sum(), 1e-12))
        if explained < 0.30:
            warnings.warn(
                f"concept {concept_to!r} only has {explained*100:.1f}% of its "
                f"anchor-offset variance inside the gauge-fixed subspace; "
                f"expect off-target drift (cf. auto_exp_49).",
                stacklevel=2,
            )

    # ---------------------------------------------------------------- steer
    def steer(
        self,
        prompt: str,
        concept: str,
        alpha: float = 1.0,
        *,
        concept_from: str | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stream: bool = True,
        dry_run: bool = False,
    ) -> SteerResult:
        """Generate a steered completion for ``prompt``.

        ``alpha`` is in units of the anchor-offset 1-sigma scale on the
        training harvest -- ``alpha=2.0`` is a +2σ push.
        """
        self._locality_warning(concept)
        v = self.direction(concept, concept_from=concept_from)
        scale = alpha * self.alpha_sigma(v)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
            "extra_body": {
                "interventions": [{
                    "layer": int(self.layer),
                    "vector": v.astype(np.float32).tolist(),
                    "scale": float(scale),
                    "position": "last",
                }],
            },
        }
        if dry_run or not self.server_url:
            # Truncated echo for legibility
            br = json.loads(json.dumps(body))
            for iv in br["extra_body"]["interventions"]:
                vec = iv["vector"]
                iv["vector"] = f"<len={len(vec)} first3={vec[:3]} last3={vec[-3:]}>"
            return SteerResult(prompt, concept, alpha, "", False, br)

        text = "".join(self._stream(body))
        return SteerResult(prompt, concept, alpha, text, True, {"layer": self.layer,
                                                                "scale": scale})

    def batch_steer(
        self,
        prompts: Sequence[str],
        concept: str,
        alpha: float = 1.0,
        **kw,
    ) -> list[SteerResult]:
        return [self.steer(p, concept, alpha, **kw) for p in prompts]

    # ----------------------------------------------------- transport / SSE
    def _stream(self, body: dict) -> Iterator[str]:
        """Stream a chat-completions response as SSE; yield text chunks."""
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = obj["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
                except (KeyError, IndexError, TypeError):
                    continue
