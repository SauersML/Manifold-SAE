"""Poincaré-ball primitives for the Hyperbolic SAE.

All formulas use curvature c > 0 (so the ball has radius 1/sqrt(c)). Negative
curvature in the geometric sense; we parameterize the magnitude. Clamps norms
strictly inside the open ball.

Numerically stable, MPS-safe (no double precision, no in-place on views).
"""

from __future__ import annotations

import torch

# Stay strictly inside the open ball.
_BALL_EPS = 1e-5
_NORM_EPS = 1e-15


def _clip_to_ball(x: torch.Tensor, c: float) -> torch.Tensor:
    """Clip x to have ‖x‖ ≤ (1/sqrt(c)) · (1-eps)."""
    radius = (1.0 - _BALL_EPS) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    factor = torch.clamp(radius / norm, max=1.0)
    return x * factor


def exp_0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at origin: tangent vector v ↦ point in ball.

    exp_0^c(v) = tanh(sqrt(c) ‖v‖) · v / (sqrt(c) ‖v‖)
    """
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return _clip_to_ball(coeff * v, c)


def log_0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithm at origin: point in ball ↦ tangent vector at origin.

    log_0^c(x) = atanh(sqrt(c) ‖x‖) · x / (sqrt(c) ‖x‖)
    """
    sqrt_c = c ** 0.5
    x = _clip_to_ball(x, c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    coeff = torch.atanh((sqrt_c * x_norm).clamp(max=1.0 - _BALL_EPS)) / (sqrt_c * x_norm)
    return coeff * x


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Möbius addition x ⊕_c y in the Poincaré ball (broadcasting)."""
    x = _clip_to_ball(x, c)
    y = _clip_to_ball(y, c)
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    den = (1.0 + 2.0 * c * xy + c * c * x2 * y2).clamp_min(_NORM_EPS)
    return _clip_to_ball(num / den, c)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Geodesic distance d_c(x, y) = (2/sqrt(c)) · atanh(sqrt(c) ‖(-x) ⊕_c y‖)."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp_min(_NORM_EPS)
    arg = (sqrt_c * diff_norm).clamp(max=1.0 - _BALL_EPS)
    return (2.0 / sqrt_c) * torch.atanh(arg)
