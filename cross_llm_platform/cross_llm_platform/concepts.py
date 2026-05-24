"""Predefined concept-label functions for prompts.

Each concept function accepts a prompt and returns numeric labels. Scalar
concepts return a one-element NumPy array; multi-axis concepts return one
value per axis. The registry exposes these functions by stable names for
CLI, server, atlas, and tests.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

ConceptFunction = Callable[[str], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class ConceptSpec:
    """Registered concept-label function metadata."""

    name: str
    axes: tuple[str, ...]
    fn: ConceptFunction
    description: str


COLOR_RGB: Mapping[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.0, 0.7, 0.1),
    "blue": (0.0, 0.15, 1.0),
    "purple": (0.55, 0.0, 0.8),
    "pink": (1.0, 0.45, 0.75),
    "brown": (0.45, 0.25, 0.08),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}

POSITIVE = frozenset("good great excellent happy joyful love wonderful bright success calm".split())
NEGATIVE = frozenset("bad awful terrible sad angry hate grim failure dark anxious".split())
FORMAL = frozenset("therefore pursuant consequently regarding sincerely shall hereby moreover".split())
INFORMAL = frozenset("hey yeah gonna kinda lol dude ok okay awesome nope".split())
PERSONA = ("doctor", "lawyer", "teacher", "engineer", "artist", "scientist", "chef", "parent")
TIME_PERIODS = {
    "ancient": -2.0,
    "medieval": -1.2,
    "renaissance": -0.7,
    "victorian": -0.3,
    "modern": 0.4,
    "future": 1.5,
    "futuristic": 1.7,
}
REGIONS = ("north america", "europe", "asia", "africa", "south america", "oceania", "middle east")


def hsv_for_color(prompt: str) -> NDArray[np.float64]:
    """Return ``[hue, saturation, value]`` inferred from color words."""
    text = prompt.lower()
    hits = [(name, rgb) for name, rgb in COLOR_RGB.items() if re.search(rf"\b{name}\b", text)]
    if not hits:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    _name, rgb = hits[0]
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return np.asarray([h, s, v], dtype=np.float64)


def valence_for_sentiment(prompt: str) -> NDArray[np.float64]:
    """Return ``[valence]`` from a small polarity lexicon in ``[-1, 1]``."""
    toks = _tokens(prompt)
    pos = sum(tok in POSITIVE for tok in toks)
    neg = sum(tok in NEGATIVE for tok in toks)
    denom = max(pos + neg, 1)
    return np.asarray([(pos - neg) / denom], dtype=np.float64)


def formality(prompt: str) -> NDArray[np.float64]:
    """Return ``[formality]`` from informal/formal lexical cues."""
    toks = _tokens(prompt)
    formal = sum(tok in FORMAL for tok in toks)
    informal = sum(tok in INFORMAL for tok in toks)
    return np.asarray([(formal - informal) / max(formal + informal, 1)], dtype=np.float64)


def persona(prompt: str) -> NDArray[np.float64]:
    """Return one-hot persona cues for common role words."""
    text = prompt.lower()
    return np.asarray([1.0 if re.search(rf"\b{role}\b", text) else 0.0 for role in PERSONA])


def time_period(prompt: str) -> NDArray[np.float64]:
    """Return ``[time_period]`` ordered from ancient past to future."""
    text = prompt.lower()
    vals = [score for word, score in TIME_PERIODS.items() if re.search(rf"\b{word}\b", text)]
    return np.asarray([float(np.mean(vals)) if vals else 0.0], dtype=np.float64)


def geographic_region(prompt: str) -> NDArray[np.float64]:
    """Return one-hot region cues."""
    text = prompt.lower().replace("-", " ")
    return np.asarray([1.0 if region in text else 0.0 for region in REGIONS], dtype=np.float64)


REGISTRY: dict[str, ConceptSpec] = {
    "hsv": ConceptSpec("hsv", ("hue", "saturation", "value"), hsv_for_color, "HSV color labels"),
    "sentiment": ConceptSpec("sentiment", ("valence",), valence_for_sentiment, "Sentiment valence"),
    "formality": ConceptSpec("formality", ("formality",), formality, "Formality score"),
    "persona": ConceptSpec("persona", PERSONA, persona, "Persona role indicators"),
    "time-period": ConceptSpec("time-period", ("period",), time_period, "Historical period score"),
    "geographic-region": ConceptSpec(
        "geographic-region", REGIONS, geographic_region, "Geographic region indicators"
    ),
}


def label_prompts(prompts: list[str] | tuple[str, ...], concept: str) -> dict[str, NDArray[np.float64]]:
    """Label prompts with a registered concept.

    Args:
        prompts: Prompt strings.
        concept: Registry key.

    Returns:
        Mapping suitable for :func:`cross_llm_platform.gauge.fit_gauge`.
    """
    if concept not in REGISTRY:
        raise KeyError(f"unknown concept {concept!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[concept]
    values = np.stack([spec.fn(prompt) for prompt in prompts], axis=0)
    return {concept: values}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())
