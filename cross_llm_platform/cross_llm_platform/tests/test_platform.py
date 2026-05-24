"""Integration coverage for the standalone cross_llm_platform package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cross_llm_platform import (
    ConceptSteerer,
    GaugeFit,
    HarvestResult,
    fit_gauge,
    label_prompts,
    validated_diagnostics,
)
from cross_llm_platform.atlas import AtlasRow, write_atlas_table
from cross_llm_platform.cli import cli
from cross_llm_platform.concepts import REGISTRY
from cross_llm_platform.ingest import load_prompts
from cross_llm_platform.server import create_app


def test_concepts_registry_numeric() -> None:
    """All predefined concept functions return numeric arrays."""
    prompt = "A formal doctor from Europe likes a bright red modern design."
    for spec in REGISTRY.values():
        value = spec.fn(prompt)
        assert value.dtype.kind == "f"
        assert value.ndim == 1
        assert value.shape[0] == len(spec.axes)


def test_gauge_steer_diagnostics_roundtrip(tmp_path: Path) -> None:
    """Fit gauge, register anchors, steer, diagnose, and save/load."""
    x, prompts = _fixture_data()
    labels = label_prompts(prompts, "hsv")
    fit = fit_gauge(x, labels, targets=["hsv"], anchor_rows={"red": [0, 3], "blue": [1, 4]})
    assert fit.d >= 1
    assert fit.axes.shape[0] == x.shape[1]
    req = ConceptSteerer(fit, layer=2).request("The next color is", "red", 2.0)
    assert req.direction.shape == (x.shape[1],)
    assert req.scale > 0.0
    report = validated_diagnostics(
        fit,
        x,
        labels,
        {"red": [0, 3], "blue": [1, 4]},
        n_perm=3,
    )
    assert "per_anchor_curvature_auto_exp_52" in report
    path = tmp_path / "gauge.npz"
    fit.save(path)
    loaded = GaugeFit.load(path)
    assert loaded.d == fit.d
    assert sorted(loaded.anchors) == ["blue", "red"]


def test_ingest_result_and_prompt_loader(tmp_path: Path) -> None:
    """HarvestResult persistence and prompt loading work without HF deps."""
    prompts = ("a", "b")
    result = HarvestResult(np.ones((2, 4), dtype=np.float32), prompts, "mock", (1,), "mean")
    path = tmp_path / "harvest.npz"
    result.save(path)
    loaded = HarvestResult.load(path)
    assert loaded.activations.shape == (2, 4)
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("one\n\ntwo\n")
    assert load_prompts(prompt_file) == ["one", "two"]


def test_cli_fit_and_steer(tmp_path: Path) -> None:
    """Click commands fit a gauge and resolve steering vectors."""
    x, prompts = _fixture_data()
    x_path = tmp_path / "x.npy"
    p_path = tmp_path / "prompts.txt"
    g_path = tmp_path / "gauge.npz"
    np.save(x_path, x)
    p_path.write_text("\n".join(prompts))
    runner = CliRunner()
    fit_res = runner.invoke(
        cli,
        [
            "platform",
            "fit-gauge",
            "--activations",
            str(x_path),
            "--prompts",
            str(p_path),
            "--concept",
            "hsv",
            "--out",
            str(g_path),
        ],
    )
    assert fit_res.exit_code == 0, fit_res.output
    fit = GaugeFit.load(g_path)
    fit.register_anchor("red", x[[0, 3]].mean(axis=0))
    fit.save(g_path)
    steer_res = runner.invoke(
        cli,
        ["platform", "steer", "--gauge", str(g_path), "--prompt", "x", "--concept", "red", "--alpha", "2"],
    )
    assert steer_res.exit_code == 0, steer_res.output
    assert "direction_len" in steer_res.output


def test_server_and_atlas_table(tmp_path: Path) -> None:
    """Server factory is importable and atlas tables are written."""
    x, prompts = _fixture_data()
    fit = fit_gauge(x, label_prompts(prompts, "hsv"), targets=["hsv"], anchor_rows={"red": [0, 3]})
    try:
        app = create_app(gauge=fit, layer=2)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert app is not None
    table = write_atlas_table(
        [AtlasRow("mock", "hsv", 2, len(prompts), x.shape[1], fit.d, 0.5, fit.d)],
        tmp_path / "atlas.csv",
    )
    assert table.read_text().startswith("model,concept")


def _fixture_data() -> tuple[np.ndarray, list[str]]:
    cached = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    prompts = [
        "A red apple is bright",
        "A blue ocean is calm",
        "A green leaf is natural",
        "The red sunset glows",
        "The blue sky is clear",
        "The green forest is quiet",
        "A yellow lemon is vivid",
        "A purple flower is soft",
        "A black stone is dark",
        "A white cloud is light",
        "An orange pumpkin is round",
        "A pink rose is delicate",
    ]
    if cached.exists():
        arr = np.load(cached, mmap_mode="r")[: len(prompts), :64].astype(np.float64)
        if np.isfinite(arr).all() and arr.std() > 0:
            return arr, prompts
    rng = np.random.default_rng(0)
    labels = label_prompts(prompts, "hsv")["hsv"]
    basis = rng.normal(size=(3, 64))
    x = labels @ basis + 0.05 * rng.normal(size=(len(prompts), 64))
    return x.astype(np.float64), prompts
