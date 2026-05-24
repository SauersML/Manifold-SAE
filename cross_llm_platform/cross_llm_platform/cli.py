"""Click CLI for the cross-LLM platform."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from .concepts import REGISTRY, label_prompts
from .gauge import GaugeFit, fit_gauge
from .ingest import harvest_activations, load_prompts
from .steer import ConceptSteerer


@click.group(name="cls")
def cli() -> None:
    """Cross-LLM concept steering tools."""


@cli.group()
def platform() -> None:
    """Harvest, fit gauges, steer, and serve."""


@platform.command()
@click.option("--model", "model_name", required=True, type=str)
@click.option("--layer", required=True, type=int)
@click.option("--prompts", "prompts_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", default="harvest.npz", type=click.Path(path_type=Path))
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--max-length", default=256, show_default=True, type=int)
@click.option("--pool", default="last_token", show_default=True, type=click.Choice(["last_token", "mean", "first_token"]))
@click.option("--trust-remote-code", is_flag=True)
def harvest(
    model_name: str,
    layer: int,
    prompts_path: Path,
    out_path: Path,
    batch_size: int,
    max_length: int,
    pool: str,
    trust_remote_code: bool,
) -> None:
    """Harvest HuggingFace model activations."""
    prompts = load_prompts(prompts_path)
    result = harvest_activations(
        model_name,
        prompts,
        layer,
        batch_size=batch_size,
        max_length=max_length,
        pool=pool,  # type: ignore[arg-type]
        trust_remote_code=trust_remote_code,
    )
    result.save(out_path)
    click.echo(f"saved {result.activations.shape} activations to {out_path}")


@platform.command("fit-gauge")
@click.option("--activations", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompts", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--concept", required=True, type=click.Choice(sorted(REGISTRY)))
@click.option("--out", "out_path", default="gauge.npz", type=click.Path(path_type=Path))
@click.option("--d", default=None, type=int)
@click.option("--k", default=None, type=int)
def fit_gauge_cmd(activations: Path, prompts: Path, concept: str, out_path: Path, d: int | None, k: int | None) -> None:
    """Fit a BIC-selected gauge for a concept."""
    x = _load_activation_matrix(activations)
    prompt_list = load_prompts(prompts)
    labels = label_prompts(tuple(prompt_list), concept)
    fit = fit_gauge(x, labels, targets=[concept], d=d, k=k)
    fit.save(out_path)
    click.echo(json.dumps({"out": str(out_path), "d": fit.d, "r2": fit.r2}, indent=2))


@platform.command()
@click.option("--gauge", "gauge_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", required=True, type=str)
@click.option("--concept", required=True, type=str)
@click.option("--alpha", default=1.0, show_default=True, type=float)
@click.option("--layer", default=1, show_default=True, type=int)
def steer(gauge_path: Path, prompt: str, concept: str, alpha: float, layer: int) -> None:
    """Resolve a steering vector for a prompt/concept/alpha request."""
    gauge = GaugeFit.load(gauge_path)
    steerer = ConceptSteerer(gauge, layer=layer)
    req = steerer.request(prompt, concept, alpha)
    click.echo(
        json.dumps(
            {
                "prompt": req.prompt,
                "concept": req.concept,
                "alpha": req.alpha,
                "layer": req.layer,
                "scale": req.scale,
                "direction_len": int(req.direction.shape[0]),
                "direction_first8": req.direction[:8].tolist(),
            },
            indent=2,
        )
    )


@platform.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--gauge", "gauge_path", type=click.Path(exists=True, path_type=Path))
@click.option("--layer", default=1, show_default=True, type=int)
def serve(host: str, port: int, gauge_path: Path | None, layer: int) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException("serve requires uvicorn; install cross-llm-platform[server]") from exc
    from .server import create_app

    app = create_app(gauge_path=gauge_path, layer=layer)
    uvicorn.run(app, host=host, port=port)


def _load_activation_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["activations"], dtype=np.float64)


if __name__ == "__main__":
    cli()
