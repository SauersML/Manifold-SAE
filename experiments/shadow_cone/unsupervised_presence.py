"""Unsupervised presence detector for the shadow-cone benchmark.

This script does not use presence labels to fit the detector. It regenerates the
existing synthetic shadow-cone data, trains the plain unsupervised BSF, extracts
the benchmark target block, learns a one-dimensional high-energy direction from
training codes, and fits a two-component Gaussian mixture to that scalar energy.
Labels are used only for held-out AUC reporting.

Run:
``.venv/bin/python experiments/shadow_cone/unsupervised_presence.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "bsf_baseline"))

from shadow_cone import match_target_block, roc_auc  # noqa: E402
from bsf import BSF, BSFConfig, TrainConfig, train_bsf  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

MFILE = HERE / "metrics.json"
PRESENCE_SETUP = dict(d=64, b=4, n_blocks=8, N=2400, n_distractor=6,
                      distractor=(0.10, 0.50), weak=(0.25, 0.55),
                      strong=(1.8, 2.6), noise=0.05)


def load_metrics() -> dict:
    return json.loads(MFILE.read_text()) if MFILE.exists() else {}


def save_metrics(metrics: dict) -> None:
    MFILE.write_text(json.dumps(metrics, indent=2))


def make_presence(seed: int):
    r = PRESENCE_SETUP
    rng = np.random.default_rng(seed)
    target_basis = np.linalg.qr(rng.standard_normal((r["d"], r["b"])))[0]
    target_direction = target_basis @ rng.standard_normal(r["b"])
    target_direction /= np.linalg.norm(target_direction)
    distractors = [
        np.linalg.qr(rng.standard_normal((r["d"], r["b"])))[0]
        for _ in range(r["n_distractor"])
    ]
    X = np.zeros((r["N"], r["d"]))
    intensity = np.zeros(r["N"])
    for i in range(r["N"]):
        t = rng.random()
        a = 0.0 if t < 1 / 3 else (rng.uniform(*r["weak"]) if t < 2 / 3 else rng.uniform(*r["strong"]))
        X[i] += a * target_direction
        intensity[i] = a
        for basis in distractors:
            X[i] += basis @ (rng.standard_normal(r["b"]) * rng.uniform(*r["distractor"]))
    X += r["noise"] * rng.standard_normal((r["N"], r["d"]))
    return X.astype(np.float64), (intensity > 0).astype(int), intensity, target_basis


@dataclass
class GaussianMixture1D:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True)), axis=axis)


def fit_gaussian_mixture_1d(x: np.ndarray, steps: int = 200) -> GaussianMixture1D:
    """Two-component EM for scalar energies, initialized from quantiles."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4:
        raise ValueError("need at least four scalar energies to fit a mixture")
    floor = max(float(np.var(x)) * 1e-6, 1e-8)
    means = np.quantile(x, [0.25, 0.75]).astype(np.float64)
    variances = np.full(2, max(float(np.var(x)), floor), dtype=np.float64)
    weights = np.full(2, 0.5, dtype=np.float64)

    for _ in range(steps):
        logp = (
            np.log(weights[None, :] + 1e-300)
            - 0.5 * np.log(2.0 * np.pi * variances[None, :])
            - 0.5 * ((x[:, None] - means[None, :]) ** 2) / variances[None, :]
        )
        resp = np.exp(logp - _logsumexp(logp, axis=1)[:, None])
        nk = resp.sum(axis=0) + 1e-300
        weights = nk / x.size
        means = (resp * x[:, None]).sum(axis=0) / nk
        variances = (resp * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
        variances = np.maximum(variances, floor)

    order = np.argsort(means)
    return GaussianMixture1D(weights=weights[order], means=means[order], variances=variances[order])


def posterior_high_component(model: GaussianMixture1D, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    logp = (
        np.log(model.weights[None, :] + 1e-300)
        - 0.5 * np.log(2.0 * np.pi * model.variances[None, :])
        - 0.5 * ((x[:, None] - model.means[None, :]) ** 2) / model.variances[None, :]
    )
    return np.exp(logp[:, 1] - _logsumexp(logp, axis=1))


def posterior_threshold(model: GaussianMixture1D) -> float:
    lo = min(model.means[0] - 6.0 * np.sqrt(model.variances[0]), model.means[1] - 6.0 * np.sqrt(model.variances[1]))
    hi = max(model.means[0] + 6.0 * np.sqrt(model.variances[0]), model.means[1] + 6.0 * np.sqrt(model.variances[1]))
    grid = np.linspace(lo, hi, 20001)
    post = posterior_high_component(model, grid)
    between = (grid >= model.means[0]) & (grid <= model.means[1])
    idx = np.argmin(np.abs(post[between] - 0.5)) if between.any() else int(np.argmin(np.abs(post - 0.5)))
    return float(grid[between][idx] if between.any() else grid[idx])


def learned_directional_energy(z_train: np.ndarray, z_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn one in-block energy direction from train codes without labels."""
    center = z_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_train - center, full_matrices=False)
    direction = vt[0]
    train_energy = z_train @ direction
    eval_energy = z_eval @ direction

    # Fix the sign by calling the heavier tail "present-like"; no labels used.
    median = float(np.median(train_energy))
    if abs(float(np.percentile(train_energy, 5)) - median) > abs(float(np.percentile(train_energy, 95)) - median):
        direction = -direction
        train_energy = -train_energy
        eval_energy = -eval_energy
    return train_energy, eval_energy, direction


def per_intensity(score: np.ndarray, labels: np.ndarray, intensity: np.ndarray) -> dict:
    absent = intensity == 0
    weak = (intensity > 0) & (intensity < 1.0)
    strong = intensity >= 1.0

    def auc_vs_absent(mask: np.ndarray) -> float:
        keep = absent | mask
        return roc_auc(score[keep], mask[keep].astype(int))

    return {
        "auc_all": roc_auc(score, labels),
        "auc_weak_vs_absent": auc_vs_absent(weak),
        "auc_strong_vs_absent": auc_vs_absent(strong),
    }


def run(steps: int) -> dict:
    X, presence, intensity, target_basis = make_presence(0)
    n_train = int(0.8 * PRESENCE_SETUP["N"])
    X_train = torch.tensor(X[:n_train])
    X_test = torch.tensor(X[n_train:])
    y_test = presence[n_train:]
    intensity_test = intensity[n_train:]

    model = BSF(BSFConfig(
        d_model=PRESENCE_SETUP["d"],
        n_blocks=PRESENCE_SETUP["n_blocks"],
        block_size=PRESENCE_SETUP["b"],
        k_blocks=3,
        mode="grassmann",
        aux_k_blocks=1,
        seed=0,
    ))
    train_bsf(model, X_train, TrainConfig(steps=steps, batch_size=512, lr=3e-3))

    target_block = match_target_block(model.decoder.detach().cpu().numpy(), target_basis)
    z_train = model.encode(X_train).detach().cpu().numpy()[:, target_block, :]
    z_test = model.encode(X_test).detach().cpu().numpy()[:, target_block, :]

    train_energy, test_energy, direction = learned_directional_energy(z_train, z_test)
    mixture = fit_gaussian_mixture_1d(train_energy)
    posterior = posterior_high_component(mixture, test_energy)
    threshold = posterior_threshold(mixture)
    hard = test_energy >= threshold

    return {
        "target_block": int(target_block),
        "train_steps": int(steps),
        "detector": "unsupervised_directional_energy_mixture",
        "fit_uses_presence_labels": False,
        "score": "posterior_high_mean_component",
        "direction": direction.tolist(),
        "mixture": {
            "weights": mixture.weights.tolist(),
            "means": mixture.means.tolist(),
            "variances": mixture.variances.tolist(),
            "posterior_0_5_energy_threshold": threshold,
        },
        "heldout": {
            **per_intensity(posterior, y_test, intensity_test),
            "hard_accuracy_at_threshold": float((hard == y_test.astype(bool)).mean()),
            "hard_positive_rate_at_threshold": float(hard.mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800, help="unsupervised BSF training steps")
    parser.add_argument("--no-save", action="store_true", help="print metrics without updating metrics.json")
    args = parser.parse_args()

    result = run(args.steps)
    if not args.no_save:
        metrics = load_metrics()
        metrics.setdefault("presence", {"setup": {k: PRESENCE_SETUP[k] for k in PRESENCE_SETUP}, "detectors": {}})
        metrics["presence"]["detectors"]["unsupervised_energy_mixture"] = result
        save_metrics(metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
