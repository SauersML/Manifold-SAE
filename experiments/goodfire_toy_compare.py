"""Compare Manifold-SAE against the BSF paper's additive-manifold toy.

This is a local reproduction harness for the "Toy model of Manifold
Superposition" in arXiv:2606.25234 Appendix H. It builds sparse sums of known
low-dimensional manifolds embedded in R^D, trains the local BSF reimplementation
and the current gamfit Manifold-SAE solver, then scores per-factor contribution
recovery against the known ground truth.

The default is a laptop-scale, figure-style six-factor run. Use
``--n-factors 128 --n-train 300000`` only on a real workstation/cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

import gamfit

ROOT = Path(__file__).resolve().parents[1]
BSF_DIR = ROOT / "experiments" / "bsf_baseline"
if str(BSF_DIR) not in sys.path:
    sys.path.insert(0, str(BSF_DIR))

from bsf import BSF, BSFConfig, TrainConfig, ev, train_bsf  # noqa: E402


@dataclass(frozen=True)
class Config:
    out_dir: Path = ROOT / "runs" / "GOODFIRE_TOY_COMPARE"
    seed: int = 0
    n_factors: int = 6
    d_ambient: int = 32
    n_train: int = 4000
    n_eval: int = 1500
    active_count: int = 3
    noise: float = 0.0
    bsf_steps: int = 1200
    bsf_lr: float = 4.0e-3
    bsf_batch_size: int = 512
    gamfit_iter: int = 25
    gamfit_methods: tuple[str, ...] = ("manifold_sae_linear", "manifold_sae_mixed", "manifold_sae_patch")
    skip_bsf: bool = False
    skip_gamfit: bool = False


@dataclass(frozen=True)
class Primitive:
    name: str
    span_dim: int
    intrinsic_dim: int
    gamfit_basis: str
    sampler: Callable[[np.random.Generator, int], np.ndarray]


def _center_scale(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    rms = math.sqrt(float(np.mean(np.sum(centered * centered, axis=1))))
    return centered / max(rms, 1.0e-12)


def _segment(rng: np.random.Generator, n: int) -> np.ndarray:
    t = rng.uniform(-1.0, 1.0, n)
    return _center_scale(t[:, None])


def _circle(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return _center_scale(np.c_[np.cos(theta), np.sin(theta)])


def _disk(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    return _center_scale(np.c_[r * np.cos(theta), r * np.sin(theta)])


def _sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.uniform(-1.0, 1.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.clip(1.0 - u * u, 0.0, None))
    return _center_scale(np.c_[s * np.cos(theta), s * np.sin(theta), u])


def _torus(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    major, minor = 2.0, 0.55
    return _center_scale(
        np.c_[
            (major + minor * np.cos(phi)) * np.cos(theta),
            (major + minor * np.cos(phi)) * np.sin(theta),
            minor * np.sin(phi),
        ]
    )


def _mobius(rng: np.random.Generator, n: int) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    t = rng.uniform(-0.45, 0.45, n)
    return _center_scale(
        np.c_[
            (1.0 + t * np.cos(phi / 2.0)) * np.cos(phi),
            (1.0 + t * np.cos(phi / 2.0)) * np.sin(phi),
            t * np.sin(phi / 2.0),
        ]
    )


def _swiss_roll(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(-1.0, 1.0, n)
    return _center_scale(np.c_[theta * np.cos(theta), h, theta * np.sin(theta)])


def _helix(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 4.0 * np.pi, n)
    return _center_scale(np.c_[np.cos(theta), np.sin(theta), theta / (2.0 * np.pi)])


ZOO: tuple[Primitive, ...] = (
    Primitive("segment", 1, 1, "linear", _segment),
    Primitive("circle", 2, 1, "periodic", _circle),
    Primitive("disk", 2, 2, "euclidean", _disk),
    Primitive("sphere", 3, 2, "sphere", _sphere),
    Primitive("torus", 3, 2, "torus", _torus),
    Primitive("mobius", 3, 2, "euclidean", _mobius),
    Primitive("swiss_roll", 3, 2, "euclidean", _swiss_roll),
    Primitive("helix", 3, 2, "euclidean", _helix),
)

# Chosen to visually resemble the six rows in the paper's Figure 4.
FIGURE_KIND_ORDER = ("torus", "sphere", "disk", "helix", "torus", "circle")


def _primitive_sequence(n_factors: int) -> list[Primitive]:
    by_name = {p.name: p for p in ZOO}
    if n_factors <= len(FIGURE_KIND_ORDER):
        return [by_name[name] for name in FIGURE_KIND_ORDER[:n_factors]]
    out = [by_name[name] for name in FIGURE_KIND_ORDER]
    while len(out) < n_factors:
        out.append(ZOO[len(out) % len(ZOO)])
    return out


def _random_frame(rng: np.random.Generator, rows: int, d_ambient: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d_ambient, rows)))
    return q[:, :rows].T.copy()


@dataclass
class ToyData:
    x_train: np.ndarray
    x_eval: np.ndarray
    active_eval: np.ndarray
    contrib_eval: np.ndarray
    primitive_names: list[str]
    primitive_bases: list[str]
    primitive_dims: list[int]
    primitive_span_dims: list[int]


def make_toy(cfg: Config) -> ToyData:
    if cfg.active_count < 1 or cfg.active_count > cfg.n_factors:
        raise ValueError("active_count must be in [1, n_factors]")
    primitives = _primitive_sequence(cfg.n_factors)
    if sum(p.span_dim for p in primitives) > cfg.d_ambient:
        # Independent random frames still work when overcomplete, but this
        # warning catches accidental tiny-D figure runs.
        print(
            f"[warn] total local span {sum(p.span_dim for p in primitives)} "
            f"exceeds d_ambient={cfg.d_ambient}; subspaces will overlap.",
            flush=True,
        )

    rng = np.random.default_rng(cfg.seed)
    frames = [_random_frame(rng, p.span_dim, cfg.d_ambient) for p in primitives]

    def sample(n: int, keep_contrib: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = np.zeros((n, cfg.n_factors), dtype=bool)
        contrib = np.zeros((n, cfg.n_factors, cfg.d_ambient), dtype=np.float64)
        x = np.zeros((n, cfg.d_ambient), dtype=np.float64)
        for row in range(n):
            sel = rng.choice(cfg.n_factors, cfg.active_count, replace=False)
            active[row, sel] = True
        for i, primitive in enumerate(primitives):
            rows = np.flatnonzero(active[:, i])
            if rows.size == 0:
                continue
            local = primitive.sampler(rng, rows.size)
            embedded = local @ frames[i]
            x[rows] += embedded
            if keep_contrib:
                contrib[rows, i, :] = embedded
        if cfg.noise > 0.0:
            x += cfg.noise * rng.standard_normal(x.shape)
        return x, active, contrib

    x_train, _, _ = sample(cfg.n_train, keep_contrib=False)
    x_eval, active_eval, contrib_eval = sample(cfg.n_eval, keep_contrib=True)
    scale = math.sqrt(float(np.mean(np.sum(x_train * x_train, axis=1))))
    scale = max(scale, 1.0e-12)
    return ToyData(
        x_train=(x_train / scale).astype(np.float64),
        x_eval=(x_eval / scale).astype(np.float64),
        active_eval=active_eval,
        contrib_eval=(contrib_eval / scale).astype(np.float64),
        primitive_names=[p.name for p in primitives],
        primitive_bases=[p.gamfit_basis for p in primitives],
        primitive_dims=[p.intrinsic_dim for p in primitives],
        primitive_span_dims=[p.span_dim for p in primitives],
    )


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    if sst <= 1.0e-12:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / sst)


def _match_from_matrix(score: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    clean = np.nan_to_num(score, nan=-1.0e9)
    row, col = linear_sum_assignment(-clean)
    pairs = [(int(r), int(c)) for r, c in zip(row, col)]
    vals = [float(score[r, c]) for r, c in pairs if np.isfinite(score[r, c])]
    return pairs, float(np.mean(vals)) if vals else float("nan")


def score_block_contrib(
    contrib: np.ndarray,
    active: np.ndarray,
    block_contrib: np.ndarray,
    block_scores: np.ndarray,
) -> dict[str, object]:
    n_factors = contrib.shape[1]
    n_blocks = block_contrib.shape[1]
    r2 = np.full((n_factors, n_blocks), np.nan, dtype=np.float64)
    auc = np.full((n_factors, n_blocks), np.nan, dtype=np.float64)
    for i in range(n_factors):
        rows = active[:, i]
        if rows.sum() < 3:
            continue
        y = active[:, i].astype(int)
        for g in range(n_blocks):
            r2[i, g] = _r2(contrib[rows, i, :], block_contrib[rows, g, :])
            s = block_scores[:, g]
            if np.unique(y).size == 2 and np.isfinite(s).all():
                auc[i, g] = roc_auc_score(y, s)
    r2_pairs, r2_mean = _match_from_matrix(r2)
    auc_pairs, auc_mean = _match_from_matrix(auc)
    return {
        "mean_recovery_r2": r2_mean,
        "mean_presence_auc": auc_mean,
        "r2_matrix": r2.tolist(),
        "auc_matrix": auc.tolist(),
        "r2_matches": r2_pairs,
        "auc_matches": auc_pairs,
    }


@torch.no_grad()
def score_bsf_model(model: BSF, data: ToyData) -> dict[str, object]:
    x = torch.tensor(data.x_eval, dtype=torch.float64)
    out = model(x, update_util=False)
    z = out.z_sparse.cpu().numpy()
    dec = model.decoder.detach().cpu().numpy()
    block_contrib = np.einsum("ngb,gbd->ngd", z, dec)
    scores = np.linalg.norm(z, axis=2)
    scored = score_block_contrib(data.contrib_eval, data.active_eval, block_contrib, scores)
    scored["global_ev"] = ev(model, x)
    return scored


def run_bsf(cfg: Config, data: ToyData) -> dict[str, object]:
    x_train = torch.tensor(data.x_train, dtype=torch.float64)
    x_eval = torch.tensor(data.x_eval, dtype=torch.float64)
    results: dict[str, object] = {}
    specs = {
        "topk_sae": ("vanilla", 1, cfg.n_factors * max(data.primitive_span_dims)),
        "vanilla_bsf": ("vanilla", max(data.primitive_span_dims), cfg.n_factors),
        "grassmann_bsf": ("grassmann", max(data.primitive_span_dims), cfg.n_factors),
    }
    for name, (mode, block_size, n_blocks) in specs.items():
        print(f"[bsf] training {name}", flush=True)
        model = BSF(
            BSFConfig(
                d_model=cfg.d_ambient,
                n_blocks=n_blocks,
                block_size=block_size,
                k_blocks=cfg.active_count if block_size > 1 else cfg.active_count * block_size,
                mode=mode,
                aux_k_blocks=1,
                seed=cfg.seed,
            )
        )
        train_bsf(
            model,
            x_train,
            TrainConfig(
                steps=cfg.bsf_steps,
                batch_size=cfg.bsf_batch_size,
                lr=cfg.bsf_lr,
                seed=cfg.seed,
                log_every=max(cfg.bsf_steps // 4, 1),
            ),
            X_val=x_eval,
            verbose=True,
        )
        results[name] = score_bsf_model(model, data)
    return results


def _fit_gamfit(data: ToyData, cfg: Config, method: str) -> object:
    if method == "manifold_sae_softmax_circle":
        kwargs = dict(
            K=cfg.n_factors,
            d_atom=1,
            atom_topology="circle",
            assignment="softmax",
            n_iter=max(1, min(cfg.gamfit_iter, 1)),
            top_k=cfg.active_count,
            sparsity_weight=0.0,
            smoothness_weight=1.0e-6,
            isometry_weight=0.0,
            ard_per_atom=False,
            decoder_incoherence_weight=0.0,
            nuclear_norm_weight=0.0,
            random_state=cfg.seed,
        )
    elif method == "manifold_sae_linear":
        kwargs = dict(
            K=cfg.n_factors,
            d_atom=1,
            atom_topology="linear",
            assignment="ibp_map",
            n_iter=max(3, min(cfg.gamfit_iter, 3)),
            random_state=cfg.seed,
        )
    else:
        kwargs = dict(
            K=cfg.n_factors,
            assignment="ibp_map",
            top_k=cfg.active_count,
            n_iter=cfg.gamfit_iter,
            sparsity_weight=0.01,
            coord_sparsity="l1",
            smoothness_weight=0.01,
            isometry_weight=0.1,
            ard_per_atom=False,
            decoder_incoherence_weight=0.1,
            nuclear_norm_weight=0.0,
            random_state=cfg.seed,
            alpha="auto",
        )
    if method == "manifold_sae_mixed":
        # gamfit 0.1.247 still routes some analytic penalties through a unified
        # latent-coordinate row block; heterogeneous atom dimensions are rejected
        # before the solver starts. Use a common two-coordinate chart and keep
        # topology/basis heterogeneous. This gives 1D factors one redundant
        # coordinate, but it tests the current mixed-basis solver instead of only
        # its validation guard.
        kwargs["d_atom"] = [max(2, d) for d in data.primitive_dims]
        kwargs["atom_basis"] = data.primitive_bases
    elif method == "manifold_sae_patch":
        kwargs["d_atom"] = 2
        kwargs["atom_topology"] = "euclidean"
    elif method not in {"manifold_sae_linear", "manifold_sae_softmax_circle"}:
        raise ValueError(f"unknown gamfit method: {method}")
    model = gamfit.sae_manifold_fit(data.x_train, **kwargs)
    return model, ""


def score_gamfit_model(model: object, data: ToyData) -> dict[str, object]:
    assignments = np.asarray(model.encode(data.x_eval), dtype=np.float64)
    atoms = []
    for k in range(len(model.atoms)):
        atom = np.asarray(model.atom_reconstruct(data.x_eval, k), dtype=np.float64)
        atoms.append(assignments[:, k : k + 1] * atom)
    block_contrib = np.stack(atoms, axis=1)
    scored = score_block_contrib(data.contrib_eval, data.active_eval, block_contrib, assignments)
    scored["global_ev"] = _r2(data.x_eval, model.reconstruct(data.x_eval))
    scored["fit_reconstruction_r2_train"] = float(getattr(model, "reconstruction_r2", float("nan")))
    scored["atom_topology"] = getattr(model, "atom_topology", None)
    scored["basis_specs"] = list(getattr(model, "basis_specs", []))
    return scored


def run_gamfit(cfg: Config, data: ToyData) -> dict[str, object]:
    results: dict[str, object] = {}
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = cfg.out_dir / "gamfit_child_data.npz"
    np.savez(
        data_path,
        x_train=data.x_train,
        x_eval=data.x_eval,
        active_eval=data.active_eval,
        contrib_eval=data.contrib_eval,
        primitive_names=np.array(data.primitive_names, dtype=object),
        primitive_bases=np.array(data.primitive_bases, dtype=object),
        primitive_dims=np.array(data.primitive_dims, dtype=np.int64),
        primitive_span_dims=np.array(data.primitive_span_dims, dtype=np.int64),
    )
    for name in cfg.gamfit_methods:
        print(f"[gamfit] fitting {name}", flush=True)
        child_out = cfg.out_dir / f"{name}.json"
        child_log = cfg.out_dir / f"{name}.log"
        child_out.unlink(missing_ok=True)
        child_log.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--gamfit-child",
            name,
            "--data-npz",
            str(data_path),
            "--child-out",
            str(child_out),
            "--seed",
            str(cfg.seed),
            "--n-factors",
            str(cfg.n_factors),
            "--d-ambient",
            str(cfg.d_ambient),
            "--n-train",
            str(cfg.n_train),
            "--n-eval",
            str(cfg.n_eval),
            "--active-count",
            str(cfg.active_count),
            "--noise",
            str(cfg.noise),
            "--gamfit-iter",
            str(cfg.gamfit_iter),
            "--skip-bsf",
        ]
        with child_log.open("w") as log_file:
            try:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                      check=False, timeout=120)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc = subprocess.CompletedProcess(cmd, returncode=124)
                timed_out = True
        if child_out.exists():
            scored = json.loads(child_out.read_text())
        else:
            tail = "\n".join(child_log.read_text(errors="replace").splitlines()[-20:])
            scored = {
                "error": "child timed out" if timed_out else f"child exited {proc.returncode}",
                "fit_log_tail": tail,
            }
        scored["child_returncode"] = proc.returncode
        scored["child_log"] = str(child_log)
        results[name] = scored
    return results


def write_outputs(cfg: Config, data: ToyData, results: dict[str, object]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**asdict(cfg), "out_dir": str(cfg.out_dir)},
        "versions": {
            "gamfit": gamfit.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "primitive_names": data.primitive_names,
        "primitive_bases": data.primitive_bases,
        "primitive_dims": data.primitive_dims,
        "results": results,
    }
    (cfg.out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for method, result in results.items():
        if not isinstance(result, dict):
            continue
        rows.append(
            {
                "method": method,
                "global_ev": result.get("global_ev"),
                "mean_recovery_r2": result.get("mean_recovery_r2"),
                "mean_presence_auc": result.get("mean_presence_auc"),
                "error": result.get("error"),
            }
        )
    with (cfg.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "global_ev", "mean_recovery_r2", "mean_presence_auc", "error"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Goodfire Toy Compare",
        "",
        f"- gamfit: `{gamfit.__version__}`",
        f"- primitives: {', '.join(data.primitive_names)}",
        f"- train/eval: {cfg.n_train}/{cfg.n_eval}, D={cfg.d_ambient}, active_count={cfg.active_count}",
        "",
        "| method | global EV | mean contribution R2 | mean presence AUC | error |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(v: object) -> str:
            return "" if v is None else (f"{float(v):.4f}" if isinstance(v, (float, int, np.floating)) and np.isfinite(v) else str(v))
        lines.append(
            f"| {row['method']} | {fmt(row['global_ev'])} | {fmt(row['mean_recovery_r2'])} | "
            f"{fmt(row['mean_presence_auc'])} | {row['error'] or ''} |"
        )
    (cfg.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[wrote] {cfg.out_dir}", flush=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gamfit-child", choices=("manifold_sae_linear", "manifold_sae_mixed", "manifold_sae_patch"), default=None, help=argparse.SUPPRESS)
    p.add_argument("--data-npz", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-dir", type=Path, default=Config.out_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--n-factors", type=int, default=Config.n_factors)
    p.add_argument("--d-ambient", type=int, default=Config.d_ambient)
    p.add_argument("--n-train", type=int, default=Config.n_train)
    p.add_argument("--n-eval", type=int, default=Config.n_eval)
    p.add_argument("--active-count", type=int, default=Config.active_count)
    p.add_argument("--noise", type=float, default=Config.noise)
    p.add_argument("--bsf-steps", type=int, default=Config.bsf_steps)
    p.add_argument("--bsf-lr", type=float, default=Config.bsf_lr)
    p.add_argument("--bsf-batch-size", type=int, default=Config.bsf_batch_size)
    p.add_argument("--gamfit-iter", type=int, default=Config.gamfit_iter)
    p.add_argument("--gamfit-methods", type=lambda s: tuple(x for x in s.split(",") if x), default=Config.gamfit_methods)
    p.add_argument("--skip-bsf", action="store_true")
    p.add_argument("--skip-gamfit", action="store_true")
    ns = p.parse_args()
    if ns.gamfit_child is not None:
        return ns  # type: ignore[return-value]
    data = vars(ns)
    data.pop("gamfit_child")
    data.pop("data_npz")
    data.pop("child_out")
    return Config(**data)


def _child_data(path: Path) -> ToyData:
    z = np.load(path, allow_pickle=True)
    return ToyData(
        x_train=np.asarray(z["x_train"], dtype=np.float64),
        x_eval=np.asarray(z["x_eval"], dtype=np.float64),
        active_eval=np.asarray(z["active_eval"], dtype=bool),
        contrib_eval=np.asarray(z["contrib_eval"], dtype=np.float64),
        primitive_names=[str(x) for x in z["primitive_names"].tolist()],
        primitive_bases=[str(x) for x in z["primitive_bases"].tolist()],
        primitive_dims=[int(x) for x in z["primitive_dims"].tolist()],
        primitive_span_dims=([int(x) for x in z["primitive_span_dims"].tolist()]
                             if "primitive_span_dims" in z.files else [int(x) for x in z["primitive_dims"].tolist()]),
    )


def _run_gamfit_child(ns: argparse.Namespace) -> None:
    if ns.data_npz is None or ns.child_out is None:
        raise ValueError("gamfit child requires --data-npz and --child-out")
    cfg = Config(
        out_dir=ns.child_out.parent,
        seed=ns.seed,
        n_factors=ns.n_factors,
        d_ambient=ns.d_ambient,
        n_train=ns.n_train,
        n_eval=ns.n_eval,
        active_count=ns.active_count,
        noise=ns.noise,
        gamfit_iter=ns.gamfit_iter,
    )
    data = _child_data(ns.data_npz)
    try:
        model, log_text = _fit_gamfit(data, cfg, method=ns.gamfit_child)
        scored = score_gamfit_model(model, data)
        scored["fit_log_tail"] = "\n".join(log_text.splitlines()[-30:])
    except Exception as exc:  # noqa: BLE001 - write failure for parent
        scored = {"error": f"{type(exc).__name__}: {exc}"}
    ns.child_out.write_text(json.dumps(scored, indent=2))


def main() -> None:
    cfg = parse_args()
    if isinstance(cfg, argparse.Namespace):
        _run_gamfit_child(cfg)
        return
    torch.set_default_dtype(torch.float64)
    data = make_toy(cfg)
    results: dict[str, object] = {}
    if not cfg.skip_bsf:
        results.update(run_bsf(cfg, data))
    if not cfg.skip_gamfit:
        results.update(run_gamfit(cfg, data))
    write_outputs(cfg, data, results)


if __name__ == "__main__":
    main()
