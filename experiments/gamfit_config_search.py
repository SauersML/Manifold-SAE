"""Search for an end-to-end working gamfit SAE-manifold configuration.

Each candidate runs in a child process because some gamfit failures terminate
the interpreter. The parent records whether the model can fit, reconstruct,
encode, and produce per-factor contribution scores on the AMM toy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.goodfire_toy_compare import Config as ToyConfig  # noqa: E402
from experiments.goodfire_toy_compare import make_toy, score_gamfit_model  # noqa: E402


@dataclass(frozen=True)
class SearchConfig:
    out_dir: Path = ROOT / "runs" / "GAMFIT_CONFIG_SEARCH"
    n_factors: int = 6
    d_ambient: int = 32
    n_train: int = 500
    n_eval: int = 200
    active_count: int = 3
    seed: int = 0
    timeout_s: int = 90


def candidates(active_count: int) -> list[dict[str, Any]]:
    base = {
        "K": 6,
        "assignment": "ibp_map",
        "n_iter": 3,
        "random_state": 0,
    }
    return [
        {
            "name": "circle_iter1",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
            "n_iter": 1,
        },
        {
            "name": "circle_minimal",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
        },
        {
            "name": "circle_topk",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
            "top_k": active_count,
        },
        {
            "name": "circle_low_regularization",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
            "top_k": active_count,
            "sparsity_weight": 0.01,
            "smoothness_weight": 0.01,
            "isometry_weight": 0.1,
            "decoder_incoherence_weight": 0.1,
            "nuclear_norm_weight": 0.0,
            "alpha": "auto",
        },
        {
            "name": "linear_minimal",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
        },
        {
            "name": "linear_iter1",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
            "n_iter": 1,
        },
        {
            "name": "linear_iter10",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
            "n_iter": 10,
        },
        {
            "name": "linear_d2_minimal",
            **base,
            "d_atom": 2,
            "atom_topology": "linear",
        },
        {
            "name": "linear_topk",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
            "top_k": active_count,
        },
        {
            "name": "linear_softmax",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
            "assignment": "softmax",
        },
        {
            "name": "linear_threshold_gate",
            **base,
            "d_atom": 1,
            "atom_topology": "linear",
            "assignment": "threshold_gate",
            "jumprelu_threshold": 0.0,
        },
        {
            "name": "euclidean_d1_minimal",
            **base,
            "d_atom": 1,
            "atom_topology": "euclidean",
        },
        {
            "name": "euclidean_d2_minimal",
            **base,
            "d_atom": 2,
            "atom_topology": "euclidean",
        },
        {
            "name": "circle_softmax",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
            "assignment": "softmax",
        },
        {
            "name": "circle_threshold_gate",
            **base,
            "d_atom": 1,
            "atom_topology": "circle",
            "assignment": "threshold_gate",
            "jumprelu_threshold": 0.0,
        },
        {
            "name": "mixed_basis_minimal",
            **base,
            "d_atom": [2, 2, 2, 2, 2, 1],
            "atom_basis": ["torus", "sphere", "euclidean", "euclidean", "torus", "periodic"],
        },
    ]


def toy_cfg(cfg: SearchConfig) -> ToyConfig:
    return ToyConfig(
        out_dir=cfg.out_dir,
        seed=cfg.seed,
        n_factors=cfg.n_factors,
        d_ambient=cfg.d_ambient,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        active_count=cfg.active_count,
        noise=0.0,
        skip_bsf=True,
        skip_gamfit=False,
    )


def run_child(data_path: Path, out_path: Path, kwargs: dict[str, Any]) -> None:
    import gamfit

    z = np.load(data_path, allow_pickle=True)
    data = make_toy(toy_cfg(SearchConfig(
        out_dir=out_path.parent,
        seed=int(z["seed"]),
        n_factors=int(z["n_factors"]),
        d_ambient=int(z["d_ambient"]),
        n_train=int(z["n_train"]),
        n_eval=int(z["n_eval"]),
        active_count=int(z["active_count"]),
    )))
    fit_kwargs = dict(kwargs)
    name = str(fit_kwargs.pop("name"))
    t0 = time.time()
    try:
        model = gamfit.sae_manifold_fit(data.x_train, **fit_kwargs)
        scored = score_gamfit_model(model, data)
        scored.update({
            "ok": True,
            "name": name,
            "seconds": round(time.time() - t0, 3),
            "gamfit_version": gamfit.__version__,
            "kwargs": fit_kwargs,
        })
    except BaseException as exc:  # noqa: BLE001 - this is a failure recorder
        scored = {
            "ok": False,
            "name": name,
            "seconds": round(time.time() - t0, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gamfit_version": getattr(gamfit, "__version__", "unknown"),
            "kwargs": fit_kwargs,
        }
    out_path.write_text(json.dumps(scored, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", action="store_true")
    p.add_argument("--data-path", type=Path)
    p.add_argument("--out-path", type=Path)
    p.add_argument("--kwargs-json", type=str)
    p.add_argument("--out-dir", type=Path, default=SearchConfig.out_dir)
    p.add_argument("--n-train", type=int, default=SearchConfig.n_train)
    p.add_argument("--n-eval", type=int, default=SearchConfig.n_eval)
    p.add_argument("--timeout-s", type=int, default=SearchConfig.timeout_s)
    return p.parse_args()


def main() -> None:
    ns = parse_args()
    if ns.child:
        if ns.data_path is None or ns.out_path is None or ns.kwargs_json is None:
            raise ValueError("child requires --data-path, --out-path, --kwargs-json")
        run_child(ns.data_path, ns.out_path, json.loads(ns.kwargs_json))
        return

    cfg = SearchConfig(out_dir=ns.out_dir, n_train=ns.n_train, n_eval=ns.n_eval, timeout_s=ns.timeout_s)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = cfg.out_dir / "search_data_meta.npz"
    np.savez(
        data_path,
        seed=cfg.seed,
        n_factors=cfg.n_factors,
        d_ambient=cfg.d_ambient,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        active_count=cfg.active_count,
    )
    rows = []
    for kwargs in candidates(cfg.active_count):
        name = kwargs["name"]
        out_path = cfg.out_dir / f"{name}.json"
        log_path = cfg.out_dir / f"{name}.log"
        out_path.unlink(missing_ok=True)
        print(f"[try] {name}", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--data-path",
            str(data_path),
            "--out-path",
            str(out_path),
            "--kwargs-json",
            json.dumps(kwargs),
        ]
        t0 = time.time()
        with log_path.open("w") as log:
            try:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=cfg.timeout_s, check=False)
                status = "exited"
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
                returncode = None
        if out_path.exists():
            row = json.loads(out_path.read_text())
        else:
            row = {
                "ok": False,
                "name": name,
                "error_type": status,
                "error": f"{status} returncode={returncode}",
            }
        row["returncode"] = returncode
        row["wall_s"] = round(time.time() - t0, 3)
        row["log_path"] = str(log_path)
        rows.append(row)
        print(json.dumps({
            "name": name,
            "ok": row.get("ok"),
            "global_ev": row.get("global_ev"),
            "mean_recovery_r2": row.get("mean_recovery_r2"),
            "error_type": row.get("error_type"),
            "returncode": returncode,
        }), flush=True)
        (cfg.out_dir / "summary.json").write_text(json.dumps({
            "config": {**asdict(cfg), "out_dir": str(cfg.out_dir)},
            "rows": rows,
        }, indent=2))


if __name__ == "__main__":
    main()
