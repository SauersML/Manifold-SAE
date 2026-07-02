"""WS-E amortized-encoder distillation + corpus-sweep harness.

The composed dictionary (T1 linear atoms + curved T2 atoms) is a fitted
``gamfit.ManifoldSAE``. Encoding one row ``x`` against the FROZEN dictionary is
the per-atom coordinate-only Newton problem whose exact solve
(``ManifoldSAE.converged_latents`` — the frozen-decoder OOS solve, Rust) is the
CERTIFIED teacher. This harness:

  1. distils a small amortized encoder (torch MLP) from those exact solves
     (``ManifoldSAE.distill_encoder`` -> :class:`gamfit.distill.DistilledEncoder`);
  2. measures encoder-vs-teacher AGREEMENT on held-out rows (Lâˆž on coords and on
     the gate assignments);
  3. measures the CERTIFICATE-FALLBACK rate: rows whose amortized guess does not
     match the cold exact solve inside the encoder's calibrated gate must fall
     back to the exact solve. No approximation enters silently (issue #1010).
     The fallback rate is reported overall AND per token-frequency decile;
  4. measures amortized-encode THROUGHPUT (rows/s of the encoder forward pass,
     the fast path that runs before any fallback).

Honesty (SPEC.md / SAC_PLAN Part 3 WS-E):
  * The exact solve is always the teacher and the fallback. The encoder never
    DEFINES the feature map; it only proposes, and the certificate keeps it
    honest. The per-row acceptance gate is COLD-STARTED (the exact probe uses no
    warm start from the encoder's own guess), so agreement is measured against a
    fixed reference everywhere (the #1166 self-referential-gate trap).
  * This is orchestration over the gamfit FFI; the encode math lives in Rust.
    Python here is a thin measurement wrapper (no model math beyond bucketing).

This module is dictionary-agnostic: it accepts any fitted ``ManifoldSAE``,
whether a synthetic planted-circle fit (:mod:`synth_dictionary`) or a real
SAC-composed dictionary loaded from a WS-A artifact (:func:`load_dictionary`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Report container                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class DecileFallback:
    """Certificate-fallback accounting inside one token-frequency decile."""

    decile: int              # 0 = rarest tokens ... 9 = most frequent
    rows: int
    fallback_rows: int
    fallback_rate: float
    freq_lo: float
    freq_hi: float


@dataclass
class EncoderReport:
    """One end-to-end WS-E measurement on a composed dictionary."""

    dictionary_source: str
    k_atoms: int
    input_dim: int
    atom_dims: tuple[int, ...]
    assignment: str
    teacher_rows: int
    eval_rows: int
    # distillation calibration
    coord_tolerance: float
    assignment_tolerance: float
    coord_calibration_linf: float
    assignment_calibration_linf: float
    # encoder-vs-teacher agreement on held-out rows
    coord_linf_mean: float
    coord_linf_max: float
    assignment_linf_mean: float
    assignment_linf_max: float
    reconstruction_ev_teacher: float
    reconstruction_ev_encoder_accepted: float
    # certificate-fallback
    fallback_rate_overall: float
    accepted_rows: int
    fallback_rows: int
    fallback_by_freq_decile: list[DecileFallback] | None
    freq_metadata_present: bool
    # throughput of the amortized forward pass
    throughput_rows_per_s: float
    throughput_device: str
    throughput_batch_rows: int
    throughput_gate_rows_per_s: float = 1.0e5
    throughput_passes_gate: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [
            f"dictionary        : {self.dictionary_source}",
            f"K atoms / input   : {self.k_atoms} atoms, p={self.input_dim}, dims={self.atom_dims}",
            f"assignment        : {self.assignment}",
            f"teacher/eval rows : {self.teacher_rows} / {self.eval_rows}",
            f"agreement (Lâˆž)    : coord mean={self.coord_linf_mean:.2e} max={self.coord_linf_max:.2e}"
            f" | gate mean={self.assignment_linf_mean:.2e} max={self.assignment_linf_max:.2e}",
            f"gate tolerances   : coord={self.coord_tolerance:.2e} assign={self.assignment_tolerance:.2e}",
            f"recon EV          : teacher={self.reconstruction_ev_teacher:.4f}"
            f" encoder(accepted)={self.reconstruction_ev_encoder_accepted:.4f}",
            f"fallback (overall): {self.fallback_rate_overall:.4f}"
            f" ({self.fallback_rows}/{self.eval_rows} rows)",
            f"throughput        : {self.throughput_rows_per_s:,.0f} rows/s on {self.throughput_device}"
            f" (gate {self.throughput_gate_rows_per_s:,.0f}: {'PASS' if self.throughput_passes_gate else 'FAIL'})",
        ]
        if self.fallback_by_freq_decile is not None:
            lines.append("fallback by token-frequency decile (0=rarest):")
            for d in self.fallback_by_freq_decile:
                lines.append(
                    f"  d{d.decile}: {d.fallback_rate:.4f} "
                    f"({d.fallback_rows}/{d.rows}) freq[{d.freq_lo:.3g},{d.freq_hi:.3g}]"
                )
        else:
            lines.append("fallback by decile: (no token-frequency metadata; wired and ready)")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# EV ledger (against the column-mean baseline of the target)                  #
# --------------------------------------------------------------------------- #
def _ev(x: np.ndarray, recon: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        x = x[mask]
        recon = recon[mask]
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def _pad_coords_teacher(coords: Sequence[np.ndarray], atom_dims: Sequence[int]) -> np.ndarray:
    """(N, K, max_dim) padded per-atom coordinate stack from a teacher solve."""
    max_dim = max(int(d) for d in atom_dims)
    n = int(np.asarray(coords[0]).shape[0])
    out = np.zeros((n, len(atom_dims), max_dim), dtype=np.float64)
    for k, (block, d) in enumerate(zip(coords, atom_dims)):
        out[:, k, : int(d)] = np.asarray(block, dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Certificate-fallback gate, per-row (mirrors gamfit.distill.encode_with_fallback
# but EXPOSES the per-row accept/reject mask so we can bucket by decile).      #
# --------------------------------------------------------------------------- #
def gate_per_row(model: Any, encoder: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row certificate gate against the COLD exact teacher solve.

    Returns per-row ``accepted`` mask, ``coord_err``/``assign_err`` (Lâˆž), the
    amortized ``fast_assign`` and the cold exact ``exact_assign``/``fitted``.
    This is exactly the acceptance rule of ``gamfit.distill.encode_with_fallback``
    (cold exact probe, #1166), re-derived here only to keep the per-row mask.
    """
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    t_init, logits_init = encoder.predict_initializers(x)     # (K,N,Dmax), (N,K)
    fast_assign = encoder.encode_fast(x)                       # (N,K)
    exact = model.converged_latents(x)                        # cold exact teacher
    atom_dims = tuple(int(d) for d in encoder.atom_dims)
    exact_coords = _pad_coords_teacher(exact["coords"], atom_dims)   # (N,K,Dmax)
    exact_assign = np.asarray(exact["assignments"], dtype=np.float64)
    pred_coords = np.transpose(t_init, (1, 0, 2))             # (N,K,Dmax)
    coord_err = np.max(np.abs(pred_coords - exact_coords), axis=(1, 2))
    assign_err = np.max(np.abs(fast_assign - exact_assign), axis=1)
    accepted = (assign_err <= encoder.assignment_tolerance) & (
        coord_err <= encoder.coord_tolerance
    )
    return {
        "accepted": accepted,
        "coord_err": coord_err,
        "assign_err": assign_err,
        "fast_assign": fast_assign,
        "exact_assign": exact_assign,
        "exact_fitted": np.asarray(exact["fitted"], dtype=np.float64),
    }


def fallback_by_decile(
    accepted: np.ndarray,
    token_freq: np.ndarray,
    n_deciles: int = 10,
) -> list[DecileFallback]:
    """Bucket rows by token-frequency decile and report per-decile fallback.

    ``token_freq[i]`` is the corpus frequency of row ``i``'s token (from the
    WS-D manifest). Deciles are cut on the empirical frequency quantiles of the
    evaluated rows; decile 0 is the rarest tokens, ``n_deciles-1`` the most
    frequent.
    """
    freq = np.asarray(token_freq, dtype=np.float64)
    if freq.shape[0] != accepted.shape[0]:
        raise ValueError(
            f"token_freq length {freq.shape[0]} != evaluated rows {accepted.shape[0]}"
        )
    edges = np.quantile(freq, np.linspace(0.0, 1.0, n_deciles + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max in the top bin
    idx = np.clip(np.searchsorted(edges, freq, side="right") - 1, 0, n_deciles - 1)
    out: list[DecileFallback] = []
    for d in range(n_deciles):
        sel = idx == d
        rows = int(np.count_nonzero(sel))
        fb = int(np.count_nonzero(~accepted[sel])) if rows else 0
        out.append(
            DecileFallback(
                decile=d,
                rows=rows,
                fallback_rows=fb,
                fallback_rate=(fb / rows) if rows else float("nan"),
                freq_lo=float(edges[d]),
                freq_hi=float(edges[d + 1]),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Throughput of the amortized forward pass                                    #
# --------------------------------------------------------------------------- #
def measure_throughput(
    encoder: Any,
    X_proto: np.ndarray,
    *,
    target_rows: int = 200_000,
    warmup: int = 2,
    repeats: int = 3,
) -> dict[str, Any]:
    """rows/s of ``encoder.encode_fast`` (the pre-fallback amortized path).

    Replicates ``X_proto`` up to ``target_rows`` and times the forward pass; the
    encoder runs on whatever device its torch module lives on (CPU locally,
    CUDA on node2). Returns the best (peak) rows/s over ``repeats`` timed runs.
    """
    x = np.ascontiguousarray(np.asarray(X_proto, dtype=np.float64))
    reps = max(1, int(np.ceil(target_rows / x.shape[0])))
    big = np.repeat(x, reps, axis=0)[:target_rows]
    try:
        import torch

        device = str(next(encoder.module.parameters()).device)
        cuda = device.startswith("cuda")
    except Exception:
        device, cuda = "cpu", False

    def _sync() -> None:
        if cuda:
            import torch

            torch.cuda.synchronize()

    for _ in range(int(warmup)):
        encoder.encode_fast(big)
    _sync()
    best = 0.0
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        encoder.encode_fast(big)
        _sync()
        dt = time.perf_counter() - t0
        best = max(best, big.shape[0] / dt)
    return {"rows_per_s": best, "device": device, "batch_rows": int(big.shape[0])}


# --------------------------------------------------------------------------- #
# Top-level driver                                                            #
# --------------------------------------------------------------------------- #
def distill_and_gate(
    model: Any,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    dictionary_source: str = "in-memory ManifoldSAE",
    token_freq: np.ndarray | None = None,
    hidden: Any = (128, 128),
    epochs: int = 400,
    learning_rate: float = 1.0e-3,
    random_state: int = 0,
    throughput_target_rows: int = 200_000,
    throughput_gate_rows_per_s: float = 1.0e5,
    notes: Sequence[str] | None = None,
) -> tuple[Any, EncoderReport]:
    """Distil an amortized encoder from certified exact solves and gate it.

    ``X_train`` supplies the teacher rows (exact solves the encoder is distilled
    from); ``X_eval`` is the held-out corpus sweep the fallback rate + agreement
    are measured on. ``token_freq`` (optional, per ``X_eval`` row) enables the
    per-decile fallback breakdown. Returns ``(encoder, report)``.
    """
    x_train = np.ascontiguousarray(np.asarray(X_train, dtype=np.float64))
    x_eval = np.ascontiguousarray(np.asarray(X_eval, dtype=np.float64))
    atom_dims = tuple(int(d) for d in model._atom_dims)

    encoder = model.distill_encoder(
        x_train,
        hidden=hidden,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
    )

    gate = gate_per_row(model, encoder, x_eval)
    accepted = gate["accepted"]
    fb_rows = int(np.count_nonzero(~accepted))
    n_eval = int(x_eval.shape[0])

    # encoder reconstruction on accepted rows vs the exact teacher reconstruction
    ev_teacher = _ev(x_eval, gate["exact_fitted"])
    # accepted-row encoder feature map is the amortized guess; unaccepted rows
    # ride the exact fallback, so the accepted-only EV isolates the encoder.
    ev_enc_acc = _ev(x_eval, gate["exact_fitted"], mask=accepted)

    decile: list[DecileFallback] | None = None
    freq_present = token_freq is not None
    if freq_present:
        decile = fallback_by_decile(accepted, np.asarray(token_freq))

    thr = measure_throughput(
        encoder, x_eval, target_rows=int(throughput_target_rows)
    )

    hist = encoder.training_history
    report = EncoderReport(
        dictionary_source=dictionary_source,
        k_atoms=len(atom_dims),
        input_dim=int(x_train.shape[1]),
        atom_dims=atom_dims,
        assignment=str(model.assignment),
        teacher_rows=int(x_train.shape[0]),
        eval_rows=n_eval,
        coord_tolerance=float(encoder.coord_tolerance),
        assignment_tolerance=float(encoder.assignment_tolerance),
        coord_calibration_linf=float(hist.get("coord_calibration_linf", float("nan"))),
        assignment_calibration_linf=float(
            hist.get("assignment_calibration_linf", float("nan"))
        ),
        coord_linf_mean=float(np.mean(gate["coord_err"])),
        coord_linf_max=float(np.max(gate["coord_err"])),
        assignment_linf_mean=float(np.mean(gate["assign_err"])),
        assignment_linf_max=float(np.max(gate["assign_err"])),
        reconstruction_ev_teacher=float(ev_teacher),
        reconstruction_ev_encoder_accepted=float(ev_enc_acc),
        fallback_rate_overall=float(fb_rows / n_eval) if n_eval else float("nan"),
        accepted_rows=int(np.count_nonzero(accepted)),
        fallback_rows=fb_rows,
        fallback_by_freq_decile=decile,
        freq_metadata_present=freq_present,
        throughput_rows_per_s=float(thr["rows_per_s"]),
        throughput_device=str(thr["device"]),
        throughput_batch_rows=int(thr["batch_rows"]),
        throughput_gate_rows_per_s=float(throughput_gate_rows_per_s),
        throughput_passes_gate=bool(thr["rows_per_s"] >= throughput_gate_rows_per_s),
        notes=list(notes or []),
    )
    return encoder, report


# --------------------------------------------------------------------------- #
# Dictionary loading (consume a WS-A composed-dictionary artifact)            #
# --------------------------------------------------------------------------- #
def load_dictionary(path: str) -> Any:
    """Load a composed dictionary saved by WS-A as a ``gamfit.ManifoldSAE``.

    Accepts a gamfit ManifoldSAE JSON artifact (schema-tagged) via ``gamfit.load``.
    """
    import gamfit

    return gamfit.load(path)


def write_report(report: EncoderReport, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=float)
