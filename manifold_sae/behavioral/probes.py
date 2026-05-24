"""Linear-probe class predicting behavior-binary labels from SAE-atom activations.

Inspired by Arditi 2024's refusal direction (a single linear direction over
the residual stream separates harmful vs. benign prompts) and by Sharma 2023
(sycophancy probes). We operate over SAE-atom activations rather than raw
residual stream, so the probe weights directly index *atoms* whose firing
patterns encode each behavior. Sparse, interpretable, and steerable.

Training: logistic regression (numpy / scikit-learn-style closed form via
PyTorch L-BFGS). Output: per-atom |weight| ranking → "behavior atoms".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import nn

BEHAVIOR_TARGETS = ("refusal", "sycophancy", "hedging", "deception")


@dataclass
class ProbeReport:
    target: str
    n_train: int
    n_val: int
    train_acc: float
    val_acc: float
    val_auc: float
    weights: np.ndarray            # (n_atoms,)
    bias: float
    top_atoms: list[tuple[int, float]] = field(default_factory=list)  # (atom_idx, signed_weight)


class BehavioralProbe(nn.Module):
    """Logistic regression on SAE-atom activations → P(behavior=1).

    Parameters
    ----------
    n_atoms : int
        Width of the SAE atom layer (e.g. F=128 or F=4096).
    target : str
        One of {"refusal", "sycophancy", "hedging", "deception"}. Only used
        as a label / for downstream report keys; the math is identical.
    l2 : float
        Ridge penalty on weights. Default 1e-3 — small because the input is
        already sparse (TopK SAE) so over-fitting risk is moderate.
    """

    def __init__(self, n_atoms: int, target: str = "refusal", l2: float = 1e-3) -> None:
        super().__init__()
        if target not in BEHAVIOR_TARGETS:
            # We don't hard-fail unknown targets — caller may want custom names.
            pass
        self.n_atoms = int(n_atoms)
        self.target = target
        self.l2 = float(l2)
        self.linear = nn.Linear(self.n_atoms, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.report: ProbeReport | None = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """atoms: (B, F) -> (B,) probability of behavior=1."""
        return torch.sigmoid(self.linear(atoms).squeeze(-1))

    def logit(self, atoms: torch.Tensor) -> torch.Tensor:
        return self.linear(atoms).squeeze(-1)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def fit(
        self,
        atoms: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        *,
        val_split: float = 0.25,
        epochs: int = 400,
        lr: float = 0.1,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> ProbeReport:
        """Fit logistic regression with L-BFGS + ridge.

        Returns a `ProbeReport` (also stashed on `self.report`).
        """
        X = torch.as_tensor(np.asarray(atoms), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=device).view(-1)
        assert X.shape[0] == y.shape[0], (X.shape, y.shape)
        assert X.shape[1] == self.n_atoms, (X.shape, self.n_atoms)
        N = X.shape[0]

        # Stratified split — keep both classes in val even on tiny N.
        rng = np.random.default_rng(seed)
        idx_pos = np.where(y.cpu().numpy() > 0.5)[0]
        idx_neg = np.where(y.cpu().numpy() <= 0.5)[0]
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        n_val_pos = max(1, int(round(val_split * len(idx_pos))))
        n_val_neg = max(1, int(round(val_split * len(idx_neg))))
        val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
        train_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)

        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[val_idx], y[val_idx]

        self.to(device)
        # L-BFGS converges in a handful of steps for logistic regression.
        opt = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=epochs,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            logits = self.logit(Xtr)
            loss = bce(logits, ytr) + self.l2 * (self.linear.weight ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            tr_pred = (self.logit(Xtr) > 0).float()
            va_logit = self.logit(Xva)
            va_pred = (va_logit > 0).float()
            train_acc = float((tr_pred == ytr).float().mean().item())
            val_acc = float((va_pred == yva).float().mean().item())
            val_auc = _roc_auc(yva.cpu().numpy(), va_logit.cpu().numpy())

        w = self.linear.weight.detach().cpu().numpy().reshape(-1)
        b = float(self.linear.bias.detach().cpu().item())
        order = np.argsort(-np.abs(w))
        top = [(int(i), float(w[i])) for i in order[:32]]
        report = ProbeReport(
            target=self.target,
            n_train=int(Xtr.shape[0]),
            n_val=int(Xva.shape[0]),
            train_acc=train_acc,
            val_acc=val_acc,
            val_auc=val_auc,
            weights=w,
            bias=b,
            top_atoms=top,
        )
        self.report = report
        return report

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def top_k_atoms(self, k: int = 10, signed: bool = True) -> list[tuple[int, float]]:
        if self.report is None:
            raise RuntimeError("Probe not fitted yet — call .fit() first.")
        w = self.report.weights
        key = w if signed else np.abs(w)
        order = np.argsort(-np.abs(w))[:k]
        return [(int(i), float(w[i])) for i in order]


# ----------------------------------------------------------------------
# stand-alone helpers
# ----------------------------------------------------------------------

def top_atoms_for(probe: BehavioralProbe, k: int = 10) -> list[tuple[int, float]]:
    """Return top-k atoms by |weight| from a fitted probe."""
    return probe.top_k_atoms(k=k)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tiny ROC-AUC (Mann-Whitney U) — avoids a sklearn dep."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # vectorised: for each pos, count negs with lower score (+ 0.5 ties).
    rank_pos = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(rank_pos / (len(pos) * len(neg)))


def cross_correlation(
    probes: dict[str, BehavioralProbe],
    *,
    top_k: int = 20,
) -> dict[str, dict[str, float]]:
    """Pairwise overlap of top-k atoms across behaviors.

    Returns Jaccard(|top_k_atoms_i|, |top_k_atoms_j|) for each pair. A value
    near 0 means behavior subspaces are roughly orthogonal in the SAE
    dictionary; values near 1 mean atoms are shared.
    """
    keys = list(probes.keys())
    out: dict[str, dict[str, float]] = {a: {} for a in keys}
    sets = {a: {int(i) for i, _ in probes[a].top_k_atoms(k=top_k)} for a in keys}
    for a in keys:
        for b in keys:
            A, B = sets[a], sets[b]
            inter = len(A & B)
            union = len(A | B) or 1
            out[a][b] = inter / union
    return out
