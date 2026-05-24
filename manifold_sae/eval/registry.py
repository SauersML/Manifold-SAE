"""Loader registry. Add a new model variant in one line by appending to
``LOADERS`` (a dict from filename glob -> loader function).

Loaders return an ``SAEWrapper`` (see ``harness.py``).
"""
from __future__ import annotations

import fnmatch
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .harness import SAEWrapper


# ---------------------------------------------------------------------------
# Lazy import of the SAE classes defined in scripts/train_sae_comparison.py
# (those classes are the ones the three checkpoint files were saved from).
# ---------------------------------------------------------------------------


def _import_comparison_module():
    """Load scripts/train_sae_comparison.py *just enough* to access classes,
    without executing its training side effects.
    """
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "train_sae_comparison.py"
    # That script has top-level training side effects. Re-implement the small
    # set of classes locally to keep this loader hermetic.
    raise NotImplementedError("Use the local class definitions below.")


# ---------------------------------------------------------------------------
# Local re-declarations of the three saved architectures. Keeping these
# in-harness avoids running the training script's globals at import time.
# ---------------------------------------------------------------------------


class _TopKSAE(torch.nn.Module):
    def __init__(self, d_in, n_feat, top_k):
        super().__init__()
        self.W_e = torch.nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        self.b_e = torch.nn.Parameter(torch.zeros(n_feat))
        self.W_d = torch.nn.Parameter(torch.randn(n_feat, d_in) * (1.0 / np.sqrt(n_feat)))
        self.b_d = torch.nn.Parameter(torch.zeros(d_in))
        self.top_k = top_k

    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        topv, topi = z.topk(self.top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topi, F.relu(topv))
        return z_sparse

    def decode(self, z):
        return z @ self.W_d + self.b_d


class _L1SAE(torch.nn.Module):
    def __init__(self, d_in, n_feat):
        super().__init__()
        self.W_e = torch.nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        self.b_e = torch.nn.Parameter(torch.zeros(n_feat))
        self.W_d = torch.nn.Parameter(torch.randn(n_feat, d_in) * (1.0 / np.sqrt(n_feat)))
        self.b_d = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        z = (x - self.b_d) @ self.W_e + self.b_e
        return F.relu(z)

    def decode(self, z):
        return z @ self.W_d + self.b_d


class _ManifoldFourierSAE(torch.nn.Module):
    """Matches the ManifoldSAE class saved in runs/sae_comparison/model_manifold.pt."""

    def __init__(self, d_in, n_feat, M_F=3):
        super().__init__()
        self.n_feat = n_feat
        self.M_F = M_F
        self.W_gate = torch.nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        self.b_gate = torch.nn.Parameter(torch.full((n_feat,), -2.0))
        self.W_theta = torch.nn.Parameter(torch.randn(d_in, n_feat * 2) * (1.0 / np.sqrt(d_in)))
        self.W_amp = torch.nn.Parameter(torch.randn(d_in, n_feat) * (1.0 / np.sqrt(d_in)))
        basis_dim = 2 * M_F + 1
        self.D_k = torch.nn.Parameter(torch.randn(n_feat, basis_dim, d_in) * (0.1 / np.sqrt(basis_dim)))
        self.b_d = torch.nn.Parameter(torch.zeros(d_in))
        self.log_ard = torch.nn.Parameter(torch.zeros(n_feat))

    def _theta(self, x):
        xc = x - self.b_d
        tp = xc @ self.W_theta
        tp = tp.view(x.shape[0], self.n_feat, 2)
        tp = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tp

    def _fourier(self, cs):
        c, s = cs[..., 0], cs[..., 1]
        feats = [torch.ones_like(c), c, s]
        ck, sk = c.clone(), s.clone()
        for _ in range(2, self.M_F + 1):
            ck_new = ck * c - sk * s
            sk_new = sk * c + ck * s
            ck, sk = ck_new, sk_new
            feats += [ck, sk]
        return torch.stack(feats, dim=-1)

    def encode_activation(self, x):
        xc = x - self.b_d
        gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
        amp = F.softplus(xc @ self.W_amp) * torch.exp(self.log_ard)
        return gate * amp

    def reconstruct(self, x):
        xc = x - self.b_d
        gate = torch.sigmoid(xc @ self.W_gate + self.b_gate)
        amp = F.softplus(xc @ self.W_amp) * torch.exp(self.log_ard)
        cs = self._theta(x)
        phi = self._fourier(cs)
        w = (gate * amp).unsqueeze(-1)
        w_phi = (w * phi).reshape(x.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        return w_phi @ D_flat + self.b_d

    def decode_from_acts(self, z, theta_basis):
        # z is the gate*amp scalar per atom (B, F). theta_basis is the
        # ``phi`` tensor (B, F, P) frozen from the input. We allow steering
        # by modifying z but keep the per-row theta_basis fixed.
        w = z.unsqueeze(-1)  # (B, F, 1)
        w_phi = (w * theta_basis).reshape(z.shape[0], -1)
        D_flat = self.D_k.reshape(-1, self.D_k.shape[-1])
        return w_phi @ D_flat + self.b_d


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class TopKWrapper(SAEWrapper):
    def __init__(self, model: _TopKSAE, name: str):
        self.model = model.eval()
        self.name = name
        self.n_features = model.W_e.shape[1]
        self.input_dim = model.W_e.shape[0]
        self.firing_threshold = 1e-3

    def encode(self, x):
        return self.model.encode(x)

    def decode_from_activations(self, z):
        return self.model.decode(z)

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))


class L1Wrapper(SAEWrapper):
    def __init__(self, model: _L1SAE, name: str):
        self.model = model.eval()
        self.name = name
        self.n_features = model.W_e.shape[1]
        self.input_dim = model.W_e.shape[0]
        self.firing_threshold = 1e-3

    def encode(self, x):
        return self.model.encode(x)

    def decode_from_activations(self, z):
        return self.model.decode(z)

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))


class ManifoldFourierWrapper(SAEWrapper):
    def __init__(self, model: _ManifoldFourierSAE, name: str):
        self.model = model.eval()
        self.name = name
        self.n_features = model.n_feat
        self.input_dim = model.b_d.shape[0]
        self.firing_threshold = 1e-3
        self._cached_theta_basis = None
        self._cached_x_id = None

    def _theta_basis(self, x):
        cs = self.model._theta(x)
        return self.model._fourier(cs)

    def encode(self, x):
        # cache theta basis keyed by tensor id so decode_from_activations can
        # reuse it for the same x. Steering / ablation pass the same z so we
        # also keep the LAST encoded basis.
        with torch.no_grad():
            z = self.model.encode_activation(x)
            self._cached_theta_basis = self._theta_basis(x).detach()
            self._cached_z_shape = z.shape
        return z

    def decode_from_activations(self, z):
        # Use the most recently cached theta basis (rows must match).
        if self._cached_theta_basis is None or self._cached_theta_basis.shape[0] != z.shape[0]:
            raise RuntimeError(
                "ManifoldFourierWrapper.decode_from_activations requires a matching"
                " preceding encode(x) call so theta_basis is cached."
            )
        return self.model.decode_from_acts(z, self._cached_theta_basis)

    def reconstruct(self, x):
        return self.model.reconstruct(x)

    # ------------------------------------------------------------------
    # Anchor-swap aware decode.
    #
    # When the caller swaps the SOURCE row's amplitudes (z block) into a
    # TARGET row, the theta cached for those atoms must ALSO come from
    # the source row -- otherwise the swapped amplitudes are read off
    # against the TARGET's theta basis and the intended hue transplant
    # doesn't happen. This is the bug discovered by the steering-bench
    # agent (anchor-swap R^2 ~ 0.025 vs TopK's 0.285).
    # ------------------------------------------------------------------
    def swap_theta_from(
        self,
        z_target: torch.Tensor,
        x_source: torch.Tensor,
        source_row_id: int,
        atom_mask,
    ) -> torch.Tensor:
        """Decode `z_target` after replacing per-atom theta_k with the
        theta_k of `x_source[source_row_id]` wherever ``atom_mask[k]``
        is True.

        Parameters
        ----------
        z_target      : (B, F) target activations (already containing the
                        amp-block swap if any).
        x_source      : (M, D) input rows from which to pull the donor
                        theta. Often the full validation matrix.
        source_row_id : index into ``x_source`` of the donor row.
        atom_mask     : bool/byte tensor or np.ndarray of shape (F,)
                        marking which atoms get the donor theta.
        """
        with torch.no_grad():
            # Build target theta basis -- prefer the cached one if it
            # already matches z_target's batch shape.
            if (
                self._cached_theta_basis is not None
                and self._cached_theta_basis.shape[0] == z_target.shape[0]
            ):
                phi_target = self._cached_theta_basis.clone()
            else:
                # Caller did not pre-encode the targets; we cannot build
                # the target theta from scratch without their X. Fall
                # back to broadcasting the source row's theta across the
                # whole batch (degenerate but safe).
                phi_src_full = self._theta_basis(x_source)
                phi_target = phi_src_full[source_row_id : source_row_id + 1].expand(
                    z_target.shape[0], -1, -1
                ).clone()

            # Compute donor theta basis from x_source.
            phi_source = self._theta_basis(x_source)
            phi_donor = phi_source[source_row_id]  # (F, P)

            # Resolve atom_mask to a torch bool tensor on the right device.
            if isinstance(atom_mask, np.ndarray):
                mask = torch.from_numpy(atom_mask.astype(np.bool_))
            else:
                mask = atom_mask.to(torch.bool)
            mask = mask.to(phi_target.device)

            # Overwrite the masked atom slots with donor theta.
            phi_target[:, mask, :] = phi_donor[mask, :].unsqueeze(0)

            return self.model.decode_from_acts(z_target, phi_target)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_topk(path: str, *, d_in: int, n_feat: int = 512, top_k: int = 32, device: str = "cpu") -> SAEWrapper:
    m = _TopKSAE(d_in, n_feat, top_k=top_k)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    return TopKWrapper(m.to(device), name=Path(path).stem)


def load_l1(path: str, *, d_in: int, n_feat: int = 512, device: str = "cpu") -> SAEWrapper:
    m = _L1SAE(d_in, n_feat)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    return L1Wrapper(m.to(device), name=Path(path).stem)


def load_manifold(path: str, *, d_in: int, n_feat: int = 512, M_F: int = 3, device: str = "cpu") -> SAEWrapper:
    m = _ManifoldFourierSAE(d_in, n_feat, M_F=M_F)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    return ManifoldFourierWrapper(m.to(device), name=Path(path).stem)


# Glob -> loader. Add new variants here in ONE line.
LOADERS = {
    "*model_topk*.pt": load_topk,
    "*model_l1*.pt": load_l1,
    "*model_manifold*.pt": load_manifold,
    # "*model_matryoshka*.pt": load_matryoshka,   # add when ready
    # "*model_equivariant*.pt": load_equivariant,
    # "*model_skip*.pt": load_skip_transcoder,
}


def loader_for(path: str):
    name = os.path.basename(path)
    for pat, fn in LOADERS.items():
        if fnmatch.fnmatch(name, pat):
            return fn
    raise KeyError(f"No loader registered for {name}. Add it to LOADERS in registry.py.")
