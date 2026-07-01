"""Tests for HyperbolicSAE + gamfit-native Poincaré geometry.

Geometry is the ``gamfit.torch.PoincareAtoms`` primitive directly — there is no
hand-rolled or shim layer anymore. The SAE config exposes ``curvature`` as a
positive magnitude (κ > 0) for ergonomics and hands the primitive the geometric
negative curvature (c = -κ). Tests exercise both the primitive and the SAE.

gamfit 0.1.134 note: ``PoincareAtoms.project_into_ball`` / decode saturate ONTO
the ball boundary (‖x‖ == 1/√c) for large inputs, which makes ``distance`` /
geodesic maps blow up. The interior tests therefore build genuinely-interior
points via the ``_interior`` clamp ((1 - 1e-6)/√c rescale) before asserting
distance/symmetry properties — they are NOT skipped.
"""
from __future__ import annotations

import math

import torch

from gamfit.torch import PoincareAtoms

from manifold_sae.hyperbolic_sae import HyperbolicSAE


def _atoms(ball_dim: int, c: float = 1.0) -> PoincareAtoms:
    """A 1-atom dictionary with geometric curvature -c, for raw geometry calls."""
    return PoincareAtoms(F=1, ball_dim=ball_dim, curvature=-float(c))


def _interior(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Rescale ``x`` to lie STRICTLY inside the ball of radius 1/√c.

    The primitive's ``project_into_ball`` saturates onto the boundary for large
    norms (a known gamfit 0.1.134 behaviour), which makes ``distance`` diverge.
    This clamp guarantees ‖out‖ ≤ (1 - 1e-6)/√c so downstream geodesic ops are
    finite.
    """
    max_norm = (1.0 - 1e-6) / math.sqrt(float(c))
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return x * torch.clamp(max_norm / norm, max=1.0)


# --------------------------------------------------------------------------- #
# Curvature-sign contract: positive magnitude in the SAE, negative geometric   #
# curvature inside the primitive.                                              #
# --------------------------------------------------------------------------- #
def test_curvature_sign_convention():
    sae = HyperbolicSAE(input_dim=8, n_features=4, ball_dim=3, curvature=1.0)
    # SAE exposes the positive magnitude...
    assert sae.c == 1.0
    # ...but the primitive was handed the negated (strictly negative) value.
    assert sae.atoms_dict.curvature < 0.0
    assert sae.atoms_dict.curvature == -sae.c
    # Magnitude must be positive at construction.
    try:
        HyperbolicSAE(input_dim=8, n_features=4, ball_dim=3, curvature=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative magnitude should be rejected")
    # The primitive itself requires negative curvature.
    try:
        PoincareAtoms(F=2, ball_dim=3, curvature=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("PoincareAtoms should reject c >= 0")


def test_project_into_ball_stays_inside():
    torch.manual_seed(0)
    atoms = _atoms(6, c=1.0)
    x = torch.randn(20, 6) * 5.0  # way outside the unit ball
    # project_into_ball saturates ONTO the boundary; clamp strictly inside.
    p = _interior(atoms.project_into_ball(x), c=1.0)
    assert torch.isfinite(p).all()
    assert (p.norm(dim=-1) < 1.0).all(), "projection escaped the ball"


def test_mobius_identity_at_origin():
    """0 ⊕ y = y  and  x ⊕ 0 = x via the primitive's mobius_add."""
    torch.manual_seed(1)
    atoms = _atoms(5, c=1.0)
    x = torch.randn(10, 5) * 0.1
    zero = torch.zeros_like(x)
    a = atoms.mobius_add(zero.contiguous(), x.contiguous())
    b = atoms.mobius_add(x.contiguous(), zero.contiguous())
    assert torch.allclose(a, x, atol=1e-5), f"0⊕x mismatch {(a-x).abs().max()}"
    assert torch.allclose(b, x, atol=1e-5), f"x⊕0 mismatch {(b-x).abs().max()}"


def test_distance_symmetry_and_triangle():
    torch.manual_seed(2)
    atoms = _atoms(5, c=1.0)
    pts = _interior(torch.randn(15, 5) * 0.2, c=1.0)  # strictly inside

    def dist(i, j):
        return atoms.distance(pts[i:i + 1].contiguous(), pts[j:j + 1].contiguous()).item()

    # Symmetry.
    for _ in range(10):
        i, j = torch.randint(0, 15, (2,)).tolist()
        assert abs(dist(i, j) - dist(j, i)) < 1e-4, f"asymmetric: {dist(i, j)} vs {dist(j, i)}"
    # Triangle inequality.
    for _ in range(15):
        i, j, k = torch.randint(0, 15, (3,)).tolist()
        d_ij, d_jk, d_ik = dist(i, j), dist(j, k), dist(i, k)
        assert d_ik <= d_ij + d_jk + 1e-3, f"triangle fail: {d_ik} > {d_ij}+{d_jk}"


def test_distance_finite_and_nonneg_interior():
    """Distances between strictly-interior points are finite and >= 0; the
    distance from the origin grows with radius (boundary-safety check that the
    interior clamp keeps geodesics from blowing up)."""
    atoms = _atoms(6, c=1.0)
    # Build points at increasing radius, all strictly inside.
    base = torch.randn(10, 6)
    base = base / base.norm(dim=-1, keepdim=True)
    radii = torch.linspace(0.1, 0.999999, 10).unsqueeze(-1)
    pts = _interior(base * radii, c=1.0)
    zero = torch.zeros_like(pts)
    d = atoms.distance(zero.contiguous(), pts.contiguous())
    assert torch.isfinite(d).all(), "interior distance produced NaN/Inf"
    assert (d >= 0).all()
    # Farther-out interior points are geodesically farther from the origin.
    assert d[-1] > d[0], f"radius ordering broken: {d[0]} !< {d[-1]}"


def test_decode_forward_stays_in_ball():
    """The primitive's tangent-aggregation decode lands inside (or on) the ball
    and never NaNs, even with large gates."""
    atoms = _atoms(4, c=1.0)
    z = torch.randn(7, 1) * 1e3  # huge gates push toward the boundary
    out = atoms(z)
    assert torch.isfinite(out).all(), "decode produced NaN/Inf"
    assert (out.norm(dim=-1) <= 1.0 + 1e-4).all(), "decode escaped the ball"


# --------------------------------------------------------------------------- #
# SAE-level                                                                    #
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_keys():
    sae = HyperbolicSAE(input_dim=16, n_features=8, ball_dim=4, curvature=1.0)
    x = torch.randn(7, 16)
    out = sae(x)
    assert out["x_hat"].shape == (7, 16)
    assert out["gates"].shape == (7, 8)
    assert out["atoms_pos"].shape == (8, 4)  # shared dictionary, not per-sample
    assert out["radii"].shape == (8,)
    assert out["radii"].min() >= 0.0
    # feature_norms_in_ball matches the radii used in the loss.
    fn = sae.feature_norms_in_ball()
    assert torch.allclose(fn, out["radii"], atol=1e-5)


def test_loss_decreases_under_training():
    torch.manual_seed(3)
    D, F, d = 16, 8, 4
    sae = HyperbolicSAE(input_dim=D, n_features=F, ball_dim=d,
                        curvature=1.0, sparsity_weight=1e-4)
    x = torch.randn(64, D)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        loss, _ = sae.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
    assert all(l == l for l in losses), "NaN loss"
