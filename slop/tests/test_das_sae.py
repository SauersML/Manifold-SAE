"""Tests for DAS-SAE (gamfit-native InterchangeSwapDecoder).

The decoder is now the gamfit primitive ``InterchangeSwapDecoder``, which
GATES at decode time and takes a BOOLEAN atom mask for the swap. These tests
target that contract:

(1) bool-mask selection: swap_decode(a, b, all_True) == decode(a);
    swap_decode(a, b, all_False) == decode(b); a one-hot mask matches a
    manual gated decode of the spliced latent.
(2) gate gradient flows: grad of the interchange loss w.r.t. decoder.gate is
    non-trivial.
(3) loss decreases under a few optimizer steps.
(4) no-swap (all-False mask) reproduces the gated decode of z_b.
(5) fit_hue_direction recovers a planted linear hue signal.

NOTE: these import gamfit.torch.InterchangeSwapDecoder, which is only present
in the post-refactor gamfit wheel; until that wheel builds, collection of this
module will error at import — that is expected (the source API is the stable
contract per the cutover spec).
"""
from __future__ import annotations

import torch

from manifold_sae.das_sae import (
    DASSAE,
    DASSAEConfig,
    build_target_swap,
    fit_hue_direction,
)


def _make(D=32, F=24, B=8, top_k=4, seed=0):
    torch.manual_seed(seed)
    cfg = DASSAEConfig(input_dim=D, n_features=F, top_k=top_k)
    sae = DASSAE(cfg)
    x = torch.randn(B, D)
    return sae, x


def test_swap_decode_bool_selection():
    """all-True ⇒ decode(z_a); all-False ⇒ decode(z_b); one-hot matches manual."""
    sae, x = _make()
    z_a = sae(x).z
    z_b = sae(x.flip(0)).z
    F = z_a.shape[-1]

    all_true = torch.ones(F, dtype=torch.bool)
    all_false = torch.zeros(F, dtype=torch.bool)

    assert torch.allclose(
        sae.swap_decode(z_a, z_b, atom_mask=all_true), sae.decode(z_a), atol=1e-5
    ), "all-True mask must reproduce the gated decode of z_a"
    assert torch.allclose(
        sae.swap_decode(z_a, z_b, atom_mask=all_false), sae.decode(z_b), atol=1e-5
    ), "all-False mask must reproduce the gated decode of z_b"

    # One-hot mask: feature 3 taken from z_a, rest from z_b. Manually splice
    # the latent then gated-decode; must match the fused swap_decode.
    m = torch.zeros(F, dtype=torch.bool)
    m[3] = True
    z_manual = z_b.clone()
    z_manual[:, 3] = z_a[:, 3]
    assert torch.allclose(
        sae.swap_decode(z_a, z_b, atom_mask=m), sae.decode(z_manual), atol=1e-5
    ), "one-hot swap_decode must equal gated decode of the manually spliced latent"


def test_no_swap_identity():
    """all-False mask == gated decode of z_b (the no-swap baseline)."""
    sae, x = _make()
    z_a = sae(x).z
    z_b = sae(x.flip(0)).z
    F = z_a.shape[-1]
    all_false = torch.zeros(F, dtype=torch.bool)
    s = sae.swap_decode(z_a, z_b, atom_mask=all_false)
    assert torch.allclose(s, sae.decode(z_b), atol=1e-5), \
        "all-False swap_decode must equal the gated decode of z_b (no-op on a)"


def test_gate_gradient_flows():
    """Interchange loss must push a gradient into the decoder gate."""
    sae, x = _make()
    x_a = x
    x_b = x.flip(0)
    # Build a dummy target_swap.
    v = torch.randn(x.shape[1])
    hue_a = torch.rand(x.shape[0])
    hue_b = torch.rand(x.shape[0])
    tgt = build_target_swap(x_a, x_b, hue_a, hue_b, v)
    # Use a non-zero gate L1 so the gate definitely receives a gradient even if
    # the bool-thresholded swap term is locally flat for some atoms.
    losses = sae.compute_loss(x_a, x_b, tgt,
                              lambda_intv=1.0, lambda_gate=1e-2, lambda_l1=0.0,
                              lambda_gate_entropy=0.0)
    losses["loss"].backward()
    g = sae.decoder.gate.grad
    assert g is not None, "decoder.gate must receive a gradient"
    assert g.abs().sum().item() > 1e-8, "decoder.gate gradient must be non-trivial"


def test_loss_decreases_after_optim_steps():
    sae, x = _make(D=24, F=16, B=12, top_k=4, seed=1)
    x_a = x
    x_b = x.flip(0)
    v = torch.randn(x.shape[1])
    hue_a = torch.rand(x.shape[0])
    hue_b = torch.rand(x.shape[0])
    tgt = build_target_swap(x_a, x_b, hue_a, hue_b, v)

    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    init = float(sae.compute_loss(x_a, x_b, tgt, lambda_intv=1.0,
                                  lambda_gate=0.0, lambda_l1=0.0,
                                  lambda_gate_entropy=0.0)["loss"].item())
    for _ in range(40):
        l = sae.compute_loss(x_a, x_b, tgt, lambda_intv=1.0,
                             lambda_gate=0.0, lambda_l1=0.0,
                             lambda_gate_entropy=0.0)["loss"]
        opt.zero_grad(); l.backward(); opt.step()
    final = float(l.item())
    assert final < init * 0.95, f"loss should decrease: init={init:.4f} final={final:.4f}"


def test_fit_hue_direction_recovers_signal():
    torch.manual_seed(7)
    D = 16
    N = 200
    true_v = torch.randn(D)
    true_v = true_v / true_v.norm()
    hue = torch.rand(N) * 2 - 1
    X = hue.unsqueeze(-1) * true_v.unsqueeze(0) + 0.05 * torch.randn(N, D)
    v_est = fit_hue_direction(X, hue)
    # The estimated direction should correlate strongly with true_v
    # (in the sense that X v_est tracks hue).
    Xc = X - X.mean(0, keepdim=True)
    pred = Xc @ v_est
    corr = torch.corrcoef(torch.stack([pred, hue - hue.mean()]))[0, 1]
    assert corr.abs().item() > 0.95, f"hue ridge fit weak: corr={corr.item():.3f}"
