"""Tests for DAS-SAE.

(1) swap op symmetry: swap(a, b, m) + swap(b, a, m) ≡ a + b for any m.
(2) mask gradient flows: grad of intv loss w.r.t. gate_logits is non-zero.
(3) loss decreases under a few optimizer steps.
(4) no-swap (mask=0) reproduces standard SAE forward (same as decode(z_a)).
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


def test_swap_symmetry():
    sae, x = _make()
    x_a = x
    x_b = x.flip(0)
    z_a = sae(x_a).z
    z_b = sae(x_b).z
    # Random mask.
    m = torch.rand(z_a.shape[-1])
    s_ab = sae.swap(z_a, z_b, mask=m)
    s_ba = sae.swap(z_b, z_a, mask=m)
    assert torch.allclose(s_ab + s_ba, z_a + z_b, atol=1e-6), \
        "swap(a,b,m) + swap(b,a,m) must equal a + b for any m"


def test_no_swap_identity():
    sae, x = _make()
    z_a = sae(x).z
    z_b = sae(x.flip(0)).z
    zero_mask = torch.zeros(z_a.shape[-1])
    s = sae.swap(z_a, z_b, mask=zero_mask)
    assert torch.allclose(s, z_a), "mask=0 swap must equal z_a (no-op)"
    # And decoding it must equal the standard SAE forward.
    x_hat_swap = sae.decode(s)
    x_hat_std = sae.decode(z_a)
    assert torch.allclose(x_hat_swap, x_hat_std), \
        "decode(swap_no_op) must equal decode(z_a) = standard SAE forward"


def test_mask_gradient_flows():
    sae, x = _make()
    x_a = x
    x_b = x.flip(0)
    # Build a dummy target_swap.
    v = torch.randn(x.shape[1])
    hue_a = torch.rand(x.shape[0])
    hue_b = torch.rand(x.shape[0])
    tgt = build_target_swap(x_a, x_b, hue_a, hue_b, v)
    losses = sae.compute_loss(x_a, x_b, tgt,
                              lambda_intv=1.0, lambda_gate=0.0, lambda_l1=0.0,
                              lambda_gate_entropy=0.0)
    losses["loss"].backward()
    g = sae.gate_logits.grad
    assert g is not None, "gate_logits must receive a gradient"
    assert g.abs().sum().item() > 1e-8, "gate_logit gradient must be non-trivial"


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
