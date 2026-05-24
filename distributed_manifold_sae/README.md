# distributed_manifold_sae

Distributed PyTorch training scaffold for a K=1,000,000-atom Manifold-SAE on
cogito-L40 residual-stream activations (D=7168).

## Status

| Deliverable                  | State                                    |
|------------------------------|------------------------------------------|
| `__init__.py`                | done                                     |
| `model.py`                   | functional (forward + backward verified) |
| `loss.py`                    | functional (all 7 terms wired)           |
| `data.py`                    | functional (memmap + DistributedSampler) |
| `train.py`                   | scaffolded; FSDP wrap policy + bf16 mixed precision; not run end-to-end at K=1M |
| `eval.py`                    | minimal stub (R², dead-atom rate)        |
| `dashboard.py`               | TensorBoard wrapper with no-op fallback  |
| `configs/k1m_circle_cogito.yaml` | done                                 |
| `scripts/train.sh`           | done                                     |
| `README.md`                  | this file                                |

## Architecture

```
x ∈ R^D
  │
  ▼  AmortizedEncoder (MLP, gradient-checkpointed)
logits, θ_raw ∈ R^{K} × R^{K × d_atom}
  │              │
  ▼              ▼  Circle retraction (L2-normalise per atom)
IBP-Gumbel    θ ∈ S^1
mask  (B, K, top_k active)
  │
  ▼
For each active (b, k):
    direction_k(θ) = anchor_k + tangent_k · θ      ∈ R^D
    contribution   = mask_{b,k} · amp_{b,k} · direction_k(θ_{b,k})
  │
  ▼  sum + b_dec
recon ∈ R^D
```

Decoder parameters that scale with K:
* `anchor` ∈ R^{K × D}             (~K·D·4 bytes  ≈ 28 GB at K=1M, D=7168 in fp32)
* `tangent` ∈ R^{K × D × d_atom}   (~K·D·d·4 bytes ≈ 57 GB at K=1M, D=7168, d=2)

Both are sharded by FSDP across N ranks. Forward uses gather-by-active-indices
so each step touches only `O(B · top_k · D)` decoder memory regardless of K.

## Composed loss

```
total = w_recon  · MSE(recon, x)
      + w_ibp    · KL(q_mask || Bern(top_k / K))
      + w_iso    · ‖T_k^T T_k − I_d‖²_F          (active atoms only)
      + w_ard    · Inverse-Gamma marginal on per-atom amp variance
      + w_mech   · L1 on per-atom activation rate across batch
      + w_anchor · ‖anchor_k‖²                    (active atoms only)
      + w_tangent· ‖tangent_k‖²_F                 (active atoms only)
```

Each penalty is its own callable in `loss.py`; the `ComposedLoss` class is just
a weighted sum harness — consistent with the gamfit composition-engine
philosophy (`project_gamfit_composition_engine.md` in memory).

## Usage

Single-process smoke test (CPU):

```bash
uv run python -m distributed_manifold_sae.train --mock-run
```

Mock 4-rank gloo launch:

```bash
torchrun --nproc_per_node=4 -m distributed_manifold_sae.train \
    --config distributed_manifold_sae/configs/k1m_circle_cogito.yaml \
    --mock-run
```

Real 4-GPU NCCL launch:

```bash
bash distributed_manifold_sae/scripts/train.sh
```

## What still needs to be done by a human

1. **Real-cluster validation.** No GPU was available during scaffolding; FSDP
   wrap policy and bf16 mixed-precision config are written but unverified at K=1M.
   In particular, `riemannian_retract` calls `torch.linalg.qr` on the full
   tangent tensor — under FSDP `use_orig_params=True` each rank sees only its
   shard, which is correct, but this needs to be checked with the actual
   sharded layout.
2. **K=1M memory budget.** AmortizedEncoder's `out_proj` produces
   `K·(1+d_atom) = 3M` outputs from a 4096-d hidden. That's 12B params in the
   output projection alone — needs explicit FSDP wrapping or replacement with
   a low-rank head (e.g. factored projection `R^{H → r} → R^{r → K(1+d)}`).
3. **Optimizer-level Riemannian step.** The current design calls
   `riemannian_retract()` after `optimizer.step()` from the training loop. A
   cleaner implementation would subclass `torch.optim.AdamW` to wrap step()
   itself, so DDP/FSDP can manage the manifold projection per-shard.
4. **Hydra integration.** Config loading is plain YAML; swap for Hydra if
   sweep / multirun is wanted.
5. **Eval coverage.** `eval.py` only reports R² + dead-atom rate. The
   project's full eval (HSV-axis alignment, name-token correlations,
   max-activating-example mining) needs to be ported.
6. **Resume.** `load_checkpoint` exists but isn't wired into `train()`'s main
   loop — add `--resume PATH` flag.

## Verified

The single-process forward + backward pass produces non-NaN losses on all 7
terms and finite gradients on `encoder`, `anchor`, and `tangent`. See
`__init__.py` for the public API.
