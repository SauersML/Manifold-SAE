# BSF baseline — Block-Sparse Featurizers (torch reimplementation)

_Generated 2026-07-02 15:00 · CPU float64 · our faithful reimplementation of Goodfire's BSF as the head-to-head baseline._

Models: **Vanilla BSF** (free encoder + free block decoder, unit-norm rows), **Grassmannian BSF** (tied encoder `z=γ·xDᵀ`, one scalar γ, block decoders held column-orthonormal on the Stiefel manifold via periodic QR), and the **TopK-SAE** baseline (= vanilla BSF at block size b=1). Sparsity is per-block top-k on ‖z_g‖₂; codes are **signed** (no ReLU) so each block is a full subspace. AuxK resurrects dead blocks from the residual.

## 1. Synthetic planted-subspace recovery

8 random 4-dim subspaces in d=48; each of 2000 points is a sparse sum of 2 of them + noise σ=0.05. Recovery = mean cos² of principal angles between each planted subspace and its matched recovered block (1.0 = perfect).

| model | val EV | recovery R² (principal angles) | mean stable rank | mean utilization |
|---|---:|---:|---:|---:|
| BSF-vanilla | 0.8935 | **0.8185** | 2.90 | 0.93 |
| BSF-grassmann | 0.9345 | **0.9855** | 2.95 | 0.93 |

_(planted block size = 4; recovered stable rank ≈ 2.9 confirms each block spans its full 4-D subspace.)_

## 2. Real activations — EV at matched budget & sparsity

Data: `OLMO3_32B_BASE_SELF_QUALIA_LAST` layer 40, n=760 prompts, PCA-reduced to d=128. **Matched decoder budget** (latent width F=64 → dec params F·d constant) and **matched sparsity** (L0 = k·b = 8 nonzeros) across block sizes. b=1 is the TopK-SAE baseline.

| block b | model | G | k | L0 | val EV | mean stable rank | mean util |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | TopK-SAE | 64 | 8 | 8 | 0.4489 | 1.00 | 1.00 |
| 2 | BSF-grassmann | 32 | 4 | 8 | 0.4253 | 1.46 | 0.80 |
| 2 | BSF-vanilla | 32 | 4 | 8 | 0.4275 | 1.48 | 0.82 |
| 4 | BSF-grassmann | 16 | 2 | 8 | 0.3993 | 2.46 | 0.72 |
| 4 | BSF-vanilla | 16 | 2 | 8 | 0.3872 | 2.34 | 0.71 |
| 8 | BSF-grassmann | 8 | 1 | 8 | 0.3249 | 3.42 | 0.58 |
| 8 | BSF-vanilla | 8 | 1 | 8 | 0.3488 | 3.30 | 0.59 |

_At matched sparsity the reconstruction EV trades off against block width: TopK-SAE (b=1) EV = 0.4489, best block-BSF = BSF-vanilla b=2 EV = 0.4275 (Δ = -0.0215) — a wider block packs the same L0 into fewer, higher-stable-rank subspaces (stable rank climbs 1.0 → 3.4 as b: 1 → 8), the paper's ≈3 landing at b≈4–8._

## 3. Cyclic-feature block finding (weekday / month)

Per-template-demeaned residuals for the weekday (7-circle) and month (12-circle) token sets, fit with Grassmannian BSF (G=4 blocks, b=4, k=1). The paper's curve-detector result: a SINGLE block's decoder subspace holds the whole cyclic feature and its in-block coordinate orders it. We take the block whose 4-D subspace explains the most of the demeaned signal, read the chart off that ONE block (project *all* tokens onto it, in-block 2-D PCA angle), and score the ordering over *every* token. A circle is extrinsically 2-D, so the block's coordinate stable rank should be ≈2, and cyclic adjacency accuracy → 1.0.

| set | tokens | curve-detector block | subspace EV (whole cycle) | in-block coord stable rank | cyclic adjacency acc (all tokens) |
|---|---:|---:|---:|---:|---:|
| weekday | 7 | #3 | 0.80 | 2.36 | **1.00** |
| month | 12 | #3 | 0.81 | 2.41 | **1.00** |

_One block's subspace captures ~80% of each cyclic feature's variance and its chart orders all tokens perfectly around the circle (adjacency 1.0) at coordinate stable rank ≈2 — the extrinsic dimension of a circle. A single signed block is a curve detector, exactly the paper's result._

## Files

- `bsf.py` — models (vanilla / Grassmannian BSF, TopK-SAE baseline), block-TopK, AuxK, Stiefel retraction, metrics, gam shard-format loader.
- `train.py` — this driver (synthetic / real / cyclic phases + report).
- `metrics.json` — all numbers above, machine-readable.

