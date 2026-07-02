# Seed stability: latent-level vs subspace-level

gamfit 0.1.242. Seeds [0, 1, 2]. Threshold cos>=0.9. Higher = more stable across seeds.

Latent = individual atom directions (Hungarian-matched); Subspace = principal-angle agreement of the spans (union, and per planted chart).

## real_olmo32b

Data: real, N=600, p=32, sha256 `8ff03780a195e5053f75d910`.

| dictionary | latent cos | latent frac>=thr | union span cos | per-chart cos | hashes identical |
|------------|-----------|------------------|----------------|---------------|------------------|
| gamfit sparse_dictionary_fit (deterministic) | 1.000 | 1.00 | 1.000 | n/a | True |
| random-init tiling SAE (aux) | 0.851 | 0.76 | 0.825 | n/a | False |
| gamfit sae_manifold_fit (circle) | NON_CONVERGENT (Timeout) | - | - | - | - |

## planted_circles

Data: planted 1-D circle manifolds (fixed real-shaped synthetic), N=3000, p=30, sha256 `6c49bf5aa19885104765cc50`.

| dictionary | latent cos | latent frac>=thr | union span cos | per-chart cos | hashes identical |
|------------|-----------|------------------|----------------|---------------|------------------|
| gamfit sparse_dictionary_fit (deterministic) | 1.000 | 1.00 | 1.000 | 1.000 | True |
| random-init tiling SAE (aux) | 0.845 | 0.85 | 0.833 | 0.889 | False |
| gamfit sae_manifold_fit (circle) | NON_CONVERGENT (NoResult) | - | - | - | - |

## Content-addressed hashes (dictionary_artifact v1 port)

- real_olmo32b / gamfit_linear: s0=c5f1fdd1df47 s1=c5f1fdd1df47 s2=c5f1fdd1df47
- real_olmo32b / aux_random_tiling: s0=741301302fff s1=d07a188a22d8 s2=7c0ad88486d9
- planted_circles / gamfit_linear: s0=a5d298bc2c31 s1=a5d298bc2c31 s2=a5d298bc2c31
- planted_circles / aux_random_tiling: s0=122ff707c072 s1=5b5072b1039a s2=f93a9af1f1d3
