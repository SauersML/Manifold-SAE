# Cross-layer chart transport vs. compute (qwen25)

Per-layer K=1 circle chart (torch backend), coordinate transport `t_{L'}=h(t_L)` via `gamfit.layer_transport`. Degree ±1 + small isometry defect ⇒ block **rotates** the circle (TRANSPORT); large defect ⇒ **recomputes** (COMPUTE). `compute_gap = R²_native − R²_transported`.


## weekday (layers [5, 8, 11, 14], cyclic=True)

| layer | chart EV | decode ok |
|---|---:|---|
| L5 | 0.6233 | True |
| L8 | 0.5637 | True |
| L11 | 0.6015 | True |
| L14 | 0.6133 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L5→L8 | -1 | 0.4714 | False | 0.5637 | -2.3165 | 2.8802 |
| L8→L11 | -1 | 0.9596 | False | 0.6015 | -3.8237 | 4.4252 |
| L11→L14 | 2 | 0.8026 | False | 0.6133 | 0.5505 | 0.0628 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L5→L8 | -1 | 0.0032 | True |
| L8→L11 | -1 | 5.7691 | False |
| L11→L14 | 1 | 2.0596 | False |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L5→L11 | 0.27467230106117385 | 3.510112002169663e-07 |
| L8→L14 | 0.8871612243871407 | 0.029129685239369207 |

## month (layers [5, 8, 11, 14], cyclic=True)

| layer | chart EV | decode ok |
|---|---:|---|
| L5 | 0.5930 | True |
| L8 | 0.5375 | True |
| L11 | 0.6204 | True |
| L14 | 0.6218 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L5→L8 | 1 | 0.1715 | True | 0.5375 | -3.4186 | 3.9560 |
| L8→L11 | 1 | 37.8902 | False | 0.6204 | 0.3396 | 0.2808 |
| L11→L14 | 1 | 2.1597 | False | 0.6218 | -2.1342 | 2.7560 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L5→L8 | -1 | 1.0151 | False |
| L8→L11 | -1 | 0.3833 | True |
| L11→L14 | 1 | 0.2396 | False |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L5→L11 | 1.2332175357221318 | 0.00023591819161927763 |
| L8→L14 | 1.7014388922103747 | 0.0 |
