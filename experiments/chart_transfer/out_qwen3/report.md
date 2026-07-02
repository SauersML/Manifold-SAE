# Cross-layer chart transport vs. compute (qwen3)

Per-layer K=1 circle chart (torch backend), coordinate transport `t_{L'}=h(t_L)` via `gamfit.layer_transport`. Degree ±1 + small isometry defect ⇒ block **rotates** the circle (TRANSPORT); large defect ⇒ **recomputes** (COMPUTE). `compute_gap = R²_native − R²_transported`.


## weekday (layers [24, 32, 40], cyclic=True)

| layer | chart EV | decode ok |
|---|---:|---|
| L24 | 0.3526 | True |
| L32 | 0.2362 | True |
| L40 | 0.3209 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L24→L32 | 0 | 1.0000 | False | 0.2362 | -4.0518 | 4.2880 |
| L32→L40 | 0 | 1.0000 | False | 0.3209 | -1.4518 | 1.7727 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L24→L32 | -1 | 0.0027 | True |
| L32→L40 | -1 | 0.0001 | True |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L24→L40 | 3.202336225027956e-07 | 1.0 |

## month (layers [24, 32, 40], cyclic=True)

| layer | chart EV | decode ok |
|---|---:|---|
| L24 | 0.2435 | True |
| L32 | 0.2413 | True |
| L40 | 0.2283 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L24→L32 | -1 | 0.3905 | True | 0.2413 | -3.9869 | 4.2283 |
| L32→L40 | 0 | 0.3888 | False | 0.2283 | -0.1773 | 0.4056 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L24→L32 | -1 | 0.0000 | True |
| L32→L40 | -1 | 0.0000 | True |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L24→L40 | 0.2693776083723147 | 0.0 |

## year (layers [24, 32, 40], cyclic=False)

| layer | chart EV | decode ok |
|---|---:|---|
| L24 | 0.1492 | True |
| L32 | 0.1114 | True |
| L40 | 0.1184 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L24→L32 | — | 0.5658 | True | 0.1114 | 0.0121 | 0.0993 |
| L32→L40 | — | 100.4161 | False | 0.1184 | -0.1523 | 0.2707 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L24→L32 | — | 0.2702 | False |
| L32→L40 | — | 0.1532 | False |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L24→L40 | 5.203508459582966 | 0.0 |

## color (layers [24, 32, 40], cyclic=True)

| layer | chart EV | decode ok |
|---|---:|---|
| L24 | 0.5694 | True |
| L32 | 0.5668 | True |
| L40 | 0.5487 | True |

**SAE-chart transport** | hop | degree | isometry defect | topo ok | R² native | R² transported | compute gap |
|---|---:|---:|---|---:|---:|---:|
| L24→L32 | 0 | 1.0000 | False | 0.5668 | 0.0223 | 0.5445 |
| L32→L40 | 0 | 0.9372 | False | 0.5487 | 0.5370 | 0.0117 |

**2D-PCA-angle transport (robustness cross-check)** | hop | degree | isometry defect | topo ok |
|---|---:|---:|---|
| L24→L32 | 1 | 0.0585 | True |
| L32→L40 | 1 | 0.0757 | True |

**Composition law:** | two-hop | composition defect | p |
|---|---:|---:|
| L24→L40 | 0.21375460030932533 | 0.0 |
