# Template-transfer report

**Primary metric:** all-fold held-out-template EV from
`template_out/template_transfer.json`. Coordinate consistency is reported as
secondary evidence only; it is not allowed to override the EV result.

## Verdict

The all-fold EV result is **null** for a robust transferable circle claim.
Weekday is explicitly fragile: its coordinate median is high, but the
2-coordinate linear baseline has higher all-fold EV. Month is the stronger
case, but it is best described as month/null-robust rather than a general
weekday-and-month robust-circle result.

| set | folds | chart EV | linear-1 EV | linear-2 EV | chart-linear1 | chart-linear2 | EV verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| month | 14 | 0.124 | 0.028 | 0.045 | 0.096 | 0.080 | chart beats 2D linear on all-fold EV |
| weekday | 14 | 0.263 | 0.014 | 0.429 | 0.249 | -0.166 | EV-null/fragile: 2D linear beats chart |

## Coordinate Readout

Coordinate consistency is useful as an interpretability diagnostic, but the
weekday result is not a robust reconstruction-transfer win.

| set | mean circ-corr | median circ-corr | frac folds > 0.8 | unseen adjacency |
|---|---:|---:|---:|---:|
| month | 0.708 | 0.812 | 0.50 | 0.464 |
| weekday | 0.781 | 0.951 | 0.64 | 0.449 |

## Clean-Fold Slice

The clean-fold slice keeps folds with coordinate consistency > 0.8. It is
reported for diagnosis, not as the headline metric.

| set | clean folds | chart EV | linear-1 EV | linear-2 EV |
|---|---:|---:|---:|---:|
| month | 7 | 0.217 | 0.001 | 0.001 |
| weekday | 9 | 0.330 | 0.007 | 0.430 |

## Reproduce

```bash
python3 experiments/chart_transfer/report_template_transfer.py
```

This command renders the markdown report from the existing JSON. Rerunning
`template_transfer.py` requires the richer harvest cache in
`experiments/chart_transfer/template_out/harvest_more/`.
