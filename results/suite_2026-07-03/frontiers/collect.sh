#!/bin/bash
# Collect finished synthetic frontier JSONs from MSI, then plot + score verdicts.
# Idempotent: re-run as jobs land; missing files are skipped. Run from repo root.
#   bash results/suite_2026-07-03/frontiers/collect.sh
set -uo pipefail
MSI=~/msi-node/msi
RROOT=/projects/standard/hsiehph/sauer354/Manifold-SAE/results/suite_2026-07-03/frontiers
FR=results/suite_2026-07-03/frontiers
PY="python"

# families: synth_p* (raw), fair_p* (single-active raw), pca_p* (single-active PCA-reduced)
TAGS="synth_p48 synth_p256 synth_p1024 fair_p48 fair_p96 fair_p192 pca_p256 pca_p1024"
for base in $TAGS; do
  for dgp in curved linear; do
    f="${base}_${dgp}.json"
    $MSI get "$RROOT/$f" "$FR/$f" 2>/dev/null && echo "[got] $f" || echo "[skip] $f"
  done
done

for base in $TAGS; do
  cf="$FR/${base}_curved.json"; lf="$FR/${base}_linear.json"
  [ -f "$cf" ] || continue
  $PY -m experiments.frontier_plots --in "$cf" --out-dir "$FR" --tag "${base}" 2>/dev/null && echo "[plot] ${base}"
  if [ -f "$lf" ]; then
    $PY -m experiments.frontier_analyze --curved "$cf" --linear "$lf" \
        --out-md "$FR/verdicts_${base}.md" --out-json "$FR/verdicts_${base}.json" 2>/dev/null && echo "[verdict] ${base}"
  else
    $PY -m experiments.frontier_analyze --curved "$cf" \
        --out-md "$FR/verdicts_${base}.md" --out-json "$FR/verdicts_${base}.json" 2>/dev/null && echo "[verdict] ${base} (curved only)"
  fi
done
echo "done. verdicts: $FR/verdicts_*.md"
