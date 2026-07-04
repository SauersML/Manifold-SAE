#!/bin/bash
# Collect finished synthetic frontier JSONs from MSI, then plot + score verdicts.
# Idempotent: re-run as jobs land; missing files are skipped. Run from repo root.
#   bash results/suite_2026-07-03/frontiers/collect.sh
set -uo pipefail
MSI=~/msi-node/msi
RROOT=/projects/standard/hsiehph/sauer354/Manifold-SAE/results/suite_2026-07-03/frontiers
FR=results/suite_2026-07-03/frontiers
PY="python"

for p in 48 256 1024; do
  for dgp in curved linear; do
    f="synth_p${p}_${dgp}.json"
    $MSI get "$RROOT/$f" "$FR/$f" 2>/dev/null && echo "[got] $f" || echo "[skip] $f (not ready)"
  done
done
# calibration (p=256, curved vs manifold_linear only)
$MSI get /projects/standard/hsiehph/sauer354/scratch/fr_calib_p256.json "$FR/synth_p256_calib.json" 2>/dev/null \
  && echo "[got] calib" || echo "[skip] calib"

for p in 48 256 1024; do
  cf="$FR/synth_p${p}_curved.json"; lf="$FR/synth_p${p}_linear.json"
  [ -f "$cf" ] || continue
  $PY -m experiments.frontier_plots --in "$cf" --out-dir "$FR" --tag "p${p}_curved" 2>/dev/null && echo "[plot] p${p}"
  if [ -f "$lf" ]; then
    $PY -m experiments.frontier_analyze --curved "$cf" --linear "$lf" \
        --out-md "$FR/verdicts_p${p}.md" --out-json "$FR/verdicts_p${p}.json" 2>/dev/null && echo "[verdict] p${p}"
  else
    $PY -m experiments.frontier_analyze --curved "$cf" \
        --out-md "$FR/verdicts_p${p}.md" --out-json "$FR/verdicts_p${p}.json" 2>/dev/null && echo "[verdict] p${p} (curved only)"
  fi
done
echo "done. verdict markdown: $FR/verdicts_p*.md"
