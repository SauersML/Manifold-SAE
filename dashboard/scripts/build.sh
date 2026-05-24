#!/usr/bin/env bash
# Build production frontend + (optionally) regenerate the shared TS schema.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Regenerate shared schema from a live backend if one is reachable.
if curl -sf http://localhost:8000/openapi.json > /dev/null 2>&1; then
  echo "[build] regenerating shared/schema.ts from live backend"
  (cd "$ROOT/frontend" && npx --yes openapi-typescript http://localhost:8000/openapi.json -o "$ROOT/shared/schema.ts.gen")
  echo "[build] wrote shared/schema.ts.gen (review before replacing schema.ts)"
fi

cd "$ROOT/frontend"
npm install
npm run build
echo "[build] frontend built to $ROOT/frontend/dist"
