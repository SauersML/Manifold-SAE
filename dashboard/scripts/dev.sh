#!/usr/bin/env bash
# Run backend (uvicorn --reload) + frontend (vite dev) side-by-side for local dev.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

trap 'kill 0' SIGINT SIGTERM EXIT

# Backend
(
  cd "$ROOT/.."
  python -m pip install --quiet -r dashboard/backend/requirements.txt
  PYTHONPATH="$ROOT" uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
) &

# Frontend
(
  cd "$ROOT/frontend"
  if [ ! -d node_modules ]; then npm install; fi
  npm run dev
) &

wait
