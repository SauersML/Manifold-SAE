#!/usr/bin/env bash
# Build and start both containers in detached mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  COMPOSE="docker compose"
fi

$COMPOSE -f docker-compose.yml build
$COMPOSE -f docker-compose.yml up -d
echo "[deploy] backend:  http://localhost:8000/api/health"
echo "[deploy] frontend: http://localhost:3000"
