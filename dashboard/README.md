# Manifold-SAE Dashboard

A real-time interactive WebGL dashboard for exploring the cogito-L40 colour
concept manifold produced by `auto_exp_38` / `auto_exp_54`.

- 3D WebGL scatter of 949 xkcd colour concepts, coloured by RGB / hue /
  modifier-count / monoword
- Side panel with HSV + modifier-count + monoword sliders, an alpha slider,
  and a free-form prompt
- Click any point to send a steering request; the resulting completion
  streams into the chat panel via WebSocket
- Tabbed views: **Manifold** (3D scene), **Diagnostics** (variance, CV R²,
  curvature trace), **Steering History**

## Three-command quickstart on a fresh machine

```bash
git clone <this-repo> Manifold-SAE
cd Manifold-SAE/dashboard
docker-compose up --build
```

Open <http://localhost:3000>. The 3D manifold appears immediately; click a
point to steer.

The backend is reachable directly at <http://localhost:8000>
(`/api/health`, `/api/manifold`, `/api/diagnostics`, `/api/steer`, `/ws`).

## Architecture

```
dashboard/
├── frontend/        React + TypeScript + three.js (Vite)
│   ├── src/
│   │   ├── App.tsx            shell, tabs, sliders, chat
│   │   ├── ManifoldView.tsx   InstancedMesh + GPU picking
│   │   ├── Diagnostics.tsx    variance / R² / curvature charts
│   │   ├── api.ts             REST + WebSocket client
│   │   └── styles.css
│   ├── vite.config.ts         proxies /api and /ws to backend in dev
│   └── package.json
├── backend/         FastAPI
│   ├── app.py                 all routes + WebSocket hub
│   └── requirements.txt
├── shared/
│   ├── schema.py              Pydantic models (re-exports)
│   └── schema.ts              hand-mirrored TS, regenerable via openapi-typescript
├── docker/
│   ├── Dockerfile.backend     python:3.11-slim + uvicorn
│   ├── Dockerfile.frontend    node build → nginx:alpine
│   ├── nginx.conf             SPA + /api + /ws upstream
│   └── docker-compose.yml
├── docker-compose.yml         convenience copy at dashboard/ root
├── scripts/
│   ├── dev.sh                 uvicorn --reload + vite dev
│   ├── build.sh               regen schema, npm build
│   └── deploy.sh              docker compose build && up -d
├── tests/
│   ├── test_backend.py        pytest + TestClient
│   ├── e2e.spec.ts            Playwright
│   └── playwright.config.ts
└── README.md
```

## Local development (no Docker)

```bash
# from repo root
bash dashboard/scripts/dev.sh
```

This launches:

- Backend on :8000 (uvicorn --reload) loading `runs/auto_exp_54_nonhsv_gauge.npz`
- Frontend on :3000 (vite dev with HMR), proxying `/api` and `/ws` to :8000

Cold-start: the backend takes <2 s because the module only imports numpy +
FastAPI and lazily loads the npz on first request.

## Click-to-steer data flow

1. User clicks a point in the WebGL canvas.
2. The picking-scene render reads back a single pixel; the RGB triple
   decodes to an instance id (`id = r + (g<<8) + (b<<16) - 1`).
3. The React layer calls `POST /api/steer` with `{point_id, hue, sat, val,
   modifier_count, monoword, alpha, prompt}`.
4. The backend resolves HSV / modifier counts from the cached manifold,
   composes a steered completion against the validated nearest-xkcd map,
   and returns `{completion, matched_name, matched_hex, matched_distance,
   concept}`.
5. The same response is broadcast over the `/ws` socket to every connected
   client, populating the chat panel and the History tab.

## Tests

```bash
# Backend
pip install -r dashboard/backend/requirements.txt
PYTHONPATH=dashboard pytest dashboard/tests/test_backend.py -v

# Frontend end-to-end (requires the dev stack running)
cd dashboard/tests
npm init -y && npm i -D @playwright/test
npx playwright install chromium
npx playwright test e2e.spec.ts
```

## Screenshots

_(placeholder — capture these after first run)_

- `docs/screenshots/manifold.png` – default RGB-coloured 3D view
- `docs/screenshots/steer.png` – chat panel after clicking a point
- `docs/screenshots/diagnostics.png` – diagnostics tab
