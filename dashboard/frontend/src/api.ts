import type {
  DiagnosticsResponse,
  ManifoldResponse,
  SteerRequest,
  SteerResponse,
} from "@shared/schema";

const BASE = ""; // proxied by Vite in dev, same-origin in prod (nginx)

export async function fetchManifold(): Promise<ManifoldResponse> {
  const r = await fetch(`${BASE}/api/manifold`);
  if (!r.ok) throw new Error(`manifold: ${r.status}`);
  return r.json();
}

export async function fetchDiagnostics(): Promise<DiagnosticsResponse> {
  const r = await fetch(`${BASE}/api/diagnostics`);
  if (!r.ok) throw new Error(`diagnostics: ${r.status}`);
  return r.json();
}

export async function postSteer(req: SteerRequest): Promise<SteerResponse> {
  const r = await fetch(`${BASE}/api/steer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`steer: ${r.status}`);
  return r.json();
}

export function openWS(onMessage: (m: unknown) => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch {
      /* ignore */
    }
  };
  return ws;
}
