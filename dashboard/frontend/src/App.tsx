import { useEffect, useMemo, useRef, useState } from "react";
import { fetchManifold, openWS, postSteer } from "./api";
import type { ManifoldResponse, SteerResponse } from "@shared/schema";
import { ManifoldView, type ColorMode } from "./ManifoldView";
import { Diagnostics } from "./Diagnostics";

type Tab = "manifold" | "diagnostics" | "history";

export default function App() {
  const [manifold, setManifold] = useState<ManifoldResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("manifold");
  const [colorMode, setColorMode] = useState<ColorMode>("rgb");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [history, setHistory] = useState<SteerResponse[]>([]);
  const [busy, setBusy] = useState(false);
  const [wsStatus, setWsStatus] = useState<"connecting" | "open" | "closed">("connecting");

  const [hue, setHue] = useState(0.55);
  const [sat, setSat] = useState(0.7);
  const [val, setVal] = useState(0.7);
  const [modCount, setModCount] = useState(0);
  const [monoword, setMono] = useState(true);
  const [alpha, setAlpha] = useState(1.0);
  const [prompt, setPrompt] = useState("The color is");

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetchManifold().then(setManifold).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    const ws = openWS((m: any) => {
      if (m?.type === "steer") {
        setHistory((h) => [m.data as SteerResponse, ...h].slice(0, 50));
      }
    });
    ws.onopen = () => setWsStatus("open");
    ws.onclose = () => setWsStatus("closed");
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  // Sync sliders from selected point
  useEffect(() => {
    if (selectedId === null || !manifold) return;
    const p = manifold.points[selectedId];
    if (!p) return;
    setHue(p.hsv[0]);
    setSat(p.hsv[1]);
    setVal(p.hsv[2]);
    setModCount(p.modifier_count);
    setMono(p.monoword);
  }, [selectedId, manifold]);

  const handleSteer = async (point_id: number | null = null) => {
    setBusy(true);
    try {
      const resp = await postSteer({
        point_id,
        hue,
        saturation: sat,
        value: val,
        modifier_count: modCount,
        monoword,
        alpha,
        prompt,
      });
      setHistory((h) => [resp, ...h].slice(0, 50));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPick = (id: number | null) => {
    setSelectedId(id);
    if (id !== null) void handleSteer(id);
  };

  const liveColor = useMemo(() => {
    const c = hsvToCss(hue, sat, val);
    return c;
  }, [hue, sat, val]);

  return (
    <div className="app">
      <header>
        <h1>Manifold-SAE · Cogito Color Dashboard</h1>
        <div className="status">
          {manifold ? `${manifold.points.length} points · joint CV R² ${manifold.mean_joint_cv.toFixed(3)}` : "…"}
          {" · ws "}
          <span style={{ color: wsStatus === "open" ? "#22d3ee" : "#f87171" }}>{wsStatus}</span>
        </div>
      </header>

      <div className="main">
        {err && <div className="hud" style={{ color: "#f87171" }}>{err}</div>}
        {manifold && tab !== "diagnostics" && (
          <>
            <ManifoldView
              points={manifold.points}
              colorMode={colorMode}
              onPick={onPick}
              selectedId={selectedId}
            />
            <div className="legend">
              {(["rgb", "hue", "modifier", "monoword"] as ColorMode[]).map((m) => (
                <button
                  key={m}
                  className={m === colorMode ? "active" : ""}
                  onClick={() => setColorMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
            <div className="hud">
              <div>drag = rotate · shift-drag / right-drag = pan · wheel = zoom</div>
              <div>
                selected: <b>{selectedId !== null ? manifold.points[selectedId].name : "—"}</b>
              </div>
            </div>
          </>
        )}
        {tab === "diagnostics" && (
          <div style={{ position: "absolute", inset: 0, padding: 24, overflow: "auto" }}>
            <Diagnostics />
          </div>
        )}
      </div>

      <aside className="sidebar">
        <div className="tabs">
          <button className={"tab " + (tab === "manifold" ? "active" : "")} onClick={() => setTab("manifold")}>
            Manifold
          </button>
          <button
            className={"tab " + (tab === "diagnostics" ? "active" : "")}
            onClick={() => setTab("diagnostics")}
          >
            Diagnostics
          </button>
          <button className={"tab " + (tab === "history" ? "active" : "")} onClick={() => setTab("history")}>
            History
          </button>
        </div>

        {tab !== "history" && (
          <div className="tab-body">
            <div className="section">
              <h2>Concept Sliders</h2>
              <div style={{
                height: 32,
                borderRadius: 6,
                background: liveColor,
                marginBottom: 12,
                border: "1px solid var(--border)",
              }} />
              <Slider label="hue" v={hue} set={setHue} />
              <Slider label="saturation" v={sat} set={setSat} />
              <Slider label="value" v={val} set={setVal} />
              <Slider label="modifiers" v={modCount} set={(x) => setModCount(Math.round(x))} min={0} max={4} step={1} fmt={(x) => String(Math.round(x))} />
              <Slider label="alpha" v={alpha} set={setAlpha} min={-2} max={2} step={0.05} />
              <div className="checkbox-row">
                <input id="mw" type="checkbox" checked={monoword} onChange={(e) => setMono(e.target.checked)} />
                <label htmlFor="mw">monoword</label>
              </div>
            </div>

            <div className="section">
              <h2>Prompt</h2>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                style={{
                  width: "100%",
                  background: "var(--panel2)",
                  border: "1px solid var(--border)",
                  color: "var(--text)",
                  padding: 8,
                  borderRadius: 6,
                  fontFamily: "inherit",
                  fontSize: 12,
                  resize: "vertical",
                }}
              />
            </div>

            <button className="primary" disabled={busy} onClick={() => handleSteer(null)}>
              {busy ? "Steering…" : "Steer with current concept"}
            </button>

            <div className="section" style={{ marginTop: 18 }}>
              <h2>Latest completion</h2>
              {history[0] ? <ChatMsg msg={history[0]} /> : <div style={{ color: "var(--muted)", fontSize: 12 }}>Click a point or press steer.</div>}
            </div>
          </div>
        )}

        {tab === "history" && (
          <div className="tab-body">
            <div className="section">
              <h2>Steering history ({history.length})</h2>
              <div className="chat">
                {history.length === 0 && <div style={{ color: "var(--muted)", fontSize: 12 }}>No steers yet.</div>}
                {history.map((m, i) => (
                  <ChatMsg key={i} msg={m} />
                ))}
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

function Slider({
  label,
  v,
  set,
  min = 0,
  max = 1,
  step = 0.01,
  fmt,
}: {
  label: string;
  v: number;
  set: (x: number) => void;
  min?: number;
  max?: number;
  step?: number;
  fmt?: (x: number) => string;
}) {
  return (
    <div className="slider-row">
      <label>{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v}
        onChange={(e) => set(parseFloat(e.target.value))}
      />
      <span className="val">{fmt ? fmt(v) : v.toFixed(2)}</span>
    </div>
  );
}

function ChatMsg({ msg }: { msg: SteerResponse }) {
  return (
    <div className="chat-msg">
      <div>
        <span className="swatch" style={{ background: msg.matched_hex }} />
        {msg.completion}
      </div>
      <div className="meta">
        nearest <b>{msg.matched_name}</b> · d={msg.matched_distance.toFixed(3)} · α={String(msg.concept.alpha)}
      </div>
    </div>
  );
}

function hsvToCss(h: number, s: number, v: number): string {
  // simple HSV->RGB
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r = 0, g = 0, b = 0;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}
