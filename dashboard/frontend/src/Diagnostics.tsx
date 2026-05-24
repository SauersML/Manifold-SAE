import { useEffect, useState } from "react";
import { fetchDiagnostics } from "./api";
import type { DiagnosticsResponse } from "@shared/schema";

export function Diagnostics() {
  const [d, setD] = useState<DiagnosticsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetchDiagnostics().then(setD).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <div style={{ color: "#f87171" }}>{err}</div>;
  if (!d) return <div>Loading diagnostics…</div>;
  const maxVar = Math.max(...d.variance_per_axis, 1e-9);
  const maxCurv = Math.max(...d.curvature_trace, 1e-9);

  return (
    <div>
      <div className="section">
        <h2>Variance per axis</h2>
        <div className="barchart">
          {d.variance_per_axis.map((v, i) => (
            <div key={i} className="bar" style={{ height: `${(v / maxVar) * 100}%` }}>
              <span>{d.axis_labels[i] ?? `a${i}`}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 22, fontSize: 11, color: "var(--muted)" }}>
          values: {d.variance_per_axis.map((v) => v.toFixed(4)).join("  ")}
        </div>
      </div>

      <div className="section">
        <h2>Cross-validated R² (per target)</h2>
        <div className="barchart">
          {d.cv_r2.map((v, i) => (
            <div
              key={i}
              className="bar"
              style={{ height: `${Math.max(0, v) * 100}%`, opacity: 0.75 + 0.25 * v }}
            >
              <span>t{i}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 22, fontSize: 11, color: "var(--muted)" }}>
          {d.cv_r2.map((v) => v.toFixed(3)).join("  ")}
        </div>
      </div>

      <div className="section">
        <h2>Curvature trace (PC-1 sorted)</h2>
        <svg className="spark" viewBox={`0 0 ${d.curvature_trace.length} 100`} preserveAspectRatio="none">
          <polyline
            fill="none"
            stroke="#22d3ee"
            strokeWidth="1"
            points={d.curvature_trace
              .map((y, x) => `${x},${100 - (y / maxCurv) * 95}`)
              .join(" ")}
          />
        </svg>
        <div style={{ marginTop: 4, fontSize: 11, color: "var(--muted)" }}>
          {d.curvature_trace.length} samples · max {maxCurv.toFixed(4)}
        </div>
      </div>
    </div>
  );
}
