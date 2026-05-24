// Hand-mirrored from backend/app.py Pydantic models.
// Regenerate with: openapi-typescript http://localhost:8000/openapi.json -o shared/schema.ts
// (See scripts/build.sh.)

export interface ManifoldPoint {
  id: number;
  name: string;
  hex: string;
  rgb: [number, number, number];
  hsv: [number, number, number];
  modifier_count: number;
  monoword: boolean;
  xyz: [number, number, number];
}

export interface ManifoldResponse {
  points: ManifoldPoint[];
  W_joint: number[][];
  d1_r2_cv: number[];
  mean_joint_cv: number;
  targets: string[];
}

export interface DiagnosticsResponse {
  variance_per_axis: number[];
  curvature_trace: number[];
  axis_labels: string[];
  cv_r2: number[];
}

export interface SteerRequest {
  point_id?: number | null;
  hue: number;
  saturation: number;
  value: number;
  modifier_count: number;
  monoword: boolean;
  alpha: number;
  prompt: string;
}

export interface SteerResponse {
  completion: string;
  matched_name: string;
  matched_hex: string;
  matched_distance: number;
  concept: Record<string, number | boolean>;
}

export type WSMessage =
  | { type: "hello"; data: { n_clients: number } }
  | { type: "steer"; data: SteerResponse }
  | { type: "client"; data: unknown };
