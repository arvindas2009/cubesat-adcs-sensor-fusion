// Mirrors docs/specs/webapp_protocol.md §1-§3 (the wire format). Quaternions are scalar-first
// [w, x, y, z], body-to-ECI. Degrees only in fields named *_deg / *_deg_s; wheel speeds in rpm.

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
/** Row-major 3x3 matrix as nested lists. */
export type Mat3 = [Vec3, Vec3, Vec3];

export type SensorPreset = 'realistic' | 'ideal';
export type FilterPreset = 'design' | 'g7';

/** §1 session parameters. */
export interface SessionParams {
  orbits: number;
  rate_deg_s: number;
  seed: number;
  sensors: SensorPreset;
  filter: FilterPreset;
  dumping: boolean;
  inertia_scale: number;
}

/** §2 static per-session information. */
export interface Info {
  period_s: number;
  t_end_s: number;
  control_period_s: number;
  n_steps: number;
  epoch_utc: string;
  body_size_m: Vec3;
  wheel_max_speed_rad_s: number;
  m_max_Am2: number;
  safe_rate_deg_s: number;
  detumble_threshold_deg_s: number;
  orbit_track_eci_m: Vec3[];
  eclipses_s: [number, number][];
}

/** §2 dynamic session status. */
export interface Status {
  k: number;
  t: number;
  done: boolean;
  playing: boolean;
  speed: number;
  connected: boolean;
}

/** Subset of a JSON-schema property as produced by pydantic (used to build the form). */
export interface SchemaProperty {
  type?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  enum?: (string | number)[];
  default?: unknown;
  title?: string;
  description?: string;
  [key: string]: unknown;
}

/** GET /api/adcs/presets. */
export interface Presets {
  params_schema: {
    properties: Record<string, SchemaProperty>;
    [key: string]: unknown;
  };
  defaults: SessionParams;
}

/** Body of POST 201 and GET /api/adcs/sessions/{id}. */
export interface SessionInfoResponse {
  id: string;
  params: SessionParams;
  info: Info;
  status: Status;
}

/** §3 first message after connect (and again after `reset`). */
export interface Hello {
  type: 'hello';
  id: string;
  params: SessionParams;
  info: Info;
  status: Status;
}

export type ModeName = 'DETUMBLE' | 'POINTING';

export interface FrameEci {
  q: Quat;
  q_est: Quat | null;
  r_m: Vec3;
  v_m_s: Vec3;
  sun: Vec3;
  b_T: Vec3;
  gmst_rad: number;
}

export interface FrameDisplay {
  C: Mat3;
  C_est: Mat3 | null;
  sun: Vec3;
  b: Vec3;
}

export interface FrameMetrics {
  nadir_err_deg: number;
  est_nadir_err_deg: number | null;
  know_err_deg: number | null;
  att_sigma_deg: number | null;
  rate_deg_s: number;
  w_deg_s: Vec3;
  w_est_deg_s: Vec3 | null;
  wheel_speed_rpm: Vec3;
  wheel_frac: number;
  m_Am2: Vec3;
  u_w_Nm: Vec3;
  sun_used: number;
}

export interface KickEvent {
  t: number;
  dw_deg_s: Vec3;
}

export interface FrameEvents {
  t_init: number | null;
  switch_times: number[];
  fallback_times: number[];
  kicks: KickEvent[];
}

/** §3.2 streamed frame. */
export interface Frame {
  type: 'frame';
  seq: number;
  k: number;
  t: number;
  orbit: number;
  mode: 0 | 1;
  mode_name: ModeName;
  dump: boolean;
  lit: boolean;
  done: boolean;
  playing: boolean;
  speed_set: number;
  speed_actual: number;
  eci: FrameEci;
  display: FrameDisplay;
  metrics: FrameMetrics;
  events: FrameEvents;
}

/** Server-side command or validation error; the socket stays open. */
export interface ErrorMsg {
  type: 'error';
  message: string;
}

export type ServerMessage = Hello | Frame | ErrorMsg;

export type Axis = 'x' | 'y' | 'z';

/** §3.1 client-to-server commands. */
export type Command =
  | { cmd: 'play' }
  | { cmd: 'pause' }
  | { cmd: 'speed'; value: number }
  | { cmd: 'kick'; axis: Axis; deg_s: number }
  | { cmd: 'step'; n: number }
  | { cmd: 'reset' };
