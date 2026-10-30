// Mirrors docs/specs/px4_replay.md §1.1 (the replay document) and §2 (the log list). Quaternions are
// scalar-first [w, x, y, z], body (FRD) to NED; degrees only in fields named *_deg. The server writes
// NaN / infinity as `null`, so every number may be `null`; EKF2 rows are null outside its time span.

/** A number that the server may have replaced by `null` (NaN or infinity). */
export type NumOrNull = number | null;
/** One row of an (n, k) array: k numbers, any of which may be `null`. */
export type Row = readonly NumOrNull[];

export interface ReplayLogMeta {
  id: string;
  role: string;
  notes: string;
  duration_s: NumOrNull;
  origin_lat_deg: NumOrNull;
  origin_lon_deg: NumOrNull;
  gps_utc_start: string | number | null;
  declination_deg: NumOrNull;
  g_m_s2: NumOrNull;
}

export interface ReplayFilter {
  increment: string;
  runtime_s: NumOrNull;
  n_imu: NumOrNull;
  n_updates: NumOrNull;
}

export interface ReplayOurs {
  p_ned_m: Row[];
  q: Row[];
  euler_deg: Row[]; // [roll, pitch, yaw]
  sigma3_att_deg: Row[];
  sigma3_pos_m: Row[];
}

export interface ReplayEkf2 {
  p_ned_m: Row[];
  q: Row[];
  euler_deg: Row[];
}

/** ours - EKF2; `null` where EKF2 is not available. */
export interface ReplayDiff {
  pos_h_m: NumOrNull[];
  pos_d_m: NumOrNull[];
  tilt_deg: NumOrNull[];
  yaw_deg: NumOrNull[];
}

export interface ReplayNis {
  t: NumOrNull[];
  nis: NumOrNull[];
  accepted: boolean[];
  dof: NumOrNull;
  band95: [NumOrNull, NumOrNull];
}

export interface ReplayEvents {
  ekf2_resets_s: number[];
  skip_init_s: number;
}

export interface Replay {
  schema_version: number;
  caveat: string;
  rate_hz: number;
  log: ReplayLogMeta;
  filter: ReplayFilter;
  t: number[];
  ours: ReplayOurs;
  ekf2: ReplayEkf2;
  diff: ReplayDiff;
  nis: Record<string, ReplayNis>;
  events: ReplayEvents;
}

/** One row of `GET /api/px4/logs`. */
export interface LogEntry {
  id: string;
  role: string;
  notes: string;
  present: boolean;
  cached: boolean;
}

export interface LogsResponse {
  logs: LogEntry[];
  download_hint: string;
  caveat: string;
}
