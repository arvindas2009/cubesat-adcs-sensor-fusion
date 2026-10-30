// Replay -> plot columns, path polylines, reset markers and readout values (docs/specs/px4_replay.md
// §3.4). Pure and DOM-free; tested on a small hand-built Replay in tests/series.test.ts. No physics: this
// only re-arranges what the server computed (the one derived quantity, height above the start, is
// -(d - d_start) as the spec's readout list defines it).
import type { Columns } from '../adcs/history';
import type { NumOrNull, Replay, Row } from './types';

/** Keep a value only if it is a finite number. */
export function fin(x: NumOrNull | undefined): number | null {
  return typeof x === 'number' && Number.isFinite(x) ? x : null;
}

/** A row as plain numbers, or null if it is missing or has any non-finite element (e.g. EKF2 outside its span). */
export function finiteRow(row: Row | null | undefined): number[] | null {
  if (!row) return null;
  const out: number[] = [];
  for (const x of row) {
    if (typeof x !== 'number' || !Number.isFinite(x)) return null;
    out.push(x);
  }
  return out;
}

const col = (xs: readonly NumOrNull[]): (number | null)[] => xs.map(fin);

// ---------------------------------------------------------------- plots 1, 2: differences

/** Plot 1 (tilt, yaw difference [deg]) and plot 2 (horizontal, vertical position difference [m]); ours - EKF2. */
export function diffColumns(rep: Replay): { att: Columns; pos: Columns } {
  return {
    att: [rep.t.slice(), col(rep.diff.tilt_deg), col(rep.diff.yaw_deg)],
    pos: [rep.t.slice(), col(rep.diff.pos_h_m), col(rep.diff.pos_d_m)],
  };
}

// ---------------------------------------------------------------- plot 4: height above the start

/** First finite down coordinate (NED d) of a position track, or null if there is none. */
export function startDown(p: readonly Row[]): number | null {
  for (const row of p) {
    const d = fin(row?.[2]);
    if (d !== null) return d;
  }
  return null;
}

/** Height above the start [m] = -(d - d_start) per sample, null where d is missing. */
export function heightOf(p: readonly Row[]): (number | null)[] {
  const d0 = startDown(p);
  return p.map((row) => {
    const d = fin(row?.[2]);
    return d === null || d0 === null ? null : d0 - d;
  });
}

/** Plot 4 columns: t, ours, EKF2 (each relative to its own first finite sample). */
export function heightColumns(rep: Replay): Columns {
  return [rep.t.slice(), heightOf(rep.ours.p_ned_m), heightOf(rep.ekf2.p_ned_m)];
}

// ---------------------------------------------------------------- plot 3: NIS

export interface NisPoints {
  t: number[];
  y: number[];
}

export interface NisSeries {
  sensor: string;
  dof: number | null;
  band95: [number, number] | null;
  accepted: NisPoints;
  rejected: NisPoints;
}

/** Per sensor: accepted and rejected samples (samples with a non-finite t or nis are dropped). */
export function nisSeries(rep: Replay): NisSeries[] {
  const out: NisSeries[] = [];
  for (const [sensor, e] of Object.entries(rep.nis)) {
    const accepted: NisPoints = { t: [], y: [] };
    const rejected: NisPoints = { t: [], y: [] };
    const n = Math.min(e.t.length, e.nis.length);
    for (let i = 0; i < n; i++) {
      const t = fin(e.t[i]);
      const y = fin(e.nis[i]);
      if (t === null || y === null) continue;
      const dst = e.accepted[i] ? accepted : rejected;
      dst.t.push(t);
      dst.y.push(y);
    }
    const lo = fin(e.band95?.[0]);
    const hi = fin(e.band95?.[1]);
    out.push({ sensor, dof: fin(e.dof), band95: lo !== null && hi !== null ? [lo, hi] : null, accepted, rejected });
  }
  return out;
}

export interface NisBand {
  dof: number;
  lo: number;
  hi: number;
}

/** One 95 % band per distinct dof, in order of first appearance. */
export function nisBands(rep: Replay): NisBand[] {
  const seen = new Map<number, NisBand>();
  for (const s of nisSeries(rep)) {
    if (s.dof === null || s.band95 === null || seen.has(s.dof)) continue;
    seen.set(s.dof, { dof: s.dof, lo: s.band95[0], hi: s.band95[1] });
  }
  return [...seen.values()];
}

/**
 * Merge series with different sample times onto one shared, sorted, de-duplicated x axis (uPlot needs a
 * common x): every series gets `null` where it has no sample. Each part's `t` must be ascending.
 */
export function alignColumns(parts: readonly NisPoints[]): Columns {
  let total = 0;
  for (const p of parts) total += p.t.length;
  const all = new Float64Array(total);
  let k = 0;
  for (const p of parts) for (const t of p.t) all[k++] = t;
  all.sort();
  const xs: number[] = [];
  for (let i = 0; i < all.length; i++) if (i === 0 || all[i] !== all[i - 1]) xs.push(all[i]);
  const cols: Columns = [xs];
  for (const p of parts) {
    const y: (number | null)[] = new Array(xs.length).fill(null);
    let j = 0;
    for (let i = 0; i < p.t.length; i++) {
      while (xs[j] < p.t[i]) j++;
      y[j] = p.y[i];
    }
    cols.push(y);
  }
  return cols;
}

// ---------------------------------------------------------------- paths

export interface PathSegment {
  /** First sample index of a run of valid points. */
  start: number;
  /** One past the last sample index of the run. */
  end: number;
}

/** Maximal runs of valid points; a point is invalid if it is null/undefined or has a non-finite element. */
export function pathSegments(points: ReadonlyArray<Row | null | undefined>): PathSegment[] {
  const out: PathSegment[] = [];
  let start = -1;
  for (let i = 0; i < points.length; i++) {
    const ok = finiteRow(points[i]) !== null;
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) {
      out.push({ start, end: i });
      start = -1;
    }
  }
  if (start >= 0) out.push({ start, end: points.length });
  return out;
}

// ---------------------------------------------------------------- reset markers

/** First EKF2 reset strictly after `t` (a small tolerance keeps a repeated click moving forward), or null. */
export function nextReset(rep: Replay, t: number, eps = 1e-6): number | null {
  let best: number | null = null;
  for (const r of rep.events.ekf2_resets_s) if (r > t + eps && (best === null || r < best)) best = r;
  return best;
}

/** Last EKF2 reset strictly before `t`, or null. */
export function prevReset(rep: Replay, t: number, eps = 1e-6): number | null {
  let best: number | null = null;
  for (const r of rep.events.ekf2_resets_s) if (r < t - eps && (best === null || r > best)) best = r;
  return best;
}

// ---------------------------------------------------------------- readouts

export interface ReadoutValues {
  t: number;
  /** Fraction of the log [0, 1]. */
  frac: number;
  oursEuler: (number | null)[];
  ekf2Euler: (number | null)[];
  tilt: number | null;
  yaw: number | null;
  posH: number | null;
  posD: number | null;
  /** Largest of the three attitude-error 3-sigma values [deg]. */
  sigma3: number | null;
  heightOurs: number | null;
  heightEkf2: number | null;
}

const startCache = new WeakMap<Replay, { ours: number | null; ekf2: number | null }>();

function starts(rep: Replay): { ours: number | null; ekf2: number | null } {
  let s = startCache.get(rep);
  if (!s) {
    s = { ours: startDown(rep.ours.p_ned_m), ekf2: startDown(rep.ekf2.p_ned_m) };
    startCache.set(rep, s);
  }
  return s;
}

function height(row: Row | undefined, d0: number | null): number | null {
  const d = fin(row?.[2]);
  return d === null || d0 === null ? null : d0 - d;
}

function triple(row: Row | undefined): (number | null)[] {
  return [0, 1, 2].map((k) => fin(row?.[k]));
}

/** All readout values at grid sample `i`; `t` defaults to the sample time (the clock may be between samples). */
export function readoutsAt(rep: Replay, i: number, tOverride?: number): ReadoutValues {
  const n = rep.t.length;
  const j = Math.min(Math.max(0, i), Math.max(0, n - 1));
  const t = tOverride ?? rep.t[j];
  const span = n > 1 ? rep.t[n - 1] - rep.t[0] : 0;
  const s3 = triple(rep.ours.sigma3_att_deg[j]).filter((x): x is number => x !== null);
  const st = starts(rep);
  return {
    t,
    frac: span > 0 ? Math.min(1, Math.max(0, (t - rep.t[0]) / span)) : 0,
    oursEuler: triple(rep.ours.euler_deg[j]),
    ekf2Euler: triple(rep.ekf2.euler_deg[j]),
    tilt: fin(rep.diff.tilt_deg[j]),
    yaw: fin(rep.diff.yaw_deg[j]),
    posH: fin(rep.diff.pos_h_m[j]),
    posD: fin(rep.diff.pos_d_m[j]),
    sigma3: s3.length ? Math.max(...s3) : null,
    heightOurs: height(rep.ours.p_ned_m[j], st.ours),
    heightEkf2: height(rep.ekf2.p_ned_m[j], st.ekf2),
  };
}
