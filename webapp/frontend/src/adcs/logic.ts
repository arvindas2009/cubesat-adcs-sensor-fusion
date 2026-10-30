// Pure, DOM-free logic behind the ADCS page (tested in tests/logic.test.ts): form validation,
// kick-command building, speed-slider mapping, timeline geometry, log-axis ranges, throttles.
import type { Axis, Command, Frame, SchemaProperty, SessionParams } from '../types';
import { modeSegments, type ModeSegment } from './history';

// ---------------------------------------------------------------- parameter form

export type FieldKind = 'number' | 'integer' | 'enum' | 'boolean';

export interface FieldSpec {
  name: string;
  label: string;
  kind: FieldKind;
  min?: number;
  max?: number;
  exclusiveMin?: number;
  exclusiveMax?: number;
  options?: string[];
  def: unknown;
  /** Short constraint text shown next to the field, e.g. "0.05 – 15". */
  hint: string;
}

export const FIELD_LABELS: Record<string, string> = {
  orbits: 'Orbits',
  rate_deg_s: 'Initial rate [°/s]',
  seed: 'Seed',
  sensors: 'Sensors',
  filter: 'Filter',
  dumping: 'Momentum dumping',
  inertia_scale: 'Inertia scale',
};

function numText(x: number): string {
  return String(x);
}

function hintFor(kind: FieldKind, p: SchemaProperty): string {
  if (kind !== 'number' && kind !== 'integer') return '';
  const lo = p.minimum ?? p.exclusiveMinimum;
  const hi = p.maximum ?? p.exclusiveMaximum;
  const range = lo !== undefined && hi !== undefined ? `${numText(lo)} – ${numText(hi)}` : '';
  return kind === 'integer' ? (range ? `integer ${range}` : 'integer') : range;
}

/** Field descriptions in schema order (pydantic keeps the order of protocol §1). */
export function fieldSpecs(props: Record<string, SchemaProperty>): FieldSpec[] {
  const out: FieldSpec[] = [];
  for (const [name, p] of Object.entries(props)) {
    let kind: FieldKind;
    if (Array.isArray(p.enum)) kind = 'enum';
    else if (p.type === 'boolean') kind = 'boolean';
    else if (p.type === 'integer') kind = 'integer';
    else if (p.type === 'number') kind = 'number';
    else continue; // not representable in this form; the server default applies
    out.push({
      name,
      label: FIELD_LABELS[name] ?? p.title ?? name,
      kind,
      min: p.minimum,
      max: p.maximum,
      exclusiveMin: p.exclusiveMinimum,
      exclusiveMax: p.exclusiveMaximum,
      options: kind === 'enum' ? (p.enum as unknown[]).map(String) : undefined,
      def: p.default,
      hint: hintFor(kind, p),
    });
  }
  return out;
}

export type FieldResult = { ok: true; value: number | string | boolean } | { ok: false; error: string };

/** Validate one raw form value (a string from an input, or a boolean from a checkbox). */
export function validateField(spec: FieldSpec, raw: string | boolean): FieldResult {
  switch (spec.kind) {
    case 'boolean':
      return { ok: true, value: Boolean(raw) };
    case 'enum': {
      const s = String(raw);
      return spec.options?.includes(s) ? { ok: true, value: s } : { ok: false, error: 'choose one of the options' };
    }
    default: {
      const s = String(raw).trim();
      if (s === '') return { ok: false, error: 'required' };
      const x = Number(s);
      if (!Number.isFinite(x)) return { ok: false, error: 'not a number' };
      if (spec.kind === 'integer' && !Number.isInteger(x)) return { ok: false, error: 'must be an integer' };
      if (spec.min !== undefined && x < spec.min) return { ok: false, error: `must be at least ${numText(spec.min)}` };
      if (spec.max !== undefined && x > spec.max) return { ok: false, error: `must be at most ${numText(spec.max)}` };
      if (spec.exclusiveMin !== undefined && x <= spec.exclusiveMin)
        return { ok: false, error: `must be greater than ${numText(spec.exclusiveMin)}` };
      if (spec.exclusiveMax !== undefined && x >= spec.exclusiveMax)
        return { ok: false, error: `must be less than ${numText(spec.exclusiveMax)}` };
      return { ok: true, value: x };
    }
  }
}

export interface FormValidation {
  values: Partial<SessionParams>;
  errors: Record<string, string>;
  valid: boolean;
}

export function validateAll(specs: readonly FieldSpec[], raw: Record<string, string | boolean>): FormValidation {
  const values: Record<string, unknown> = {};
  const errors: Record<string, string> = {};
  for (const spec of specs) {
    const r = validateField(spec, raw[spec.name] ?? '');
    if (r.ok) values[spec.name] = r.value;
    else errors[spec.name] = r.error;
  }
  return { values: values as Partial<SessionParams>, errors, valid: Object.keys(errors).length === 0 };
}

// ---------------------------------------------------------------- kick

export const KICK_MAX_DEG_S = 20;
export const KICK_DEFAULT_DEG_S = 5;

export type KickResult = { ok: true; cmd: Extract<Command, { cmd: 'kick' }> } | { ok: false; error: string };

/**
 * Build the `kick` command (body axes, signed magnitude in deg/s). The server validates again and
 * enforces the 30 deg/s total-rate cap [B-25]; its error comes back as a toast.
 */
export function buildKickCommand(axis: Axis, sign: 1 | -1, magnitudeDegS: number): KickResult {
  if (axis !== 'x' && axis !== 'y' && axis !== 'z') return { ok: false, error: 'axis must be x, y or z' };
  if (!Number.isFinite(magnitudeDegS)) return { ok: false, error: 'magnitude is not a number' };
  if (magnitudeDegS <= 0) return { ok: false, error: 'magnitude must be greater than 0' };
  if (magnitudeDegS > KICK_MAX_DEG_S) return { ok: false, error: `magnitude must be at most ${KICK_MAX_DEG_S} °/s` };
  return { ok: true, cmd: { cmd: 'kick', axis, deg_s: sign * magnitudeDegS } };
}

// ---------------------------------------------------------------- speed slider (log scale)

export const SPEED_MIN = 1;
export const SPEED_MAX = 2000;
export const SLIDER_MAX = 1000;

function round3(x: number): number {
  if (x === 0) return 0;
  const m = 10 ** (2 - Math.floor(Math.log10(Math.abs(x))));
  return Math.round(x * m) / m;
}

/** Slider position 0..SLIDER_MAX -> speed 1..2000 on a log scale, rounded to 3 significant digits. */
export function sliderToSpeed(pos: number): number {
  const u = Math.min(1, Math.max(0, pos / SLIDER_MAX));
  const v = SPEED_MIN * (SPEED_MAX / SPEED_MIN) ** u;
  return Math.min(SPEED_MAX, Math.max(SPEED_MIN, round3(v)));
}

/** Inverse of `sliderToSpeed` (continuous, not rounded). */
export function speedToSlider(speed: number): number {
  const s = Math.min(SPEED_MAX, Math.max(SPEED_MIN, speed));
  return (Math.log(s / SPEED_MIN) / Math.log(SPEED_MAX / SPEED_MIN)) * SLIDER_MAX;
}

// ---------------------------------------------------------------- timeline geometry

/** Map a time in [0, tEnd] to x in [x0, x1] (clamped). */
export function timeToX(t: number, tEnd: number, x0: number, x1: number): number {
  if (!(tEnd > 0)) return x0;
  const u = Math.min(1, Math.max(0, t / tEnd));
  return x0 + u * (x1 - x0);
}

/** Rectangle of an interval [a, b]; at least `minWidth` px wide so short events stay visible. */
export function intervalRect(
  a: number,
  b: number,
  tEnd: number,
  x0: number,
  x1: number,
  minWidth = 1,
): { x: number; w: number } {
  const xa = timeToX(a, tEnd, x0, x1);
  const xb = timeToX(b, tEnd, x0, x1);
  return { x: xa, w: Math.max(minWidth, xb - xa) };
}

/** Orbit boundaries inside (0, tEnd): n = 1, 2, ... at t = n x period. */
export function orbitTicks(tEnd: number, period: number): { t: number; n: number }[] {
  const out: { t: number; n: number }[] = [];
  if (!(period > 0) || !(tEnd > 0)) return out;
  for (let n = 1; n * period < tEnd; n++) out.push({ t: n * period, n });
  return out;
}

/**
 * Compact record of the mode over time: only samples where the mode changes are stored (plus the
 * latest time), so memory does not grow with run length. `segments()` goes through `modeSegments`.
 */
export class ModeTrack {
  private pts: { t: number; mode: number }[] = [];
  private lastT = 0;
  private lastMode = -1;

  push(t: number, mode: number): void {
    if (t < this.lastT) this.clear(); // time went backwards: a reset we did not hear about
    if (mode !== this.lastMode) this.pts.push({ t, mode });
    this.lastT = t;
    this.lastMode = mode;
  }

  segments(): ModeSegment[] {
    if (this.pts.length === 0) return [];
    return modeSegments([...this.pts, { t: this.lastT, mode: this.lastMode }]);
  }

  clear(): void {
    this.pts = [];
    this.lastT = 0;
    this.lastMode = -1;
  }
}

// ---------------------------------------------------------------- plots

/** Values at or below `floor` are drawn at the floor on a log axis (they cannot be placed); null stays null. */
export function clampLog(v: number | null, floor: number): number | null {
  return v === null ? null : v < floor ? floor : v;
}

/** Smallest of 1, 2, 5 x 10^n that is >= x (x > 0). */
function niceCeil(x: number): number {
  const e = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 5, 10]) if (m * e >= x * (1 - 1e-12)) return m * e;
  return 10 * e;
}

/**
 * y range for a log axis from the data extremes (plus any threshold lines): the lower bound is
 * the decade below the smallest value, the upper bound the next 1-2-5 value above the largest.
 * Never below `floor`; always at least one decade tall; a default when there is no data.
 */
export function logRange(
  dmin: number | null | undefined,
  dmax: number | null | undefined,
  floor: number,
  extras: readonly number[] = [],
): [number, number] {
  const vals = [dmin, dmax, ...extras].filter((x): x is number => typeof x === 'number' && Number.isFinite(x) && x > 0);
  if (vals.length === 0) return [Math.max(floor, 0.01), 1000];
  const lo = Math.max(floor, Math.min(...vals));
  const hi = Math.max(...vals);
  const a = Math.max(floor, 10 ** Math.floor(Math.log10(lo)));
  let b = niceCeil(hi);
  if (b < a * 10) b = a * 10;
  return [a, b];
}

/** x range of the plots: from the first retained sample to the run horizon (aligned with the timeline). */
export function timeRange(first: number | null | undefined, last: number | null | undefined, tEnd: number): [number, number] {
  const lo = typeof first === 'number' && Number.isFinite(first) ? first : 0;
  let hi = Math.max(tEnd, typeof last === 'number' && Number.isFinite(last) ? last : 0);
  if (!(hi > lo)) hi = lo + 1;
  return [lo, hi];
}

// ---------------------------------------------------------------- scheduling

/** Keeps only the newest frame between two animation frames (every frame still goes to History). */
export class FrameLatch {
  private pending: Frame | null = null;
  /** Number of frames replaced before being rendered. */
  dropped = 0;

  offer(f: Frame): void {
    if (this.pending) this.dropped++;
    this.pending = f;
  }

  take(): Frame | null {
    const f = this.pending;
    this.pending = null;
    return f;
  }
}

/** Rate limiter: `shouldRun` is true at most once per `minIntervalMs`, or always when forced. */
export class Throttle {
  private last = -Infinity;
  constructor(readonly minIntervalMs: number) {}

  shouldRun(nowMs: number, force = false): boolean {
    if (force || nowMs - this.last >= this.minIntervalMs || nowMs < this.last) {
      this.last = nowMs;
      return true;
    }
    return false;
  }
}

// ---------------------------------------------------------------- close codes

export const CLOSE_REPLACED = 4001;
export const CLOSE_NOT_FOUND = 4404;

/** Banner text for a WebSocket close code. */
export function closeBanner(code: number): string {
  if (code === CLOSE_REPLACED)
    return 'This session was taken over by another connection (code 4001, replaced). Start a new session to continue.';
  if (code === CLOSE_NOT_FOUND)
    return 'Session not found (code 4404): the server restarted or the session was evicted. Start a new session.';
  return `Connection to the server was lost (code ${code}). Start a new session.`;
}
