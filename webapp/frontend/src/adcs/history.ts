// Ring buffer of frame samples for the plots, plus mode-timeline helpers
// (docs/specs/frontend.md §3). Pure, DOM-free.
import type { Frame } from '../types';

export interface Sample {
  t: number;
  nadir: number | null;
  estNadir: number | null;
  know: number | null;
  sigma3: number | null;
  rate: number | null;
  wheel0: number | null;
  wheel1: number | null;
  wheel2: number | null;
}

/** Columns in uPlot's aligned-data order: t, then the series below, all the same length. */
export type Columns = (number | null)[][];

/** Series order of `columns()` after the time column. */
export const SERIES = ['nadir', 'estNadir', 'know', 'sigma3', 'rate', 'wheel0', 'wheel1', 'wheel2'] as const;

function fin(x: number | null | undefined): number | null {
  return x == null || !Number.isFinite(x) ? null : x;
}

/** A finite value, or null. The 3-sigma bound is 3 x att_sigma_deg. */
export function sampleFromFrame(f: Frame): Sample {
  const m = f.metrics;
  const sig = fin(m.att_sigma_deg);
  return {
    t: f.t,
    nadir: fin(m.nadir_err_deg),
    estNadir: fin(m.est_nadir_err_deg),
    know: fin(m.know_err_deg),
    sigma3: sig === null ? null : 3 * sig,
    rate: fin(m.rate_deg_s),
    wheel0: fin(m.wheel_speed_rpm[0]),
    wheel1: fin(m.wheel_speed_rpm[1]),
    wheel2: fin(m.wheel_speed_rpm[2]),
  };
}

export class History {
  private readonly buf: Sample[];
  private head = 0; // index of the oldest sample once full
  private n = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error('capacity must be a positive integer');
    this.buf = new Array<Sample>(capacity);
  }

  get length(): number {
    return this.n;
  }

  /** Append one frame's sample, overwriting the oldest when full. */
  push(frame: Frame): void {
    const s = sampleFromFrame(frame);
    if (this.n < this.capacity) {
      this.buf[(this.head + this.n) % this.capacity] = s;
      this.n++;
    } else {
      this.buf[this.head] = s;
      this.head = (this.head + 1) % this.capacity;
    }
  }

  /** Samples oldest to newest. */
  samples(): Sample[] {
    const out: Sample[] = [];
    for (let i = 0; i < this.n; i++) out.push(this.buf[(this.head + i) % this.capacity]);
    return out;
  }

  /** uPlot aligned data: [t, nadir, estNadir, know, sigma3, rate, wheel0, wheel1, wheel2]. Nulls stay null (gaps). */
  columns(): Columns {
    const cols: Columns = Array.from({ length: 1 + SERIES.length }, () => new Array<number | null>(this.n));
    for (let i = 0; i < this.n; i++) {
      const s = this.buf[(this.head + i) % this.capacity];
      cols[0][i] = s.t;
      for (let j = 0; j < SERIES.length; j++) cols[j + 1][i] = s[SERIES[j]];
    }
    return cols;
  }

  clear(): void {
    this.head = 0;
    this.n = 0;
  }
}

export interface ModeSegment {
  mode: number;
  t0: number;
  t1: number;
}

/**
 * Merge consecutive samples with equal mode into segments [t0, t1] (first and last sample time).
 * A change of mode closes the previous segment at the new sample's time, so segments abut.
 */
export function modeSegments(samples: readonly { t: number; mode: number }[]): ModeSegment[] {
  const out: ModeSegment[] = [];
  for (const s of samples) {
    const last = out[out.length - 1];
    if (last && last.mode === s.mode) {
      last.t1 = s.t;
    } else {
      if (last) last.t1 = s.t;
      out.push({ mode: s.mode, t0: s.t, t1: s.t });
    }
  }
  return out;
}

/** Fraction of [0, t_end] spent in eclipse; intervals are clipped to the horizon and may not overlap. */
export function eclipseFraction(eclipses: readonly (readonly [number, number])[], t_end: number): number {
  if (!(t_end > 0)) return 0;
  let total = 0;
  for (const [a, b] of eclipses) {
    const lo = Math.max(0, a);
    const hi = Math.min(t_end, b);
    if (hi > lo) total += hi - lo;
  }
  return total / t_end;
}
