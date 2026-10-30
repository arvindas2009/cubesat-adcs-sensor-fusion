// Readout formatting (pure). Missing or non-finite values print as an em dash, never as 0.
export const DASH = '—';

function ok(x: number | null | undefined): x is number {
  return typeof x === 'number' && Number.isFinite(x);
}

/** Fixed decimals, right-aligned to `width` characters (0 = no padding). */
export function fmtFixed(x: number | null | undefined, digits: number, width = 0, signed = false): string {
  if (!ok(x)) return DASH.padStart(width, ' ');
  let s = x.toFixed(digits);
  if (/^-0(\.0+)?$/.test(s)) s = s.slice(1); // no negative zero
  if (signed && !s.startsWith('-')) s = '+' + s;
  return s.padStart(width, ' ');
}

/** Angle in degrees with magnitude-dependent precision (keeps ~4 significant digits). */
export function fmtAngle(x: number | null | undefined): string {
  if (!ok(x)) return DASH;
  const a = Math.abs(x);
  const digits = a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
  return fmtFixed(x, digits);
}

/** Three components, each right-aligned in `width`, separated by two spaces; null prints three dashes. */
export function fmtVec3(v: readonly number[] | null | undefined, digits: number, width: number): string {
  if (!v) return [DASH, DASH, DASH].map((d) => d.padStart(width, ' ')).join('  ');
  return [0, 1, 2].map((i) => fmtFixed(v[i], digits, width, true)).join('  ');
}

/** Seconds -> HH:MM:SS (mission elapsed time). */
export function fmtClock(t: number | null | undefined): string {
  if (!ok(t) || t < 0) return DASH;
  const s = Math.floor(t);
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(hh)}:${p(mm)}:${p(ss)}`;
}

/** Sim speed multiplier, e.g. "500x", "1.5x" (3 significant digits at most). */
export function fmtSpeed(x: number | null | undefined): string {
  if (!ok(x)) return DASH;
  let s = x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return `${s}×`;
}

export function fmtPercent(frac: number | null | undefined, digits = 0): string {
  if (!ok(frac)) return DASH;
  return `${(100 * frac).toFixed(digits)} %`;
}
