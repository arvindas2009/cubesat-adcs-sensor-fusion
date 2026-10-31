// Pure text of the replay header pills (docs/specs/ui_theme.md §2): role, short id, duration, grid rate and
// reset count as separate chips instead of one middle-dot string. Every value comes from the replay document.
import { DASH, fmtFixed } from '../adcs/format';

export interface MetaInput {
  role: string;
  id: string;
  /** `replay.log.duration_s`, or the span of the time grid when the server did not send it. */
  durationS: number;
  rateHz: number;
  resets: number;
}

export function metaChips(m: MetaInput): string[] {
  return [
    m.role,
    m.id.slice(0, 8),
    `${fmtFixed(m.durationS, 0)} s`,
    `${m.rateHz > 0 ? fmtFixed(m.rateHz, 0) : DASH} Hz grid`,
    `${m.resets} EKF2 reset${m.resets === 1 ? '' : 's'}`,
  ];
}
