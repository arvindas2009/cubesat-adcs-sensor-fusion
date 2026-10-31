// Pure accessibility helpers (docs/specs/ui_console.md §2, §7). DOM-free so vitest can test them in node.
// `nextTab` (tabs), `placeTip` (help tips), the PX4 slider text and keys, and the text alternatives of the timeline and plots.

/**
 * Roving-tabindex arrow handling for a horizontal tab list of `n` tabs with the active index `i`:
 * Right / Left move (wrapping), Home / End jump to the ends. Returns the new index, or null for any other key.
 */
export function nextTab(key: string, i: number, n: number): number | null {
  if (n <= 0) return null;
  switch (key) {
    case 'ArrowRight':
      return (i + 1) % n;
    case 'ArrowLeft':
      return (i - 1 + n) % n;
    case 'Home':
      return 0;
    case 'End':
      return n - 1;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------- tip placement (§6.1)

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface TipPlacement {
  left: number;
  top: number;
  placement: 'below' | 'above';
}

/**
 * Where to put a help tip of size `tip` for an `anchor` in a `viewport`, all in px: below the anchor, flipped
 * above it when it would run off the bottom (and there is room above), and clamped to the viewport edges.
 */
export function placeTip(
  anchor: Rect,
  tip: { width: number; height: number },
  viewport: { width: number; height: number },
  gap = 6,
  margin = 8,
): TipPlacement {
  const maxLeft = Math.max(margin, viewport.width - tip.width - margin);
  const left = Math.min(Math.max(anchor.left, margin), maxLeft);
  const belowTop = anchor.bottom + gap;
  if (belowTop + tip.height <= viewport.height - margin) return { left, top: belowTop, placement: 'below' };
  const aboveTop = anchor.top - gap - tip.height;
  if (aboveTop >= margin) return { left, top: aboveTop, placement: 'above' };
  // neither side has room: stay below, clamped into the viewport
  const top = Math.max(margin, Math.min(belowTop, viewport.height - tip.height - margin));
  return { left, top, placement: 'below' };
}

// ---------------------------------------------------------------------- PX4 timeline slider (§7)

/** `aria-valuetext` of the replay timeline: "12.3 s of 280.0 s". */
export function sliderText(elapsedS: number, durationS: number): string {
  return `${elapsedS.toFixed(1)} s of ${durationS.toFixed(1)} s`;
}

export type SliderMove = { delta: number } | { to: 'start' | 'end' };

/** Keys of the replay timeline slider: Left / Right move 1 s (Shift: 10 s), Home / End jump to the ends. */
export function sliderKey(key: string, shift: boolean): SliderMove | null {
  switch (key) {
    case 'ArrowRight':
      return { delta: shift ? 10 : 1 };
    case 'ArrowLeft':
      return { delta: shift ? -10 : -1 };
    case 'Home':
      return { to: 'start' };
    case 'End':
      return { to: 'end' };
    default:
      return null;
  }
}

// ---------------------------------------------------------------------- text alternatives (§7)

const DASH = '—';

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

function sec(x: number): string {
  return `${Math.round(x)} s`;
}

/** What the ADCS timeline canvas shows, as text (the `aria-label` of its `role="img"`). */
export function timelineSummary(
  info: { t_end_s: number; period_s: number; eclipses_s: readonly (readonly [number, number])[] } | null,
  segments: readonly { mode: number; t0: number; t1: number }[],
  events: { t_init: number | null; switch_times: readonly number[]; fallback_times: readonly number[]; kicks: readonly unknown[] } | null,
  t: number,
): string {
  if (!info) return 'Mission timeline: no session yet.';
  const parts = [`Mission timeline, 0 to ${sec(info.t_end_s)}`];
  const orbit = info.period_s > 0 ? ` (orbit ${(t / info.period_s).toFixed(2)} of ${(info.t_end_s / info.period_s).toFixed(1)})` : '';
  parts.push(`now ${sec(t)}${orbit}`);
  const last = segments[segments.length - 1];
  if (last) parts.push(`mode ${last.mode === 1 ? 'POINTING' : 'DETUMBLE'} since ${sec(last.t0)}`);
  parts.push(plural(info.eclipses_s.length, 'eclipse'));
  if (events) {
    parts.push(events.t_init === null ? 'TRIAD start pending' : `TRIAD start at ${sec(events.t_init)}`);
    parts.push(plural(events.switch_times.length, 'mode switch', 'mode switches'));
    parts.push(plural(events.fallback_times.length, 'fallback'));
    parts.push(plural(events.kicks.length, 'kick'));
  }
  return `${parts.join('. ')}.`;
}

/** Hidden summary of a plot card: its title and the latest value of each series (null or missing: a dash). */
export function chartSummary(title: string, labels: readonly string[], latest: readonly (string | number | null | undefined)[]): string {
  const items = labels.map((l, i) => {
    const v = latest[i];
    return `${l} ${v === null || v === undefined || v === '' ? DASH : v}`;
  });
  return items.length ? `${title}. Latest: ${items.join(', ')}.` : `${title}.`;
}
