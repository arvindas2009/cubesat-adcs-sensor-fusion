// Pure helpers for the value a bound chart legend shows (docs/specs/ui_console.md §5.1).

export type Column = ArrayLike<number | null | undefined>;

function present(v: number | null | undefined): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Last finite value of a column at index <= `from` (default: the end); null when there is none. */
export function latestValue(col: Column, from = col.length - 1): number | null {
  for (let k = Math.min(from, col.length - 1); k >= 0; k--) {
    const v = col[k];
    if (present(v)) return v;
  }
  return null;
}

/** Index of the last x <= `x` in an ascending column (-1 when `x` precedes all of them). */
export function indexAtOrBefore(xs: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/**
 * The legend value of one series. Hovering: the value at the cursor index (null when that sample is
 * missing). Not hovering: the latest finite sample, or, when `restX` is given (a replay's playback time),
 * the latest finite sample at or before it.
 */
export function legendValue(xs: ArrayLike<number>, col: Column, hover: number | null | undefined, restX?: number | null): number | null {
  if (hover !== null && hover !== undefined) {
    const v = col[hover];
    return present(v) ? v : null;
  }
  if (restX !== null && restX !== undefined && Number.isFinite(restX)) {
    const i = indexAtOrBefore(xs, restX);
    return i < 0 ? null : latestValue(col, i);
  }
  return latestValue(col);
}
