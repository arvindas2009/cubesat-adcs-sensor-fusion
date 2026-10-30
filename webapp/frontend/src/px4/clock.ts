// Playback clock and sample lookup for the PX4 replay (docs/specs/px4_replay.md §3.4). Pure, DOM-free.

/** Playback speeds offered by the transport (multiples of real time). */
export const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4, 8, 16];
export const DEFAULT_SPEED = 1;

export class PlaybackClock {
  /** Current time [s] on the log's own time axis, always within [t0, t1]. */
  t: number;
  playing = false;
  speed = DEFAULT_SPEED;

  constructor(
    readonly t0: number,
    readonly t1: number,
  ) {
    this.t = t0;
  }

  /** Start playing; at the end of the log this restarts from the beginning. */
  play(): void {
    if (this.t >= this.t1) this.t = this.t0;
    this.playing = true;
  }

  pause(): void {
    this.playing = false;
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  /** Jump to `t` (clamped to [t0, t1]); the play state is unchanged. */
  seek(t: number): void {
    if (!Number.isFinite(t)) return;
    this.t = Math.min(this.t1, Math.max(this.t0, t));
  }

  /** Advance by `speed * dtWallSeconds` while playing; stops (paused) at the end. Returns the new time. */
  tick(dtWallSeconds: number): number {
    if (!this.playing || !(dtWallSeconds > 0)) return this.t;
    this.t += this.speed * dtWallSeconds;
    if (this.t >= this.t1) {
      this.t = this.t1;
      this.playing = false;
    }
    return this.t;
  }
}

/**
 * Index of the last sample of the sorted array `tArr` that is <= t (binary search). 0 below the start,
 * last index above the end, 0 for an empty array.
 */
export function indexAt(tArr: ArrayLike<number>, t: number): number {
  const n = tArr.length;
  if (n === 0 || !(t >= tArr[0])) return 0;
  if (t >= tArr[n - 1]) return n - 1;
  let lo = 0; // tArr[lo] <= t
  let hi = n - 1; // tArr[hi] > t
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tArr[mid] <= t) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** Linear map of a pixel x in [x0, x1] to a time in [t0, t1] (clamped); the inverse of the timeline cursor. */
export function xToTime(x: number, x0: number, x1: number, t0: number, t1: number): number {
  if (!(x1 > x0)) return t0;
  const u = Math.min(1, Math.max(0, (x - x0) / (x1 - x0)));
  return t0 + u * (t1 - t0);
}
