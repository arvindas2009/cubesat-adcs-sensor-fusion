import { describe, expect, it } from 'vitest';
import { DEFAULT_SPEED, PlaybackClock, SPEEDS, indexAt, xToTime } from '../src/px4/clock';

describe('PlaybackClock', () => {
  it('starts paused at t0 with speed 1', () => {
    const c = new PlaybackClock(10, 20);
    expect(c.t).toBe(10);
    expect(c.playing).toBe(false);
    expect(c.speed).toBe(1);
    expect(DEFAULT_SPEED).toBe(1);
    expect(SPEEDS).toEqual([0.25, 0.5, 1, 2, 4, 8, 16]);
  });

  it('does not advance while paused', () => {
    const c = new PlaybackClock(0, 10);
    expect(c.tick(1)).toBe(0);
  });

  it('advances speed * dt while playing', () => {
    const c = new PlaybackClock(0, 100);
    c.play();
    expect(c.tick(0.5)).toBeCloseTo(0.5, 12);
    c.speed = 4;
    expect(c.tick(0.5)).toBeCloseTo(2.5, 12);
    c.speed = 0.25;
    expect(c.tick(2)).toBeCloseTo(3, 12);
    expect(c.playing).toBe(true);
  });

  it('clamps at t1 and pauses there', () => {
    const c = new PlaybackClock(0, 10);
    c.speed = 16;
    c.play();
    expect(c.tick(5)).toBe(10);
    expect(c.playing).toBe(false);
    expect(c.tick(5)).toBe(10);
  });

  it('play() at t1 restarts from t0', () => {
    const c = new PlaybackClock(2, 5);
    c.seek(5);
    c.play();
    expect(c.t).toBe(2);
    expect(c.playing).toBe(true);
  });

  it('toggle flips the play state', () => {
    const c = new PlaybackClock(0, 5);
    c.toggle();
    expect(c.playing).toBe(true);
    c.toggle();
    expect(c.playing).toBe(false);
  });

  it('seek clamps to [t0, t1], ignores NaN and keeps the play state', () => {
    const c = new PlaybackClock(3, 8);
    c.seek(-100);
    expect(c.t).toBe(3);
    c.seek(100);
    expect(c.t).toBe(8);
    c.seek(5.5);
    expect(c.t).toBe(5.5);
    c.seek(Number.NaN);
    expect(c.t).toBe(5.5);
    c.play();
    c.seek(4);
    expect(c.playing).toBe(true);
  });

  it('ignores non-positive dt', () => {
    const c = new PlaybackClock(0, 10);
    c.play();
    expect(c.tick(0)).toBe(0);
    expect(c.tick(-1)).toBe(0);
  });
});

describe('indexAt', () => {
  const t = [0, 0.02, 0.04, 0.06, 0.08];

  it('returns the last sample <= t', () => {
    expect(indexAt(t, 0)).toBe(0);
    expect(indexAt(t, 0.01)).toBe(0);
    expect(indexAt(t, 0.02)).toBe(1);
    expect(indexAt(t, 0.0399)).toBe(1);
    expect(indexAt(t, 0.04)).toBe(2);
    expect(indexAt(t, 0.07)).toBe(3);
  });

  it('is 0 below the start and last above the end', () => {
    expect(indexAt(t, -5)).toBe(0);
    expect(indexAt(t, 0.08)).toBe(4);
    expect(indexAt(t, 99)).toBe(4);
  });

  it('handles empty, single-sample and NaN input', () => {
    expect(indexAt([], 1)).toBe(0);
    expect(indexAt([7], 3)).toBe(0);
    expect(indexAt([7], 9)).toBe(0);
    expect(indexAt(t, Number.NaN)).toBe(0);
  });

  it('agrees with a linear scan on a larger grid', () => {
    const grid = Array.from({ length: 1000 }, (_, i) => 100 + i / 50);
    for (const x of [99, 100, 100.001, 105.5, 119.98, 119.99, 120, 130]) {
      let expected = 0;
      for (let i = 0; i < grid.length; i++) if (grid[i] <= x) expected = i;
      expect(indexAt(grid, x)).toBe(expected);
    }
  });
});

describe('xToTime', () => {
  it('maps the pixel span linearly onto the time span and clamps', () => {
    expect(xToTime(12, 12, 212, 100, 150)).toBe(100);
    expect(xToTime(112, 12, 212, 100, 150)).toBe(125);
    expect(xToTime(212, 12, 212, 100, 150)).toBe(150);
    expect(xToTime(-50, 12, 212, 100, 150)).toBe(100);
    expect(xToTime(999, 12, 212, 100, 150)).toBe(150);
    expect(xToTime(5, 10, 10, 3, 4)).toBe(3); // degenerate width
  });
});
