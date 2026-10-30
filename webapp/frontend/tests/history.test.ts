import { describe, expect, it } from 'vitest';
import { History, eclipseFraction, modeSegments, sampleFromFrame } from '../src/adcs/history';
import type { Frame } from '../src/types';

interface Over {
  t: number;
  nadir?: number;
  est?: number | null;
  know?: number | null;
  sigma?: number | null;
  rate?: number;
  rpm?: [number, number, number];
}

function mkFrame(o: Over): Frame {
  return {
    metrics: {
      nadir_err_deg: o.nadir ?? 10,
      est_nadir_err_deg: o.est === undefined ? 9 : o.est,
      know_err_deg: o.know === undefined ? 1 : o.know,
      att_sigma_deg: o.sigma === undefined ? 0.5 : o.sigma,
      rate_deg_s: o.rate ?? 2,
      wheel_speed_rpm: o.rpm ?? [100, -200, 300],
    },
    t: o.t,
  } as unknown as Frame;
}

describe('sampleFromFrame', () => {
  it('maps metrics, with 3 sigma = 3 x att_sigma_deg', () => {
    const s = sampleFromFrame(mkFrame({ t: 5, sigma: 0.4 }));
    expect(s).toEqual({
      t: 5, nadir: 10, estNadir: 9, know: 1, sigma3: 3 * 0.4, rate: 2, wheel0: 100, wheel1: -200, wheel2: 300,
    });
  });

  it('keeps null as null (not zero) and maps non-finite numbers to null', () => {
    const s = sampleFromFrame(mkFrame({ t: 1, est: null, know: null, sigma: null, nadir: NaN }));
    expect(s.estNadir).toBeNull();
    expect(s.know).toBeNull();
    expect(s.sigma3).toBeNull();
    expect(s.nadir).toBeNull();
  });
});

describe('History ring buffer', () => {
  it('rejects a bad capacity', () => {
    expect(() => new History(0)).toThrow();
    expect(() => new History(2.5)).toThrow();
  });

  it('is empty at the start, with aligned empty columns', () => {
    const h = new History(4);
    expect(h.length).toBe(0);
    const c = h.columns();
    expect(c.length).toBe(9);
    c.forEach((col) => expect(col).toEqual([]));
  });

  it('grows up to capacity in insertion order', () => {
    const h = new History(5);
    for (let i = 0; i < 3; i++) h.push(mkFrame({ t: i }));
    expect(h.length).toBe(3);
    expect(h.columns()[0]).toEqual([0, 1, 2]);
  });

  it('wraps around, dropping the oldest and keeping order', () => {
    const h = new History(4);
    for (let i = 0; i < 11; i++) h.push(mkFrame({ t: i, nadir: 100 + i }));
    expect(h.length).toBe(4);
    const c = h.columns();
    expect(c[0]).toEqual([7, 8, 9, 10]);
    expect(c[1]).toEqual([107, 108, 109, 110]);
  });

  it('exactly full, then one more push', () => {
    const h = new History(3);
    for (let i = 0; i < 3; i++) h.push(mkFrame({ t: i }));
    expect(h.columns()[0]).toEqual([0, 1, 2]);
    h.push(mkFrame({ t: 3 }));
    expect(h.columns()[0]).toEqual([1, 2, 3]);
  });

  it('columns are aligned: every series has one entry per time stamp, in uPlot order', () => {
    const h = new History(6);
    for (let i = 0; i < 9; i++) h.push(mkFrame({ t: i, rpm: [i, 2 * i, 3 * i], rate: i / 10 }));
    const c = h.columns();
    expect(c.length).toBe(9); // t + 8 series
    const n = c[0].length;
    expect(n).toBe(6);
    c.forEach((col) => expect(col.length).toBe(n));
    // order: t, nadir, estNadir, know, sigma3, rate, wheel0, wheel1, wheel2
    expect(c[5]).toEqual([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    expect(c[6]).toEqual([3, 4, 5, 6, 7, 8]);
    expect(c[7]).toEqual([6, 8, 10, 12, 14, 16]);
    expect(c[8]).toEqual([9, 12, 15, 18, 21, 24]);
  });

  it('keeps null values as gaps in the columns, also after wrap-around', () => {
    const h = new History(3);
    h.push(mkFrame({ t: 0, est: null, know: null, sigma: null }));
    h.push(mkFrame({ t: 1 }));
    h.push(mkFrame({ t: 2, est: null }));
    h.push(mkFrame({ t: 3, know: null, sigma: null }));
    const c = h.columns();
    expect(c[0]).toEqual([1, 2, 3]);
    expect(c[2]).toEqual([9, null, 9]); // estNadir
    expect(c[3]).toEqual([1, 1, null]); // know
    expect(c[4]).toEqual([1.5, 1.5, null]); // sigma3
  });

  it('clear() empties the buffer and it can be reused', () => {
    const h = new History(3);
    for (let i = 0; i < 5; i++) h.push(mkFrame({ t: i }));
    h.clear();
    expect(h.length).toBe(0);
    expect(h.columns()[0]).toEqual([]);
    h.push(mkFrame({ t: 42 }));
    expect(h.columns()[0]).toEqual([42]);
  });

  it('works with capacity 1', () => {
    const h = new History(1);
    h.push(mkFrame({ t: 1 }));
    h.push(mkFrame({ t: 2 }));
    expect(h.columns()[0]).toEqual([2]);
  });
});

describe('modeSegments', () => {
  it('returns nothing for no samples', () => {
    expect(modeSegments([])).toEqual([]);
  });

  it('merges consecutive equal modes into one segment', () => {
    const segs = modeSegments([
      { t: 0, mode: 0 },
      { t: 1, mode: 0 },
      { t: 2, mode: 0 },
    ]);
    expect(segs).toEqual([{ mode: 0, t0: 0, t1: 2 }]);
  });

  it('splits at a mode change; segments abut', () => {
    const segs = modeSegments([
      { t: 0, mode: 0 },
      { t: 10, mode: 0 },
      { t: 20, mode: 1 },
      { t: 30, mode: 1 },
      { t: 40, mode: 0 },
      { t: 50, mode: 0 },
    ]);
    expect(segs).toEqual([
      { mode: 0, t0: 0, t1: 20 },
      { mode: 1, t0: 20, t1: 40 },
      { mode: 0, t0: 40, t1: 50 },
    ]);
  });

  it('handles a single sample', () => {
    expect(modeSegments([{ t: 7, mode: 1 }])).toEqual([{ mode: 1, t0: 7, t1: 7 }]);
  });
});

describe('eclipseFraction', () => {
  it('is the covered share of [0, t_end]', () => {
    expect(eclipseFraction([[100, 200], [500, 600]], 1000)).toBeCloseTo(0.2, 12);
  });

  it('is 0 without eclipses or for a non-positive horizon', () => {
    expect(eclipseFraction([], 1000)).toBe(0);
    expect(eclipseFraction([[1, 2]], 0)).toBe(0);
  });

  it('clips intervals to the horizon', () => {
    expect(eclipseFraction([[900, 1200]], 1000)).toBeCloseTo(0.1, 12);
    expect(eclipseFraction([[-50, 50]], 1000)).toBeCloseTo(0.05, 12);
  });
});
