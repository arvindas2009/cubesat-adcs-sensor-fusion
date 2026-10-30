import { describe, expect, it } from 'vitest';
import {
  alignColumns,
  diffColumns,
  finiteRow,
  heightColumns,
  nextReset,
  nisBands,
  nisSeries,
  pathSegments,
  prevReset,
  readoutsAt,
} from '../src/px4/series';
import type { Replay } from '../src/px4/types';

const NUL3 = [null, null, null];
const NUL4 = [null, null, null, null];

/** A hand-built replay of 5 samples (10 Hz) following docs/specs/px4_replay.md §1.1; EKF2 is null for i = 0, 1. */
function makeReplay(): Replay {
  return {
    schema_version: 1,
    caveat: 'EKF2 is a reference, not truth',
    rate_hz: 10,
    log: {
      id: 'abcd1234-0000',
      role: 'hover',
      notes: 'test',
      duration_s: 0.4,
      origin_lat_deg: null,
      origin_lon_deg: null,
      gps_utc_start: null,
      declination_deg: null,
      g_m_s2: 9.8,
    },
    filter: { increment: 'x', runtime_s: 1, n_imu: 5, n_updates: 2 },
    t: [100, 100.1, 100.2, 100.3, 100.4],
    ours: {
      p_ned_m: [
        [0, 0, -1],
        [1, 0, -2],
        [2, 1, -3],
        [3, 1, -2],
        [4, 2, -1],
      ],
      q: [1, 2, 3, 4, 5].map(() => [1, 0, 0, 0]),
      euler_deg: [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 2, 6],
        [1, 2, 7],
      ],
      sigma3_att_deg: [
        [0.5, 0.2, 0.9],
        [0.4, 0.2, 0.8],
        [0.3, 0.2, 0.7],
        [0.3, 0.2, 0.6],
        [0.1, 0.4, 0.2],
      ],
      sigma3_pos_m: [1, 2, 3, 4, 5].map(() => [0.1, 0.1, 0.2]),
    },
    ekf2: {
      p_ned_m: [NUL3, NUL3, [2, 1, -2.5], [3, 1, -1.5], [4, 2, -0.5]],
      q: [NUL4, NUL4, [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
      euler_deg: [NUL3, NUL3, [1, 2, 4], [1, 2, 5], [1, 2, 6]],
    },
    diff: {
      pos_h_m: [null, null, 0, 0, 0],
      pos_d_m: [null, null, -0.5, -0.5, -0.5],
      tilt_deg: [null, null, 0.1, 0.2, 0.3],
      yaw_deg: [null, null, 1, 1, 1],
    },
    nis: {
      gps_pos: {
        t: [100.0, 100.2, 100.4],
        nis: [2.5, 40, 3],
        accepted: [true, false, true],
        dof: 3,
        band95: [0.2158, 9.3484],
      },
      baro: {
        t: [100.1, 100.2],
        nis: [0.5, 1.5],
        accepted: [true, true],
        dof: 1,
        band95: [0.000982, 5.0239],
      },
      mag: {
        t: [100.3],
        nis: [null],
        accepted: [true],
        dof: 3,
        band95: [0.2158, 9.3484],
      },
    },
    events: { ekf2_resets_s: [100.25, 100.1], skip_init_s: 5 },
  };
}

describe('diffColumns', () => {
  it('returns aligned [t, tilt, yaw] and [t, horizontal, vertical] columns with nulls kept', () => {
    const rep = makeReplay();
    const { att, pos } = diffColumns(rep);
    expect(att).toEqual([rep.t, [null, null, 0.1, 0.2, 0.3], [null, null, 1, 1, 1]]);
    expect(pos).toEqual([rep.t, [null, null, 0, 0, 0], [null, null, -0.5, -0.5, -0.5]]);
    expect(att[0]).not.toBe(rep.t); // a copy
  });
});

describe('heightColumns', () => {
  it('is -(d - d_start) for each track relative to its own first finite sample', () => {
    const cols = heightColumns(makeReplay());
    expect(cols[0]).toEqual([100, 100.1, 100.2, 100.3, 100.4]);
    expect(cols[1]).toEqual([0, 1, 2, 1, 0]); // ours: d = -1, -2, -3, -2, -1
    expect(cols[2]).toEqual([null, null, 0, -1, -2]); // EKF2: d0 = -2.5, it descends
  });

  it('is all null for a track without finite samples', () => {
    const rep = makeReplay();
    rep.ekf2.p_ned_m = rep.ekf2.p_ned_m.map(() => NUL3);
    expect(heightColumns(rep)[2]).toEqual([null, null, null, null, null]);
  });
});

describe('nisSeries and nisBands', () => {
  it('splits accepted and rejected samples and drops non-finite ones', () => {
    const s = nisSeries(makeReplay());
    expect(s.map((x) => x.sensor)).toEqual(['gps_pos', 'baro', 'mag']);
    const gps = s[0];
    expect(gps.accepted).toEqual({ t: [100.0, 100.4], y: [2.5, 3] });
    expect(gps.rejected).toEqual({ t: [100.2], y: [40] });
    expect(gps.dof).toBe(3);
    expect(gps.band95).toEqual([0.2158, 9.3484]);
    expect(s[2].accepted).toEqual({ t: [], y: [] }); // the null NIS sample
  });

  it('lists one band per distinct dof', () => {
    expect(nisBands(makeReplay())).toEqual([
      { dof: 3, lo: 0.2158, hi: 9.3484 },
      { dof: 1, lo: 0.000982, hi: 5.0239 },
    ]);
  });
});

describe('alignColumns', () => {
  it('merges times, de-duplicates them and fills gaps with null', () => {
    const cols = alignColumns([
      { t: [1, 3, 5], y: [10, 30, 50] },
      { t: [2, 3], y: [20, 31] },
      { t: [], y: [] },
    ]);
    expect(cols[0]).toEqual([1, 2, 3, 5]);
    expect(cols[1]).toEqual([10, null, 30, 50]);
    expect(cols[2]).toEqual([null, 20, 31, null]);
    expect(cols[3]).toEqual([null, null, null, null]);
  });

  it('handles no parts', () => {
    expect(alignColumns([])).toEqual([[]]);
  });
});

describe('pathSegments', () => {
  it('splits at null rows and at rows with a null element', () => {
    const pts = [[0, 0, 0], [1, 0, 0], null, [2, 0, 0], [null, 0, 0], [3, 0, 0], [4, 0, 0]];
    expect(pathSegments(pts)).toEqual([
      { start: 0, end: 2 },
      { start: 3, end: 4 },
      { start: 5, end: 7 },
    ]);
  });

  it('keeps a fully valid path as one segment and a fully null path as none', () => {
    expect(pathSegments([[0, 0, 0], [1, 1, 1]])).toEqual([{ start: 0, end: 2 }]);
    expect(pathSegments([NUL3, NUL3])).toEqual([]);
    expect(pathSegments([])).toEqual([]);
  });

  it('works on the replay EKF2 track', () => {
    expect(pathSegments(makeReplay().ekf2.p_ned_m)).toEqual([{ start: 2, end: 5 }]);
  });
});

describe('finiteRow', () => {
  it('returns numbers or null', () => {
    expect(finiteRow([1, 2, 3])).toEqual([1, 2, 3]);
    expect(finiteRow([1, null, 3])).toBeNull();
    expect(finiteRow(null)).toBeNull();
    expect(finiteRow([1, Number.NaN])).toBeNull();
  });
});

describe('nextReset / prevReset', () => {
  const rep = makeReplay(); // resets are given unsorted: 100.25, 100.1
  it('find the neighbouring reset strictly after / before t', () => {
    expect(nextReset(rep, 100)).toBe(100.1);
    expect(nextReset(rep, 100.1)).toBe(100.25); // strictly after
    expect(nextReset(rep, 100.25)).toBeNull();
    expect(prevReset(rep, 100.4)).toBe(100.25);
    expect(prevReset(rep, 100.25)).toBe(100.1); // strictly before
    expect(prevReset(rep, 100.1)).toBeNull();
    expect(prevReset(rep, 99)).toBeNull();
  });
});

describe('readoutsAt', () => {
  it('collects the values of one grid sample', () => {
    const r = readoutsAt(makeReplay(), 3);
    expect(r.t).toBe(100.3);
    expect(r.frac).toBeCloseTo(0.75, 12);
    expect(r.oursEuler).toEqual([1, 2, 6]);
    expect(r.ekf2Euler).toEqual([1, 2, 5]);
    expect(r.tilt).toBe(0.2);
    expect(r.yaw).toBe(1);
    expect(r.posH).toBe(0);
    expect(r.posD).toBe(-0.5);
    expect(r.sigma3).toBe(0.6);
    expect(r.heightOurs).toBe(1);
    expect(r.heightEkf2).toBe(-1);
  });

  it('shows missing EKF2 values as null and uses the max of the three 3-sigma values', () => {
    const r = readoutsAt(makeReplay(), 0);
    expect(r.ekf2Euler).toEqual([null, null, null]);
    expect(r.tilt).toBeNull();
    expect(r.yaw).toBeNull();
    expect(r.posH).toBeNull();
    expect(r.posD).toBeNull();
    expect(r.heightEkf2).toBeNull();
    expect(r.heightOurs).toBe(0);
    expect(r.sigma3).toBe(0.9);
    expect(r.frac).toBe(0);
  });

  it('takes the clock time for t and clamps the index', () => {
    const rep = makeReplay();
    expect(readoutsAt(rep, 99).t).toBe(100.4);
    expect(readoutsAt(rep, 99).frac).toBe(1);
    expect(readoutsAt(rep, 1, 100.15).t).toBe(100.15);
    expect(readoutsAt(rep, 1, 100.15).frac).toBeCloseTo(0.375, 12);
    expect(readoutsAt(rep, -4).t).toBe(100);
  });
});
