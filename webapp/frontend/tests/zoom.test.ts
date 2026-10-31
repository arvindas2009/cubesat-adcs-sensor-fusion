// Pure tests of the chart zoom bookkeeping (docs/specs/ui_console.md §5.2, §8) and of the small pure helpers
// the bound legends and the NIS tags use.
import { describe, expect, it } from 'vitest';
import { mergeBandTags } from '../src/px4/tags';
import { indexAtOrBefore, latestValue, legendValue } from '../src/ui/sample';
import { XZoom } from '../src/ui/zoom';

describe('XZoom', () => {
  it('starts inactive and returns the default range', () => {
    const z = new XZoom();
    expect(z.active).toBe(false);
    expect(z.range([0, 100])).toEqual([0, 100]);
  });

  it('select zooms to the selection and range returns it', () => {
    const z = new XZoom();
    z.select(20, 40, 0, 100, 1);
    expect(z.active).toBe(true);
    expect([z.min, z.max]).toEqual([20, 40]);
    expect(z.range([0, 100])).toEqual([20, 40]);
  });

  it('accepts a selection dragged right to left', () => {
    const z = new XZoom();
    z.select(40, 20, 0, 100, 1);
    expect([z.min, z.max]).toEqual([20, 40]);
  });

  it('clamps to the data span', () => {
    const z = new XZoom();
    z.select(-50, 30, 0, 100, 1);
    expect([z.min, z.max]).toEqual([0, 30]);
    z.select(70, 400, 0, 100, 1);
    expect([z.min, z.max]).toEqual([70, 100]);
    z.select(-5, 500, 10, 90, 1);
    expect([z.min, z.max]).toEqual([10, 90]);
  });

  it('keeps the minimum span, widening around the centre', () => {
    const z = new XZoom();
    z.select(50, 50.2, 0, 100, 4);
    expect(z.max - z.min).toBeCloseTo(4, 12);
    expect((z.min + z.max) / 2).toBeCloseTo(50.1, 12);
  });

  it('shifts a too-narrow selection at an edge to stay inside the data', () => {
    const z = new XZoom();
    z.select(0, 0.1, 0, 100, 4);
    expect([z.min, z.max]).toEqual([0, 4]);
    z.select(99.9, 100, 0, 100, 4);
    expect([z.min, z.max]).toEqual([96, 100]);
  });

  it('never asks for more than the data span', () => {
    const z = new XZoom();
    z.select(1, 2, 0, 10, 50);
    expect([z.min, z.max]).toEqual([0, 10]);
  });

  it('ignores non-finite input and empty data', () => {
    const z = new XZoom();
    z.select(Number.NaN, 5, 0, 10, 1);
    z.select(1, 5, 0, 0, 1);
    expect(z.active).toBe(false);
  });

  it('reset deactivates and range returns the default again', () => {
    const z = new XZoom();
    z.select(10, 20, 0, 100, 1);
    z.reset();
    expect(z.active).toBe(false);
    expect(z.range([0, 100])).toEqual([0, 100]);
  });
});

describe('legend sample helpers', () => {
  const xs = [0, 1, 2, 3, 4];

  it('latestValue skips missing samples from the end', () => {
    expect(latestValue([1, 2, 3])).toBe(3);
    expect(latestValue([1, 2, null, null])).toBe(2);
    expect(latestValue([null, undefined])).toBeNull();
    expect(latestValue([])).toBeNull();
    expect(latestValue([1, 2, 3], 1)).toBe(2);
    expect(latestValue([1, Number.NaN, Number.NaN])).toBe(1);
  });

  it('indexAtOrBefore is a binary search for the last x <= value', () => {
    expect(indexAtOrBefore(xs, -1)).toBe(-1);
    expect(indexAtOrBefore(xs, 0)).toBe(0);
    expect(indexAtOrBefore(xs, 2.5)).toBe(2);
    expect(indexAtOrBefore(xs, 4)).toBe(4);
    expect(indexAtOrBefore(xs, 99)).toBe(4);
    expect(indexAtOrBefore([], 1)).toBe(-1);
  });

  it('legendValue: hover reads the cursor sample, rest shows the latest (or the one at restX)', () => {
    const col = [10, 11, null, 13, 14];
    expect(legendValue(xs, col, 1)).toBe(11);
    expect(legendValue(xs, col, 2)).toBeNull(); // hovering a missing sample: a dash, not the neighbour
    expect(legendValue(xs, col, null)).toBe(14);
    expect(legendValue(xs, col, undefined)).toBe(14);
    expect(legendValue(xs, col, null, 2.5)).toBe(11); // latest finite at or before x = 2.5
    expect(legendValue(xs, col, null, -3)).toBeNull();
    expect(legendValue(xs, col, 3, 0)).toBe(13); // hovering wins over restX
  });
});

describe('mergeBandTags', () => {
  it('keeps well-separated tags as they are, top first', () => {
    expect(mergeBandTags([{ y: 100, dof: 3 }, { y: 20, dof: 1 }], 12)).toEqual([
      { y: 20, text: 'dof 1' },
      { y: 100, text: 'dof 3' },
    ]);
  });

  it('merges tags closer than the gap into one and never overlaps', () => {
    const out = mergeBandTags([{ y: 50, dof: 3 }, { y: 58, dof: 1 }, { y: 200, dof: 2 }], 12);
    expect(out).toEqual([
      { y: 50, text: 'dof 1, 3' },
      { y: 200, text: 'dof 2' },
    ]);
    for (let i = 1; i < out.length; i++) expect(out[i].y - out[i - 1].y).toBeGreaterThanOrEqual(12);
  });

  it('a gap of exactly the minimum does not merge; non-finite positions are dropped', () => {
    expect(mergeBandTags([{ y: 0, dof: 1 }, { y: 12, dof: 2 }], 12)).toHaveLength(2);
    expect(mergeBandTags([{ y: Number.NaN, dof: 1 }], 12)).toEqual([]);
    expect(mergeBandTags([], 12)).toEqual([]);
  });

  it('chains merge and the same dof is listed once', () => {
    expect(mergeBandTags([{ y: 0, dof: 1 }, { y: 10, dof: 2 }, { y: 20, dof: 2 }], 12)).toEqual([{ y: 0, text: 'dof 1, 2' }]);
  });
});
