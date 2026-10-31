// Pure accessibility helpers (docs/specs/ui_console.md §8): the tab-list keyboard model, the text
// alternatives of the timeline and the plots, the help-tip placement and the replay slider.
import { describe, expect, it } from 'vitest';
import { chartSummary, nextTab, placeTip, sliderKey, sliderText, timelineSummary } from '../src/ui/a11y';

describe('nextTab', () => {
  it('arrows move and wrap', () => {
    expect(nextTab('ArrowRight', 0, 2)).toBe(1);
    expect(nextTab('ArrowRight', 1, 2)).toBe(0);
    expect(nextTab('ArrowLeft', 1, 2)).toBe(0);
    expect(nextTab('ArrowLeft', 0, 2)).toBe(1);
    expect(nextTab('ArrowRight', 2, 4)).toBe(3);
    expect(nextTab('ArrowLeft', 0, 4)).toBe(3);
  });

  it('Home and End jump to the ends from anywhere', () => {
    expect(nextTab('Home', 3, 4)).toBe(0);
    expect(nextTab('Home', 0, 4)).toBe(0);
    expect(nextTab('End', 0, 4)).toBe(3);
    expect(nextTab('End', 3, 4)).toBe(3);
  });

  it('other keys and an empty list give null (the key keeps its default)', () => {
    for (const k of ['ArrowDown', 'ArrowUp', 'Tab', 'Enter', ' ', 'a', 'PageDown']) expect(nextTab(k, 0, 2)).toBeNull();
    expect(nextTab('ArrowRight', 0, 0)).toBeNull();
  });

  it('a single tab stays put', () => {
    expect(nextTab('ArrowRight', 0, 1)).toBe(0);
    expect(nextTab('ArrowLeft', 0, 1)).toBe(0);
  });
});

describe('timelineSummary', () => {
  const info = {
    t_end_s: 5400,
    period_s: 5400 / 3,
    eclipses_s: [
      [600, 1300],
      [2400, 3100],
      [4200, 4900],
    ] as [number, number][],
  };
  const events = { t_init: 120, switch_times: [800], fallback_times: [] as number[], kicks: [{}, {}] };

  it('says no session before a hello', () => {
    expect(timelineSummary(null, [], null, 0)).toBe('Mission timeline: no session yet.');
  });

  it('summarises the span, now, the mode, eclipses and events', () => {
    const text = timelineSummary(
      info,
      [
        { mode: 0, t0: 0, t1: 800 },
        { mode: 1, t0: 800, t1: 1234 },
      ],
      events,
      1234,
    );
    expect(text).toContain('0 to 5400 s');
    expect(text).toContain('now 1234 s (orbit 0.69 of 3.0)');
    expect(text).toContain('mode POINTING since 800 s');
    expect(text).toContain('3 eclipses');
    expect(text).toContain('TRIAD start at 120 s');
    expect(text).toContain('1 mode switch');
    expect(text).not.toContain('1 mode switches');
    expect(text).toContain('0 fallbacks');
    expect(text).toContain('2 kicks');
    expect(text.endsWith('.')).toBe(true);
  });

  it('handles an empty history and a pending TRIAD start', () => {
    const text = timelineSummary(info, [], { t_init: null, switch_times: [], fallback_times: [], kicks: [] }, 0);
    expect(text).toContain('TRIAD start pending');
    expect(text).toContain('0 kicks');
    expect(text).not.toContain('mode DETUMBLE');
    expect(timelineSummary({ ...info, eclipses_s: [[1, 2]] }, [], null, 5)).toContain('1 eclipse.');
  });

  it('never calls an attitude quantity altitude', () => {
    expect(timelineSummary(info, [{ mode: 0, t0: 0, t1: 1 }], events, 1).toLowerCase()).not.toContain('altitude');
  });
});

describe('chartSummary', () => {
  it('lists the latest value of each series, a dash for a missing one', () => {
    expect(chartSummary('Nadir pointing error (°, log scale)', ['true', 'estimated'], ['1.234', null])).toBe(
      'Nadir pointing error (°, log scale). Latest: true 1.234, estimated —.',
    );
    expect(chartSummary('t', ['a', 'b', 'c'], [3, undefined, ''])).toBe('t. Latest: a 3, b —, c —.');
  });

  it('a zero is a value, not a dash; no series gives just the title', () => {
    expect(chartSummary('t', ['a'], [0])).toBe('t. Latest: a 0.');
    expect(chartSummary('Height', [], [])).toBe('Height.');
  });

  it('labels without a value (short list) get a dash', () => {
    expect(chartSummary('t', ['a', 'b'], ['1'])).toBe('t. Latest: a 1, b —.');
  });
});

describe('placeTip', () => {
  const vp = { width: 1000, height: 700 };
  const tip = { width: 300, height: 100 };
  const anchor = (left: number, top: number) => ({ left, top, right: left + 18, bottom: top + 18 });

  it('goes below the anchor, left-aligned with it', () => {
    expect(placeTip(anchor(200, 100), tip, vp)).toEqual({ left: 200, top: 124, placement: 'below' });
  });

  it('flips above when it would run off the bottom and there is room above', () => {
    const p = placeTip(anchor(200, 640), tip, vp);
    expect(p.placement).toBe('above');
    expect(p.top).toBe(640 - 6 - 100);
  });

  it('clamps to the right and left edges', () => {
    expect(placeTip(anchor(900, 100), tip, vp).left).toBe(1000 - 300 - 8);
    expect(placeTip(anchor(-20, 100), tip, vp).left).toBe(8);
  });

  it('a tip wider than the viewport sits at the margin', () => {
    expect(placeTip(anchor(50, 100), { width: 2000, height: 50 }, vp).left).toBe(8);
  });

  it('stays inside the viewport when neither side has room', () => {
    const p = placeTip(anchor(100, 30), { width: 200, height: 680 }, vp);
    expect(p.placement).toBe('below');
    expect(p.top).toBe(12); // pushed up so its bottom edge keeps the 8 px margin (700 - 680 - 8)
  });
});

describe('replay timeline slider', () => {
  it('sliderText formats elapsed of duration', () => {
    expect(sliderText(12.34, 280)).toBe('12.3 s of 280.0 s');
    expect(sliderText(0, 0.04)).toBe('0.0 s of 0.0 s');
  });

  it('sliderKey: arrows move 1 s, Shift 10 s, Home / End jump', () => {
    expect(sliderKey('ArrowRight', false)).toEqual({ delta: 1 });
    expect(sliderKey('ArrowLeft', false)).toEqual({ delta: -1 });
    expect(sliderKey('ArrowRight', true)).toEqual({ delta: 10 });
    expect(sliderKey('ArrowLeft', true)).toEqual({ delta: -10 });
    expect(sliderKey('Home', false)).toEqual({ to: 'start' });
    expect(sliderKey('End', true)).toEqual({ to: 'end' });
  });

  it('other keys are not handled (Space stays the play toggle)', () => {
    for (const k of [' ', 'ArrowUp', 'ArrowDown', 'PageDown', 'a', 'Tab']) expect(sliderKey(k, false)).toBeNull();
  });
});
