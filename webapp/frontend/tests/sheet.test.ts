// Pure tests of the rising plot sheet (docs/specs/ui_console.md §3.3): the motion model (follow, snap,
// reduced motion, drag, toggle), wheel normalisation, and the wheel / key routing truth tables.
import { describe, expect, it } from 'vitest';
import { SheetModel, easeOutCubic, normalizeWheel, routeKey, routeWheel, type SheetConfig } from '../src/ui/sheetModel';

const CFG: Partial<SheetConfig> = { travelPx: 400, durationMs: 260, idleMs: 140, commitFrac: 0.08, discretePx: 40 };

/** Tick until the model stops moving; returns the last tick and how many settle events were seen. */
function run(m: SheetModel, from: number, stepMs = 16, maxMs = 2000): { last: ReturnType<SheetModel['tick']>; settles: number; t: number } {
  let settles = 0;
  let t = from;
  let last = m.tick(t);
  if (last.settledNow) settles++;
  while ((last.moving || m.tick(t).moving) && t < from + maxMs) {
    t += stepMs;
    last = m.tick(t);
    if (last.settledNow) settles++;
  }
  return { last, settles, t };
}

describe('easeOutCubic', () => {
  it('runs 0 to 1, fast at the start, clamped outside', () => {
    expect(easeOutCubic(0)).toBe(0);
    expect(easeOutCubic(1)).toBe(1);
    expect(easeOutCubic(0.5)).toBeCloseTo(0.875, 12);
    expect(easeOutCubic(-3)).toBe(0);
    expect(easeOutCubic(9)).toBe(1);
    expect(easeOutCubic(0.2)).toBeGreaterThan(0.2);
  });
});

describe('SheetModel: wheel follow and snap', () => {
  it('starts lowered and idle', () => {
    const m = new SheetModel(CFG);
    expect(m.tick(0)).toEqual({ p: 0, state: 'lowered', moving: false, settledNow: false });
    expect(new SheetModel(CFG, 'raised').tick(0)).toEqual({ p: 1, state: 'raised', moving: false, settledNow: false });
  });

  it('follows the wheel 1:1 over the travel', () => {
    const m = new SheetModel(CFG);
    m.wheel(100, 0);
    let r = m.tick(1);
    expect(r.p).toBeCloseTo(0.25, 12);
    expect(r.moving).toBe(true);
    expect(r.state).toBe('lowered'); // not committed yet
    m.wheel(40, 20);
    r = m.tick(21);
    expect(r.p).toBeCloseTo(0.35, 12);
    m.wheel(-20, 40);
    expect(m.tick(41).p).toBeCloseTo(0.3, 12);
  });

  it('commits to the gesture direction after the idle time, ending exactly at 1 with one settle', () => {
    const m = new SheetModel(CFG);
    m.wheel(100, 0);
    expect(m.tick(100).moving).toBe(true); // still within idleMs
    expect(m.tick(139).state).toBe('lowered');
    const r = m.tick(140); // idle reached: snap starts
    expect(r.state).toBe('raised');
    expect(r.moving).toBe(true);
    const { last, settles } = run(m, 140);
    expect(last.p).toBe(1);
    expect(last.state).toBe('raised');
    expect(last.moving).toBe(false);
    expect(settles).toBe(1);
    expect(m.tick(9999).settledNow).toBe(false);
  });

  it('returns when the gesture moved less than commitFrac', () => {
    const m = new SheetModel(CFG);
    m.wheel(20, 0); // 0.05 < 0.08
    expect(m.tick(1).p).toBeCloseTo(0.05, 12);
    const { last, settles } = run(m, 200);
    expect(last.p).toBe(0);
    expect(last.state).toBe('lowered');
    expect(settles).toBe(1);
  });

  it('a small push from raised returns to raised; a larger upward one lowers', () => {
    const m = new SheetModel(CFG, 'raised');
    m.wheel(-20, 0);
    let res = run(m, 200);
    expect(res.last.p).toBe(1);
    expect(res.last.state).toBe('raised');
    m.wheel(-200, 1000);
    res = run(m, 1200);
    expect(res.last.p).toBe(0);
    expect(res.last.state).toBe('lowered');
  });

  it('the animation uses the full duration and is monotonic', () => {
    const m = new SheetModel(CFG);
    m.wheel(100, 0);
    m.tick(140);
    let prev = m.tick(140).p;
    for (let t = 150; t < 140 + 260; t += 10) {
      const p = m.tick(t).p;
      expect(p).toBeGreaterThanOrEqual(prev);
      expect(p).toBeLessThan(1);
      prev = p;
    }
    const end = m.tick(140 + 260);
    expect(end.p).toBe(1);
    expect(end.settledNow).toBe(true);
  });

  it('is clamped to [0, 1] however hard the wheel is pushed', () => {
    const m = new SheetModel(CFG);
    m.wheel(100000, 0);
    expect(m.tick(1).p).toBe(1);
    m.wheel(-100000, 20);
    expect(m.tick(21).p).toBe(0);
    m.wheel(100000, 40);
    expect(m.tick(41).p).toBe(1);
  });

  it('a wheel at an end that never leaves it does not settle', () => {
    const m = new SheetModel(CFG);
    m.wheel(-50, 0);
    expect(m.tick(1)).toEqual({ p: 0, state: 'lowered', moving: false, settledNow: false });
    expect(m.tick(500).settledNow).toBe(false);
  });

  it('a new wheel during the snap animation takes over from where it is', () => {
    const m = new SheetModel(CFG);
    m.wheel(100, 0);
    m.tick(140);
    const mid = m.tick(200).p;
    expect(mid).toBeGreaterThan(0.25);
    expect(mid).toBeLessThan(1);
    m.wheel(-40, 210); // user reverses
    const r = m.tick(211);
    expect(r.p).toBeLessThan(mid);
    expect(r.moving).toBe(true);
    const { last, settles } = run(m, 400);
    expect(last.p).toBe(0); // moved up by 0.1 (>= commitFrac) -> lowers
    expect(settles).toBe(1);
  });

  it('ignores zero, NaN and a zero travel', () => {
    const m = new SheetModel(CFG);
    m.wheel(0, 0);
    m.wheel(Number.NaN, 1);
    expect(m.tick(2).p).toBe(0);
    m.setTravel(0);
    m.wheel(100, 3);
    expect(m.tick(4).p).toBe(0);
    m.setTravel(200);
    m.wheel(100, 5);
    expect(m.tick(6).p).toBeCloseTo(0.5, 12);
  });
});

describe('SheetModel: drag', () => {
  it('follows the pointer (down = lowering) and snaps on release', () => {
    const m = new SheetModel(CFG, 'raised');
    m.dragStart(0);
    m.dragMove(100, 10); // pointer 100 px down: p = 1 - 0.25
    expect(m.tick(11).p).toBeCloseTo(0.75, 12);
    m.dragMove(-30, 20); // back up past the start: clamped
    expect(m.tick(21).p).toBe(1);
    m.dragMove(200, 30);
    expect(m.tick(31).p).toBeCloseTo(0.5, 12);
    m.dragEnd(40);
    const { last, settles } = run(m, 40);
    expect(last.p).toBe(0); // moved down by half: commits to lowered
    expect(last.state).toBe('lowered');
    expect(settles).toBe(1);
  });

  it('a short drag returns to the start state', () => {
    const m = new SheetModel(CFG);
    m.dragStart(0);
    m.dragMove(-20, 5); // 5 % up
    m.dragEnd(10);
    const { last } = run(m, 10);
    expect(last.p).toBe(0);
    expect(last.state).toBe('lowered');
  });

  it('a drag does not end on idle (only on dragEnd)', () => {
    const m = new SheetModel(CFG);
    m.dragStart(0);
    m.dragMove(-200, 1);
    const r = m.tick(5000);
    expect(r.p).toBeCloseTo(0.5, 12);
    expect(r.state).toBe('lowered');
    expect(r.moving).toBe(true);
  });

  it('ignores moves without dragStart', () => {
    const m = new SheetModel(CFG);
    m.dragMove(-200, 1);
    expect(m.tick(2).p).toBe(0);
  });
});

describe('SheetModel: toggle and set', () => {
  it('toggle animates to the other end and settles once', () => {
    const m = new SheetModel(CFG);
    m.toggle(0);
    expect(m.tick(0).state).toBe('raised');
    const mid = m.tick(130);
    expect(mid.p).toBeGreaterThan(0);
    expect(mid.p).toBeLessThan(1);
    const { last, settles } = run(m, 130);
    expect(last.p).toBe(1);
    expect(settles).toBe(1);
    m.toggle(1000);
    const down = run(m, 1000);
    expect(down.last.p).toBe(0);
    expect(down.last.state).toBe('lowered');
    expect(down.settles).toBe(1);
  });

  it('set to the current state is a no-op', () => {
    const m = new SheetModel(CFG);
    m.set('lowered', 0);
    expect(m.tick(1)).toEqual({ p: 0, state: 'lowered', moving: false, settledNow: false });
    m.set('raised', 2);
    run(m, 2);
    m.set('raised', 1000);
    expect(m.tick(1001)).toEqual({ p: 1, state: 'raised', moving: false, settledNow: false });
  });

  it('exposes the committed state immediately', () => {
    const m = new SheetModel(CFG);
    expect(m.state).toBe('lowered');
    m.set('raised', 0);
    expect(m.state).toBe('raised');
  });
});

describe('SheetModel: reduced motion', () => {
  const RM: Partial<SheetConfig> = { ...CFG, reducedMotion: true };

  it('flips at the first discretePx of a wheel gesture, once per gesture', () => {
    const m = new SheetModel(RM);
    m.wheel(20, 0);
    expect(m.tick(1).p).toBe(0); // below 40 px: nothing yet
    m.wheel(25, 10); // 45 px accumulated
    let r = m.tick(11);
    expect(r).toEqual({ p: 1, state: 'raised', moving: false, settledNow: true });
    m.wheel(-500, 20); // same gesture (< idleMs): ignored
    r = m.tick(21);
    expect(r.p).toBe(1);
    expect(r.settledNow).toBe(false);
    m.wheel(-500, 100);
    expect(m.tick(101).p).toBe(1);
  });

  it('the next gesture (after idle) can flip back', () => {
    const m = new SheetModel(RM);
    m.wheel(100, 0);
    expect(m.tick(1).state).toBe('raised');
    m.wheel(-100, 1000); // idle passed: a new gesture
    const r = m.tick(1001);
    expect(r).toEqual({ p: 0, state: 'lowered', moving: false, settledNow: true });
  });

  it('never reports intermediate positions', () => {
    const m = new SheetModel(RM);
    for (let t = 0; t < 100; t += 10) {
      m.wheel(15, t);
      const p = m.tick(t + 1).p;
      expect(p === 0 || p === 1).toBe(true);
    }
  });

  it('toggle and set are instant', () => {
    const m = new SheetModel(RM);
    m.toggle(0);
    expect(m.tick(0)).toEqual({ p: 1, state: 'raised', moving: false, settledNow: true });
    m.set('lowered', 5);
    expect(m.tick(5)).toEqual({ p: 0, state: 'lowered', moving: false, settledNow: true });
  });

  it('a drag flips once past discretePx (upwards raises)', () => {
    const m = new SheetModel(RM);
    m.dragStart(0);
    m.dragMove(-20, 1);
    expect(m.tick(2).p).toBe(0);
    m.dragMove(-60, 3);
    expect(m.tick(4)).toEqual({ p: 1, state: 'raised', moving: false, settledNow: true });
    m.dragMove(80, 5); // ignored for the rest of the gesture
    expect(m.tick(6).p).toBe(1);
    m.dragEnd(7);
    expect(m.tick(8).p).toBe(1);
  });

  it('switching to reduced motion mid-animation finishes it at once', () => {
    const m = new SheetModel(CFG);
    m.toggle(0);
    m.tick(100);
    m.setReducedMotion(true);
    expect(m.tick(101)).toEqual({ p: 1, state: 'raised', moving: false, settledNow: true });
  });

  it('switching to reduced motion mid-gesture jumps to the committed end', () => {
    const m = new SheetModel(CFG);
    m.wheel(120, 0); // 0.3 raised, not yet snapped
    m.setReducedMotion(true);
    const r = m.tick(1);
    expect(r.p).toBe(1);
    expect(r.settledNow).toBe(true);
  });
});

describe('normalizeWheel', () => {
  it('pixels pass through, lines count 16 px, pages the page height', () => {
    expect(normalizeWheel(53, 0, 700)).toBe(53);
    expect(normalizeWheel(-3, 1, 700)).toBe(-48);
    expect(normalizeWheel(1, 2, 700)).toBe(700);
    expect(normalizeWheel(-0.5, 2, 640)).toBe(-320);
  });
});

describe('routeWheel', () => {
  const base = { enabled: true, dialogOpen: false, overCanvas: false, zoomModifier: false, overOwnScroller: false };

  it('truth table', () => {
    const bools = [false, true];
    for (const enabled of bools)
      for (const dialogOpen of bools)
        for (const overCanvas of bools)
          for (const zoomModifier of bools)
            for (const overOwnScroller of bools) {
              const c = { enabled, dialogOpen, overCanvas, zoomModifier, overOwnScroller };
              let want: 'sheet' | 'camera' | 'native';
              if (!enabled || dialogOpen) want = 'native';
              else if (zoomModifier) want = overCanvas ? 'camera' : 'native';
              else if (overOwnScroller) want = 'native';
              else want = 'sheet';
              expect(routeWheel(c), JSON.stringify(c)).toBe(want);
            }
  });

  it('the documented cases', () => {
    expect(routeWheel(base)).toBe('sheet'); // plain wheel on the stage, bar or sheet
    expect(routeWheel({ ...base, overCanvas: true })).toBe('sheet'); // plain wheel over a 3D canvas
    expect(routeWheel({ ...base, overCanvas: true, zoomModifier: true })).toBe('camera'); // Ctrl / Cmd / pinch
    expect(routeWheel({ ...base, zoomModifier: true })).toBe('native'); // browser zoom elsewhere
    expect(routeWheel({ ...base, overOwnScroller: true })).toBe('native'); // the overflowing side column
    expect(routeWheel({ ...base, dialogOpen: true })).toBe('native');
    expect(routeWheel({ ...base, enabled: false, overCanvas: true })).toBe('native'); // fallback layout
  });
});

describe('routeKey', () => {
  const base = { key: 'PageDown', enabled: true, dialogOpen: false, target: 'other' as const, state: 'lowered' as const };

  it('PageDown raises a lowered sheet, PageUp lowers a raised one', () => {
    expect(routeKey(base)).toBe('raise');
    expect(routeKey({ ...base, key: 'PageUp', state: 'raised' })).toBe('lower');
  });

  it('does nothing when there is nothing to do (the browser keeps the key)', () => {
    expect(routeKey({ ...base, state: 'raised' })).toBeNull();
    expect(routeKey({ ...base, key: 'PageUp' })).toBeNull();
  });

  it('is ignored in text and number inputs, selects, ranges and sliders', () => {
    for (const target of ['text', 'select', 'range', 'slider'] as const) {
      expect(routeKey({ ...base, target }), target).toBeNull();
      expect(routeKey({ ...base, key: 'PageUp', state: 'raised', target }), target).toBeNull();
    }
  });

  it('is ignored with a dialog open or when the layout is disabled', () => {
    expect(routeKey({ ...base, dialogOpen: true })).toBeNull();
    expect(routeKey({ ...base, enabled: false })).toBeNull();
  });

  it('other keys are not routed', () => {
    for (const key of ['ArrowDown', ' ', 'Home', 'End', 'a']) expect(routeKey({ ...base, key })).toBeNull();
  });
});
