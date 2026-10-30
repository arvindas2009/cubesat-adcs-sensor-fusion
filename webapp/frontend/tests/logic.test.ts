import { describe, expect, it } from 'vitest';
import {
  FrameLatch,
  ModeTrack,
  SLIDER_MAX,
  Throttle,
  buildKickCommand,
  clampLog,
  closeBanner,
  fieldSpecs,
  intervalRect,
  logRange,
  orbitTicks,
  sliderToSpeed,
  speedToSlider,
  timeRange,
  timeToX,
  validateAll,
  validateField,
} from '../src/adcs/logic';
import type { Frame, SchemaProperty } from '../src/types';

// What GET /api/adcs/presets returns (pydantic JSON schema of SessionParams).
const PROPS: Record<string, SchemaProperty> = {
  orbits: { default: 6, maximum: 15, minimum: 0.05, title: 'Orbits', type: 'number' },
  rate_deg_s: { default: 10, maximum: 30, minimum: 0, title: 'Rate Deg S', type: 'number' },
  seed: { default: 0, maximum: 2147483647, minimum: 0, title: 'Seed', type: 'integer' },
  sensors: { default: 'realistic', enum: ['realistic', 'ideal'], title: 'Sensors', type: 'string' },
  filter: { default: 'design', enum: ['design', 'g7'], title: 'Filter', type: 'string' },
  dumping: { default: true, title: 'Dumping', type: 'boolean' },
  inertia_scale: { default: 1, maximum: 1.5, minimum: 0.5, title: 'Inertia Scale', type: 'number' },
};

describe('form specs and validation', () => {
  const specs = fieldSpecs(PROPS);
  const good = {
    orbits: '6',
    rate_deg_s: '10',
    seed: '0',
    sensors: 'realistic',
    filter: 'design',
    dumping: true,
    inertia_scale: '1',
  };

  it('keeps schema order and kinds', () => {
    expect(specs.map((s) => s.name)).toEqual([
      'orbits',
      'rate_deg_s',
      'seed',
      'sensors',
      'filter',
      'dumping',
      'inertia_scale',
    ]);
    expect(specs.map((s) => s.kind)).toEqual(['number', 'number', 'integer', 'enum', 'enum', 'boolean', 'number']);
    expect(specs[0].hint).toBe('0.05 – 15');
    expect(specs[2].hint).toBe('integer 0 – 2147483647');
  });

  it('accepts the defaults', () => {
    const r = validateAll(specs, good);
    expect(r.valid).toBe(true);
    expect(r.values).toEqual({
      orbits: 6,
      rate_deg_s: 10,
      seed: 0,
      sensors: 'realistic',
      filter: 'design',
      dumping: true,
      inertia_scale: 1,
    });
  });

  it('rejects every kind of violation with a message', () => {
    const orbits = specs[0];
    expect(validateField(orbits, '0.01')).toEqual({ ok: false, error: 'must be at least 0.05' });
    expect(validateField(orbits, '15.5')).toEqual({ ok: false, error: 'must be at most 15' });
    expect(validateField(orbits, '')).toEqual({ ok: false, error: 'required' });
    expect(validateField(orbits, 'abc')).toEqual({ ok: false, error: 'not a number' });
    expect(validateField(specs[2], '1.5')).toEqual({ ok: false, error: 'must be an integer' });
    expect(validateField(specs[3], 'bogus').ok).toBe(false);
    expect(validateField(orbits, '0.05').ok).toBe(true); // bounds are inclusive
    expect(validateField(orbits, '15').ok).toBe(true);
  });

  it('collects errors per field', () => {
    const r = validateAll(specs, { ...good, orbits: '99', seed: '-3' });
    expect(r.valid).toBe(false);
    expect(Object.keys(r.errors).sort()).toEqual(['orbits', 'seed']);
  });

  it('honours exclusive bounds and skips unsupported types', () => {
    const s = fieldSpecs({ a: { type: 'number', exclusiveMinimum: 0 }, b: { type: 'array' } });
    expect(s).toHaveLength(1);
    expect(validateField(s[0], '0')).toEqual({ ok: false, error: 'must be greater than 0' });
    expect(validateField(s[0], '0.1').ok).toBe(true);
  });
});

describe('buildKickCommand', () => {
  it('builds signed commands', () => {
    expect(buildKickCommand('x', 1, 5)).toEqual({ ok: true, cmd: { cmd: 'kick', axis: 'x', deg_s: 5 } });
    expect(buildKickCommand('z', -1, 12.5)).toEqual({ ok: true, cmd: { cmd: 'kick', axis: 'z', deg_s: -12.5 } });
  });
  it('rejects zero, negative magnitude, NaN and anything above 20 deg/s', () => {
    expect(buildKickCommand('y', 1, 20).ok).toBe(true);
    for (const m of [0, -1, NaN, Infinity, 20.01]) expect(buildKickCommand('y', 1, m).ok).toBe(false);
  });
});

describe('speed slider', () => {
  it('maps the ends and the middle logarithmically', () => {
    expect(sliderToSpeed(0)).toBe(1);
    expect(sliderToSpeed(SLIDER_MAX)).toBe(2000);
    expect(sliderToSpeed(SLIDER_MAX / 2)).toBeCloseTo(44.7, 1);
  });
  it('clamps out-of-range input', () => {
    expect(sliderToSpeed(-50)).toBe(1);
    expect(sliderToSpeed(5000)).toBe(2000);
    expect(speedToSlider(0.1)).toBe(0);
    expect(speedToSlider(1e6)).toBeCloseTo(SLIDER_MAX, 9);
  });
  it('is monotone and round-trips within the 3-digit rounding', () => {
    let prev = 0;
    for (let p = 0; p <= SLIDER_MAX; p += 25) {
      const v = sliderToSpeed(p);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
      expect(Math.abs(speedToSlider(v) - p)).toBeLessThan(4);
    }
    expect(speedToSlider(500)).toBeGreaterThan(speedToSlider(50));
  });
  it('speeds sent to the server are always within 1..2000', () => {
    for (let p = 0; p <= SLIDER_MAX; p++) {
      const v = sliderToSpeed(p);
      expect(v).toBeGreaterThanOrEqual(1);
      expect(v).toBeLessThanOrEqual(2000);
    }
  });
});

describe('timeline geometry', () => {
  it('maps time to x and clamps', () => {
    expect(timeToX(0, 100, 10, 110)).toBe(10);
    expect(timeToX(50, 100, 10, 110)).toBe(60);
    expect(timeToX(100, 100, 10, 110)).toBe(110);
    expect(timeToX(500, 100, 10, 110)).toBe(110);
    expect(timeToX(-5, 100, 10, 110)).toBe(10);
    expect(timeToX(5, 0, 10, 110)).toBe(10);
  });
  it('gives intervals a minimum width', () => {
    expect(intervalRect(20, 40, 100, 0, 200)).toEqual({ x: 40, w: 40 });
    expect(intervalRect(50, 50, 100, 0, 200, 2)).toEqual({ x: 100, w: 2 });
  });
  it('places orbit ticks strictly inside the horizon', () => {
    expect(orbitTicks(350, 100)).toEqual([
      { t: 100, n: 1 },
      { t: 200, n: 2 },
      { t: 300, n: 3 },
    ]);
    expect(orbitTicks(300, 100)).toEqual([
      { t: 100, n: 1 },
      { t: 200, n: 2 },
    ]);
    expect(orbitTicks(50, 100)).toEqual([]);
    expect(orbitTicks(100, 0)).toEqual([]);
  });
});

describe('ModeTrack', () => {
  it('stores only mode changes and closes the last segment at the latest time', () => {
    const m = new ModeTrack();
    expect(m.segments()).toEqual([]);
    for (let t = 0; t <= 4; t++) m.push(t, 0);
    for (let t = 5; t <= 9; t++) m.push(t, 1);
    m.push(10, 0);
    expect(m.segments()).toEqual([
      { mode: 0, t0: 0, t1: 5 },
      { mode: 1, t0: 5, t1: 10 },
      { mode: 0, t0: 10, t1: 10 },
    ]);
  });
  it('clears when time goes backwards or on demand', () => {
    const m = new ModeTrack();
    m.push(0, 0);
    m.push(100, 1);
    m.push(0, 0);
    expect(m.segments()).toEqual([{ mode: 0, t0: 0, t1: 0 }]);
    m.clear();
    expect(m.segments()).toEqual([]);
  });
});

describe('log axis helpers', () => {
  it('clamps non-positive values to the floor and keeps null', () => {
    expect(clampLog(null, 1e-3)).toBeNull();
    expect(clampLog(0, 1e-3)).toBe(1e-3);
    expect(clampLog(-2, 1e-3)).toBe(1e-3);
    expect(clampLog(0.5, 1e-3)).toBe(0.5);
  });
  it('returns a decade below the minimum and a 1-2-5 value above the maximum, at least a decade tall', () => {
    expect(logRange(0.3, 150, 1e-3)).toEqual([0.1, 200]);
    expect(logRange(0.3, 3, 1e-3)).toEqual([0.1, 5]);
    expect(logRange(2, 40, 1e-3)).toEqual([1, 50]);
    expect(logRange(2, 7, 1e-3)).toEqual([1, 10]);
    expect(logRange(5, 5, 1e-3)).toEqual([1, 10]);
    expect(logRange(10, 10, 1e-3)).toEqual([10, 100]);
  });
  it('includes threshold lines and respects the floor', () => {
    expect(logRange(3, 8, 1e-3, [0.5, 2])).toEqual([0.1, 10]);
    expect(logRange(1e-6, 1, 1e-3)[0]).toBe(1e-3);
  });
  it('has a default for empty data', () => {
    const [a, b] = logRange(null, undefined, 1e-3);
    expect(a).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(a);
  });
  it('time range runs to the horizon and never collapses', () => {
    expect(timeRange(0, 100, 600)).toEqual([0, 600]);
    expect(timeRange(120, 700, 600)).toEqual([120, 700]);
    expect(timeRange(null, null, 0)).toEqual([0, 1]);
  });
});

describe('FrameLatch', () => {
  const f = (t: number) => ({ t }) as unknown as Frame;
  it('keeps only the newest frame between takes and counts drops', () => {
    const l = new FrameLatch();
    expect(l.take()).toBeNull();
    l.offer(f(1));
    l.offer(f(2));
    l.offer(f(3));
    expect(l.take()?.t).toBe(3);
    expect(l.take()).toBeNull();
    expect(l.dropped).toBe(2);
  });
});

describe('Throttle', () => {
  it('runs at most once per interval unless forced', () => {
    const th = new Throttle(100);
    expect(th.shouldRun(1000)).toBe(true);
    expect(th.shouldRun(1050)).toBe(false);
    expect(th.shouldRun(1099)).toBe(false);
    expect(th.shouldRun(1100)).toBe(true);
    expect(th.shouldRun(1110, true)).toBe(true);
    expect(th.shouldRun(1150)).toBe(false);
  });
});

describe('closeBanner', () => {
  it('describes the protocol close codes', () => {
    expect(closeBanner(4001)).toContain('4001');
    expect(closeBanner(4404)).toContain('4404');
    expect(closeBanner(1006)).toContain('1006');
  });
});
