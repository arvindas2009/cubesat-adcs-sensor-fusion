// Dashboard + soft-UI pass (docs/specs/ui_dashboard.md): hash routes, sourced headline numbers, the
// detailed models' geometry and frames, the optional-model manifest, and the radius tokens in style.css.
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { routeFromHash } from '../src/router';
import { FACTS } from '../src/dashboard/facts';
import { buildCubeSat } from '../src/models/cubesat';
import { ARM, buildQuadcopter, rotorPositions } from '../src/models/quadcopter';
import { GLTF_TO_FRD, validEntries } from '../src/models/gltf';

describe('routeFromHash', () => {
  it('maps the three routes and defaults to home', () => {
    expect(routeFromHash('')).toBe('home');
    expect(routeFromHash('#')).toBe('home');
    expect(routeFromHash('#home')).toBe('home');
    expect(routeFromHash('#adcs')).toBe('adcs');
    expect(routeFromHash('#px4')).toBe('px4');
    expect(routeFromHash('#/PX4')).toBe('px4');
    expect(routeFromHash('#nonsense')).toBe('home');
  });
});

describe('dashboard facts', () => {
  const progress = readFileSync(new URL('../../../PROGRESS.md', import.meta.url), 'utf8');
  it.each(FACTS.map((f) => [f.label, f] as const))('%s is quoted verbatim from PROGRESS.md', (_l, f) => {
    expect(progress).toContain(f.value);
    expect(f.source).toMatch(/^PROGRESS\.md · Group \d+$/);
  });
});

describe('detailed CubeSat', () => {
  const sat = buildCubeSat({ size: [1, 1, 3.4] });
  it('the bus is exactly the body size, centred on the origin', () => {
    const bb = new THREE.Box3().setFromObject(sat.bus);
    [-0.5, -0.5, -1.7].forEach((v, i) => expect(bb.min.toArray()[i]).toBeCloseTo(v, 6)); // float32 geometry
    [0.5, 0.5, 1.7].forEach((v, i) => expect(bb.max.toArray()[i]).toBeCloseTo(v, 6));
  });
  it('the two wings deploy along +y and -y, no longer than the bus', () => {
    expect(sat.wings).toHaveLength(2);
    // the panel (first child) clears the bus; the hinge brackets bridge the gap on purpose
    const [pa, pb] = sat.wings.map((w) => new THREE.Box3().setFromObject(w.children[0]));
    expect(pa.min.y).toBeGreaterThan(0.5);
    expect(pb.max.y).toBeLessThan(-0.5);
    const [a, b] = sat.wings.map((w) => new THREE.Box3().setFromObject(w));
    for (const bb of [a, b]) {
      expect(bb.max.z - bb.min.z).toBeLessThanOrEqual(3.4);
      expect(bb.max.x - bb.min.x).toBeLessThan(0.2); // thin plates, cells facing +-x
    }
  });
  it('reports half extents that contain every part', () => {
    const bb = new THREE.Box3().setFromObject(sat.group);
    expect(sat.halfExtent[0]).toBeGreaterThanOrEqual(bb.max.x);
    expect(sat.halfExtent[1]).toBeGreaterThanOrEqual(bb.max.y);
    expect(sat.halfExtent[2]).toBeGreaterThanOrEqual(bb.max.z);
    expect(sat.halfExtent[2]).toBeLessThan(2.2); // nothing absurdly long
  });
  it('can be built without wings', () => {
    expect(buildCubeSat({ size: [1, 1, 3.4], wings: false }).wings).toHaveLength(0);
  });
});

describe('detailed quadcopter (FRD)', () => {
  const q = buildQuadcopter({ solid: true });
  const named = (n: string) => q.children.filter((c) => c.name === n);
  it('the nose marks +x (forward)', () => {
    const [nose] = named('nose');
    expect(nose.position.x).toBeGreaterThan(0.3);
    expect(Math.abs(nose.position.y)).toBeLessThan(1e-9);
  });
  it('four motors sit on the X arms, one per quadrant', () => {
    const motors = named('motor');
    expect(motors).toHaveLength(4);
    const quads = new Set(motors.map((m) => `${Math.sign(m.position.x)}${Math.sign(m.position.y)}`));
    expect(quads.size).toBe(4);
    for (const p of rotorPositions()) expect(Math.hypot(p.x, p.y)).toBeCloseTo(ARM, 9);
  });
  it('landing legs hang down (+z) and the ghost has no detail parts', () => {
    const legs = named('leg');
    expect(legs).toHaveLength(4);
    legs.forEach((l) => expect(l.position.z).toBeGreaterThan(0));
    expect(buildQuadcopter({ solid: false }).children.some((c) => c.name === 'leg')).toBe(false);
  });
});

describe('optional glTF models', () => {
  it('maps glTF forward / up / +x to FRD forward / up (-z) / left (-y), a proper rotation', () => {
    const map = (v: [number, number, number]) =>
      new THREE.Vector3(...v)
        .applyMatrix4(GLTF_TO_FRD)
        .toArray()
        .map((x) => x + 0); // +0 turns -0 into 0
    expect(map([0, 0, 1])).toEqual([1, 0, 0]);
    expect(map([0, 1, 0])).toEqual([0, 0, -1]);
    expect(map([1, 0, 0])).toEqual([0, -1, 0]);
    expect(GLTF_TO_FRD.determinant()).toBeCloseTo(1, 12);
  });
  it('keeps only licensed, complete manifest entries', () => {
    const ok = { role: 'quadcopter', file: 'quad.glb', license: 'CC0-1.0', source: 'https://example.org', author: 'a' };
    expect(validEntries({ models: [ok] })).toHaveLength(1);
    expect(validEntries({ models: [{ ...ok, license: 'All rights reserved' }] })).toHaveLength(0);
    expect(validEntries({ models: [{ ...ok, file: '../x.glb' }] })).toHaveLength(0);
    expect(validEntries(null)).toHaveLength(0);
  });
  it('the shipped manifest is valid (every entry licensed)', () => {
    const m = JSON.parse(readFileSync(new URL('../public/models/manifest.json', import.meta.url), 'utf8'));
    expect(validEntries(m)).toHaveLength(m.models.length);
  });
});

describe('soft UI tokens (style.css)', () => {
  const css = readFileSync(new URL('../src/style.css', import.meta.url), 'utf8');
  it('defines the radius tokens', () => {
    for (const t of ['--r-xs: 4px', '--r-sm: 8px', '--r-md: 12px', '--r-lg: 20px', '--r-pill: 999px']) expect(css).toContain(t);
  });
  it('has no hard-coded 1-4 px radius left', () => {
    expect(css).not.toMatch(/border-radius:\s*[1-4]px/);
  });
});

// ---- landing page v2: motion helpers and sourced copy -------------------------------------------
import { Spring, nearest, project, rubberband, velocityOf } from '../src/dashboard/spring';
import { LOOP, PHASES } from '../src/dashboard/content';

describe('Spring', () => {
  const run = (s: Spring, seconds: number) => {
    let max = -Infinity;
    for (let t = 0; t < seconds; t += 1 / 60) max = Math.max(max, s.step(1 / 60));
    return max;
  };
  it('critically damped: reaches the target without overshoot', () => {
    const s = new Spring(0, { damping: 1, response: 0.4 });
    s.target = 1;
    expect(run(s, 3)).toBeLessThanOrEqual(1 + 1e-6);
    expect(s.value).toBeCloseTo(1, 4);
    expect(s.settled).toBe(true);
  });
  it('under-damped: overshoots, then settles', () => {
    const s = new Spring(0, { damping: 0.5, response: 0.4 });
    s.target = 1;
    expect(run(s, 4)).toBeGreaterThan(1.05);
    expect(s.value).toBeCloseTo(1, 3);
  });
  it('re-targets from its current value and velocity (no jump)', () => {
    const s = new Spring(0, { damping: 1, response: 0.4 });
    s.target = 1;
    s.step(0.1);
    const v = s.value;
    s.target = -1;
    s.configure({ damping: 0.8, response: 0.3 });
    expect(s.value).toBe(v);
    s.step(1 / 240);
    expect(Math.abs(s.value - v)).toBeLessThan(0.05);
  });
  it('a larger response is slower', () => {
    const fast = new Spring(0, { response: 0.2 });
    const slow = new Spring(0, { response: 0.8 });
    fast.target = slow.target = 1;
    fast.step(0.15);
    slow.step(0.15);
    expect(fast.value).toBeGreaterThan(slow.value);
  });
});

describe('gesture helpers', () => {
  it('project: 1000 px/s coasts 499 px at 0.998, nothing at rest', () => {
    expect(project(1000)).toBeCloseTo(499, 6);
    expect(project(0)).toBe(0);
    expect(project(-500)).toBeCloseTo(-249.5, 6);
  });
  it('nearest picks the closest snap point', () => {
    expect(nearest([0, 1 / 3, 2 / 3, 1], 0.55)).toBeCloseTo(2 / 3, 12);
    expect(nearest([0, 1], -5)).toBe(0);
  });
  it('rubberband resists more the further past the edge, and never exceeds the pull', () => {
    const a = rubberband(50, 400);
    const b = rubberband(200, 400);
    expect(a).toBeGreaterThan(0);
    expect(a).toBeLessThan(50);
    expect(b / 200).toBeLessThan(a / 50);
    expect(rubberband(-50, 400)).toBeCloseTo(-a, 12);
  });
  it('velocityOf uses the recent window', () => {
    const h = [
      { x: 0, t: 0 },
      { x: 10, t: 500 },
      { x: 20, t: 550 },
      { x: 30, t: 600 },
    ];
    expect(velocityOf(h)).toBeCloseTo(200, 6); // (30 - 10) px / 0.1 s; the t = 0 sample is outside the window
    expect(velocityOf([{ x: 1, t: 0 }])).toBe(0);
  });
});

describe('landing copy is sourced', () => {
  const progress = readFileSync(new URL('../../../PROGRESS.md', import.meta.url), 'utf8');
  const all = [...LOOP.flatMap((s) => s.evidence), ...PHASES.flatMap((p) => p.evidence)];
  it.each(all)('"%s" appears in PROGRESS.md', (e) => {
    expect(progress).toContain(e);
  });
  it('the loop has the four stages in order and phases start with release', () => {
    expect(LOOP.map((s) => s.id)).toEqual(['sense', 'estimate', 'control', 'actuate']);
    expect(PHASES[0].id).toBe('release');
    expect(PHASES.at(-1)!.id).toBe('point');
  });
});
