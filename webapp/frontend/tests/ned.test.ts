import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { N, bodyNedToWorld, det3, nedToWorld } from '../src/math/ned';
import raw from './fixtures/ned_fixtures.json';

type Vec = number[];
interface QuatFx {
  q: Vec;
  dcm: Vec[];
  world_cols: Vec[];
}
interface NamedFx {
  name: string;
  euler_in_deg: Vec;
  q: Vec;
  euler_deg: Vec;
  nose_world: Vec;
  wing_world: Vec;
  down_world: Vec;
}
interface VecFx {
  ned: Vec;
  world: Vec;
}
const fx = raw as unknown as { N: Vec[]; quats: QuatFx[]; named: NamedFx[]; vectors: VecFx[] };

const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

function named(name: string): NamedFx {
  const f = fx.named.find((x) => x.name === name);
  if (!f) throw new Error(`missing fixture ${name}`);
  return f;
}

function expectVec(v: THREE.Vector3, e: Vec, tol: number): void {
  expect(Math.abs(v.x - e[0])).toBeLessThan(tol);
  expect(Math.abs(v.y - e[1])).toBeLessThan(tol);
  expect(Math.abs(v.z - e[2])).toBeLessThan(tol);
}

describe('N matrix', () => {
  it('equals the Python fixture and is a proper rotation', () => {
    expect(N.map((r) => [...r])).toEqual(fx.N);
    expect(det3(N)).toBe(1);
    expect(det3(fx.N)).toBe(1);
  });

  it('maps north to +X, east to +Z and down to -Y', () => {
    expectVec(nedToWorld([1, 0, 0]), [1, 0, 0], 1e-15);
    expectVec(nedToWorld([0, 1, 0]), [0, 0, 1], 1e-15);
    expectVec(nedToWorld([0, 0, 1]), [0, -1, 0], 1e-15);
  });
});

describe('nedToWorld', () => {
  it('equals N v from the fixture vectors', () => {
    for (const v of fx.vectors) expectVec(nedToWorld(v.ned), v.world, 1e-12);
  });

  it('applies the scale', () => {
    const w = nedToWorld([2, 4, -6], 0.5);
    expectVec(w, [1, 3, 2], 1e-15);
  });
});

describe('bodyNedToWorld', () => {
  it('applied to the unit axes equals the columns of N @ to_dcm(q)', () => {
    expect(fx.quats.length).toBeGreaterThanOrEqual(20);
    for (const f of fx.quats) {
      const qw = bodyNedToWorld(f.q);
      AXES.forEach((a, k) => expectVec(a.clone().applyQuaternion(qw), f.world_cols[k], 1e-12));
    }
  });

  it('is a unit quaternion', () => {
    for (const f of fx.quats) expect(Math.abs(bodyNedToWorld(f.q).length() - 1)).toBeLessThan(1e-12);
  });

  it('level north: nose along world +X, body z (down) along world -Y', () => {
    const f = named('level_north');
    const qw = bodyNedToWorld(f.q);
    expectVec(AXES[0].clone().applyQuaternion(qw), [1, 0, 0], 1e-12);
    expectVec(AXES[2].clone().applyQuaternion(qw), [0, -1, 0], 1e-12);
    expectVec(AXES[1].clone().applyQuaternion(qw), [0, 0, 1], 1e-12); // right wing points east
  });

  it('yaw 90 deg: nose points east (world +Z)', () => {
    const qw = bodyNedToWorld(named('yaw_90_east').q);
    expectVec(AXES[0].clone().applyQuaternion(qw), [0, 0, 1], 1e-12);
  });

  it('pitch +30 deg: nose up (positive world y)', () => {
    const f = named('pitch_up_30');
    const nose = AXES[0].clone().applyQuaternion(bodyNedToWorld(f.q));
    expect(nose.y).toBeCloseTo(Math.sin(Math.PI / 6), 12);
    expectVec(nose, f.nose_world, 1e-12);
  });

  it('roll +30 deg: right wing down (negative world y)', () => {
    const f = named('roll_right_30');
    const wing = AXES[1].clone().applyQuaternion(bodyNedToWorld(f.q));
    expect(wing.y).toBeCloseTo(-Math.sin(Math.PI / 6), 12);
    expectVec(wing, f.wing_world, 1e-12);
  });

  it('named attitudes: every body axis matches the Python world axes, Euler angles round-trip', () => {
    for (const f of fx.named) {
      const qw = bodyNedToWorld(f.q);
      expectVec(AXES[0].clone().applyQuaternion(qw), f.nose_world, 1e-12);
      expectVec(AXES[1].clone().applyQuaternion(qw), f.wing_world, 1e-12);
      expectVec(AXES[2].clone().applyQuaternion(qw), f.down_world, 1e-12);
      // to_euler_zyx (oracle) gives back the input angles for these non-singular attitudes
      f.euler_in_deg.forEach((d, k) => expect(Math.abs(f.euler_deg[k] - d)).toBeLessThan(1e-9));
    }
  });
});
