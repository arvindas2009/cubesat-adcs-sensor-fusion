import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  M,
  bodyInEciToWorld,
  displayMatrixToWorld,
  earthRotationY,
  eciToWorld,
  latLonToEcef,
  toThreeQuaternion,
  zUpToWorld,
} from '../src/math/frames';
import raw from './fixtures/frames_fixtures.json';

type Vec = number[];
interface QuatFx {
  q: Vec;
  dcm: Vec[];
}
interface FrameFx {
  label: string;
  phase: string;
  frame: {
    eci: { q: Vec; r_m: Vec; gmst_rad: number };
    display: { C: Vec[]; sun: Vec; b: Vec };
  };
  q_unit: Vec;
  display_world_cols: Vec[];
  eci_world_cols: Vec[];
  display_sun_world: Vec;
  display_b_world: Vec;
  r_world_m: Vec;
}
interface EarthFx {
  gmst_rad: number;
  lat_deg: number;
  lon_deg: number;
  lat_rad: number;
  lon_rad: number;
  ecef: Vec;
  world: Vec;
  u: number;
  v: number;
}
const fx = raw as unknown as {
  M: Vec[];
  display_flip: Vec[];
  quats: QuatFx[];
  frames: FrameFx[];
  earth: EarthFx[];
};

const UNIT_AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

function expectVec(v: THREE.Vector3, expected: Vec, tol: number): void {
  expect(Math.abs(v.x - expected[0])).toBeLessThan(tol);
  expect(Math.abs(v.y - expected[1])).toBeLessThan(tol);
  expect(Math.abs(v.z - expected[2])).toBeLessThan(tol);
}

describe('fixtures', () => {
  it('cover the phases the page shows', () => {
    const phases = new Set(fx.frames.map((f) => f.phase));
    expect(phases.has('pre-init')).toBe(true);
    expect(phases.has('detumble')).toBe(true);
    expect(phases.has('pointing')).toBe(true);
    expect(fx.quats.length).toBeGreaterThanOrEqual(20);
  });

  it('M in the fixtures equals M in frames.ts', () => {
    expect(fx.M).toEqual(M.map((r) => [...r]));
  });
});

describe('toThreeQuaternion', () => {
  it('gives the rotation matrix of common.quaternion.to_dcm (1e-12)', () => {
    for (const { q, dcm } of fx.quats) {
      const m = new THREE.Matrix4().makeRotationFromQuaternion(toThreeQuaternion(q));
      const e = m.elements; // column-major
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(Math.abs(e[j * 4 + i] - dcm[i][j])).toBeLessThan(1e-12);
        }
      }
    }
  });

  it('stores scalar-first [w, x, y, z] as Three.js (x, y, z, w)', () => {
    const t = toThreeQuaternion([0.1, 0.2, 0.3, 0.4]);
    expect([t.x, t.y, t.z, t.w]).toEqual([0.2, 0.3, 0.4, 0.1]);
  });
});

describe('displayMatrixToWorld', () => {
  it('columns equal the fixture world body axes (1e-6) and form a proper rotation', () => {
    for (const f of fx.frames) {
      const m = displayMatrixToWorld(f.frame.display.C);
      const e = m.elements;
      for (let k = 0; k < 3; k++) {
        const col = new THREE.Vector3(e[4 * k], e[4 * k + 1], e[4 * k + 2]);
        expectVec(col, f.display_world_cols[k], 1e-6);
      }
      expect(Math.abs(m.determinant() - 1)).toBeLessThan(1e-5);
    }
  });

  it('maps body axes of a body-fixed mesh the same way via applyMatrix4', () => {
    for (const f of fx.frames) {
      const m = displayMatrixToWorld(f.frame.display.C);
      UNIT_AXES.forEach((a, k) => expectVec(a.clone().applyMatrix4(m), f.display_world_cols[k], 1e-6));
    }
  });

  it('DISPLAY_FLIP puts body z at world (0, -1, 0): nadir is down', () => {
    const m = displayMatrixToWorld(fx.display_flip);
    const z = new THREE.Vector3(0, 0, 1).applyMatrix4(m);
    expectVec(z, [0, -1, 0], 1e-12);
    expect(Math.abs(m.determinant() - 1)).toBeLessThan(1e-12);
  });
});

describe('bodyInEciToWorld', () => {
  it('matches M @ to_dcm(q) on the unit axes (1e-12), frame quaternions normalised', () => {
    for (const f of fx.frames) {
      const q = bodyInEciToWorld(f.q_unit);
      UNIT_AXES.forEach((a, k) => expectVec(a.clone().applyQuaternion(q), f.eci_world_cols[k], 1e-12));
    }
  });

  it('matches for the random quaternions too', () => {
    for (const { q, dcm } of fx.quats) {
      const w = bodyInEciToWorld(q);
      UNIT_AXES.forEach((a, k) => {
        const colEci = [dcm[0][k], dcm[1][k], dcm[2][k]];
        expectVec(a.clone().applyQuaternion(w), [colEci[0], colEci[2], -colEci[1]], 1e-12);
      });
    }
  });

  it('wire quaternions (7 significant digits, not exactly unit) agree to 1e-6', () => {
    for (const f of fx.frames) {
      const q = bodyInEciToWorld(f.frame.eci.q).normalize();
      UNIT_AXES.forEach((a, k) => expectVec(a.clone().applyQuaternion(q), f.eci_world_cols[k], 1e-6));
    }
  });
});

describe('zUpToWorld / eciToWorld', () => {
  it('apply M = [[1,0,0],[0,0,1],[0,-1,0]]', () => {
    const v = zUpToWorld([1, 2, 3]);
    expect([v.x, v.y, v.z]).toEqual([1, 3, -2]);
    expect(Math.abs(new THREE.Matrix3().set(1, 0, 0, 0, 0, 1, 0, -1, 0).determinant() - 1)).toBe(0);
  });

  it('match the fixture vectors', () => {
    for (const f of fx.frames) {
      expectVec(zUpToWorld(f.frame.display.sun), f.display_sun_world, 1e-12);
      expectVec(zUpToWorld(f.frame.display.b), f.display_b_world, 1e-12);
      expectVec(eciToWorld(f.frame.eci.r_m, 1), f.r_world_m, 1e-6);
      const scaled = eciToWorld(f.frame.eci.r_m, 1e-6);
      expectVec(scaled, f.r_world_m.map((x) => x * 1e-6), 1e-12);
    }
  });

  it('zenith is world-up in the display scene, nadir is (0, -1, 0)', () => {
    expectVec(zUpToWorld([0, 0, 1]), [0, 1, 0], 1e-15);
    expectVec(zUpToWorld([0, 0, -1]), [0, -1, 0], 1e-15);
  });
});

describe('latLonToEcef', () => {
  it('matches the fixture ground points', () => {
    for (const p of fx.earth) {
      const e = latLonToEcef(p.lat_rad, p.lon_rad, 1);
      for (let i = 0; i < 3; i++) expect(Math.abs(e[i] - p.ecef[i])).toBeLessThan(1e-12);
    }
    expect(latLonToEcef(0, 0, 2)).toEqual([2, 0, 0]);
  });
});

describe('Earth texture orientation (earth.rotation.y = gmst)', () => {
  const geom = new THREE.SphereGeometry(1, 64, 32);
  const pos = geom.getAttribute('position');
  const uv = geom.getAttribute('uv');

  /** Index of the vertex whose UV is nearest (u, v) with v = Three.js uv.y (1 at the north pole). */
  function nearestVertex(u: number, v: number): number {
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < uv.count; i++) {
      const du = uv.getX(i) - u;
      const dv = uv.getY(i) - v;
      const d = du * du + dv * dv;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  it('textured vertices land on the ground points M p_eci', () => {
    expect(fx.earth.length).toBeGreaterThanOrEqual(12);
    for (const p of fx.earth) {
      const mesh = new THREE.Mesh(geom);
      mesh.rotation.y = earthRotationY(p.gmst_rad);
      mesh.updateMatrixWorld();
      // fixture v is measured from the top (north); Three.js uv.y is 1 at the north pole
      const i = nearestVertex(p.u, 1 - p.v);
      const w = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
      const expected = new THREE.Vector3(p.world[0], p.world[1], p.world[2]);
      const angle = w.angleTo(expected);
      expect(angle, `lat ${p.lat_deg} lon ${p.lon_deg} gmst ${p.gmst_rad}`).toBeLessThan(0.06);
    }
  });

  it('is sensitive: a wrong sign or offset in rotation.y fails the same check', () => {
    // guards against the test passing trivially (e.g. all angles small because gmst is small)
    const bad = (rot: (g: number) => number): number => {
      let worst = 0;
      for (const p of fx.earth) {
        const mesh = new THREE.Mesh(geom);
        mesh.rotation.y = rot(p.gmst_rad);
        mesh.updateMatrixWorld();
        const i = nearestVertex(p.u, 1 - p.v);
        const w = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
        worst = Math.max(worst, w.angleTo(new THREE.Vector3(p.world[0], p.world[1], p.world[2])));
      }
      return worst;
    };
    expect(bad((g) => -g)).toBeGreaterThan(0.5);
    expect(bad((g) => g + Math.PI / 2)).toBeGreaterThan(0.5);
  });

  it('local sphere axes are M p_ecef: u = 0.5 is +X, u = 0.75 is -Z, north pole is +Y', () => {
    const at = (u: number, v: number): THREE.Vector3 => {
      const i = nearestVertex(u, v);
      return new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i));
    };
    expectVec(at(0.5, 0.5), [1, 0, 0], 1e-6);
    expectVec(at(0.75, 0.5), [0, 0, -1], 1e-6);
    expectVec(at(0.5, 1), [0, 1, 0], 1e-6);
  });
});
