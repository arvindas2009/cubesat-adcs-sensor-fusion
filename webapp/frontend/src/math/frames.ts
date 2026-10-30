// ALL coordinate conversions between the simulation frames and Three.js world coordinates live here
// (docs/specs/frontend.md §1). Pure functions, no DOM; tested against Python-generated fixtures.
//
// Three.js is Y-up. M maps a right-handed "z-up" frame into world coordinates:
//   M (x, y, z) = (x, z, -y),  det M = +1  (a -90 degree rotation about x).
// Quaternions on the wire are scalar-first [w, x, y, z]; Three.js stores (x, y, z, w).
import * as THREE from 'three';

/** Rows of M, as a plain 3x3 (row-major). */
export const M: readonly (readonly [number, number, number])[] = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

/** M as a quaternion: rotation by -90 degrees about +x. */
const M_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

/** Wire quaternion [w, x, y, z] -> THREE.Quaternion(x, y, z, w). Same rotation as `to_dcm(q)`. */
export function toThreeQuaternion(q: ArrayLike<number>): THREE.Quaternion {
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

/** M v: a vector in a z-up frame (ECI, display) as world coordinates. */
export function zUpToWorld(v: ArrayLike<number>): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[2], -v[1]);
}

/** ECI position (m) -> world position, `scale` world units per metre (1 unit = 1000 km: 1e-6). */
export function eciToWorld(r_m: ArrayLike<number>, scale: number): THREE.Vector3 {
  return zUpToWorld(r_m).multiplyScalar(scale);
}

/**
 * `rotation.y` of the Earth mesh (which sits directly in world coordinates, with a SphereGeometry
 * whose local coordinates are M p_ecef): equals GMST. p_eci = Rz(gmst)^T p_ecef is an active rotation by
 * +gmst about ECI z, which is a rotation by +gmst about world +Y after M. Verified by the Earth
 * texture test in tests/frames.test.ts.
 */
export function earthRotationY(gmst_rad: number): number {
  return gmst_rad;
}

/**
 * Display-frame body attitude C (row-major 3x3; column k = body axis k in display coordinates) ->
 * the local-to-world rotation M C of a body-fixed mesh in the display scene.
 */
export function displayMatrixToWorld(C: readonly (readonly number[])[]): THREE.Matrix4 {
  const R = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += M[i][k] * C[k][j];
      R[i][j] = s;
    }
  }
  return new THREE.Matrix4().set(
    R[0][0], R[0][1], R[0][2], 0,
    R[1][0], R[1][1], R[1][2], 0,
    R[2][0], R[2][1], R[2][2], 0,
    0, 0, 0, 1,
  );
}

/** Attitude of a body-fixed mesh in the ECI scene: quat(M) * q_wire. */
export function bodyInEciToWorld(q: ArrayLike<number>): THREE.Quaternion {
  return M_QUAT.clone().multiply(toThreeQuaternion(q));
}

/** Spherical lat/lon (rad) and radius R -> ECEF [x, y, z] (z = polar axis, lon 0 on +x). */
export function latLonToEcef(lat: number, lon: number, R: number): [number, number, number] {
  const c = Math.cos(lat);
  return [R * c * Math.cos(lon), R * c * Math.sin(lon), R * Math.sin(lat)];
}
