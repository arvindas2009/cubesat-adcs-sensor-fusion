// NED -> Three.js world conversions for the PX4 replay page (docs/specs/px4_replay.md §3.2).
// Pure functions, no DOM; tested against Python-generated fixtures (tests/ned.test.ts).
//
// Three.js is Y-up. One fixed proper rotation N maps NED into world coordinates:
//   N (n, e, d) = (n, -d, e),  det N = +1.
// North is world +X, up is world +Y, east is world +Z. Quaternions on the wire are scalar-first
// [w, x, y, z], body (FRD) to NED; the Three.js quaternion goes through frames.ts only.
import * as THREE from 'three';
import { toThreeQuaternion } from './frames';

/** Rows of N, as a plain 3x3 (row-major). */
export const N: readonly (readonly [number, number, number])[] = [
  [1, 0, 0],
  [0, 0, -1],
  [0, 1, 0],
];

/** N as a quaternion. N maps (n, e, d) -> (n, -d, e): a +90 degree rotation about world +X. */
const N_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

/** det of the 3x3 matrix `A` (used by the tests to confirm N is a proper rotation). */
export function det3(A: readonly (readonly number[])[]): number {
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

/** N v * scale: a NED vector (metres) as world coordinates. */
export function nedToWorld(v: ArrayLike<number>, scale = 1): THREE.Vector3 {
  return new THREE.Vector3(v[0], -v[2], v[1]).multiplyScalar(scale);
}

/**
 * Attitude of a mesh modelled in FRD body coordinates (x forward, y right, z down): quat(N) * q_wire.
 * The mesh then shows N R(q) p_body.
 */
export function bodyNedToWorld(q: ArrayLike<number>): THREE.Quaternion {
  return N_QUAT.clone().multiply(toThreeQuaternion(q));
}
