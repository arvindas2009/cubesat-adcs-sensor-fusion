"""Quaternion algebra (scalar-first, Hamilton) for attitude representation.

Conventions (see ``CONVENTIONS.md`` section 2 and 3)
----------------------------------------------------
- Storage: scalar-first ``q = [w, x, y, z]``, float arrays of shape ``(4,)`` or
  batched ``(..., 4)``. Leading dimensions broadcast with numpy rules.
- Algebra: Hamilton product, ``i^2 = j^2 = k^2 = ijk = -1``.
- Meaning: a quaternion attached to a body is the **body-to-reference**
  rotation, ``v_ref = R(q) v_body = q (x) [0, v_body] (x) q*``. Composition is
  ``q_AC = q_AB (x) q_BC``.
- Unit norm is the working assumption; ``to_dcm``, ``rotate``, ``to_rotvec``
  and friends normalise internally. Where a unique representative of a rotation
  is needed (``from_dcm``, ``to_rotvec``) the sign is chosen so that ``w >= 0``.
- Kinematics: ``q_dot = 1/2 q (x) [0, omega_b]`` with ``omega_b`` the body rate
  relative to the reference frame, expressed in body axes. Constant-rate
  propagation: ``q_{k+1} = q_k (x) exp(omega_b dt)``.
- Error state: attitude errors are **right / local** perturbations expressed
  in body axes, ``q_true = q_est (x) exp(d_theta)``; ``error_rotvec(q_ref, q)``
  returns ``e`` with ``q = q_ref (x) exp(e)``.
- Euler angles (display only): aerospace ZYX, ``R = Rz(yaw) Ry(pitch) Rx(roll)``.

Angles are in radians, angular rates in rad/s, time in seconds. Functions never
mutate their inputs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "identity",
    "skew",
    "multiply",
    "conjugate",
    "inverse",
    "norm",
    "normalize",
    "canonicalize",
    "rotate",
    "rotate_inverse",
    "to_dcm",
    "from_dcm",
    "from_axis_angle",
    "from_rotvec",
    "to_rotvec",
    "from_euler_zyx",
    "to_euler_zyx",
    "small_angle_quat",
    "error_rotvec",
    "angle_between",
    "derivative",
    "omega_matrix",
    "propagate",
    "slerp",
    "inject_attitude_error",
    "attitude_reset_jacobian",
]

_NORM_EPS = 1e-12
_TAYLOR_ANGLE = 1e-8  # |phi| below which from_rotvec uses its Taylor series
_SMALL_VEC = 1e-4  # |v| below which to_rotvec uses its series
_GIMBAL_EPS = 1e-8  # cos(pitch) below which to_euler_zyx treats gimbal lock


def _f(x) -> np.ndarray:
    """Convert an array-like to a float ndarray (no copy if already float)."""
    return np.asarray(x, dtype=float)


def identity() -> np.ndarray:
    """Identity rotation ``[1, 0, 0, 0]``, shape ``(4,)``."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def skew(v) -> np.ndarray:
    """Cross-product matrix ``[v x]`` of ``v (..., 3)`` -> ``(..., 3, 3)``.

    ``skew(a) @ b == np.cross(a, b)``. Units are those of ``v``.
    """
    v = _f(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )


def multiply(p, q) -> np.ndarray:
    """Hamilton product ``p (x) q`` of quaternions ``(..., 4)`` (broadcast)."""
    p = _f(p)
    q = _f(q)
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        axis=-1,
    )


def conjugate(q) -> np.ndarray:
    """Conjugate ``[w, -x, -y, -z]`` of ``q (..., 4)``."""
    return _f(q) * np.array([1.0, -1.0, -1.0, -1.0])


def inverse(q) -> np.ndarray:
    """Inverse ``conjugate(q) / |q|^2`` of ``q (..., 4)``."""
    q = _f(q)
    return conjugate(q) / np.sum(q * q, axis=-1, keepdims=True)


def norm(q) -> np.ndarray:
    """Euclidean norm of ``q (..., 4)`` -> ``(...)``."""
    return np.linalg.norm(_f(q), axis=-1)


def normalize(q) -> np.ndarray:
    """Return ``q / |q|`` for ``q (..., 4)``.

    Raises
    ------
    ValueError
        If any quaternion has norm below 1e-12.
    """
    q = _f(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < _NORM_EPS):
        raise ValueError("cannot normalize quaternion with norm < 1e-12")
    return q / n


def canonicalize(q) -> np.ndarray:
    """Return ``q`` or ``-q`` such that ``w >= 0``.

    If ``w == 0`` the first non-zero component of ``(x, y, z)`` is made positive.
    """
    q = _f(q)
    key = np.where(
        q[..., 0] != 0,
        q[..., 0],
        np.where(q[..., 1] != 0, q[..., 1], np.where(q[..., 2] != 0, q[..., 2], q[..., 3])),
    )
    return np.where((key < 0)[..., None], -q, q)


def rotate(q, v) -> np.ndarray:
    """Rotate ``v (..., 3)`` by ``q``: ``R(q) v`` (body -> reference).

    ``q`` is normalised first; equals ``to_dcm(q) @ v``.
    """
    q = normalize(q)
    v = _f(v)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rotate_inverse(q, v) -> np.ndarray:
    """Rotate ``v (..., 3)`` by the inverse of ``q``: ``R(q)^T v`` (reference -> body)."""
    return rotate(conjugate(q), v)


def to_dcm(q) -> np.ndarray:
    """Direction cosine matrix ``R(q)`` (body -> reference), ``(..., 3, 3)``.

    ``q`` is normalised first. Layout per CONVENTIONS section 2.
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def from_dcm(R) -> np.ndarray:
    """Quaternion ``(..., 4)`` (canonical, ``w >= 0``) of rotation matrix ``R (..., 3, 3)``.

    Shepperd's method, vectorised: all four candidate quaternions are formed
    and the one with the largest pivot (``w^2, x^2, y^2, z^2`` from the trace and
    diagonal) is selected per element. Robust for all rotations including 180 deg.
    """
    R = _f(R)
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4 * (pivot)^2 for each candidate.
    c = np.stack(
        [
            1 + r00 + r11 + r22,
            1 + r00 - r11 - r22,
            1 - r00 + r11 - r22,
            1 - r00 - r11 + r22,
        ],
        axis=-1,
    )
    k = np.argmax(c, axis=-1)
    cmax = np.take_along_axis(c, k[..., None], axis=-1)[..., 0]
    s = 2.0 * np.sqrt(np.maximum(cmax, 0.0))  # = 4 * pivot
    s = np.where(s > 0, s, 1.0)  # guard (only reachable for degenerate input)

    q_w = np.stack([s / 4, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s], axis=-1)
    q_x = np.stack([(r21 - r12) / s, s / 4, (r01 + r10) / s, (r02 + r20) / s], axis=-1)
    q_y = np.stack([(r02 - r20) / s, (r01 + r10) / s, s / 4, (r12 + r21) / s], axis=-1)
    q_z = np.stack([(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, s / 4], axis=-1)
    cands = np.stack([q_w, q_x, q_y, q_z], axis=-2)  # (..., 4 candidates, 4)
    q = np.take_along_axis(cands, k[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return canonicalize(q)


def from_axis_angle(axis, angle) -> np.ndarray:
    """Quaternion of a rotation by ``angle`` [rad] about ``axis (..., 3)``.

    ``axis`` is normalised internally (must be non-zero); ``axis`` and ``angle``
    broadcast over leading dimensions.
    """
    axis = _normalize_vec(axis)
    angle = _f(angle)
    half = 0.5 * angle
    xyz = np.sin(half)[..., None] * axis
    w = np.broadcast_to(np.cos(half)[..., None], xyz.shape[:-1] + (1,))
    return np.concatenate([w, xyz], axis=-1)


def _normalize_vec(v) -> np.ndarray:
    """Unit vector ``v / |v|`` for ``v (..., 3)``; ``ValueError`` if ``|v| < 1e-12``."""
    v = _f(v)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < _NORM_EPS):
        raise ValueError("cannot normalize vector with norm < 1e-12")
    return v / n


def from_rotvec(phi) -> np.ndarray:
    """Exponential map: rotation vector ``phi (..., 3)`` [rad] -> quaternion ``(..., 4)``.

    ``exp(phi) = [cos(|phi|/2), sin(|phi|/2) phi/|phi|]``. For ``|phi| < 1e-8``
    a Taylor series is used (no division by zero); ``phi = 0`` gives exactly the
    identity.
    """
    phi = _f(phi)
    a = np.linalg.norm(phi, axis=-1)
    small = a < _TAYLOR_ANGLE
    a_safe = np.where(small, 1.0, a)
    half = 0.5 * a
    # sin(a/2)/a  ->  1/2 - a^2/48
    scale = np.where(small, 0.5 - a * a / 48.0, np.sin(0.5 * a_safe) / a_safe)
    w = np.where(small, 1.0 - a * a / 8.0, np.cos(half))
    return np.concatenate([w[..., None], scale[..., None] * phi], axis=-1)


def to_rotvec(q) -> np.ndarray:
    """Logarithm map: quaternion ``(..., 4)`` -> rotation vector ``(..., 3)`` [rad].

    Inverse of :func:`from_rotvec` with ``|phi| <= pi`` (``q`` is normalised and
    canonicalised to ``w >= 0`` first). Uses ``angle = 2 atan2(|v|, w)`` and a
    series for tiny ``|v|`` to avoid 0/0.
    """
    q = canonicalize(normalize(q))
    w = q[..., 0]
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    small = s < _SMALL_VEC
    s_safe = np.where(small, 1.0, s)
    # angle / |v| = 2 atan(s/w)/s ;  series: (2/w)(1 - r^2/3 + r^4/5), r = s/w
    w_safe = np.where(w > 0, w, 1.0)
    r2 = (s / w_safe) ** 2
    scale_small = (2.0 / w_safe) * (1.0 - r2 / 3.0 + r2 * r2 / 5.0)
    scale = np.where(small, scale_small, 2.0 * np.arctan2(s_safe, w) / s_safe)
    return scale[..., None] * v


def from_euler_zyx(roll, pitch, yaw) -> np.ndarray:
    """Quaternion of ``R = Rz(yaw) Ry(pitch) Rx(roll)``; angles [rad], broadcast.

    Returns shape ``broadcast(roll, pitch, yaw).shape + (4,)``.
    """
    roll, pitch, yaw = np.broadcast_arrays(_f(roll), _f(pitch), _f(yaw))
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def to_euler_zyx(q):
    """Aerospace ZYX Euler angles ``(roll, pitch, yaw)`` [rad] of ``q (..., 4)``.

    ``R(q) = Rz(yaw) Ry(pitch) Rx(roll)``; roll and yaw in ``(-pi, pi]``, pitch in
    ``[-pi/2, pi/2]``. The arcsin argument is clipped to ``[-1, 1]``. At gimbal
    lock (``|pitch| = pi/2``) roll is set to 0 and the remaining rotation is
    assigned to yaw.
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    r10 = 2 * (x * y + w * z)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)

    pitch = -np.arcsin(np.clip(r20, -1.0, 1.0))
    gimbal = np.hypot(r21, r22) < _GIMBAL_EPS
    roll = np.where(gimbal, 0.0, np.arctan2(r21, r22))
    yaw = np.where(gimbal, np.arctan2(-r01, r11), np.arctan2(r10, r00))
    return roll, pitch, yaw


def small_angle_quat(dtheta) -> np.ndarray:
    """Quaternion ``normalize([1, dtheta/2])`` for small rotation vector ``dtheta (..., 3)`` [rad]."""
    dtheta = _f(dtheta)
    one = np.ones(dtheta.shape[:-1] + (1,))
    return normalize(np.concatenate([one, 0.5 * dtheta], axis=-1))


def error_rotvec(q_ref, q) -> np.ndarray:
    """Body-frame attitude error ``e`` [rad] with ``q = q_ref (x) exp(e)``.

    Computed as ``to_rotvec(conjugate(q_ref) (x) q)``; ``|e| <= pi``.
    """
    return to_rotvec(multiply(conjugate(q_ref), q))


def angle_between(q1, q2) -> np.ndarray:
    """Principal angle [rad] in ``[0, pi]`` of the relative rotation ``q1* (x) q2``.

    Invariant to the sign of either quaternion and to their norms.
    """
    rel = multiply(conjugate(q1), q2)
    return 2.0 * np.arctan2(np.linalg.norm(rel[..., 1:], axis=-1), np.abs(rel[..., 0]))


def derivative(q, omega) -> np.ndarray:
    """Quaternion rate ``1/2 q (x) [0, omega]`` [1/s]; ``omega`` is body rate [rad/s] in body axes."""
    omega = _f(omega)
    p = np.concatenate([np.zeros(omega.shape[:-1] + (1,)), omega], axis=-1)
    return 0.5 * multiply(q, p)


def omega_matrix(omega) -> np.ndarray:
    """Matrix ``Omega(omega)`` (..., 4, 4) with ``derivative(q, omega) == 1/2 Omega @ q``.

    ``Omega = [[0, -omega^T], [omega, -[omega x]]]``; ``omega`` is body rate [rad/s].
    """
    omega = _f(omega)
    batch = omega.shape[:-1]
    out = np.zeros(batch + (4, 4))
    out[..., 0, 1:] = -omega
    out[..., 1:, 0] = omega
    out[..., 1:, 1:] = -skew(omega)
    return out


def propagate(q, omega, dt) -> np.ndarray:
    """Propagate ``q`` over ``dt`` [s] at constant body rate ``omega`` [rad/s].

    ``normalize(q (x) from_rotvec(omega dt))``; exact for constant body rate.
    """
    omega = _f(omega)
    dt = _f(dt)
    return normalize(multiply(q, from_rotvec(omega * dt[..., None])))


def slerp(q0, q1, t) -> np.ndarray:
    """Spherical linear interpolation along the shortest arc from ``q0`` to ``q1``.

    ``t`` in ``[0, 1]`` (array ``t`` broadcasts against the quaternion batch
    dimensions). ``t = 0`` gives ``q0``; ``t = 1`` gives ``q1`` or ``-q1`` (whichever
    lies on the shortest arc). Inputs are normalised; the result is unit norm.
    """
    q0 = normalize(q0)
    q1 = normalize(q1)
    t = _f(t)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    # half-angle between them, accurate for both small and large angles
    theta = 2.0 * np.arctan2(
        np.linalg.norm(q1 - q0, axis=-1, keepdims=True),
        np.linalg.norm(q1 + q0, axis=-1, keepdims=True),
    )
    small = theta < 1e-6
    sin_theta = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
    w1 = np.where(small, t, np.sin(t * theta) / sin_theta)
    return normalize(w0 * q0 + w1 * q1)


def inject_attitude_error(q, dtheta) -> np.ndarray:
    """Inject a body-frame attitude error estimate into a nominal quaternion.

    ``normalize(q (x) exp(dtheta))`` -- the error-state convention
    ``q_true = q_est (x) exp(dtheta)`` of CONVENTIONS.md section 3. Shared by
    every error-state filter in the repo so the injection sign is defined once.
    ``q (..., 4)``, ``dtheta (..., 3)`` [rad].
    """
    return normalize(multiply(q, from_rotvec(dtheta)))


def attitude_reset_jacobian(dtheta) -> np.ndarray:
    """Attitude block of the error-state reset Jacobian, ``I - 1/2 [dtheta x]``.

    After injecting ``dtheta`` (see :func:`inject_attitude_error`) the attitude
    error covariance block is mapped with ``P_att <- G P_att G^T`` (Sola 2017,
    "Quaternion kinematics for the error-state Kalman filter", Eq. 287).
    ``dtheta (..., 3)`` -> ``(..., 3, 3)``.
    """
    dtheta = _f(dtheta)
    return np.broadcast_to(np.eye(3), dtheta.shape[:-1] + (3, 3)) - 0.5 * skew(dtheta)
