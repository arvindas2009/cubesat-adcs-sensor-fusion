"""Strapdown dead reckoning (Project A, flat non-rotating Earth, NED as inertial).

Frames and units (CONVENTIONS sections 1, 2, 4): body = PX4 FRD, navigation frame =
local NED; time [s], gyro [rad/s] (FRD), accel = specific force [m/s^2] (FRD),
position [m], velocity [m/s] (NED), quaternions scalar-first Hamilton FRD -> NED.

Sensor model: ``f_b = R(q)^T (a_n - g_n) + b_a``, gyro ``= w_b + b_g``, hence
``a_n = R(q) (f_b - b_a) + g_n`` with ``g_n = [0, 0, +g]``.

Integration scheme (second order in dt for smooth signals)
----------------------------------------------------------
For each interval ``[t_k, t_{k+1}]`` with actual ``dt_k = t_{k+1} - t_k``
(non-uniform grids are fine):

* Attitude -- **trapezoid gyro**: ``w_k = 0.5 (w~_k + w~_{k+1}) - b_g`` is the mean of
  the two bracketing samples, held constant over the interval, and
  ``q_{k+1} = q_k (x) exp(w_k dt_k)`` (this is exactly
  ``common.quaternion.propagate(q_k, w_k, dt_k)``; the unit-norm renormalisation is
  applied once to the accumulated product). The cumulative quaternion product is
  evaluated with a vectorised Hillis-Steele prefix scan (log2 N batched
  ``multiply`` passes, no Python loop over samples).
* Mid-interval attitude: ``q_mid = q_k (x) exp(w_k dt_k / 2)``.
* Velocity -- the specific force is the mean of the two bracketing samples,
  ``f_k = 0.5 (f~_k + f~_{k+1}) - b_a``, rotated by ``q_mid``:
  ``a_k = R(q_mid) f_k + g_n``; ``v_{k+1} = v_k + a_k dt_k``.
* Position -- trapezoid on velocity: ``p_{k+1} = p_k + 0.5 (v_k + v_{k+1}) dt_k``.

Exact for constant ``a_n`` and constant attitude (and for constant body rate with no
specific force); no Earth-rate / transport-rate terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import quaternion as quat
from common.frames import G0


@dataclass
class DeadReckoningResult:
    """Strapdown output on the input time grid.

    t : (N,) s; q : (N, 4) body(FRD)-to-NED quaternion (unit norm, continuous in
    time -- not sign-canonicalised); v : (N, 3) m/s NED; p : (N, 3) m NED, with
    ``p[0] = p0`` (positions are relative to the NED origin the caller uses).
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    p: np.ndarray


def _prefix_product(dq: np.ndarray) -> np.ndarray:
    """Inclusive cumulative Hamilton product ``dq[0] (x) ... (x) dq[k]``, shape (M, 4).

    Hillis-Steele scan: associative, order-preserving, O(M log M) vectorised work.
    """
    out = dq.copy()
    s = 1
    while s < out.shape[0]:
        out[s:] = quat.multiply(out[:-s], out[s:])
        s *= 2
    return out


def strapdown(t, gyro, accel, q0, v0=None, p0=None, *,
              gyro_bias=None, accel_bias=None, g_ned=(0, 0, G0)) -> DeadReckoningResult:
    """Integrate gyro and accelerometer samples into attitude, velocity, position.

    Parameters
    ----------
    t : (N,) s, strictly increasing (need not be uniform).
    gyro : (N, 3) rad/s, FRD body angular rate (raw, incl. bias).
    accel : (N, 3) m/s^2, FRD specific force (raw, incl. bias).
    q0 : (4,) initial FRD -> NED quaternion (required, normalised internally).
    v0, p0 : (3,) initial NED velocity [m/s] / position [m]; default zeros.
    gyro_bias, accel_bias : (3,) constant biases [rad/s], [m/s^2] subtracted from the
        samples; default zeros.
    g_ned : (3,) gravity in NED [m/s^2]; default ``[0, 0, G0]``.

    Returns ``DeadReckoningResult`` (see class docstring). Scheme: see module docstring.
    Raises ``ValueError`` for mismatched lengths/shapes or non-increasing ``t``.
    """
    t = np.asarray(t, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    accel = np.asarray(accel, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("t must be a non-empty 1-D array")
    n = t.size
    if gyro.shape != (n, 3) or accel.shape != (n, 3):
        raise ValueError(
            f"gyro {gyro.shape} and accel {accel.shape} must both be ({n}, 3) to match t")

    def _vec3(x, name, default):
        if x is None:
            return np.asarray(default, dtype=float).copy()
        x = np.asarray(x, dtype=float)
        if x.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {x.shape}")
        return x

    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (4,):
        raise ValueError(f"q0 must have shape (4,), got {q0.shape}")
    q0 = quat.normalize(q0)
    v0 = _vec3(v0, "v0", np.zeros(3))
    p0 = _vec3(p0, "p0", np.zeros(3))
    b_g = _vec3(gyro_bias, "gyro_bias", np.zeros(3))
    b_a = _vec3(accel_bias, "accel_bias", np.zeros(3))
    g = _vec3(g_ned, "g_ned", np.zeros(3))

    q = np.empty((n, 4)); v = np.empty((n, 3)); p = np.empty((n, 3))
    q[0], v[0], p[0] = q0, v0, p0
    if n == 1:
        return DeadReckoningResult(t=t.copy(), q=q, v=v, p=p)

    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError("t must be strictly increasing")

    # Attitude: trapezoid (mean of bracketing samples) body rate per interval.
    w = 0.5 * (gyro[:-1] + gyro[1:]) - b_g
    phi = w * dt[:, None]
    dq = quat.from_rotvec(phi)
    q[1:] = quat.normalize(quat.multiply(q0, _prefix_product(dq)))
    q_mid = quat.normalize(quat.multiply(q[:-1], quat.from_rotvec(0.5 * phi)))

    # Velocity: mean specific force rotated by the mid-interval attitude.
    f = 0.5 * (accel[:-1] + accel[1:]) - b_a
    a_n = quat.rotate(q_mid, f) + g
    v[1:] = v0 + np.cumsum(a_n * dt[:, None], axis=0)
    # Position: trapezoid on velocity.
    p[1:] = p0 + np.cumsum(0.5 * (v[:-1] + v[1:]) * dt[:, None], axis=0)
    return DeadReckoningResult(t=t.copy(), q=q, v=v, p=p)


def initial_attitude_from_accel_mag(accel, mag) -> np.ndarray:
    """Coarse static initial attitude (FRD -> NED quaternion) from accel and magnetometer.

    Parameters
    ----------
    accel : (3,) or (N, 3) m/s^2, FRD specific force of a (near-)static vehicle; an
        ``(N, 3)`` window is averaged over axis 0.
    mag : (3,) or (N, 3) magnetic field, FRD (any consistent unit, e.g. T); averaged
        likewise.

    Method: at rest ``f_b = -g R^T e_z``, so the body-frame down axis is
    ``d = -f/|f|``. East is ``e = (d x m)/|d x m|`` (removes the vertical field
    component, i.e. tilt compensation), north ``n = e x d``. The rows of the
    body-to-NED DCM are ``[n; e; d]``. Heading is relative to **magnetic** north
    (declination = 0). Returns a canonical (w >= 0) unit quaternion, shape (4,).

    Raises ``ValueError`` if the specific force is ~0 or the field is (anti)parallel
    to gravity (heading undefined).
    """
    f = np.asarray(accel, dtype=float)
    m = np.asarray(mag, dtype=float)
    f = f.mean(axis=0) if f.ndim == 2 else f
    m = m.mean(axis=0) if m.ndim == 2 else m
    if f.shape != (3,) or m.shape != (3,):
        raise ValueError("accel and mag must be (3,) or (N, 3)")
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        raise ValueError("specific force has ~zero norm; cannot level")
    d = -f / nf
    e = np.cross(d, m)
    ne = np.linalg.norm(e)
    if ne < 1e-12 * max(np.linalg.norm(m), 1e-300):
        raise ValueError("magnetic field parallel to gravity; heading undefined")
    e /= ne
    n = np.cross(e, d)
    return quat.canonicalize(quat.from_dcm(np.vstack([n, e, d])))
