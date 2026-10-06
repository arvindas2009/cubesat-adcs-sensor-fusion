"""Attitude control laws for the 3U CubeSat (Project B, Group 6 / M7).

Conventions (CONVENTIONS.md): SI units; ``q`` = body-to-ECI quaternion,
scalar-first Hamilton; ``w`` = body angular velocity relative to ECI in body
axes (rad/s); ``h_w`` = total reaction-wheel momentum in body axes (N m s);
magnetic field ``b_b`` in tesla, body axes; dipoles in A m^2; torques in N m.
Spec: ``docs/specs/adcs_control.md`` section 2.

Laws
----
* **B-dot detumble** on the field direction (Avanzini & Giulietti 2012)::

      m = -(k/|B|) d(b_hat)/dt,   k = 2 n (1 + sin xi_m) J_min

  With ``d(b_hat)/dt = -w x b_hat`` (the inertial field change is ~2n, small
  while tumbling) the torque is ``m x B = k (w x b_hat) x b_hat = -k w_perp``:
  the component of ``w`` perpendicular to ``B`` decays with time constant
  ``J/k`` and ``w . tau = -k |w_perp|^2 <= 0`` (kinetic energy never grows).
* **Nadir pointing**: rate-limited quaternion-error PD with gyroscopic and
  reference-rate feed-forward. With ``err = rotvec(conj(q_ref) q)`` (body
  axes, CONVENTIONS section 3) and ``dw = w - w_ref_b``::

      tau = -Kd (dw - w_cmd) + w x (I w + h_w) - I (w x w_ref_b),
      w_cmd = -sat(wn/(2 zeta) err, rate_limit),  Kp = wn^2 I, Kd = 2 zeta wn I

  Derivation: ``w_ref_b = C w_ref`` with ``C = R(dq)^T``; ``C_dot = -[dw x] C``
  and ``w_ref`` is constant in LVLH axes (circular orbit), so
  ``d(w_ref_b)/dt = -dw x w_ref_b = -w x w_ref_b``. With
  ``I w_dot = -w x (I w + h_w) + tau + tau_dist`` this gives
  ``I d(dw)/dt = -Kp err - Kd dw + tau_dist`` while ``w_cmd`` is unsaturated.
  ``rate_limit`` bounds the slew rate so that a large initial error cannot
  push the (small) wheels into saturation.
* **Momentum dumping** (cross-product law)::

      m = (k/|B|^2) (h_w - h_target) x B   ->   m x B = -k (h_w - h_target)_perp

  The external torque removes the wheel momentum perpendicular to ``B``; the
  pointing loop keeps the attitude, so the wheels unload. Switched on/off
  with hysteresis on the largest wheel speed (``dump_active``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import quaternion as quat
from common.frames import GM_EARTH, dcm_eci_to_lvlh
from adcs.actuators import saturate_dipole
from adcs.dynamics import _cross

__all__ = [
    "bdot_gain",
    "bdot_dipole",
    "nadir_reference",
    "PointingGains",
    "pointing_torque",
    "dump_dipole",
    "dump_active",
]


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def bdot_gain(inertia, a: float, inc: float, mu: float = GM_EARTH) -> float:
    """B-dot gain ``k = 2 n (1 + sin xi_m) J_min`` (N m s), Avanzini & Giulietti (2012).

    ``inertia`` (3, 3) kg m^2; ``a`` semi-major axis (m); ``inc`` orbit
    inclination (rad), used as the geomagnetic inclination ``xi_m`` (the dipole
    tilt of ~10 deg moves the true value around it).
    """
    n = np.sqrt(mu / float(a) ** 3)
    j_min = float(np.min(np.linalg.eigvalsh(_v(inertia))))
    return float(2.0 * n * (1.0 + np.sin(float(inc))) * j_min)


def bdot_dipole(b_b, b_b_prev, dt: float, k: float, m_max: float) -> np.ndarray:
    """B-dot dipole command (A m^2, body axes) from two field samples ``dt`` s apart.

    ``m = -(k/|B|) (b_hat - b_hat_prev)/dt`` with ``b_b`` the current and
    ``b_b_prev`` the previous field sample (T, body axes), ``k`` in N m s;
    saturated by scaling so that ``max |m_i| <= m_max``.
    """
    b = _v(b_b)
    bp = _v(b_b_prev)
    bn = np.linalg.norm(b)
    bhat_dot = (b / bn - bp / np.linalg.norm(bp)) / float(dt)
    return saturate_dipole(-(float(k) / bn) * bhat_dot, m_max, "scale")


def nadir_reference(r_eci, v_eci):
    """Nadir (LVLH) attitude reference.

    Returns ``(q_ref, w_ref)``: ``q_ref`` (..., 4) body-to-ECI quaternion of a body
    aligned with LVLH (w >= 0); ``w_ref`` (..., 3) angular velocity of LVLH
    relative to ECI expressed **in LVLH axes**, ``[0, -|r x v|/|r|^2, 0]`` rad/s
    (LVLH y is the negative orbit normal).
    """
    r = _v(r_eci)
    v = _v(v_eci)
    q_ref = quat.from_dcm(np.swapaxes(dcm_eci_to_lvlh(r, v), -1, -2))
    rate = np.linalg.norm(np.cross(r, v), axis=-1) / np.sum(r * r, axis=-1)
    w_ref = np.zeros(r.shape)
    w_ref[..., 1] = -rate
    return q_ref, w_ref


@dataclass(frozen=True)
class PointingGains:
    """Rate-limited PD gains: ``Kp = wn^2 I``, ``Kd = 2 zeta wn I``."""

    wn: float = 0.1                                  # rad/s, closed-loop natural frequency
    zeta: float = 0.7                                # damping ratio
    rate_limit: float = float(np.deg2rad(0.5))       # rad/s, maximum commanded slew rate

    def __post_init__(self):
        if not (self.wn > 0 and self.zeta > 0 and self.rate_limit > 0):
            raise ValueError("PointingGains: wn, zeta and rate_limit must be > 0")


def _sat_norm(x: np.ndarray, limit: float) -> np.ndarray:
    nx = np.linalg.norm(x)
    return x * (limit / nx) if nx > limit else x


def pointing_torque(q, w, h_w, q_ref, w_ref, inertia, gains: PointingGains):
    """Desired body torque for tracking ``q_ref`` (N m, body axes) and the attitude error.

    ``q`` body-to-ECI; ``w`` body rate (rad/s, body axes); ``h_w`` wheel
    momentum (N m s, body axes); ``q_ref`` reference body-to-ECI attitude;
    ``w_ref`` reference angular velocity relative to ECI in **reference axes**
    (rad/s, constant in those axes); ``inertia`` (3, 3) kg m^2.
    Returns ``(tau, err)`` with ``err = to_rotvec(conj(q_ref) q)`` (rad, body
    axes, shortest rotation). See the module docstring for the law.
    """
    inertia = _v(inertia)
    w = _v(w)
    dq = quat.multiply(quat.conjugate(_v(q_ref)), quat.normalize(_v(q)))
    err = quat.to_rotvec(dq)
    w_ref_b = quat.rotate_inverse(dq, _v(w_ref))
    dw = w - w_ref_b
    w_cmd = -_sat_norm(gains.wn / (2.0 * gains.zeta) * err, gains.rate_limit)
    kd = 2.0 * gains.zeta * gains.wn * inertia
    tau = (-kd @ (dw - w_cmd)
           + _cross(w, inertia @ w + _v(h_w))
           - inertia @ _cross(w, w_ref_b))
    return tau, err


def dump_dipole(h_w, b_b, k: float, m_max: float, h_target=0.0) -> np.ndarray:
    """Cross-product momentum-dumping dipole (A m^2, body axes).

    ``m = (k/|B|^2) ((h_w - h_target) x b_b)``, ``k`` in 1/s, ``h_w`` N m s,
    ``b_b`` T (body axes); saturated by scaling (``max |m_i| <= m_max``).
    Unsaturated, ``m x b_b = -k (h_w - h_target)_perp``.
    """
    b = _v(b_b)
    dh = _v(h_w) - _v(h_target)
    m = (float(k) / float(b @ b)) * _cross(dh, b)
    return saturate_dipole(m, m_max, "scale")


def dump_active(prev: bool, speed_frac: float, on_frac: float, off_frac: float) -> bool:
    """Hysteresis switch for momentum dumping.

    ``speed_frac = max_i |Omega_i| / max_speed``. Returns ``True`` at or above
    ``on_frac``, ``False`` at or below ``off_frac``, otherwise ``prev``.
    """
    if speed_frac >= on_frac:
        return True
    if speed_frac <= off_frac:
        return False
    return bool(prev)
