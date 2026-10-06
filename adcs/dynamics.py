"""Rigid-body attitude dynamics with reaction wheels (Project B).

State vector ``y = [q (4), w (3), h_w (3)]`` (length 10):

* ``q``   body-to-ECI quaternion, scalar-first Hamilton (CONVENTIONS.md section 2),
* ``w``   body angular velocity relative to ECI, **body axes** (rad/s),
* ``h_w`` total reaction-wheel angular momentum, **body axes** (N m s).

``I`` is the 3x3 spacecraft inertia about the CoM (kg m^2, body axes); it
includes the wheel rotors' transverse inertia (wheels are modelled only by their
angular momentum). ``h_w`` is the *total* wheel momentum, with the rotors'
spin-axis inertia **excluded** from ``I`` (that is what makes ``u_w`` the motor
torque). Group 6 gets the wheel speed from ``Omega = (h_w . a)/J_s - a . w``
for spin axis ``a`` and spin inertia ``J_s``.

Equations of motion (the contract, torques in N m, body axes)::

    q_dot   = 1/2 q (x) [0, w]
    I w_dot = -w x (I w + h_w) - u_w + tau_ext
    h_w_dot = u_w

``u_w`` is the torque applied *to the wheels*; the reaction ``-u_w`` acts on the
body. With ``u_w = 0`` and ``tau_ext = 0`` the inertial angular momentum
``H_i = R(q) (I w + h_w)`` is constant, and so is ``T = 1/2 w' I w`` while
``h_w`` is constant (``w . (w x h_w) = 0``). Wheel speed limits and saturation
are not modelled here (Group 6).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from common import quaternion as quat
from common.frames import dcm_eci_to_lvlh

__all__ = [
    "derivative",
    "rk4_step",
    "integrate",
    "angular_momentum_inertial",
    "kinetic_energy",
    "lvlh_error",
    "nadir_angle",
]


def _as_torque_fn(torque):
    """Return ``f(t, q, w, h_w) -> (3,)`` from a callable, a 3-vector or ``None`` (zeros)."""
    if torque is None:
        const = np.zeros(3)
    elif callable(torque):
        return lambda t, q, w, h: np.asarray(torque(t, q, w, h), dtype=float).reshape(3)
    else:
        const = np.asarray(torque, dtype=float).reshape(3)
    return lambda t, q, w, h: const


def derivative(t, y, inertia, wheel_torque=None, ext_torque=None) -> np.ndarray:
    """Time derivative of the state ``y = [q, w, h_w]`` (shape ``(10,)``).

    ``t`` s; ``q`` body-to-ECI; ``w`` rad/s (body axes); ``h_w`` N m s (body
    axes); ``inertia`` (3, 3) kg m^2. ``wheel_torque`` (``u_w``, applied to the
    wheels) and ``ext_torque`` (``tau_ext`` on the body) are N m in body axes:
    callables ``f(t, q, w, h_w) -> (3,)`` or constant 3-vectors; ``None`` is zero.
    Returns ``[q_dot, w_dot, h_w_dot]``. ``q`` is used as given (not normalised).
    """
    y = np.asarray(y, dtype=float)
    q, w, h = y[0:4], y[4:7], y[7:10]
    inertia = np.asarray(inertia, dtype=float)
    u = _as_torque_fn(wheel_torque)(t, q, w, h)
    tau = _as_torque_fn(ext_torque)(t, q, w, h)
    w_dot = np.linalg.solve(inertia, -np.cross(w, inertia @ w + h) - u + tau)
    out = np.empty(10)
    out[0:4] = quat.derivative(q, w)
    out[4:7] = w_dot
    out[7:10] = u
    return out


def rk4_step(t, y, dt, inertia, wheel_torque=None, ext_torque=None) -> np.ndarray:
    """One classic RK4 step of size ``dt`` (s) from time ``t`` (s); returns ``y_next`` (10,).

    Same units/frames as :func:`derivative`. ``q`` is renormalised afterwards.
    """
    y = np.asarray(y, dtype=float)
    uf = _as_torque_fn(wheel_torque)
    ef = _as_torque_fn(ext_torque)

    def f(tt, yy):
        return derivative(tt, yy, inertia, uf, ef)

    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    y_next[0:4] = quat.normalize(y_next[0:4])
    return y_next


def integrate(y0, t, inertia, wheel_torque=None, ext_torque=None,
              rtol=1e-10, atol=1e-12, normalize=True) -> np.ndarray:
    """Integrate the dynamics with ``solve_ivp`` (DOP853) at the times ``t``.

    ``y0`` (10,) initial state at ``t[0]``; ``t`` increasing 1-D array of times
    (s). Torques as in :func:`derivative` (N m, body axes). Returns ``y`` of
    shape ``(N, 10)`` with ``q`` renormalised at the output unless
    ``normalize=False`` (raw DOP853 output, to measure the integrator's own
    quaternion-norm drift: ~1.5e-10 over 10 orbits at ``rtol=1e-11``).
    """
    y0 = np.asarray(y0, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if t.ndim != 1:
        raise ValueError("t must be a 1-D array")
    if t.size == 1:
        out = y0[None, :].copy()
    else:
        uf = _as_torque_fn(wheel_torque)
        ef = _as_torque_fn(ext_torque)
        sol = solve_ivp(lambda tt, yy: derivative(tt, yy, inertia, uf, ef),
                        (t[0], t[-1]), y0, method="DOP853", t_eval=t,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"dynamics integration failed: {sol.message}")
        out = sol.y.T.copy()
    if normalize:
        out[:, 0:4] = quat.normalize(out[:, 0:4])
    return out


def angular_momentum_inertial(y, inertia) -> np.ndarray:
    """Total angular momentum in ECI axes, ``R(q) (I w + h_w)`` (N m s).

    ``y`` shape ``(..., 10)``; returns ``(..., 3)``.
    """
    y = np.asarray(y, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    h_b = y[..., 4:7] @ inertia.T + y[..., 7:10]
    return quat.rotate(quat.normalize(y[..., 0:4]), h_b)


def kinetic_energy(y, inertia) -> np.ndarray:
    """Body rotational kinetic energy ``1/2 w' I w`` (J); ``y`` shape ``(..., 10)`` -> ``(...)``."""
    y = np.asarray(y, dtype=float)
    w = y[..., 4:7]
    inertia = np.asarray(inertia, dtype=float)
    return 0.5 * np.sum(w * (w @ inertia.T), axis=-1)


def lvlh_error(q, r_eci, v_eci) -> np.ndarray:
    """Attitude error rotation vector w.r.t. the LVLH nadir target (rad, body axes).

    ``q`` body-to-ECI, ``r_eci`` (m), ``v_eci`` (m/s), batched over leading dims.
    ``q_lvlh = from_dcm(dcm_eci_to_lvlh(r, v).T)`` is the body-to-ECI attitude of
    a body aligned with LVLH; returns ``to_rotvec(conj(q_lvlh) (x) q)`` so that
    ``q = q_lvlh (x) exp(err)`` (CONVENTIONS.md section 3, error in body axes).
    """
    q_lvlh = quat.from_dcm(np.swapaxes(dcm_eci_to_lvlh(r_eci, v_eci), -1, -2))
    return quat.to_rotvec(quat.multiply(quat.conjugate(q_lvlh), quat.normalize(q)))


def nadir_angle(q, r_eci, v_eci) -> np.ndarray:
    """Total attitude error angle (rad) w.r.t. the LVLH nadir target: ``|lvlh_error|``."""
    return np.linalg.norm(lvlh_error(q, r_eci, v_eci), axis=-1)
