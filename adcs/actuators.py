"""Actuator models for the 3U CubeSat ADCS (Project B): magnetorquers and reaction wheels.

Frames and units (CONVENTIONS.md): SI throughout, all vectors in **spacecraft body
axes**, angular rates are body rates relative to ECI (rad/s), time on axis 0 for
series, leading dimensions broadcast where stated.

Hardware (recorded in ``assumptions.md`` [B-05], [B-06]):

* **Magnetorquers**: three orthogonal coils along body x, y, z, ISIS iMTQ board,
  nominal actuation level 0.2 A m^2 per axis (20 degC, 5 V).
* **Reaction wheels**: three identical CubeSpace CubeWheel Gen 2 CW0017 wheels,
  maximum torque 0.23 mN m, rated momentum 1.77 mN m s at 8000 rpm.  The rated
  point is the operational speed limit (the 10000 rpm maximum is not used).
  The spin inertia ``J_s = 1.77e-3 / (8000 * 2 pi / 60) = 2.1128e-6 kg m^2`` is
  **derived** from the rated momentum and speed, not a datasheet number.  Wheel
  spin axes are the body x, y, z axes; no friction, torque ripple or motor
  dynamics.

Wheel momentum convention (``adcs/dynamics.py``): ``h_w`` is the *total* wheel
angular momentum along the body axes, including the body-rate part, so the
wheel speed relative to the body is ``Omega = h_w / J_s - w`` and
``h_w_dot = u_w`` with ``u_w`` the torque applied **to** the wheels (the body
feels ``-u_w``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RPM",
    "MagnetorquerConfig",
    "WheelConfig",
    "saturate_dipole",
    "magnetorquer_torque",
    "wheel_speed",
    "wheel_momentum",
    "limit_wheel_torque",
]

RPM: float = 2.0 * np.pi / 60.0
"""Conversion factor, rad/s per rpm."""


@dataclass(frozen=True)
class MagnetorquerConfig:
    """Magnetorquer (ISIS iMTQ class) parameters.

    ``m_max`` maximum dipole per body axis (A m^2); ``control_period`` (s);
    ``duty`` fraction of each control period the coils are driven (the
    magnetometer is sampled in the off part).
    """

    m_max: float = 0.2
    control_period: float = 1.0
    duty: float = 0.8

    def __post_init__(self) -> None:
        if not self.m_max > 0:
            raise ValueError("m_max must be > 0")
        if not self.control_period > 0:
            raise ValueError("control_period must be > 0")
        if not (0 < self.duty <= 1):
            raise ValueError("duty must satisfy 0 < duty <= 1")

    @property
    def on_time(self) -> float:
        """Coil-on time per control period, s."""
        return self.duty * self.control_period


@dataclass(frozen=True)
class WheelConfig:
    """Reaction-wheel (CubeWheel Gen 2 CW0017 class) parameters.

    ``max_torque`` N m; ``rated_momentum`` N m s at ``rated_speed_rpm`` rpm.
    """

    max_torque: float = 0.23e-3
    rated_momentum: float = 1.77e-3
    rated_speed_rpm: float = 8000.0

    def __post_init__(self) -> None:
        if not (self.max_torque > 0 and self.rated_momentum > 0 and self.rated_speed_rpm > 0):
            raise ValueError("max_torque, rated_momentum and rated_speed_rpm must be > 0")

    @property
    def max_speed(self) -> float:
        """Operational wheel speed limit, rad/s."""
        return self.rated_speed_rpm * RPM

    @property
    def spin_inertia(self) -> float:
        """Derived wheel spin inertia ``rated_momentum / max_speed``, kg m^2."""
        return self.rated_momentum / self.max_speed

    @property
    def h_max(self) -> float:
        """Rated wheel momentum, N m s."""
        return self.rated_momentum


def saturate_dipole(m_cmd, m_max: float, mode: str = "scale") -> np.ndarray:
    """Limit a commanded magnetic dipole, shape ``(..., 3)``, A m^2, body axes.

    ``"scale"``: if ``max_i |m_i| > m_max`` scale the whole vector by
    ``m_max / max_i |m_i|`` (direction preserved).  ``"clip"``: clip each
    component to ``[-m_max, m_max]``.  Returns a new array (input not modified).
    """
    m = np.array(m_cmd, dtype=float)
    if mode == "scale":
        peak = np.max(np.abs(m), axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(peak > m_max, m_max / peak, 1.0)
        return m * factor
    if mode == "clip":
        return np.clip(m, -m_max, m_max)
    raise ValueError(f"unknown saturation mode {mode!r}; use 'scale' or 'clip'")


def magnetorquer_torque(m_b, b_b) -> np.ndarray:
    """Magnetorquer torque ``m x B`` (N m, body axes).

    ``m_b`` dipole (A m^2, body axes), ``b_b`` magnetic flux density (T, body
    axes); shapes ``(..., 3)``, broadcast.
    """
    return np.cross(np.asarray(m_b, dtype=float), np.asarray(b_b, dtype=float))


def wheel_speed(h_w, w, cfg: WheelConfig = WheelConfig()) -> np.ndarray:
    """Wheel speeds relative to the body (rad/s), shape ``(..., 3)``.

    ``Omega = h_w / J_s - w``; ``h_w`` total wheel momentum (N m s, body axes),
    ``w`` body rate relative to ECI (rad/s, body axes).  Wheel i spins about
    body axis i.
    """
    return np.asarray(h_w, dtype=float) / cfg.spin_inertia - np.asarray(w, dtype=float)


def wheel_momentum(speed, w, cfg: WheelConfig = WheelConfig()) -> np.ndarray:
    """Total wheel momentum (N m s, body axes) from wheel speeds, shape ``(..., 3)``.

    Inverse of :func:`wheel_speed`: ``h_w = J_s (Omega + w)`` with ``Omega``
    the speed relative to the body (rad/s) and ``w`` the body rate (rad/s).
    """
    return cfg.spin_inertia * (np.asarray(speed, dtype=float) + np.asarray(w, dtype=float))


def limit_wheel_torque(u_cmd, h_w, w, cfg: WheelConfig = WheelConfig(), dt=None) -> np.ndarray:
    """Torque actually delivered to the wheels, shape ``(3,)``, N m, body axes.

    ``u_cmd`` commanded torque applied **to** the wheels (``h_w_dot = u_w``),
    ``h_w`` total wheel momentum (N m s), ``w`` body rate (rad/s).

    1. Each component is clipped to ``+-max_torque``.
    2. Speed limit, with ``h_hi = J_s (max_speed + w)`` and
       ``h_lo = J_s (-max_speed + w)`` per axis.  If ``dt is None`` the
       component is zeroed where the wheel is at/beyond a limit and the command
       pushes it further out.  If ``dt`` (s) is given the command is held for
       ``dt`` and each component is clipped to
       ``[min(0, (h_lo-h)/dt), max(0, (h_hi-h)/dt)]``, so a held command cannot
       carry a wheel past the limit, never pushes a wheel already beyond a limit
       further out, and never forces a reversal.

    The result satisfies both limits.
    """
    u = np.clip(np.array(u_cmd, dtype=float), -cfg.max_torque, cfg.max_torque)
    h = np.asarray(h_w, dtype=float)
    w = np.asarray(w, dtype=float)
    js = cfg.spin_inertia
    h_hi = js * (cfg.max_speed + w)
    h_lo = js * (-cfg.max_speed + w)
    if dt is None:
        blocked = ((h >= h_hi) & (u > 0)) | ((h <= h_lo) & (u < 0))
        return np.where(blocked, 0.0, u)
    if not dt > 0:
        raise ValueError("dt must be > 0")
    u_hi = np.maximum(0.0, (h_hi - h) / dt)
    u_lo = np.minimum(0.0, (h_lo - h) / dt)
    return np.clip(u, u_lo, u_hi)
